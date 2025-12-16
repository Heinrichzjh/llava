#!/usr/bin/env python3
"""
convert_driveaction_to_llava_onevision_final.py

LiAuto-DriveAction 数据集转 LLaVA-OneVision 训练格式（最终完整版）

功能：
1. 将 Parquet/HuggingFace Dataset 中的多视角图片解压保存。
2. 生成符合 LLaVA-OneVision SFT 要求的 JSON 数据。
3. 针对选择题：自动拼接选项内容，并添加 "只回答选项字母" 的指令。
4. 针对判断题：自动标准化答案为 True/False，并添加 "回答 True or False" 的指令。

用法示例：
  python convert_driveaction_to_llava_onevision_final.py \
    --dataset ./drive-action \
    --out-dir ./drive_action_output \
    --lang en \
    --max-samples 5000
"""

import os
import json
import io
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 尝试导入 datasets 库，如果没有安装也不影响 parquet 模式运行
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

import pandas as pd


def ensure_dir(p: Path):
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)


def save_image_from_field(image_field, save_path: Path):
    """
    保存图片到磁盘。
    支持输入格式：PIL.Image, dict({'bytes': ...}), bytes
    """
    if image_field is None:
        raise ValueError("image_field is None")

    # 1. 识别图片格式并打开
    if isinstance(image_field, Image.Image):
        img = image_field
    elif isinstance(image_field, dict) and 'bytes' in image_field:
        img = Image.open(io.BytesIO(image_field['bytes']))
    elif isinstance(image_field, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_field))
    else:
        # 尝试直接读取（兼容某些 dataset 返回路径的情况）
        try:
            img = Image.open(image_field)
        except Exception as e:
            raise ValueError(f"无法解析 image_field 类型: {type(image_field)} - {e}")

    # 2. 强制转为 RGB (去除透明通道，统一格式)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. 保存为 JPEG
    img.save(save_path, format='JPEG', quality=95)


def row_to_qa(row, lang='en'):
    """
    解析问答内容，进行答案标准化和指令增强。
    """
    # 1. 确定语言列
    key = 'content_en' if lang == 'en' else 'content_cn'
    if key not in row or row[key] is None:
        for alt in ['content_en', 'content_cn', 'content']:
            if alt in row and row[alt] is not None:
                key = alt
                break

    qa = row.get(key)
    
    # 兜底：如果没找到 QA 结构
    if qa is None:
        q = row.get('question') or row.get('question_text') or ''
        a = row.get('answer') or row.get('label') or ''
        return str(q), str(a)

    # 如果是字符串形式的字典，先 eval 解析
    if isinstance(qa, str):
        try:
            qa = eval(qa)
        except Exception:
            return qa, ''

    # 2. 核心解析逻辑
    if isinstance(qa, dict):
        question = str(qa.get('question') or qa.get('Q') or '')
        raw_answer = qa.get('answer') or qa.get('A')
        
        # --- 答案标准化 (Normalization) ---
        # 无论原始是 "正确", "true", "True", "1", 全部统一为 "True"/"False"
        ans_str = str(raw_answer).strip()
        ans_lower = ans_str.lower()
        
        if ans_lower in ['true', '1', 'yes', 't', '正确', '是']:
            answer = "True"
        elif ans_lower in ['false', '0', 'no', 'f', '错误', '否']:
            answer = "False"
        else:
            answer = ans_str # 其他情况（如选项 A/B/C/D）保持原样

        options = qa.get('options')

        # --- 情况 A: 选择题 (Options 存在且不为空) ---
        if options and isinstance(options, dict):
            formatted_options = []
            keys = sorted(options.keys()) # 确保 A, B, C, D 顺序
            
            # 拼接选项内容: "A. Left turn"
            for opt_key in keys:
                opt_val = options[opt_key]
                formatted_options.append(f"{opt_key}. {opt_val}")
            
            if formatted_options:
                question = question + "\n" + "\n".join(formatted_options)
            
            # 增加强指令：只回答字母
            if lang == 'en':
                opt_str = "/".join(keys)
                question = question + f"\nPlease answer with the option letter only ({opt_str})."
            else:
                question = question + "\n请直接回答选项字母。"
        
        # --- 情况 B: 判断题 (Options 为空，且答案是 T/F) ---
        elif answer in ['True', 'False']:
            # 增加强指令：回答 True/False
            if lang == 'en':
                question = question + "\nPlease answer True or False."
            else:
                question = question + "\n请判断对错 (回答 True 或 False)。"

        return question, answer

    raise ValueError('无法解析 qa 内容')


def process_item(item, idx, out_images_dir, lang):
    """
    处理单条数据：保存图片 + 生成 Conversation
    """
    # 1. 保存三张图片
    saved_paths = []
    for cam in range(3):
        key = f'image_{cam}'
        if key not in item or item[key] is None:
            continue
        try:
            # 文件名：idx_image_0.jpg
            fname = f"{idx}_{key}.jpg"
            dest = out_images_dir / fname
            save_image_from_field(item[key], dest)
            # 记录相对路径
            saved_paths.append(str(Path("images") / fname))
        except Exception as e:
            print(f"Warning: 无法保存图片 idx={idx} key={key}: {e}")

    if len(saved_paths) == 0:
        return None

    # 2. 解析 QA
    try:
        q, a = row_to_qa(item, lang=lang)
    except Exception as e:
        print(f"Warning: 无法解析 QA idx={idx}: {e}")
        return None

    # 3. 构造 Prompt (LLaVA-OneVision 格式)
    # 使用 "View N: <image>" 来区分多图
    view_prompts = []
    for i in range(len(saved_paths)):
        view_prompts.append(f"View {i+1}: <image>") 
    image_tags = '\n'.join(view_prompts)
    
    human_value = f"{image_tags}\n{q}"

    conversation = [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": a} # 注意：Role 必须是 gpt
    ]

    record_id = item.get('question_slice_id') or item.get('id') or f'idx_{idx}'
    
    # 4. 构造完整 Record
    record = {
        'id': f'{record_id}_{idx}',
        'image': saved_paths, # 注意：键名必须是 image (单数)
        'conversations': conversation
    }

    return record


def convert_from_dataset(dataset, out_images_dir: Path, lang='en', max_samples=None):
    records = []
    ensure_dir(out_images_dir)

    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    for idx in tqdm(range(n), desc='Converting Dataset'):
        item = dataset[idx]
        record = process_item(item, idx, out_images_dir, lang)
        if record:
            records.append(record)
    return records


def convert_from_parquet(parquet_path: Path, out_images_dir: Path, lang='en', max_samples=None):
    # 使用 Pandas 读取 Parquet
    df = pd.read_parquet(parquet_path)
    records = []
    ensure_dir(out_images_dir)

    total = len(df)
    if max_samples is not None:
        total = min(total, max_samples)

    for idx in tqdm(range(total), desc='Converting Parquet'):
        row = df.iloc[idx]
        item = row.to_dict() # 转为 dict 统一处理
        record = process_item(item, idx, out_images_dir, lang)
        if record:
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert LiAuto-DriveAction to LLaVA format")
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace repo id 或 本地数据集文件夹路径')
    parser.add_argument('--parquet', type=str, default=None, help='直接指定单个 .parquet 文件路径')
    parser.add_argument('--out-dir', type=str, default='./drive_action_llava', help='输出目录')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (默认 train)')
    parser.add_argument('--lang', type=str, choices=['en', 'cn'], default='en', help='选择问题语言 (默认 en)')
    parser.add_argument('--max-samples', type=int, default=None, help='仅转换前 N 条数据 (测试用)')
    parser.add_argument('--shuffle', action='store_true', help='打乱数据顺序')
    parser.add_argument('--json-name', type=str, default='drive_action_train.json', help='输出 JSON 文件名')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images'
    ensure_dir(images_dir)

    records = []

    print(f"Output Directory: {out_dir.absolute()}")

    # 模式 A: 从 Dataset (HF 或 本地文件夹) 加载
    if args.dataset:
        if load_dataset is None:
            raise RuntimeError('datasets 库未安装，请执行 `pip install datasets`')
        print(f'Loading dataset from: {args.dataset} ...')
        
        # 尝试加载
        try:
            ds = load_dataset(args.dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if args.split not in ds:
            print(f"Split '{args.split}' 不存在。可用 splits: {list(ds.keys())}")
            return
        
        dataset = ds[args.split]
        if args.shuffle:
            dataset = dataset.shuffle(seed=42)
            
        records = convert_from_dataset(dataset, images_dir, lang=args.lang, max_samples=args.max_samples)

    # 模式 B: 从 Parquet 文件加载
    elif args.parquet:
        parquet_path = Path(args.parquet)
        if not parquet_path.exists():
            raise FileNotFoundError(f"文件未找到: {parquet_path}")
        print(f'Loading parquet from: {parquet_path} ...')
        records = convert_from_parquet(parquet_path, images_dir, lang=args.lang, max_samples=args.max_samples)
    
    else:
        raise ValueError('请指定 --dataset 或 --parquet 参数')

    # 保存 JSON
    out_json = out_dir / args.json_name
    print(f'Saving {len(records)} records to {out_json} ...')
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print('✅ Conversion Complete!')
    print(f'Images: {images_dir}')
    print(f'JSON:   {out_json}')


if __name__ == '__main__':
    main()