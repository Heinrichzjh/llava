from multiprocessing import Value
from typing import Any
from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict,List,Tuple,Any,Union
from platform import processor
from transformers import AutoProcessor,LlavaProcessor
from torch.nn.utils.rnn import pad_sequence


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("drive_action_train.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        # --- 修改部分：支持多个图片 ---
        image_list = cur_data.get("image")   # 可能是 list 或 str
        image_paths = []

        if isinstance(image_list, list):
            for img in image_list:
                # img 是类似 "images/10_image_0.jpg"，取文件名
                img_path = self.image_dir / Path(img).name
                image_paths.append(img_path)
        else:
            # 单图片情况（兼容旧数据）
            image_paths = [self.image_dir / Path(image_list).name]
        return human_input, chatbot_output, image_paths



@dataclass
class QaImageOutput:
    input_ids: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_sizes: torch.Tensor


def build_qaimage(processor: AutoProcessor, 
                  q_text_from_json: str, 
                  answer: str, 
                  image_paths: List[Union[str, Path]]) -> QaImageOutput:
    
    # 1. System Prompt
    system_prompt = (
        "You are an autonomous driving assistant. "
        "Analyze the multi-view images and answer the user's question accurately."
    )
    
    # 2. 构造 User Content (修改为：构造纯字符串)
    # -----------------------------------------------------------
    # [修改说明]: Tokenizer 模板不支持 List，我们需要手动构造
    # 包含 <image> 占位符的字符串。
    # -----------------------------------------------------------
    
    # 清洗文本 (移除 json 里的旧 <image> 占位符，防止重复)
    clean_q_text = q_text_from_json.replace("<image>", "").strip()
    
    # 构造图片占位符字符串。
    # 如果有 3 张图，这就是 "<image>\n<image>\n<image>\n"
    # 注意：LLaVA-OneVision 通常使用 <image> 作为占位符，Processor 后续会处理它
    image_tokens_str = ("<image>\n" * len(image_paths))
    
    # 拼接成最终的用户输入字符串
    user_content_str = image_tokens_str + clean_q_text

    # 3. 构造 Messages
    messages = [
        {"role": "system", "content": system_prompt}, 
        # [修改说明]: 这里 content 传入 str，而不是 list
        {"role": "user",   "content": user_content_str},
    ]

    # 4. 生成 Prompt 文本
    # 此时传入的是纯文本，apply_chat_template 不会再报 concatenation 错误
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5. 构造完整文本 (Prompt + Answer + EOS)
    full_text = prompt_text + answer + processor.tokenizer.eos_token

    # 6. 加载图片
    raw_images = [Image.open(Path(p)).convert("RGB") for p in image_paths]

    # 7. Processor 一键处理
    # LLaVA-OneVision 的 processor 会查找 text 中的 <image> 字符串，
    # 并将其替换为对应的 vision token (如 <|vision_start|>...<|vision_end|>)
    inputs = processor(
        text=full_text,
        images=raw_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"][0]
    pixel_values = inputs["pixel_values"][0] 
    image_sizes = inputs["image_sizes"][0]
    if image_sizes.ndim == 1:
        image_sizes = image_sizes.unsqueeze(0) # 确保形状是 (Num_Images, 2)
    
    # 8. Mask Labels (SFT 逻辑)
    # 计算 prompt 的真实长度
    prompt_inputs = processor(
        text=prompt_text,
        images=raw_images,
        return_tensors="pt",
        padding=True 
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = input_ids.clone()
    # 这里的 mask 逻辑保持不变
    if prompt_len < len(labels):
        labels[:prompt_len] = -100
    else:
        labels[:] = -100

    return QaImageOutput(
        input_ids=input_ids,
        labels=labels,
        pixel_values=pixel_values,
        image_sizes=image_sizes
    )






class TrainLLavaOneVisionCollator:
    def __init__(self, processor, IGNORE_INDEX: int = -100):
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
        self.pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        # --- 🔴 DEBUG START ---
        print(f"\n[DEBUG] Collator called with {len(features)} features")
        try:
            input_ids_list = []
            labels_list = []
            pixel_values_list = []
            image_sizes_list = []

            for i, feature in enumerate(features):
                print(f"[DEBUG] Processing feature {i}...")
                # feature[0]: q_text, feature[1]: answer, feature[2]: paths
                
                # 打印一下路径看看对不对
                # print(f"  -> Images: {feature[2]}") 

                qaimage_output = build_qaimage(
                    processor=self.processor,
                    q_text_from_json=feature[0],
                    answer=feature[1],
                    image_paths=feature[2]
                )
                
                # 检查输出是否为 None
                if qaimage_output is None:
                    print(f"[ERROR] Feature {i} returned None!")
                    continue

                input_ids_list.append(qaimage_output.input_ids)
                labels_list.append(qaimage_output.labels)
                pixel_values_list.append(qaimage_output.pixel_values)
                image_sizes_list.append(qaimage_output.image_sizes)

            print("[DEBUG] Stacking tensors...")
            final_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
            final_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
            attention_mask = final_input_ids.ne(self.pad_token_id).long()
            final_pixel_values = torch.cat(pixel_values_list, dim=0)
            final_image_sizes = torch.cat(image_sizes_list, dim=0)
            
            #print(f"[DEBUG] Batch prepared. Input shape: {final_input_ids.shape}")
            
            return {
                "input_ids": final_input_ids,
                "labels": final_labels,
                "attention_mask": attention_mask,
                "pixel_values": final_pixel_values,
                "image_sizes": final_image_sizes
            }
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Collator failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
        # --- 🔴 DEBUG END ---
   