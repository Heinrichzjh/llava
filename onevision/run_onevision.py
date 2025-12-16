import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence,Any
# 必须引入 OneVision 专用类
from transformers import LlavaOnevisionForConditionalGeneration,LlavaOnevisionProcessor

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
)

#from train_llava.custom_trainer import WebTrainer
from show_onevision.data_onevision import LlavaDataset, TrainLLavaOneVisionCollator
# from train_llava.data_websend import DatasetReceiveByWeb, TrainLlavaModelCollatorByWeb
from show_onevision.utils_onevision import print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    # 修改默认路径为您下载的本地路径
    model_name_or_path: Optional[str] = field(default="/home/ZJH/llava/onevision/llava_onevision")
    train_type: Optional[str] = field(
        default="use_lora", # 推荐默认使用 lora
        metadata={
            "help": """
            1. use_lora: 使用 LoRA 微调 LLM，冻结 Vision Tower (推荐);
            2. none: 全量参数训练 (显存需求极大);
            3. freeze_vision: 冻结 Vision Tower，全量训练 LLM (显存需求大)
            """
        },
    )

@dataclass
class DataArguments:
    # 指向包含 images/ 和 drive_action_train.json 的目录
    data_path: str = field(
        default="./drive_action_output", 
        metadata={"help": "Path to the training data dir."}
    )
    # data_json_name: str = field(default="drive_action_train.json") # 可选扩展

def load_model_processor(modelargs: ModelArguments):

    config = AutoConfig.from_pretrained(
        modelargs.model_name_or_path,
        trust_remote_code=True 
    )

    model =LlavaOnevisionForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,

    )

    processor = AutoProcessor.from_pretrained(
        modelargs.model_name_or_path,
        #fix_mistral_regex=True     
    )

    if not hasattr(processor, "image_token"):
        processor.image_token = "<image>"

    if modelargs.train_type == "use_lora":
        logging.warning("Configuring LORA...")
        
        # 冻结 Vision Tower (通常做法)
        # OneVision 的视觉部分通常叫 vision_tower
        if hasattr(model, "vision_tower"):
            logging.info("Freezing Vision Tower...")
            for param in model.vision_tower.parameters():
                param.requires_grad = False
        
        from peft import LoraConfig, get_peft_model

        # 针对 Qwen2 (OneVision基座) 的推荐配置
        TARGET_MODULES = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]

        lora_config = LoraConfig(
            r=32, # 建议加大一点，32 也可以
            lora_alpha=128,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"], 
        )

        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        
        # 确保 Projector 是 float32 或者 bfloat16 参与训练
        # 有时候 PEFT 会把 modules_to_save 重置类型，确保它们 requires_grad=True
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name:
                param.requires_grad = True
                param.data = param.data.to(torch.bfloat16)
    
    elif modelargs.train_type == "freeze_vision":
        logging.warning("Freezing Vision Tower, Training LLM fully...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
            
    # 打印可训练参数情况
    print_trainable_parameters(model)

    return model, processor

def load_dataset_collator(processor, dataargs: DataArguments):

    llava_dataset = LlavaDataset(
        dataargs.data_path  
    )
    data_collator = TrainLLavaOneVisionCollator(processor, -100)

    return llava_dataset, data_collator
   


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.max_steps = -1
    training_args.remove_unused_columns = False
    # 开启 Gradient Checkpointing 节省显存
    training_args.gradient_checkpointing = True

    if training_args.num_train_epochs is None:
        training_args.num_train_epochs = 1
    print("DEBUG max_steps =", training_args.max_steps)
    print("DEBUG num_train_epochs =", training_args.num_train_epochs)

    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    print("Train dataset length =", len(train_dataset))
    print("First sample =", train_dataset[0])

    training_args.max_steps = -1
    training_args.num_train_epochs = int(training_args.num_train_epochs)
    training_args.do_train = True
    training_args.remove_unused_columns = True


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    logging.info("Start Training...")
    trainer.train()
    
    logging.info("Saving Model...")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    # 同时保存 processor，方便推理时直接加载
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()