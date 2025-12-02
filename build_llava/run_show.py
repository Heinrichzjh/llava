import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)

#from train_llava.custom_trainer import WebTrainer
from show_llava.data import LlavaDataset, TrainLLavaModelCollator
# from train_llava.data_websend import DatasetReceiveByWeb, TrainLlavaModelCollatorByWeb
from show_llava.util import print_trainable_parameters

logger = logging.getLogger(__name__)

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="show_model_v1/model001")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)


def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)

    if hasattr(processor, "image_processor"):
        processor.image_processor.patch_size = 14
        # 防止某些版本用的是 image_patch_size 这个名字
        if not hasattr(processor.image_processor, "image_patch_size"):
             processor.image_processor.image_patch_size = 14

    # 2. 【关键】强制给 processor 本身也赋值
    # 新版 transformers 可能会直接读取 self.patch_size
    # 我们直接在实例字典里覆盖它，不管它原来是不是 None
    try:
        processor.patch_size = 14
    except Exception:
        pass # 如果是只读属性，可能赋值失败，忽略

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)
        # model.print_trainable_parameters()

    elif modelargs.train_type == "none":
        logging.warning("使用全量参数进行训练")

        pass
    elif modelargs.train_type == "freeze_vision":
        logging.warning("冻结vision_tower网络层，剩下的网络权重进行训练")

        for param in model.vision_tower.parameters():
            param.requires_grad = False
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):

    llava_dataset = LlavaDataset(
        dataargs.data_path  # "data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
    )
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,  
    )
    


    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()