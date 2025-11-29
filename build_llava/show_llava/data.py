from multiprocessing import Value
from typing import Any
from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict,List,Tuple
from platform import processor
from transformers import AutoProcessor

class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images_dl")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return human_input, chatbot_output, image_path




@dataclass
class QaImageOutput:
    q_input_ids:torch.Tensor
    pixel_values:torch.Tensor
    a_input_ids:torch.Tensor

def build_qaiamge(processor:LlavaProcessor,q_text:str,a_text:str,image_path:Path):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_file=image_path
    raw_image=Image.open(fp=image_file)

    inputs=processor(prompt,raw_image,return_tensors="pt")#根据词表（Vocabulary），把字符串切分并映射成对应的整数索引（Index）

    a_input_ids=processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res

class TrainLLavaModelCollator:
    def __init__(self,processor:AutoProcessor,IGNORE_INDEX:int)->None:
        self.processor=processor
        self.ignore_index=IGNORE_INDEX

    def convert_one_piece(
            self,
            q_input_ids:torch.Tensor,
            a_input_ids:torch.Tensor)->None:
        input_ids=torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor(llava_processor.tokenizer.eos_token_id).reshape(1,-1)
            ],
            axis=1,#在第二维度拼接
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ignore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )           
        return input_ids,labels



    def __call__(self,features:List)->Dict[str, torch.Tensor]:

        input_ids_list=[]
        label_list=[]
        pixel_values=[]
        max_input_len_list=[]

        for feature in features:
            qaimage_output=build_qaiamge(#图片文本放一起
                processor=llava_processor,
                q_text=feature[0],
                a_text=feature[1],
                image_path=feature[2]
            )
            temp_input_ids,temp_labels=self.convert_one_piece(#q&a拼接
                q_input_ids=qaimage_output.q_input_ids,
                a_input_ids=qaimage_output.a_input_ids
            )

            max_input_len_list.append(temp_input_ids.shape[1])#???
            input_ids_list.append(temp_input_ids)
            label_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len=max(max_input_len_list)

        final_input_ids=torch.concat(#应对长度不一样的语句，进行填充 list->tensor
            [
                torch.concat(
                    [
                        torch.full(
                            (1,max_input_len-max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),value
                    ],axis=1
                )
                for index,value in enumerate(input_ids_list)
            ]
        )

        final_labels=torch.concat(#labels同样要进行填充对齐  list->tensor
            [
                torch.concat(
                    [
                        torch.full(
                            (1,max_input_len-max_input_len_list[index]),
                            self.ignore_index,
                        ),value
                    ],axis=1
                )
                for index,value in enumerate(label_list)
            ]
        )

        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return{
            "input_ids":final_input_ids,
            "labels":final_labels,
            "pixel_values":final_pixel_values,
            "attention_mask":attention_mask
        }

 