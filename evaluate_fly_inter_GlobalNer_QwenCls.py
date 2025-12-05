"""
Date: 2021-06-11 13:54:00
LastEditors: GodK
LastEditTime: 2021-07-19 21:53:18
"""
import os
import config_fly as config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForCausalLM
from common.utils_lower import Preprocessor, multilabel_categorical_crossentropy
from models.GlobalPointer_clip_qk_fly_arg import Merge, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import wandb
# from evaluate_2 import load_model
import time
import numpy as np
import matplotlib.pyplot as plt

from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
import random

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.backends.cudnn.deterministic = True

############################
tokenizer_bert = BertTokenizerFast.from_pretrained(config["bert_path"])
tokenizer_qwen = AutoTokenizer.from_pretrained(config["qwen_path"], padding_side='left')
############################

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def augment_data(self, sample, mask_token='[MASK]', data_type="train", arg=True):
        """
        对样本进行数据增强（在训练时使用）
        train: 50% 不变，40% 替换为实体原名称，10% 替换为 [MASK]
        valid: 100% 不变

        Inputs:
        train和valid中一条数据格式：{"text":"...", "label": {label:{entity:[[start, end], ...], ..., "scene": [...]}
        Returns:
        train和valid中一条数据格式：{"text":"...", "entity_list":[(start, end, label), ...], "scene":[...]}
        """
        text = sample['text']
        entity_list = []

        offset = 0
        new_text = text
        modified_text = list(text)
        modified_spans = []

        all_entities = []
        for label_type, entities in sample["label"].items():
            if label_type == "scene":
                scene = sample["label"]["scene"]
                continue
            for ent, positions in entities.items():
                for pos in positions:
                    all_entities.append({
                        "start": pos[0],
                        "end": pos[1],
                        "type": label_type,
                        "text": text[pos[0]:pos[1] + 1],
                        "entity_name": ent
                    })
                    
        all_entities.sort(key=lambda x: x["start"])  # 按起始位置排序，避免位置错乱

        new_text = ""
        last_idx = 0
        new_entity_list = []

        my_entity_list = {}  # 👈

        span_airline_code = []  ###
        set_call_sign = set()  ###
        for ent in all_entities:
            start, end = ent["start"], ent["end"]

            if ent['type'] == 'NER LABEL (call_sign)':  ####
                set_call_sign.add(ent["text"])
                continue

            if data_type == "train" and arg == True:
                # 训练时：50%不变，40%替换为实体名，10%替换为[MASK]
                prob = random.random()

                if prob < 0.5:
                    # 不变
                    replacement = ent["text"]  ####
                    new_text += text[last_idx:end + 1]
                    new_start = len(new_text) - (end - start + 1)
                    new_end = len(new_text) - 1
                elif prob < 0.5 + 0.4:
                    # 替换为实体名称 d2
                    replacement = ent["entity_name"]
                    new_text += text[last_idx:start] + replacement
                    new_start = len(new_text) - len(replacement)
                    new_end = len(new_text) - 1
                else:
                    # 替换为 [MASK] d3
                    replacement = mask_token
                    new_text += text[last_idx:start] + replacement
                    new_start = len(new_text) - len(replacement)
                    new_end = len(new_text) - 1

            else:
                # 非训练时：100% 不变
                replacement = ent["text"]  ####
                new_text += text[last_idx:end + 1]
                new_start = len(new_text) - (end - start + 1)
                new_end = len(new_text) - 1

            new_entity_list.append((new_start, new_end, ent["type"]))
            last_idx = end + 1

            if ent["type"] not in my_entity_list:  # 👈
                my_entity_list[ent["type"]] = [replacement]
            else:
                my_entity_list[ent["type"]].append(replacement)

            if ent['type'] == 'NER LABEL (icao_code)':  ###
                span_airline_code.append((new_start, new_end, ent["text"], replacement))

        new_text += text[last_idx:]  # 加上剩余部分

        if span_airline_code != [] and set_call_sign != set():  ###
            for new_start, new_end, ent_text, replacement in span_airline_code:
                if ent_text in set_call_sign:
                    new_entity_list.append((new_start, new_end, 'NER LABEL (call_sign)'))

                    if 'NER LABEL (call_sign)' not in my_entity_list:  # 👈
                        my_entity_list['NER LABEL (call_sign)'] = [replacement]
                    else:
                        my_entity_list['NER LABEL (call_sign)'].append(replacement)

        return {
            "text": new_text,
            "entity_list": new_entity_list,
            "scene": scene if data_type != "predict" else None,
            "my_entity_list": my_entity_list,  # 👈
        }

    def generate_inputs_bert(self, datas, max_seq_len, ent2id, scene2id, data_type="train"):
        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in datas:
            if data_type != "predict":
                sample = self.augment_data(sample, data_type=data_type)
            else:
                sample = {"text": sample["text"], "entity_list": []}

            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            labels = None
            if data_type != "predict":
                # === 1. NER 标签 ===
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                ner_labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    ner_labels[ent2id[label], start, end] = 1

                # === 2. Scene 多标签分类 ===
                scene_labels = np.zeros(scene_type_size)
                for scene_label in sample.get("scene", []):
                    if scene_label in scene2id:
                        scene_labels[scene2id[scene_label]] = 1

                labels = {
                    "ner": torch.tensor(ner_labels).clone().detach().long(),
                    "scene": torch.tensor(scene_labels).clone().detach().float()  # 多标签分类一般用float
                }

            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).clone().detach().long()
            attention_mask = torch.tensor(inputs["attention_mask"]).clone().detach().long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).clone().detach().long()
            if labels is not None:
                # labels = torch.tensor(inputs["labels"]).clone().detach().long()
                labels = {
                    "ner": torch.tensor(inputs["labels"]["ner"]).clone().detach().long(),
                    "scene": torch.tensor(inputs["labels"]["scene"]).clone().detach().float()
                }

            # == New: generate label descriptions ==
            id2ent = {v: k for k, v in ent2id.items()}
            id2scene2 = {v: k for k, v in scene2id.items()}

            max_label_len = 0
            for i in range(ent_type_size):
                label_text = id2ent[i]
                tokens = self.tokenizer(label_text)["input_ids"]
                max_label_len = max(max_label_len, len(tokens))

            labels_input_ids = []
            labels_attention_mask = []
            labels_token_type_ids = []

            for i in range(ent_type_size):
                label_text = id2ent[i]

                label_inputs = self.tokenizer(
                    label_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_label_len,
                )

                labels_input_ids.append(torch.tensor(label_inputs["input_ids"]).clone().detach().long())
                labels_attention_mask.append(torch.tensor(label_inputs["attention_mask"]).clone().detach().long())
                labels_token_type_ids.append(torch.tensor(label_inputs["token_type_ids"]).clone().detach().long())

            # Stack all label encodings for this sample (shape: num_cls, label_len)
            labels_input_ids = torch.stack(labels_input_ids, dim=0)
            labels_attention_mask = torch.stack(labels_attention_mask, dim=0)
            labels_token_type_ids = torch.stack(labels_token_type_ids, dim=0)

            # == New: generate label descriptions ==
            max_scene_len = 0
            for i in range(scene_type_size):
                scene_text = id2scene2[i]
                tokens = self.tokenizer(scene_text)["input_ids"]
                max_scene_len = max(max_scene_len, len(tokens))

            scene_labels_input_ids = []
            scene_labels_attention_mask = []
            scene_labels_token_type_ids = []

            for i in range(scene_type_size):
                scene_label_text = id2scene2[i]

                scene_label_inputs = self.tokenizer(
                    scene_label_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_scene_len,
                )

                scene_labels_input_ids.append(torch.tensor(scene_label_inputs["input_ids"]).clone().detach().long())
                scene_labels_attention_mask.append(
                    torch.tensor(scene_label_inputs["attention_mask"]).clone().detach().long())
                scene_labels_token_type_ids.append(
                    torch.tensor(scene_label_inputs["token_type_ids"]).clone().detach().long())

            # Stack all label encodings for this sample (shape: num_cls, label_len)
            scene_labels_input_ids = torch.stack(scene_labels_input_ids, dim=0)
            scene_labels_attention_mask = torch.stack(scene_labels_attention_mask, dim=0)
            scene_labels_token_type_ids = torch.stack(scene_labels_token_type_ids, dim=0)

            sample_input = {
                "sample": sample,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "token_type_ids": token_type_ids,
                "labels_input_ids": labels_input_ids,
                "labels_attention_mask": labels_attention_mask,
                "labels_token_type_ids": labels_token_type_ids,
                "scene_input_ids": scene_labels_input_ids,
                "scene_attention_mask": scene_labels_attention_mask,
                "scene_token_type_ids": scene_labels_token_type_ids,
            }

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch_bert(self, batch_data, max_seq_len, ent2id, scene2id, data_type="train"):
        batch_data = self.generate_inputs_bert(batch_data, max_seq_len, ent2id, scene2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        # labels_list = []

        ner_labels_list = []
        scene_labels_list = []

        labels_input_ids_list = []
        labels_attention_mask_list = []
        labels_token_type_ids_list = []

        scene_labels_input_ids_list = []
        scene_labels_attention_mask_list = []
        scene_labels_token_type_ids_list = []

        for sample in batch_data:
            sample_list.append(sample["sample"])
            input_ids_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            token_type_ids_list.append(sample["token_type_ids"])
            if data_type != "predict":
                # labels_list.append(sample[4])
                ner_labels_list.append(sample["labels"]["ner"])
                scene_labels_list.append(sample["labels"]["scene"])


        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        # batch_labels = torch.stack(labels_list, dim=0) if data_type != "predict" else None

        if data_type != "predict":
            batch_ner_labels = torch.stack(ner_labels_list, dim=0)
            batch_scene_labels = torch.stack(scene_labels_list, dim=0)
            batch_labels = {
                "ner": batch_ner_labels,
                "scene": batch_scene_labels
            }
        else:
            batch_labels = None

        labels_input_ids_list.append(sample["labels_input_ids"])
        labels_attention_mask_list.append(sample["labels_attention_mask"])
        labels_token_type_ids_list.append(sample["labels_token_type_ids"])

        labels_input_ids = labels_input_ids_list[0]
        labels_attention_mask = labels_attention_mask_list[0]
        labels_token_type_ids = labels_token_type_ids_list[0]

        scene_labels_input_ids_list.append(sample["scene_input_ids"])
        scene_labels_attention_mask_list.append(sample["scene_attention_mask"])
        scene_labels_token_type_ids_list.append(sample["scene_token_type_ids"])

        scene_labels_input_ids = scene_labels_input_ids_list[0]
        scene_labels_attention_mask = scene_labels_attention_mask_list[0]
        scene_labels_token_type_ids = scene_labels_token_type_ids_list[0]

        # return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels
        return (
            sample_list,
            batch_input_ids,
            batch_attention_mask,
            batch_token_type_ids,
            batch_labels,
            labels_input_ids,
            labels_attention_mask,
            labels_token_type_ids,
            scene_labels_input_ids,
            scene_labels_attention_mask,
            scene_labels_token_type_ids
        )

    def build_prompt(self, text, scene2id, my_entity_list, data_type):
        class_content = '\n'.join([label for label, _ in sorted(scene2id.items(), key=lambda x: x[1])])
        entity_content = "；".join(
            f"{label}: {'、'.join(label_text_list)}"
            for label, label_text_list in my_entity_list.items()
        )

        if data_type != "predict":
            prompt = (
                "<|im_start|>system\n"
                "你是一个飞行与管制文本的多标签分类器，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。多个标签用顿号「、」分隔。不要输出多余内容。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"# 类别\n{class_content}\n\n"
                f"# 文本的实体\n{entity_content}\n\n"
                f"# 文本\n{text}\n\n"
                "请输出此文本对应的类别：\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            prompt = (
                "<|im_start|>system\n"
                "你是一个飞行与管制文本的多标签分类器，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。多个标签用顿号「、」分隔。不要输出多余内容。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"# 类别\n{class_content}\n\n"
                "# 新增示例\n文本的实体：<NER LABEL 1>：<NER TEXT>；<NER LABEL 2>：<NER TEXT>。文本：<TEXT>。此文本对应类别：<CLS LABEL N+1>。\n\n"
                f"# 文本的实体\n{entity_content}\n\n"
                f"# 文本\n{text}\n\n"
                "请输出此文本对应的类别：\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"

                # "<|im_start|>system\n"
                # "你是一个小朋友\n"
                # "<|im_end|>\n"
                # "<|im_start|>user\n"
                # f"{text}\n\n"
                # "<|im_end|>\n"
                # "<|im_start|>assistant\n"

                # "<|im_start|>system\n"
                # "你是一个飞行与管制文本的多标签分类器，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。多个标签用顿号「、」分隔。不要输出多余内容。\n"
                # "<|im_end|>\n"
                # "<|im_start|>user\n"
                # f"# 类别\n{class_content}\n也可以在新增类别中选择：<CLS LABEL N+1>\n\n"
                # f"# 文本的实体\n{entity_content}\n\n"
                # f"# 文本\n{text}\n\n"
                # "请输出此文本对应的类别：\n"
                # "<|im_end|>\n"
                # "<|im_start|>assistant\n"
            )

        return prompt

    def generate_inputs_qwen(self, datas, max_seq_len, ent2id, scene2id, data_type="train"):
        ent_type_size = len(ent2id)  # 实体类别
        scene_type_size = len(scene2id)

        all_inputs = []
        for sample in datas:
            # if data_type != "predict":
            sample = self.augment_data(sample, data_type=data_type)

            text = self.build_prompt(sample["text"], scene2id, sample["my_entity_list"], data_type=data_type)  # 👈

            labels = None
            if data_type != "predict":
                scene_labels = []
                for scene_label in sample.get("scene", []):
                    scene_labels.append(scene_label)
                scene_labels = '、'.join(scene_labels)

                labels = {
                  "scene": scene_labels
                }

            sample_input = {
                "sample": sample,
                "text": text,
                "labels": labels,
            }

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch_qwen(self, batch_data, max_seq_len, ent2id, scene2id, data_type="train", task_type="classification"):
        '''
        list: [(sample, text, labels), ...]
            sample: ...
            text: "prompt..."
            labels: {"scene": "label1、label2、..."}
        '''
        batch_data = self.generate_inputs_qwen(batch_data, max_seq_len, ent2id, scene2id, data_type)
        sample_list = []
        text_list = []

        ner_labels_list = []
        scene_labels_list = []

        for sample in batch_data:
            sample_list.append(sample["sample"])
            text_list.append(sample["text"])
            if data_type != "predict":
                # labels_list.append(sample[4])
                # ner_labels_list.append(sample["labels"]["ner"])
                scene_labels_list.append(sample["labels"]["scene"])

        # batch_input_ids = torch.stack(input_ids_list, dim=0)
        # batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        # batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        # batch_labels = torch.stack(labels_list, dim=0) if data_type != "predict" else None

        is_lm_output = data_type != "predict" and not(data_type == "valid" and task_type=="classification")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id

        # 🔁 在这里进行 batched tokenization
        inputs = self.tokenizer(
            text_list,
            add_special_tokens=False,
        )

        if is_lm_output:
            # batch_ner_labels = torch.stack(ner_labels_list, dim=0)
            # batch_scene_labels = torch.stack(scene_labels_list, dim=0)
            # batch_labels = {
            #     "ner": batch_ner_labels,
            #     "scene": batch_scene_labels
            # }

            outputs = self.tokenizer(
                scene_labels_list,
                add_special_tokens=False,
            )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i in range(len(inputs['input_ids'])):
            if is_lm_output:
                inputs_ids = inputs['input_ids'][i] + outputs['input_ids'][i] + [im_end_id]
                attention_mask = inputs['attention_mask'][i] + outputs['attention_mask'][i] + [1]
                labels = [-100] * len(inputs['input_ids'][i]) + outputs['input_ids'][i] + [im_end_id]  # causal lm 的标签中，prompt 部分为 -100
            else :
                inputs_ids = inputs['input_ids'][i]
                attention_mask = inputs['attention_mask'][i]

            # 截断
            if len(inputs_ids) > max_seq_len:
                inputs_ids = inputs_ids[:max_seq_len]
                attention_mask = attention_mask[:max_seq_len]
                if is_lm_output:
                    labels = labels[:max_seq_len]

            batch_input_ids.append(inputs_ids)
            batch_attention_mask.append(attention_mask)
            if is_lm_output:
                batch_labels.append(labels)

        # Step 4: Pad to the longest sequence in batch
        longest = max(len(ids) for ids in batch_input_ids)
        for i in range(len(batch_input_ids)):
            batch_input_ids[i] = [self.tokenizer.pad_token_id] * (longest - len(batch_input_ids[i])) + batch_input_ids[i]
            batch_attention_mask[i] = [0] * (longest - len(batch_attention_mask[i])) + batch_attention_mask[i]
            if is_lm_output:
                batch_labels[i] = [-100] * (longest - len(batch_labels[i])) + batch_labels[i]


        # Convert to tensors
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long) if is_lm_output else None

        return (
            sample_list,
            batch_input_ids,
            batch_attention_mask,
            None,  # token_type_ids not used
            batch_labels,
            scene_labels_list if data_type != "predict" else None,
        )

    def decode_ent(self, pred_matrix):
        pass


def load_data(data_path, data_type="train"):
    if data_type in ["train", "valid", "predict"]:
        datas = []
        with open(data_path, encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)  # 回到文件开头
            if first_char == "[":
                # 新格式：JSON数组
                data_list = json.load(f)
            else:
                # 旧格式：JSON lines
                data_list = [json.loads(line) for line in f]

        return data_list
    else:
        return json.load(open(data_path, encoding="utf-8"))


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)

scene2id_path = os.path.join(config["data_home"], config["exp_name"], config["scene2id"])
with open(scene2id_path, encoding="utf-8") as f:
    scene2id = json.load(f)

scene_type_size = len(scene2id)
id2scene = {v: k for k, v in scene2id.items()}


def data_generator(data_type="predict"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "predict":
        predict_data_path = os.path.join(config["data_home"], config["exp_name"], config["predict_data"])
        predict_data = load_data(predict_data_path, "predict")

    all_data = predict_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer_bert.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer_bert)

    if data_type == "predict":
        predict_dataloader = DataLoader(MyDataset(predict_data),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch_bert(x, max_seq_len, ent2id, scene2id,
                                                                                    data_type="predict")
                                     )
        return predict_dataloader


def decode_ent(text, pred_matrix, tokenizer, threshold=0):
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    # ent_list = {}
    # for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
    #     ent_type = id2ent[ent_type_id]
    #     ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
    #     ent_text = text[ent_char_span[0]:ent_char_span[1]]
    #
    #     ent_type_dict = ent_list.get(ent_type, {})
    #     ent_text_list = ent_type_dict.get(ent_text, [])
    #     ent_text_list.append(ent_char_span)
    #     ent_type_dict.update({ent_text: ent_text_list})
    #     ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)

    ent_list = OrderedDict()
    entities = []

    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]
        entities.append({
            "start": ent_char_span[0],
            "end": ent_char_span[1],
            "type": ent_type,
            "text": ent_text
        })

    # 按 start 排序
    entities.sort(key=lambda x: x["start"])

    for ent in entities:
        ent_type = ent["type"]
        ent_text = ent["text"]
        span = [ent["start"], ent["end"]]
        if ent_type not in ent_list:
            ent_list[ent_type] = {}
        if ent_text not in ent_list[ent_type]:
            ent_list[ent_type][ent_text] = []
        ent_list[ent_type][ent_text].append(span)

    return ent_list


############################
text_encoder = BertModel.from_pretrained(config["bert_path"])
label_encoder = BertModel.from_pretrained(config["bert_path"])
model_bert = Merge(text_encoder, label_encoder, 128, 128, ent_type_size)
model_bert.load_state_dict(torch.load(config["model_state_ner"]), strict=False)
model_bert = model_bert.to(device)

model_qwen_base = AutoModelForCausalLM.from_pretrained(config["qwen_path"])
model_qwen = PeftModel.from_pretrained(model_qwen_base, config["model_lora_cls"])
model_qwen = model_qwen.to(device)
############################


def interactive_predict():
    while True:
        try:
            text = input("\n\n请输入指令文本（输入 exit 退出）：\n=================================\n")
            if text.lower() == "exit":
                break

            # 构造样本
            sample = {"text": text, "label": {}}
            data_maker = DataMaker(tokenizer_bert)
            max_seq_len = hyper_parameters["max_seq_len"]
            batch = data_maker.generate_batch_bert([sample], max_seq_len, ent2id, scene2id, data_type="predict")


            (batch_input_ids, batch_attention_mask, batch_token_type_ids,
             batch_labels,
             labels_input_ids, labels_attention_mask, labels_token_type_ids,
             scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = batch[1:]

            (batch_input_ids, batch_attention_mask, batch_token_type_ids,
             labels_input_ids, labels_attention_mask, labels_token_type_ids,
             scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = (
                batch_input_ids.to(device), batch_attention_mask.to(device), batch_token_type_ids.to(device),
                labels_input_ids.to(device), labels_attention_mask.to(device), labels_token_type_ids.to(device),
                scene_labels_input_ids.to(device), scene_labels_attention_mask.to(device),
                scene_labels_token_type_ids.to(device),
            )

            with torch.no_grad():
                logits_ner, logits_scene = model_bert(
                    batch_input_ids, batch_attention_mask, batch_token_type_ids,
                    labels_input_ids, labels_attention_mask, labels_token_type_ids,
                    scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids
                )

            # 解码
            pred_matrix = logits_ner[0]
            label = decode_ent(text, pred_matrix, tokenizer_bert)
            # scene = decode_scene(logits_scene)[0]
            # label['scene'] = scene

            # 打印 JSON 格式
            print("=================================")
            # print("结构化 JSON 输出：")
            # print("-----------------")
            # print(json.dumps({"text": text, "label": label}, ensure_ascii=False))
            # print("-----------------")

            # 打印格式化输出
            print("\n------------")
            print("格式化输出：")
            print("------------")
            print(f"指令文本：{text}")
            print("识别结果：")
            # print("指令场景：", "\t".join(scene))
            print("关键词：")
            for ent_type, entities in label.items():
                if ent_type == "scene":
                    continue
                for ent_text in entities:
                    print(f"{ent_type}\t{ent_text}")
            #################################################
            sample = {"text": text, "label": dict(label)}
            data_maker = DataMaker(tokenizer_qwen)
            ###
            my_scene2id = load_data(scene2id_path, "scene2id")
            my_scene2id["CLSn+1"] = 1000

            batch = data_maker.generate_batch_qwen([sample], 512, ent2id, my_scene2id, data_type="predict")
            (batch_samples,
             batch_input_ids, batch_attention_mask, _,
             _, true_texts) = batch

            (batch_input_ids, batch_attention_mask) = (
                batch_input_ids.to(device), batch_attention_mask.to(device)
            )

            with torch.no_grad():
                scene_ids = model_qwen.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer_qwen.pad_token_id
                )

            # 解码
            # 假设 batch_input_ids.shape = (batch_size, seq_len)
            prompt_len = batch_input_ids.shape[1]
            # 去掉 prompt，只保留 response
            output_ids = [scene_ids[i, prompt_len:] for i in range(scene_ids.size(0))]

            # padded 到相同长度（可选，例如为了 batch_decode）
            output_ids = pad_sequence(output_ids, batch_first=True, padding_value=tokenizer_qwen.pad_token_id)

            predict_texts = tokenizer_qwen.batch_decode(output_ids, skip_special_tokens=True)
            
            predict_labels = [predict_text.split("assistant\n")[1].split("、") if ("assistant\n" in predict_text) else predict_text.split("、") for predict_text in predict_texts]

            print("------------")
            print("指令场景：", "\t".join(predict_labels[0]))  # [[...]]
            print("------------")

        except Exception as e:
            print("发生错误：", e)



if __name__ == '__main__':
    # evaluate()

    interactive_predict()