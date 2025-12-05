"""
Date: 2021-06-02 00:33:09
LastEditors: GodK
"""
import sys

sys.path.append("../")
from common.utils_lower import Preprocessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import random

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
                        "text": text[pos[0]:pos[1]+1],
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
            "scene": scene,
            "my_entity_list": my_entity_list, # 👈
        }

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
            )

        return prompt

    def generate_inputs(self, datas, max_seq_len, ent2id, scene2id, data_type="train"):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]

        将 labels 改为一个 字典结构
        labels = {
            "ner": 实体识别标签张量,  # shape: (ent_type_size, max_len, max_len)
            "scene": 场景多标签分类张量  # shape: (scene_type_size,)
        }

        """

        ent_type_size = len(ent2id)  # 实体类别
        scene_type_size = len(scene2id)

        all_inputs = []
        for sample in datas:
            #if data_type != "predict":
            sample = self.augment_data(sample, data_type=data_type)

            sample["text"] = self.build_prompt(sample["text"], scene2id, sample["my_entity_list"], data_type=data_type)  # 👈

            labels = None
            if data_type != "predict":
                scene_labels = []
                for scene_label in sample.get("scene", []):
                    scene_labels.append(scene_label)
                scene_labels = '、'.join(scene_labels)

                labels = {
                    # "ner": torch.tensor(ner_labels).clone().detach().long(),
                    # "scene": torch.tensor(scene_labels).clone().detach().float()  # 多标签分类一般用float
                    "scene": scene_labels
                }

            sample_input = {
                "sample": sample,
                "text": sample["text"],
                "labels": labels,
            }

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, scene2id, data_type="train", task_type="classification"):
        '''
        list: [(sample, text, labels), ...]
            sample: ...
            text: "prompt..."
            labels: {"scene": "label1、label2、..."}
        '''
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, scene2id, data_type)
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

        is_lm_output = data_type != "predict" and not(data_type == "valid" and task_type=="classification")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id

        # 🔁 在这里进行 batched tokenization
        inputs = self.tokenizer(
            text_list,
            add_special_tokens=False,
        )

        if is_lm_output:
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



class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        """
        多标签 scene 分类的 F1（micro）
        """
        y_pred = torch.gt(y_pred, 0).clone().detach().float()
        return 2 * torch.sum(y_true * y_pred) / (torch.sum(y_true + y_pred) + 1e-10)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).clone().detach().float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1e-10)

    def get_sample_recall(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).clone().detach().float()
        return torch.sum(y_pred[y_true == 1]) / (y_true.sum() + 1e-10)

    def get_scene_metrics(self, predict_labels, true_labels, scene_type_size, scene2id):
        """
        多标签 scene 分类的整体 F1, P, R（micro）
        Args:
            predict_labels: [[label1, label2, ...], ...]
            true_labels: [[label1, label2, ...], ...]
        """
        batch_size = len(predict_labels)
        pred_ids = [[scene2id[label] for label in labels if label in scene2id] for labels in predict_labels]
        true_ids = [[scene2id[label] for label in labels if label in scene2id] for labels in true_labels]

        y_pred = torch.zeros(batch_size, scene_type_size)
        y_true = torch.zeros(batch_size, scene_type_size)

        for i in range(batch_size):
            if pred_ids[i]:
                y_pred[i, pred_ids[i]] = 1
            if true_ids[i]:
                y_true[i, true_ids[i]] = 1

        f1 = self.get_sample_f1(y_pred, y_true)
        precision = self.get_sample_precision(y_pred, y_true)
        recall = self.get_sample_recall(y_pred, y_true)
        return f1.item(), precision.item(), recall.item()

    def get_evaluate_fpr(self, y_pred, y_true, threshold=0):
        """
        NER 的 F1、Precision、Recall
        """
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > threshold)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1 = 2 * X / (Y + Z + 1e-10)
        precision = X / (Y + 1e-10)
        recall = X / (Z + 1e-10)
        return f1, precision, recall

    def decode_scene(self, logits_scene, threshold=0):
        # probs = torch.sigmoid(logits_scene)
        preds = (logits_scene > threshold).int().cpu().numpy()
        results = []
        for pred_vec in preds:
            labels = [scene_label for i, scene_label in enumerate(self.id2scene.values()) if pred_vec[i] == 1]
            results.append(labels)
        return results