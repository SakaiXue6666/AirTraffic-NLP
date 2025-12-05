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
                    })  # [{"start": 3, "end": 6, "type": "<NER LABEL k>", "text": "<NER TEXT>", "entity_name": "<NER NAME>"}]

        all_entities.sort(key=lambda x: x["start"])  # 按起始位置排序，避免位置错乱

        new_text = ""
        last_idx = 0
        new_entity_list = []

        # my_entity_list = []

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
                    # <NER TEXT> --> <NER NAME>
                    replacement = ent["entity_name"]
                    new_text += text[last_idx:start] + replacement
                    new_start = len(new_text) - len(replacement)
                    new_end = len(new_text) - 1
                else:
                    # 替换为 [MASK] d3
                    # <NER TEXT> --> [mask]
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

            # my_entity_list.append((replacement, ent["type"]))

            if ent['type'] == 'NER LABEL (icao_code)':  ###
                span_airline_code.append((new_start, new_end, ent["text"], replacement))

        new_text += text[last_idx:]  # 加上剩余部分

        if span_airline_code != [] and set_call_sign != set():  ###
            for new_start, new_end, ent_text, replacement in span_airline_code:
                if ent_text in set_call_sign:
                    new_entity_list.append((new_start, new_end, 'NER LABEL (call_sign)'))

        return {
            "text": new_text,  # <TEXT AUG>
            "entity_list": new_entity_list,  # [(4, 8, "<NER LABEL k>")]
            "scene": scene  # ["CLS LABEL n"]
        }

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
            if data_type != "predict":
                sample = self.augment_data(sample, data_type=data_type)

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
                # label_text = "这是一个" + ent2id_cn[i]
                label_text = id2ent[i]
                tokens = self.tokenizer(label_text)["input_ids"]
                max_label_len = max(max_label_len, len(tokens))

            max_scene_len = 0
            for i in range(scene_type_size):
                # label_text = "这是一个" + ent2id_cn[i]
                scene_text = id2scene2[i]
                tokens = self.tokenizer(scene_text)["input_ids"]
                max_scene_len = max(max_scene_len, len(tokens))

            labels_input_ids = []
            labels_attention_mask = []
            labels_token_type_ids = []

            for i in range(ent_type_size):
                # label_text = "这是一个" + ent2id_cn[i]
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


            scene_labels_input_ids = []
            scene_labels_attention_mask = []
            scene_labels_token_type_ids = []

            for i in range(scene_type_size):
                # label_text = "这是一个" + ent2id_cn[i]
                scene_label_text = id2scene2[i]

                scene_label_inputs = self.tokenizer(
                    scene_label_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_scene_len,
                )

                scene_labels_input_ids.append(torch.tensor(scene_label_inputs["input_ids"]).clone().detach().long())
                scene_labels_attention_mask.append(torch.tensor(scene_label_inputs["attention_mask"]).clone().detach().long())
                scene_labels_token_type_ids.append(torch.tensor(scene_label_inputs["token_type_ids"]).clone().detach().long())

            # Stack all label encodings for this sample (shape: num_cls, label_len)
            scene_labels_input_ids = torch.stack(scene_labels_input_ids, dim=0)
            scene_labels_attention_mask = torch.stack(scene_labels_attention_mask, dim=0)
            scene_labels_token_type_ids = torch.stack(scene_labels_token_type_ids, dim=0)


            sample_input = (
                sample, input_ids, attention_mask, token_type_ids, labels,
                labels_input_ids, labels_attention_mask, labels_token_type_ids,
                scene_labels_input_ids,  scene_labels_attention_mask,  scene_labels_token_type_ids
            )

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, scene2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, scene2id, data_type)
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
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "predict":
                # labels_list.append(sample[4])
                ner_labels_list.append(sample[4]["ner"])
                scene_labels_list.append(sample[4]["scene"])

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

        labels_input_ids_list.append(sample[5])
        labels_attention_mask_list.append(sample[6])
        labels_token_type_ids_list.append(sample[7])

        labels_input_ids = labels_input_ids_list[0]
        labels_attention_mask = labels_attention_mask_list[0]
        labels_token_type_ids = labels_token_type_ids_list[0]

        scene_labels_input_ids_list.append(sample[8])
        scene_labels_attention_mask_list.append(sample[9])
        scene_labels_token_type_ids_list.append(sample[10])

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

    def get_scene_metrics(self, logits, y_true, threshold=0):
        """
        多标签 scene 分类的整体 F1, P, R（micro）
        Args:
            logits: raw scene预测输出 (batch_size, scene_type_size)
            y_true: ground truth (batch_size, scene_type_size)
            threshold: sigmoid阈值，默认0.5
        """
        y_pred = (logits > threshold).clone().detach().float()

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


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        # self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    # def forward(self, input_ids, attention_mask, token_type_ids):
    def forward(self, last_hidden_state, attention_mask):
        # self.device = input_ids.device
        self.device = last_hidden_state.device

        # context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        # last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        # 变成了一个长度为 ent_type_size 的 tuple
        # 每个元素 shape 是 (batch_size, seq_len, inner_dim*2)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        '''
        b	batch size
        m	起点 token 位置 (start index)
        n	终点 token 位置 (end index)
        h	实体类别 (entity type)
        d	特征维度 (inner_dim)
        
        对于每个 batch、每个实体类别，枚举所有起点-终点位置，
        把起点query向量和终点key向量做点积，得到一个打分
        '''
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # # padding mask
        # pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # # pad_mask = pad_mask_v&pad_mask_h
        # logits = logits * pad_mask - (1 - pad_mask) * 1e12
        #
        # # 排除下三角，起点在终点后面
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class TextGobalPointerEncoder(nn.Module):
    def __init__(self, encoder, ent_dim, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.ent_dim = ent_dim
        self.globalpointer = GlobalPointer(self.hidden_size, ent_dim, inner_dim, RoPE)

        self.dense = nn.Linear(self.ent_dim, self.ent_dim)

        # # 冻结 encoder 参数（只训练 prompt embeddings）
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.scene_classifier = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]
        logits = self.globalpointer(last_hidden_state, attention_mask)

        # (batch_size, ent_dim, seq_len, seq_len) -> (batch_size, seq_len, seq_len, ent_dim)
        logits = logits.permute(0, 2, 3, 1)
        # # (batch_size, seq_len, seq_len, ent_dim) -> (batch_size * seq_len * seq_len, ent_dim)
        # logits = logits.reshape(-1, self.ent_dim)
        logits = self.dense(logits)

        cls_output = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        scene_logits = self.scene_classifier(cls_output)  # (batch_size, self.ent_dim)

        return logits, scene_logits

class NerLabelEncoder(nn.Module):
    def __init__(self, encoder, ent_dim):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(hidden_size, ent_dim)

        # # 冻结 encoder 参数（只训练 prompt embeddings）
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] # (num_cls, seqlen, hidden_size)
        cls_token = last_hidden_state[:, 0, :] # (num_cls, hidden_size)

        # outputs: (num_cls, hidden_size) -> (num_cls, ent_dim)
        outputs = self.dense(cls_token)

        return outputs


class Merge(nn.Module):
    def __init__(self, encoder1, encoder2, ent_dim, inner_dim, ent_type_size):
        super().__init__()
        """
        :param encoder1: 用于编码文本的模型（函数或nn.Module）
        :param encoder2: 用于编码标签的模型（函数或nn.Module）
        """
        self.text_encoder = TextGobalPointerEncoder(encoder1, ent_dim, inner_dim)
        self.label_encoder = NerLabelEncoder(encoder2, ent_dim)
        self.scene_encoder = NerLabelEncoder(encoder2, encoder2.config.hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.ent_type_size = ent_type_size

    def forward(self, text_input_ids, text_attention_mask, text_token_type_ids,
              label_input_ids, label_attention_mask, label_token_type_ids,
                label_input_ids_scene,
                label_attention_mask_scene,
                label_token_type_ids_scene
                ):

        # 文本编码（输出实体 logits）
        entity_logits, scene_logits = self.text_encoder(
            text_input_ids,
            text_attention_mask,
            text_token_type_ids
        )
        # shape:
        # entity_logits: (batch_size, seq_len, seq_len, ent_dim)
        # scene_logits: (batch_size, ent_dim)


        # 标签编码（每个标签变成向量）
        label_embeddings = self.label_encoder(
            label_input_ids,
            label_attention_mask,
            label_token_type_ids
        )  # shape: (num_labels, ent_dim)

        label_embeddings_scene = self.scene_encoder(
            label_input_ids_scene,
            label_attention_mask_scene,
            label_token_type_ids_scene
        )  # shape: (scene_type_size, hidden_size)

        entity_logits = F.normalize(entity_logits, dim=-1)
        label_embeddings = F.normalize(label_embeddings, dim=-1)

        # # 计算相似度 (batch_size * seq_len * seq_len, num_labels)
        # similarity = entity_logits @ label_embeddings.t()

        logit_scale = self.logit_scale.exp()
        # 计算相似度 (batch_size, seq_len, seq_len, num_labels)
        similarity = logit_scale * entity_logits @ label_embeddings.t()
        # (batch_size, num_labels, seq_len, seq_len)
        similarity = similarity.permute(0, 3, 1, 2)

        batch_size = text_input_ids.size()[0]
        seq_len = text_input_ids.size()[1]
        # padding mask
        pad_mask = text_attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        similarity = similarity * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角，起点在终点后面
        mask = torch.tril(torch.ones_like(similarity), -1)
        similarity = similarity - mask * 1e12

        # ==================
        # 2. Scene 多标签分类
        # ==================
        scene_logits = F.normalize(scene_logits, dim=-1)
        label_embeddings_scene = F.normalize(label_embeddings_scene, dim=-1)

        # 计算相似度 (batch_size, scene_type_size)
        similarity_scene = logit_scale * scene_logits @ label_embeddings_scene.t()


        return similarity, similarity_scene