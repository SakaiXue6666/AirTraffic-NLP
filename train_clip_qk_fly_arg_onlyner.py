"""
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""

import os
import config_fly as config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
from common.utils_lower import Preprocessor, multilabel_categorical_crossentropy
from models.GlobalPointer_clip_qk_fly_arg import DataMaker, MyDataset, Merge, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import wandb
# from evaluate_2 import load_model
import time
import numpy as np
import matplotlib.pyplot as plt

config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
# 避免因算法选择或硬件优化导致的非确定性行为
torch.backends.cudnn.deterministic = True

if config["logger"] == "wandb" and config["run_type"] == "train":
    # init wandb
    wandb.init(project="GlobalPointer_" + config["exp_name"],
               config=hyper_parameters  # Initialize config
               )
    wandb.run.name = config["run_name"] + "_" + wandb.run.id

    model_state_dict_dir = wandb.run.dir
    logger = wandb
elif config["run_type"] == "train":
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

"""加载tokenizer"""
if config["encoder"] == "Bert":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
elif config["encoder"] == "Roberta":
    tokenizer = RobertaTokenizer.from_pretrained(config["bert_path"])

def load_data(data_path, data_type="train"):
    """
    读取json数据
    """
    if data_type in ["train", "valid"]:
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

"""读取实体、场景的id字典"""
ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)
id2ent = {v: k for k, v in ent2id.items()}

# 未记录scene类型
scene2id_path = os.path.join(config["data_home"], config["exp_name"], config["scene2id"])
scene2id = load_data(scene2id_path, "scene2id")
scene_type_size = len(scene2id)
id2scene = {v: k for k, v in scene2id.items()}


def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """
    """加载json"""
    if data_type == "train":
        train_data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        train_data = load_data(train_data_path, "train")
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid")
    elif data_type == "valid":
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []
    elif data_type == "test":
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []

    all_data = train_data + valid_data

    """句子截取"""
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num + 10, hyper_parameters["max_seq_len"])   ###

    """初始化数据处理对象"""
    data_maker = DataMaker(tokenizer)

    """生成dataloader"""
    if data_type == "train":
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, scene2id, data_type="train")
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, scene2id, data_type="valid")
                                      )
        return train_dataloader, valid_dataloader
    else:
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, scene2id)
                                      )
        return valid_dataloader

"""初始化评估对象"""
metrics = MetricsCalculator()

"""加载model"""
if config["encoder"] == "Bert":
    text_encoder = BertModel.from_pretrained(config["bert_path"])
    label_encoder = BertModel.from_pretrained(config["bert_path"])
elif config["encoder"] == "Roberta":
    text_encoder = RobertaModel.from_pretrained(config["bert_path"])
    label_encoder = RobertaModel.from_pretrained(config["bert_path"])
"""加载自定义模型"""
model = Merge(text_encoder, label_encoder, 128, 128, ent_type_size)
model = model.to(device)

if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.watch(model)

"""train"""
def train_step(batch_train, model, optimizer, criterion):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    (batch_samples,
     batch_input_ids, batch_attention_mask, batch_token_type_ids,
     batch_labels,
     labels_input_ids, labels_attention_mask, labels_token_type_ids,
     scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = batch_train

    (batch_input_ids, batch_attention_mask, batch_token_type_ids,
     labels_input_ids, labels_attention_mask, labels_token_type_ids,
     scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = (
        batch_input_ids.to(device), batch_attention_mask.to(device), batch_token_type_ids.to(device),
        labels_input_ids.to(device), labels_attention_mask.to(device), labels_token_type_ids.to(device),
        scene_labels_input_ids.to(device), scene_labels_attention_mask.to(device), scene_labels_token_type_ids.to(device),
    )

    batch_labels["ner"] = batch_labels["ner"].to(device)
    batch_labels["scene"] = batch_labels["scene"].to(device)


    # logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    logits = model(
        batch_input_ids, batch_attention_mask, batch_token_type_ids,
        labels_input_ids, labels_attention_mask, labels_token_type_ids,
        scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids,
    )

    loss = criterion(batch_labels, logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, dataloader, epoch, optimizer):
    model.train()

    def loss_fun(labels, preds):
        """
        labels: dict with keys 'ner' and 'scene'
            - labels['ner']: (batch_size, ent_type_size, seq_len, seq_len)
            - labels['scene']: (batch_size, scene_type_size)
        preds:
            - tuple: (logits_ner, logits_scene)
        """
        logits_ner, logits_scene = preds
        loss_ner = multilabel_categorical_crossentropy(
            labels["ner"].reshape(labels["ner"].size(0) * labels["ner"].size(1), -1),
            logits_ner.reshape(logits_ner.size(0) * logits_ner.size(1), -1)
        )

        loss_scene = multilabel_categorical_crossentropy(labels["scene"], logits_scene)

        # return loss_ner + loss_scene
        return loss_ner

    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        scheduler = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        loss = train_step(batch_data, model, optimizer, loss_fun)

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        if scheduler is not None:
            scheduler.step()

        pbar.set_description(
            f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

        if config["logger"] == "wandb" and batch_ind % config["log_interval"] == 0:
            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

    return avg_loss

"""valid"""
def valid_step(batch_valid, model):
    (batch_samples,
     batch_input_ids, batch_attention_mask, batch_token_type_ids,
     batch_labels,
     labels_input_ids, labels_attention_mask, labels_token_type_ids,
     scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = batch_valid

    (batch_input_ids, batch_attention_mask, batch_token_type_ids,
     labels_input_ids, labels_attention_mask, labels_token_type_ids,
     scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids) = (
        batch_input_ids.to(device), batch_attention_mask.to(device), batch_token_type_ids.to(device),
        labels_input_ids.to(device), labels_attention_mask.to(device), labels_token_type_ids.to(device),
        scene_labels_input_ids.to(device), scene_labels_attention_mask.to(device), scene_labels_token_type_ids.to(device),
    )

    batch_labels["ner"] = batch_labels["ner"].to(device)
    batch_labels["scene"] = batch_labels["scene"].to(device)


    with torch.no_grad():
        logits_ner, logits_scene = model(
            batch_input_ids, batch_attention_mask, batch_token_type_ids,
            labels_input_ids, labels_attention_mask, labels_token_type_ids,
            scene_labels_input_ids, scene_labels_attention_mask, scene_labels_token_type_ids
        )

    # sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)
    # 评估
    f1_ner, p_ner, r_ner = metrics.get_evaluate_fpr(logits_ner, batch_labels["ner"])
    f1_scene, p_scene, r_scene = metrics.get_scene_metrics(logits_scene, batch_labels["scene"])

    # 获取预测标签
    # scene_probs = torch.sigmoid(logits_scene)
    scene_preds = (logits_scene > 0).int().cpu().numpy()
    scene_trues = batch_labels["scene"].cpu().numpy()

    # ner_preds = logits_ner.cpu()
    # ner_trues = batch_labels["ner"].cpu()
    ner_preds = (logits_ner > 0).int().cpu()
    ner_trues = batch_labels["ner"].cpu()

    # return sample_f1, sample_precision, sample_recall
    return {
        "metrics": {
            "ner": {"f1": f1_ner, "p": p_ner, "r": r_ner},
            "scene": {"f1": f1_scene, "p": p_scene, "r": r_scene}
        },
        "ner_preds": ner_preds,
        "ner_trues": ner_trues,
        "scene_preds": scene_preds,
        "scene_trues": scene_trues,
        "samples": batch_samples
    }


def valid(model, dataloader):
    model.eval()

    # total_f1, total_precision, total_recall = 0., 0., 0.

    total_ner_f1 = total_ner_p = total_ner_r = 0.
    total_scene_f1 = total_scene_p = total_scene_r = 0.

    print_count = 0

    for batch_data in tqdm(dataloader, desc="Validating"):
        result = valid_step(batch_data, model)
        metrics_result = result["metrics"]

        total_ner_f1 += metrics_result["ner"]["f1"]
        total_ner_p += metrics_result["ner"]["p"]
        total_ner_r += metrics_result["ner"]["r"]

        total_scene_f1 += metrics_result["scene"]["f1"]
        total_scene_p += metrics_result["scene"]["p"]
        total_scene_r += metrics_result["scene"]["r"]

        # 🔍 打印预测 vs 真实
        # for i, sample in enumerate(result["samples"]):
        #     print("\n================= Sample =================")
        #     print("Text:", sample["text"])
        #
        #     # 1. 打印 scene 分类
        #     pred_scene_label_ids = result["scene_preds"][i]
        #     true_scene_label_ids = result["scene_trues"][i]
        #
        #     pred_scenes = [id2scene[j] for j, v in enumerate(pred_scene_label_ids) if v == 1] if id2scene else pred_scene_label_ids.tolist()
        #     true_scenes = [id2scene[j] for j, v in enumerate(true_scene_label_ids) if v == 1] if id2scene else true_scene_label_ids.tolist()
        #
        #     print("[Scene]")
        #     print("真实标签:", true_scenes)
        #     print("预测标签:", pred_scenes)
        #
        #     # 2. 打印 NER 识别
        #     if id2ent:
        #         ner_pred_matrix = result["ner_preds"][i]
        #         ner_true_matrix = result["ner_trues"][i]
        #         pred_entities = decode_ent(sample["text"], ner_pred_matrix, tokenizer, 0)
        #         true_entities = decode_ent(sample["text"], ner_true_matrix, tokenizer, 0)
        #         print("[NER]")
        #         print("真实实体:", true_entities)
        #         print("预测实体:", pred_entities)
        #
        #     print("=========================================")
        #     print_count += 1

    n = len(dataloader)
    avg_ner_f1 = total_ner_f1 / n
    avg_ner_p = total_ner_p / n
    avg_ner_r = total_ner_r / n

    avg_scene_f1 = total_scene_f1 / n
    avg_scene_p = total_scene_p / n
    avg_scene_r = total_scene_r / n

    print("\n***************** Evaluation Results *****************")
    print("[NER]")
    print(f'Precision: {avg_ner_p:.4f}, Recall: {avg_ner_r:.4f}, F1: {avg_ner_f1:.4f}')
    print("[Scene]")
    print(f'Precision: {avg_scene_p:.4f}, Recall: {avg_scene_r:.4f}, F1: {avg_scene_f1:.4f}')
    print("******************************************************")

    return {
        "ner_f1": avg_ner_f1,
        "scene_f1": avg_scene_f1
    }

def decode_ent(text, pred_matrix, tokenizer, threshold=0):
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}

    binary_matrix = (pred_matrix > threshold).astype(float)

    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)
    return ent_list

# 新增绘图函数
def plot_epoch_loss_curve(epoch_losses, save_dir="loss_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Avg Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_per_epoch.png"))
    plt.close()

    print(f"Loss curve saved to {os.path.join(save_dir, 'loss_per_epoch.png')}")

if __name__ == '__main__':
    if config["run_type"] == "train":
        train_dataloader, valid_dataloader = data_generator()

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

        max_avg_f1 = 0.0

        all_epoch_losses = []
        for epoch in range(hyper_parameters["epochs"]):
            avg_loss = train(model, train_dataloader, epoch, optimizer)
            all_epoch_losses.append(avg_loss)

            valid_results = valid(model, valid_dataloader)  # 返回 dict
            valid_ner_f1 = valid_results["ner_f1"]
            valid_scene_f1 = valid_results["scene_f1"]
            # avg_f1 = (valid_ner_f1 + valid_scene_f1) / 2
            avg_f1 = valid_ner_f1

            if avg_f1 > max_avg_f1:
                max_avg_f1 = avg_f1
                if avg_f1 > config["f1_2_save"]:  # 判断是否需要保存
                    model_state_num = len(glob.glob(model_state_dict_dir + "/ner_model_state_dict_*.pt"))
                    """f1 > 配置f1，保存输出权重"""
                    torch.save(model.state_dict(),
                               os.path.join(model_state_dict_dir, f"ner_model_state_dict_{model_state_num}.pt"))

            print(f"[Epoch {epoch}] Best Avg F1: {max_avg_f1:.4f}")
            print("******************************************")

            if config.get("logger", "") == "wandb":
                logger.log({
                    "valid_ner_f1": valid_ner_f1,
                    "valid_scene_f1": valid_scene_f1,
                    "valid_avg_f1": avg_f1,
                    "best_avg_f1": max_avg_f1
                })

            plot_epoch_loss_curve(all_epoch_losses)
