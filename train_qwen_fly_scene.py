"""
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""

import os
import config_fly as config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForCausalLM
from common.utils_lower import Preprocessor, multilabel_categorical_crossentropy
from models.Qwen_fly_scene_2_ner import DataMaker, MyDataset, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import wandb
# from evaluate_2 import load_model
import time
import numpy as np
import matplotlib.pyplot as plt

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from torch.nn.utils.rnn import pad_sequence

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
tokenizer = AutoTokenizer.from_pretrained(config["qwen_path"], padding_side='left')
"""加载model"""
model = AutoModelForCausalLM.from_pretrained(
    config["qwen_path"],
    # quantization_config = quant_config,
    # device_map = "auto"
)

"""设置 LoRA 配置"""
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],  # 对 Q/V 矩阵加 LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

"""将 LoRA 应用到模型"""
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model = model.to(device)

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

    """初始化数据处理对象"""
    data_maker = DataMaker(tokenizer)

    """生成dataloader"""
    if data_type == "train":
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, 512, ent2id, scene2id, data_type="train")
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, 512, ent2id, scene2id, data_type="valid")
                                      )
        return train_dataloader, valid_dataloader
    else:
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, 512, ent2id, scene2id)
                                      )
        return valid_dataloader

"""初始化评估对象"""
metrics = MetricsCalculator()

"""train"""
def train_step(batch_train, model, optimizer, criterion):
    (batch_samples,
     batch_input_ids, batch_attention_mask, _,
     batch_labels, _) = batch_train

    (batch_input_ids, batch_attention_mask, batch_labels) = (
        batch_input_ids.to(device), batch_attention_mask.to(device), batch_labels.to(device),
    )

    logits_scene = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        labels=batch_labels
    )

    loss = logits_scene.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.watch(model)


def train(model, dataloader, epoch, optimizer):
    model.train()

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

        loss = train_step(batch_data, model, optimizer, None)

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
     batch_input_ids, batch_attention_mask, _,
     _, true_texts) = batch_valid

    (batch_input_ids, batch_attention_mask) = (
        batch_input_ids.to(device), batch_attention_mask.to(device)
    )

    with torch.no_grad():
        scene_ids = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # # 获取每个样本的 prompt 实际长度（非 padding 的 token 数）
    # prompt_lengths = batch_attention_mask.sum(dim=1)  # shape: (batch_size,)
    # # 去掉 prompt，只保留 response
    # output_ids = []
    # for i, prompt_len in enumerate(prompt_lengths):
    #     output_ids.append(scene_ids[i, prompt_len:])  # 从 prompt_len 开始截取

    # 假设 batch_input_ids.shape = (batch_size, seq_len)
    prompt_len = batch_input_ids.shape[1]
    # 去掉 prompt，只保留 response
    output_ids = [scene_ids[i, prompt_len:] for i in range(scene_ids.size(0))]

    # padded 到相同长度（可选，例如为了 batch_decode）
    output_ids = pad_sequence(output_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    predict_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    predict_labels = [predict_text.split("assistant\n")[1].split("、") if ("assistant\n" in predict_text) else predict_text.split("、") for predict_text in predict_texts]
    true_labels = [true_text.split("、") for true_text in true_texts]

    # print(predict_labels)

    f1_scene, p_scene, r_scene = metrics.get_scene_metrics(predict_labels, true_labels, scene_type_size, scene2id)

    return {
        "metrics": {
            # "ner": {"f1": f1_ner, "p": p_ner, "r": r_ner},
            "scene": {"f1": f1_scene, "p": p_scene, "r": r_scene}
        },
        # "ner_preds": ner_preds,
        # "ner_trues": ner_trues,
        "scene_preds": predict_labels,
        "scene_trues": true_labels,
        "samples": batch_samples
    }


def valid(model, dataloader):
    model.eval()

    total_scene_f1 = total_scene_p = total_scene_r = 0.

    for batch_data in tqdm(dataloader, desc="Validating"):
        result = valid_step(batch_data, model)
        metrics_result = result["metrics"]

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

    avg_scene_f1 = total_scene_f1 / n
    avg_scene_p = total_scene_p / n
    avg_scene_r = total_scene_r / n

    print("\n***************** Evaluation Results *****************")
    # print("[NER]")
    # print(f'Precision: {avg_ner_p:.4f}, Recall: {avg_ner_r:.4f}, F1: {avg_ner_f1:.4f}')
    print("[Scene]")
    print(f'Precision: {avg_scene_p:.4f}, Recall: {avg_scene_r:.4f}, F1: {avg_scene_f1:.4f}')
    print("******************************************************")

    return {
        # "ner_f1": avg_ner_f1,
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
            # valid_ner_f1 = valid_results["ner_f1"]
            valid_scene_f1 = valid_results["scene_f1"]
            # avg_f1 = (valid_ner_f1 + valid_scene_f1) / 2
            avg_f1 = valid_scene_f1

            if avg_f1 > max_avg_f1:
                max_avg_f1 = avg_f1
                if avg_f1 > config["f1_2_save"]:  # 判断是否需要保存
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    """f1 > 配置f1，保存输出权重"""
                    torch.save(model.state_dict(),
                               os.path.join(model_state_dict_dir, f"model_state_dict_{model_state_num}.pt"))

            print(f"[Epoch {epoch}] Best Avg F1: {max_avg_f1:.4f}")
            print("******************************************")

            if config.get("logger", "") == "wandb":
                logger.log({
                    # "valid_ner_f1": valid_ner_f1,
                    "valid_scene_f1": valid_scene_f1,
                    "valid_avg_f1": avg_f1,
                    "best_avg_f1": max_avg_f1
                })

            plot_epoch_loss_curve(all_epoch_losses)