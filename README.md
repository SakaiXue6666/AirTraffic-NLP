# 📘 **智能空管系统（Intelligent Air Traffic Management System）**

本项目聚焦于民航**空中交通管制（ATC）**场景下的文本语义理解任务，目标是在语音识别之后，对管制员与飞行员之间的指令文本进行结构化解析与语义建模。

由于空管指令语言具有高度专业化、口语化读法特殊、数据稀缺且噪声较大的特点，通用语言模型或基于规则的方法在该场景下效果有限。针对这一问题，本项目面向空管领域，构建了一套空管文本理解的 NLP 系统。

本项目包含两个核心任务：

1. **实体命名识别（NER）**：提取航班号、机场、地点等关键字段
2. **场景识别（Scene Classification）**：识别飞行场景、状态类型等类别

系统整体用于构建基础的空管文本理解模块。

------

## 🚀 **项目结构与说明**

### **1. 实体命名识别（NER）训练**

使用 GlobalPointer + CLIP 编码器的结构，支持航班文本的实体抽取。

训练脚本：

```
train_clip_qk_fly_arg_onlyner.py
```

模型结构：

```
models/GlobalPointer_clip_qk_fly_arg.py
```

------

### **2. 场景识别（Scene Classification）训练**

使用 Qwen 语言模型作为分类 backbone，支持识别飞行状态/场景类别。

训练脚本：

```
train_qwen_fly_scene.py
```

模型结构：

```
models/Qwen_fly_scene_2_ner.py
```

------

### **3. 联合测试（NER + Scene）**

用于评估 NER + Scene 模型的联合推理能力。

测试脚本：

```
evaluate_fly_inter_GlobalNer_QwenCls.py
```

------

## 📦 **模型权重**

预训练模型放在：

```
pretrained_models/
```

训练后的结果权重、日志输出放在：

```
outputs/
```

------

## ⚙️ **模型配置**

所有训练/测试的配置文件存放于：

```
config_fly/
```

包含超参数设置、模型路径、训练策略等。

------

## 🛠️ **工具函数**

常用数据预处理、lowercase 工具等位于：

```
common/utils_lower.py
```
