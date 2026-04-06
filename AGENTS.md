# Chandra OCR Fine-Tuning - 架构概述

## 1. 项目概述

Chandra是一个基于Qwen3-VL-8B架构的OCR模型，专为从文档图像中提取文本而设计。本代码库提供了一个生产就绪的微调脚本，使用Unsloth和LoRA技术进行高效微调。

### 核心功能
- 使用LoRA技术进行高效微调，仅训练少量参数
- 支持本地和HuggingFace Hub数据集
- 自动检测和转换多种数据集格式
- 内置早期停止机制防止过拟合
- 计算OCR指标（CER/WER）进行性能评估
- 支持多种模型格式导出（LoRA适配器、16位合并模型、GGUF等）

### 技术栈
- **深度学习框架**: PyTorch 2.10+
- **模型库**: Transformers 4.57.6+
- **高效训练**: Unsloth 2026.2.1+
- **LoRA支持**: PEFT 0.18.1+
- **训练器**: TRL 0.24.0+
- **数据集处理**: Datasets 4.3.0+
- **OCR指标**: Jiwer 4.0.0+

## 2. 代码结构与主要组件

### 主要文件

| 文件名 | 功能描述 |
|--------|----------|
| `train_chandra.py` | 主微调脚本，包含完整的训练流程 |
| `train_chandra_v2.py` | 微调脚本的第二个版本（扩展功能） |
| `inference_chandra.py` | 推理脚本，用于使用微调后的模型进行OCR |
| `convert-xml-chandra-hf.py` | 将PAGE-XML/ALTO-XML注释转换为HuggingFace数据集 |
| `requirements.txt` | 项目依赖项 |
| `README.md` | 详细文档和使用说明 |

### 训练流程

`train_chandra.py`实现了以下核心流程：

1. **模型加载**: 使用Unsloth的`FastVisionModel`加载Chandra模型和处理器
2. **配置验证**: 验证模型配置与预期参数匹配
3. **LoRA设置**: 应用LoRA适配器到模型的视觉和语言层
4. **数据集准备**: 加载、检测格式并转换为Qwen3-VL对话格式
5. **预训练推理测试**: 验证模型在训练前的性能
6. **训练**: 使用SFTTrainer进行训练，支持早停和检查点恢复
7. **模型保存**: 保存LoRA适配器、分词器和处理器
8. **后训练推理**: 比较训练前后的性能

## 3. 构建与命令

### 创建虚拟环境(项目初始化时执行一次)
```bash
# macOS/Linux (zsh/bash)
python -m venv .venv
```

### 激活虚拟环境
```bash
# macOS/Linux (zsh/bash)
source .venv/bin/activate
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练命令

#### 基本用法
```bash
# 使用HuggingFace数据集进行微调
python train_chandra.py --dataset_name unsloth/LaTeX_OCR --output_dir ./chandra_output

# 使用本地数据集并启用早停
python train_chandra.py --dataset_dir ./my_ocr_dataset --output_dir ./chandra_output --eval_steps 50 --early_stopping_patience 3

# 恢复中断的训练
python train_chandra.py --dataset_name unsloth/LaTeX_OCR --output_dir ./chandra_output --resume_from_checkpoint latest
```

#### 导出模型
```bash
# 保存合并的16位模型和GGUF格式
python train_chandra.py --dataset_dir ./my_hf_dataset --output_dir ./chandra_production --save_merged_16bit --save_gguf

# 推送到HuggingFace Hub
python train_chandra.py --dataset_dir ./my_hf_dataset --output_dir ./chandra_production --push_to_hub myuser/chandra-ocr-finetuned --hf_token $HF_TOKEN
```

### 数据转换命令
```bash
# 将XML注释转换为HuggingFace数据集
python convert-xml-chandra-hf.py --input_dir /path/to/xml_and_images --output_dir ./chandra_dataset --include_full_pages --include_paragraphs
```

### 验证命令
```bash
# 语法检查
python -m py_compile train_chandra.py

# 导入检查
python -c "from train_chandra import load_model, verify_config, train; print('OK')"

# 快速帮助
python train_chandra.py --help

# 快速试运行
python train_chandra.py --dataset_name unsloth/LaTeX_OCR --output_dir ./test_chandra --max_steps 1 --early_stopping_patience 0 --skip_pre_eval --verbose
```

## 4. 代码风格与最佳实践

### 代码结构
- 脚本采用模块化设计，分为10个主要部分
- 延迟导入重型库（如torch、unsloth）以加快`--help`响应速度
- 使用类型提示提高代码可读性和可维护性
- 详细的日志记录，支持不同的日志级别

### 命名约定
- 函数名使用下划线分隔的小写字母（snake_case）
- 常量使用全大写字母，下划线分隔
- 参数名使用描述性名称，避免缩写

### 最佳实践
- 使用4位量化减少内存占用
- 启用梯度检查点以节省显存
- 使用混合精度训练（bf16）提高训练速度
- 自动检测数据集格式，支持简单格式和finevision格式
- 实现早期停止机制防止过拟合
- 提供详细的文档和使用示例

## 5. 测试

### 验证机制
- 模型配置验证：确保加载的模型符合Chandra的预期配置
- 预训练推理测试：在训练前验证模型的前向传递
- 后训练推理测试：比较训练前后的性能差异

### 评估指标
- **损失函数**: 交叉熵损失
- **OCR指标**: CER（字符错误率）和WER（词错误率）
  - 使用jiwer库计算
  - 教师强制（teacher-forced）指标，比实际生成更乐观但仍能可靠跟踪训练进度
  - 可通过`--no_cer_wer`禁用

### 早停机制
- 当`early_stopping_patience > 0`时自动启用
- 数据集自动分割为训练/验证集（默认10%验证集）
- 每`eval_steps`步评估一次模型
- 如果`eval_loss`（或指定的其他指标）在`patience`次评估中没有改善，则停止训练
- 自动加载最佳检查点

## 6. 安全性与数据保护

### 数据处理
- 支持本地和HuggingFace Hub数据集
- 自动检测数据集格式，避免手动格式转换错误
- 实现内存高效的数据处理，避免加载整个数据集到内存

### 模型安全
- 使用`trust_remote_code=True`加载模型时进行配置验证
- 提供跳过验证的选项，但默认启用
- 支持检查点恢复，确保训练中断后可以继续

### 环境变量
- 使用环境变量（如`HF_TOKEN`）管理敏感信息
- 不在代码中硬编码敏感信息

## 7. 配置管理

### 环境要求
- Python 3.12+
- CUDA支持（推荐24GB+ VRAM）

### 模型配置
Chandra模型的关键配置参数：
- 架构：Qwen3VLForConditionalGeneration
- 基础模型：Qwen/Qwen3-VL-8B-Instruct
- 最大序列长度：2048
- 图像标记ID：151655
- 视觉开始/结束标记ID：151652/151653
- 补丁大小：16
- 空间合并大小：2
- 数据类型：bfloat16

### 训练配置
- 默认学习率：2e-4
- 默认批量大小：1（每个GPU）
- 默认梯度累积步数：8
- 默认LoRA参数：r=16, alpha=16, dropout=0.05
- 默认图像分辨率：2048px（最长边）

## 8. 使用示例：端到端工作流

### 步骤1：转换XML注释到HF数据集
```bash
python convert-xml-chandra-hf.py --input_dir /path/to/xml_and_images --output_dir ./chandra_dataset --include_full_pages --include_paragraphs --val_ratio 0.1
```

### 步骤2：微调Chandra模型
```bash
python train_chandra.py --dataset_dir ./chandra_dataset/hf_dataset_chandra --output_dir ./chandra_finetuned --num_epochs 3 --eval_steps 100 --early_stopping_patience 5 --verbose
```

### 步骤3：使用微调后的模型
```python
from train_chandra import run_ocr, load_model
model, tokenizer, processor = load_model("./chandra_finetuned")
print(run_ocr("document.png", model=model, tokenizer=tokenizer, processor=processor))
```

## 9. 训练建议

1. **学习率**: 从2e-4开始，如过拟合则降低到5e-5
2. **视觉层微调**: 对于领域适应（手写、特定布局）保持`--finetune_vision`，对于分布内数据使用`--no_finetune_vision`
3. **批量大小**: 使用RTX 3090（24GB）时，`batch_size=2` + `grad_accum=8`可获得16的有效批量
4. **图像分辨率**: 2048px是24GB VRAM的最佳选择，24GB+ GPU可使用4096px获得更好的准确性
5. **温度**: 推理时使用0.3以获得确定性的OCR输出
6. **早停**: 推荐使用，耐心值=3，eval_steps=50是良好的起点
7. **序列长度**: Chandra的最大序列长度为2048令牌，过长的文档可能会被截断

## 10. 扩展功能

### train_chandra_v2.py
提供了扩展功能的第二个版本的微调脚本，可能包含额外的特性和改进。

### inference_chandra.py
专门用于推理的脚本，提供了更简单的接口来使用微调后的Chandra模型进行OCR任务。

## 总结

Chandra OCR微调代码库提供了一个高效、灵活且生产就绪的解决方案，用于微调基于Qwen3-VL-8B的OCR模型。通过使用Unsloth和LoRA技术，它能够在有限的计算资源上实现高效的微调，并支持多种数据集格式和模型导出选项。详细的文档和丰富的配置选项使其适用于各种OCR微调任务。