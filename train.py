#!/usr/bin/env python3
"""
Qwen2.5 量化模型微调脚本 (Unsloth)
======================================
支持 4-bit QLoRA / 16-bit LoRA 微调 Qwen3.5 小模型。
支持 Alpaca / ShareGPT / Text 格式的数据。

用法:
    python train.py --config config.yaml
    python train.py --config config.yaml --model Qwen/Qwen2.5-1.5B-Instruct
"""

import os
import sys
import json
import yaml
import argparse
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# 必须在其他库之前导入 unsloth，以确保 patching 正确生效
import unsloth  # noqa: F401
import torch

# ============================================================
# Logging 配置
# ============================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置加载
# ============================================================
@dataclass
class Config:
    model: Dict[str, Any] = field(default_factory=dict)
    lora: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 2048
    hf_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # 展平嵌套结构
        flat = {}
        for section in ["model", "lora", "training", "data", "output"]:
            flat[section] = raw.get(section, {})
        flat["max_seq_length"] = raw.get("max_seq_length", 2048)
        flat["hf_token"] = raw.get("hf_token", None)
        return cls(**flat)


# ============================================================
# 数据加载
# ============================================================
def load_alpaca_dataset(data_path: str, combine_input: bool = True):
    """
    加载 Alpaca 格式数据集（JSON / JSONL）
    格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    from datasets import Dataset

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            records = json.load(f)

    logger.info(f"Loaded {len(records)} records from {data_path}")

    def format_alpaca(example):
        if combine_input and example.get("input"):
            instruction = f"{example['instruction']}\n\n{example['input']}"
        else:
            instruction = example.get("instruction", "")
        output = example.get("output", "")

        # Alpaca 格式模板
        text = (
            f"以下是用户指令的描述。\n\n"
            f"### 指令:\n{instruction}\n\n"
            f"### 回答:\n{output}"
        )
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    return dataset


def load_sharegpt_dataset(data_path: str):
    """
    加载 ShareGPT 格式数据集（JSON / JSONL）
    格式: {"conversations": [{"from": "human"/"gpt", "value": "..."}]}
    """
    from datasets import Dataset

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            records = json.load(f)

    logger.info(f"Loaded {len(records)} records from {data_path}")

    def format_sharegpt(example):
        conversations = example.get("conversations", [])
        texts = []
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            if role == "human":
                texts.append(f"### 用户:\n{content}")
            elif role == "gpt":
                texts.append(f"### 助手:\n{content}")
            elif role == "system":
                texts.append(f"### 系统:\n{content}")
        return {"text": "\n\n".join(texts)}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_sharegpt, remove_columns=dataset.column_names)
    return dataset


def load_text_dataset(data_path: str):
    """加载纯文本数据集（每行一个文本，或 JSON 中 text 字段）"""
    from datasets import Dataset

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        elif data_path.endswith(".json"):
            records = json.load(f)
        else:
            # 纯文本，每行一个
            texts = [line.strip() for line in f if line.strip()]
            records = [{"text": t} for t in texts]

    logger.info(f"Loaded {len(records)} records from {data_path}")
    return Dataset.from_list(records)


def load_rag_dataset(data_path: str):
    """
    加载 RAG 格式数据集（JSON / JSONL）
    格式: {"context": "...", "instruction": "...", "input": "...", "output": "..."}

    context: 可选的规则上下文（为空时退化为标准 QA）
    模板:
      {% if context %}
      ### 参考规则:
      {context}

      {% endif %}
      ### 指令:
      {instruction}

      ### 回答:
      {output}
    """
    from datasets import Dataset

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            records = json.load(f)

    logger.info(f"Loaded {len(records)} RAG records from {data_path}")

    def format_rag(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        context = example.get("context", "").strip()

        if inp:
            instruction = f"{instruction}\n\n{inp}"

        if context:
            text = (
                f"### 参考规则:\n{context}\n\n"
                f"### 指令:\n{instruction}\n\n"
                f"### 回答:\n{output}"
            )
        else:
            text = (
                f"以下是用户指令的描述。\n\n"
                f"### 指令:\n{instruction}\n\n"
                f"### 回答:\n{output}"
            )
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_rag, remove_columns=dataset.column_names)
    return dataset


def load_dataset(cfg: Config) -> tuple:
    """加载训练集和验证集"""
    data_cfg = cfg.data
    train_path = data_cfg.get("train_file", "./data/train.jsonl")
    eval_path = data_cfg.get("eval_file", None)
    fmt = data_cfg.get("format", "alpaca")
    combine = data_cfg.get("alpaca_combine_input", True)

    loaders = {
        "alpaca": lambda p: load_alpaca_dataset(p, combine),
        "sharegpt": load_sharegpt_dataset,
        "text": load_text_dataset,
        "rag": load_rag_dataset,
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported data format: {fmt}, 支持: {list(loaders.keys())}")

    # 加载训练集
    dataset = loader(train_path)

    # 划分训练/验证集
    if eval_path and os.path.exists(eval_path):
        eval_dataset = loader(eval_path)
        train_dataset = dataset
    else:
        split_ratio = data_cfg.get("train_split_ratio", 0.9)
        split = dataset.train_test_split(test_size=1 - split_ratio, seed=cfg.training.get("seed", 42))
        train_dataset = split["train"]
        eval_dataset = split["test"]

    logger.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    return train_dataset, eval_dataset


# ============================================================
# 模型与 Tokenizer 加载
# ============================================================
def load_model_and_tokenizer(cfg: Config):
    """加载量化模型和 tokenizer"""
    from unsloth import FastLanguageModel
    import torch

    model_cfg = cfg.model
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_seq_length = cfg.max_seq_length
    load_in = model_cfg.get("load_in", "4bit")
    hf_token = cfg.hf_token

    # 量化配置
    dtype = None
    load_in_4bit = False
    load_in_8bit = False

    if load_in == "4bit":
        load_in_4bit = True
    elif load_in == "8bit":
        load_in_8bit = True
    elif load_in == "16bit":
        dtype = torch.bfloat16 if cfg.training.get("bf16", True) else torch.float16
    else:
        raise ValueError(f"不支持的加载方式: {load_in}，可选: 4bit, 8bit, 16bit")

    logger.info(f"Loading model: {model_name} ({load_in})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        token=hf_token,
        device_map="auto",
    )

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # 配置 tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # 为 DeepSeek 设置默认 chat template
        tokenizer.chat_template = (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}\n\n### 指令:\n{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}\n\n### 回答:\n{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}\n\n### 回答:\n{% endif %}"
        )

    return model, tokenizer


def _get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


# ============================================================
# LoRA 配置
# ============================================================
def setup_lora(model, cfg: Config):
    """配置 LoRA / QLoRA"""
    from unsloth import FastLanguageModel

    lora_cfg = cfg.lora

    logger.info(f"Configuring LoRA: rank={lora_cfg.get('r', 16)}, "
                f"alpha={lora_cfg.get('lora_alpha', 16)}, "
                f"dropout={lora_cfg.get('lora_dropout', 0)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get("target_modules"),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_rslora=lora_cfg.get("use_rslora", False),
        use_dora=lora_cfg.get("use_dora", False),
        loftq_config=None,  # LoftQ 可选
        max_seq_length=cfg.max_seq_length,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} ({trainable / sum(p.numel() for p in model.parameters()) * 100:.2f}%)")

    return model


# ============================================================
# 训练
# ============================================================
def train(model, tokenizer, train_dataset, eval_dataset, cfg: Config):
    """执行训练"""
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    from transformers import Trainer

    training_cfg = cfg.training

    # 构建 TrainingArguments 参数
    args = TrainingArguments(
        output_dir=cfg.output.get("output_dir", "./outputs"),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        max_steps=training_cfg.get("max_steps", -1),
        learning_rate=training_cfg.get("learning_rate", 2.0e-4),
        optim=training_cfg.get("optim", "adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        seed=training_cfg.get("seed", 42),
        logging_steps=training_cfg.get("logging_steps", 1),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 50),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 50),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", is_bfloat16_supported()),
        report_to=training_cfg.get("report_to", "none"),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=True,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    # 预分词数据集

    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=cfg.max_seq_length,
            return_attention_mask=True,
        )
        # 为 CausalLM 设置 labels = input_ids（深拷贝避免共享引用）
        labels = [ids.copy() for ids in tokenized["input_ids"]]
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )

    # 自定义数据整理器 - 手动 padding
    def data_collator(features):
        """手动 padding，兼容所有 HuggingFace tokenizer"""
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        # 收集各字段并转 tensor
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # 找到 batch 中最大序列长度
        max_len = max(ids.size(0) for ids in input_ids)
        batch_size = len(features)

        # 分配 padding 后的张量
        padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i in range(batch_size):
            seq_len = input_ids[i].size(0)
            padded_input_ids[i, :seq_len] = input_ids[i]
            padded_attention_mask[i, :seq_len] = attention_mask[i]
            padded_labels[i, :seq_len] = labels[i]

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer_stats = trainer.train()

    logger.info(f"Training completed. Loss: {trainer_stats.training_loss:.4f}")

    # 保存 LoRA 权重（用 unsloth 的方法绕过 pickle 兼容问题）
    logger.info(f"Saving LoRA weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练指标
    with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
        json.dump({
            "train_loss": trainer_stats.training_loss,
            "train_runtime": trainer_stats.metrics.get("train_runtime", 0),
            "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second", 0),
        }, f, ensure_ascii=False, indent=2)

    return trainer


# ============================================================
# 合并模型
# ============================================================
def save_merged_model(model, tokenizer, cfg: Config):
    """合并 LoRA 权重到基础模型并保存"""
    from unsloth import FastLanguageModel

    output_cfg = cfg.output
    save_format = output_cfg.get("save_format", "hf")
    merged_dir = output_cfg.get("merged_dir", "./outputs/merged_model")

    logger.info(f"Saving merged model to {merged_dir} (format: {save_format})")

    if save_format in ("hf", "both"):
        # 保存为 HuggingFace 格式（合并权重）
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved (16-bit) at {merged_dir}")

    if save_format in ("gguf", "both"):
        # 保存为 GGUF 格式（供 ollama/llama.cpp 使用）
        gguf_dir = merged_dir + "_gguf"
        model.save_pretrained_gguf(gguf_dir, tokenizer)
        logger.info(f"GGUF model saved at {gguf_dir}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 + Unsloth 微调脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--data_file", type=str, default=None, help="覆盖训练数据路径")
    parser.add_argument("--only_inference", action="store_true", help="仅运行推理（不训练）")
    args = parser.parse_args()

    # 加载配置
    cfg = Config.from_yaml(args.config)

    # CLI 参数覆盖
    if args.model:
        cfg.model["name"] = args.model
    if args.output_dir:
        cfg.output["output_dir"] = args.output_dir
    if args.data_file:
        cfg.data["train_file"] = args.data_file

    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model.get('name')}")
    logger.info(f"Load in: {cfg.model.get('load_in')}")
    logger.info(f"LoRA rank: {cfg.lora.get('r', 16)}")
    logger.info(f"Max seq length: {cfg.max_seq_length}")
    logger.info(f"Output dir: {cfg.output.get('output_dir')}")
    logger.info(f"Data file: {cfg.data.get('train_file')}")
    logger.info(f"Data format: {cfg.data.get('format')}")
    logger.info("=" * 60)

    # 检测 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU detected! Training on CPU will be very slow.")

    if args.only_inference:
        # 仅推理模式
        from inference import run_inference
        model, tokenizer = load_model_and_tokenizer(cfg)
        model = setup_lora(model, cfg)
        run_inference(model, tokenizer)
        return

    # 1. 加载数据集
    train_dataset, eval_dataset = load_dataset(cfg)

    # 2. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 3. 配置 LoRA
    model = setup_lora(model, cfg)

    # 4. 训练
    trainer = train(model, tokenizer, train_dataset, eval_dataset, cfg)

    # 5. 合并并保存模型（可选）
    save_merged = cfg.output.get("save_merged", True)
    if save_merged:
        save_merged_model(model, tokenizer, cfg)

    logger.info("All done! 🎉")


if __name__ == "__main__":
    main()
