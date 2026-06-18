#!/usr/bin/env python3
"""
Qwen2.5 Mac M 系列微调脚本 — 基于 MLX
========================================

在 Mac M1/M2/M3/M4 上使用 Apple MLX 框架高效微调模型。

限制:
  - 无 CUDA，使用 Metal GPU
  - 统一内存架构 (CPU/GPU 共享 RAM)

依赖:
    pip install mlx mlx-lm
    (国内) pip install modelscope

用法:
    python train_mac.py --config config_mac.yaml
    python train_mac.py --model Qwen/Qwen2.5-0.5B-Instruct --hf-endpoint https://hf-mirror.com

默认模型:
    Qwen/Qwen2.5-0.5B-Instruct (约 0.5B 参数，~1GB 内存)
    16GB Mac 可用; 24GB 以上可换 1.5B/3B

数据格式:
    Alpaca JSONL → 自动转为 OpenAI messages 格式
    {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
    mlx-lm 的 ChatDataset 只对 assistant 部分算 loss，训练正确。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("❌ 需要 pyyaml: pip install pyyaml")
    sys.exit(1)


def load_alpaca_jsonl(path: str, combine_input: bool = True) -> list[dict]:
    """
    读取 Alpaca 格式 JSONL → OpenAI messages 格式。

    输出:
        {"messages": [
            {"role": "user",      "content": "{instruction}"},
            {"role": "assistant", "content": "{output}"}
        ]}
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  [{path}] 读取 {len(records)} 条")

    out = []
    for r in records:
        inst = r.get("instruction", "")
        inp = r.get("input", "")
        out_text = r.get("output", "")
        if combine_input and inp:
            inst = f"{inst}\n\n{inp}"
        out.append({
            "messages": [
                {"role": "user",      "content": inst},
                {"role": "assistant", "content": out_text},
            ]
        })
    return out


def load_sharegpt_jsonl(path: str) -> list[dict]:
    """
    读取 ShareGPT 格式 JSONL → OpenAI messages 格式。
    输入: {"conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  [{path}] 读取 {len(records)} 条")

    out = []
    for r in records:
        messages = []
        for turn in r.get("conversations", []):
            role = turn.get("from", "")
            content = turn.get("value", "")
            if role == "human":
                messages.append({"role": "user",      "content": content})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": content})
            elif role == "system":
                messages.append({"role": "system",   "content": content})
        out.append({"messages": messages})
    return out


def load_rag_jsonl(path: str, combine_input: bool = True) -> list[dict]:
    """
    读取 RAG 格式 JSONL to OpenAI messages 格式。
    输入: {"context": "...", "instruction": "...", "input": "...", "output": "..."}
    当 context 非空时添加 system message 作为参考规则。
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  [{path}] 读取 {len(records)} 条 (RAG)")

    out = []
    for r in records:
        context = r.get("context", "").strip()
        inst = r.get("instruction", "")
        inp = r.get("input", "")
        out_text = r.get("output", "")
        if combine_input and inp:
            inst = f"{inst}\n\n{inp}"
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"参考规则:\n{context}"
            })
        messages.append({"role": "user", "content": inst})
        messages.append({"role": "assistant", "content": out_text})
        out.append({"messages": messages})
    return out


def prepare_data(
    train_path: str,
    eval_path: str | None,
    data_dir: str,
    fmt: str = "alpaca",
    combine_input: bool = True,
    split_ratio: float = 0.9,
):
    """将项目数据转为 mlx-lm ChatDataset 格式，写入 data_dir"""
    loader_map = {
        "alpaca":   lambda p: load_alpaca_jsonl(p, combine_input),
        "sharegpt": load_sharegpt_jsonl,
        "rag":      lambda p: load_rag_jsonl(p, combine_input),
    }
    loader = loader_map.get(fmt)
    if not loader:
        raise ValueError(f"不支持的数据格式: {fmt} (可选: alpaca, sharegpt, rag)")

    all_data = loader(train_path)
    if eval_path and os.path.exists(eval_path):
        train_data = all_data
        eval_data = loader(eval_path)
    else:
        split = int(len(all_data) * split_ratio)
        train_data = all_data[:split]
        eval_data = all_data[split:]

    print(f"  训练集: {len(train_data)} 条  验证集: {len(eval_data)} 条")
    if train_data:
        print(f"  格式示例: messages[0]={train_data[0]['messages'][0]['role']}, "
              f"messages[1]={train_data[0]['messages'][1]['role']}")

    os.makedirs(data_dir, exist_ok=True)
    for name, rows in [("train", train_data), ("valid", eval_data)]:
        p = os.path.join(data_dir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  写入 {p}")
    return train_data, eval_data


def download_model(model_name: str, hf_endpoint: str | None = None) -> str | None:
    """用 huggingface_hub 预下载模型到缓存目录"""
    from huggingface_hub import snapshot_download, HfApi

    print(f"  下载模型: {model_name} ...")

    old = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"  镜像源: {hf_endpoint}")

    try:
        local_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=[
                "*.json", "*.safetensors", "*.py",
                "tokenizer.model", "*.tiktoken", "*.txt",
            ],
            resume_download=True,
        )
        print(f"  下载完成: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print(f"  建议: (1) --hf-endpoint https://hf-mirror.com")
        print(f"        (2) pip install modelscope")
        return None
    finally:
        if hf_endpoint and not old:
            os.environ.pop("HF_ENDPOINT", None)


def build_mlx_config(model: str, data_dir: str, output_dir: str, **kw) -> dict:
    """生成 mlx-lm 0.31+ 配置字典"""
    return {
        "model": model,
        "train": True,
        "fine_tune_type": kw.get("fine_tune_type", "lora"),
        "optimizer": kw.get("optimizer", "adamw"),
        "data": data_dir,
        "seed": kw.get("seed", 42),
        "num_layers": kw.get("num_layers", -1),
        "batch_size": kw.get("batch_size", 1),
        "iters": kw.get("iters", 1000),
        "val_batches": kw.get("val_batches", -1),
        "learning_rate": kw.get("lr", 2e-4),
        "steps_per_report": kw.get("steps_per_report", 5),
        "steps_per_eval": kw.get("steps_per_eval", 100),
        "adapter_path": os.path.join(output_dir, "adapter"),
        "save_every": kw.get("save_every", 500),
        "max_seq_length": kw.get("max_seq_length", 2048),
        "grad_accumulation_steps": kw.get("grad_accum", 8),
        # 🔑 关键: 只对 assistant 回复算 loss
        "mask_prompt": True,
        "grad_checkpoint": kw.get("grad_checkpoint", False),
        "lr_schedule": kw.get("lr_schedule"),
        "lora_parameters": {
            "rank": kw.get("lora_rank", 16),
            "scale": kw.get("lora_scale", 20.0),
            "dropout": kw.get("lora_dropout", 0.0),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mac 微调 (MLX)")
    parser.add_argument("--config", help="YAML 配置文件")

    # 模型
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hf-endpoint", default=None,
                        help="HF 镜像: https://hf-mirror.com")
    parser.add_argument("--download-only", action="store_true",
                        help="只下载模型，不训练")

    # 数据
    parser.add_argument("--train-file", default="./data/train.jsonl")
    parser.add_argument("--eval-file", default="./data/eval.jsonl")
    parser.add_argument("--data-format", choices=["alpaca", "sharegpt", "rag"], default="rag")
    parser.add_argument("--split-ratio", type=float, default=0.9)

    # 训练
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-batches", type=int, default=-1)
    parser.add_argument("--fine-tune-type", choices=["lora", "dora", "full"],
                        default="lora")
    parser.add_argument("--optimizer", default="adamw")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    # 输出
    parser.add_argument("--output-dir", default="./outputs_mac")
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--dry-run", action="store_true", help="只准备数据")
    args = parser.parse_args()

    # ---- YAML 覆盖 ----
    cfg = vars(args)
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            yml = yaml.safe_load(f) or {}

        # YAML→CLI 键名映射：YAML 里的命名与 argparse 不一致，需要翻译
        YAML_KEY_MAP = {
            "learning_rate": "lr",
            "grad_accumulation_steps": "grad_accum",
        }

        for k, v in yml.items():
            # 直接匹配 argparse 字段
            if k in cfg and v is not None:
                cfg[k] = v
            # 需要翻译键名的字段
            elif k in YAML_KEY_MAP and v is not None:
                cfg[YAML_KEY_MAP[k]] = v
            # 嵌套的 lora_parameters.rank → lora_rank, etc.
            elif k == "lora_parameters" and isinstance(v, dict):
                for lk, lv in v.items():
                    cfg_key = f"lora_{lk}"  # rank→lora_rank, scale→lora_scale
                    if cfg_key in cfg:
                        cfg[cfg_key] = lv
            # lr_schedule 和 grad_checkpoint YAML 中有但 argparse 中没有对应字段
            elif k in ("lr_schedule", "grad_checkpoint") and v is not None:
                cfg[k] = v

    # ---- HF 镜像 ----
    hf_endpoint = cfg["hf_endpoint"] or os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"  HF_ENDPOINT: {hf_endpoint}")

    # ---- 打印配置 ----
    print("=" * 60)
    print("Qwen2.5 Mac M-Series Fine-tuning  (mlx-lm)")
    print("=" * 60)
    for k in ["model", "fine_tune_type", "lr", "iters", "batch_size",
              "grad_accum", "num_layers", "lora_rank", "max_seq_length",
              "output_dir"]:
        print(f"  {k}: {cfg[k]}")
    print(f"  mask_prompt: true (只对 assistant 回复算 loss)")
    if hf_endpoint:
        print(f"  hf_endpoint: {hf_endpoint}")
    print()

    # ---- 下载 ----
    if cfg["download_only"]:
        print("[仅下载模型]")
        download_model(cfg["model"], hf_endpoint or None)
        return

    # ---- 准备数据 ----
    print("[1/3] 准备数据 (messages 格式 + mask_prompt)...")
    data_dir = os.path.join(cfg["output_dir"], "_mlx_data")
    prepare_data(
        train_path=cfg["train_file"],
        eval_path=cfg.get("eval_file"),
        data_dir=data_dir,
        fmt=cfg["data_format"],
        combine_input=True,
        split_ratio=cfg["split_ratio"],
    )

    if cfg["dry_run"]:
        print(f"\nDry-run, 数据已准备: {data_dir}/{{train,valid}}.jsonl")
        return

    # ---- 生成配置 ----
    print("\n[2/3] 生成 mlx-lm 配置 ...")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    mlx_cfg = build_mlx_config(**cfg, data_dir=data_dir)

    config_path = os.path.join(cfg["output_dir"], "_mlx_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(mlx_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  配置写入 {config_path}")
    print(f"  mask_prompt: {mlx_cfg['mask_prompt']}")

    # ---- 预下载模型 ----
    print("\n  检查模型缓存 ...")
    download_model(cfg["model"], hf_endpoint or None)

    # ---- 训练 ----
    print("\n[3/3] 启动 mlx-lm 训练 ...")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", config_path]
    print(f"  运行: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "true"
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint

    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"  {line.rstrip()}")

    proc.wait()
    elapsed = time.time() - start

    if proc.returncode == 0:
        adapter_dir = os.path.join(cfg["output_dir"], "adapter")
        print(f"\n{'='*60}")
        print(f"训练完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Adapter: {adapter_dir}")
        print("=" * 60)
    else:
        print(f"\n训练失败 (exit {proc.returncode})")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
