#!/usr/bin/env python3
"""
推理脚本 - 使用微调后的模型进行对话生成，支持语义向量检索
============================================================

用法:
    # 使用 LoRA 权重（需要 adapter 目录）
    python inference.py --lora_dir ./outputs \
        --model_name Qwen/Qwen2.5-1.5B-Instruct

    # 纯向量检索（不微调）
    python inference.py --no-adapter --vector --interactive

    # 启动交互模式
    python inference.py --lora_dir ./outputs --vector --interactive
"""

import os
import re
import sys
import json
import argparse
import logging
from typing import Optional, List, Dict
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model(
    model_name: str = None,
    model_dir: str = None,
    lora_dir: str = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """加载模型"""
    from unsloth import FastLanguageModel

    if model_dir:
        logger.info(f"Loading merged model from {model_dir}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    else:
        model_name = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        logger.info(f"Loading base model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        if lora_dir:
            logger.info(f"Loading LoRA adapter from {lora_dir}")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
            logger.info(f"LoRA adapter loaded successfully")

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_rag_prompt(instruction: str, context: str = "", input_text: str = "") -> str:
    """根据是否有上下文构造对应 prompt"""
    if input_text:
        instruction = f"{instruction}\n\n{input_text}"

    if context:
        return (
            f"### 参考规则:\n{context}\n\n"
            f"### 指令:\n{instruction}\n\n"
            f"### 回答:\n"
        )
    else:
        return (
            f"以下是用户指令的描述。\n\n"
            f"### 指令:\n{instruction}\n\n"
            f"### 回答:\n"
        )


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    context: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """生成回复"""
    prompt_text = get_rag_prompt(instruction, context, input_text)

    inputs = tokenizer([prompt_text], return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt_text):].strip()
    return response


def chat_format(messages: List[Dict[str, str]]) -> str:
    """对话列表 → 模型输入文本（无上下文）"""
    text = "以下是用户指令的描述。\n\n"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            text += f"### 系统:\n{content}\n\n"
        elif role == "user":
            text += f"### 指令:\n{content}\n\n"
        elif role == "assistant":
            text += f"### 回答:\n{content}\n\n"
    text += "### 回答:\n"
    return text


def chat_format_rag(messages: List[Dict[str, str]], context: str = "") -> str:
    """RAG 对话格式（带参考规则）"""
    if context:
        text = f"### 参考规则:\n{context}\n\n"
    else:
        text = "以下是用户指令的描述。\n\n"

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            text += f"### 系统:\n{content}\n\n"
        elif role == "user":
            text += f"### 指令:\n{content}\n\n"
        elif role == "assistant":
            text += f"### 回答:\n{content}\n\n"
    text += "### 回答:\n"
    return text


def interactive_mode(model, tokenizer, vdb=None):
    """交互式对话模式"""
    title = "  桌游规则助手 (向量检索)" if vdb else "  桌游规则助手"
    print("\n" + "=" * 60)
    print(title)
    print("  输入 'quit' 退出, 'clear' 清空历史")
    if vdb:
        games = ", ".join(vdb.get_game_names())
        print(f"  向量索引: {games}")
    print("=" * 60 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("🧑 用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            conversation_history = []
            print("历史已清空。\n")
            continue
        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        context = ""
        if vdb:
            results = vdb.retrieve(user_input, top_k=3)
            from vectors.store import format_context
            context = format_context(results, max_chars=2000)
            prompt = chat_format_rag(conversation_history, context)
            if context:
                print("  📖 参考规则已加载")
        else:
            context = ""
            prompt = chat_format(conversation_history)

        inputs = tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        response = re.sub(r'<\|im_end\|>|</s>|<\|endoftext\|>', '', response).strip()

        print(f"🤖 助手: {response}\n")

        conversation_history.append({"role": "assistant", "content": response})


def batch_inference(model, tokenizer, input_file: str, output_file: str = None,
                    vdb=None):
    """批量推理"""
    with open(input_file, "r", encoding="utf-8") as f:
        if input_file.endswith(".jsonl"):
            samples = [json.loads(line) for line in f if line.strip()]
        else:
            samples = json.load(f)

    results = []
    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", sample.get("question", ""))
        input_text = sample.get("input", "")
        context = ""
        if vdb:
            vdb_results = vdb.retrieve(instruction, top_k=3)
            from vectors.store import format_context
            context = format_context(vdb_results, max_chars=2000)
        logger.info(f"Inferencing [{i+1}/{len(samples)}]: {instruction[:50]}...")

        response = generate_response(model, tokenizer, instruction, input_text,
                                     context=context)
        results.append({
            **sample,
            "generated_output": response,
        })

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Results saved to {output_file}")
    else:
        for r in results:
            print(f"\n{'='*40}")
            print(f"指令: {r.get('instruction', '')}")
            print(f"生成: {r['generated_output']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="桌游规则问答推理脚本")
    parser.add_argument("--vector", action="store_true",
                        help="启用语义向量检索")
    parser.add_argument("--vector-index", type=str, default="vectors/indexes/",
                        help="向量索引目录")
    parser.add_argument("--no-adapter", action="store_true",
                        help="不使用 LoRA adapter（纯 base model）")
    parser.add_argument("--model_name", type=str, default=None,
                        help="基础模型名称（HuggingFace 模型 ID）")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="合并后的模型目录")
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="LoRA 权重目录")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式")
    parser.add_argument("--instruction", type=str, default=None,
                        help="单次推理指令")
    parser.add_argument("--input_text", type=str, default="",
                        help="指令的额外输入")
    parser.add_argument("--batch_file", type=str, default=None,
                        help="批量推理的输入文件")
    parser.add_argument("--output_file", type=str, default=None,
                        help="批量推理的输出文件")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    args = parser.parse_args()

    # 参数校验
    if not any([args.model_dir, args.model_name, args.lora_dir]):
        if os.path.exists("./outputs/merged_model"):
            args.model_dir = "./outputs/merged_model"
        elif os.path.exists("./outputs"):
            args.lora_dir = "./outputs"
            args.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            parser.error("请指定 --model_name、--model_dir 或 --lora_dir 之一")

    # 加载向量检索引擎
    vdb = None
    if args.vector:
        from vectors.store import MultiGameVectorStore
        vdb = MultiGameVectorStore(args.vector_index)
        vdb.load_all()

    # 单次推理 + 向量检索
    vector_context = None
    if args.vector and args.instruction and vdb:
        results = vdb.retrieve(args.instruction, top_k=3)
        from vectors.store import format_context
        vector_context = format_context(results, max_chars=2000)
    else:
        vector_context = None

    # 加载模型
    model, tokenizer = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        lora_dir=None if args.no_adapter else args.lora_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.cpu,
    )

    if args.interactive:
        interactive_mode(model, tokenizer, vdb=vdb)
    elif args.batch_file:
        batch_inference(model, tokenizer, args.batch_file, args.output_file,
                        vdb=vdb)
    elif args.instruction:
        response = generate_response(
            model, tokenizer,
            instruction=args.instruction,
            input_text=args.input_text,
            context=vector_context or "",
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n🤖 {response}")
    else:
        interactive_mode(model, tokenizer, vdb=vdb)


if __name__ == "__main__":
    main()
