#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac 推理脚本 — 支持语义向量检索

用法:
    python infer_mac.py --vector --interactive
    python infer_mac.py --vector --prompt "农家乐怎么扩建房屋？"
    python infer_mac.py --no-adapter --vector --interactive
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__) or ".")

# ---------- 游戏名检测 ----------
GAME_KEYWORDS = {
    "agricola":     ["agricola", "农家乐", "农场"],
    "lotr journeys": ["lotr", "中洲", "魔戒", "journeys in middle earth",
                      "中洲征途", "journeys in middle"],
    "stone age":    ["stone age", "石器时代"],
}


def _detect_game(query: str) -> str | None:
    """从 query 中猜测用户问的是哪个桌游"""
    q = query.lower().strip()
    for game, keywords in GAME_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return game
    return None


def main():
    parser = argparse.ArgumentParser(description="Mac 推理")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="./outputs_mac_rag/adapter")
    parser.add_argument("--no-adapter", action="store_true", help="不使用 adapter")
    parser.add_argument("--prompt", help="单次推理")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--vector", action="store_true", help="启用语义向量检索")
    parser.add_argument("--vector-index", default="vectors/indexes/",
                        help="向量索引目录")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # ---------- 加载向量检索引擎 ----------
    vdb = None
    if args.vector:
        try:
            from vectors.store import MultiGameVectorStore, format_context
            vdb = MultiGameVectorStore(args.vector_index)
            vdb.load_all()
            games = ", ".join(vdb.get_game_names())
            print(f"  [向量检索] 就绪: {games}")
        except Exception as e:
            print(f"  ⚠️  向量检索加载失败: {e}")
            vdb = None

    # ---------- 加载模型 ----------
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"加载模型: {args.model}")
    try:
        model, tokenizer = load(args.model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    if not args.no_adapter:
        if os.path.isdir(args.adapter):
            try:
                from mlx_lm.utils import load_adapters
                model = load_adapters(model, args.adapter)
                print(f"已加载 adapter: {args.adapter}")
            except Exception as e:
                print(f"adapter 加载失败: {e}")
                print("  继续使用 base 模型推理")

    sampler = make_sampler(temp=args.temperature)

    # ---------- 推理函数 ----------
    def ask(prompt: str):
        nonlocal vdb
        messages = []
        if vdb:
            # 自动识别游戏名，避免跨游戏污染
            game = _detect_game(prompt)
            game_names = [game] if game else None
            tag = f" [{game}]" if game else ""
            results = vdb.retrieve(prompt, top_k=3, game_names=game_names)
            context = format_context(results, max_chars=2000)
            if context:
                messages.append({
                    "role": "system",
                    "content": f"参考规则:\n{context}"
                })
                print(f"[RAG] [向量检索]{tag} 📖 参考规则已加载")
        messages.append({"role": "user", "content": prompt})

        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )
        response = generate(
            model, tokenizer,
            prompt=input_text,
            max_tokens=args.max_tokens,
            sampler=sampler,
        )
        print(f"{response}\n")

    if args.prompt:
        ask(args.prompt)

    if args.interactive:
        mode = "向量检索" if vdb else "标准"
        print(f"\n交互模式 ({mode}) 输入 exit 退出\n")
        while True:
            try:
                p = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if p.lower() in ("exit", "quit", "q"):
                break
            if p.strip():
                ask(p)


if __name__ == "__main__":
    main()
