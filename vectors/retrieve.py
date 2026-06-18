#!/usr/bin/env python3
"""
向量检索 CLI — 测试中文 query 对规则原文的语义检索效果

用法:
    # 交互模式
    python vectors/retrieve.py --interactive

    # 单次查询
    python vectors/retrieve.py --query "农家乐每回合能做什么"
"""

import argparse
import sys

sys.path.insert(0, ".")
from vectors.store import MultiGameVectorStore


def main():
    parser = argparse.ArgumentParser(description="桌游规则向量检索测试")
    parser.add_argument("--index", default="vectors/indexes/",
                        help="向量索引目录")
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="嵌入模型名称")
    parser.add_argument("--query", help="单次查询语句")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式")
    parser.add_argument("--top-k", type=int, default=3,
                        help="返回 top-K 结果")
    parser.add_argument("--game", nargs="*",
                        help="限定游戏名（多值用空格分隔）")
    args = parser.parse_args()

    # ---- 加载索引 ----
    print("加载向量索引...")
    vdb = MultiGameVectorStore(args.index, model_name=args.model)
    vdb.load_all()
    games = ", ".join(vdb.get_game_names())
    print(f"  就绪: {games}")

    # ---- 检索函数 ----
    def retrieve(query: str):
        print(f"\n[查询] {query}")
        game_names = args.game if args.game else None
        results = vdb.retrieve(query, top_k=args.top_k,
                               game_names=game_names)
        if not results:
            print("  (无结果)")
            return
        for i, r in enumerate(results):
            game = r.get("game", "?")
            section = r.get("section", "")
            score = r["score"]
            preview = r["chunk"][:150].replace("\n", " ")
            print(f"\n  [{i+1}] {game} | {section} | score={score:.4f}")
            print(f"      {preview}...")

    # ---- 执行 ----
    if args.query:
        retrieve(args.query)

    if args.interactive:
        print("\n交互模式 (输入 exit 退出)")
        while True:
            try:
                q = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if q.strip():
                retrieve(q)


if __name__ == "__main__":
    main()
