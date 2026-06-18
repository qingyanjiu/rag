#!/usr/bin/env python3
"""
构建向量索引 — 从规则文件生成每个桌游的语义向量库

用法:
    python vectors/build.py
    python vectors/build.py --rules data/docs/ --index vectors/indexes/
"""

import argparse
import sys

sys.path.insert(0, ".")
from vectors.store import MultiGameVectorStore


def main():
    parser = argparse.ArgumentParser(description="构建桌游规则向量索引")
    parser.add_argument("--rules", default="data/docs/",
                        help="规则文件目录 (.txt)")
    parser.add_argument("--index", default="vectors/indexes/",
                        help="向量索引输出目录")
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="嵌入模型名称")
    args = parser.parse_args()

    print("=" * 50)
    print("构建桌游规则向量索引")
    print("=" * 50)
    print(f"  规则目录: {args.rules}")
    print(f"  索引目录: {args.index}")
    print(f"  嵌入模型: {args.model}")
    print()

    vdb = MultiGameVectorStore(args.index, model_name=args.model)
    vdb.build_from_rules(args.rules)

    print()
    print(f"  覆盖游戏: {', '.join(vdb.get_game_names())}")
    print(f"  总块数: {sum(len(s.chunks) for s in vdb.games.values())}")

    vdb.save_all()

    print()
    print("=" * 50)
    print("完成！")
    print(f"  索引已保存到: {args.index}")
    print(f"  使用:  python vectors/retrieve.py --index {args.index}")
    print("=" * 50)


if __name__ == "__main__":
    main()
