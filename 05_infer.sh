#!/usr/bin/env bash
# ============================================================
# 05_infer.sh — RAG 推理：检索 + LLM 润色答案
# ============================================================
# 用法:
#   export OPENAI_API_KEY="sk-..."
#   bash 05_infer.sh "免赔额多少"
#   bash 05_infer.sh -q "怎么报销" --model deepseek-chat
#   bash 05_infer.sh "保险范围" --top-k 8 --no-sources
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -ge 1 ]; then
    python infer/run.py "$@"
else
    echo "用法: bash 05_infer.sh \"你的问题\""
    echo "      bash 05_infer.sh -q \"问题\" --model gpt-4o-mini"
    exit 1
fi
