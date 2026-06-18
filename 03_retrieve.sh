#!/usr/bin/env bash
# ============================================================
# 03_retrieve.sh — 检索测试
# ============================================================
# 用法:
#   bash 03_retrieve.sh              交互模式
#   bash 03_retrieve.sh "你的问题"   单次查询
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -ge 1 ]; then
    python vectors/retrieve.py --query "$*"
else
    python vectors/retrieve.py --interactive
fi
