#!/usr/bin/env bash
# ============================================================
# 04_reranker_test.sh — 对比测试 bge-m3 vs reranker 重排序
# 用法:
#   bash 04_reranker_test.sh                                          # 默认查询
#   bash 04_reranker_test.sh -q "免赔额多少" -q "保险范围"            # 自定义查询
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"
python reranker/test.py "$@"
