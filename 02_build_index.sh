#!/usr/bin/env bash
# ============================================================
# 02_build_index.sh — 构建向量索引（bge-m3 多语言嵌入）
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=""
for p in /Users/louisliu/miniforge3/bin/python3 /usr/bin/python3; do
    if $p -c "import sentence_transformers" 2>/dev/null; then
        PYTHON=$p
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "请安装 sentence-transformers: pip install sentence-transformers"
    exit 1
fi

$PYTHON vectors/build.py --model BAAI/bge-m3
