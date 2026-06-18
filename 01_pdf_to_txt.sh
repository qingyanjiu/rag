#!/usr/bin/env bash
# ============================================================
# 01_pdf_to_txt.sh — 将 data/docs/ 下的 PDF 批量转为 TXT
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/data/docs"

PYTHON=""
for p in /Users/louisliu/miniforge3/bin/python3 /usr/bin/python3 python3; do
    if command -v "$p" &>/dev/null && $p -c "import pypdf" 2>/dev/null; then
        PYTHON=$p
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "需要安装 pypdf: pip install pypdf"
    exit 1
fi

converted=0
skipped=0
failed=0

for pdf in *.pdf; do
    [ -f "$pdf" ] || continue
    txt="${pdf%.pdf}.txt"
    if [ -f "$txt" ]; then
        echo "  ⏭ 跳过（已存在）: $txt"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  → 转换: $pdf"
    if $PYTHON -c "
from pypdf import PdfReader
import sys
try:
    reader = PdfReader('$pdf')
    with open('$txt', 'w') as f:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                f.write(text + '\n')
    print('ok')
except Exception as e:
    print(f'error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        n_pages=$($PYTHON -c "from pypdf import PdfReader; print(len(PdfReader('$pdf').pages))")
        echo "     ✅ $n_pages 页 → $txt"
        converted=$((converted + 1))
    else
        echo "     ❌ 转换失败"
        failed=$((failed + 1))
    fi
done

echo ""
echo "完成: $converted 个转换, $skipped 个跳过, $failed 个失败"
