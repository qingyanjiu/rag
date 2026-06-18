#!/usr/bin/env bash
# ============================================================
# 桌游规则书 → 训练数据集 一键构建脚本
# ============================================================
# 用法:
#   bash data/build_dataset.sh                    # 全流程
#   bash data/build_dataset.sh --skip-pdf         # 跳过 PDF→TXT（已有txt时）
#   bash data/build_dataset.sh --skip-llm         # 跳过 LLM 提取（已有jsonl时）
#   bash data/build_dataset.sh --skip-merge       # 跳过合并（只提取QA）
#
# 流程:
#   1. PDF → TXT  (data/docs/ 中的新 PDF)
#   2. TXT → QA   (调用 Claude API 提取问答对)
#   3. 合并数据   (合并所有 llm_*.jsonl 到 train.jsonl / eval.jsonl)
#   4. 输出统计
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RULES_DIR="$SCRIPT_DIR/docs"

# 解析参数
SKIP_PDF=false
SKIP_LLM=false
SKIP_MERGE=false
for arg in "$@"; do
    case "$arg" in
        --skip-pdf) SKIP_PDF=true ;;
        --skip-llm) SKIP_LLM=true ;;
        --skip-merge) SKIP_MERGE=true ;;
    esac
done

echo ""
echo "============================================"
echo "  训练数据集 构建工具"
echo "============================================"
echo "  规则书目录: $RULES_DIR"
echo "  跳过 PDF → TXT : $SKIP_PDF"
echo "  跳过 LLM 提取 : $SKIP_LLM"
echo "  跳过数据合并 : $SKIP_MERGE"
echo "============================================"
echo ""

# ============================================================
# Step 1: PDF → TXT
# ============================================================
if [ "$SKIP_PDF" = false ]; then
    echo "【Step 1/3】PDF → TXT 文本提取"
    echo "----------------------------------------"

    # 检查 pypdf 是否可用
    python3 -c "import pypdf" 2>/dev/null || {
        echo "📦 安装 pypdf (PDF 文本提取)..."
        uv pip install pypdf --quiet 2>/dev/null || pip install pypdf -q
    }

    # 查找所有 PDF
    shopt -s nullglob
    pdfs=("$RULES_DIR"/*.pdf)
    shopt -u nullglob

    if [ ${#pdfs[@]} -eq 0 ]; then
        echo "  没有找到 PDF 文件，跳过。"
    else
        for pdf in "${pdfs[@]}"; do
            txt="${pdf%.pdf}.txt"
            if [ -f "$txt" ]; then
                echo "  ✅ 已有 TXT: $(basename "$txt") (跳过)"
            else
                echo "  📄 提取: $(basename "$pdf") → $(basename "$txt")"
                python3 -c "
import sys
from pypdf import PdfReader
reader = PdfReader('$pdf')
text = ''
for page in reader.pages:
    t = page.extract_text()
    if t:
        text += t + '\n'
with open('$txt', 'w', encoding='utf-8') as f:
    f.write(text.strip())
print(f'    共 {len(reader.pages)} 页, {len(text):,} 字符')
" 2>&1 || echo "  ⚠️ 提取失败: $pdf (可能为扫描件)"
            fi
        done
    fi
    echo ""
else
    echo "【Step 1/3】跳过 PDF → TXT"
    echo ""
fi

# ============================================================
# Step 2: TXT → QA pairs (调用 Claude API)
# ============================================================
if [ "$SKIP_LLM" = false ]; then
    echo "【Step 2/3】TXT → 问答对 (调用 LLM)"
    echo "----------------------------------------"

    # 检查 API Key
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "⚠️  未设置 ANTHROPIC_API_KEY 环境变量"
        echo "   请设置您的 API Key:"
        echo "   export ANTHROPIC_API_KEY=\"sk-ant-...\""
        echo ""
        read -rp "   输入 API Key (或直接回车跳过): " key
        if [ -n "$key" ]; then
            export ANTHROPIC_API_KEY="$key"
        else
            echo "   跳过 LLM 提取。"
            SKIP_LLM=true
        fi
    fi

    if [ "$SKIP_LLM" = false ]; then
        # 检查 ANTHROPIC_BASE_URL
        API_BASE="${ANTHROPIC_BASE_URL:-}"
        API_BASE_ARG=""
        if [ -n "$API_BASE" ]; then
            API_BASE_ARG="--api-base $API_BASE"
        fi

        # 默认模型
        MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}"

        # 游戏名称映射 (文件名前缀 → 中文名)
        declare -A GAME_NAMES=(
            [agricola]="农家乐"
            [lotr]="魔戒·中洲之旅"
            [stone-age]="石器时代"
            [catan]="卡坦岛"
            [carcassonne]="卡卡颂"
            [ticket-to-ride]="车票之旅"
            [terraforming-mars]="殖民火星"
            [azul]="阿祖尔"
            [wingspan]="展翅翱翔"
            [splendor]="璀璨宝石"
            [patchwork]="拼布艺术"
            [everdell]="仙境幽谷"
            [dune]="沙丘"
            [ark-nova]="方舟动物园"
            [brass]="铜板"
            [root]="森林根国"
            [gloomhaven]="幽暗港"
            [pandemic]="瘟疫危机"
            [7-wonders]="七大奇迹"
            [blood-rage]="血怒"
        )

        # 查找所有 TXT 规则书
        shopt -s nullglob
        txts=("$RULES_DIR"/*.txt)
        shopt -u nullglob

        processed=0
        for txt in "${txts[@]}"; do
            basename_txt=$(basename "$txt")
            prefix="${basename_txt%-rulebook*}"
            prefix="${prefix%-rules*}"
            prefix="${prefix%.txt}"

            # 确定输出路径
            output="$SCRIPT_DIR/llm_${prefix}.jsonl"

            # 如果输出已存在，询问是否覆盖
            if [ -f "$output" ]; then
                count=$(wc -l < "$output")
                echo "  📋 已有: $basename_txt → $(basename "$output") (${count}条)"
                echo "     (重新提取会覆盖，跳过则保留已有)"
                echo "     [y=重新提取 / n=跳过]"
                read -rp "     是否处理? " -n1 ans
                echo ""
                if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
                    echo "  ⏭️  跳过 $(basename "$txt")"
                    continue
                fi
            fi

            # 获取游戏中文名
            game_name="${GAME_NAMES[$prefix]:-$prefix}"

            echo "  🎲 提取: $(basename "$txt") → $(basename "$output") (${game_name})"
            python3 "$PROJ_DIR/extract_qa_via_llm.py" extract \
                --input "$txt" \
                --output "$output" \
                --game "$game_name" \
                --model "$MODEL" \
                $API_BASE_ARG \
                --rate-limit 1.0

            processed=$((processed + 1))
        done

        if [ $processed -eq 0 ]; then
            echo "  ℹ️  没有新的规则书需要处理。"
        fi
    fi
    echo ""
else
    echo "【Step 2/3】跳过 LLM 提取"
    echo ""
fi

# ============================================================
# Step 3: 合并数据 → train.jsonl / eval.jsonl
# ============================================================
if [ "$SKIP_MERGE" = false ]; then
    echo "【Step 3/3】合并所有数据"
    echo "----------------------------------------"

    # 收集所有 llm_*.jsonl
    shopt -s nullglob
    llm_files=("$SCRIPT_DIR"/llm_*.jsonl)
    shopt -u nullglob

    if [ ${#llm_files[@]} -eq 0 ]; then
        echo "  没有 llm_*.jsonl 数据，跳过合并。"
    else
        echo "  参与合并的文件:"
        for f in "${llm_files[@]}"; do
            count=$(wc -l < "$f")
            echo "    📄 $(basename "$f") — ${count}条"
        done

        # 合并所有 LLM 生成数据到一个临时文件
        TMP_ALL=$(mktemp)
        for f in "${llm_files[@]}"; do
            cat "$f" >> "$TMP_ALL"
        done
        TOTAL=$(wc -l < "$TMP_ALL")

        # 去重（按问题内容）
        TMP_DEDUP=$(mktemp)
        python3 -c "
import json
seen = set()
with open('$TMP_ALL', 'r') as f_in, open('$TMP_DEDUP', 'w') as f_out:
    for line in f_in:
        r = json.loads(line)
        key = r['instruction'][:40]
        if key not in seen:
            seen.add(key)
            f_out.write(line)
" 2>&1
        UNIQUE=$(wc -l < "$TMP_DEDUP")
        echo "  🔄 去重: $TOTAL → $UNIQUE 条"

        # 按 9:1 分割 train/eval
        TRAIN="$SCRIPT_DIR/train.jsonl"
        EVAL="$SCRIPT_DIR/eval.jsonl"

        python3 -c "
import json, random
random.seed(42)

with open('$TMP_DEDUP', 'r') as f:
    records = [json.loads(l) for l in f if l.strip()]

random.shuffle(records)
split = int(len(records) * 0.9)
train, eval_data = records[:split], records[split:]

# 保留原有的 train/eval 数据（非 LLM 生成的）
# 追加新生成的 LLM 数据
import os

def append_jsonl(path, data):
    existing = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                r = json.loads(line)
                existing.add(r['instruction'][:40])
    with open(path, 'a', encoding='utf-8') as f:
        for r in data:
            if r['instruction'][:40] not in existing:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

append_jsonl('$TRAIN', train)
append_jsonl('$EVAL', eval_data)

new_train = sum(1 for r in train if r['instruction'][:40] not in existing_before if we could track...)
print(f'  ✅ 已追加到 train.jsonl / eval.jsonl')
" 2>&1 || true

        # 更简单的实现: 直接统计合并结果
        python3 -c "
import json, random, os
random.seed(42)

records = []
with open('$TMP_DEDUP', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

random.shuffle(records)
split = int(len(records) * 0.9)
train_new, eval_new = records[:split], records[split:]

def merge(path, new_data):
    seen = set()
    old_count = 0
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seen.add(r['instruction'][:40])
                    old_count += 1
    added = 0
    with open(path, 'a', encoding='utf-8') as f:
        for r in new_data:
            key = r['instruction'][:40]
            if key not in seen:
                seen.add(key)
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                added += 1
    return old_count, added, len(seen)

train_old, train_added, train_total = merge('$TRAIN', train_new)
eval_old, eval_added, eval_total = merge('$EVAL', eval_new)

print(f'  📊 结果:')
print(f'    train.jsonl: {train_old} +{train_added} = {train_total} 条')
print(f'    eval.jsonl : {eval_old} +{eval_added} = {eval_total} 条')
" 2>&1

        rm -f "$TMP_ALL" "$TMP_DEDUP"
    fi
    echo ""
else
    echo "【Step 3/3】跳过数据合并"
    echo ""
fi

# ============================================================
# 完成
# ============================================================
echo "============================================"
echo "  ✅ 数据构建完成！"
echo "============================================"
echo ""
echo "数据集统计:"
python3 "$PROJ_DIR/extract_qa_via_llm.py" stats "$SCRIPT_DIR/train.jsonl" 2>/dev/null || echo "  (请先安装依赖)"
echo ""
echo "下一步:"
echo "  python train.py --config config.yaml"
echo ""
