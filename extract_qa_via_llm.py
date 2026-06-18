#!/usr/bin/env python3
"""
LLM 规则书 → 问答对 自动提取工具
=================================
用法:
  1. 设置 API Key:
     export OPENAI_API_KEY="sk-..."          # OpenAI
     export ANTHROPIC_API_KEY="sk-ant-..."   # Anthropic (二选一)

  2. 运行:
     python extract_qa_via_llm.py --input data/docs/agricola-rulebook.txt \\
                                  --output data/llm_generated.jsonl \\
                                  --game "农家乐" \\
                                  --model gpt-4o

  3. 查看统计:
     python extract_qa_via_llm.py --stats data/llm_generated.jsonl

流程:
  规则书 .txt → 按章节智能分块
              → 每块调 LLM 生成 3-5 条问答对
              → 解析/校验 → 输出 Alpaca JSONL
"""

import os, re, json, time, argparse, hashlib, random
from typing import List, Optional
from pathlib import Path

random.seed(42)


# ============================================================
# 文本分块
# ============================================================
def chunk_rulebook(text: str, min_chunk: int = 800, max_chunk: int = 2500) -> List[str]:
    """
    将规则书文本智能分块。
    优先按章节标题切割，每块控制在 max_chunk 字符以内。
    """
    # 尝试识别章节标题模式
    section_patterns = [
        r'^#+\s+.+$',                    # Markdown heading
        r'^[A-Z][A-Za-z\s/]+:$',         # "Overview:" 格式
        r'^\d+\.\s+[A-Z][A-Za-z\s/]+$',  # "1. The Actions" 格式
        r'^[A-Z][A-Za-z\s/]+\n[-=]+\n',  # 带下划线的标题
        r'^第[一二三四五六七八九十]+[章节].*$',  # 中文章节
    ]

    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        is_section = any(re.match(p, line.strip()) for p in section_patterns)

        # 如果是新章节且当前块已够大，切分
        if is_section and current_len >= min_chunk and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line) + 1

            # 如果超过最大长度，找最近的句号切分
            if current_len > max_chunk:
                chunk_text = '\n'.join(current_chunk)
                # 尝试在最后一个句号处切
                last_period = max(
                    chunk_text.rfind('。'), chunk_text.rfind('. '),
                    chunk_text.rfind('\n\n')
                )
                if last_period > min_chunk:
                    chunks.append(chunk_text[:last_period+1])
                    remaining = chunk_text[last_period+1:].strip()
                    current_chunk = [remaining] if remaining else []
                    current_len = len(remaining)
                else:
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_len = 0

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


# ============================================================
# LLM 调用
# ============================================================
def call_llm(prompt: str, model: str = "deepseek-chat", temperature: float = 0.3,
             api_base: Optional[str] = None) -> Optional[str]:
    """调用 LLM API。默认走 Anthropic 协议（兼容 DeepSeek /anthropic 端点）"""
    if model.startswith("gpt-") or model.startswith("o"):
        return _call_openai(prompt, model, temperature)
    else:
        return _call_anthropic(prompt, model, temperature, api_base)


def _call_openai(prompt: str, model: str, temperature: float) -> Optional[str]:
    import httpx
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("需要设置 OPENAI_API_KEY 环境变量")

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, model: str, temperature: float,
                    api_base: Optional[str] = None) -> Optional[str]:
    import httpx
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = api_base or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    if not api_key:
        raise ValueError("需要设置 ANTHROPIC_API_KEY 环境变量")

    resp = httpx.post(
        f"{base_url}/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        json={
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = """你是一个桌游规则专家。你的任务是从规则书片段中提取问答对。

要求:
1. 每段生成 3-5 条问答对
2. 问题要覆盖不同角度（"是什么"、"怎么玩"、"有什么限制"、"常见误解"）
3. 答案必须严格基于原文，不能编造
4. 包含具体数字和例子
5. 用 Alpaca 格式输出 JSON

输出格式 (JSON 数组):
[
  {{"instruction": "问题", "output": "答案"}},
  {{"instruction": "问题", "output": "答案"}}
]

注意: 只输出 JSON，不要有额外的文字说明。"""


def build_prompt(game_name: str, chunk_text: str) -> str:
    return f"""以下是「{game_name}」规则书的一段内容，请生成 3-5 条问答对。

规则内容:
---
{chunk_text}
---

请输出 JSON 数组，格式: [{{"instruction": "问题", "output": "答案"}}]
答案要详细、准确，使用中文。"""


# ============================================================
# 解析 LLM 输出
# ============================================================
def parse_llm_output(text: str) -> List[dict]:
    """从 LLM 回复中提取 JSON 问答对"""
    # 尝试提取 JSON 数组
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            records = json.loads(json_match.group())
            if isinstance(records, list):
                # 校验格式
                valid = []
                for r in records:
                    if isinstance(r, dict) and "instruction" in r and "output" in r:
                        valid.append({
                            "instruction": r["instruction"].strip(),
                            "input": "",
                            "output": r["output"].strip(),
                        })
                return valid
        except json.JSONDecodeError:
            pass

    # 如果 JSON 解析失败，尝试逐行解析
    qa_pairs = []
    lines = text.strip().split('\n')
    current_q = None
    current_a = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[QQ][：:]\s*', line) or line.startswith('"instruction"'):
            if current_q and current_a:
                qa_pairs.append({
                    "instruction": current_q,
                    "input": "",
                    "output": '\n'.join(current_a).strip()
                })
            current_q = re.sub(r'^[QQ][：:]\s*', '', line)
            current_a = []
        elif re.match(r'^[AA][：:]\s*', line):
            current_a.append(re.sub(r'^[AA][：:]\s*', '', line))
        elif current_q:
            current_a.append(line)

    if current_q and current_a:
        qa_pairs.append({
            "instruction": current_q,
            "input": "",
            "output": '\n'.join(current_a).strip()
        })

    return qa_pairs


# ============================================================
# 主逻辑
# ============================================================
def extract_qa(
    input_file: str,
    output_file: str,
    game_name: str = "桌游",
    model: str = "deepseek-chat",
    max_chunks: int = 0,
    rate_limit: float = 1.0,
    api_base: Optional[str] = None,
) -> int:
    """从规则书提取问答对"""
    print(f"\n📖 读取规则书: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"   总长度: {len(text):,} 字符")

    # 分块
    chunks = chunk_rulebook(text)
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    print(f"   分块: {len(chunks)} 块 (显示前3块大小: {[len(c) for c in chunks[:3]]})")

    all_records = []
    errors = 0

    for idx, chunk in enumerate(chunks):
        print(f"\n[{idx+1}/{len(chunks)}] 处理块 ({len(chunk):,} 字符)...")

        prompt = build_prompt(game_name, chunk)
        try:
            response = call_llm(prompt, model=model, api_base=api_base)
            if response:
                records = parse_llm_output(response)
                if records:
                    all_records.extend(records)
                    print(f"   ✅ 生成 {len(records)} 条")
                else:
                    print(f"   ⚠️ 解析结果为空")
                    errors += 1
            else:
                print(f"   ❌ LLM 返回空")
                errors += 1
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            errors += 1

        # 限速
        if idx < len(chunks) - 1:
            time.sleep(rate_limit)

    # 保存
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"✅ 完成!")
    print(f"   输入: {input_file}")
    print(f"   输出: {output_file}")
    print(f"   总块数: {len(chunks)}")
    print(f"   错误数: {errors}")
    print(f"   生成问答: {len(all_records)} 条")
    print(f"{'='*50}")

    return len(all_records)


def show_stats(jsonl_file: str):
    """显示数据集统计"""
    with open(jsonl_file, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    total = len(records)
    avg_q = sum(len(r["instruction"]) for r in records) // total
    avg_a = sum(len(r["output"]) for r in records) // total
    max_len = max(len(r["instruction"]) + len(r["output"]) for r in records)

    print(f"\n📊 数据集统计: {jsonl_file}")
    print(f"   总样本数: {total}")
    print(f"   平均问题长度: {avg_q} 字")
    print(f"   平均答案长度: {avg_a} 字")
    print(f"   最大总长度: {max_len} 字")
    print(f"   建议 max_seq_length: {(max_len // 512 + 1) * 512}")


def dedup(jsonl_file: str):
    """去重"""
    with open(jsonl_file, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    seen = set()
    deduped = []
    for r in records:
        key = r["instruction"][:30]  # 按问题前30字去重
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"   去重: {len(records)} → {len(deduped)} 条")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 规则书 → 问答对 提取工具")
    sub = parser.add_subparsers(dest="cmd")

    # extract 命令
    p_extract = sub.add_parser("extract", help="从规则书提取问答对")
    p_extract.add_argument("--input", "-i", required=True, help="规则书 .txt 文件路径")
    p_extract.add_argument("--output", "-o", default="./data/llm_generated.jsonl", help="输出 JSONL 路径")
    p_extract.add_argument("--game", default="桌游", help="游戏名称（用于 prompt）")
    p_extract.add_argument("--model", default="deepseek-chat", help="LLM 模型 (deepseek-chat / gpt-4o / claude-sonnet-4-6)")
    p_extract.add_argument("--api-base", default=None, help="API 地址 (默认取 ANTHROPIC_BASE_URL 环境变量，否则 https://api.anthropic.com)")
    p_extract.add_argument("--max-chunks", type=int, default=0, help="最多处理块数（0=全部）")
    p_extract.add_argument("--rate-limit", type=float, default=1.0, help="API 调用间隔（秒）")

    # stats 命令
    p_stats = sub.add_parser("stats", help="查看数据集统计")
    p_stats.add_argument("input", help="JSONL 文件路径")

    # dedup 命令
    p_dedup = sub.add_parser("dedup", help="去重")
    p_dedup.add_argument("input", help="JSONL 文件路径")

    args = parser.parse_args()

    if args.cmd == "extract":
        api_base = args.api_base or os.environ.get("ANTHROPIC_BASE_URL")
        extract_qa(args.input, args.output, args.game, args.model,
                   args.max_chunks, args.rate_limit, api_base)
    elif args.cmd == "stats":
        show_stats(args.input)
    elif args.cmd == "dedup":
        dedup(args.input)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python extract_qa_via_llm.py extract -i data/docs/agricola-rulebook.txt -o data/llm_agricola.jsonl --game 农家乐 --model gpt-4o")
        print("  python extract_qa_via_llm.py stats data/llm_agricola.jsonl")
        print("  python extract_qa_via_llm.py dedup data/llm_agricola.jsonl")
