#!/usr/bin/env python3
"""
RAG 推理 — 检索 + LLM 润色答案
================================
流程: query → bge-m3 检索 → LLM 生成最终答案

环境变量:
    OPENAI_API_KEY      OpenAI API 密钥（必填）
    OPENAI_BASE_URL     API 地址（默认 https://api.openai.com/v1）
    RAG_MODEL           模型名（默认 gpt-4o-mini）

用法:
    export OPENAI_API_KEY="sk-..."
    python infer/run.py "免赔额多少"
    python infer/run.py -q "怎么报销" --model deepseek-chat --top-k 8
"""
import argparse, json, os, sys, textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectors.store import MultiGameVectorStore

# ── API 配置 ──
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")


def call_llm(messages, model, temperature=0.3, max_tokens=1024):
    """调用 OpenAI 兼容 API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except ImportError:
        pass

    # fallback: 直接用 requests
    import requests as req
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = req.post(f"{BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers, json=body, timeout=120)
    try:
        resp.raise_for_status()
    except Exception:
        print(f"  [错误] API 请求失败: {resp.status_code}")
        print(f"  [错误] 请求地址: {resp.request.url}")
        print(f"  [错误] 模型: {model}")
        print(f"  [错误] 响应: {resp.text[:500]}")
        print(f"  [提示] 检查模型名是否正确，或查看 API 文档确认调用格式")
        if "OPENAI_BASE_URL" in os.environ:
            print(f"  [提示] 当前 OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', '')}")
        if "RAG_MODEL" in os.environ:
            print(f"  [提示] 当前 RAG_MODEL={os.environ.get('RAG_MODEL', '')}")
        print(f"  [提示] 示例: DeepSeek 用 deepseek-chat, 通义千问用 qwen-plus")
        sys.exit(1)
    return resp.json()["choices"][0]["message"]["content"]


def build_prompt(query, chunks):
    """构建 RAG prompt，包含检索到的上下文"""
    context_parts = []
    for i, c in enumerate(chunks):
        source = f"[{i+1}]"
        meta = c.get("section", "") or ""
        if meta:
            source += f" ({meta})"
        context_parts.append(f"{source} {c['chunk']}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是一个文档问答助手。请根据以下文档内容回答问题。

要求：
- 回答要简洁准确，基于文档内容
- 如果文档内容不足以回答，请如实说明
- 必要时引用文档中的具体条款或数据
- 用中文回答

===== 文档内容 =====

{context_text}

===== 问题 =====

{query}

===== 回答 ====="""
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="RAG 推理 — 检索 + LLM 润色答案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例：
              python infer/run.py "免赔额多少"
              python infer/run.py -q "保险范围是什么" --model deepseek-chat --top-k 8
              python infer/run.py -q "怎么报销" --no-sources
        """))
    parser.add_argument("query", nargs="?", help="查询问题")
    parser.add_argument("-q", "--query", dest="query2", help="查询问题（备用）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索段落数（默认 5）")
    parser.add_argument("--model", default=None,
                        help=f"LLM 模型（默认 {MODEL}）")
    parser.add_argument("--no-sources", action="store_true",
                        help="不展示检索来源")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM 温度（默认 0.3）")
    parser.add_argument("--index", default="vectors/indexes",
                        help="向量索引目录")
    args = parser.parse_args()

    # 解析 query
    query = args.query or args.query2
    if not query:
        parser.print_help()
        print("\n错误：请提供查询问题")
        sys.exit(1)

    model = args.model or MODEL

    # 检查 API key
    if not API_KEY:
        print("错误：未设置 OPENAI_API_KEY 环境变量")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # ── 1. 加载索引 ──
    print("加载向量索引...", end=" ", flush=True)
    vdb = MultiGameVectorStore(args.index)
    vdb.load_all()
    print(f"就绪 ({', '.join(vdb.get_game_names())})")

    # ── 2. 检索 ──
    print(f"检索 top-{args.top_k}...", end=" ", flush=True)
    results = vdb.retrieve(query, top_k=args.top_k)
    if not results:
        print("\n未找到相关文档内容")
        sys.exit(1)
    print(f"找到 {len(results)} 条")

    api_url = f"{BASE_URL.rstrip(chr(47))}/chat/completions"
    print(f"调用 LLM → {api_url}")
    print(f"  模型: {model}")
    print(f"  参数: temperature={args.temperature}, top_k={args.top_k}")

    # flush before LLM call so the user sees where we are, flush=True)
    prompt = build_prompt(query, results)
    messages = [
        {"role": "system", "content": "你是一个专业的文档问答助手。"},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages, model, temperature=args.temperature)

    # ── 4. 输出 ──
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"{'='*60}")
    print(f"\n{answer}\n")

    if not args.no_sources:
        print(f"{'─'*60}")
        print("检索来源:")
        for i, r in enumerate(results):
            src = r.get("section", "") or ""
            preview = r["chunk"][:120].replace("\n", " ").strip()
            print(f"  [{i+1}] score={r['score']:.4f}  {preview}...")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()
