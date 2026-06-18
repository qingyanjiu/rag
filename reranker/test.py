#!/usr/bin/env python3
"""
Reranker 模型测试脚本 v3 — 通用版
===============================
测试 (bge-m3 检索 → cross-encoder 重排) 流程。

用法:
    python reranker/test.py                                             # 默认查询
    python reranker/test.py -q "保险免赔额是多少"                        # 自定义单查询
    python reranker/test.py -q "问题1" -q "问题2"                      # 自定义多查询
    bash 04_reranker_test.sh                                           # 默认
    bash 04_reranker_test.sh -q "免赔额多少" -q "保险范围"              # 自定义
"""
import json, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from vectors.store import MultiGameVectorStore

# ── 缓存 embedding 模型 ──
_EMBED_MODEL = None

def get_embed_model(model_name="BAAI/bge-m3"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"加载嵌入模型: {model_name}")
        t0 = time.time()
        _EMBED_MODEL = SentenceTransformer(model_name, local_files_only=True)
        print(f"  完成 ({time.time()-t0:.1f}s)")
    return _EMBED_MODEL

# ── 缓存 cross-encoder ──
_CE_MODEL = None
_CE_TOKENIZER = None

def get_ce():
    global _CE_MODEL, _CE_TOKENIZER
    if _CE_MODEL is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ce_path = os.path.join(os.path.dirname(__file__), "checkpoints")
        print(f"加载 cross-encoder: {ce_path}")
        t0 = time.time()
        _CE_TOKENIZER = AutoTokenizer.from_pretrained(ce_path, local_files_only=True)
        _CE_MODEL = AutoModelForSequenceClassification.from_pretrained(
            ce_path, local_files_only=True)
        _CE_MODEL.eval()
        print(f"  完成 ({time.time()-t0:.1f}s, {_CE_MODEL.num_parameters():,} 参数)")
    return _CE_MODEL, _CE_TOKENIZER

def rerank(model, tokenizer, query, candidates, batch_size=8):
    import torch
    pairs = [(query, c["chunk"]) for c in candidates]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = tokenizer(*zip(*batch), padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        batch_scores = logits.squeeze(-1).tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: -x[0])
    return scored

def test_one(vdb, query, top_k_bge=10, top_k_final=5):
    ce_model, ce_tok = get_ce()
    print(f"\n{'─'*60}")
    print(f"查询: {query}")
    print(f"{'─'*60}")

    # Step 1: bge-m3 检索
    t0 = time.time()
    model = get_embed_model()
    q_vec = model.encode([query], normalize_embeddings=True)

    all_results = []
    for gname, gstore in vdb.games.items():
        if gstore.embeddings is None:
            continue
        scores = np.dot(gstore.embeddings, q_vec.T).flatten()
        top_idx = np.argsort(scores)[-top_k_bge:][::-1]
        for i in top_idx:
            all_results.append({
                "chunk": gstore.chunks[i],
                "score": round(float(scores[i]), 4),
                "game": gstore.game,
                **gstore.metas[i],
            })
    all_results.sort(key=lambda x: -x["score"])
    all_results = all_results[:top_k_bge]
    t_bge = time.time() - t0
    print(f"▸ bge-m3 检索 ({len(all_results)} 条, {t_bge:.2f}s):")
    for i, r in enumerate(all_results):
        pre = r["chunk"][:100].replace("\n", " ")
        print(f"  [{i+1}] bge={r['score']:.4f} | {pre}...")

    # Step 2: cross-encoder 重排序
    t0 = time.time()
    reranked = rerank(ce_model, ce_tok, query, all_results)
    t_ce = time.time() - t0
    print(f"\n▸ cross-encoder 重排结果 ({t_ce:.2f}s, top-{top_k_final}):")
    for i, (ce_score, r) in enumerate(reranked[:top_k_final]):
        pre = r["chunk"][:100].replace("\n", " ")
        bge_rank = next((j+1 for j, br in enumerate(all_results)
                         if br["chunk"] == r["chunk"]), None)
        arrow = "⬆" if (bge_rank and i+1 < bge_rank) else ("⬇" if (bge_rank and i+1 > bge_rank) else "→")
        print(f"  [{i+1}] ce={ce_score:.4f} (bge={r['score']:.4f}) {arrow} bge#{bge_rank or '?'}")
        print(f"      {pre}")

    # Step 3: 分数对比
    reranked_dict = {r["chunk"]: s for s, r in reranked}
    print(f"\n▸ bge-m3 vs cross-encoder 分数对比 (全部 {len(all_results)} 条):")
    for r in all_results:
        pre = r["chunk"][:50].replace("\n", " ")
        ce_s = reranked_dict.get(r["chunk"], -999)
        ce_rank = next((j+1 for j, (s, cr) in enumerate(reranked) if cr["chunk"] == r["chunk"]), None)
        print(f"  bge={r['score']:.3f}  ce={ce_s:.3f}  ce_rank={ce_rank} | {pre}")

def main():
    parser = argparse.ArgumentParser(description="Reranker 对比测试")
    parser.add_argument("--query", "-q", action="append", dest="queries",
                       help="自定义查询（可重复使用指定多个）")
    args = parser.parse_args()

    # 1. 加载索引
    print("="*50)
    print("加载向量索引")
    print("="*50)
    vdb = MultiGameVectorStore("vectors/indexes")
    vdb.load_all()
    print(f"已加载文档: {vdb.get_game_names()}")

    # 2. 确定查询列表
    if args.queries:
        queries = args.queries
        print(f"\n使用自定义查询 ({len(queries)} 个):")
        for i, q in enumerate(queries):
            print(f"  [{i+1}] {q}")
    else:
        doc_names = vdb.get_game_names()
        queries = []
        print(f"\n未指定查询，使用文档名生成默认查询 ({len(doc_names)} 个)")
        for doc in doc_names:
            query = f"查询文档 {doc} 的相关内容"
            queries.append(query)
            print(f"  [{queries.index(query)+1}] {query}")

    # 3. 加载 embedding 模型
    get_embed_model()

    # 4. 执行测试
    for q in queries:
        test_one(vdb, q)

if __name__ == "__main__":
    main()
