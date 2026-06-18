#!/usr/bin/env python3
"""
Reranker 推理模块 — Cross-Encoder 重排
========================================

用法:
    from reranker.inference import Reranker

    reranker = Reranker("reranker/checkpoints")
    scores = reranker.score("query", ["chunk1", "chunk2", ...])
    # 或直接重排检索结果
    results = reranker.rerank("query", raw_results, top_k=3)
"""

import logging, os
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 重排器"""

    def __init__(self, model_path: str = "reranker/checkpoints", device: str = "cpu"):
        self.model_path = model_path
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        if os.path.exists(self.model_path):
            logger.info(f"加载本地 reranker 模型: {self.model_path}")
            self._model = CrossEncoder(self.model_path)
        else:
            logger.info(f"使用默认模型: BAAI/bge-reranker-v2-m3")
            self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1,
                                       local_files_only=True)
        logger.info("Reranker 加载完成")

    def score(self, query: str, chunks: List[str]) -> List[float]:
        """
        对 (query, chunk) 对打分
        Returns: [score_0, score_1, ...] 越高越相关
        """
        self._load()
        pairs = [(query, chunk) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    def rerank(self, query: str, results: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对检索结果重排
        Args:
            query: 用户查询
            results: 原始检索结果列表（每项含 "chunk" 字段）
            top_k: 返回 top-k
        Returns:
            重排后的结果列表（按得分降序，含新增 "rerank_score" 字段）
        """
        if not results:
            return []

        chunks = [r.get("chunk", "") for r in results]
        scores = self.score(query, chunks)

        # 附加 rerank score
        for r, s in zip(results, scores):
            r["rerank_score"] = round(float(s), 4)

        # 按 rerank score 降序
        results.sort(key=lambda x: -x.get("rerank_score", 0))

        return results[:top_k]


def format_context(results: List[dict], max_chars: int = 2000) -> str:
    """格式化重排结果为单段文本（跟 store.py 的 format_context 一致）"""
    parts = []
    total = 0
    for r in results:
        block = r.get("chunk", "")
        if not block:
            continue
        total += len(block)
        if total > max_chars and parts:
            break
        parts.append(block)
    return "\n\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
