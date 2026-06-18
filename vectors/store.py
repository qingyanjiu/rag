#!/usr/bin/env python3
"""
GameVectorStore — 按桌游组织的向量检索模块
============================================

每个桌游独立建索引，支持跨游戏检索。
使用 sentence-transformers 生成语义嵌入，numpy 做余弦相似度检索。

用法:
    from vectors.store import MultiGameVectorStore

    vdb = MultiGameVectorStore("vectors/indexes")
    vdb.build_from_rules("data/docs/")
    vdb.save_all()

    results = vdb.retrieve("农家乐每回合能做什么", top_k=3)
    for r in results:
        print(r["game"], r["score"], r["chunk"][:80])
"""

import json
import os
import re
from typing import List, Optional
from glob import glob
from pathlib import Path

import numpy as np
import torch

# ============================================================
# Chunking utilities — moved from rag_index.py
# ============================================================

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """按段落分块，尝试在句子/段落边界处分割"""
    lines = text.split("\n")
    chunks, current, current_len = [], [], 0
    for line in lines:
        is_heading = bool(re.match(r"^#{1,4}\s+", line.strip()))
        if is_heading and current_len >= chunk_size * 0.5 and current:
            chunks.append("\n".join(current))
            current, current_len = [line], len(line)
            continue
        current.append(line)
        current_len += len(line) + 1
        if current_len > chunk_size:
            block = "\n".join(current)
            split_at = -1
            for sep in ["。", ".\n", "\n\n"]:
                idx = block.rfind(sep)
                if idx > chunk_size * 0.4:
                    split_at = idx + 1
                    break
            if split_at > 0:
                chunks.append(block[:split_at])
                remaining = block[max(0, split_at - overlap):].strip()
                current, current_len = [remaining] if remaining else [], len(remaining)
            else:
                chunks.append(block)
                current, current_len = [], 0
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def _detect_section_headers(lines: list) -> list:
    """检测规则书中自然出现的章节标题，返回行号列表"""
    known_sections = {
        "Object of the game", "Preparing to play", "Cards",
        "Starting player", "Play of the game",
        "The Actions", "Action A", "Action B", "Action C", "Action D",
        "Components", "Component List", "Setup", "Gameplay",
        "Game overview", "The game rounds",
        "Overview", "The App", "Using This Document",
        "First Campaign Setup", "Roles", "Game Rules",
        "Playing the Game", "How to Play", "Game End",
        "Scoring", "Victory Points", "Winning the Game",
    }
    headers = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) < 3 or len(s) > 90:
            continue
        if s.startswith(("•", "-", "♦", "×", "(", "*", "+")):
            continue
        if s[0].isdigit() and "/" in s:
            continue
        if s.isdigit():
            continue
        preceded_by_blank = (i == 0) or (i > 0 and not lines[i - 1].strip())
        is_header = False
        # 规则 1: 全大写短标题
        if (preceded_by_blank and s.isupper()
                and sum(1 for c in s if c.isalpha()) > 5):
            is_header = True
        # 规则 2: Title Case
        if (preceded_by_blank and " " in s and len(s) < 60
                and all(w[0].isupper() if w[0].isalpha() else True
                        for w in s.split() if w)):
            is_header = True
        # 规则 3: 数字标题
        if (preceded_by_blank or i < 5) and re.match(r"^\d+\.\s+[A-Z]", s):
            is_header = True
        # 规则 4: 已知固定章节名
        if s in known_sections:
            is_header = True
        # 规则 5: "Phase N:" / "Action X" 模式
        if re.match(r"^(Phase \d+|Action [A-D])\b", s):
            is_header = True
        if is_header:
            headers.append(i)
    return headers


def smart_chunk_text(text: str, game: str = "",
                     max_chunk_size: int = 800,
                     min_chunk_size: int = 200) -> List[dict]:
    """
    智能分段：按章节语义保持完整，替代 sliding-window。

    Phase 1 — 检测章节标题
    Phase 2 — 按标题分组内容
    Phase 3 — 长段在段落边界处拆分

    返回 List[dict]，每项含 text, section, game, chunk_id
    """
    lines = text.split("\n")
    header_lines = _detect_section_headers(lines)
    sections = []
    prev_h = 0
    for h in header_lines:
        if prev_h < h:
            sections.append({
                "header": lines[prev_h].strip() if prev_h > 0 else "General",
                "content": "\n".join(lines[prev_h:h]),
            })
            prev_h = h
    if prev_h < len(lines):
        header = lines[prev_h].strip() if prev_h > 0 else "General"
        sections.append({
            "header": header,
            "content": "\n".join(lines[prev_h:]),
        })
    if not sections:
        raw = chunk_text(text)
        return [{
            "text": c, "section": game or "General",
            "game": game, "chunk_id": i,
        } for i, c in enumerate(raw)]
    chunks = []
    chunk_id = 0

    def emit_chunk(header: str, content: str):
        nonlocal chunk_id
        content = content.strip()
        if not content or len(content) < min_chunk_size:
            return
        if len(content) <= max_chunk_size:
            chunks.append({
                "text": content, "section": header,
                "game": game, "chunk_id": chunk_id,
            })
            chunk_id += 1
            return
        paragraphs = re.split(r"\n\s*\n", content)
        if len(paragraphs) <= 1:
            step = max_chunk_size
            for i in range(0, len(content), step):
                block = content[i:i + step]
                if len(block) >= min_chunk_size:
                    chunks.append({
                        "text": block, "section": header,
                        "game": game, "chunk_id": chunk_id,
                    })
                    chunk_id += 1
            return
        current, current_len = [], 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > max_chunk_size and current:
                block = "\n\n".join(current)
                if len(block) >= min_chunk_size:
                    chunks.append({
                        "text": block, "section": header,
                        "game": game, "chunk_id": chunk_id,
                    })
                    chunk_id += 1
                current, current_len = [para], len(para)
            else:
                current.append(para)
                current_len += len(para) + 2
        if current:
            block = "\n\n".join(current)
            if len(block) >= min_chunk_size:
                chunks.append({
                    "text": block, "section": header,
                    "game": game, "chunk_id": chunk_id,
                })
                chunk_id += 1
    for sec in sections:
        header = sec["header"]
        content = sec["content"]
        lines_c = content.split("\n")
        if lines_c and lines_c[0].strip() == header:
            content = "\n".join(lines_c[1:])
        emit_chunk(header, content)
    if not chunks:
        chunks.append({
            "text": text.strip()[:max_chunk_size],
            "section": game or "General",
            "game": game, "chunk_id": 0,
        })
    return chunks




# ============================================================
# 单游戏向量存储
# ============================================================

class GameVectorStore:
    """单个桌游的向量索引"""

    def __init__(self, game_name: str = "",
                 model_name: str = "BAAI/bge-m3"):
        self.game = game_name
        self.model_name = model_name
        self._model = None
        self.chunks: List[str] = []
        self.metas: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    # ---- 内部 ----

    def _load_model(self, local_files_only: bool = False):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        # 优先尝试在线下载，失败后回退到本地缓存
        if not local_files_only:
            try:
                self._model = SentenceTransformer(self.model_name)
                return self._model
            except Exception:
                pass
        # 本地缓存模式
        self._model = SentenceTransformer(self.model_name, local_files_only=True)
        return self._model

    # ---- 构建 ----

    # ---- 截断 ----

    def _truncate_chunks(self, chunks: List[str],
                         max_tokens: int = 512) -> List[str]:
        """截断超长 chunk 到模型能处理的最大 token 数，避免 OOM"""
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True)
        except Exception:
            max_chars = max_tokens * 4
            truncated = []
            for c in chunks:
                if len(c) > max_chars:
                    truncated.append(c[:max_chars])
                else:
                    truncated.append(c)
            diff = sum(len(c) - len(t) for c, t in zip(chunks, truncated)
                       if len(c) > len(t))
            if diff:
                print(f"    ⚠ 回退截断 {diff} 字符")
            return truncated
        truncated = []
        over_count = 0
        for c in chunks:
            tokens = tok.encode(c, truncation=True, max_length=max_tokens)
            t = tok.decode(tokens, skip_special_tokens=True)
            if len(t) < len(c):
                over_count += 1
            truncated.append(t)
        if over_count:
            print(f"    ⚠ 截断 {over_count}/{len(chunks)} 块 (max {max_tokens} tokens)")
        return truncated

    # ---- 构建 ----

    def build(self, chunks: List[str], metas: Optional[List[dict]] = None,
              max_tokens: int = 512):
        """嵌入并建索引（自动截断超长 chunk）"""
        self.chunks = self._truncate_chunks(list(chunks), max_tokens=max_tokens)
        self.metas = metas or [{} for _ in chunks]
        for i, m in enumerate(self.metas):
            if "game" not in m:
                m["game"] = self.game
        model = self._load_model(local_files_only=True)
        # 显存不足时切到 CPU 编码
        device = getattr(model, "device", None)
        if device is not None and str(device).startswith("cuda"):
            free_mem = torch.cuda.get_device_properties(0).total_memory                        - torch.cuda.memory_allocated(0)
            if free_mem < 4 * 1024 ** 3:
                print(f"    ⚠ GPU 显存不足 ({free_mem/1024**3:.1f} GB)，切到 CPU 编码")
                model = model.to("cpu")
        self.embeddings = model.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=False
        )
        print(f"  [{self.game}] {len(self.chunks)} 块 → 嵌入 shape {self.embeddings.shape}")

    # ---- 检索 ----

    def retrieve(self, query: str, top_k: int = 5,
                 min_score: float = 0.15) -> List[dict]:
        """语义检索，返回 [{"chunk", "score", ...meta}, ...]"""
        if self.embeddings is None:
            return []
        model = self._load_model(local_files_only=True)
        q_vec = model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_vec.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for i in top_indices:
            if scores[i] >= min_score:
                results.append({
                    "chunk": self.chunks[i],
                    "score": round(float(scores[i]), 4),
                    "game": self.game,
                    **self.metas[i],
                })
        return results

    # ---- 持久化 ----

    def save(self, dir_path: str):
        """保存嵌入 + 元数据到目录"""
        os.makedirs(dir_path, exist_ok=True)
        safe = re.sub(r"[^\w]", "_", self.game.lower().strip())
        emb_path = os.path.join(dir_path, f"{safe}.npy")
        meta_path = os.path.join(dir_path, f"{safe}.json")
        np.save(emb_path, self.embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "game": self.game,
                "model": self.model_name,
                "chunks": self.chunks,
                "metas": self.metas,
            }, f, ensure_ascii=False, indent=1)
        return meta_path

    def load(self, dir_path: str):
        """从目录加载嵌入 + 元数据"""
        safe = re.sub(r"[^\w]", "_", self.game.lower().strip())
        emb_path = os.path.join(dir_path, f"{safe}.npy")
        meta_path = os.path.join(dir_path, f"{safe}.json")
        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"索引文件未找到: {dir_path}/{safe}.*")
        self.embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metas = data["metas"]
        self.game = data["game"]
        self.model_name = data.get("model", self.model_name)
        print(f"  [{self.game}] 已加载 {len(self.chunks)} 块 ({self.embeddings.shape})")
        return self


# ============================================================
# 多游戏向量存储
# ============================================================

class MultiGameVectorStore:
    """管理多个桌游的向量索引，支持跨游戏检索"""

    def __init__(self, index_dir: str = "vectors/indexes",
                 model_name: str = "BAAI/bge-m3"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.games: dict[str, GameVectorStore] = {}

    # ---- 构建 ----

    def build_from_rules(self, rules_dir: str = "data/docs/") -> dict:
        """
        从规则文件构建所有桌游的向量索引。

        对每个 .txt 文件：
          1. smart_chunk_text 分块
          2. 构建 GameVectorStore
          3. 保存到 self.games
        """
        for fp in sorted(glob(os.path.join(rules_dir, "*.txt"))):
            name = Path(fp).stem
            game_name = re.sub(r"-rulebook$", "", name).replace("_", " ").strip()

            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()

            chunks_data = smart_chunk_text(text, game=game_name)
            chunks = [c["text"] for c in chunks_data]
            metas = [{
                "section": c.get("section", ""),
                "source": Path(fp).name,
                "chunk_id": c["chunk_id"],
            } for c in chunks_data]

            store = GameVectorStore(game_name, self.model_name)
            store.build(chunks, metas)
            self.games[game_name] = store

        return self.games

    def add_game(self, store: GameVectorStore):
        """手动添加一个已构建的游戏索引"""
        self.games[store.game] = store

    # ---- 检索 ----

    def retrieve(self, query: str, top_k: int = 5,
                 game_names: Optional[List[str]] = None,
                 min_score: float = 0.15) -> List[dict]:
        """
        跨游戏或指定游戏检索。

        Args:
            query: 中文或英文问题
            top_k: 返回结果数
            game_names: 限制检索哪些游戏（None = 全部）
            min_score: 余弦相似度最低阈值

        Returns:
            [{chunk, score, game, section, ...}, ...]
        """
        target_games = game_names or list(self.games.keys())
        all_results: List[dict] = []
        for name in target_games:
            if name in self.games:
                results = self.games[name].retrieve(query, top_k=top_k,
                                                     min_score=min_score)
                all_results.extend(results)
        all_results.sort(key=lambda x: -x["score"])
        return all_results[:top_k]

    # ---- 持久化 ----

    def save_all(self):
        """保存所有游戏索引到 self.index_dir"""
        os.makedirs(self.index_dir, exist_ok=True)
        for store in self.games.values():
            store.save(self.index_dir)
        print(f"  [存储] 已保存 {len(self.games)} 个游戏索引 → {self.index_dir}/")

    def load_all(self):
        """从 self.index_dir 加载所有游戏索引"""
        self.games = {}
        for fp in sorted(glob(os.path.join(self.index_dir, "*.json"))):
            with open(fp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            game_name = meta["game"]
            store = GameVectorStore(game_name, self.model_name)
            store.load(self.index_dir)
            self.games[game_name] = store
        print(f"  [加载] 已加载 {len(self.games)} 个游戏索引 ← {self.index_dir}/")
        return self.games

    def get_game_names(self) -> List[str]:
        return list(self.games.keys())


# ============================================================
# 上下文格式化
# ============================================================

def format_context(results: List[dict], max_chars: int = 2000) -> str:
    """格式化检索结果为单段文本，用于拼接 prompt"""
    parts = []
    total = 0
    for r in results:
        block = r["chunk"]
        total += len(block)
        if total > max_chars and parts:
            break
        parts.append(block)
    return "\n\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
