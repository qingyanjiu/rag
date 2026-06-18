# 文档问答 — RAG + Cross-Encoder 重排序检索系统

文档 → 向量索引 → 提问 → **bge-m3 语义检索** → **XLM-RoBERTa reranker 重排序** → LLM 生成回答。

使用 `BAAI/bge-m3` 多语言嵌入 + 余弦相似度 + **Cross-Encoder 重排** + OpenAI 兼容接口 LLM。

---

## 向量处理完整流程

### 构建阶段

```
data/docs/*.txt     ← 英文规则书原文
        │
        ▼
smart_chunk_text()             ← 智能分块
  Phase 1  检测章节标题（全大写/Title Case/数字编号/已知章节名）
  Phase 2  按标题分组内容
  Phase 3  长段在段落边界处拆分（max 800 chars）
        │
        ▼
bge-m3 encode                  ← 多语言嵌入
  • 自动截断超长 chunk（max 512 tokens，避免 GPU OOM）
  • 每块生成 1024 维语义向量
  • 余弦归一化（normalize_embeddings=True）
        │
        ▼
vectors/indexes/               ← 持久化到磁盘
  agricola.npy + agricola.json   (42 块)
  lotr_journeys.npy + .json      (74 块)
  stone_age.npy + .json          (12 块)
        │
        ▼
MultiGameVectorStore.save_all()
  • .npy = 嵌入矩阵
  • .json = chunk 原文 + section + source + chunk_id
```

**构建一条命令**：
```bash
bash 02_build_index.sh
# → 自动遍历 data/docs/*.txt → 分块 → 嵌入 → 存入 vectors/indexes/
```

### 推理阶段（含 Reranker 重排序）

```
用户提问                     ← "中洲征途中，阿拉贡有哪些技能卡"
        │
        ▼
_detect_game(query)          ← 关键词匹配
  • "农家乐"/"agricola"       → 只搜 agricola
  • "中洲"/"lotr"/"魔戒"     → 只搜 lotr journeys
  • "石器时代"/"stone age"   → 只搜 stone age
  • 未匹配                    → 搜全部后按分数排序
        │
        ▼
bge-m3 encode(query)         ← 将中文问题编码为 1024 维向量
        │
        ▼
余弦相似度检索                 ← query 向量 × 所有 chunk 向量
  bge-m3 粗筛 → 取 top_k=20
        │
        ▼
XLM-RoBERTa Cross-Encoder    ← 对每条候选精细打分（全连接注意力）
  reranker.score(query, candidates)
        │
        ▼
重排序取 top_k=3              ← 按 cross-encoder 分数降序
        │
        ▼
format_context(results)      ← 拼接参考规则文本（max 2000 字符）
        │
        ▼
system: "参考规则:\n{context}"
user:   "{query}"
        │
        ▼
LLM 生成回答                    ← 基于检索到的规则原文作答
```

---

## 向量检索模块

`vectors/` 是纯向量检索子系统，不依赖大模型或训练框架。

### 文件结构

```
vectors/
├── store.py          GameVectorStore + MultiGameVectorStore + smart_chunk_text
├── build.py          构建向量索引 CLI
├── retrieve.py       检索测试 CLI（纯引擎，不涉及 LLM）
└── indexes/          已构建的向量索引
    ├── agricola.npy / .json
    ├── lotr_journeys.npy / .json
    └── stone_age.npy / .json
```

### 核心设计

| 特性 | 说明 |
|---|---|
| **每游戏独立索引** | 各游戏分开建库，检索时可全搜或按 game 过滤 |
| **规则原文直接存向量** | 不用翻译或摘要，原文 chunk → bge-m3 嵌入 |
| **跨语言语义匹配** | bge-m3 支持 100+ 语言，中文问题直查英文原文 |
| **自动截断** | 嵌入前截断超长 chunk（max 512 tokens） |
| **持久化** | `.npy` + `.json`，加载无需重新编码 |
| **GPU 显存保护** | 空闲显存 < 4GB 时自动切到 CPU 编码 |

### 用法

```python
from vectors.store import MultiGameVectorStore, format_context

# 构建（首次）
vdb = MultiGameVectorStore("vectors/indexes")
vdb.build_from_rules("data/docs/")
vdb.save_all()

# 加载（后续）
vdb = MultiGameVectorStore("vectors/indexes")
vdb.load_all()

# 检索
results = vdb.retrieve("农家乐每回合能做什么", top_k=3)
for r in results:
    print(r["game"], r["score"], r["chunk"][:80])

# 格式化上下文
context = format_context(results, max_chars=2000)
```

### CLI 工具

```bash
# 构建索引
bash 02_build_index.sh

# 交互式检索测试（纯引擎，不涉及大模型）
python vectors/retrieve.py --interactive

# 单次查询
python vectors/retrieve.py --query "农家乐每回合能做什么"
python vectors/retrieve.py --query "中洲征途中战斗规则" --game lotr-journeys
```

---

## Reranker 重排序模块

`reranker/inference.py` 封装了一个基于 [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large) 的 Cross-Encoder 重排器（567M 参数），
对 bge-m3 的初筛结果做精细二分类判分，消除跨游戏噪声和语义模糊的段落。

与 bge-m3 的对比：

| 维度 | bge-m3（初筛） | XLM-RoBERTa（精排） |
|------|---------------|-------------------|
| 模型类型 | 双编码器（Bi-Encoder） | 交叉编码器（Cross-Encoder） |
| 输入 | `[query]`, `[chunk]` 分别编码 | `[CLS] query [SEP] chunk [SEP]` |
| 速度 | 快，一次编码后余弦点积 | 慢，每对从头计算 |
| 精度 | 中等，适合大规模召回 | 高，适合精排 |
| 跨语言 | 100+ 语言（bge-m3 原生支持） | 100 语言（XLM-RoBERTa 原生支持） |
| 在项目中的角色 | 全库召回 top-20 | 重排 top-5 |

训练好的模型权重存放在 `reranker/checkpoints/`（2.27 GB），由 `data/reranker_train.jsonl`
（2400 条 query-positive-negative 三元组）微调得到。

### 文件结构

```
reranker/                       ← Reranker 重排序模块
├── __init__.py
├── inference.py                ← Cross-Encoder 重排序封装
├── train.py                    ← 训练脚本
├── test.py                     ← 对比测试脚本
└── checkpoints/                ← 训练好的 Cross-Encoder 模型
    ├── config.json             ← XLM-RoBERTa-large 架构配置
    ├── model.safetensors       ← 2.27 GB 权重（567M 参数）
    ├── tokenizer.json          ← 250K 词表
    └── ...

data/reranker_train.jsonl      ← 2400 条训练三元组（query, positive, negative）
```

### 使用方式

```python
from reranker.inference import Reranker

reranker = Reranker("reranker/checkpoints")

# 打分
scores = reranker.score("农家乐怎么获得木材", [
    "Build room(s) and/or Build Stable(s)...",
    "Forest, clay mound, quarry and river...",
])
# scores ≈ [0.90, 0.49] → 第一条确实更相关

# 或直接重排检索结果
results = vdb.retrieve("农家乐怎么获得木材", top_k=20)
reranked = reranker.rerank("农家乐怎么获得木材", results, top_k=5)
```

### CLI 测试

```bash
# 对比 bge-m3 与 cross-encoder 的检索排序效果
python reranker/test.py
```

### 重新训练

```bash
python reranker/train.py \
  --data data/reranker_train.jsonl \
  --output models/reranker \
  --epochs 2 \
  --batch-size 4 \
  --lr 2e-5
```

训练数据格式（每行一条 JSON）：
```json
{"query": "在《农家乐》游戏中，行动空间有哪些来源？",
 "positive": "Some actions are printed directly on the game boards...",
 "negative": "symbol to bake his two Grain into bread..."}
```

---

## 快速开始

```bash
cd rag

# 1. PDF 转 TXT
bash 01_pdf_to_txt.sh

# 2. 构建向量索引（自动安装 bge-m3）
bash 02_build_index.sh

# 3. 测试检索
bash 03_retrieve.sh

# 4. （可选）验证 reranker 重排序效果
bash 04_reranker_test.sh
```

### 搭配大模型推理（推荐：05_infer.sh）

```bash
# 设置 API 密钥
export OPENAI_API_KEY="sk-..."
export RAG_MODEL="gpt-4o-mini"        # 可选，默认 gpt-4o-mini
# export OPENAI_BASE_URL="..."          # 可选，可用于本地模型

# 提问
bash 05_infer.sh "免赔额多少"
bash 05_infer.sh -q "保险范围" --model deepseek-chat --top-k 8
bash 05_infer.sh "怎么报销" --no-sources
```

也可以直接用 `python infer/run.py` 传更多参数。

旧版 `infer_mac.py` / `inference.py` 仍保留可用。

# CUDA — 同上
python inference.py --no-adapter --vector --interactive
python inference.py --lora_dir ./outputs --vector --interactive
```

---

## 环境要求

| 组件 | 要求 |
|---|---|
| Python | 3.10+ |
| Mac | Apple Silicon（M1+），MLX |
| GPU | NVIDIA GPU ≥8GB，CUDA 11.8+ |
| 向量检索 | `pip install sentence-transformers`（bge-m3 自动下载） |
| 磁盘 | vectors/indexes/ 约 15 MB（3 个游戏） |

---

## 使用流程（PDF → TXT → 向量索引 → 检索）

把 PDF 文档放入 `data/docs/`，然后依次执行以下步骤：

### 1. 安装 PDF 转 TXT 工具（二选一）

```bash
# 方式 A：pypdf（轻量，纯 Python）
pip install pypdf

# 方式 B：poppler（更快，含 pdftotext 命令）
brew install poppler
```

### 2. PDF 转 TXT

```bash
# 方式 A：pypdf
python3 -c "
from pypdf import PdfReader
reader = PdfReader('data/docs/你的文件.pdf')
with open('data/docs/你的文件.txt', 'w') as f:
    for page in reader.pages:
        f.write(page.extract_text() + '\n')
"

# 方式 B：pdftotext
pdftotext -layout data/docs/你的文件.pdf data/docs/你的文件.txt
```

### 3. 构建向量索引

```bash
bash 02_build_index.sh
```

自动完成：
- 遍历 `data/docs/*.txt`
- `smart_chunk_text` 按章节分段
- `bge-m3` 生成 1024 维语义嵌入
- 持久化到 `vectors/indexes/`（每文档一个 `.npy` + `.json`）

### 4. 测试检索

```bash
# 交互模式
python vectors/retrieve.py --interactive

# 单次查询
python vectors/retrieve.py --query "你的问题"
```

### 5. RAG 推理 — 检索 + LLM 润色答案

```bash
# 设置 API 密钥（支持 OpenAI / 任意兼容格式的 API）
export OPENAI_API_KEY="sk-..."

# 指定模型（可选，默认 gpt-4o-mini）
export RAG_MODEL="gpt-4o-mini"

# 自定义 API 地址（可选，可用于本地模型）
# export OPENAI_BASE_URL="http://localhost:8000/v1"

# 提问
bash 05_infer.sh "免赔额多少"
bash 05_infer.sh -q "保险范围是什么" --model deepseek-chat
```

自动完成：bge-m3 检索 → 拼接上下文 → LLM 生成回答。

### 6. 验证 Reranker 重排序效果（可选）

```bash
bash 04_reranker_test.sh -q "你的问题"
bash 04_reranker_test.sh -q "问题1" -q "问题2"  # 多个问题
```

对比 bge-m3 初筛与 cross-encoder 精排的排序差异。

---

## 项目结构

```
deepseek-finetune/
├── 01_pdf_to_txt.sh             # PDF → TXT
│
├── 02_build_index.sh             # 构建向量索引（bge-m3 嵌入）
│
├── 03_retrieve.sh                # 检索测试
│
├── 04_reranker_test.sh           # Reranker 对比测试
│
├── 05_infer.sh                    # RAG 推理：检索 + LLM 润色答案
│
├── vectors/                     # 向量检索模块（核心）
│   ├── __init__.py
│   ├── store.py                 # GameVectorStore + MultiGameVectorStore + 智能分块
│   ├── retrieve.py              # 检索测试 CLI（纯引擎，不涉及 LLM）
│   ├── build.py                 # 构建索引 CLI
│   └── indexes/                 # 已构建的向量索引
│       ├── agricola.npy / .json
│       ├── lotr_journeys.npy / .json
│       └── stone_age.npy / .json
│
├── reranker/                    # Reranker 重排序模块
│   ├── __init__.py
│   ├── inference.py             # Cross-Encoder 重排序封装
│   ├── train.py                 # Reranker 训练脚本
│   ├── test.py                  # bge-m3 vs reranker 对比测试
│   └── checkpoints/             # 训练好的 Cross-Encoder 模型权重
│       ├── config.json
│       ├── model.safetensors    # 2.27 GB（567M 参数）
│       ├── tokenizer.json
│       └── ...
│
├── infer/                        # RAG 推理模块
│   └── run.py                    # 检索 + LLM 润色答案（OpenAI 兼容接口）
│

├── infer_mac.py                 # 推理脚本（Mac MLX）
├── inference.py                 # 推理脚本（CUDA）
│
├── data/
│   ├── docs/                    # 文档来源（放入你的文档 .txt）
│   ├── reranker_train.jsonl     # Reranker 训练数据（2400 条三元组）
│   ├── train.jsonl              # LLM 微调训练数据
│   └── eval.jsonl               # LLM 微调验证数据
│
├── train_mac.py                 # 训练脚本（Mac MLX，可选）
├── train.py                     # 训练脚本（CUDA，可选）
├── config_mac_rag.yaml          # 训练配置（Mac）
├── config_rag.yaml              # 训练配置（CUDA）
│
├── requirements.txt
├── setup_env.sh                 # CUDA 环境安装
└── outputs_mac_rag/             # Mac 训练输出
    └── adapter/```

## 常见问题

**Q: 检索会混到其他游戏的结果吗？**
默认按游戏名自动过滤。以"农家乐"开头的提问只搜 agricola，带"中洲"只搜 lotr journeys。
如果无法判断，才搜全部后按分数排序。

**Q: 向量检索结果不准？**
- 检查规则书 .txt 提取是否完整（PDF 转 TXT 质量是瓶颈）
- 调整 `top_k`（默认 3）或 `min_score`（默认 0.15）
- 检查文本有无乱码或 OCR 错误
- bge-m3 对中文问英文原文的跨语言场景效果较好

**Q: bge-m3 初筛后为什么还要加 reranker？**
bge-m3 把 query 和 chunk 各自编码成向量再比余弦相似度，
这相当于"压缩"了信息做匹配。Cross-Encoder 让 query 和 chunk 互相看对方（全连接注意力），
判断更精细。实测中，跨游戏检索时 bge-m3 可能会把"木材"相关的石器时代规则排到前面，
reranker 能正确地把农业革命的建房规则提到前面。

**Q: Reranker 需要 GPU 吗？**
强推荐 GPU。XLM-RoBERTa-large 在 CPU 上给 20 条候选重排需要 25-37 秒。
如果跑在 Mac 上，可以在 MLX 框架下跑推理（目前还未适配）。

**Q: Reranker 的训练数据怎么来的？**
`data/reranker_train.jsonl` 包含 2400 条 `query-positive-negative` 三元组。
query 来自 `data/train.jsonl` 和 `data/eval.jsonl` 中的问题，
positive 是 bge-m3 检索到的正确段落（人工或规则筛选），
negative 是从相同或不同游戏中采样的不相关段落。

**Q: 规则文本超长导致 OOM？**
已内置自动截断 — 超过 512 tokens 的 chunk 会被截断再嵌入。
GPU 空闲显存低于 4GB 时自动切到 CPU 编码。

**Q: 每次提问都要重新加载向量吗？**
不需要。索引在启动时一次性加载到内存（约 15 MB），后续只做查询编码 + 点积搜索。

**Q: 不想微调，直接用 LLM 回答？**
```bash
export OPENAI_API_KEY="sk-..."
bash 05_infer.sh "你的问题"
```
基于检索到的文档内容，通过 LLM 生成回答。支持 OpenAI / 任意兼容格式的 API。

**Q: 添加新文档的完整流程？**
```bash
# 1. PDF 转 TXT
bash 01_pdf_to_txt.sh
# 2. 构建索引
bash 02_build_index.sh
# 3. 测试检索
bash 03_retrieve.sh "新文档的问题"
#
# （可选）RAG 推理
bash 05_infer.sh "新文档的问题"
```

---

## 参考

- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Framework
- [Unsloth](https://github.com/unslothai/unsloth) — 高效微调框架
- [Qwen2.5 模型系列](https://huggingface.co/Qwen)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — 多语言语义嵌入模型
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large) — 跨语言预训练模型
