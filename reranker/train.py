#!/usr/bin/env python3
"""
Cross-Encoder Reranker 微调
"""
import json, os, sys, time, random, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/reranker_train.jsonl")
    parser.add_argument("--output", default="reranker/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    t0 = time.time()

    # 1. 数据
    triples = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                triples.append(json.loads(line))
    logger.info(f"已加载 {len(triples)} 条三元组")

    examples = []
    for t in triples:
        examples.append(InputExample(texts=[t["query"], t["positive"]], label=1.0))
        examples.append(InputExample(texts=[t["query"], t["negative"]], label=0.0))
    random.shuffle(examples)
    logger.info(f"训练样本: {len(examples)} 对")

    n_eval = max(1, len(examples) // 20)
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]
    logger.info(f"  训练: {len(train_examples)}  验证: {len(eval_examples)}")

    # 2. 模型
    backbone = os.path.join(os.path.dirname(__file__), "checkpoints")
    model = CrossEncoder(backbone, num_labels=1, local_files_only=True)
    logger.info("模型加载完成")

    # 3. Dataloader
    train_dl = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)

    # 4. 训练
    logger.info(f"开始微调 (epochs={args.epochs}, batch_size={args.batch_size})")
    model.fit(
        train_dataloader=train_dl,
        epochs=args.epochs,
        warmup_steps=100,
        optimizer_params={"lr": args.lr},
        weight_decay=0.01,
        show_progress_bar=True,
    )

    # 5. 验证
    correct = 0
    for ex in eval_examples:
        score = model.predict([(ex.texts[0], ex.texts[1])])[0]
        pred = 1.0 if score >= 0.5 else 0.0
        correct += 1 if pred == ex.label else 0
    acc = correct / len(eval_examples)
    logger.info(f"验证准确率: {acc:.4f}")

    # 6. 保存
    os.makedirs(args.output, exist_ok=True)
    model.save(args.output)

    # 7. 测试
    test = [
        ("农家乐怎么喂养", "must be fed. Growing your family is important..."),
        ("农家乐怎么喂养", "Stone Age scoring rules: each hut costs..."),
    ]
    scores = model.predict(test)
    for (q, c), s in zip(test, scores):
        logger.info(f"  ({s:.4f}) {q[:30]} | {c[:40]}...")

    elapsed = time.time() - t0
    logger.info(f"✅ 完成! 耗时={elapsed:.0f}s  acc={acc:.4f}  保存到 {args.output}")

if __name__ == "__main__":
    main()
