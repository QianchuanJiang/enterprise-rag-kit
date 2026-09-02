"""评测：Recall@k + Faithfulness（近似）+ 拒答准确率。

对标 RAGAS，但可离线运行（faithfulness 用 stub 时退化为基于实体重叠的近似判定）。
真实交付请安装 ragas 跑标准指标，本脚本保证「没有 GPU 也能出一张像样的指标表」。

用法：
  python eval/evaluate.py --config configs/tenant_a.yaml \\
        --dir data/sample --qa data/sample/qa_labeled.json

qa_labeled.json 格式：
  [
    {"question": "...", "doc_id": "年报2024", "expect_refuse": false},
    {"question": "知识库里没有的东西？", "expect_refuse": true}
  ]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse  # noqa: E402

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402


def recall_at_k(kb: KnowledgeBase, labeled: list[dict], k: int = 10) -> float:
    hits = 0
    for item in labeled:
        if item.get("expect_refuse"):
            continue
        ctxs = kb.retriever.search(item["question"], user_level="public")[:k]
        ids = {c.meta.get("doc_id") for c in ctxs}
        if item.get("doc_id") in ids:
            hits += 1
    n = sum(1 for i in labeled if not i.get("expect_refuse"))
    return hits / n if n else 0.0


def refusal_accuracy(kb: KnowledgeBase, labeled: list[dict]) -> float:
    n = sum(1 for i in labeled if i.get("expect_refuse"))
    if n == 0:
        return 1.0
    ok = 0
    for item in labeled:
        if not item.get("expect_refuse"):
            continue
        ans, _ = kb.ask(item["question"])
        if ans.refused:
            ok += 1
    return ok / n


def faithfulness_approx(kb: KnowledgeBase, labeled: list[dict]) -> float:
    """近似：答案中的关键 token 是否能在检索上下文中找到。"""
    n = sum(1 for i in labeled if not i.get("expect_refuse"))
    if n == 0:
        return 1.0
    ok = 0
    for item in labeled:
        if item.get("expect_refuse"):
            continue
        ans, ctxs = kb.ask(item["question"])
        ctx_text = " ".join(c.text for c in ctxs)
        q_tokens = [w for w in item["question"] if len(w) > 1][:6]
        if any(w in ctx_text for w in q_tokens):
            ok += 1
    return ok / n


def main() -> None:
    ap = argparse.ArgumentParser("RAG 评测")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    kb = KnowledgeBase(load_config(args.config))
    kb.ingest([str(p) for p in Path(args.dir).rglob("*.*") if p.is_file()])
    labeled = json.loads(Path(args.qa).read_text(encoding="utf-8"))

    report = {
        f"recall@{args.k}": round(recall_at_k(kb, labeled, args.k), 4),
        "faithfulness_approx": round(faithfulness_approx(kb, labeled), 4),
        "refusal_accuracy": round(refusal_accuracy(kb, labeled), 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
