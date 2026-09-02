#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知擎 RAG 框架 · 真实链路第一批指标（样例 A）。

与 smoke_test.py 的区别：这里用真实模型，产出可复现的真实指标。
- 嵌入：本地 Ollama BGE-M3（1024 维，零成本）
- 生成：云端智谱 GLM-4.6V-Flash（OpenAI 兼容，免费档）
- 语料：data/smoke/ 下 2 份合成年报（.md）

运行：
    cd rag-kit
    python scripts/real_run.py

说明：本脚本所有指标均由真实模型跑出，可复现；语料为极小样本（2 份），
仅用于验证「本地 BGE-M3 + 云端 GLM」链路真实可用并产出基线数字。
真实交付指标请在 5–10 份真实年报上重跑，并如实记录。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# 嵌入走本地 Ollama BGE-M3；密钥从 .env 读取
os.environ.setdefault("RAG_EMBEDDER", "ollama")
load_dotenv(ROOT / ".env")

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402

CONFIG = ROOT / "configs" / "real.yaml"
SAMPLE_DIR = ROOT / "data" / "smoke"


def _files(d: Path) -> list[str]:
    return [str(p) for p in Path(d).rglob("*.*") if p.is_file()]


def _norm_nums(s: str) -> set[str]:
    """抽取数字序列（去逗号），用于轻量 grounding 校验。"""
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*\.?\d*", s) if len(m) >= 3}


def recall_at_k(retrieved_per_q: list[list[str]], gold: list[str], k: int) -> float:
    """retrieved_per_q[i] 是第 i 个 query 的 top-k chunk 文本列表；
    gold[i] 是应被命中的关键词；命中即 Recall=1。"""
    if not gold:
        return 0.0
    hit = 0
    for chunks, g in zip(retrieved_per_q, gold):
        top = chunks[:k]
        if any(g in c for c in top):
            hit += 1
    return hit / len(gold)


def main() -> int:
    print("=== 知擎 RAG 框架 · 真实链路第一批指标（样例 A）===\n")
    cfg = load_config(str(CONFIG))
    kb = KnowledgeBase(cfg, backend="memory")

    # 1) 入库
    rep = kb.ingest(_files(SAMPLE_DIR))
    n = rep["ingested_chunks"]
    print(f"[入库] chunks={n}  quality={rep.get('quality')}\n")
    if n == 0:
        print("❌ 入库为空，无法评测")
        return 1

    # 2) 检索 Recall@k —— 黄金集（query -> 应命中的关键词）
    gold_set = [
        ("平安银行 2024 年营业收入是多少？", "营业收入"),
        ("平安银行 2024 年净利润是多少？", "净利润"),
        ("平安银行核心一级资本充足率是多少？", "核心一级资本充足率"),
        ("贵州茅台 2024 年毛利率是多少？", "毛利率"),
        ("贵州茅台 2024 年营业收入是多少？", "营业收入"),
    ]
    retrieved_texts: list[list[str]] = []
    for q, _ in gold_set:
        ctx = kb.retriever.search(q, user_level="public")
        retrieved_texts.append([c.text for c in ctx])

    print("=== 检索指标（真实 BGE-M3 向量 + BM25 + RRF）===")
    for k in (1, 3, 5, 8):
        r = recall_at_k(retrieved_texts, [g for _, g in gold_set], k)
        print(f"  Recall@{k:<2} = {r*100:5.1f}%")
    print()

    # 3) 生成（真实 GLM-4.6V）+ 溯源 + grounding 校验
    print("=== 生成指标（真实 GLM-4.6V-Flash）===")
    gen_queries = [
        "平安银行 2024 年净利润是多少？",
        "贵州茅台 2024 年营业收入是多少？",
    ]
    nonempty = 0
    grounded = 0
    cited = 0
    for q in gen_queries:
        try:
            ans, ctx = kb.ask(q)
        except Exception as exc:  # noqa: BLE001
            print(f"  [生成失败] {q} -> {exc}")
            continue
        context_blob = "\n".join(c.text for c in ctx)
        nums_ans = _norm_nums(ans.text)
        nums_ctx = _norm_nums(context_blob)
        is_grounded = bool(nums_ans & nums_ctx) or len(nums_ans) == 0
        nonempty += bool(ans.text)
        cited += len(ans.citations) > 0
        grounded += is_grounded
        print(f"\n  Q: {q}")
        print(f"  A: {ans.text[:200]}")
        print(f"  溯源: {ans.citations}")
        print(f"  数值 grounding: {'OK' if is_grounded else '⚠ 答案数字未在上下文找到'}"
              + (f"  (ans={sorted(nums_ans)[:5]} ctx={sorted(nums_ctx)[:5]})" if not is_grounded else ""))
    n_gen = len(gen_queries)
    print(f"\n  非空率={nonempty}/{n_gen}  溯源率={cited}/{n_gen}  数值 grounding={grounded}/{n_gen}\n")

    # 4) 拒答（无依据）
    ans_empty, _ = kb.ask("请告诉我 2025 年火星基地的运营预算。")
    refused_ok = ans_empty.refused is True
    print(f"=== 拒答测试 ===\n  无依据问题拒答: {'OK' if refused_ok else 'FAIL'}  (reason={ans_empty.reason})\n")

    # 5) 汇总
    print("=== 指标汇总（样例 A · 真实链路基线）===")
    print(f"  入库 chunks          : {n}")
    print(f"  Recall@1 / @3 / @5   : "
          f"{recall_at_k(retrieved_texts, [g for _,g in gold_set], 1)*100:.0f}% / "
          f"{recall_at_k(retrieved_texts, [g for _,g in gold_set], 3)*100:.0f}% / "
          f"{recall_at_k(retrieved_texts, [g for _,g in gold_set], 5)*100:.0f}%")
    print(f"  生成非空率           : {nonempty}/{n_gen}")
    print(f"  溯源率               : {cited}/{n_gen}")
    print(f"  数值 grounding 率    : {grounded}/{n_gen}")
    print(f"  无依据拒答           : {'通过' if refused_ok else '未通过'}")
    print("\n注：语料为 2 份合成 .md，仅验证链路；真实交付请在 5–10 份真实年报上重跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
