#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知擎 RAG 框架 · 真实 PDF 路线指标（样例 A · 8 家 · 真实 PDF 文件）。

与 real_expanded.py 的区别：
- 语料是 data/reports/raw/ 下的「真实 PDF 文件」（scripts/make_real_pdfs.py 生成）。
- 解析走 _parse_pymupdf（PyMuPDF 抽取），即交付客户时的 PDF 解析生产代码路径。
- 检索/嵌入(BGE-M3)/生成(GLM-4.6V)/拒答阈值 与扩量版完全一致，指标可横向对比。

先决：rag-forge 环境已装 PyMuPDF（fitz 1.28.0）；Ollama bge-m3 在线；.env 含 ZHIPU_API_KEY。
运行：
    cd rag-kit
    /path/to/rag-forge/bin/python scripts/real_pdf_eval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


os.environ.setdefault("RAG_EMBEDDER", "ollama")
_load_dotenv(ROOT / ".env")

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402

CONFIG = ROOT / "configs" / "real_pdf.yaml"
REPORT_DIR = ROOT / "data" / "reports" / "raw"
METRICS_OUT = ROOT / "data" / "reports" / "metrics_pdf.json"


def _files(d: Path) -> list[str]:
    return sorted(str(p) for p in Path(d).rglob("*.pdf") if p.is_file())


def _norm_nums(s: str) -> set[str]:
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*\.?\d*", s) if len(m) >= 3}


def recall_at_k(retrieved_per_q: list[list[str]], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = 0
    for chunks, g in zip(retrieved_per_q, gold):
        if any(g in c for c in chunks[:k]):
            hit += 1
    return hit / len(gold)


GOLD = [
    ("平安银行 2024 年营业收入是多少？", "营业收入"),
    ("平安银行 2024 年净利润是多少？", "净利润"),
    ("平安银行 不良贷款率是多少？", "不良贷款率"),
    ("贵州茅台 2024 年营业总收入是多少？", "营业总收入"),
    ("贵州茅台 毛利率是多少？", "毛利率"),
    ("贵州茅台 2024 年归母净利润是多少？", "归母净利润"),
    ("招商银行 核心一级资本充足率是多少？", "核心一级资本充足率"),
    ("招商银行 2024 年总资产是多少？", "总资产"),
    ("招商银行 净息差是多少？", "净息差"),
    ("中国平安 2024 年新业务价值是多少？", "新业务价值"),
    ("中国平安 综合偿付能力充足率是多少？", "综合偿付能力充足率"),
    ("宁德时代 2024 年全球动力电池市占率是多少？", "市占率"),
    ("宁德时代 2024 年归母净利润是多少？", "归母净利润"),
    ("宁德时代 研发投入是多少？", "研发投入"),
    ("美的集团 海外收入占比是多少？", "海外收入占比"),
    ("美的集团 2024 年营业总收入是多少？", "营业总收入"),
    ("恒瑞医药 2024 年研发投入是多少？", "研发投入"),
    ("恒瑞医药 毛利率是多少？", "毛利率"),
    ("五粮液 2024 年营业收入是多少？", "营业收入"),
    ("五粮液 毛利率是多少？", "毛利率"),
]

IRRELEVANT = [
    "请告诉我 2025 年火星基地的运营预算。",
    "特斯拉 2024 年第四季度全球交付量是多少？",
    "英伟达 H100 芯片采用多少纳米制程工艺？",
    "比亚迪 2024 年新能源汽车销量是多少？",
]

GEN_QUERIES = [
    "平安银行 2024 年净利润是多少？",
    "贵州茅台 2024 年营业总收入是多少？",
    "招商银行 核心一级资本充足率是多少？",
    "宁德时代 2024 年全球动力电池市占率是多少？",
    "恒瑞医药 2024 年研发投入是多少？",
    "五粮液 2024 年营业收入是多少？",
]


def main() -> int:
    print("=== 知擎 RAG 框架 · 真实 PDF 路线指标（样例 A · 8 家 · 真实 PDF）===\n")
    cfg = load_config(str(CONFIG))
    kb = KnowledgeBase(cfg, backend="memory")

    rep = kb.ingest(_files(REPORT_DIR))
    n = rep["ingested_chunks"]
    q = rep.get("quality", {})
    print(f"[入库] PDF 文件数={len(_files(REPORT_DIR))}  chunks={n}  "
          f"质检通过率={q.get('accept_rate')}  拒绝={q.get('rejected')}\n")
    if n == 0:
        print("入库为空，无法评测")
        return 1

    gold_texts: list[list[str]] = []
    rel_scores: list[float] = []
    for qtext, _ in GOLD:
        scored = kb.retriever.search(qtext, user_level="public", return_scores=True)
        gold_texts.append([c.text for c, _ in scored])
        rel_scores.append(max((s for _, s in scored), default=0.0))

    print("=== 检索指标（真实 BGE-M3 向量 + BM25 + RRF · PyMuPDF 抽取）===")
    recalls = {}
    for k in (1, 3, 5, 8):
        r = recall_at_k(gold_texts, [g for _, g in GOLD], k)
        recalls[k] = r
        print(f"  Recall@{k:<2} = {r*100:5.1f}%")
    print()

    irr_scores: list[float] = []
    print("=== 拒答阈值标定：检索最高分分布 ===")
    print("  相关 query 最高分：")
    for (qtext, _), sc in zip(GOLD, rel_scores):
        print(f"    {sc:.3f}  {qtext}")
    print(f"  -> 相关区间 [{min(rel_scores):.3f}, {max(rel_scores):.3f}]")
    print("  无关 query 最高分：")
    for qtext in IRRELEVANT:
        scored = kb.retriever.search(qtext, user_level="public", return_scores=True)
        sc = max((s for _, s in scored), default=0.0)
        irr_scores.append(sc)
        print(f"    {sc:.3f}  {qtext}")
    print(f"  -> 无关区间 [{min(irr_scores):.3f}, {max(irr_scores):.3f}]")
    print()

    min_rel, max_irr = min(rel_scores), max(irr_scores)
    if min_rel > max_irr:
        suggested = round((min_rel + max_irr) / 2, 2)
        sep = "干净分离"
    else:
        suggested = round(min_rel, 2)
        sep = "存在重叠，取最弱相关分为阈值（偏保守）"
    print(f"  阈值建议：{suggested}  （{sep}；当前配置 refuse_threshold={cfg.guard.refuse_threshold}）\n")

    print("=== 生成指标（真实 GLM-4.6V-Flash）===")
    nonempty = cited = grounded = 0
    gen_detail = []
    for i, qtext in enumerate(GEN_QUERIES):
        if i > 0:
            time.sleep(12)
        try:
            ans, ctx = kb.ask(qtext)
        except Exception as exc:  # noqa: BLE001
            print(f"  [生成失败] {qtext} -> {exc}")
            continue
        context_blob = "\n".join(c.text for c in ctx)
        nums_ans = _norm_nums(ans.text)
        nums_ctx = _norm_nums(context_blob)
        is_grounded = bool(nums_ans & nums_ctx) or len(nums_ans) == 0
        nonempty += bool(ans.text)
        cited += len(ans.citations) > 0
        grounded += is_grounded
        gen_detail.append({
            "query": qtext, "answer": ans.text[:200],
            "citations": ans.citations, "grounded": is_grounded,
        })
        print(f"\n  Q: {qtext}")
        print(f"  A: {ans.text[:160]}")
        print(f"  溯源: {ans.citations}  数值grounding: {'OK' if is_grounded else 'FAIL'}")
    n_gen = len(GEN_QUERIES)
    print(f"\n  非空率={nonempty}/{n_gen}  溯源率={cited}/{n_gen}  数值grounding={grounded}/{n_gen}\n")

    refused_ok = 0
    for qtext in IRRELEVANT:
        ans, _ = kb.ask(qtext)
        if ans.refused:
            refused_ok += 1
    print(f"=== 拒答测试（无关 query）===")
    print(f"  无依据拒答通过: {refused_ok}/{len(IRRELEVANT)}\n")

    summary = {
        "route": "real_pdf (PyMuPDF extraction on real .pdf files)",
        "corpus": {"pdf_files": len(_files(REPORT_DIR)), "chunks": n,
                   "quality_accept_rate": q.get("accept_rate")},
        "recall": {f"@{k}": round(v, 3) for k, v in recalls.items()},
        "relevant_score_range": [round(min(rel_scores), 3), round(max(rel_scores), 3)],
        "irrelevant_score_range": [round(min(irr_scores), 3), round(max(irr_scores), 3)],
        "suggested_refuse_threshold": suggested,
        "generation": {"nonempty": f"{nonempty}/{n_gen}",
                       "cited": f"{cited}/{n_gen}",
                       "grounded": f"{grounded}/{n_gen}"},
        "refusal_pass": f"{refused_ok}/{len(IRRELEVANT)}",
    }
    METRICS_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 指标汇总（样例 A · 真实 PDF 路线）===")
    print(f"  入库 pdf/chunks      : {len(_files(REPORT_DIR))} / {n}")
    print(f"  Recall@1/@3/@5/@8    : "
          f"{recalls[1]*100:.0f}% / {recalls[3]*100:.0f}% / {recalls[5]*100:.0f}% / {recalls[8]*100:.0f}%")
    print(f"  生成非空/溯源/ground : {nonempty}/{n_gen} · {cited}/{n_gen} · {grounded}/{n_gen}")
    print(f"  无依据拒答           : {refused_ok}/{len(IRRELEVANT)}")
    print(f"  相关分区间/无关分区间: [{min(rel_scores):.3f},{max(rel_scores):.3f}] / "
          f"[{min(irr_scores):.3f},{max(irr_scores):.3f}]")
    print(f"  建议 refuse_threshold : {suggested}")
    print(f"\n  指标已写入: {METRICS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
