#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知擎 RAG 框架 · 场景 B 权限审计指标（合规 Guard）。

特点：
- 真实 BGE-M3 嵌入（Ollama 本地），检索/ACL/审计全部真实跑出，可复现。
- 核心安全指标（越权拦截/最小权限/审计/越权可溯）**不依赖 LLM**——
  这正是设计要点：权限安全不应寄托在模型「懂事」上，而在检索层硬拦截。
- 可选：末尾用真实 GLM-4.6V 演示管理员视角下能给出带溯源的答案（best-effort，429 不致命）。

先决：rag-forge 环境；Ollama bge-m3 在线；.env 含 ZHIPU_API_KEY（仅可选演示用）。
运行：
    cd rag-kit
    /path/to/rag-forge/bin/python scripts/real_b_eval.py
"""

from __future__ import annotations

import json
import os
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
from permissions import LEVEL_RANK, can_access  # noqa: E402

CONFIG = ROOT / "configs" / "tenant_b.yaml"
CORPUS_DIR = ROOT / "data" / "compliance"
METRICS_OUT = ROOT / "data" / "compliance" / "metrics_b.json"


def _files(d: Path) -> list[str]:
    return sorted(str(p) for p in Path(d).rglob("*.md") if p.is_file())


def _stem(c) -> str:
    return Path(c.meta.get("source", "")).stem


# 每条查询：问题、目标文档 stem、目标文档密级、敏感度类别
QUERIES = [
    {"q": "云启科技 2024 年营业收入和净利润是多少？", "stem": "pub_云启科技_年报", "lvl": "public", "cat": "public"},
    {"q": "云启科技 2025 年内部营收目标是多少？", "stem": "int_云启科技_经营计划", "lvl": "internal", "cat": "internal"},
    {"q": "云启科技 拟收购的标的公司和交易对价是多少？", "stem": "res_云启科技_并购预案", "lvl": "restricted", "cat": "restricted"},
    {"q": "远东智造 2024 年营业收入和毛利率是多少？", "stem": "pub_远东智造_年报", "lvl": "public", "cat": "public"},
    {"q": "远东智造 2025 年内部营收目标是多少？", "stem": "int_远东智造_经营计划", "lvl": "internal", "cat": "internal"},
    {"q": "远东智造 拟收购的标的公司和交易对价是多少？", "stem": "res_远东智造_并购预案", "lvl": "restricted", "cat": "restricted"},
    {"q": "康元生物 2024 年营业收入和毛利率是多少？", "stem": "pub_康元生物_年报", "lvl": "public", "cat": "public"},
    {"q": "康元生物 2025 年内部研发投入预算是多少？", "stem": "int_康元生物_经营计划", "lvl": "internal", "cat": "internal"},
    {"q": "康元生物 拟收购的标的公司和交易对价是多少？", "stem": "res_康元生物_并购预案", "lvl": "restricted", "cat": "restricted"},
]

IRRELEVANT = [
    "请告诉我 2025 年火星基地的运营预算。",
    "特斯拉 2024 年第四季度全球交付量是多少？",
    "英伟达 H100 芯片采用多少纳米制程工艺？",
    "比亚迪 2024 年新能源汽车销量是多少？",
]

LEVELS = ["public", "internal", "restricted"]


def main() -> int:
    print("=== 知擎 RAG 框架 · 场景 B 权限审计指标（合规 Guard）===\n")
    cfg = load_config(str(CONFIG))
    kb = KnowledgeBase(cfg, backend="memory")

    rep = kb.ingest(_files(CORPUS_DIR))
    n = rep["ingested_chunks"]
    q = rep.get("quality", {})
    print(f"[入库] 文件数={len(_files(CORPUS_DIR))}  chunks={n}  "
          f"质检通过率={q.get('accept_rate')}  拒绝={q.get('rejected')}")
    # 打印各 chunk 密级分布，验证 acl_map 生效
    dist = {}
    for c in kb.retriever.chunks:
        lv = c.meta.get("acl_level", "public")
        dist[lv] = dist.get(lv, 0) + 1
    print(f"[密级分布] {dist}\n")
    if n == 0:
        print("入库为空，无法评测")
        return 1

    # ---------- 跑矩阵：每条查询 × 每个身份级别 ----------
    ac1_ok = ac2_ok = ac5_ok = 0
    ac1_n = ac2_n = ac5_n = 0
    for item in QUERIES:
        qtext, target_stem, target_lvl, cat = item["q"], item["stem"], item["lvl"], item["cat"]
        for lv in LEVELS:
            ans, ctx = kb.ask(qtext, user_level=lv, user_id=f"user_{lv}")
            # AC-2 最小权限：返回 chunk 密级全部 <= 用户级别
            ac2_n += 1
            if all(can_access(lv, c.meta.get("acl_level", "public")) for c in ctx):
                ac2_ok += 1
            # AC-1 越权拦截：受限类问题，普通员工视角不得出现 restricted chunk
            if cat == "restricted" and lv == "public":
                ac1_n += 1
                if not any(c.meta.get("acl_level") == "restricted" for c in ctx):
                    ac1_ok += 1
        # AC-5 管理员可用性：restricted 视角下，目标文档应进 top5
        admin_scored = kb.retriever.search(qtext, user_level="restricted", return_scores=True)
        top5_stems = [_stem(c) for c, _ in admin_scored[:5]]
        ac5_n += 1
        if target_stem in top5_stems:
            ac5_ok += 1

    # ---------- AC-7 无据拒答（LLM-independent，靠空召回/低分阈值） ----------
    refused_ok = 0
    for qtext in IRRELEVANT:
        ans, _ = kb.ask(qtext, user_level="public", user_id="user_public")
        if ans.refused:
            refused_ok += 1

    # ---------- AC-3 审计覆盖 + AC-4 越权可溯：读审计日志 ----------
    audit_lines = [
        json.loads(ln.rsplit("|H:", 1)[0])
        for ln in kb.audit.path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    total_asks = len(QUERIES) * len(LEVELS) + len(IRRELEVANT)
    ac3_ok = 1 if len(audit_lines) == total_asks else 0
    ac4_ok = 0
    ac4_n = 0
    for item in QUERIES:
        if item["cat"] != "restricted":
            continue
        # 该受限问题、以 public 身份发起的那条审计
        for rec in audit_lines:
            if rec.get("question") == item["q"] and rec.get("user_level") == "public":
                ac4_n += 1
                if rec.get("denied") is True:
                    ac4_ok += 1
                break

    # ---------- AC-6 acl_selfcheck 自检 ----------
    sc = kb.acl_selfcheck("云启科技 拟收购的标的公司和交易对价是多少？")
    sc_restricted = sc.get("restricted", [])
    sc_public = sc.get("public", [])
    ac6_ok = 1 if (len(sc_restricted) > 0 and "restricted" not in sc_public) else 0

    print("=== 验收结果（场景 B · 真实 BGE-M3 嵌入）===")
    print(f"  AC-1 越权拦截（受限问题·员工视角 0 受限命中） : {ac1_ok}/{ac1_n}")
    print(f"  AC-2 最小权限（返回密级全部 <= 用户级别）    : {ac2_ok}/{ac2_n}")
    print(f"  AC-3 审计覆盖（审计条数 == 查询数）          : {ac3_ok}  ({len(audit_lines)}/{total_asks})")
    print(f"  AC-4 越权可溯（被拦查询审计标记 denied）      : {ac4_ok}/{ac4_n}")
    print(f"  AC-5 管理员可用性（目标文档进 top5）         : {ac5_ok}/{ac5_n}")
    print(f"  AC-6 acl_selfcheck（受限可见/公开不可见）     : {ac6_ok}/1")
    print(f"  AC-7 无据拒答（无关问题正确拒答）             : {refused_ok}/{len(IRRELEVANT)}")
    print()

    # ---------- 双身份演示（核心卖点） ----------
    print("=== 双身份演示：同一受限问题，不同可见范围 ===")
    demo_q = "云启科技 拟收购的标的公司和交易对价是多少？"
    for who, lv in (("普通员工", "public"), ("合规管理员", "restricted")):
        ans, ctx = kb.ask(demo_q, user_level=lv, user_id=f"demo_{lv}")
        levels = sorted({c.meta.get("acl_level", "public") for c in ctx})
        snippet = ans.text[:80].replace("\n", " ")
        print(f"  [{who} | {lv}] 可见密级={levels}")
        print(f"          回答={snippet}")
    print()

    # ---------- 可选：真实 GLM 管理员视角答案（best-effort） ----------
    real_answer = None
    try:
        ans, ctx = kb.ask(demo_q, user_level="restricted", user_id="admin_real")
        real_answer = ans.text[:200]
        print(f"[可选·真实GLM] 管理员视角答案：{real_answer}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[可选·真实GLM] 调用失败（不影响核心指标）：{exc}\n")

    summary = {
        "sample": "B 合规Guard",
        "corpus": {"md_files": len(_files(CORPUS_DIR)), "chunks": n,
                   "acl_distribution": dist, "quality_accept_rate": q.get("accept_rate")},
        "acceptance": {
            "AC1_越权拦截": f"{ac1_ok}/{ac1_n}",
            "AC2_最小权限": f"{ac2_ok}/{ac2_n}",
            "AC3_审计覆盖": f"{ac3_ok} ({len(audit_lines)}/{total_asks})",
            "AC4_越权可溯": f"{ac4_ok}/{ac4_n}",
            "AC5_管理员可用性": f"{ac5_ok}/{ac5_n}",
            "AC6_自检": f"{ac6_ok}/1",
            "AC7_无据拒答": f"{refused_ok}/{len(IRRELEVANT)}",
        },
        "acl_selfcheck": sc,
        "optional_real_glm_answer": real_answer,
    }
    METRICS_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已写入: {METRICS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
