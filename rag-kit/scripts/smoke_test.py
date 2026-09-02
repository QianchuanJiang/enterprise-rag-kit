#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知擎 RAG 框架 · 端到端冒烟测试（零网络 / 零模型 / 零失败）。

目的：在没有任何 GPU、模型权重、API key 的情况下，验证整条链路
    解析 → 切片 → 质检 → 向量化(fake) → 检索(BM25+RRF) → 生成(stub) → 溯源/拒答
  真实可用，且行为符合预期。

运行：
    cd rag-kit
    python scripts/smoke_test.py

设计要点：
- 语料用 data/smoke/ 下的合成 Markdown（与真实年报版式无关，仅验证管线）
- 嵌入用 fake 哈希向量、生成用 stub 抽取式 —— 二者都零依赖
- 真实链路（智谱 GLM-4.7-Flash + 本地 BGE-M3）见 configs/dev.yaml，需 API key / Ollama
- 退出码：全部通过 0，存在失败 1（可直接接 CI）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402

CONFIG = ROOT / "configs" / "smoke.yaml"
SAMPLE_DIR = ROOT / "data" / "smoke"


def _files(d: Path) -> list[str]:
    return [str(p) for p in Path(d).rglob("*.*") if p.is_file()]


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    print("=== 知擎 RAG 框架 · 端到端冒烟测试 ===\n")
    ok = True

    cfg = load_config(str(CONFIG))
    kb = KnowledgeBase(cfg, backend="memory")

    # 1) 入库：解析 → 切片 → 质检
    rep = kb.ingest(_files(SAMPLE_DIR))
    n = rep["ingested_chunks"]
    ok &= check("入库产出 chunks > 0", n > 0, f"chunks={n}")
    ok &= check("质检报告存在", bool(rep.get("quality")))
    print(f"  ingest report: {rep}\n")

    # 2) 有依据问答：检索 → 生成 → 溯源
    # 注：fake 向量下相关性是近似的（标题/事实的语义区分需 BGE-M3+重排）。
    # 这里用一个在 fake 模式下也能干净命中的查询做展示（"净利润"只出现在事实片段）。
    ans, ctx = kb.ask("2024 年净利润是多少？")
    print(f"[ask] 净利润 -> {ans.text[:80]}...")
    print(f"        citations={ans.citations}\n")
    ok &= check("有依据问题返回非空答案", bool(ans.text), f"len={len(ans.text)}")
    ok &= check("答案带回溯引用", len(ans.citations) > 0, f"citations={ans.citations}")
    ok &= check("未误触发拒答", ans.refused is False)
    # stub 应命中含「净利润」的片段，而非标题/随机片段
    ok &= check(
        "溯源片段与问题相关(含'净利润')",
        any("净利润" in c.text for c in ctx),
    )

    # 3) 拒答（无依据）：用空 KB 触发「无召回即拒答」分支
    empty_kb = KnowledgeBase(cfg, backend="memory")  # 不入库
    ans_empty, _ = empty_kb.ask("请告诉我 2025 年火星基地的运营预算。")
    ok &= check("无依据问题触发拒答", ans_empty.refused is True, f"reason={ans_empty.reason}")
    ok &= check("拒答返回友好提示(非空)", bool(ans_empty.text))

    # 4) 生成层拒答逻辑（合规铁律的直接验证）
    direct = kb.generator.generate("任意问题", [], refuse_if_empty=True)
    ok &= check("generate([]) 直接拒答", direct.refused is True)

    # 5) 审计链
    if kb.audit:
        cnt = kb.audit.count()
        ok &= check("审计日志已写入", cnt >= 1, f"count={cnt}")
        ok &= check("审计哈希链可校验", kb.audit.verify() is True)
    else:
        ok &= check("审计日志已启用", False, "audit 为 None")

    print("\n" + ("✅ 冒烟全部通过" if ok else "❌ 存在失败项，请检查上方 FAIL 项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
