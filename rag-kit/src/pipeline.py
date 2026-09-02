"""编排层：把一个租户的配置收敛成一套可运行的知识库。

一个 KnowledgeBase 实例 = 解析 + 切片 + 质检 + 向量化 + 检索 + 生成 + 审计。
新增客户只改 config/tenants/*.yaml，不改代码——这是边际交付时间从 60h 压到 20h 的前提。

场景映射：
- 场景 A（异构解析）：默认配置即可
- 场景 B（权限审计）：security.acl_enabled=true，ingest 时给 chunk 打 acl_level
- 场景 C（离线私有化）：backend=memory 或 qdrant，配合 docker-compose 一键起
- 场景 D（分层调度）：接 Router，按路由结果切不同 llm.model
"""

from __future__ import annotations

from pathlib import Path

from chunking import build_chunks
from config import TenantConfig
from embeddings import Embedder
from generator import Generator
from parsers import parse_file
from permissions import AuditLogger, can_access
from quality import validate
from retriever import Retriever


class KnowledgeBase:
    def __init__(self, cfg: TenantConfig, backend: str = "memory"):
        self.cfg = cfg
        self.backend = backend
        self.embedder = Embedder(cfg.retrieval)
        self.retriever = Retriever(
            cfg.retrieval, self.embedder, cfg.security, backend=backend,
            qdrant_url=cfg.qdrant_url, collection=cfg.collection,
        )
        self.generator = Generator(cfg.llm, cfg.guard)
        self.audit = (
            AuditLogger(cfg.audit_path) if cfg.security.audit_enabled else None
        )

    # ---------- 入库 ----------
    def ingest(self, files: list) -> dict:
        total = 0
        report = None
        for fp in files:
            p = Path(fp)
            if not p.exists():
                continue
            doc = parse_file(p, self.cfg.parser)
            if not doc.parse_ok:
                continue
            chunks = build_chunks(doc, self.cfg.chunk)
            qr = validate(chunks)
            report = qr
            acl_map = self.cfg.security.acl_map or {}
            for c in qr.accepted:
                # 场景 B：入库即按「配置映射」给 chunk 打密级（预留 IAM 对接点）。
                # 优先级：chunk 自带 > acl_map(按 source stem) > 租户默认级。
                src = c.meta.get("source", "")
                stem = Path(src).stem if src else ""
                level = (
                    c.meta.get("acl_level")
                    or (acl_map.get(stem) if stem else None)
                    or (acl_map.get(src) if src else None)
                    or self.cfg.security.default_level
                )
                c.meta["acl_level"] = level
            self.retriever.add(qr.accepted)
            total += len(qr.accepted)
        return {
            "ingested_chunks": total,
            "quality": report.summary() if report else {},
        }

    # ---------- 问答 ----------
    def ask(
        self, question: str, user_level: str | None = None, user_id: str = "anonymous"
    ):
        user_level = user_level or self.cfg.security.default_level
        scored = self.retriever.search(question, user_level=user_level, return_scores=True)
        contexts = [c for c, _ in scored]
        best = max((s for _, s in scored), default=0.0)
        hit_levels = sorted({c.meta.get("acl_level", "public") for c in contexts})
        denied = False
        if self.cfg.security.acl_enabled and user_level != "restricted":
            # 以最高权限视角再检索一次，判断本用户是否被拦截了部分内容
            priv = self.retriever.search(question, user_level="restricted", return_scores=True)
            denied = len(priv) > len(scored)

        # 无召回，或最高检索分低于阈值 -> 拒答（场景 A/B 的「无据拒答」铁律）
        # fake 向量下分数量纲不可比，仅按「空召回」拒答，避免误杀；
        # 真实嵌入（BGE-M3 等）下启用分数阈值，确保资料无依据时拒答。
        fake_mode = self.embedder.mode == "fake"
        score_refuse = (
            self.cfg.guard.refuse_threshold
            and not fake_mode
            and best < self.cfg.guard.refuse_threshold
        )
        if not contexts or score_refuse:
            ans = self.generator.generate(question, [], refuse_if_empty=True)
        else:
            ans = self.generator.generate(question, contexts)

        if self.audit:
            self.audit.log(
                {
                    "user_id": user_id,
                    "user_level": user_level,
                    "question": question,
                    "retrieved": [c.chunk_id for c in contexts],
                    "hit_levels": hit_levels,
                    "denied": denied,
                    "refused": ans.refused,
                    "answer_len": len(ans.text),
                }
            )
        return ans, contexts

    # ---------- 从生产后端恢复（场景 C 持久化） ----------
    def load(self) -> "KnowledgeBase":
        """从 Qdrant 回灌已入库的向量与原文，无需重新入库即可检索。

        用于服务重启/横向扩容场景：向量库是单一事实源。
        """
        self.retriever.load_from_qdrant()
        return self

    # ---------- 权限自检（演示用） ----------
    def acl_selfcheck(self, question: str, levels=("public", "internal", "restricted")) -> dict:
        out = {}
        for lv in levels:
            ctx = self.retriever.search(question, user_level=lv)
            out[lv] = [
                c.meta.get("acl_level", "public") for c in ctx
            ]
        return out
