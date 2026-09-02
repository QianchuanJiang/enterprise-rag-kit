"""检索层：混合检索 + 重排。

纯向量召回率不够——型号、代号、条款号、人名靠字面匹配，语义相似度很难命中。
因此必须加一条关键词路（BM25）+ 一段重排。

融合策略：
- 稠密：Qdrant 或内存向量（余弦）
- 稀疏：BM25（对字面 token 友好）
- 融合：RRF（Reciprocal Rank Fusion），不依赖分数量纲
- 重排：bge-reranker-v2-m3 做精排，输出 Top-K 最终片段

后端：backend="memory" 走 numpy 风格内存实现（零依赖，可测试/演示）；
      backend="qdrant" 走 Qdrant（生产，支持 payload 过滤与扩展）。
"""

from __future__ import annotations

import math
from collections import defaultdict

from chunking import Chunk
from config import RetrievalConfig, SecurityConfig
from embeddings import Embedder
from permissions import can_access, LEVEL_RANK


class BM25:
    """轻量 BM25，纯 Python 实现，无外部依赖。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl = 0.0

    @staticmethod
    def _tok(text: str) -> list[str]:
        # 中文无空格，按字粒度切分，否则「营业收入/型号/条款号」这类字面匹配完全失效。
        # 阿拉伯数字与字母保留为整体 token（便于精确命中「2024」「Q3」等）。
        toks: list[str] = []
        buf = ""
        for ch in text.lower():
            if "一" <= ch <= "鿿":  # CJK 统一表意文字
                toks.append(ch)
            elif ch.isalnum():
                buf += ch
            else:
                if buf:
                    toks.append(buf)
                    buf = ""
        if buf:
            toks.append(buf)
        return toks

    def fit(self, corpus: list[str]) -> None:
        self.docs = [self._tok(c) for c in corpus]
        n = len(self.docs)
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()
        }
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / n if n else 0.0

    def score(self, query: str) -> list[float]:
        q = self._tok(query)
        scores = [0.0] * len(self.docs)
        for t in q:
            if t not in self.idf:
                continue
            w = self.idf[t]
            for i, d in enumerate(self.docs):
                f = d.count(t)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += w * f * (self.k1 + 1) / denom
        return scores


class Retriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        embedder: Embedder,
        security: SecurityConfig | None = None,
        backend: str = "memory",
        qdrant_url: str = "http://localhost:6333",
        collection: str = "kb_default",
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.security = security
        self.backend = backend
        self.qdrant_url = qdrant_url
        self.collection = collection

        self.chunks: list[Chunk] = []
        self._vecs: list[list[float]] = []
        self.bm25 = BM25()
        self._qdrant = None

    # ---------- 入库 ----------
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        # BM25 在全量 chunk 上重拟合（规模小时可接受）
        self.bm25.fit([c.text for c in self.chunks])
        if self.backend == "memory":
            self._vecs.extend(self.embedder.embed_texts([c.text for c in chunks]))
        elif self.backend == "qdrant":
            self._upsert_qdrant(chunks)

    # ---------- 检索 ----------
    def search(
        self, query: str, user_level: str = "public", return_scores: bool = False
    ) -> list[Chunk] | list[tuple[Chunk, float]]:
        top_k = self.cfg.top_k_coarse

        # 1) 稠密
        qvec = self.embedder.embed_query(query)
        if self.backend == "memory":
            dense_scores = [self._cosine(qvec, v) for v in self._vecs]
        else:
            dense_scores = self._qdrant_scores(qvec, top_k * 2, user_level)

        # 2) 稀疏
        sparse_scores = self.bm25.score(query) if self.cfg.sparse else [0.0] * len(self.chunks)

        # 3) RRF 融合
        dense_ranked = self._rank(dense_scores)
        sparse_ranked = self._rank(sparse_scores)
        fused = self._rrf(dense_ranked)
        for i, idx in enumerate(sparse_ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (i + 60)

        ranked = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

        # 4) ACL 硬过滤（权限合规 核心优势：检索层就拦截，不是生成后过滤）
        if self.security and self.security.acl_enabled:
            ranked = [
                i
                for i in ranked
                if can_access(user_level, self.chunks[i].meta.get("acl_level", "public"))
            ]

        coarse = ranked[:top_k]
        if self.cfg.rerank_enabled and coarse:
            coarse = self._rerank(query, coarse)
        final = coarse[: self.cfg.top_k_final]
        result_chunks = [self.chunks[i] for i in final]
        if return_scores:
            # 对最终召回片段重算真实余弦分（与后端无关），供生成层依据
            # guard.refuse_threshold 做「无据拒答」。避免稠密路仅返回 TopK 时
            # 因 BM25 贡献进入 final 的片段得分为 0 而误拒。
            final_vecs = self.embedder.embed_texts([self.chunks[i].text for i in final])
            return list(
                zip(result_chunks, [self._cosine(qvec, fv) for fv in final_vecs])
            )
        return result_chunks

    # ---------- 工具 ----------
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    @staticmethod
    def _minmax(scores: list[float]) -> list[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    @staticmethod
    def _rank(scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    @staticmethod
    def _rrf(ranked: list[int], k: int = 60) -> dict[int, float]:
        return {idx: 1.0 / (rank + k) for rank, idx in enumerate(ranked)}

    def _rerank(self, query: str, indices: list[int]) -> list[int]:
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            return indices  # 重排模型缺失时退化为 RRF 结果，不阻断
        model = CrossEncoder(self.cfg.rerank_model)
        pairs = [(query, self.chunks[i].text) for i in indices]
        scores = model.predict(pairs)
        return [i for _, i in sorted(zip(scores, indices), reverse=True)]

    # ---------- Qdrant 后端（生产，支持 payload 权限过滤与扩展） ----------
    def _upsert_qdrant(self, chunks: list[Chunk]) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)
        vecs = self.embedder.embed_texts([c.text for c in chunks])
        # chunk 在 self.chunks 中的起始索引即其 Qdrant point id（稳定、可对齐）
        base = len(self.chunks) - len(chunks)
        payloads = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "page": c.page,
                "acl_level": c.meta.get("acl_level", "public"),
                "source": c.meta.get("source", ""),
            }
            for c in chunks
        ]
        if not self._qdrant.collection_exists(self.collection):
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=len(vecs[0]), distance=Distance.COSINE
                ),
            )
        self._qdrant.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(id=base + j, vector=v, payload=p)
                for j, (v, p) in enumerate(zip(vecs, payloads))
            ],
        )

    def _qdrant_scores(
        self, qvec: list[float], top_k: int, user_level: str = "public"
    ) -> list[float]:
        """走 Qdrant 做稠密向量检索，返回与 self.chunks 对齐的全量分数。

        仅返回 TopK 片段的余弦分（其余为 0）供 RRF 融合；
        权限过滤优先下沉到 Qdrant payload 层（受限片段根本不返回），
        与检索层 can_access 硬拦截形成双保险。
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)
        # 仅当 ACL 开启时才把权限下沉为 Qdrant payload 过滤；
        # 否则不过滤（与内存后端行为一致，保证Qdrant 生产后端 无 ACL 场景检索完整）。
        acl_on = bool(self.security and self.security.acl_enabled)
        qfilter = None
        if acl_on:
            allowed = [
                lvl
                for lvl, r in LEVEL_RANK.items()
                if r <= LEVEL_RANK.get(user_level, 0)
            ]
            qfilter = Filter(
                must=[FieldCondition(key="acl_level", match=MatchAny(any=allowed))]
            )
        try:
            hits = self._qdrant.query_points(
                collection_name=self.collection,
                query=qvec,
                limit=top_k,
                query_filter=qfilter,
                with_payload=False,
                with_vectors=False,
            ).points
        except Exception:
            # 字段/版本不兼容时退化为不过滤，由检索层 can_access 兜底
            hits = self._qdrant.query_points(
                collection_name=self.collection,
                query=qvec,
                limit=top_k,
                with_payload=False,
                with_vectors=False,
            ).points
        scores = [0.0] * len(self.chunks)
        for h in hits:
            idx = h.id
            if isinstance(idx, int) and 0 <= idx < len(self.chunks):
                scores[idx] = h.score
        return scores

    def load_from_qdrant(self) -> None:
        """从 Qdrant 回灌 chunk 文本与载荷到内存（BM25 + 索引）。

        Qdrant 生产后端（信创/离线私有化）的核心价值：服务重启后，向量与原文仍在
        Qdrant 中，无需重新解析/切片/入库即可检索。本方法在启动时把
        payload 重建为 Chunk，使检索层（稠密+BM25+ACL）立即可用。
        """
        from qdrant_client import QdrantClient
        from chunking import Chunk

        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)
        if not self._qdrant.collection_exists(self.collection):
            return
        self.chunks = []
        offset = None
        while True:
            pts, offset = self._qdrant.scroll(
                collection_name=self.collection,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in pts:
                pl = p.payload or {}
                self.chunks.append(
                    Chunk(
                        chunk_id=pl.get("chunk_id", str(p.id)),
                        doc_id=pl.get("doc_id", ""),
                        text=pl.get("text", ""),
                        page=int(pl.get("page", 0) or 0),
                        char_len=len(pl.get("text", "")),
                        meta={
                            "acl_level": pl.get("acl_level", "public"),
                            "source": pl.get("source", ""),
                        },
                    )
                )
            if offset is None:
                break
        self.bm25.fit([c.text for c in self.chunks])
