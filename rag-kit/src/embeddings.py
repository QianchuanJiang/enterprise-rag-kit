"""向量化层。

默认 BGE-M3（中英混排 SOTA，离线可用），支持四种运行模式：
- ollama : 本地 Ollama 起 BGE-M3（OpenAI 兼容接口），**开发期默认**，零成本、数据不出本机
- local  : sentence-transformers 本地加载（HuggingFace 权重），断网可用（生产默认备选）
- cloud  : OpenAI-compatible embedding 接口（智谱 embedding-3 / 硅基流动 bge-m3 等云端免费档）
- fake   : 哈希确定性向量（仅用于测试与无模型演示，**不可用于生产**）

设计要点：Embedder 对外只暴露 embed_texts / embed_query，上层不感知后端。
这让您可以在「本地 BGE-M3」「云端 embedding」之间零改代码切换（仅改环境变量）。
"""

from __future__ import annotations

import hashlib
import math
import os

from config import RetrievalConfig


def _embed_requests(base_url: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    """OpenAI 兼容 /v1/embeddings 的 requests 直连（openai SDK 缺失时回退）。"""
    import requests

    url = base_url.rstrip("/") + "/embeddings"
    resp = requests.post(
        url,
        json={"model": model, "input": texts},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


class Embedder:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self._model = None
        # RAG_EMBEDDER 环境变量可强制覆盖：local | cloud | fake
        self.mode = os.getenv("RAG_EMBEDDER", "auto")
        if self.mode == "auto":
            # 默认走本地 Ollama BGE-M3（零成本、数据不出本机）；无 Ollama 时改 RAG_EMBEDDER=cloud/fake
            self.mode = "fake" if cfg.dense_model == "fake" else "ollama"

    # ---------- 对外接口 ----------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "fake":
            return [self._fake(t) for t in texts]
        if self.mode == "ollama":
            return self._ollama(texts)
        if self.mode == "cloud":
            return self._cloud(texts)
        self._ensure_local()
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    # ---------- 本地模型 ----------
    def _ensure_local(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.dense_model)

    # ---------- 云端接口 ----------
    def _cloud(self, texts: list[str]) -> list[list[float]]:
        base_url = os.getenv("EMBED_BASE_URL", "")
        api_key = os.getenv("EMBED_API_KEY", "sk-no-key")
        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        try:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except ImportError:
            return _embed_requests(base_url, api_key, model, texts)

    # ---------- 本地 Ollama（OpenAI 兼容 /v1/embeddings） ----------
    def _ollama(self, texts: list[str]) -> list[list[float]]:
        base_url = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("EMBED_API_KEY", "ollama")
        model = os.getenv("EMBED_MODEL", "bge-m3")
        try:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except ImportError:
            return _embed_requests(base_url, api_key, model, texts)

    # ---------- 哈希向量（测试/演示用） ----------
    def _fake(self, text: str) -> list[float]:
        """字符 3-gram 哈希到定长向量，保证相似文本有相似向量。

        仅为让脚手架在无 GPU / 无模型下载时也能端到端跑通；
        相似度是粗近似，绝对不能用于生产评测。
        """
        dim = 256
        vec = [0.0] * dim
        for i in range(0, max(1, len(text) - 2)):
            gram = text[i : i + 3]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0 if (h >> 1) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
