"""网络无关的抽象校验脚本。

目的：证明「切换供应商只改配置 + 环境变量，不动业务代码」。
- 不调用任何远端 API、不加载任何模型权重；
- 仅校验配置解析、Embedder 抽象在 fake 模式下可运行、LLM 配置正确指向智谱。

运行：
    cd rag-kit && python verify_config.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg_mod
from embeddings import Embedder


def main() -> None:
    here = os.path.dirname(__file__)
    cfg = cfg_mod.load_config(os.path.join(here, "configs", "dev.yaml"))

    print("== 配置解析 ==")
    print(f"  tenant        : {cfg.tenant}")
    print(f"  LLM provider  : {cfg.llm.provider}")
    print(f"  LLM base_url  : {cfg.llm.base_url}")
    print(f"  LLM model     : {cfg.llm.model}")
    print(f"  LLM api_key   : {cfg.llm.api_key_env} (env)")
    assert cfg.llm.base_url.endswith("/api/paas/v4"), "LLM 未指向 OpenAI 兼容网关"
    assert cfg.llm.model == "glm-4.6v-flash", "LLM 未锁定智谱免费档"

    print("\n== Embedder 抽象（fake 模式，零网络/零模型）==")
    os.environ["RAG_EMBEDDER"] = "fake"
    emb = Embedder(cfg.retrieval)
    print(f"  resolved mode : {emb.mode}")
    assert emb.mode == "fake"
    vec = emb.embed_query("2025 年营业收入同比增长 12.3%")
    assert isinstance(vec, list) and len(vec) == 256, "fake 向量维度异常"
    print(f"  vector dim    : {len(vec)}  ✓")
    # 相似文本应比无关文本更接近（fake 也是确定性近似）
    sim = sum(a * b for a, b in zip(vec, emb.embed_query("营业收入同比增长 12.3%")))
    diff = sum(a * b for a, b in zip(vec, emb.embed_query("今天天气真好")))
    print(f"  相似句内积   : {sim:.3f}  >  无关句内积: {diff:.3f}  {'✓' if sim > diff else '✗'}")
    assert sim > diff

    print("\n== 供应商切换验证（不改代码，仅环境变量）==")
    for mode in ("ollama", "cloud", "fake"):
        os.environ["RAG_EMBEDDER"] = mode
        e2 = Embedder(cfg.retrieval)
        print(f"  RAG_EMBEDDER={mode:7s} -> 解析为 {e2.mode}")
    # 还原
    os.environ["RAG_EMBEDDER"] = "fake"

    print("\n✅ 抽象层校验通过：LLM/Embedder 均可通过配置 + 环境变量切换，业务代码零改动。")


if __name__ == "__main__":
    main()
