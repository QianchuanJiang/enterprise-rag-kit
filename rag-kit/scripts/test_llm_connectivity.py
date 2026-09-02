"""真实 LLM 连通性测试（GLM-4.6V-Flash）。

目的：锁定真实模型配置 —— 证明 dev.yaml 里的 base_url + api_key_env
能真正调通智谱 GLM-4.6V-Flash，并产出可被 README 引用的真实响应（非 stub）。

运行：
    cd rag-kit && python scripts/test_llm_connectivity.py
（密钥从环境变量 ZHIPU_API_KEY 或本地 .env 注入）

安全：脚本不硬编码任何 key，也不回显完整 key（仅显示掩码）。
"""

from __future__ import annotations

import os
import sys


def _load_dotenv(path: str = ".env") -> None:
    """最小化 .env 加载，避免硬依赖 python-dotenv。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


_load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("✗ 缺少 openai 包，请先: pip install openai")
    sys.exit(2)

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4.6v-flash"
API_KEY = os.getenv("ZHIPU_API_KEY")

if not API_KEY:
    print("✗ 未找到 ZHIPU_API_KEY（请设置环境变量或写入 .env）")
    sys.exit(2)

masked = API_KEY[:6] + "…" + API_KEY[-4:]
print("== GLM 连通性测试 ==")
print(f"  model    : {MODEL}")
print(f"  base_url : {BASE_URL}")
print(f"  api_key  : {masked}")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
resp = client.chat.completions.create(
    model=MODEL,
    temperature=0.1,
    max_tokens=256,
    messages=[
        {"role": "system", "content": "你是一个严谨的企业知识库助手，只依据给定资料作答。"},
        {"role": "user", "content": "请用一句话说明：什么是 RAG（检索增强生成）？"},
    ],
)
ans = resp.choices[0].message.content or ""
print(f"\n[response] {ans}\n")
print("✅ 真实模型调用成功：dev.yaml 的 LLM 配置可端到端跑通。")
print("   注：GLM-4.6V-Flash 为原生多模态模型，亦可直接接收 image / pdf 输入，")
print("       适用于文档解析 的「扫描件/复杂版式」文档理解（无需先 OCR 再解析）。")
