# 脚本说明

## 1. 端到端冒烟测试（零网络 / 零模型）

```bash
cd rag-kit
python scripts/smoke_test.py
```

- 语料：`data/smoke/*.md`（合成样例，仅验证管线，非真实年报）
- 配置：`configs/smoke.yaml`（fake 向量 + stub 生成 + memory 后端）
- 验证项：入库 → 质检 → 检索 → 生成 → 溯源 → 无据拒答 → 审计链
- 退出码：全过 0 / 有失败 1（可直接接 CI）
- 注意：fake 向量下**相关性是近似的**（标题与事实的语义区分需真实 BGE-M3 + 重排）。
  本测试只保证「链路可跑、行为正确」，不评估检索精度。

## 2. 取真实年报语料（需联网）

```bash
pip install requests
python scripts/fetch_cninfo.py                 # 默认大盘股 2024 年报
python scripts/fetch_cninfo.py --codes 000001 600519 --year 2024 --out data/cninfo
```

- 来源：巨潮网(cninfo) 公开 API，无需登录/付费/验证码
- 下载的 PDF 放入 `data/sample/` 后，用真实链路跑（见下）

## 3. 切到真实链路（智谱 GLM-4.7-Flash + 本地 BGE-M3）

开发期真实模型配置在 `configs/dev.yaml`：

| 组件 | 配置 | 前置 |
|---|---|---|
| LLM | `provider=openai` + 智谱 `glm-4.7-flash` | 设环境变量 `ZHIPU_API_KEY` |
| Embedding | 本地 BGE-M3（`ollama pull bge-m3`） | 本地起 Ollama；或 `RAG_EMBEDDER=cloud` 走云端 |
| 解析 | `engine=docling` | `pip install docling` |

```bash
export ZHIPU_API_KEY=你的key
ollama pull bge-m3
python cli.py run --config configs/dev.yaml --dir data/sample \
      --question "2024 年营业收入是多少？"
```

> 真实指标（召回率/ Faithfulness 等）只在接真实模型后跑一次、如实记录，写进 README。
> 数据用公开年报 + 合成密级标注即可，**绝不能编造指标**。
