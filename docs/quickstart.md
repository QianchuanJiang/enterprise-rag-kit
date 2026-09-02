# 快速开始

从零跑通「入库 → 检索 → 生成 → 评测」全链路，约 10 分钟。

---

## 1. 环境要求

| 依赖 | 版本 | 必需 | 说明 |
|---|---|---|---|
| Python | 3.11+ | ✅ | 推荐 conda / venv 隔离环境 |
| Docker | 最新稳定版 | 生产后端需要 | 运行 Qdrant；仅用内存后端可跳过 |
| Ollama | 最新稳定版 | ✅ | 本地嵌入模型 `bge-m3`（1024 维） |
| LLM API Key | — | 可选 | 用于生成答案；无 key 时可用 `stub` 模式跑通检索链路 |

## 2. 安装依赖

```bash
cd rag-kit
pip install -r requirements.txt
```

核心依赖：`pymupdf`（PDF 解析）、`qdrant-client`（生产向量库）、`requests`、`numpy`、`PyYAML`。

## 3. 准备语料

仓库不内置 PDF（体积大且可复现生成），用脚本合成 8 份跨行业年报：

```bash
python scripts/make_real_pdfs.py     # 产出 data/reports/raw/*.pdf
```

> 语料为合成数据，但 PDF 文件格式、解析链路与真实年报完全一致。

## 4. 启动 Qdrant（生产后端，可选）

```bash
docker run -d -p 6333:6333 --name qdrant \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant

curl http://localhost:6333/          # 返回版本信息即就绪
```

`-v` 挂载卷是**必须的**：否则容器删除后向量数据会丢失。

## 5. 运行评测

```bash
# 内存后端（文档解析链路）
python scripts/real_pdf_eval.py

# Qdrant 生产后端（含持久化与重启恢复验证）
python scripts/real_c_eval.py

# 权限合规（检索层 ACL 拦截）
python scripts/real_b_eval.py
```

指标落盘到 `data/reports/metrics_c.json`，包含 Recall、拒答阈值、生成质量等。

## 6. 启动可观测控制台

```bash
PORT=8080 python web_demo.py         # 浏览器打开 http://localhost:8080
```

首次启动逻辑：若 Qdrant 中已有数据则直接恢复（秒级，免重新嵌入）；否则从 PDF 入库。

## 7. 配置说明

配置文件位于 `configs/`：

| 配置 | 用途 |
|---|---|
| `tenant_a.yaml` | 文档解析（内存后端） |
| `tenant_b.yaml` | 权限合规（启用 ACL） |
| `tenant_c.yaml` | Qdrant 生产后端 |

关键参数：

```yaml
backend: memory          # memory | qdrant
security:
  acl_enabled: false     # 是否启用检索层权限过滤
refuse_threshold: 0.62   # 低于该相似度直接拒答，不编造
llm:
  model: glm-4.6v-flash  # 亦可用 stub 模式离线跑通检索
embedding:
  model: bge-m3          # Ollama 本地 1024 维
```

## 8. 常见问题

**Qdrant 连不上 / 返回 502**
若本机配置了 HTTP 代理，Python 的 `requests` 会把 `localhost` 请求也发给代理，代理对本地端口返回 502（`curl` 不受影响，所以直连测试可能"看起来正常"）。解决：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  PORT=8080 python web_demo.py
```

或仅为 `localhost,127.0.0.1` 设置 `NO_PROXY`。

**Qdrant 数据丢失**
容器被删除时数据会丢失。务必挂载数据卷；若已丢失，重新入库即可：`python scripts/real_c_eval.py`。

**生成阶段出现 429**
LLM 免费档限流。生成器内置指数退避重试，最终会成功，不影响指标。

**没有 API Key**
生成器支持 `stub` 模式：断网 / 无 key 时仍可完整跑通检索、溯源、权限拦截与审计链路，仅答案生成为占位。

## 9. 验证安装

```bash
curl http://localhost:8080/api/status
```

返回包含 `backend`、`chunks`、向量点数与拒答阈值，即表示全链路就绪。
