# enterprise-rag-kit

> 面向信创与金融合规场景的**企业级本地化 RAG 引擎**，数据不出域。

`enterprise-rag-kit` 是一套可直接落地的检索增强生成（RAG）基础框架，重点解决生产环境最棘手的三个工程问题：**真实文档解析**、**权限合规**与**生产级向量存储**。配套一个可观测的 Web 控制台，把"黑盒问答"变成"可解释的检索工作台"。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 真实 PDF 解析链路 | 基于 PyMuPDF 抽取年报 / 研报，文本干净、页码可溯源 |
| 权限合规（ACL） | 多级访问控制，受限内容对低权限用户透明拦截 |
| 生产级向量后端 | Qdrant 持久化存储，进程重启免重新嵌入入库 |
| 可观测控制台 | 检索透视（耗时/阈值/召回）、向量空间投影、溯源闭环、自动化评测台 |

## 真实评测指标（8 家跨行业年报语料 · BGE-M3 嵌入 + GLM 生成）

| 指标 | 内存后端 | Qdrant 生产后端 |
|------|---------|----------------|
| Recall@1 / @3 / @5 / @8 | 75% / 90% / **100%** / 100% | 75% / 90% / **100%** / 100% |
| 无据拒答 | 4 / 4 | 4 / 4 |
| 生成非空 / 带溯源 / 接地（grounding） | 6 / 6 / 6 | 6 / 6 / 6 |
| 拒答阈值 | 0.62（相关 0.66–0.82 / 无关 0.37–0.58，干净分离） | 0.62 |

> 指标均由端到端脚本真实跑出、`metrics_c.json` 落盘，可复现。

## 架构总览

```
                ┌─────────────── 知识入库 ───────────────┐
  真实 PDF ──▶  PyMuPDF 抽取 ──▶ 分块 ──▶ BGE-M3 嵌入 ──┐
                                                      │
                                                      ▼
                      内存向量库  ◀──┐        Qdrant（生产，持久化）
                                      │
   用户提问 ──▶ 嵌入 ──▶ 检索（含权限过滤）──▶ 阈值判定 ──┬── 低于阈值：拒答
                                                      └── 高于阈值：GLM 生成（带 [n] 溯源）
                                                                  │
                                                                  ▼
                                             Web 控制台：流式输出 / 检索透视 / PDF 溯源
```

详见 [docs/architecture.md](docs/architecture.md)。

## 三个落地场景

- **场景一 · 真实 PDF 解析**：端到端 PDF → 抽取 → 检索链路，验证解析质量对召回的影响。见 [docs/case-pdf-parsing.md](docs/case-pdf-parsing.md)
- **场景二 · 权限合规**：员工 / 管理员双身份，受限内容（如未公开交易对价）对低权限用户透明拦截。见 [docs/case-access-control.md](docs/case-access-control.md)
- **场景三 · Qdrant 生产后端**：向量库从内存切换为 Qdrant，验证持久化与重启免重入库。见 [docs/case-qdrant-backend.md](docs/case-qdrant-backend.md)

## 快速开始

```bash
# 1. 启动 Qdrant（需 Docker）
docker run -d -p 6333:6333 --name qdrant -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 2. 入库（8 份真实 PDF）
python scripts/real_pdf_eval.py        # 内存后端
python scripts/real_c_eval.py          # Qdrant 生产后端

# 3. 启动可观测控制台
PORT=8080 python web_demo.py           # 打开 http://localhost:8080
```

完整依赖与算力说明见 [docs/quickstart.md](docs/quickstart.md) 与 [docs/tech-stack.md](docs/tech-stack.md)。

## 目录结构

```
enterprise-rag-kit/
├── README.md
├── .gitignore
├── rag-kit/                 # 框架代码
│   ├── src/                 # 检索 / 生成 / 权限 / 管道 / 路由
│   ├── scripts/             # 各场景评测与入库脚本
│   ├── configs/             # 租户与场景配置（tenant_a/b/c）
│   ├── web_demo.py          # 可观测 Web 控制台（零依赖）
│   └── docker-compose.yml   # Qdrant 一键起
└── docs/                    # 设计 / 场景 / 评测文档
```

## 评测

```bash
python scripts/real_c_eval.py     # 输出 Recall / 拒答率 / 生成质量 / 持久化证明
```

自动化评测台亦可在 Web 控制台内一键触发（[docs/web-console.md](docs/web-console.md)）。

## 技术栈

- 嵌入：Ollama 本地 `bge-m3`（1024 维）
- 生成：智谱 GLM-4.6V-Flash（OpenAI 兼容）
- 向量库：内存 / Qdrant 1.19
- 解析：PyMuPDF
- 控制台：Python 标准库 `http.server`（零额外依赖）

详见 [docs/tech-stack.md](docs/tech-stack.md)。
