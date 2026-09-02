# Qdrant 生产后端 · 知擎·信创Edge（离线私有化）—— Qdrant 生产后端

## 1. 这个场景解决什么
把文档解析与权限合规 的「内存向量库」升级为**生产级向量数据库 Qdrant**：
- **持久化**：向量与原文落盘，进程退出 / 服务重启不丢（内存后端反之）。
- **横向扩展**：百万级向量、横向扩容，内存后端百级尚可、千级变慢。
- **payload 权限过滤**：`acl_level` 随向量入库，开启 ACL 时受限片段在检索层就被 Qdrant 过滤掉，根本不返回（与检索层 `can_access` 双保险）。
- 上游解析/切片/质检/嵌入、下游生成**完全不动**——这正是底座抽象的价值。

## 2. 与文档解析 的唯一差别
| 层 | 文档解析（内存） | Qdrant 生产后端（Qdrant） |
|----|---------------|------------------|
| 向量存储 | 进程内 Python 列表 | Qdrant 服务（localhost:6333） |
| 运行时切换 | `KnowledgeBase(cfg, backend="memory")` | `KnowledgeBase(cfg, backend="qdrant")` |
| 检索/嵌入/生成/拒答 | 完全一致 | 完全一致（同一套 BGE-M3 + BM25 + RRF + GLM） |

**结论：切后端不改变任何检索/生成指标，只改变向量的存放位置与生命周期。**

## 3. 如何起 Qdrant 服务（本环境实测可用）
> 注意：Docker Hub 直连被墙（curl 返回 000），但本机 Docker 配了国内镜像源
> （`docker.1ms.run` / `docker.xuanyuan.me` / `docker.m.daocloud.io`），经镜像源可正常拉取。

**方式一：一键 compose（推荐，带数据卷持久化）**
```bash
cd rag-kit
docker compose up -d        # 起 Qdrant，数据落在 docker 卷 qdrant_storage
docker compose down         # 停服务（卷保留，数据不丢）
```

**方式二：单条命令**
```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
curl -s http://localhost:6333/ | head -c 80   # 健康检查，返回 qdrant 版本即 OK
```

## 4. 代码改动清单（本次Qdrant 生产后端 落地）
`src/retriever.py`：
- 给 `Retriever.__init__` 增加显式 `collection` 参数（原先错用 `self.cfg.collection`，而 `RetrievalConfig` 无该字段，桩代码必崩）。
- `_upsert_qdrant`：用 `query_points`（qdrant-client 1.19 已废弃 `search`）+ `PointStruct`；按 chunk 在 `self.chunks` 中的索引作为稳定 point id；首次入库自动 `create_collection`（1024 维 / 余弦距离）。
- `_qdrant_scores`：真实走 Qdrant 稠密检索返回对齐分数；**仅当 `acl_enabled` 时**才把权限下沉为 `acl_level` payload 过滤，否则不过滤（保证无 ACL 场景检索完整）。
- 新增 `load_from_qdrant`：从 Qdrant 回灌 chunk 文本 + 载荷，重建 BM25 与索引——实现「重启免入库即可检索」（唯一事实源在向量库）。

`src/pipeline.py`：
- `Retriever` 构造时传入 `collection=cfg.collection`。
- 新增 `KnowledgeBase.load()`，委托 `retriever.load_from_qdrant()`。

## 5. 评测命令
```bash
cd rag-kit
/path/to/rag-forge/bin/python scripts/real_c_eval.py
```
脚本会：连通预检 → 重建 `kb_c` collection → 入库 8 份真实 PDF → 跑检索/阈值/生成/拒答 →
**验证持久化（Qdrant 向量点数量 + 抽样 payload.acl_level）→ 模拟重启（新建空 KB 仅 `load()` 再检索）**。

## 6. 实测指标（8 份真实 PDF · 45→8 chunks · 真实 BGE-M3 + GLM）
| 指标 | 文档解析（内存/PDF） | Qdrant 生产后端（Qdrant） |
|------|-------------------|------------------|
| 入库 chunks / 质检 | 8 / 1.0 | 8 / 1.0 |
| Recall@1 / @3 / **@5** / @8 | 75% / 90% / 100% / 100% | 75% / 90% / 100% / 100% |
| 拒答阈值（数据标定） | 0.62（干净分离） | 0.62（干净分离） |
| 生成 非空/溯源/grounding | 6/6 · 6/6 · 6/6 | 6/6 · 6/6 · 6/6 |
| 无据拒答 | 4/4 | 4/4 |
| **Qdrant 向量点** | — | **8** |
| **重启免入库检索** | — | **OK** |

> 检索指标与内存后端**逐位一致**——因为向量相同、融合逻辑相同，仅存储位置不同。

## 7. 换成「真实下载的年报 PDF」
1. 拿到真实 PDF，丢进 `data/reports/raw/`（覆盖同名或新增）。
2. `python scripts/make_real_pdfs.py` 这一步可跳过（那是合成 PDF 用的）；直接重跑 `real_c_eval.py`。
3. 真实语料上**重新标定一次** `refuse_threshold`（配置在 `configs/tenant_c.yaml`）。
4. 若客户要求离线纯内网：用 `ollama` 本地对话模型替换智谱 `llm`，把 `qdrant_url` 指向内网服务即可——数据全程不出域。

## 8. 要点总结
> “框架同时支持内存后端（零依赖、可演示）与 Qdrant 生产后端（持久化、payload 权限过滤、横向扩展），按客户算力与合规要求切换。本次把向量库从内存切到 Qdrant 后，检索/生成指标逐位一致，且验证了持久化（重启免入库即可检索）与 payload 级权限过滤——满足数据不出域的私有化部署要求。”
