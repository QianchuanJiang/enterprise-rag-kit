# 可观测 Web 控制台

`rag-kit/web_demo.py` —— 单文件、零额外依赖（仅 Python 标准库 `http.server`）的可观测控制台。

设计目标：把"黑盒问答"变成**可解释的检索工作台**，让检索机制本身（耗时、相似度、阈值判定、权限过滤、召回片段）直接可见。

---

## 1. 能力概览

| 能力 | 说明 |
|---|---|
| 检索透视 | 每次问答展示嵌入 / 检索 / 生成各阶段耗时、最高相似度 vs 拒答阈值、放行 / 拒答 / 权限拦截判定，以及 Top-K 召回片段与相关性分数条 |
| 向量空间投影 | 把 chunk 的 1024 维向量做 PCA 降维成 2D 散点，问题向量叠加为星标，直观呈现"为什么召回这几段" |
| 溯源闭环 | 点击答案中的 `[n]` 引用，右侧直接渲染真实 PDF 对应页，并用 PyMuPDF 高亮命中片段 |
| 阈值实时调节 | 拖动滑块即时改变拒答阈值，观察召回与拒答行为如何变化 |
| 后端与身份切换 | 内存 / Qdrant 后端切换；公开 / 员工 / 管理员身份切换，用于验证检索层 ACL |
| 内置评测台 | 一键运行测试集，实时输出 Recall、溯源率、无据拒答等指标 |

## 2. 启动

```bash
cd rag-kit
PORT=8080 python web_demo.py      # 浏览器打开 http://localhost:8080
```

启动逻辑：若 Qdrant 中已有数据则直接 `load()` 恢复（秒级，免重新嵌入）；否则从 PDF 解析并入库。

## 3. 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 控制台页面 |
| `/api/status` | GET | 后端状态、chunk 数、向量点数、拒答阈值、评测指标 |
| `/api/ask` | POST | SSE 流式：`trace`（耗时 / 分数 / 判定）→ `token`（逐字答案）→ `done`（引用与上下文） |
| `/api/vectors` | GET | PCA 投影坐标（chunk 向量 + 问题向量） |
| `/api/source` | GET | 渲染指定 PDF 页并高亮命中片段（返回 PNG） |
| `/api/eval` | GET / POST | 读取或触发评测 |

SSE 事件序列示例：

```
event: trace   data: {"embed_ms":..,"retrieve_ms":..,"best":0.814,"threshold":0.62,"refused":false}
event: token    data: {"delta":"平安银行"}
event: done     data: {"answer":"..","citations":[..],"contexts":[..]}
```

## 4. 前端实现

原生 HTML + JavaScript，无框架、无构建步骤。SSE 逐字渲染；答案中的 `[n]` 渲染为可点击引用徽章，点击后在右侧面板打开 PDF 原文。

## 5. 设计约束

1. **必须调用真实 pipeline** —— 禁止写死答案或返回预设结果。
2. **降级可用** —— 生成器支持 `stub` 模式：断网 / 无 API key 时，检索、溯源、权限拦截与审计链路仍完整可跑。
3. **零额外依赖** —— 不引入 FastAPI / Streamlit，保证在最小 Python 环境中即可运行。
