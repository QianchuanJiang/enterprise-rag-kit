"""FastAPI 服务：把 KnowledgeBase 暴露成 HTTP 接口。

端点：
- POST /ingest  后台入库（传入 config 路径与文档目录）
- POST /ask     问答（带 user_level 走权限隔离）
- GET  /audit   管理员查审计日志（仅示例，生产需鉴权）
- GET  /health  健康检查

样例 C 离线部署时，这条服务与 Qdrant / Ollama 一起由 docker-compose 拉起。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402

app = FastAPI(title="RAG 企业知识库", version="0.1.0")
KB = None


class IngestReq(BaseModel):
    config: str
    dir: str


class AskReq(BaseModel):
    question: str
    user_level: str = "public"
    user_id: str = "anonymous"


@app.on_event("startup")
def _load_default():
    global KB
    default = Path(__file__).resolve().parent.parent / "configs" / "tenant_a.yaml"
    if default.exists():
        KB = KnowledgeBase(load_config(default))


@app.post("/ingest")
def ingest(req: IngestReq, bg: BackgroundTasks):
    global KB
    cfg = load_config(req.config)
    KB = KnowledgeBase(cfg)
    files = [str(p) for p in Path(req.dir).rglob("*.*") if p.is_file()]
    bg.add_task(KB.ingest, files)
    return {"status": "ingesting", "files": len(files)}


@app.post("/ask")
def ask(req: AskReq):
    if KB is None:
        raise HTTPException(503, "知识库未加载，请先调用 /ingest")
    ans, _ = KB.ask(req.question, req.user_level, req.user_id)
    return ans.to_dict()


@app.get("/audit")
def audit(limit: int = 50):
    if KB is None or KB.audit is None:
        raise HTTPException(503, "审计未启用")
    lines = KB.audit.path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines[-limit:]:
        if not ln.strip():
            continue
        body = ln.rsplit("|H:", 1)[0]
        out.append(body)
    return {"count": len(out), "records": out}


@app.get("/health")
def health():
    return {"ok": True, "loaded": KB is not None}
