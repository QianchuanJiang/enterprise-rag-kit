#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知擎 RAG 控制台 · 单文件零依赖演示服务（样例 A/B/C 统一入口）。

功能：
  1. 流式问答（SSE）：逐字输出 + 检索透视（分数/耗时/拒答判定/溯源）。
  2. 向量空间投影：把 1024 维 chunk 向量做 PCA 降维成 2D，query 作为星标叠放。
  3. 溯源闭环：点答案里的 [n] 直接打开真实 PDF 对应页并高亮命中片段。
  4. 阈值实时滑块：拖动改变拒答阈值，看召回/拒答如何变化。
  5. 后端切换：Qdrant 生产 / 内存；角色切换：公开/员工/管理员（样例 B 权限双身份）。
  6. 直播评测台：一键跑 10 题测试集，实时读出 Recall/拒答/生成指标。

运行：
  cd rag-kit
  docker run -d -p 6333:6333 qdrant/qdrant   # 样例 C 需要
  /path/to/rag-forge/bin/python web_demo.py
  浏览器打开 http://localhost:8080
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("RAG_EMBEDDER", "ollama")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402

try:
    import fitz  # noqa: E402
except Exception:  # pragma: no cover
    fitz = None

CONFIG_C = ROOT / "configs" / "tenant_c.yaml"
RAW_DIR = ROOT / "data" / "reports" / "raw"
METRICS = ROOT / "data" / "reports" / "metrics_c.json"
PY = sys.executable

PORT = int(os.environ.get("PORT", "8080"))

# ---------------- KB 管理（懒加载 + 缓存）----------------
KB_CACHE: dict = {}
VIZ_CACHE: dict = {}
_KB_LOCK = threading.Lock()

EVAL_PROC = None
EVAL_LOCK = threading.Lock()


def _files() -> list:
    return sorted(str(p) for p in RAW_DIR.rglob("*.pdf") if p.is_file())


def build_kb(kind: str) -> KnowledgeBase:
    if kind in KB_CACHE:
        return KB_CACHE[kind]
    cfg = load_config(str(CONFIG_C))
    if kind == "qdrant_c":
        kb = KnowledgeBase(cfg, backend="qdrant")
        kb.load()  # 从 Qdrant 回灌，秒级，免重嵌入
    elif kind == "memory_c":
        kb = KnowledgeBase(cfg, backend="memory")
        kb.ingest(_files())
    elif kind == "acl":
        # 样例 B 权限双身份：把两家公司标为受限，演示员工被拦/管理员可见
        cfg.security.acl_enabled = True
        cfg.security.acl_map = {"贵州茅台": "restricted", "宁德时代": "restricted"}
        kb = KnowledgeBase(cfg, backend="memory")
        kb.ingest(_files())
    else:
        raise ValueError(kind)
    KB_CACHE[kind] = kb
    return kb


def select_kb(backend: str, role: str):
    """返回 (kb, user_level)。非公开角色强制走 ACL 知识库。"""
    if role in ("employee", "admin"):
        lvl = "internal" if role == "employee" else "restricted"
        return build_kb("acl"), lvl
    kind = "qdrant_c" if backend == "qdrant" else "memory_c"
    return build_kb(kind), "public"


# ---------------- 向量投影（PCA）----------------
def _pca(vecs):
    """返回 (mean, comps, coords)。优先 numpy，缺失时退化为前两维。"""
    try:
        import numpy as np
        X = np.asarray(vecs, dtype=float)
        mean = X.mean(0)
        Xc = X - mean
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:2].T
        except Exception:
            comps = np.eye(len(mean), 2)
        coords = Xc @ comps
        return mean, comps, coords
    except Exception:
        d = len(vecs[0])
        mean = [0.0] * d
        comps = [[1.0, 0.0], [0.0, 1.0]][:d] if d >= 2 else None
        coords = [[v[0], v[1]] for v in vecs]
        return mean, comps, coords


def get_viz(kb: KnowledgeBase):
    """返回 [(chunk, vector)] 并对该 KB 缓存 PCA 基。"""
    key = id(kb)
    if key in VIZ_CACHE:
        return VIZ_CACHE[key]
    chunks = kb.retriever.chunks
    if kb.backend == "qdrant":
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=kb.cfg.qdrant_url)
        if not qc.collection_exists(kb.cfg.collection):
            VIZ_CACHE[key] = []
            return []
        pts, _ = qc.scroll(
            collection_name=kb.cfg.collection,
            with_vectors=True, with_payload=True, limit=256,
        )
        vecs = [p.vector for p in pts]
    else:
        vecs = kb.retriever._vecs
    out = list(zip(chunks, vecs))
    VIZ_CACHE[key] = out
    return out


def pca_basis_for(kb: KnowledgeBase):
    viz = get_viz(kb)
    if not viz:
        return None, None, []
    vecs = [v for _, v in viz]
    mean, comps, coords = _pca(vecs)
    return mean, comps, coords


# ---------------- 工具 ----------------
def _chunk_text(s: str, size: int = 8):
    return [s[i : i + size] for i in range(0, len(s), size)] or [""]


def _norm_nums(s: str) -> set:
    import re
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*\.?\d*", s) if len(m) >= 3}


# ---------------- HTTP 处理 ----------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ZhqingRAG/1.0"

    # ---- 响应辅助 ----
    def _send_json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        for name, payload in events:
            try:
                data = json.dumps(payload, ensure_ascii=False)
                self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                break

    def log_message(self, *args):  # 静默
        pass

    # ---- 路由 ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._serve_page()
        elif u.path == "/api/status":
            self._api_status(u)
        elif u.path == "/api/vectors":
            self._api_vectors(u)
        elif u.path == "/api/source":
            self._api_source(u)
        elif u.path == "/api/eval":
            self._api_eval_get()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            data = {}
        if u.path == "/api/ask":
            self._api_ask(data)
        elif u.path == "/api/eval/run":
            self._api_eval_run()
        else:
            self._send_json({"error": "not found"}, 404)

    # ---- 页面 ----
    def _serve_page(self):
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- 状态 ----
    def _api_status(self, u):
        global EVAL_PROC
        qp = parse_qs(u.query)
        backend = qp.get("backend", ["qdrant"])[0]
        role = qp.get("role", ["public"])[0]
        try:
            kb, lvl = select_kb(backend, role)
            n_chunks = len(kb.retriever.chunks)
            acl = bool(kb.cfg.security.acl_enabled)
            thr = kb.cfg.guard.refuse_threshold
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, 500)
            return
        running = bool(EVAL_PROC and EVAL_PROC.poll() is None)
        metrics = None
        if METRICS.exists():
            try:
                metrics = json.loads(METRICS.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
        self._send_json({
            "ok": True,
            "backend": kb.backend,
            "role": role,
            "user_level": lvl,
            "chunks": n_chunks,
            "acl_enabled": acl,
            "threshold": thr,
            "qdrant_url": kb.cfg.qdrant_url,
            "metrics": metrics,
            "eval_running": running,
        })

    # ---- 向量投影 ----
    def _api_vectors(self, u):
        qp = parse_qs(u.query)
        backend = qp.get("backend", ["qdrant"])[0]
        role = qp.get("role", ["public"])[0]
        q = qp.get("q", [""])[0].strip()
        try:
            kb, _ = select_kb(backend, role)
            mean, comps, coords = pca_basis_for(kb)
            viz = get_viz(kb)
            if not viz:
                self._send_json({"points": [], "query": None})
                return
            points = []
            for i, (c, _) in enumerate(viz):
                xy = coords[i]
                points.append({
                    "id": i,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "source": os.path.basename(c.meta.get("source", "")),
                    "page": c.page,
                    "acl": c.meta.get("acl_level", "public"),
                    "snippet": c.text[:48],
                })
            qcoord = None
            if q and comps is not None:
                try:
                    import numpy as np
                    qv = kb.embedder.embed_query(q)
                    qc = (np.asarray(qv, float) - np.asarray(mean, float)) @ np.asarray(comps, float)
                    qcoord = {"x": float(qc[0]), "y": float(qc[1])}
                except Exception:
                    qcoord = None
            self._send_json({"points": points, "query": qcoord})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    # ---- PDF 溯源 ----
    def _api_source(self, u):
        if fitz is None:
            self._send_json({"error": "fitz unavailable"}, 500)
            return
        qp = parse_qs(u.query)
        src = _unquote(qp.get("src", [""])[0])
        page = int(qp.get("page", [0])[0] or 0)
        text = _unquote(qp.get("text", [""])[0])
        if not os.path.exists(src):
            cand = RAW_DIR / os.path.basename(src)
            if cand.exists():
                src = str(cand)
        try:
            doc = fitz.open(src)
            idx = max(0, page - 1) if page > 0 else 0
            idx = min(idx, doc.page_count - 1)
            pg = doc[idx]
            rects = pg.search_for(text[:40]) if text else []
            for r in rects:
                try:
                    pg.add_highlight_annot(r)
                except Exception:
                    pass
            pix = pg.get_pixmap(dpi=110)
            png = base64.b64encode(pix.tobytes("png")).decode("ascii")
            self._send_json({"png": png, "found": len(rects) > 0, "pages": doc.page_count})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    # ---- 评测 ----
    def _api_eval_get(self):
        global EVAL_PROC
        running = bool(EVAL_PROC and EVAL_PROC.poll() is None)
        metrics = None
        if METRICS.exists():
            try:
                metrics = json.loads(METRICS.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
        self._send_json({"running": running, "metrics": metrics})

    def _api_eval_run(self):
        global EVAL_PROC
        with EVAL_LOCK:
            if EVAL_PROC and EVAL_PROC.poll() is None:
                self._send_json({"running": True})
                return
            try:
                EVAL_PROC = subprocess.Popen(
                    [PY, str(ROOT / "scripts" / "real_c_eval.py")],
                    cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._send_json({"started": True})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)

    # ---- 问答（SSE）----
    def _api_ask(self, data):
        question = (data.get("question") or "").strip()
        backend = data.get("backend", "qdrant")
        role = data.get("role", "public")
        try:
            threshold = float(data.get("threshold", 0.62))
        except Exception:
            threshold = 0.62
        if not question:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._sse([("error", {"msg": "问题为空"})])
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            kb, user_level = select_kb(backend, role)
            kb.cfg.guard.refuse_threshold = threshold
        except Exception as exc:
            self._sse([("error", {"msg": str(exc)})])
            return

        # --- 嵌入计时 ---
        t0 = time.time()
        try:
            qvec = kb.embedder.embed_query(question)
        except Exception as exc:
            self._sse([("error", {"msg": f"嵌入失败: {exc}"})])
            return
        t_emb = time.time() - t0

        # --- 检索计时 ---
        t0 = time.time()
        try:
            scored = kb.retriever.search(question, user_level=user_level, return_scores=True)
        except Exception as exc:
            self._sse([("error", {"msg": f"检索失败: {exc}"})])
            return
        t_ret = time.time() - t0

        contexts = [c for c, _ in scored]
        scores = [s for _, s in scored]
        best = max(scores) if scores else 0.0
        fake = getattr(kb.embedder, "mode", None) == "fake"
        score_refuse = bool(threshold) and not fake and best < threshold

        denied = False
        if kb.cfg.security.acl_enabled and user_level != "restricted":
            try:
                priv = kb.retriever.search(question, user_level="restricted", return_scores=True)
                denied = len(priv) > len(scored)
            except Exception:
                denied = False

        top = []
        for c, s in list(zip(contexts, scores))[:6]:
            top.append({
                "source": os.path.basename(c.meta.get("source", "")),
                "page": c.page,
                "score": round(s, 3),
                "acl": c.meta.get("acl_level", "public"),
                "snippet": c.text[:140],
                "text": c.text[:400],
                "full_source": c.meta.get("source", ""),
                "citation": c.citation,
            })

        refused = (not contexts) or score_refuse
        self._sse([("trace", {
            "embed_ms": round(t_emb * 1000),
            "retrieve_ms": round(t_ret * 1000),
            "best": round(best, 3),
            "threshold": threshold,
            "refused": refused,
            "denied": denied,
            "acl_enabled": kb.cfg.security.acl_enabled,
            "user_level": user_level,
            "contexts": top,
        })])

        if refused:
            ans = kb.generator.generate(question, [], refuse_if_empty=True)
            answer = ans.text
            self._sse([("delta", {"text": answer})])
            gen_ms = 0
        else:
            t0 = time.time()
            ans = kb.generator.generate(question, contexts)
            gen_ms = time.time() - t0
            answer = ans.text
            for piece in _chunk_text(answer, 8):
                self._sse([("delta", {"text": piece})])
                time.sleep(0.02)

        citations = [c.citation for c in contexts] if not refused else []
        self._sse([("genmeta", {"generate_ms": round(gen_ms * 1000)})])
        self._sse([("done", {
            "answer": answer,
            "refused": refused or ans.refused,
            "reason": getattr(ans, "reason", ""),
            "citations": citations,
            "contexts": top,
        })])


def _unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


# ---------------- 前端页面 ----------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>知擎 RAG 控制台</title>
<style>
  :root{
    --bg:#f6f7fb; --card:#ffffff; --ink:#0f172a; --mut:#64748b; --line:#e6e8ef;
    --blue:#185fa5; --blue2:#378add; --green:#3b6d11; --red:#a32d2d; --amber:#854f0b;
    --chip:#eef2f8;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:18px}
  .top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 16px}
  .brand{font-weight:600;font-size:16px}
  .pill{font-size:12px;padding:4px 10px;border-radius:999px;background:var(--chip);color:var(--mut);border:1px solid var(--line)}
  .pill.on{background:#e7f4ec;color:var(--green);border-color:#bfe3c8}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden}
  .seg button{border:0;background:#fff;padding:5px 12px;font-size:12px;color:var(--mut);cursor:pointer}
  .seg button.active{background:var(--blue);color:#fff}
  .slider{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mut)}
  input[type=range]{width:130px}
  .metrics{margin-left:auto;display:flex;gap:14px;font-size:12px;color:var(--mut)}
  .metrics b{color:var(--ink);font-size:14px}
  .grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:16px}
  @media(max-width:880px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
  .card h3{margin:0 0 10px;font-size:14px;font-weight:600}
  .chat{height:430px;overflow:auto;display:flex;flex-direction:column;gap:12px}
  .msg{display:flex;gap:8px}
  .msg .b{flex:1;padding:10px 12px;border-radius:10px}
  .u .b{background:#eaf1fb;margin-left:36px}
  .a .b{background:#f3f5f9;margin-right:8px}
  .badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;background:#fbeaea;color:var(--red);border:1px solid #f3c4c4}
  .badge.ok{background:#e7f4ec;color:var(--green);border-color:#bfe3c8}
  .cite{color:var(--blue);cursor:pointer;font-weight:600}
  .caret{display:inline-block;width:8px;height:14px;background:var(--blue2);margin-left:1px;vertical-align:-2px;animation:bl 1s steps(1) infinite}
  @keyframes bl{50%{opacity:0}}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .chips button{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 12px;font-size:12px;color:var(--mut);cursor:pointer}
  .inputbar{display:flex;gap:8px;margin-top:10px}
  .inputbar input{flex:1;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px}
  .inputbar button{border:0;background:var(--blue);color:#fff;border-radius:10px;padding:0 18px;cursor:pointer;font-size:14px}
  canvas{width:100%;height:240px;background:#fbfcfe;border:1px solid var(--line);border-radius:10px}
  .trace{margin-top:8px;font-size:12px;color:var(--mut)}
  .trace .row{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  .tg{background:var(--chip);border-radius:6px;padding:4px 8px}
  .ctx{border-top:1px dashed var(--line);margin-top:10px;padding-top:8px}
  .ctx .c{font-size:12px;padding:6px 8px;border-radius:8px;background:#fafbfd;margin-bottom:6px;cursor:pointer}
  .ctx .c:hover{background:#eef4fb}
  .bar{height:6px;background:#eef1f6;border-radius:4px;overflow:hidden;margin:4px 0}
  .bar > i{display:block;height:100%;background:var(--blue2)}
  .modal{position:fixed;inset:0;background:rgba(15,23,42,.55);display:none;align-items:center;justify-content:center;z-index:50}
  .modal.show{display:flex}
  .modal .box{background:#fff;border-radius:12px;padding:12px;max-width:90vw;max-height:88vh;overflow:auto}
  .modal img{max-width:100%;border-radius:8px}
  .evalgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
  .ev{background:#fafbfd;border:1px solid var(--line);border-radius:10px;padding:10px;text-align:center}
  .ev .n{font-size:20px;font-weight:600}
  .ev .l{font-size:11px;color:var(--mut)}
  .muted{color:var(--mut);font-size:12px}
  .runbtn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 14px;cursor:pointer;font-size:13px}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <span class="brand">知擎 RAG 控制台</span>
    <span id="bkPill" class="pill on">Qdrant ●</span>
    <span id="aclPill" class="pill">ACL 关</span>
    <div class="seg" id="bkSeg">
      <button data-bk="qdrant" class="active">生产 Qdrant</button>
      <button data-bk="memory">内存</button>
    </div>
    <div class="seg" id="roleSeg">
      <button data-role="public" class="active">公开</button>
      <button data-role="employee">员工</button>
      <button data-role="admin">管理员</button>
    </div>
    <span class="slider">拒答阈值 <input id="thr" type="range" min="0.30" max="0.90" step="0.01" value="0.62"><b id="thrV">0.62</b></span>
    <span class="metrics" id="metrics"></span>
  </div>

  <div class="grid">
    <div class="card">
      <h3>对话</h3>
      <div class="chat" id="chat"></div>
      <div class="chips" id="chips"></div>
      <div class="inputbar">
        <input id="q" placeholder="问点什么，例如：平安银行 2024 年净利润是多少？">
        <button id="send">发送</button>
      </div>
    </div>

    <div class="card">
      <h3>检索透视</h3>
      <canvas id="scatter" width="600" height="240"></canvas>
      <div class="muted" id="scInfo">PCA 投影 · 蓝点=召回片段，★=你的问题</div>
      <div class="trace" id="trace"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>直播评测台 <button class="runbtn" id="runEval">运行 10 题测试集</button> <span class="muted" id="evalState"></span></h3>
    <div class="evalgrid" id="evalGrid"></div>
  </div>
</div>

<div class="modal" id="modal"><div class="box"><div id="modalBody"></div></div></div>

<script>
const state = { backend:"qdrant", role:"public", threshold:0.62 };
const EXAMPLES = [
  "平安银行 2024 年净利润是多少？",
  "贵州茅台 2024 年营业总收入是多少？",
  "招商银行 核心一级资本充足率是多少？",
  "宁德时代 2024 年归母净利润是多少？",
  "特斯拉 2024 年第四季度交付量是多少？"
];
let curAnsEl=null, curTraceEl=null, lastContexts=[];

function $(s){return document.querySelector(s);}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function highlight(s){
  return esc(s).replace(/\[(\d+)\]/g,'<span class="cite" data-i="$1">[$1]</span>');
}

async function loadStatus(){
  const r = await fetch(`/api/status?backend=${state.backend}&role=${state.role}`);
  const d = await r.json();
  if(!d.ok) return;
  $("#bkPill").textContent = d.backend==="qdrant" ? "Qdrant ●" : "内存 ●";
  $("#aclPill").textContent = d.acl_enabled ? "ACL 开" : "ACL 关";
  $("#aclPill").className = "pill " + (d.acl_enabled?"on":"");
  $("#metrics").innerHTML = `chunks <b>${d.chunks}</b> · 阈值 <b>${d.threshold}</b>`;
  drawScatter(null);
}

// ---- 散点图 ----
async function drawScatter(q){
  const r = await fetch(`/api/vectors?backend=${state.backend}&role=${state.role}&q=${encodeURIComponent(q||"")}`);
  const d = await r.json();
  const cv = $("#scatter"); const ctx = cv.getContext("2d");
  ctx.clearRect(0,0,cv.width,cv.height);
  if(!d.points || !d.points.length){ $("#scInfo").textContent="暂无向量（请先运行评测或切换后端）"; return; }
  const xs=d.points.map(p=>p.x), ys=d.points.map(p=>p.y);
  const minx=Math.min(...xs,...(d.query?[d.query.x]:[])), maxx=Math.max(...xs,...(d.query?[d.query.x]:[]));
  const miny=Math.min(...ys,...(d.query?[d.query.y]:[])), maxy=Math.max(...ys,...(d.query?[d.query.y]:[]));
  const sx=v=>30+(v-minx)/((maxx-minx)||1)*(cv.width-60);
  const sy=v=>cv.height-20-(v-miny)/((maxy-miny)||1)*(cv.height-40);
  // 召回片段（来自最近一次 trace 的 id 集合）
  const hot = new Set((lastContexts||[]).map(c=>c._id));
  d.points.forEach((p,i)=>{
    p._id=i;
    const isHot = hot.has(i);
    ctx.beginPath(); ctx.arc(sx(p.x),sy(p.y),isHot?5:3.5,0,7);
    ctx.fillStyle = p.acl==="restricted" ? "#a32d2d" : (isHot ? "#185fa5" : "#9fb6cf");
    ctx.fill();
  });
  if(d.query){
    const qx=sx(d.query.x), qy=sy(d.query.y);
    ctx.fillStyle="#0f172a"; ctx.font="14px sans-serif";
    ctx.fillText("★", qx-5, qy+5);
  }
  cv.onmousemove = (e)=>{
    const rect=cv.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(cv.width/rect.width);
    const my=(e.clientY-rect.top)*(cv.height/rect.height);
    let best=null,bd=1e9;
    d.points.forEach(p=>{const dx=sx(p.x)-mx,dy=sy(p.y)-my;const dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=p;}});
    if(best && bd<200){ $("#scInfo").textContent=`${best.source} · P${best.page} · ${best.acl} · ${best.snippet}…`; }
    else { $("#scInfo").textContent="PCA 投影 · 蓝点=召回片段，★=你的问题"; }
  };
  // 把坐标回填给 trace 高亮
  lastCoords = d;
}

// ---- 发送 ----
async function ask(){
  const q = $("#q").value.trim();
  if(!q) return;
  $("#q").value="";
  addUser(q);
  const aEl=document.createElement("div"); aEl.className="msg a";
  const b=document.createElement("div"); b.className="b";
  b.innerHTML='<span class="caret"></span>';
  aEl.appendChild(b); $("#chat").appendChild(aEl);
  curAnsEl=b; curAnsEl._full="";
  curTraceEl=null; lastContexts=[];

  const resp = await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({question:q,backend:state.backend,role:state.role,threshold:state.threshold})});
  const reader = resp.body.getReader(); const dec=new TextDecoder(); let buf="";
  while(true){
    const {done,value}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let idx;
    while((idx=buf.indexOf("\n\n"))>=0){
      const chunk=buf.slice(0,idx); buf=buf.slice(idx+2);
      const m=chunk.match(/event: (\w+)\ndata: ([\s\S]*)/);
      if(m) handleEvent(m[1], JSON.parse(m[2]));
    }
  }
  drawScatter(q);
}

function handleEvent(name, d){
  if(name==="trace"){
    renderTrace(d);
  } else if(name==="delta"){
    if(curAnsEl){ curAnsEl._full=(curAnsEl._full||"")+d.text; curAnsEl.innerHTML=highlight(curAnsEl._full)+'<span class="caret"></span>'; $("#chat").scrollTop=$("#chat").scrollHeight; }
  } else if(name==="genmeta"){
    if(curTraceEl) curTraceEl.querySelector(".gen").textContent=`生成 ${d.generate_ms}ms`;
  } else if(name==="done"){
    if(curAnsEl){ curAnsEl.innerHTML=highlight(curAnsEl._full||""); }
    if(d.refused){
      const bd=document.createElement("span"); bd.className="badge"; bd.textContent="拒答";
      curAnsEl.insertBefore(bd, curAnsEl.firstChild);
    }
    lastContexts=d.contexts||[];
    bindCites();
  } else if(name==="error"){
    if(curAnsEl) curAnsEl.innerHTML='<span class="badge">错误</span> '+esc(d.msg);
  }
}

function renderTrace(d){
  const t=document.createElement("div"); t.className="trace";
  const verdict = d.refused ? '<span class="badge">拒答</span>' : (d.denied?'<span class="badge">权限拦截</span>':'<span class="badge ok">放行</span>');
  t.innerHTML = `<div>判定：${verdict} · 最高分 <b>${d.best}</b> / 阈值 <b>${d.threshold}</b> · 用户级 ${d.user_level}</div>
    <div class="row">
      <span class="tg">嵌入 ${d.embed_ms}ms</span>
      <span class="tg">检索 ${d.retrieve_ms}ms</span>
      <span class="tg gen">生成 —</span>
    </div>
    <div class="ctx" id="ctxBox"></div>`;
  $("#trace").innerHTML=""; $("#trace").appendChild(t);
  curTraceEl=t;
  const box=t.querySelector("#ctxBox");
  (d.contexts||[]).forEach((c,i)=>{
    const el=document.createElement("div"); el.className="c";
    el.dataset.i=i; el.dataset.src=c.full_source; el.dataset.page=c.page; el.dataset.text=c.text;
    const pct=Math.max(4,Math.round(c.score*100));
    el.innerHTML=`<div>${i+1}. ${esc(c.source)} · P${c.page} · ${c.acl} · 分 ${c.score}</div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="muted">${esc(c.snippet)}</div>`;
    box.appendChild(el);
  });
}

function addUser(q){
  const m=document.createElement("div"); m.className="msg u";
  const b=document.createElement("div"); b.className="b"; b.textContent=q;
  m.appendChild(b); $("#chat").appendChild(m); $("#chat").scrollTop=$("#chat").scrollHeight;
}

function bindCites(){
  document.querySelectorAll(".cite").forEach(s=>{
    s.onclick=()=>{
      const i=+s.dataset.i-1; const c=(lastContexts||[])[i];
      if(c) openSource(c.full_source, c.page, c.text);
    };
  });
  document.querySelectorAll(".ctx .c").forEach(el=>{
    el.onclick=()=>openSource(el.dataset.src, +el.dataset.page, el.dataset.text);
  });
}

async function openSource(src,page,text){
  const r=await fetch(`/api/source?src=${encodeURIComponent(src)}&page=${page}&text=${encodeURIComponent(text)}`);
  const d=await r.json();
  if(d.png){ $("#modalBody").innerHTML=`<img src="data:image/png;base64,${d.png}"><div class="muted">${d.found?"已高亮命中片段":"未精确匹配，已展示该页"} · ${esc(src)} P${page}</div>`; $("#modal").classList.add("show"); }
  else { $("#modalBody").textContent="无法打开："+ (d.error||"未知"); $("#modal").classList.add("show"); }
}
$("#modal").onclick=()=>$("#modal").classList.remove("show");

// ---- 评测台 ----
async function loadEval(){
  const r=await fetch("/api/eval"); const d=await r.json();
  const g=$("#evalGrid");
  if(d.running){ $("#evalState").textContent="评测进行中…"; setTimeout(loadEval,3000); return; }
  if(!d.metrics){ $("#evalState").textContent="尚未运行"; g.innerHTML='<div class="muted">点“运行 10 题测试集”生成指标</div>'; return; }
  const m=d.metrics;
  const rec=m.recall||{}; const gen=m.generation||{};
  g.innerHTML=`
    <div class="ev"><div class="n">${(rec["@5"]!=null?Math.round(rec["@5"]*100):"—")}%</div><div class="l">Recall@5</div></div>
    <div class="ev"><div class="n">${gen.cited||"—"}</div><div class="l">溯源率</div></div>
    <div class="ev"><div class="n">${m.refusal_pass||"—"}</div><div class="l">无据拒答</div></div>
    <div class="ev"><div class="n">${m.qdrant_points!=null?m.qdrant_points:"—"}</div><div class="l">Qdrant 向量点</div></div>`;
  $("#evalState").textContent=`相关[${m.relevant_score_range}] / 无关[${m.irrelevant_score_range}] · 建议阈值 ${m.suggested_refuse_threshold}`;
}
$("#runEval").onclick=async()=>{
  $("#evalState").textContent="已启动…";
  await fetch("/api/eval/run",{method:"POST"});
  setTimeout(loadEval,2000);
};

// ---- 控件 ----
$("#bkSeg").onclick=e=>{ const b=e.target.dataset.bk; if(!b) return;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle("active",x===e.target));
  state.backend=b; loadStatus(); };
$("#roleSeg").onclick=e=>{ const r=e.target.dataset.role; if(!r) return;
  [...e.currentTarget.children].forEach(x=>x.classList.toggle("active",x===e.target));
  state.role=r; loadStatus(); };
$("#thr").oninput=e=>{ state.threshold=+e.target.value; $("#thrV").textContent=(+e.target.value).toFixed(2); };
$("#send").onclick=ask; $("#q").addEventListener("keydown",e=>{ if(e.key==="Enter") ask(); });
EXAMPLES.forEach(q=>{ const b=document.createElement("button"); b.textContent=q; b.onclick=()=>{ $("#q").value=q; ask(); }; $("#chips").appendChild(b); });

loadStatus(); loadEval();
</script>
</body>
</html>"""


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"知擎 RAG 控制台已启动: http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
