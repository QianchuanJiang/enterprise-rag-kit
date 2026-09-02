"""命令行入口：本地验证与演示用。

用法：
  # 入库（解析 + 切片 + 质检 + 向量化）
  python cli.py ingest --config configs/tenant_a.yaml --dir data/sample

  # 一键演示：入库后立刻问答（无持久化，适合快速验证）
  python cli.py run --config configs/tenant_a.yaml --dir data/sample \\
        --question "2024 年公司营业收入是多少？"

  # 带权限级别的提问（权限合规）
  python cli.py run --config configs/tenant_b.yaml --dir data/sample \\
        --question "受限条款的具体内容？" --user-level public

说明：脚手架默认跑 memory 后端 + fake/stub，无需 GPU 与模型下载即可端到端跑通。
真实部署把 config 里的 dense_model / llm.provider 改成真实模型即可。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402

from config import load_config  # noqa: E402
from pipeline import KnowledgeBase  # noqa: E402


def _files(directory: str) -> list[str]:
    return [str(p) for p in Path(directory).rglob("*.*") if p.is_file()]


def cmd_ingest(args) -> None:
    kb = KnowledgeBase(load_config(args.config))
    rep = kb.ingest(_files(args.dir))
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def cmd_run(args) -> None:
    kb = KnowledgeBase(load_config(args.config))
    rep = kb.ingest(_files(args.dir))
    print(f"[ingest] chunks={rep['ingested_chunks']} quality={rep['quality']}")
    ans, ctx = kb.ask(args.question, user_level=args.user_level)
    print("\n[answer]")
    print(ans.text)
    print("\n[citations]", ans.citations)
    if args.user_level and kb.cfg.security.acl_enabled:
        print("\n[acl selfcheck]", kb.acl_selfcheck(args.question))


def main() -> None:
    ap = argparse.ArgumentParser("RAG 企业知识库脚手架")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="入库")
    pi.add_argument("--config", required=True)
    pi.add_argument("--dir", required=True)
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("run", help="入库并问答（演示）")
    pr.add_argument("--config", required=True)
    pr.add_argument("--dir", required=True)
    pr.add_argument("--question", required=True)
    pr.add_argument("--user-level", default="public")
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
