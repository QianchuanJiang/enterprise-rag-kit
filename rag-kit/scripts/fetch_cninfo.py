#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从巨潮网(cninfo)下载公开年报 PDF —— 文档解析 的真实语料来源。

特点：
- 使用 cninfo 公开 API，无需登录 / 付费 / 验证码
- 默认拉取几只大盘股的 2024 年年报，作为端到端验证语料
- 仅负责下载，不解析；解析走 rag-kit 的 docling / PyMuPDF 管线

运行示例：
    # 下载默认清单（平安银行 / 贵州茅台 / 浦发银行 2024 年报）
    python scripts/fetch_cninfo.py

    # 指定个股与年份
    python scripts/fetch_cninfo.py --codes 000001 600519 --year 2024 --out data/cninfo

依赖：requests  （pip install requests）
注意：需要联网；cninfo 偶有限流，可加 --sleep 1.0 拉开间隔。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 默认大盘股清单（代码, 简称）
DEFAULT_STOCKS = [
    ("000001", "平安银行"),
    ("600519", "贵州茅台"),
    ("600000", "浦发银行"),
]

CNINFO_LIST_API = "https://webapi.cninfo.com.cn/api-cloud-platform/api/info/disclosure/resDocList"
CNINFO_STATIC = "https://static.cninfo.com.cn"  # adjunctUrl 前缀


def _get_session():
    import requests

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "http://www.cninfo.com.cn",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return s


def find_annual_report(s, code: str, year: int) -> dict | None:
    """在公告列表里定位指定年份的年度报告。"""
    params = {"stock": code, "subtype": "05", "pageNum": 1, "pageSize": 10}
    try:
        r = s.get(CNINFO_LIST_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {code} 列表请求失败: {exc}")
        return None

    announcements = (data.get("data") or {}).get("announcements") or []
    for a in announcements:
        title = a.get("announcementTitle", "")
        if "年度报告" in title and str(year) in title:
            return a
    return None


def download(s, a: dict, out_dir: Path) -> str | None:
    code = a.get("secCode") or a.get("stock", "")
    title = a.get("announcementTitle", "report")
    adjunct = a.get("adjunctUrl", "")
    if not adjunct:
        return None
    url = CNINFO_STATIC + adjunct
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{code}_{title}".replace("/", "_").replace(" ", "_")
    dest = out_dir / f"{safe}.PDF"
    try:
        r = s.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return str(dest)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 下载失败 {url}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser("cninfo 年报下载器")
    ap.add_argument("--codes", nargs="*", default=None, help="股票代码清单，缺省用默认大盘股")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default=str(ROOT / "data" / "cninfo"))
    ap.add_argument("--sleep", type=float, default=0.5, help="请求间隔（秒），防限流")
    args = ap.parse_args()

    stocks = DEFAULT_STOCKS
    if args.codes:
        # 允许 --codes 000001 600519 形式；若只给代码则简称留空
        stocks = [(c, "") for c in args.codes]

    out_dir = Path(args.out)
    s = _get_session()
    ok = 0
    for code, name in stocks:
        print(f"[fetch] {code} {name or ''} {args.year} 年报 ...")
        a = find_annual_report(s, code, args.year)
        if not a:
            print(f"  [skip] 未找到 {code} {args.year} 年度报告")
            continue
        path = download(s, a, out_dir)
        if path:
            print(f"  [ok] -> {path}")
            ok += 1
        time.sleep(args.sleep)

    print(f"\n完成：成功下载 {ok} / {len(stocks)} 份。输出目录：{out_dir}")
    print("下一步：把 PDF 放入 data/sample 后，用 configs/dev.yaml 跑真实链路（需 Ollama + 智谱 key）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
