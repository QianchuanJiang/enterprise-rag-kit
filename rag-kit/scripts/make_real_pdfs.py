#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 data/reports/ 下的合成年报 .md 渲染成「真正的 PDF 文件」到 data/reports/raw/。

目的：在当前环境无法从巨潮/东方财富下载真实 PDF（接口鉴权/JS 渲染）的情况下，
用 PyMuPDF 把合成文本写成货真价实的 PDF，从而跑通
    PDF 文件 -> PyMuPDF 抽取(_parse_pymupdf) -> 切片 -> 质检 -> BGE-M3 -> GLM
这条与「下载来的真实 PDF」完全一致的「真实 PDF 路线」生产代码链路。

注意：PDF 的 *文本来源* 是合成逼真版（因下载被服务端拦截），但 *文件格式与解析*
100% 真实 —— _parse_pymupdf 面对的是标准 PDF，与交付客户时一致。
等环境恢复（能拿到真实年报 PDF）时，只需把真实 PDF 丢进 raw/ 覆盖同名文件，重跑评测即可。

依赖：PyMuPDF（rag-forge 环境已装：fitz 1.28.0）+ 系统中文矢量字体。
运行：
    cd rag-kit
    /path/to/rag-forge/bin/python scripts/make_real_pdfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
SRC_MD = ROOT / "data" / "reports"
OUT_PDF = ROOT / "data" / "reports" / "raw"
OUT_PDF.mkdir(parents=True, exist_ok=True)

# macOS 自带中文矢量字体（TrueType Collection）；按可用性回退
for _f in (
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
):
    if Path(_f).exists():
        FONT = _f
        break
else:
    FONT = None  # 退化为默认字体（中文可能乱码，仅演示）

PAGE_W, PAGE_H = 595, 842  # A4
MARGIN = 50
LINE_H = 20


def _render(doc: "fitz.Document", lines: list[str]) -> None:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN
    for raw in lines:
        ln = raw.rstrip()
        if y > PAGE_H - MARGIN:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN
        if not ln.strip():
            y += LINE_H * 0.5
            continue
        if ln.startswith("# "):
            size, text = 18, ln[2:].strip()
        elif ln.startswith("## "):
            size, text = 15, ln[3:].strip()
        elif ln.startswith("### "):
            size, text = 13, ln[4:].strip()
        elif ln.strip().startswith("|") and "|" in ln[1:]:
            size, text = 10, ln  # 表格行：等宽小字
        else:
            size, text = 11, ln
        try:
            page.insert_text((MARGIN, y), text, fontname="cjk", fontfile=FONT, fontsize=size)
        except Exception:
            page.insert_text((MARGIN, y), text, fontsize=size)
        y += LINE_H + size * 0.35


def main() -> int:
    print(f"字体: {FONT or '默认(中文可能乱码)'}")
    count = 0
    for md in sorted(SRC_MD.glob("*.md")):
        if md.name.upper() == "README.MD":
            continue
        text = md.read_text(encoding="utf-8")
        doc = fitz.open()
        _render(doc, text.splitlines())
        try:
            doc.subset_fonts()  # 子集化：55MB -> 数百 KB，中文抽取不受影响
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] subset_fonts failed for {md.name}: {e}")
        out = OUT_PDF / (md.stem + ".pdf")
        doc.save(str(out))
        doc.close()
        print(f"  wrote {out.name}")
        count += 1
    print(f"完成：{count} 个真实 PDF -> {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
