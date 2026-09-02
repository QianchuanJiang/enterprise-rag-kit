"""文档解析层。

经验：RAG 效果 80% 的问题出在这一层，不在模型。
表格、扫描件、页眉页脚是三个最大的坑，必须逐个处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import ParserConfig

# 常见噪声：页眉、页脚、独立页码、水印
_NOISE_PATTERNS = [
    r"^\s*第?\s*\d+\s*页?(\s*/\s*共?\s*\d+\s*页?)?\s*$",
    r"^\s*-?\s*\d+\s*-?\s*$",
    r"^\s*目录\s*$",
    r"^\s*\.{5,}\s*\d*\s*$",
]


@dataclass
class Block:
    """统一的中间表示：所有格式最终都归一成 Block 列表。"""

    type: str  # heading | paragraph | table | image_caption
    text: str
    page: int = 0
    level: int = 0  # heading 层级
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedDoc:
    doc_id: str
    source: str
    blocks: list[Block]
    parse_ok: bool = True
    fail_reason: str = ""


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(re.match(p, stripped) for p in _NOISE_PATTERNS)


def _clean(text: str, drop_header_footer: bool) -> str:
    if not drop_header_footer:
        return text
    lines = [ln for ln in text.splitlines() if not _is_noise(ln)]
    return "\n".join(lines).strip()


def _table_to_kv(rows: list[list[str]]) -> str:
    """大表格拆成「属性-值对」单独存储。

    整表向量化几乎必然失效：问「2024 年毛利率」时，
    语义相似度很难命中一整张合并报表。
    """
    if not rows:
        return ""
    header = rows[0]
    parts = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        pairs = []
        for i, cell in enumerate(row):
            key = header[i].strip() if i < len(header) and header[i].strip() else f"列{i + 1}"
            val = cell.strip()
            if val:
                pairs.append(f"{key}：{val}")
        if pairs:
            parts.append("；".join(pairs))
    return "\n".join(parts)


def _parse_docling(path: Path, cfg: ParserConfig) -> list[Block]:
    """复杂版式（跨页表格、图文混排）优先用 Docling。"""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    blocks: list[Block] = []
    for item, level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()
        text = getattr(item, "text", "") or ""
        page = getattr(getattr(item, "prov", [None])[0], "page_no", 0) or 0

        if "table" in label:
            table = getattr(item, "data", None)
            rows = getattr(table, "grid", None) if table else None
            if rows:
                text = _table_to_kv(rows) if cfg.table_mode == "kv_pairs" else text
                blocks.append(Block(type="table", text=text, page=page, meta={"rows": len(rows)}))
                continue
        if "title" in label or "heading" in label or "section_header" in label:
            blocks.append(Block(type="heading", text=text.strip(), page=page, level=1))
            continue
        if text.strip():
            blocks.append(Block(type="paragraph", text=text.strip(), page=page))
    return blocks


def _parse_pymupdf(path: Path, cfg: ParserConfig) -> list[Block]:
    """文版 PDF 用 PyMuPDF，轻量快速。

    判断依据：直接抽取有文字层就不用 OCR，没有才上 OCR。
    """
    import fitz

    doc = fitz.open(str(path))
    blocks: list[Block] = []
    for page_no in range(len(doc)):
        page = doc[page_no]
        text = page.get_text("text")

        if not text.strip() and cfg.ocr:
            text = _ocr_page(page, cfg)
        if not text.strip():
            continue

        # 表格单独抽取：PyMuPDF 的 find_tables 对有线框表格效果好
        tables = page.find_tables()
        table_zones = [(t.bbox, t.extract()) for t in tables]

        body = text
        for bbox, rows in table_zones:
            if not rows:
                continue
            rendered = _table_to_kv(rows) if cfg.table_mode == "kv_pairs" else "\n".join(
                " | ".join(c or "" for c in r) for r in rows
            )
            if rendered:
                blocks.append(
                    Block(type="table", text=rendered, page=page_no + 1, meta={"rows": len(rows)})
                )

        cleaned = _clean(body, cfg.drop_header_footer)
        if cleaned and len(cleaned) >= cfg.min_text_len:
            blocks.append(Block(type="paragraph", text=cleaned, page=page_no + 1))
    doc.close()
    return blocks


def _ocr_page(page, cfg: ParserConfig) -> str:
    """扫描件走 OCR。中文场景 PaddleOCR 效果最好。"""
    from paddleocr import PaddleOCR

    engine = PaddleOCR(use_angle_cls=True, lang=cfg.ocr_lang, show_log=False)
    pix = page.get_pixmap(dpi=200)
    tmp = Path("/tmp/_ocr_page.png")
    tmp.write_bytes(pix.tobytes("png"))
    result = engine.ocr(str(tmp), cls=True)
    if not result or not result[0]:
        return ""
    return "\n".join(line[1][0] for line in result[0] if line and len(line) > 1)


def parse_file(path: Path, cfg: ParserConfig) -> ParsedDoc:
    """统一入口：按扩展名与配置分派到具体解析器。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            if cfg.engine == "docling":
                try:
                    blocks = _parse_docling(path, cfg)
                except Exception:
                    # Docling 失败时降级到 PyMuPDF，不要让单个文件中断整批入库
                    blocks = _parse_pymupdf(path, cfg)
            else:
                blocks = _parse_pymupdf(path, cfg)
        elif suffix in {".docx", ".doc"}:
            blocks = _parse_docx(path, cfg)
        elif suffix in {".html", ".htm", ".md"}:
            blocks = _parse_text_like(path, cfg)
        else:
            return ParsedDoc(
                doc_id=path.stem,
                source=str(path),
                blocks=[],
                parse_ok=False,
                fail_reason=f"unsupported format: {suffix}",
            )
    except Exception as exc:  # noqa: BLE001
        return ParsedDoc(
            doc_id=path.stem, source=str(path), blocks=[], parse_ok=False, fail_reason=str(exc)
        )

    if not blocks:
        return ParsedDoc(
            doc_id=path.stem, source=str(path), blocks=[], parse_ok=False, fail_reason="empty"
        )
    return ParsedDoc(doc_id=path.stem, source=str(path), blocks=blocks)


def _parse_docx(path: Path, cfg: ParserConfig) -> list[Block]:
    import docx

    document = docx.Document(str(path))
    blocks: list[Block] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if "heading" in style:
            level = 1
            m = re.search(r"(\d+)", style)
            if m:
                level = int(m.group(1))
            blocks.append(Block(type="heading", text=text, level=level))
        else:
            blocks.append(Block(type="paragraph", text=text))
    return blocks


def _parse_text_like(path: Path, cfg: ParserConfig) -> list[Block]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks: list[Block] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(Block(type="heading", text=stripped.lstrip("# ").strip(), level=level))
        else:
            blocks.append(Block(type="paragraph", text=stripped))
    return blocks
