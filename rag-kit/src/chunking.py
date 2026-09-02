"""切片层。

核心策略：父子 Chunk。
- 子 chunk（128 token）用于精确检索，定位准
- 父 chunk（768 token）用于给 LLM 提供完整上下文，避免答案被截断的片段带偏

一刀切 512 是最常见的错误：一段含两张表的正文被切散，问数字必错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import ChunkConfig
from parsers import Block, ParsedDoc


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    char_len: int
    meta: dict

    @property
    def citation(self) -> str:
        return f"{self.meta.get('source', self.doc_id)} · P{self.page}"


def _approx_tokens(text: str) -> int:
    """中英混排的粗估：中文按字计，英文按 4 字符 1 token。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + other // 4


def _split_by_length(text: str, size: int, overlap: int) -> list[str]:
    if _approx_tokens(text) <= size:
        return [text]
    pieces, buf = [], []
    for sent in re.split(r"(?<=[。！？；\n])", text):
        buf.append(sent)
        if _approx_tokens("".join(buf)) >= size:
            pieces.append("".join(buf))
            tail = "".join(buf)[-overlap * 4 :] if overlap else ""
            buf = [tail]
    if buf:
        pieces.append("".join(buf))
    return [p for p in pieces if p.strip()]


def build_chunks(doc: ParsedDoc, cfg: ChunkConfig) -> list[Chunk]:
    """把 ParsedDoc 转成 chunk 列表。表格整体成块，不参与文本切分。"""
    chunks: list[Chunk] = []
    seq = 0

    if cfg.strategy == "fixed":
        flat = "\n".join(b.text for b in doc.blocks)
        for piece in _split_by_length(flat, cfg.child_size, cfg.overlap):
            chunks.append(_mk(doc, seq, piece, 0, cfg, kind="flat"))
            seq += 1
        return chunks

    if cfg.strategy == "recursive":
        for blk in doc.blocks:
            for piece in _split_by_length(blk.text, cfg.parent_size, cfg.overlap):
                chunks.append(_mk(doc, seq, piece, blk.page, cfg, kind=blk.type))
                seq += 1
        return chunks

    # parent_child：按标题层级聚合正文，表格独立成块
    section: list[Block] = []
    current_page = 0

    def flush():
        nonlocal seq, section, current_page
        if not section:
            return
        parent_text = "\n".join(b.text for b in section)
        for piece in _split_by_length(parent_text, cfg.parent_size, cfg.overlap):
            for child in _split_by_length(piece, cfg.child_size, cfg.overlap):
                chunks.append(_mk(doc, seq, child, current_page, cfg, kind="child", parent=piece))
                seq += 1
        section, current_page = [], 0

    for blk in doc.blocks:
        if blk.type == "table":
            # 表格不切分，整块入库；否则跨页表格必然被切散
            chunks.append(_mk(doc, seq, blk.text, blk.page, cfg, kind="table"))
            seq += 1
            continue
        if blk.type == "heading" and cfg.respect_headings:
            flush()
            section.append(blk)
            current_page = blk.page
            continue
        if not current_page:
            current_page = blk.page
        section.append(blk)
    flush()

    return chunks


def _mk(
    doc: ParsedDoc,
    seq: int,
    text: str,
    page: int,
    cfg: ChunkConfig,
    kind: str,
    parent: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{seq}",
        doc_id=doc.doc_id,
        text=text.strip(),
        page=page,
        char_len=len(text),
        meta={
            "source": doc.source,
            "kind": kind,
            "strategy": cfg.strategy,
            "parent": parent,
        },
    )
