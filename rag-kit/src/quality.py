"""入库质量校验。

垃圾进垃圾出。上线前不清洗，问个数字答错率 30%。
这里定义「拒绝入库」的规则，并且必须在项目早期就跟客户方讲清楚——
否则客户方会问「为什么我的 500 份文档只进了 430 份」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chunking import Chunk


@dataclass
class RejectRule:
    name: str
    reason: str

    def check(self, c: Chunk) -> bool:
        """返回 True 表示命中规则，应被拒入。"""
        raise NotImplementedError


class TooShort(RejectRule):
    def __init__(self, min_chars: int = 20):
        super().__init__(name="too_short", reason=f"有效字符少于 {min_chars}")
        self.min_chars = min_chars

    def check(self, c: Chunk) -> bool:
        return len(c.text.strip()) < self.min_chars


class MostlyBlank(RejectRule):
    """空白字符占比过高，通常是解析失败或纯表格占位。"""

    def __init__(self, threshold: float = 0.5):
        super().__init__(name="mostly_blank", reason="空白字符占比过高")
        self.threshold = threshold

    def check(self, c: Chunk) -> bool:
        if not c.text:
            return True
        blank = sum(1 for ch in c.text if ch.isspace())
        return blank / len(c.text) > self.threshold


class HighDuplicate(RejectRule):
    """行重复率高，通常是页眉页脚残留或目录页。"""

    def __init__(self, threshold: float = 0.6):
        super().__init__(name="high_duplicate", reason="行重复率过高，疑似噪声")
        self.threshold = threshold

    def check(self, c: Chunk) -> bool:
        lines = [ln.strip() for ln in c.text.splitlines() if ln.strip()]
        if len(lines) < 4:
            return False
        return 1 - len(set(lines)) / len(lines) > self.threshold


class Garbled(RejectRule):
    """乱码检测：控制字符或异常字符占比过高，常见于 OCR 失败。"""

    def __init__(self, threshold: float = 0.3):
        super().__init__(name="garbled", reason="疑似乱码，OCR 或编码异常")
        self.threshold = threshold

    def check(self, c: Chunk) -> bool:
        if not c.text:
            return True
        bad = sum(1 for ch in c.text if ord(ch) < 32 and ch not in "\n\t")
        bad += len(re.findall(r"[\ufffd]", c.text))
        return bad / len(c.text) > self.threshold


DEFAULT_RULES = [TooShort(20), MostlyBlank(0.5), HighDuplicate(0.6), Garbled(0.3)]


@dataclass
class QualityReport:
    total: int
    accepted: list[Chunk]
    rejected: list[tuple[Chunk, str]]

    @property
    def accept_rate(self) -> float:
        return len(self.accepted) / self.total if self.total else 0.0

    def summary(self) -> dict:
        by_reason: dict[str, int] = {}
        for _, reason in self.rejected:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {
            "total": self.total,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "accept_rate": round(self.accept_rate, 4),
            "reject_reasons": by_reason,
        }


def validate(chunks: list[Chunk], rules: list[RejectRule] | None = None) -> QualityReport:
    rules = rules or DEFAULT_RULES
    accepted, rejected = [], []
    for c in chunks:
        hit = next((r for r in rules if r.check(c)), None)
        if hit:
            rejected.append((c, hit.reason))
        else:
            accepted.append(c)
    return QualityReport(total=len(chunks), accepted=accepted, rejected=rejected)
