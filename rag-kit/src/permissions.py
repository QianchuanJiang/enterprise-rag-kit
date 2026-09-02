"""权限隔离与审计（样例 B 护城河）。

为什么这是壁垒：平台上 95% 的竞争者做出来的知识库是「所有人看到所有内容」，
这在金融、医疗、政企是上不了线的。本模块提供：

1. 分级密级：public / internal / restricted，写入 chunk 元信息
2. 检索层硬过滤：越权片段在检索阶段就被剔除，而非生成后「打个码」
3. 分级拒答：无依据 -> 拒答；有依据但越权 -> 「无权查看」，且不泄露内容是否存在
4. 全链路审计：谁、何时、问什么、召回什么、返回什么、耗时多少，追加写 + 哈希链防篡改

can_access 被 retriever.search 直接调用，保证 ACL 不可被上层绕过。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LEVEL_RANK = {"public": 0, "internal": 1, "restricted": 2}


def can_access(user_level: str, chunk_level: str) -> bool:
    """用户密级 >= 片段密级才可见。"""
    return LEVEL_RANK.get(user_level, 0) >= LEVEL_RANK.get(chunk_level, 0)


class AuditLogger:
    """追加写审计日志，每条记录带上一条的哈希，形成不可篡改链。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _prev_hash(self) -> str:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return lines[-1].split("|H:")[-1] if lines else "GENESIS"

    def log(self, record: dict) -> None:
        record.setdefault("ts", round(time.time(), 3))
        body = json.dumps(record, ensure_ascii=False)
        h = hashlib.sha256((self._prev_hash() + "|" + body).encode("utf-8")).hexdigest()[:16]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{body}|H:{h}\n")

    def verify(self) -> bool:
        """重算哈希链，任一条被篡改即返回 False。"""
        prev = "GENESIS"
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            body, _, h = ln.rpartition("|H:")
            calc = hashlib.sha256((prev + "|" + body).encode("utf-8")).hexdigest()[:16]
            if calc != h:
                return False
            prev = h
        return True

    def count(self) -> int:
        return sum(1 for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip())
