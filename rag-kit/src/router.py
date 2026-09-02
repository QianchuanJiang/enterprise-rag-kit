"""分层调度（样例 D 加分项）：规则 -> 7B -> 72B 三级路由。

复刻您简历里的「规则引擎 + 轻模型 7B + 重模型 72B 三层分工」：
- L0 规则引擎：格式化查询、精确匹配、模板化回答，成本几乎为零
- L1 轻模型 7B：意图识别、Query 改写、简单问答、结果汇总，成本低
- L2 重模型 72B：复杂推理、多跳检索、最终生成，成本高

目标（写进验收指标）：
- L2 调用占比 <= 30%
- 综合成本下降 >= 50%
- 准确率损失 <= 2%

真实部署时，不同层级对应不同 model（见 config.llm），Router 只决定走哪一级。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteResult:
    level: int  # 0 / 1 / 2
    tier: str  # L0 / L1 / L2
    reason: str
    cost_tier: str  # free / low / high


# L0 规则：命中即直接返回，不进模型。按业务补充。
_RULES = [
    ("状态查询", ["状态", "查一下", "进度", "到哪了"], "rule_status"),
    ("问候", ["你好", "您好", "hi", "hello"], "rule_greet"),
    ("帮助", ["帮助", "能做什么", "怎么用"], "rule_help"),
]


class Router:
    def __init__(self, l0_threshold: float = 0.4, l2_threshold: float = 0.55):
        self.l0 = l0_threshold
        self.l2 = l2_threshold

    def route(self, question: str, complexity: float | None = None) -> RouteResult:
        q = question.lower()
        for name, kws, tag in _RULES:
            if any(k in q for k in kws):
                return RouteResult(0, "L0", f"命中规则：{name}", "free")

        comp = complexity if complexity is not None else self._estimate(question)
        if comp >= self.l2:
            return RouteResult(2, "L2", "复杂推理/多跳", "high")
        return RouteResult(1, "L1", "常规问答/改写", "low")

    @staticmethod
    def _estimate(q: str) -> float:
        score = 0.0
        score += min(len(q) / 200.0, 0.3)
        score += 0.15 * min(q.count("？") + q.count("?"), 2)
        for w in ["为什么", "如何", "比较", "区别", "原因", "多", "综合", "分析", "推导", "影响"]:
            if w in q:
                score += 0.1
        return min(score, 1.0)
