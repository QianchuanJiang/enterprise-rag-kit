"""生成层：强制引用溯源 + 无据拒答。

两个合规铁律（样例 B 的核心卖点）：
1. 每个结论必须带来源（doc · Pn），否则等于没做知识库
2. 资料里没有依据就明确拒答，绝不能让模型「编一个看起来对的」

三种后端：
- stub   : 抽取式生成（取与问题重叠最高的片段），离线演示/测试用，零依赖
- ollama : 本地 Ollama（vLLM 也走 OpenAI 兼容接口）
- openai : 云端 OpenAI 兼容接口
"""

from __future__ import annotations

import os
import time

from chunking import Chunk
from config import GuardConfig, LLMConfig

ANSWER_NO_CONTEXT = "抱歉，知识库中未找到相关依据，无法回答该问题。"
ANSWER_NO_PERM = "您当前权限不足以查看该内容。"


class Answer:
    def __init__(self, text: str, contexts: list[Chunk], refused: bool = False, reason: str = ""):
        self.text = text
        self.contexts = contexts
        self.refused = refused
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "citations": [c.citation for c in self.contexts],
            "refused": self.refused,
            "reason": self.reason,
        }

    @property
    def citations(self) -> list[str]:
        return [c.citation for c in self.contexts]


class Generator:
    def __init__(self, llm: LLMConfig, guard: GuardConfig):
        self.llm = llm
        self.guard = guard
        self._openai_client = None

    # ---------- 对外接口 ----------
    def generate(
        self, question: str, contexts: list[Chunk], refuse_if_empty: bool = True
    ) -> Answer:
        if not contexts:
            if refuse_if_empty:
                return Answer(ANSWER_NO_CONTEXT, [], refused=True, reason="no_context")
            return Answer("", [], refused=True, reason="no_context")

        if self.llm.provider == "stub":
            return self._stub(question, contexts)
        return self._llm(question, contexts)

    # ---------- stub 抽取式 ----------
    def _stub(self, question: str, contexts: list[Chunk]) -> Answer:
        # 检索层已把候选收窄到 top-k；抽取式生成从中挑最相关片段。
        # 用字符 bigram 重叠打分，能精确命中「营业收入」这类组合词，
        # 避免被公司名/年份等高频词抢走相关性（fake 向量下尤其明显）。
        best = max(contexts, key=lambda c: self._score(question, c.text))
        snippet = best.text[: self.guard.max_context_chars]
        return Answer(f"{snippet}\n\n（来源：{best.citation}）", [best])

    @staticmethod
    def _score(q: str, t: str) -> int:
        def tg(s: str) -> set[str]:
            # 字符 trigram：能命中「营业收入」这类唯一组合词，
            # 而「平安银行」等高频词不会与之冲突，相关性判定更稳。
            s = "".join(c for c in s if "一" <= c <= "鿿")
            return set(s[i : i + 3] for i in range(len(s) - 2))

        return len(tg(q) & tg(t))

    # ---------- 真实 LLM ----------
    def _get_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.llm.base_url, api_key=os.getenv(self.llm.api_key_env, "sk-no-key")
            )
        return self._openai_client

    def _chat_completion(self, sys_prompt: str, user: str) -> str:
        """调用真实 LLM；openai SDK 缺失时回退到 requests 直连 OpenAI 兼容接口。"""
        try:
            resp = self._get_client().chat.completions.create(
                model=self.llm.model,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content
        except ImportError:
            import requests

            url = self.llm.base_url.rstrip("/") + "/chat/completions"
            key = os.getenv(self.llm.api_key_env, "sk-no-key")
            resp = requests.post(
                url,
                json={
                    "model": self.llm.model,
                    "temperature": self.llm.temperature,
                    "max_tokens": self.llm.max_tokens,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user},
                    ],
                },
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _llm(self, question: str, contexts: list[Chunk]) -> Answer:
        ctx = "\n\n".join(
            f"[{i + 1}] ({c.citation}) {c.text}" for i, c in enumerate(contexts)
        )[: self.guard.max_context_chars]

        sys_prompt = (
            "你是企业知识库问答助手。请严格依据「资料」回答：\n"
            "1. 每个结论必须标注来源编号，如 [1]；\n"
            "2. 资料已为你检索出最相关片段，请优先从中提取答案；"
            "只有当给出的资料确实不包含问题所需信息时，才按第 3 条拒答；\n"
            "3. 若资料中没有依据，必须回答："
            f"「{ANSWER_NO_CONTEXT}」，不得编造；\n"
            "4. 语言简洁，不发散。"
        )
        user = f"资料：\n{ctx}\n\n问题：{question}"

        last_err = None
        for attempt in range(5):
            try:
                content = self._chat_completion(sys_prompt, user)
                return Answer(content, contexts)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                # 免费档常见限流（429）或瞬时网络抖动：退避后重试，保证演示不翻车
                msg = str(exc)
                if "429" in msg or "timeout" in msg.lower() or "overloaded" in msg.lower():
                    wait = 4 * (attempt + 1)
                    print(f"  [生成重试 {attempt + 1}/5] {msg[:60]}… 等待 {wait}s")
                    time.sleep(wait)
                    continue
                raise
        raise last_err  # type: ignore[misc]
