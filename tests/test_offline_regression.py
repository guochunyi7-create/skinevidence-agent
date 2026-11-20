"""离线回归：确认原 RAG 路径（agent=None）的回答结构不变。

测试直接构造显式关闭 Agent 的 ChatService，不依赖 .env 当前值——
这样即使 .env 切到 Agent 模式实测，回归测试依然稳定可复现。
"""

from __future__ import annotations

import asyncio
import unittest

from app.core.config import Settings
from app.core.deps import get_embedder, get_glossary, get_knowledge_repo
from app.repositories.order_repo import OrderRepo
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.guardrail_service import GuardrailService
from app.services.llm_gateway import ExtractiveLLM
from app.services.retrieval_service import RetrievalService


def _settings() -> Settings:
    return Settings(
        compliance_file="data/compliance/banned_claims.yaml",
        glossary_file="data/knowledge/glossary/ingredients.yaml",
        index_file="data/index/chunks.jsonl",
    )


class OfflineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = _settings()
        cls.svc = ChatService(
            settings=s,
            llm=ExtractiveLLM(),
            retrieval=RetrievalService(
                repo=get_knowledge_repo(),
                embedder=get_embedder(),
                settings=s,
                glossary=get_glossary(),
            ),
            guardrail=GuardrailService(s),
            orders=OrderRepo(),
            agent=None,  # 显式关闭 Agent，走原 RAG 路径
        )

    def test_offline_structure(self):
        """离线问答：核心字段齐全、trajectory 为空、无工具调用。"""
        resp = asyncio.run(
            self.svc.answer(ChatRequest(message="烟酰胺的限量是多少", channel="api"))
        )
        data = resp.model_dump()
        for key in (
            "answer", "intent", "citations", "confidence",
            "needs_human", "guard_flags", "trace_id", "trajectory",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["trajectory"], [], "离线模式不应有工具调用轨迹")
        self.assertIsInstance(data["answer"], str)
        self.assertTrue(data["answer"])

    def test_medical_question_turns_human(self):
        """医疗类提问仍走人工，护栏未被 Agent 改造绕过。"""
        resp = asyncio.run(
            self.svc.answer(ChatRequest(message="这个产品能治疗湿疹吗", channel="api"))
        )
        self.assertTrue(resp.needs_human)
