"""Agent 循环单测：mock 模型返回 tool_call 序列，不消耗任何 API 额度。

用标准库 unittest + asyncio.run，避免引入 pytest 等新依赖。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.config import ConfigError, Settings
from app.repositories.order_repo import OrderRepo
from app.repositories.knowledge_repo import Chunk, ScoredChunk
from app.schemas.chat import ChatRequest
from app.services.agent_service import AgentService
from app.services.guardrail_service import GuardrailService, GuardResult
from app.services.llm_gateway import ToolCall, ToolChatResult
from app.services.retrieval_service import RetrievalResult
from app.services.toolkit import Toolkit


def _settings(**overrides) -> Settings:
    """用临时合规词库构造 Settings，避免依赖真实数据文件。"""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "banned.yaml").write_text(
            "categories:\n  medical_claim:\n    level: block\n    terms: [治愈, 包治]\n"
            "sensitive_topics:\n  pregnancy:\n    terms: [孕妇]\n    desc: 孕期提示\n",
            encoding="utf-8",
        )
        base = dict(
            compliance_file=str(Path(td, "banned.yaml")),
            index_file=str(Path(td, "none.jsonl")),
            glossary_file=str(Path(td, "none.yaml")),
        )
        base.update(overrides)
        return Settings(**base)


class _FakeLLM:
    """按预设脚本依次返回 ToolChatResult。"""

    def __init__(self, script: list[ToolChatResult]) -> None:
        self.script = script
        self.calls = []

    async def chat_with_tools(self, messages, tools) -> ToolChatResult:
        self.calls.append(messages)
        return self.script[min(len(self.calls) - 1, len(self.script) - 1)]


def _search_result() -> RetrievalResult:
    chunk = Chunk("1", "01_regulation/x", "", "技术规范", "限量", "烟酰胺限量 5%")
    return RetrievalResult(hits=[ScoredChunk(chunk, 0.9, 0.8)], confidence=0.82)


def _call(name: str, args: dict, cid: str = "call_1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings(agent_enabled=True, llm_provider="openai_compatible", llm_api_key="x")
        self.retrieval = AsyncMock()
        self.retrieval.search.return_value = _search_result()
        self.toolkit = Toolkit(retrieval=self.retrieval, orders=OrderRepo())

    def _agent(self, script: list[ToolChatResult], max_steps: int = 6) -> AgentService:
        settings = _settings(
            agent_enabled=True, llm_provider="openai_compatible", llm_api_key="x",
            agent_max_steps=max_steps,
            compliance_file=self.settings.compliance_file,
            index_file=self.settings.index_file,
            glossary_file=self.settings.glossary_file,
        )
        return AgentService(settings, _FakeLLM(script), self.toolkit)

    def test_normal_loop_search_then_finish(self):
        """模型先检索、再 finish：循环两步结束，引用与轨迹都被收集。"""
        script = [
            ToolChatResult(tool_calls=[_call("search_knowledge", {"query": "烟酰胺限量"})]),
            ToolChatResult(tool_calls=[_call("finish", {"answer": "资料显示限量为5%。", "intent": "regulation"})]),
        ]
        outcome = asyncio.run(self._agent(script).run("烟酰胺限量多少", False, "t1"))

        self.assertEqual(outcome.draft, "资料显示限量为5%。")
        self.assertEqual(outcome.intent, "regulation")
        self.assertAlmostEqual(outcome.confidence, 0.82)
        self.assertEqual(len(outcome.hits), 1)
        self.assertEqual([s.tool for s in outcome.trajectory], ["search_knowledge", "finish"])
        self.retrieval.search.assert_awaited_once_with("烟酰胺限量")

    def test_step_limit_forces_human(self):
        """模型一直调检索不 finish：达到步数上限强制转人工，不会死循环。"""
        script = [ToolChatResult(tool_calls=[_call("search_knowledge", {"query": "烟酰胺"})])]
        outcome = asyncio.run(self._agent(script, max_steps=3).run("烟酰胺", False, "t2"))

        self.assertTrue(outcome.needs_human)
        self.assertIn("转人工", outcome.draft)
        self.assertEqual(len(outcome.trajectory), 3)

    def test_order_tool_reuses_repo(self):
        """订单工具真实调用 OrderRepo，返回物流信息。"""
        script = [ToolChatResult(tool_calls=[_call("query_order", {"order_no": "202611080001"})]),
                  ToolChatResult(tool_calls=[_call("finish", {"answer": "已发货。", "intent": "order_query"})])]
        outcome = asyncio.run(self._agent(script).run("订单 202611080001 到哪了", False, "t3"))
        self.assertIn("顺丰", outcome.trajectory[0].result_summary)
        self.assertEqual(outcome.intent, "order_query")

    def test_model_direct_text_is_final(self):
        """模型不调工具直接给文本：当作最终回复。"""
        script = [ToolChatResult(content="资料里没有这项信息。")]
        outcome = asyncio.run(self._agent(script).run("随便问问", False, "t4"))
        self.assertEqual(outcome.draft, "资料里没有这项信息。")


class ConfigGuardTests(unittest.TestCase):
    def test_agent_requires_openai_compatible(self):
        """开 Agent 却用抽取式模型：启动校验必须报错。"""
        s = _settings(agent_enabled=True, llm_provider="extractive")
        with self.assertRaises(ConfigError) as ctx:
            s.validate_runtime()
        self.assertIn("AGENT_ENABLED", str(ctx.exception))

    def test_default_off_passes(self):
        """默认配置校验通过，离线模式不受影响。"""
        s = _settings()
        s.validate_runtime()  # 不抛异常即通过


if __name__ == "__main__":
    unittest.main(verbosity=2)
