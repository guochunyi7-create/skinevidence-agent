"""依赖装配：单例在这里创建，路由只声明“我需要什么”。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

from app.core.config import Settings, get_settings
from app.ingestion.glossary import Glossary, build_glossary
from app.repositories.knowledge_repo import KnowledgeRepo, build_knowledge_repo
from app.repositories.order_repo import OrderRepo
from app.services.chat_service import ChatService
from app.services.guardrail_service import GuardrailService
from app.services.llm_gateway import BaseEmbedder, BaseLLM, ToolCapableLLM, build_embedder, build_llm
from app.services.retrieval_service import RetrievalService
from app.services.agent_service import AgentService
from app.services.formulation_service import FormulationService
from app.services.history_store import HistoryStore
from app.services.toolkit import Toolkit


@lru_cache
def get_llm() -> BaseLLM:
    return build_llm(get_settings(), get_glossary())


@lru_cache
def get_embedder() -> BaseEmbedder:
    return build_embedder(get_settings())


@lru_cache
def get_knowledge_repo() -> KnowledgeRepo:
    return build_knowledge_repo(get_settings())


@lru_cache
def get_guardrail() -> GuardrailService:
    return GuardrailService(get_settings())


@lru_cache
def get_orders() -> OrderRepo:
    return OrderRepo()


@lru_cache
def get_glossary() -> Glossary:
    return build_glossary(Path(get_settings().glossary_file))


@lru_cache
def get_retrieval() -> RetrievalService:
    settings: Settings = get_settings()
    return RetrievalService(
        get_knowledge_repo(), get_embedder(), settings, get_glossary()
    )


@lru_cache
def get_formulation() -> FormulationService:
    return FormulationService()


@lru_cache
def get_history_store() -> HistoryStore:
    return HistoryStore(get_settings().artifact_dir)


@lru_cache
def get_agent() -> AgentService | None:
    """仅当 Agent 开关打开且模型支持工具调用时装配。

    配置校验已保证 agent_enabled=True 时 llm_provider=openai_compatible，
    这里再做一次 isinstance 判断属于防御性编程——即使将来校验规则变化，
    也不会把一个不具备 chat_with_tools 的模型传进 Agent 循环。
    """
    settings = get_settings()
    if not settings.agent_enabled:
        return None
    llm = get_llm()
    if not isinstance(llm, ToolCapableLLM):
        logger.warning("模型不支持工具调用，Agent 未装配")
        return None
    return AgentService(
        settings=settings,
        llm=llm,
        toolkit=Toolkit(
            retrieval=get_retrieval(), orders=get_orders(), formulation=get_formulation()
        ),
        brand=settings.app_name,
    )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        settings=get_settings(),
        llm=get_llm(),
        retrieval=get_retrieval(),
        guardrail=get_guardrail(),
        orders=get_orders(),
        agent=get_agent(),
    )


async def shutdown() -> None:
    await get_llm().aclose()
    await get_embedder().aclose()
