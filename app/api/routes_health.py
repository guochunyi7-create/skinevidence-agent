"""健康检查与知识库状态：上线后运维与发布流水线依赖这两个端点。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.deps import get_chat_service, get_knowledge_repo
from app.repositories.knowledge_repo import KnowledgeRepo
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/health", summary="存活探针")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="就绪探针：知识库可用才返回 200")
async def ready(repo: KnowledgeRepo = Depends(get_knowledge_repo)) -> dict:
    chunks = await repo.count()
    return {"status": "ready", "chunks": chunks}


@router.get("/api/v1/kb/status", summary="知识库概览")
async def kb_status(
    repo: KnowledgeRepo = Depends(get_knowledge_repo),
    service: ChatService = Depends(get_chat_service),
) -> dict:
    return {
        "chunks": await repo.count(),
        "brand": service.brand,
        "mode": "agent" if (service.settings.agent_enabled and service.agent is not None) else "rag",
        "llm_model": service.settings.llm_model if service.settings.agent_enabled else "extractive",
    }
