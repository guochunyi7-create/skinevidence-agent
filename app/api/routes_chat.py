"""会话接口：问答 + 按访客隔离的历史。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.deps import get_chat_service, get_history_store
from app.core.logging import log_event
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.history_store import HistoryStore, normalize_vid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


def _vid_from(request: Request, vid: str | None = None) -> str:
    """访客 ID 优先取 query，其次取 header；格式非法返回 400。"""
    raw = vid or request.headers.get("X-Visitor-Id") or ""
    try:
        return normalize_vid(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="visitor_id 非法")


@router.post("/chat", response_model=ChatResponse, summary="坐席助手问答")
async def chat(
    payload: ChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service),
    history: HistoryStore = Depends(get_history_store),
) -> ChatResponse:
    vid = _vid_from(request)
    result = await service.answer(payload)
    log_event(
        logger,
        "chat_completed",
        trace_id=result.trace_id,
        intent=result.intent,
        confidence=result.confidence,
        needs_human=result.needs_human,
        flags=result.guard_flags,
        channel=payload.channel,
        visitor=vid,
    )
    # 历史写入失败不影响问答主流程
    history.append(
        vid=vid,
        question=payload.message.strip(),
        answer=result.answer,
        intent=result.intent,
        response=result.model_dump(mode="json"),
        session_id=payload.session_id,
    )
    return result


@router.get("/history", summary="当前访客的会话列表")
async def list_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    vid: str | None = None,
    history: HistoryStore = Depends(get_history_store),
):
    visitor = _vid_from(request, vid)
    return {"sessions": history.sessions(visitor, limit), "stats": history.stats(visitor)}


@router.get("/history/{session_id}", summary="某个会话的完整记录")
async def get_history_detail(
    session_id: str,
    request: Request,
    vid: str | None = None,
    history: HistoryStore = Depends(get_history_store),
):
    visitor = _vid_from(request, vid)
    return {"session_id": session_id, "turns": history.session_detail(visitor, session_id)}


@router.delete("/history/{session_id}", summary="删除某个会话")
async def delete_history(
    session_id: str,
    request: Request,
    vid: str | None = None,
    history: HistoryStore = Depends(get_history_store),
):
    visitor = _vid_from(request, vid)
    ok = history.delete_session(visitor, session_id)
    return {"ok": ok}
