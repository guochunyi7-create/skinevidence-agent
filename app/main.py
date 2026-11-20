"""应用入口：中间件、异常处理、路由装配。"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_chat, routes_health
from app.core import deps
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import request_id_ctx, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "服务启动",
        extra={
            "extra_fields": {
                "env": settings.env,
                "llm_provider": settings.llm_provider,
                "embed_provider": settings.embed_provider,
                "vector_store": settings.vector_store,
                "agent_enabled": settings.agent_enabled,
            }
        },
    )
    try:
        chunks = await deps.get_knowledge_repo().count()
        logger.info("知识库就绪", extra={"extra_fields": {"chunks": chunks}})
    except Exception as exc:  # 知识库没建好也要能起来，/ready 会返回失败
        logger.warning("知识库未就绪：%s", exc)
    yield
    await deps.shutdown()
    logger.info("服务已停止")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="化妆品客服坐席知识助手：RAG + 合规护栏 + 工具调用",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*", "X-Visitor-Id"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "http_request",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "elapsed_ms": elapsed_ms,
                    }
                },
            )
            request_id_ctx.reset(token)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning(
            "业务异常",
            extra={"extra_fields": {"code": exc.code, "path": request.url.path}},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": exc.code, "message": exc.message, "detail": exc.detail},
                "request_id": request_id_ctx.get(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "请求参数不合法",
                    "detail": [
                        {"field": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
                        for e in exc.errors()
                    ],
                },
                "request_id": request_id_ctx.get(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("未捕获异常", extra={"extra_fields": {"path": request.url.path}})
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "internal_error", "message": "服务暂时不可用，请稍后重试"},
                "request_id": request_id_ctx.get(),
            },
        )

    app.include_router(routes_health.router)
    app.include_router(routes_chat.router)

    # 坐席侧栏（同源托管，避免 file:// 打开时的 CORS 问题）
    console_dir = Path(__file__).resolve().parents[1] / "console"
    if console_dir.exists():
        app.mount("/console", StaticFiles(directory=str(console_dir), html=True), name="console")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/console/")

    return app


app = create_app()
