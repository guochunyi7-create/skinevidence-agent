"""类型化错误体系：客户端只能看到规范化结构，永远拿不到堆栈。"""

from __future__ import annotations


class AppError(Exception):
    code = "internal_error"
    status_code = 500
    message = "服务暂时不可用，请稍后重试"

    def __init__(self, message: str | None = None, *, detail: dict | None = None):
        self.message = message or self.message
        self.detail = detail or {}
        super().__init__(self.message)


class BadRequestError(AppError):
    code = "bad_request"
    status_code = 400
    message = "请求参数不合法"


class NotFoundError(AppError):
    code = "not_found"
    status_code = 404
    message = "未找到对应资源"


class IndexNotReadyError(AppError):
    code = "index_not_ready"
    status_code = 503
    message = "知识库尚未构建，请先执行入库脚本"


class UpstreamError(AppError):
    code = "upstream_error"
    status_code = 502
    message = "上游模型服务异常，请稍后重试"


class GuardrailRejectedError(AppError):
    """命中合规红线且无法安全回答时抛出，由会话层转换为转人工。"""

    code = "guardrail_rejected"
    status_code = 200
    message = "该问题涉及合规红线，已转人工处理"
