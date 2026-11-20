"""接口契约：请求 / 响应的唯一事实来源，前后端共用。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Intent = Literal[
    "material_info", "regulation", "technology", "order_query", "formulation", "other"
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000, description="用户原始提问")
    session_id: str | None = Field(default=None, max_length=64)
    user_id: str | None = Field(default=None, max_length=64)
    channel: Literal["seat_console", "mini_app", "wecom", "api"] = "api"


class Citation(BaseModel):
    source: str
    title: str
    section: str = ""
    snippet: str
    score: float


class ToolStep(BaseModel):
    """Agent 单次工具调用的观测记录，用于前端展示推理链与问题排查。"""

    tool: str
    arguments: dict = Field(default_factory=dict)
    result_summary: str = ""
    ok: bool = True


class FormulaItem(BaseModel):
    """配方草案中的单行原料。百分比由确定性引擎计算，不由模型生成。"""

    name: str
    category: str
    pct: float
    role: str = ""
    note: str = ""


class FormulaDraft(BaseModel):
    """结构化配方草案，由 build_formula 工具产出，前端据此渲染成分表。"""

    product_type: str
    product_label: str
    brief: str = ""
    items: list[FormulaItem] = Field(default_factory=list)
    total: float = 100.0
    rationale: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human: bool
    guard_flags: list[str] = Field(default_factory=list)
    trace_id: str
    trajectory: list[ToolStep] = Field(default_factory=list)
    formula: FormulaDraft | None = None


class IngestReport(BaseModel):
    files: int
    chunks: int
    index_file: str
