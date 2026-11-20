"""Agent 工具集：把现有服务包装成模型可调用的工具。

工具层不含业务逻辑，只做三件事：
  1. 声明 JSON Schema，告诉模型有哪些工具、参数长什么样；
  2. 把模型的参数校验后转给现有 RetrievalService / OrderRepo；
  3. 把执行结果裁剪成简短文本回填给模型——原始命中块很长，直接塞回
     对话会迅速吃满上下文，且模型容易在长文本里迷失重点。

之所以把 finish 也做成工具：让"结束"成为一个显式动作而非靠猜模型语气，
便于统一收集 intent / citations，也便于统计轨迹。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.repositories.order_repo import OrderRepo
from app.schemas.chat import FormulaDraft
from app.services.formulation_service import FormulationService
from app.services.retrieval_service import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)

SEARCH_KNOWLEDGE = "search_knowledge"
QUERY_ORDER = "query_order"
SEARCH_FORMULATION_INGREDIENTS = "search_formulation_ingredients"
GET_FORMULATION_INGREDIENT = "get_formulation_ingredient"
BUILD_FORMULA = "build_formula"
FINISH = "finish"

# 剂型模板的展示名，供模型选择 product_type 时参考
_PRODUCT_TYPES = "serum(精华液)/cream(面霜)/lotion(乳液)/eye_cream(眼霜)/eye_mask(眼膜)/mask(面膜)/cleanser(洁面)/toner(化妆水)"

# 工具结果的文本截断上限：足够模型判断，又不至于撑爆上下文
_RESULT_CHAR_LIMIT = 1500
_HIT_CHAR_LIMIT = 420


@dataclass
class ToolOutcome:
    """工具执行结果。

    text 回填给模型；finished / final_* 供 AgentService 构造最终响应。
    """

    text: str
    finished: bool = False
    final_answer: str = ""
    final_intent: str = "other"
    evidence: RetrievalResult | None = None
    formula: FormulaDraft | None = None
    ok: bool = True


@dataclass
class Toolkit:
    retrieval: RetrievalService
    orders: OrderRepo
    formulation: FormulationService | None = None
    last_evidence: RetrievalResult | None = field(default=None, init=False)
    last_formula: FormulaDraft | None = field(default=None, init=False)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": SEARCH_KNOWLEDGE,
                    "description": (
                        "在产品资料、MSDS、法规文件中检索，回答成分、限量、毒理、"
                        "储存、法规等问题。不确定资料里有没有时先调用它，不要凭记忆回答。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "检索问题，尽量包含原料名与要查的属性，如「烟酰胺的限量」",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": QUERY_ORDER,
                    "description": "查询订单状态、物流、运单号。用户给出 12 位订单号时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_no": {"type": "string", "description": "12 位订单号"}
                        },
                        "required": ["order_no"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": SEARCH_FORMULATION_INGREDIENTS,
                    "description": (
                        "研制配方时按功效或功能查找候选原料，返回每种原料的常规用量区间、相态、"
                        "配伍注意与是否有 MSDS。选料前必须先调用它，不要凭记忆编造原料。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "benefit": {
                                "type": "string",
                                "description": "想要的功效，如 抗皱/保湿/舒缓/提亮/控油/修护/抗氧化/清洁",
                            },
                            "category": {
                                "type": "string",
                                "description": "功能分类（可选）：保湿剂/润肤剂/增稠剂/防腐剂/活性成分/舒缓成分/表面活性剂/乳化剂",
                            },
                            "query": {"type": "string", "description": "补充关键词，如原料名或质地要求（可选）"},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": GET_FORMULATION_INGREDIENT,
                    "description": "查询某个具体原料的常规用量区间、相态、工艺与配伍禁忌，确认后再纳入配方。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "原料中文名或英文名，如 玻色因 / Niacinamide"}
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": BUILD_FORMULA,
                    "description": (
                        "把选定原料组装成完整配方草案。引擎会自动补齐该剂型必需组分（保湿/防腐/螯合等）、"
                        "用水补足至 100%、校验用量区间与配伍冲突，返回结构化成分表。"
                        "pct 省略则用常规常用值；只有用户明确指定浓度时才填 pct。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "description": "选定的原料列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "原料中文名，须来自检索结果"},
                                        "pct": {"type": "number", "description": "用户明确指定的百分比，可选"},
                                    },
                                    "required": ["name"],
                                },
                            },
                            "product_type": {
                                "type": "string",
                                "description": f"剂型代码：{_PRODUCT_TYPES}",
                            },
                            "brief": {"type": "string", "description": "用户需求一句话概括，如 抗皱保湿眼膜"},
                        },
                        "required": ["items", "product_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": FINISH,
                    "description": (
                        "信息足够了就调用它结束，给出最终回复。调用前必须先检索到依据；"
                        "资料不足时直接说明并建议转人工，不要编造。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "给用户的中文回复草稿"},
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "material_info", "regulation", "technology",
                                    "order_query", "formulation", "other",
                                ],
                                "description": "本次问题的类别，配方研制填 formulation",
                            },
                        },
                        "required": ["answer", "intent"],
                    },
                },
            },
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if name == SEARCH_KNOWLEDGE:
            return await self._search(str(arguments.get("query") or ""))
        if name == QUERY_ORDER:
            return await self._order(str(arguments.get("order_no") or ""))
        if name == SEARCH_FORMULATION_INGREDIENTS:
            return self._search_formulation(
                benefit=str(arguments.get("benefit") or ""),
                category=str(arguments.get("category") or ""),
                query=str(arguments.get("query") or ""),
            )
        if name == GET_FORMULATION_INGREDIENT:
            return self._get_formulation(str(arguments.get("name") or ""))
        if name == BUILD_FORMULA:
            return self._build_formula(
                items=arguments.get("items") or [],
                product_type=str(arguments.get("product_type") or "serum"),
                brief=str(arguments.get("brief") or ""),
            )
        if name == FINISH:
            return ToolOutcome(
                text="已记录最终回复。",
                finished=True,
                final_answer=str(arguments.get("answer") or "").strip(),
                final_intent=str(arguments.get("intent") or "other"),
            )
        return ToolOutcome(text=f"未知工具：{name}", ok=False)

    async def _search(self, query: str) -> ToolOutcome:
        if not query.strip():
            return ToolOutcome(text="检索失败：query 不能为空。", ok=False)
        result = await self.retrieval.search(query)
        self.last_evidence = result  # finish 时用它生成引用，避免重复检索
        if not result.hits:
            return ToolOutcome(
                text=f"没有检索到与「{query}」相关的资料。请如实告知用户并建议转人工。",
                evidence=result,
            )
        lines = [f"检索到 {len(result.hits)} 条资料，置信度 {result.confidence:.2f}："]
        for idx, hit in enumerate(result.hits[:5], start=1):
            content = " ".join(hit.chunk.content.split())
            lines.append(
                f"[{idx}] {hit.chunk.title} / {hit.chunk.section}\n"
                f"{content[:_HIT_CHAR_LIMIT]}"
            )
        return ToolOutcome(text="\n".join(lines)[:_RESULT_CHAR_LIMIT], evidence=result)

    async def _order(self, order_no: str) -> ToolOutcome:
        if not order_no.strip():
            return ToolOutcome(text="查询失败：order_no 不能为空。", ok=False)
        try:
            info = await self.orders.get(order_no)
        except Exception as exc:
            return ToolOutcome(text=f"没有查到订单 {order_no}（{exc}）。请如实告知用户。", ok=False)
        return ToolOutcome(text=info.to_sentence())

    # ---- 配方研制工具 ----

    def _formulation(self) -> FormulationService | None:
        if self.formulation is None:
            logger.warning("配方引擎未装配，配方工具不可用")
        return self.formulation

    def _search_formulation(self, benefit: str, category: str, query: str) -> ToolOutcome:
        fs = self._formulation()
        if fs is None:
            return ToolOutcome(text="配方原料库暂不可用，请转人工。", ok=False)
        found = fs.search(
            query=query,
            category=category or None,
            benefit=benefit or None,
            limit=8,
        )
        return ToolOutcome(text=fs.format_candidates(found)[:_RESULT_CHAR_LIMIT])

    def _get_formulation(self, name: str) -> ToolOutcome:
        fs = self._formulation()
        if fs is None:
            return ToolOutcome(text="配方原料库暂不可用，请转人工。", ok=False)
        if not name.strip():
            return ToolOutcome(text="查询失败：name 不能为空。", ok=False)
        ing = fs.get(name)
        if ing is None:
            return ToolOutcome(text=f"原料库未收录「{name}」。请如实告知用户，不要编造该原料用量。")
        return ToolOutcome(text=ing.describe(fs.category_labels.get(ing.cat, ing.cat)))

    def _build_formula(self, items: list, product_type: str, brief: str) -> ToolOutcome:
        fs = self._formulation()
        if fs is None:
            return ToolOutcome(text="配方引擎未装配，无法生成草案，请转人工。", ok=False)
        if not isinstance(items, list) or not items:
            return ToolOutcome(text="生成失败：items 不能为空，请先选料。", ok=False)
        draft = fs.build(items=items, product_type=product_type, brief=brief)
        self.last_formula = draft  # finish 时随响应返回给前端
        lines = [
            f"已生成 {draft.product_label}配方草案，合计 {draft.total:.2f}%（已自动闭合到 100%）："
        ]
        for row in draft.items:
            lines.append(
                f"- {row.category} {row.name} {row.pct:g}%"
                + (f"（{row.role}）" if row.role else "")
            )
        if draft.rationale:
            lines.append("系统补齐：" + "；".join(draft.rationale))
        if draft.risk_notes:
            lines.append("风险提示：" + "；".join(draft.risk_notes))
        lines.append("依据：" + draft.disclaimer)
        return ToolOutcome(text="\n".join(lines)[:_RESULT_CHAR_LIMIT * 2], formula=draft)
