"""会话编排：护栏 → 意图路由 → 工具/检索 → 生成 → 输出护栏。

这一层是业务逻辑所在，不 import 任何 FastAPI 类型，便于单元测试。
"""

from __future__ import annotations

import logging
import re
import uuid

from app.core.config import Settings
from app.repositories.order_repo import OrderRepo
from app.repositories.knowledge_repo import ScoredChunk
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.services.guardrail_service import (
    NO_KNOWLEDGE,
    GuardrailService,
    with_disclaimers,
)
from app.services.llm_gateway import BaseLLM
from app.services.retrieval_service import RetrievalService
from app.services.agent_service import AgentOutcome, AgentService
from app.services.router import QuestionRouter

logger = logging.getLogger(__name__)

_ORDER_NO_RE = re.compile(r"\b(20\d{10})\b")
_MEDICAL_NOTE = "以上仅为产品信息说明，不构成医疗建议；如有皮肤疾病请及时就医。"
# 问到疾病、药效、「有没有激素」这类问题，坐席需要人来回复：
# 化妆品既不能承诺疗效，也不能替品牌背书"绝对不含某成分"
_MEDICAL_POLICY = (
    "这个问题涉及疾病或药效判断，超出化妆品可以承诺的范围，我不能给出结论。\n"
    "能提供的是：原料资料与《化妆品安全技术规范》的原文依据（见右侧引用），"
    "以及该批次的实际检测报告。\n建议转人工，由质量或法规同事确认后回复。"
)


def _esc(value: str) -> str:
    """资料属性要进 XML 风格标签，引号不转义会把标签拆坏（文件名里常见引号）。"""
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("order_query", re.compile(r"订单|物流|快递|发货|签收|运单|到货|什么时候到")),
    # 顺序即优先级：法规 > 技术资料 > 原料信息
    ("regulation", re.compile(r"限量|禁用|限用|准用|技术规范|规范|备案|注册证|国妆|法规|检测方法|最大允许浓度")),
    ("technology", re.compile(r"电子皮肤|促渗|离子导入|临床|功效评价|专利|认证|拉曼|生物电|膜")),
    ("material_info", re.compile(r"毒理|危害|CAS|分子量|熔点|沸点|闪点|密度|储存|运输|组成|成分|原料|刺激性")),
]

SYSTEM_PROMPT = """你是「{brand}」的资深客服知识助手，为人工坐席生成回复草稿。

必须遵守的铁律：
1. 只能依据下方「检索到的资料」作答。资料没写的内容，直接说“资料里没有这项信息”，禁止推测、禁止编造成分或浓度。
2. 严禁任何医疗、疾病、治疗、药效类表述。化妆品不是药品，不得暗示可治疗皮肤疾病。
3. 涉及孕妇、哺乳期、儿童、过敏体质时，必须提示咨询医生或先做局部测试，并建议转人工。
4. 功效表述要保守：只能复述资料原文的功效描述，不得夸大或延伸。
5. 答案用口语中文，先给结论再给依据，控制在 150 字以内，不要罗列无关内容。
{extra}

检索到的资料：
{docs}"""


class ChatService:
    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM,
        retrieval: RetrievalService,
        guardrail: GuardrailService,
        orders: OrderRepo,
        brand: str = "示例品牌",
        agent: AgentService | None = None,
        router: QuestionRouter | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.retrieval = retrieval
        self.guard = guardrail
        self.orders = orders
        self.brand = brand
        self.agent = agent
        self.router = router or QuestionRouter(settings)

    async def answer(self, req: ChatRequest) -> ChatResponse:
        trace_id = uuid.uuid4().hex[:16]
        text = req.message.strip()

        # 1. 输入护栏：敏感人群提示 + 医疗类提问识别
        topic_guard = self.guard.check_input(text)
        input_risk = self.guard.check_output(text)
        medical_inquiry = any(f.category in {"medical", "disease"} for f in input_risk.flags)

        # 1.3 问题类型路由：当前知识库没有成品配方表，这类问题给出统一答复，
        # 避免 Agent 与离线模式各说各话、把原料 MSDS 硬凑成"成分表"。
        decision = self.router.route(text)
        if decision.kind == "formula":
            cleaned, output_guard = self.guard.apply_output_guard(decision.answer)
            merged = topic_guard.merge(output_guard)
            final = with_disclaimers(cleaned, merged)
            return self._response(
                final, "material_info", merged, 0.0, True, trace_id,
                query=text, blocked=output_guard.blocked,
            )

        # 1.5 Agent 分流：开关打开且装配了 Agent 时，交给模型自主决策。
        # 研制新配方时 force_formula=True，Agent 会强制走配方工具链。
        # 输入护栏已在上面执行，输出护栏在 _from_agent 里执行——位置不变。
        if self.settings.agent_enabled and self.agent is not None:
            outcome = await self.agent.run(
                text, medical_inquiry, trace_id, force_formula=decision.formulation
            )
            return self._from_agent(outcome, topic_guard, medical_inquiry, trace_id, text)

        # 2. 意图路由（原 RAG 路径，以下逻辑保持不变）
        intent = self._classify(text)

        # 3. 走工具：订单类不查知识库
        if intent == "order_query":
            order_no = self._extract_order_no(text)
            if order_no:
                try:
                    info = await self.orders.get(order_no)
                except Exception:
                    answer = f"没有查到订单 {order_no}，请核对订单号，或转人工帮您查询。"
                    return self._response(answer, intent, topic_guard, 0.4, True, trace_id)
                return self._response(
                    info.to_sentence() + "\n\n请确认信息是否与您看到的一致。",
                    intent,
                    topic_guard,
                    0.9,
                    False,
                    trace_id,
                )

        # 4. 检索
        result = await self.retrieval.search(text)
        in_domain = self.retrieval.glossary.is_in_domain(text) if self.retrieval.glossary else True
        # 超范围提问（如"公司营收"）即使字面命中法规，也不该硬答
        off_scope = (not in_domain) and result.confidence < 0.45
        if not result.hits or result.confidence < self.settings.min_answer_score or off_scope:
            logger.info(
                "检索置信度不足，转人工",
                extra={"extra_fields": {"confidence": result.confidence, "query_len": len(text)}},
            )
            return self._response(
                NO_KNOWLEDGE, intent, topic_guard, result.confidence, True, trace_id, result.hits
            )

        # 5. 生成
        extra = (
            "\n6. 用户提到疾病或医疗诉求：先说明化妆品不能替代药品，再只回答产品本身的信息。"
            if medical_inquiry
            else ""
        )
        system = SYSTEM_PROMPT.format(brand=self.brand, extra=extra, docs=self._render_docs(result.hits))
        draft = (await self.llm.complete(system, text)).strip()

        # 抽取式引擎找不到可引用的原文时返回空串：这比硬凑一段更像"没有依据"
        if not draft:
            logger.info(
                "检索有命中但抽不出可引用原文，转人工",
                extra={"extra_fields": {"confidence": result.confidence}},
            )
            return self._response(
                NO_KNOWLEDGE, intent, topic_guard, result.confidence, True, trace_id, result.hits, query=text
            )

        # 6. 输出护栏
        cleaned, output_guard = self.guard.apply_output_guard(draft)
        merged = topic_guard.merge(output_guard)
        if medical_inquiry:
            # 涉及疾病/药效时，低置信度的资料摘录只会误导坐席，直接给话术并转人工
            if result.confidence < 0.6:
                return self._response(
                    _MEDICAL_POLICY,
                    intent,
                    merged,
                    result.confidence,
                    True,
                    trace_id,
                    result.hits,
                    query=text,
                )
            cleaned = f"{cleaned}\n\n{_MEDICAL_NOTE}"
        final = with_disclaimers(cleaned, merged)

        return self._response(
            final,
            intent,
            merged,
            result.confidence,
            merged.needs_human or medical_inquiry,
            trace_id,
            result.hits,
            output_guard.blocked,
            query=text,
        )

    def _from_agent(
        self,
        outcome: AgentOutcome,
        topic_guard,
        medical_inquiry: bool,
        trace_id: str,
        query: str,
    ) -> ChatResponse:
        """Agent 路径的后处理：与原 RAG 路径共用同一套护栏、免责声明与引用构造。"""
        cleaned, output_guard = self.guard.apply_output_guard(outcome.draft)
        merged = topic_guard.merge(output_guard)
        needs_human = merged.needs_human or medical_inquiry or outcome.needs_human
        blocked = output_guard.blocked

        if medical_inquiry and outcome.confidence < 0.6:
            cleaned = _MEDICAL_POLICY
            needs_human = True
        final = with_disclaimers(cleaned, merged)

        return self._response(
            final,
            outcome.intent,
            merged,
            outcome.confidence,
            needs_human,
            trace_id,
            outcome.hits,
            blocked,
            query=query,
            trajectory=outcome.trajectory,
            formula=outcome.formula,
        )

    @staticmethod
    def _classify(text: str) -> str:
        if _ORDER_NO_RE.search(text):
            return "order_query"
        for intent, pattern in _INTENT_RULES:
            if pattern.search(text):
                return intent
        return "other"

    @staticmethod
    def _extract_order_no(text: str) -> str | None:
        match = _ORDER_NO_RE.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _render_docs(hits: list[ScoredChunk]) -> str:
        blocks = []
        for hit in hits:
            c = hit.chunk
            attrs = {
                "sku": c.sku,
                "title": c.title,
                "section": c.section,
                "source": c.source,
            }
            rendered = " ".join(
                f'{k}="{_esc(v)}"' for k, v in attrs.items() if v
            )
            blocks.append(f"<doc {rendered}>\n{c.content}\n</doc>")
        return "\n".join(blocks)

    @staticmethod
    def _snippet(content: str, query: str, width: int = 150) -> str:
        """引用片段应以问题相关处为中心，而不是永远截头 120 字。

        「CAS 号是多少」这类问题，答案往往在第 300 字之后，截头就看不到依据了。
        """
        tokens = [t for t in re.split(r"[\s，。？、；：,?]+", query) if len(t) >= 2]
        positions = [content.find(t) for t in tokens]
        positions = [p for p in positions if p >= 0]
        if not positions:
            return content[:width]
        start = max(0, min(positions) - 30)
        end = min(len(content), start + width)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(content) else ""
        return f"{prefix}{content[start:end]}{suffix}"

    @classmethod
    def _citations(cls, hits: list[ScoredChunk], query: str = "", limit: int = 3) -> list[Citation]:
        return [
            Citation(
                source=h.chunk.source,
                title=h.chunk.title,
                section=h.chunk.section,
                snippet=cls._snippet(h.chunk.content, query),
                score=max(h.raw_score, 0.0),
            )
            for h in hits[:limit]
        ]

    def _response(
        self,
        answer: str,
        intent: str,
        guard,
        confidence: float,
        needs_human: bool,
        trace_id: str,
        hits: list[ScoredChunk] | None = None,
        blocked: bool = False,
        query: str = "",
        trajectory: list | None = None,
        formula=None,
    ) -> ChatResponse:
        flags = list(guard.flag_names)
        if blocked:
            flags.append("output_blocked")
        return ChatResponse(
            answer=answer,
            intent=intent,  # type: ignore[arg-type]
            citations=self._citations(hits or [], query),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            needs_human=needs_human,
            guard_flags=flags,
            trace_id=trace_id,
            trajectory=trajectory or [],
            formula=formula,
        )
