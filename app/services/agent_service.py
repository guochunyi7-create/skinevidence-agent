"""Agent 编排：模型自主决策 → 调工具 → 观察结果 → 继续，直到给出答案。

这一层只负责 ReAct 循环本身：驱动模型、执行工具、收集引用与轨迹。
合规护栏、免责声明、医疗话术、引用构造全部交回 ChatService 统一处理——
这样 Agent 路径与原 RAG 路径共用同一套后处理，护栏位置不因开关而改变。

所有循环都有硬上限：步数、单次工具超时，避免模型反复调用工具烧 token
或陷入死循环。本层不 import 任何 FastAPI 类型。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.repositories.knowledge_repo import ScoredChunk
from app.schemas.chat import ToolStep
from app.services.llm_gateway import ToolCapableLLM, ToolChatResult
from app.services.retrieval_service import RetrievalResult
from app.services.toolkit import Toolkit

logger = logging.getLogger(__name__)

_AGENT_SYSTEM = """你是「{brand}」的客服知识助手，可以自主调用工具查资料，再回答人工坐席的提问。

工作方式：
1. 涉及成分、限量、法规、毒理等问题，先用 search_knowledge 检索，依据检索结果作答；
2. 用户给出订单号（12 位数字）时用 query_order 查询，不要凭空描述物流状态；
3. 信息足够后调用 finish 结束，answer 写给用户看的中文草稿。

铁律：
- 只能依据工具返回的资料作答，资料没有就说没有，禁止编造成分、浓度、功效；
- 严禁医疗、疾病、治疗、药效类表述；
- 涉及孕妇、儿童、过敏体质，提示咨询医生或局部测试并建议转人工；
- 资料不足或检索置信度低时，在 answer 里明确说明并建议转人工；
- 回答控制在 150 字以内，先结论后依据。
"""

_FORMULATION_SYSTEM = """你是「{brand}」的化妆品配方研发助手，为人类研发人员生成**配方草案讨论稿**。

任务：根据用户需求研制配方。严格按以下流程调用工具，快速收敛：
1. 判断剂型与目标功效；剂型代码用 serum(精华)/cream(面霜)/lotion(乳液)/
   eye_cream(眼霜)/eye_mask(眼膜)/mask(面膜)/cleanser(洁面)/toner(化妆水)。
   用户说"眼霜"必须用 eye_cream，说"眼膜"才用 eye_mask。
2. 用 search_formulation_ingredients 按功效检索候选原料，最多检索 2 次。
   检索结果已含每种原料的用量区间、相态、配伍与注意事项，足以直接选料。
3. 通常无需逐个查询原料详情；仅在确实拿不准某个原料时才用
   get_formulation_ingredient，且最多用 1 次。
4. 检索后立即用 build_formula 生成草案，传入选定原料 name 列表和 product_type。
   用户明确指定某原料浓度时才填 pct，否则留空让引擎用常规值。
5. 调用 finish 结束，intent 填 formulation，answer 按下面的结构写。

【效率要求】查询类工具累计不超过 3 次，第 4 次调用必须是 build_formula。
不要在选料阶段反复查询，否则会耗尽步数而无法产出草案。

【answer 写作要求】
- 先一句话概括配方定位与适用剂型；
- 再用要点说明选料逻辑：为什么选这些活性物与基础原料（依据工具返回的功效标签）；
- 说明哪些组分是引擎按剂型骨架自动补齐的（保湿/防腐/螯合等）；
- 如果工具返回了风险提示（配伍、特殊化妆品、肤感偏差），逐条如实转述；
- 结尾必须写明：这是 AI 生成的草案讨论稿，用量为行业常规参考区间而非法规限值，
  须经研发与法规人员做稳定性、安全性、功效评价与备案核查后方可使用。

【铁律】
- 所有百分比只能来自 build_formula 的返回结果，绝不自行编造、修改或估算浓度；
- 严禁医疗、疾病、治疗、药效类表述；涉及美白祛斑等须提示特殊化妆品管理；
- 涉及孕妇、儿童、敏感肌，提示先做局部测试并转人工；
- 若用户要的原料在库里查不到，如实说明并建议补充资料，不得用近似原料冒充。
"""

_HUMAN_NOTE = "达到最大推理步数仍未得出结论，为避免猜测，已转人工处理。"
_EMPTY_NOTE = "本轮未能从模型获得有效回复，为避免猜测，已转人工处理。"
_FORMULA_GUARD_NOTE = "配方草案尚未生成，为避免凭空给出浓度，已转人工处理。"


@dataclass
class AgentOutcome:
    """Agent 循环的原始产物，交给 ChatService 做统一后处理。"""

    draft: str
    intent: str
    evidence: RetrievalResult | None
    trajectory: list[ToolStep] = field(default_factory=list)
    needs_human: bool = False
    formula: Any = None  # FormulaDraft，延迟导入避免循环；由 ChatService 透传给前端

    @property
    def hits(self) -> list[ScoredChunk]:
        return self.evidence.hits if self.evidence else []

    @property
    def confidence(self) -> float:
        return self.evidence.confidence if self.evidence else 0.0


class AgentService:
    def __init__(
        self,
        settings: Settings,
        llm: ToolCapableLLM,
        toolkit: Toolkit,
        brand: str = "示例品牌",
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.toolkit = toolkit
        self.brand = brand

    async def run(
        self,
        question: str,
        medical_inquiry: bool,
        trace_id: str,
        force_formula: bool = False,
    ) -> AgentOutcome:
        trajectory: list[ToolStep] = []
        self.toolkit.last_evidence = None  # 每轮重置，避免误引上一次的检索结果
        self.toolkit.last_formula = None

        system = (_FORMULATION_SYSTEM if force_formula else _AGENT_SYSTEM).format(brand=self.brand)
        if medical_inquiry and not force_formula:
            system += (
                "\n补充：用户提到疾病或医疗诉求，先说明化妆品不能替代药品，"
                "再只回答产品本身信息，并建议转人工。"
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        if force_formula:
            messages.append({
                "role": "system",
                "content": (
                    "执行约束：这是配方研制任务。你必须先调用 search_formulation_ingredients 选料，"
                    "再调用 build_formula 生成草案，最后调用 finish。"
                    "禁止直接输出文字答案，禁止自行写出任何百分比浓度。"
                ),
            })
        tools = self.toolkit.schemas()
        forced_build = False
        query_tools = {"search_formulation_ingredients", "get_formulation_ingredient"}

        for step in range(1, self.settings.agent_max_steps + 1):
            # 软强制：小模型容易在选料阶段反复查询而耗尽步数。
            # 查够 3 次就直接要求它进入草案生成，避免"还没开始做就结束"。
            if (
                force_formula and not forced_build and self.toolkit.last_formula is None
                and sum(1 for s in trajectory if s.tool in query_tools) >= 3
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "查询已经足够，请不要再查任何原料。现在就用前面检索到的原料中文名调用 "
                        "build_formula（product_type 填对应剂型代码）生成草案，然后调用 finish。"
                    ),
                })
                forced_build = True

            try:
                result: ToolChatResult = await self.llm.chat_with_tools(messages, tools)
            except UpstreamError:
                raise  # 上游故障交给全局错误处理器，不伪装成转人工

            # 配方任务：模型想跳过工具直接作答时，拦下来并引导它走配方工具链。
            # 这是安全约束——否则模型会用"看起来专业"的数字编造浓度。
            if force_formula and result.is_final and self.toolkit.last_formula is None:
                logger.info(
                    "配方任务拦截直接作答，引导调用工具",
                    extra={"extra_fields": {"trace_id": trace_id, "step": step}},
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "你还没有调用 build_formula 生成配方草案。请先调用 search_formulation_ingredients "
                        "选料，再调用 build_formula，最后调用 finish。不要直接回答，也不要自己写百分比。"
                    ),
                })
                continue

            if result.is_final:
                # 模型直接给文本（没调 finish 工具）是常见情况：glm-4-flash 之类
                # 的模型更倾向直接作答。此时 intent 与转人工判断不能丢，交由
                # 系统侧规则兜底——这属于"观测字段"，本就该由确定性代码负责，
                # 而不是赌模型一定按格式调用 finish。
                answer = result.content
                evidence = self.toolkit.last_evidence
                low_confidence = bool(evidence and evidence.confidence < self.settings.min_answer_score)
                logger.info(
                    "Agent 直接给出最终回复",
                    extra={"extra_fields": {"trace_id": trace_id, "step": step, "low_confidence": low_confidence}},
                )
                return AgentOutcome(
                    draft=answer,
                    intent="formulation" if force_formula else _infer_intent(question),
                    evidence=evidence,
                    trajectory=trajectory,
                    needs_human=not bool(answer) or low_confidence,
                    formula=self.toolkit.last_formula,
                )

            # 一轮可能有多个 tool_call，逐个执行并回填
            for call in result.tool_calls:
                step_record = ToolStep(tool=call.name, arguments=call.arguments)
                try:
                    outcome = await asyncio.wait_for(
                        self.toolkit.execute(call.name, call.arguments),
                        timeout=self.settings.agent_tool_timeout_s,
                    )
                except asyncio.TimeoutError:
                    outcome = None
                    step_record.ok = False
                    step_record.result_summary = "工具执行超时"

                step_record.result_summary = step_record.result_summary or _summarize(
                    outcome.text if outcome else ""
                )
                step_record.ok = step_record.ok and bool(outcome and outcome.ok)
                trajectory.append(step_record)

                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": outcome.text if outcome else "工具执行超时，请改用其他方式或转人工。",
                    }
                )

                if outcome and outcome.finished:
                    # 配方任务：finish 了却没生成草案，同样不放行
                    if force_formula and self.toolkit.last_formula is None:
                        messages.append({
                            "role": "user",
                            "content": (
                                "你调用了 finish 但尚未调用 build_formula，没有配方草案就不能给出浓度。"
                                "请改用 search_formulation_ingredients 与 build_formula 完成草案后再结束。"
                            ),
                        })
                        continue
                    if not outcome.final_answer:
                        return AgentOutcome(
                            draft=_EMPTY_NOTE,
                            intent="other",
                            evidence=self.toolkit.last_evidence,
                            trajectory=trajectory,
                            needs_human=True,
                            formula=self.toolkit.last_formula,
                        )
                    logger.info(
                        "Agent 正常结束",
                        extra={"extra_fields": {"trace_id": trace_id, "step": step, "tools": len(trajectory)}},
                    )
                    evidence = self.toolkit.last_evidence
                    low_confidence = bool(evidence and evidence.confidence < self.settings.min_answer_score)
                    return AgentOutcome(
                        draft=outcome.final_answer,
                        intent="formulation" if force_formula else (
                            outcome.final_intent
                            if outcome.final_intent != "other"
                            else _infer_intent(question)
                        ),
                        evidence=evidence,
                        trajectory=trajectory,
                        needs_human=low_confidence,
                        formula=self.toolkit.last_formula,
                    )

        # 跑满步数还没 finish：强制转人工，不允许模型带着半成品继续
        logger.warning("Agent 达到步数上限，转人工", extra={"extra_fields": {"trace_id": trace_id}})
        has_formula = self.toolkit.last_formula is not None
        return AgentOutcome(
            draft=_HUMAN_NOTE if has_formula else (
                _FORMULA_GUARD_NOTE if force_formula else _HUMAN_NOTE
            ),
            intent="formulation" if force_formula else "other",
            evidence=self.toolkit.last_evidence,
            trajectory=trajectory,
            needs_human=True,
            formula=self.toolkit.last_formula,
        )


def _infer_intent(question: str) -> str:
    """模型没给 intent 时，用与 RAG 路径相同的规则推断，保证两边口径一致。"""
    from app.services.chat_service import ChatService

    return ChatService._classify(question)


def _summarize(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")
