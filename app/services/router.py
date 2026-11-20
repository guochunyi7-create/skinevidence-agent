"""问题类型路由：区分「查已有成品配方表」与「研制新配方」。

两类问题长得像，但能力完全不同：
  - 查询已有配方表（"这个眼膜的成分表是什么"）：知识库只有法规、MSDS、技术资料，
    没有成品配方数据，检索会把"玻色因原料 MSDS"错当成答案（置信度虚高），
    因此在入口识别并给出统一、诚实的答复。
  - 研制新配方（"帮我做一个抗皱保湿眼霜配方"）：这是本系统支持的能力，
    放行给 Agent，由它调用配方工具选料、计算剂量、校验合规后产出草案。

判断顺序：研发动词白名单 > 原料属性词 > 成品配方查询词 > 默认 Agent。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings

# 是否已有成品配方/成分表数据源。补数据后改 True 即可让查询类问题也走检索。
FORMULA_DATA_AVAILABLE = False

FORMULA_ANSWER = (
    "资料库中目前没有收录成品的配方表或完整成分表，"
    "为避免给您错误结论，这部分信息需要由研发或法规同事确认，建议转人工处理。\n"
    "如果您是想「研制」一款新品，可以直接告诉我目标功效和剂型"
    "（例如「帮我做一个抗皱保湿眼霜配方」），我会基于原料库为您生成配方草案；"
    "想了解某个原料的安全资料、毒理或法规限量，我也可以继续查询。"
)

# 命中这些模式即判定为"问已有成品配方/成分表"
_FORMULA_PATTERNS = [
    re.compile(r"(配方|成分表|成分组成|全成分|配料表|加了什么|含有哪些成分|有哪些成分)"),
    re.compile(r"(成分|配方).{0,6}(是什么|有哪些|多少|表)"),
    re.compile(r"(眼膜|精华|面霜|产品).{0,8}(配方|成分表|成分组成)"),
]

# 这些是"问单个原料属性"，不算配方问题
_INGREDIENT_QUERY_HINT = re.compile(
    r"(毒理|危害|CAS|分子量|熔点|沸点|限量|最大允许|储存|运输|刺激性|安全|MSDS)"
)

# 研发动词白名单：出现这些词说明用户要"从零研制"，不是"查已有配方表"
# 放在最优先判断，避免被上面的"配方"关键词误伤
_DEVELOP_INTENT = re.compile(
    r"(研制|研发|开发|设计|生成|推荐|组方|打样|配方建议|配方草案|新配方|"
    r"帮我配|给我配|想配|要配|怎么配|如何配|怎么设计|如何设计|用什么原料|"
    r"做一个|做个|做一?款|配一个|配个|配一?版|出一?版|拟一?个|给一?个|"
    r"想做|要做|想开发|想研制|开发一?款|做一?个|出一?个)"
)


@dataclass
class RouteDecision:
    kind: str  # "formula" | "agent"
    answer: str = ""
    formulation: bool = False  # True 表示研制新配方，需强制走配方工具链


class QuestionRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def route(self, question: str) -> RouteDecision:
        if FORMULA_DATA_AVAILABLE:
            return RouteDecision(kind="agent")

        # 1. 研制新配方：最高优先级，直接放行给 Agent 的配方工具链
        if _DEVELOP_INTENT.search(question):
            return RouteDecision(kind="agent", formulation=True)

        # 2. 问单个原料属性：即使带"成分"二字也走 Agent
        if _INGREDIENT_QUERY_HINT.search(question):
            return RouteDecision(kind="agent")

        # 3. 问已有成品配方表：知识库无此数据，统一拒答
        if any(p.search(question) for p in _FORMULA_PATTERNS):
            return RouteDecision(kind="formula", answer=FORMULA_ANSWER)

        return RouteDecision(kind="agent")
