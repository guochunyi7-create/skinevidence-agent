"""真实调用智谱 GLM 的 Agent 实测脚本（会消耗 token）。

用法：
    python scripts/live_agent_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.deps import get_chat_service
from app.schemas.chat import ChatRequest


QUESTIONS = [
    ("知识检索", "烟酰胺在化妆品里的最大允许浓度是多少？"),
    ("订单查询", "帮我查一下订单 202611080001 的物流状态"),
    ("资料不足", "这个牌子今年双十一的销售额是多少？"),
]


async def main() -> None:
    svc = get_chat_service()
    print(f"Agent 模式: {svc.settings.agent_enabled} | 模型: {svc.settings.llm_model}\n")
    for tag, question in QUESTIONS:
        print("=" * 68)
        print(f"[{tag}] {question}")
        try:
            resp = await svc.answer(ChatRequest(message=question, channel="api"))
        except Exception as exc:
            print(f"  异常: {type(exc).__name__}: {exc}")
            continue
        print(f"  回答: {resp.answer}")
        print(f"  意图: {resp.intent} | 置信度: {resp.confidence} | 转人工: {resp.needs_human}")
        print(f"  引用数: {len(resp.citations)} | 护栏: {resp.guard_flags}")
        print("  工具轨迹:")
        for i, step in enumerate(resp.trajectory, 1):
            args = json.dumps(step.arguments, ensure_ascii=False)
            print(f"    {i}. {step.tool}({args}) ok={step.ok}")
            print(f"       → {step.result_summary[:120]}")
        print()
    await svc.llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
