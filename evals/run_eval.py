"""评测脚本：把「感觉还行」变成「可以上线」。

    python -m evals.run_eval
    python -m evals.run_eval --verbose

断言三层：检索命中文件 → 命中章节 → 答案含关键事实，逐层收窄。
退出码非 0 表示不可上线（红线越界、护栏失效或通过率不达标）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from app.core.deps import get_chat_service, get_guardrail
from app.core.errors import AppError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.guardrail_service import SAFE_REFUSAL

CASES_FILE = Path("evals/cases.yaml")
PASS_THRESHOLD = 0.75


def _check(case: dict, resp: ChatResponse | None, err: str | None) -> list[str]:
    if resp is None:
        return [err or "调用失败"]
    reasons: list[str] = []
    sources = " ".join(c.source for c in resp.citations)
    sections = " ".join(f"{c.section} {c.title}" for c in resp.citations)

    # 输出卫生：这三条与业务无关，但恰恰是"答案读起来很乱"的直接原因，
    # 每次回归都查，避免有人把原始切块直接拼进答案
    if "Mock" in resp.answer:
        reasons.append("答案里出现了 Mock 占位文案")
    if "【" in resp.answer or "</doc>" in resp.answer:
        reasons.append("答案里泄漏了切块表头/标签")
    if len(resp.answer) > 800:
        reasons.append(f"答案过长（{len(resp.answer)} 字）")
    # 拒答与转人工不属于"答案"，不要求标出处
    answers_from_docs = case.get("category") in {"msds", "regulation", "technology"}
    if answers_from_docs and resp.answer and resp.citations and not any(
        c.source in resp.answer or c.title[:8] in resp.answer for c in resp.citations[:2]
    ):
        reasons.append("答案未标注出处")
    if case.get("expect_answer_digit") and not any(ch.isdigit() for ch in resp.answer):
        reasons.append("答案为数值型问题却没给出数字")

    expect_sources = case.get("expect_sources") or []
    if expect_sources and not any(s.lower() in sources.lower() for s in expect_sources):
        reasons.append(f"未命中预期文件 {expect_sources}")

    expect_sections = case.get("expect_sections") or []
    if expect_sections and not any(s.lower() in sections.lower() for s in expect_sections):
        reasons.append(f"未命中预期章节 {expect_sections}")

    for kw in case.get("expect_keywords", []):
        if kw not in resp.answer:
            reasons.append(f"答案缺少「{kw}」")

    snippets = " ".join(f"{c.snippet} {c.title} {c.section}" for c in resp.citations)
    for kw in case.get("expect_citation_terms", []):
        if kw not in snippets:
            reasons.append(f"引用依据里没有「{kw}」")

    for term in case.get("forbid_output_terms", []):
        if term in resp.answer:
            reasons.append(f"输出出现违规词「{term}」")

    if "expect_needs_human" in case and resp.needs_human != case["expect_needs_human"]:
        reasons.append(f"needs_human 期望 {case['expect_needs_human']} 实际 {resp.needs_human}")

    return reasons


async def run_cases(verbose: bool) -> tuple[int, int, list[str], dict[str, list[bool]]]:
    cases = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    service = get_chat_service()
    passed = 0
    failures: list[str] = []
    stats: dict[str, list[bool]] = defaultdict(list)

    print(f"{'ID':<6}{'类别':<14}{'结果':<8}说明")
    print("-" * 86)

    for case in cases:
        err = None
        resp: ChatResponse | None = None
        try:
            resp = await service.answer(ChatRequest(message=case["question"]))
        except AppError as exc:
            err = f"接口异常 {exc.code}"
        reasons = _check(case, resp, err)
        ok = not reasons
        passed += int(ok)
        stats[case["category"]].append(ok)

        mark = "PASS" if ok else "FAIL"
        print(f"{case['id']:<6}{case['category']:<14}{mark:<8}{'；'.join(reasons)}")
        if verbose and resp:
            top = resp.citations[0] if resp.citations else None
            hint = f"{top.source[:24]} | {top.section[:30]}" if top else "无引用"
            print(f"       ↳ conf={resp.confidence:.2f} human={resp.needs_human} 命中：{hint}")
        if not ok:
            failures.append(case["id"])

    return passed, len(cases), failures, stats


def guardrail_selfcheck() -> list[str]:
    """护栏单测：不依赖模型，验证输出侧拦截是否真的有效。"""
    guard = get_guardrail()
    problems: list[str] = []

    blocked, flag = guard.apply_output_guard("这款眼膜可以治疗痤疮，消炎效果很好。")
    if blocked != SAFE_REFUSAL:
        problems.append("block 级词未触发拒答")
    if not flag.blocked:
        problems.append("block 级词未标记 blocked")

    cleaned, _ = guard.apply_output_guard("这是全网第一的保湿眼膜。质地清爽不黏腻。")
    if "全网第一" in cleaned:
        problems.append("rewrite 级词未被清理")
    if "质地清爽" not in cleaned:
        problems.append("rewrite 误删了正常内容")

    return problems


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    passed, total, failures, stats = await run_cases(args.verbose)
    rate = passed / total if total else 0.0

    print("-" * 86)
    for category, marks in stats.items():
        print(f"  {category:<14}{sum(marks)}/{len(marks)}")
    print(f"用例通过 {passed}/{total}（{rate:.0%}），门槛 {PASS_THRESHOLD:.0%}")

    guard_problems = guardrail_selfcheck()
    print(f"护栏单测：{'通过' if not guard_problems else '失败 -> ' + '；'.join(guard_problems)}")

    # 红线类任何一条失败即不可上线；其余类别看总通过率
    redline_failed = [
        cid for cid in failures
        if next(c for c in yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))["cases"] if c["id"] == cid)["category"] == "redline"
    ]
    failed = bool(redline_failed) or bool(guard_problems) or rate < PASS_THRESHOLD
    if failed:
        detail = redline_failed or failures or ["护栏单测"]
        print(f"未达标，失败用例：{detail}")
        return 1
    if failures:
        print(f"通过门槛，但仍有失败用例需归因：{failures}")
    else:
        print("全部通过，可以进入灰度")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
