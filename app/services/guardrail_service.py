"""合规护栏：输入侧识别红线意图，输出侧拦截违规宣称。

原则：宁可拒答并转人工，也不能让 Agent 说出违反法规的话。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.core.config import Settings

logger = logging.getLogger(__name__)

SAFE_REFUSAL = (
    "这个问题涉及医疗或疾病相关表述，超出化妆品可以回答的范围，我不能给出功效结论。"
    "已为您转接人工顾问，请稍等。"
)
NO_KNOWLEDGE = (
    "产品资料里没有找到能支撑这个问题的信息。为了避免给您错误结论，建议转人工确认。"
)

LEVEL_BLOCK = "block"
LEVEL_REWRITE = "rewrite"
LEVEL_REVIEW = "review"
LEVEL_INFO = "info"


@dataclass
class GuardFlag:
    category: str
    level: str
    matched: list[str] = field(default_factory=list)
    desc: str = ""


@dataclass
class GuardResult:
    flags: list[GuardFlag] = field(default_factory=list)
    blocked: bool = False
    needs_human: bool = False
    disclaimers: list[str] = field(default_factory=list)

    @property
    def flag_names(self) -> list[str]:
        return [f"{f.category}:{f.level}" for f in self.flags]

    def merge(self, other: "GuardResult") -> "GuardResult":
        return GuardResult(
            flags=self.flags + other.flags,
            blocked=self.blocked or other.blocked,
            needs_human=self.needs_human or other.needs_human,
            disclaimers=list(dict.fromkeys(self.disclaimers + other.disclaimers)),
        )


_DISCLAIMERS = {
    "pregnancy": "孕期、哺乳期使用任何化妆品前，建议先咨询产科或皮肤科医生。",
    "children": "儿童使用请认准儿童化妆品“小金盾”标识，并在成人监护下使用。",
    "allergy": "建议先在耳后或手腕内侧做局部测试，24 小时无异常再全脸使用。",
}


class GuardrailService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._categories: dict[str, dict] = {}
        self._topics: dict[str, dict] = {}
        self._patterns: dict[str, re.Pattern[str]] = {}
        self._topic_patterns: dict[str, re.Pattern[str]] = {}
        self._load(Path(settings.compliance_file))

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"缺少合规词库：{path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._categories = data.get("categories", {})
        self._topics = data.get("sensitive_topics", {})
        for name, cfg in self._categories.items():
            self._patterns[name] = self._compile(cfg.get("terms", []))
        for name, cfg in self._topics.items():
            self._topic_patterns[name] = self._compile(cfg.get("terms", []))
        logger.info(
            "合规词库加载完成",
            extra={"extra_fields": {"categories": len(self._categories), "topics": len(self._topics)}},
        )

    @staticmethod
    def _compile(terms: list[str]) -> re.Pattern[str]:
        if not terms:
            return re.compile(r"(?!)")
        return re.compile("|".join(f"(?:{t})" for t in terms))

    def check_input(self, text: str) -> GuardResult:
        """用户提问侧：只识别敏感人群，不因用户提到医疗词就直接拒答。"""
        result = GuardResult()
        for name, pattern in self._topic_patterns.items():
            hits = {m.group(0) for m in pattern.finditer(text)}
            if hits:
                cfg = self._topics[name]
                result.flags.append(
                    GuardFlag(name, LEVEL_REVIEW, sorted(hits), cfg.get("desc", ""))
                )
                result.needs_human = True
                if name in _DISCLAIMERS:
                    result.disclaimers.append(_DISCLAIMERS[name])
        return result

    def check_output(self, text: str) -> GuardResult:
        """模型输出侧：block 级命中即整段拒答，rewrite 级删除命中句。"""
        result = GuardResult()
        for name, pattern in self._patterns.items():
            hits = {m.group(0) for m in pattern.finditer(text)}
            if not hits:
                continue
            cfg = self._categories[name]
            level = cfg.get("level", LEVEL_REVIEW)
            result.flags.append(GuardFlag(name, level, sorted(hits), cfg.get("desc", "")))
            if level == LEVEL_BLOCK:
                result.blocked = True
                result.needs_human = True
            elif level == LEVEL_REWRITE:
                result.needs_human = True
            elif level == LEVEL_INFO:
                # 特殊化妆品功效：只留痕备查，不打扰坐席
                pass
            else:
                result.needs_human = True
        return result

    def apply_output_guard(self, answer: str) -> tuple[str, GuardResult]:
        """返回经过清理的答案与护栏结果；block 级直接替换为安全话术。"""
        guard = self.check_output(answer)
        if guard.blocked:
            return SAFE_REFUSAL, guard
        cleaned = answer
        rewrite_terms = [t for f in guard.flags if f.level == LEVEL_REWRITE for t in f.matched]
        if rewrite_terms:
            cleaned = self._strip_sentences(cleaned, rewrite_terms)
            if not cleaned.strip():
                return SAFE_REFUSAL, guard
        return cleaned, guard

    @staticmethod
    def _strip_sentences(text: str, terms: list[str]) -> str:
        parts = re.split(r"(?<=[。！？\n])", text)
        kept = [p for p in parts if not any(t in p for t in terms)]
        return "".join(kept).strip()


def with_disclaimers(answer: str, guard: GuardResult) -> str:
    if not guard.disclaimers:
        return answer
    tail = "\n\n" + "\n".join(f"· {d}" for d in dict.fromkeys(guard.disclaimers))
    return answer + tail
