"""中英成分术语表：解决「中文提问检索英文 MSDS」的核心障碍。

两个方向都要通：
  入库时 —— 把中文名、别名写进该原料每个切块的表头，中文查询可字面命中
  检索时 —— 查询里出现中文成分名，扩展出英文/INCI 别名后再嵌入
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Ingredient:
    zh: str
    aliases: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    cas: str = ""
    function: str = ""
    file_hint: str = ""

    @property
    def all_zh(self) -> list[str]:
        return list(dict.fromkeys([self.zh] + self.aliases))

    def header(self) -> str:
        """写进切块表头的中文标识。"""
        return f"{self.zh}（{'/'.join(self.en[:2])}）"


class Glossary:
    def __init__(self, path: Path) -> None:
        self.ingredients: list[Ingredient] = []
        self.attributes: list[tuple[str, list[str]]] = []
        self.domain_keywords: list[str] = []
        self.substances: list[str] = []
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for row in data.get("attributes", []) or []:
                self.attributes.append((row.get("zh", ""), row.get("en", []) or []))
            self.domain_keywords = [str(k) for k in (data.get("domain_keywords") or [])]
            self.substances = [str(k) for k in (data.get("substances") or [])]
            for row in data.get("ingredients", []):
                self.ingredients.append(
                    Ingredient(
                        zh=row.get("zh", ""),
                        aliases=row.get("aliases", []) or [],
                        en=row.get("en", []) or [],
                        cas=str(row.get("cas", "") or ""),
                        function=row.get("function", "") or "",
                        file_hint=row.get("file_hint", "") or "",
                    )
                )
        else:
            logger.warning("术语表不存在，跨语言检索能力将受限", extra={"extra_fields": {"path": str(path)}})

    def match_document(self, title: str) -> Ingredient | None:
        """按 file_hint / 英文名 / 中文名匹配该文档对应的原料。"""
        probe = title.lower()
        for item in self.ingredients:
            if item.file_hint and item.file_hint.lower() in probe:
                return item
        for item in self.ingredients:
            for name in item.en + item.all_zh:
                token = name.lower()
                if len(token) >= 4 and token in probe:
                    return item
        return None

    def match_query(self, query: str) -> Ingredient | None:
        """查询里点名的原料（比 match_document 宽松：中文名 2 字即可）。

        「卡拉胶的储存条件」里"卡拉胶"只有 3 个字，按文件名的 4 字阈值会漏掉，
        结果答成别的原料的储存条件。
        """
        probe = query.lower()
        for item in self.ingredients:
            for name in item.all_zh:
                if len(name) >= 2 and name in query:
                    return item
            for name in item.en:
                if len(name) >= 4 and name.lower() in probe:
                    return item
        return None

    def expand_query(self, query: str) -> list[str]:
        """找出查询中提到的成分，返回需要追加的对照词汇。"""
        extra: list[str] = []
        low = query.lower()
        for item in self.ingredients:
            hit_zh = [a for a in item.all_zh if a and a in query]
            hit_en = [e for e in item.en if len(e) >= 3 and e.lower() in low]
            if hit_zh:
                extra.extend(item.en)
                if not hit_en:
                    extra.append(item.zh)
            elif hit_en:
                extra.extend(item.all_zh)
        return list(dict.fromkeys(t for t in extra if t))

    def is_in_domain(self, query: str) -> bool:
        """提问是否落在本知识库的业务范围内。"""
        low = query.lower()
        if any(k in query for k in self.domain_keywords):
            return True
        if self.expand_query(query) or self.expand_attributes(query):
            return True
        return any(k.lower() in low for k in self.domain_keywords if k.isascii())

    def expand_attributes(self, query: str) -> list[str]:
        """属性词跨语言扩展：问"分子量"要能命中英文的 Molecular Weight。"""
        extra: list[str] = []
        for zh, en in self.attributes:
            if zh and zh in query:
                extra.extend(en)
        return list(dict.fromkeys(extra))

    def rewrite_query(self, query: str) -> str:
        extra = self.expand_query(query) + self.expand_attributes(query)
        # 单字物质名单独追加：分词按二元组切，"化妆品中砷"只会切出"中砷"，
        # 没有独立的"砷"，倒排索引里就永远匹配不到"砷 2"这一行
        extra += [s for s in self.substances if s and s in query]
        extra = list(dict.fromkeys(extra))
        return f"{query} {' '.join(extra)}" if extra else query


def build_glossary(path: Path) -> Glossary:
    return Glossary(path)
