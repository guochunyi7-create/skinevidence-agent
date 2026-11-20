"""检索层：向量召回 + 关键词召回，用 RRF 融合，避免单一召回的盲区。

向量负责"说法不同但意思一样"（暗沉 vs 提亮），关键词负责"精确命中"
（SKU 编号、成分 INCI 名、色号），两者互补。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.core.config import Settings
from app.ingestion.glossary import Glossary
from app.repositories.knowledge_repo import KnowledgeRepo, ScoredChunk
from app.services.extractive_reader import _concepts, coverage
from app.services.llm_gateway import BaseEmbedder

logger = logging.getLogger(__name__)

# 法规类问题的信号词：命中后优先在《化妆品安全技术规范》里检索
_REGULATION_HINT = re.compile(r"限量|限值|上限|最大允许|禁用|限用|准用|技术规范|备案|注册证|国妆")

# 这些词太泛，几乎每份资料都有一节沾边（MSDS 的 "Exposure limits" 就含 limit），
# 拿它们做章节加权会把真正相关的文件挤下去
_GENERIC_SECTION_TERMS = {
    "limit", "limits", "limit value", "maximum", "use", "uses",
    "application", "function", "company", "manufacturer",
}


@dataclass
class RetrievalResult:
    hits: list[ScoredChunk] = field(default_factory=list)
    confidence: float = 0.0
    vector_top: float = 0.0
    keyword_top: float = 0.0
    coverage: float = 0.0

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


class RetrievalService:
    def __init__(
        self,
        repo: KnowledgeRepo,
        embedder: BaseEmbedder,
        settings: Settings,
        glossary: Glossary | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.settings = settings
        self.glossary = glossary

    async def search(self, query: str) -> RetrievalResult:
        pool = self.settings.retrieval_pool
        # 中文提问检索英文 MSDS 的关键一步：先按成分术语表扩展查询
        expanded = self.glossary.rewrite_query(query) if self.glossary else query
        vectors = await self.embedder.embed([expanded])
        vector_hits = await self.repo.search_vector(vectors[0], pool)
        keyword_hits = await self.repo.search_keyword(expanded, pool)

        vector_top = vector_hits[0].score if vector_hits else 0.0
        keyword_top = keyword_hits[0].score if keyword_hits else 0.0

        fused = self._rrf([vector_hits, keyword_hits])
        fused = self._constrain_entity(query, fused)
        fused = self._source_route(query, fused)
        fused = await self._section_route(query, fused)
        fused, coverage_value = self._rerank(query, fused)

        # 置信度以"问题里的概念被覆盖了多少"为主。
        # 只用向量相似度不行：余弦值随模型变化，且对"答案在哪一块"不敏感；
        # 覆盖率则是模型无关的——问到的实体和属性都出现在同一块里，才配回答。
        confidence = round(
            min(1.0, 0.65 * coverage_value + 0.2 * min(1.0, keyword_top) + 0.1 * max(0.0, vector_top)),
            4,
        )

        logger.info(
            "检索完成",
            extra={
                "extra_fields": {
                    "vector_hits": len(vector_hits),
                    "keyword_hits": len(keyword_hits),
                    "coverage": coverage_value,
                    "confidence": confidence,
                }
            },
        )
        return RetrievalResult(
            hits=fused[: self.settings.retrieval_top_k],
            confidence=confidence,
            vector_top=round(vector_top, 4),
            keyword_top=round(keyword_top, 4),
            coverage=coverage_value,
        )

    async def _section_route(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """把"章节名正好对上"的块直接拉进候选。

        MSDS 是标准十六节结构：问"危害"就该看第 3 节，问"毒理"就该看第 11 节。
        只靠相似度排序时，第 1 节（同义词、公司名）常常压过目标章节——因为
        它同样包含原料名。按章节路由比调相似度阈值更可靠。
        """
        if not self.glossary or not hits:
            return hits
        attr_terms = [
            term
            for zh, en in self.glossary.attributes
            if zh and zh in query
            for term in en
        ]
        if not attr_terms:
            return hits

        source = hits[0].chunk.source
        seen = {h.chunk.id for h in hits}
        extra: list[ScoredChunk] = []
        for term in dict.fromkeys(attr_terms):
            for chunk in await self.repo.by_section(source, term, limit=2):
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                extra.append(ScoredChunk(chunk, 0.9, 0.9))
        if extra:
            logger.info(
                "按章节路由补齐候选",
                extra={"extra_fields": {"source": source[:40], "added": len(extra)}},
            )
        return (extra + hits)[: self.settings.retrieval_pool]

    def _source_route(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """法规类问题优先在《技术规范》里找。

        "汞的限量"在原料 MSDS 第 8 节也有命中（职业接触限值），但用户问的是
        化妆品成品限量，答案只可能在法规里。按问题类型限定资料域，比调权重稳。
        """
        if not _REGULATION_HINT.search(query):
            return hits
        in_reg = [h for h in hits if h.chunk.source.startswith("01_regulation")]
        return in_reg if len(in_reg) >= 2 else hits

    def _constrain_entity(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """问题点名了某个原料时，只保留该原料的文件。

        「卡拉胶的储存条件」如果放任排序，很容易答成透明质酸钠的储存条件——
        因为所有 MSDS 的第 7 节文本几乎一样，区分它们的只有文件名。
        """
        if not self.glossary or not hits:
            return hits
        item = self.glossary.match_query(query)
        if item is None:
            return hits
        tokens = [t for t in [item.file_hint, *item.en, *item.all_zh] if t and len(t) >= 3]
        if not tokens:
            return hits
        kept = [
            h
            for h in hits
            if any(t.lower() in f"{h.chunk.title} {h.chunk.source}".lower() for t in tokens)
        ]
        if not kept:
            return hits
        logger.info(
            "命中原料实体，已收窄候选文件",
            extra={"extra_fields": {"ingredient": item.zh, "before": len(hits), "after": len(kept)}},
        )
        return kept

    def _rerank(self, query: str, hits: list[ScoredChunk]) -> tuple[list[ScoredChunk], float]:
        """按"该块覆盖了问题概念的多少"重排，并把这个覆盖率作为置信度主信号。"""
        concepts = _concepts(query, self.glossary)
        if not hits or not concepts:
            return hits, 0.0

        # 章节名直接点中属性（"3. Hazards Identification" 之于「危害」）是最强信号，
        # 它比正文里顺带提一句可靠得多，单独给一档加权。
        # 两个限制：只对"榜首那份文件"生效（否则 MSDS 的 "Exposure limits" 会把
        # 法规表格挤掉），并排除过于宽泛的词（limit / use / maximum）。
        attr_terms = [
            term.lower()
            for zh, en in (self.glossary.attributes if self.glossary else [])
            if zh and zh in query
            for term in en
            if term.lower() not in _GENERIC_SECTION_TERMS
        ]
        focus_source = hits[0].chunk.source

        scored: list[tuple[float, float, ScoredChunk]] = []
        for hit in hits:
            text = f"{hit.chunk.title} {hit.chunk.section} {hit.chunk.content}"
            cover = coverage(text, concepts)
            combined = 0.5 * min(1.0, hit.score * 6) + 0.5 * cover
            section_low = hit.chunk.section.lower()
            if (
                attr_terms
                and hit.chunk.source == focus_source
                and any(t in section_low for t in attr_terms)
            ):
                combined += 0.35
            scored.append((combined, cover, hit))
        scored.sort(key=lambda x: x[0], reverse=True)

        ordered = [ScoredChunk(h.chunk, h.score, h.raw_score) for _, _, h in scored]

        # 一个问题可能同时问两件事（"危害 + 毒理"），单块只覆盖一半；
        # 置信度看的是"这批候选合起来能不能答上"，所以取前几块的并集覆盖率
        head = " ".join(
            f"{h.chunk.title} {h.chunk.section} {h.chunk.content}" for h in ordered[:3]
        )
        union_cover = coverage(head, concepts)
        top_cover = max((c for _, c, _ in scored[:2]), default=0.0)
        return ordered, round(min(1.0, max(union_cover, top_cover)), 4)

    def _rrf(self, ranked_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        k = self.settings.rrf_k
        scores: dict[str, float] = {}
        raw: dict[str, float] = {}
        store: dict[str, ScoredChunk] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                scores[item.chunk.id] = scores.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
                raw[item.chunk.id] = max(raw.get(item.chunk.id, 0.0), item.score)
                store.setdefault(item.chunk.id, item)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            ScoredChunk(store[cid].chunk, round(score, 6), round(raw[cid], 4))
            for cid, score in ordered
        ]
