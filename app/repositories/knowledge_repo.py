"""知识库仓储层：memory（本地/CI）与 pg（生产）两套实现，接口一致。"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.errors import IndexNotReadyError

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    source: str
    sku: str
    title: str
    section: str
    content: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    raw_score: float = 0.0


_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\.\-]{1,}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _terms(text: str) -> set[str]:
    """混合分词：英文/代码整词 + 中文二元组。

    只按双字切会把 "CAS" 拆成 "CA"+"AS"，在 MSDS 语料里几乎每篇都有，
    IDF 归零，问「CAS 号」就指不到第 1 节。英文必须按整词处理。
    """
    terms: set[str] = set()
    for token in _ASCII_TOKEN.findall(text):
        token = token.lower().strip(".-")
        if len(token) >= 2:
            terms.add(token)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _split_content(content: str) -> str:
    """关键词打分文本 = 章节名 + 正文。

    刻意丢掉【文件标题｜...】里的文件标题部分：同一文件的所有块带着相同标题，
    会让它们的得分被拉平（"毒性"和"运输"分不出来）；章节名是章节级信息，要保留
    （PPT 的中文主题标签就写在章节名里）。
    """
    if not content.startswith("【"):
        return content
    end = content.find("】")
    if end < 0:
        return content
    head = content[1:end]
    body = content[end + 1 :].lstrip("\n")
    # 只剥掉第一段文件标题，后面整段章节名保留（章节名本身可能含「｜」）
    section = head.split("｜", 1)[1] if "｜" in head else head
    return f"{section}\n{body}"


def load_index(path: Path) -> tuple[list[Chunk], np.ndarray]:
    if not path.exists():
        raise IndexNotReadyError(f"索引文件不存在：{path}，请先运行入库脚本")
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append(
                Chunk(
                    id=row["id"],
                    source=row.get("source", ""),
                    sku=row.get("sku", ""),
                    title=row.get("title", ""),
                    section=row.get("section", ""),
                    content=row["content"],
                )
            )
            vectors.append(row.get("embedding") or [])
    if not chunks:
        raise IndexNotReadyError("索引为空，请检查 data/knowledge 下是否有可解析文档")
    dims = {len(v) for v in vectors if v}
    if len(dims) != 1:
        raise IndexNotReadyError(f"向量维度不一致：{dims}，请重新入库")
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    return chunks, matrix


class KnowledgeRepo:
    """仓储接口，服务层只依赖这里声明的方法。"""

    async def search_vector(self, vector: list[float], pool: int) -> list[ScoredChunk]:
        raise NotImplementedError

    async def search_keyword(self, query: str, pool: int) -> list[ScoredChunk]:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError

    async def by_section(self, source: str, needle: str, limit: int = 3) -> list[Chunk]:
        """取同一份文件里章节名含 needle 的块（如 MSDS 的 Hazards 节）。"""
        raise NotImplementedError


class MemoryKnowledgeRepo(KnowledgeRepo):
    """进程内倒排索引 + 向量检索，零外部依赖，用于本地开发与单元测试。

    关键词侧刻意做了两件事：
      1. 打分只看正文，剥掉【标题｜章节】词头 —— 否则同一文件所有块得分相同
      2. 用 IDF 加权 —— 否则「化妆品」这类高频词会把真正的关键词（如「铅」）淹掉
    """

    def __init__(self, settings: Settings) -> None:
        self.chunks, self.matrix = load_index(Path(settings.index_file))
        self._bodies = [_split_content(c.content) for c in self.chunks]
        self._doc_grams: list[set[str]] = [_terms(b) for b in self._bodies]
        self._postings: dict[str, list[int]] = {}
        for idx, grams in enumerate(self._doc_grams):
            for g in grams:
                self._postings.setdefault(g, []).append(idx)
        logger.info(
            "内存索引加载完成",
            extra={
                "extra_fields": {
                    "chunks": len(self.chunks),
                    "dim": self.matrix.shape[1],
                    "terms": len(self._postings),
                }
            },
        )

    async def count(self) -> int:
        return len(self.chunks)

    async def by_section(self, source: str, needle: str, limit: int = 3) -> list[Chunk]:
        probe = needle.lower()
        return [
            c
            for c in self.chunks
            if c.source == source and probe in c.section.lower()
        ][:limit]

    async def search_vector(self, vector: list[float], pool: int) -> list[ScoredChunk]:
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scores = self.matrix @ query
        top = np.argsort(-scores)[:pool]
        return [
            ScoredChunk(self.chunks[i], float(scores[i]))
            for i in top
            if scores[i] > 0
        ]

    async def search_keyword(self, query: str, pool: int) -> list[ScoredChunk]:
        q_grams = _terms(query)
        if not q_grams:
            return []
        total = len(self.chunks)
        scores: dict[int, float] = {}
        denom = 0.0
        for gram in q_grams:
            docs = self._postings.get(gram)
            if not docs:
                continue
            idf = math.log(1.0 + total / len(docs))
            denom += idf
            for idx in docs:
                scores[idx] = scores.get(idx, 0.0) + idf
        if not scores or denom <= 0:
            return []
        # 先按查询词自身权重归一化（落到 0~1 量级），再做长度归一化，
        # 否则关键词得分（可到 5.0）会直接压过向量余弦（0~1）
        ranked = sorted(
            (
                (score / denom / math.sqrt(max(len(self._doc_grams[i]), 30) / 30.0), i)
                for i, score in scores.items()
            ),
            reverse=True,
        )
        return [ScoredChunk(self.chunks[i], min(s, 1.0)) for s, i in ranked[:pool]]


class PgKnowledgeRepo(KnowledgeRepo):
    """pgvector 实现，生产环境替换 memory 时不需要改动服务层代码。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.settings.database_url, min_size=1, max_size=8, init=self._init_conn
            )
        return self._pool

    @staticmethod
    async def _init_conn(conn) -> None:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def count(self) -> int:
        pool = await self._get_pool()
        return int(await pool.fetchval("SELECT count(*) FROM kb_chunk"))

    async def search_vector(self, vector: list[float], pool_size: int) -> list[ScoredChunk]:
        pool = await self._get_pool()
        vec = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
        rows = await pool.fetch(
            """
            SELECT id, source, sku, title, section, content, 1 - (embedding <=> $1::vector) AS score
            FROM kb_chunk
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec,
            pool_size,
        )
        return [
            ScoredChunk(
                Chunk(r["id"], r["source"], r["sku"], r["title"], r["section"], r["content"]),
                float(r["score"]),
            )
            for r in rows
        ]

    async def search_keyword(self, query: str, pool_size: int) -> list[ScoredChunk]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT id, source, sku, title, section, content,
                   ts_rank(to_tsvector('simple', title || ' ' || section || ' ' || content),
                           plainto_tsquery('simple', $1)) AS score
            FROM kb_chunk
            WHERE to_tsvector('simple', title || ' ' || section || ' ' || content)
                  @@ plainto_tsquery('simple', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query,
            pool_size,
        )
        return [
            ScoredChunk(
                Chunk(r["id"], r["source"], r["sku"], r["title"], r["section"], r["content"]),
                float(r["score"]),
            )
            for r in rows
        ]


    async def by_section(self, source: str, needle: str, limit: int = 3) -> list[Chunk]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT id, source, sku, title, section, content
            FROM kb_chunk
            WHERE source = $1 AND section ILIKE $2
            ORDER BY id
            LIMIT $3
            """,
            source,
            f"%{needle}%",
            limit,
        )
        return [
            Chunk(r["id"], r["source"], r["sku"], r["title"], r["section"], r["content"])
            for r in rows
        ]


def build_knowledge_repo(settings: Settings) -> KnowledgeRepo:
    if settings.vector_store == "pg":
        return PgKnowledgeRepo(settings)
    return MemoryKnowledgeRepo(settings)
