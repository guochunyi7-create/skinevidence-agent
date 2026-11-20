"""切块：按「一份文档 × 一个章节」切，长文再按句子边界二次切分。

不同资料用不同粒度：
  法规   —— 单页信息密度高（表格多），块要大，否则一条限量要求被切散
  MSDS   —— 一个十六节标题下的内容就是天然答案单元
  演示稿 —— 一页一个主题，块要小，避免串味
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.ingestion.glossary import Glossary
from app.ingestion.loader import RawDoc

_SENT_RE = re.compile(r"(?<=[。！？；\n])")

PROFILE_PARAMS: dict[str, tuple[int, int]] = {
    "msds": (600, 80),
    "regulation": (800, 100),
    "deck": (450, 60),
    "page": (350, 50),
}


@dataclass
class Chunk:
    id: str
    source: str
    sku: str
    title: str
    section: str
    content: str
    embed_text: str = ""
    profile: str = "page"


def _tail_overlap(buffer: str, overlap: int) -> str:
    """取上一块的尾部做重叠，但必须落在词边界上。

    按字符硬切会把 "respiratory" 切成 "piratory"，这种碎片进索引后
    既检索不到，被当成答案引用也很荒谬。
    """
    if overlap <= 0 or not buffer:
        return ""
    tail = buffer[-overlap:]
    for sep in ("\n", " ", "。"):
        cut = tail.find(sep)
        if 0 <= cut < len(tail) - 1:
            return tail[cut + 1 :].strip()
    return tail.strip()


def _hard_split(block: str, max_len: int, overlap: int) -> list[str]:
    """单句超过块长时的兜底切分：优先在空白处断开。"""
    pieces: list[str] = []
    start, total = 0, len(block)
    while start < total:
        end = min(start + max_len, total)
        if end < total:
            cut = max(block.rfind(" ", start, end), block.rfind("\n", start, end))
            if cut > start + int(max_len * 0.6):
                end = cut + 1
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _split_text(text: str, max_len: int, overlap: int) -> list[str]:
    if len(text) <= max_len:
        return [text]

    # 先按行、再按句积累：MSDS 是逐行的键值结构，行粒度切出来的块最好用
    atoms: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_len:
            atoms.append(line)
        else:
            atoms.extend(s.strip() for s in _SENT_RE.split(line) if s.strip())

    pieces: list[str] = []
    buffer = ""
    for atom in atoms:
        if len(atom) > max_len:
            if buffer.strip():
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_hard_split(atom, max_len, overlap))
            continue
        if len(buffer) + len(atom) + 1 <= max_len:
            buffer = f"{buffer}\n{atom}" if buffer else atom
        else:
            if buffer.strip():
                pieces.append(buffer)
            tail = _tail_overlap(buffer, overlap)
            buffer = f"{tail}\n{atom}" if tail else atom
    if buffer.strip():
        pieces.append(buffer)
    return [p for p in pieces if p.strip()]


def build_chunks(
    docs: list[RawDoc],
    glossary: Glossary | None = None,
    min_len: int = 20,
    max_len: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        doc_len, doc_overlap = PROFILE_PARAMS.get(doc.profile, PROFILE_PARAMS["page"])
        doc_len = max_len or doc_len
        doc_overlap = overlap if overlap is not None else doc_overlap
        ingredient = glossary.match_document(doc.title) if glossary else None
        label = f"{doc.title}｜{ingredient.header()}" if ingredient else doc.title
        # 向量化文本 = 章节级标签 + 正文。标签必须章节级（而非文件级），
        # 否则同一文件所有块的向量会被词头拉平，"毒性"和"运输"分不开
        short = f"{ingredient.zh} {ingredient.en[0]}" if ingredient else ""
        for idx, section in enumerate(doc.sections):
            for sub_idx, piece in enumerate(_split_text(section.text, doc_len, doc_overlap)):
                body = piece.strip()
                if len(body) < min_len:
                    continue
                content = f"【{label}｜{section.heading}】\n{body}"
                head = f"{short}｜{section.heading}" if short else section.heading
                embed_text = f"{head}\n{body}"
                digest = hashlib.sha1(f"{doc.source}:{idx}:{sub_idx}".encode()).hexdigest()[:10]
                chunks.append(
                    Chunk(
                        id=f"{doc.meta.get('cas') or doc.sku or 'NA'}-{digest}",
                        source=doc.source,
                        sku=doc.sku,
                        title=doc.title,
                        section=section.heading,
                        content=content,
                        embed_text=embed_text,
                        profile=doc.profile,
                    )
                )
    return chunks
