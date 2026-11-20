"""入库流水线：文档 → 切块 → 向量化 → 索引文件（可选写入 pgvector）。

用法：
    python -m app.ingestion.pipeline                       # 自动扫描 data/knowledge 下各子目录
    python -m app.ingestion.pipeline --input 法规=data/knowledge/01_regulation
    python -m app.ingestion.pipeline --to-db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingestion.chunker import build_chunks
from app.ingestion.glossary import build_glossary
from app.ingestion.loader import Source, load_sources
from app.services.llm_gateway import build_embedder

logger = logging.getLogger("ingest")


def _discover(root: Path) -> list[Source]:
    """把 knowledge 下的一级子目录当成知识域标签，散落文件归到 default（不递归）。"""
    if not root.exists():
        raise SystemExit(f"知识库目录不存在：{root}")
    sources: list[Source] = []
    loose = False
    for child in sorted(root.iterdir()):
        if child.is_dir():
            if child.name in {"glossary", "index"}:
                continue
            sources.append(Source(child.name, child, recursive=True))
        elif child.is_file() and child.suffix.lower() in {".pdf", ".docx", ".pptx", ".md", ".txt"}:
            loose = True
    if loose:
        sources.append(Source("default", root, recursive=False))
    return sources


async def run(
    sources: list[Source], out_file: Path, to_db: bool, max_len: int | None, overlap: int | None
) -> dict:
    settings = get_settings()
    glossary = build_glossary(Path(settings.glossary_file))
    docs = load_sources(sources)
    if not docs:
        raise SystemExit("没有解析到任何文档，请检查输入目录")

    chunks = build_chunks(docs, glossary, max_len=max_len, overlap=overlap)
    if not chunks:
        raise SystemExit("切块结果为空，请检查文档内容是否过短")

    embedder = build_embedder(settings)
    logger.info("开始向量化", extra={"extra_fields": {"chunks": len(chunks)}})
    vectors = await embedder.embed([c.embed_text or c.content for c in chunks])
    await embedder.aclose()

    if len(vectors) != len(chunks):
        raise SystemExit(f"向量数量不匹配：{len(vectors)} != {len(chunks)}")
    if vectors and len(vectors[0]) != settings.embed_dim:
        raise SystemExit(
            f"向量维度 {len(vectors[0])} 与 EMBED_DIM={settings.embed_dim} 不一致"
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for chunk, vec in zip(chunks, vectors):
            fh.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "source": chunk.source,
                        "sku": chunk.sku,
                        "title": chunk.title,
                        "section": chunk.section,
                        "profile": chunk.profile,
                        "content": chunk.content,
                        "embedding": vec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if to_db:
        await _write_pg(settings, chunks, vectors)

    by_profile: dict[str, int] = {}
    for c in chunks:
        by_profile[c.profile] = by_profile.get(c.profile, 0) + 1
    report = {
        "sources": {s.label: str(s.path) for s in sources},
        "files": len(docs),
        "chunks": len(chunks),
        "by_profile": by_profile,
        "dim": len(vectors[0]) if vectors else 0,
        "index_file": str(out_file),
        "to_db": to_db,
    }
    logger.info("入库完成", extra={"extra_fields": report})
    return report


async def _write_pg(settings, chunks, vectors) -> None:
    import asyncpg

    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.executemany(
            """
            INSERT INTO kb_chunk (id, source, sku, title, section, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
            ON CONFLICT (id) DO UPDATE
              SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
            """,
            [
                (
                    c.id, c.source, c.sku, c.title, c.section, c.content,
                    "[" + ",".join(f"{v:.6f}" for v in vec) + "]",
                )
                for c, vec in zip(chunks, vectors)
            ],
        )
    finally:
        await conn.close()


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="化妆品知识库入库")
    parser.add_argument(
        "--input", action="append", default=[],
        help="标签=目录，可重复；不传则自动扫描 KNOWLEDGE_DIR 下的一级子目录",
    )
    parser.add_argument("--out", default=settings.index_file)
    parser.add_argument("--max-len", type=int, default=None, help="覆盖切块长度（一般不需要）")
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--to-db", action="store_true", help="同时写入 pgvector")
    args = parser.parse_args()

    if args.input:
        sources = []
        for item in args.input:
            if "=" not in item:
                raise SystemExit(f"--input 需要 标签=目录 形式，收到：{item}")
            label, path = item.split("=", 1)
            sources.append(Source(label.strip(), Path(path.strip())))
    else:
        sources = _discover(Path(settings.knowledge_dir))

    try:
        report = asyncio.run(
            run(sources, Path(args.out), args.to_db, args.max_len, args.overlap)
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
