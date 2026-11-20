-- pgvector 初始化脚本
-- 注意：vector 维度必须与 .env 中的 EMBED_DIM 完全一致，否则入库会失败。
-- 更换向量模型 = 换维度 = 需要重建表并重新入库。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunk (
    id          text PRIMARY KEY,
    source      text NOT NULL,
    sku         text NOT NULL DEFAULT '',
    title       text NOT NULL,
    section     text NOT NULL DEFAULT '',
    content     text NOT NULL,
    embedding   vector(1024) NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- 向量索引：HNSW 在召回率与延迟之间更均衡，数据量 < 100 万条足够
CREATE INDEX IF NOT EXISTS idx_kb_chunk_embedding
    ON kb_chunk USING hnsw (embedding vector_cosine_ops);

-- 关键词索引：用于混合检索的关键词召回
CREATE INDEX IF NOT EXISTS idx_kb_chunk_fts
    ON kb_chunk USING gin (
        to_tsvector('simple', title || ' ' || section || ' ' || content)
    );

CREATE INDEX IF NOT EXISTS idx_kb_chunk_sku ON kb_chunk (sku);
