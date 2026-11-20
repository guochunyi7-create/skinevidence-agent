"""集中配置：全部来自环境变量，启动时集中校验并快速失败。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """配置不合法，进程不应启动。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- 运行时 ----
    app_name: str = "cosmetic-agent"
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    cors_origins_raw: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- 对话模型：统一走 OpenAI 兼容协议，换供应商只改环境变量 ----
    llm_provider: Literal["extractive", "mock", "openai_compatible"] = "extractive"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout_s: float = 30.0
    llm_max_tokens: int = 800

    # ---- 向量模型：同上，可独立于对话模型切换 ----
    embed_provider: Literal["mock", "openai_compatible"] = "mock"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "text-embedding-v3"
    embed_dim: int = 1024
    embed_batch: int = 16

    # ---- 存储 ----
    vector_store: Literal["memory", "pg"] = "memory"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/cosmetic"
    index_file: str = "data/index/chunks.jsonl"
    knowledge_dir: str = "data/knowledge"
    glossary_file: str = "data/knowledge/glossary/ingredients.yaml"
    compliance_file: str = "data/compliance/banned_claims.yaml"
    artifact_dir: str = "artifacts"

    # ---- 检索与回答策略 ----
    retrieval_pool: int = Field(default=30, ge=5, le=100)
    retrieval_top_k: int = Field(default=10, ge=1, le=20)
    min_answer_score: float = Field(default=0.30, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

    # ---- Agent 模式（灰度开关：默认关闭，与原 RAG 流程行为完全一致）----
    agent_enabled: bool = False
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    agent_tool_timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    def validate_runtime(self) -> None:
        """启动时校验：宁可起不来，也不要带着错误配置对外服务。"""
        problems: list[str] = []

        if self.llm_provider == "openai_compatible" and not self.llm_api_key:
            problems.append("LLM_PROVIDER=openai_compatible 时必须提供 LLM_API_KEY")
        if self.embed_provider == "openai_compatible":
            if not self.embed_api_key:
                problems.append(
                    "EMBED_PROVIDER=openai_compatible 时必须提供 EMBED_API_KEY"
                )
            if not self.embed_base_url:
                problems.append(
                    "EMBED_PROVIDER=openai_compatible 时必须提供 EMBED_BASE_URL"
                )
        if self.env == "prod" and "*" in self.cors_origins:
            problems.append("生产环境禁止 CORS 使用通配符 *")
        if self.vector_store == "pg" and not self.database_url:
            problems.append("VECTOR_STORE=pg 时必须提供 DATABASE_URL")
        # Agent 靠模型自主选工具，抽取式模型没有这个能力；开了开关却用离线模型时
        # 必须快速失败，否则会静默退化成"没有工具调用的空转"，比直接报错更难排查。
        if self.agent_enabled and self.llm_provider != "openai_compatible":
            problems.append(
                "AGENT_ENABLED=true 时必须设置 LLM_PROVIDER=openai_compatible"
                "（extractive 抽取式模型不支持工具调用）"
            )

        if problems:
            raise ConfigError("配置校验失败：\n- " + "\n- ".join(problems))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime()
    return settings
