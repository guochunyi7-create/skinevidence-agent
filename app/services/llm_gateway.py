"""模型网关：对话模型与向量模型都可切换，业务层只依赖抽象。

统一走 OpenAI 兼容协议（DeepSeek / 通义 / 豆包 / vLLM / Ollama 均支持），
切供应商只需改环境变量，不用改一行业务代码。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.ingestion.glossary import Glossary
from app.services.extractive_reader import DocBlock, compose

logger = logging.getLogger(__name__)

# 必须带属性才算资料块，避免把提示词里的 <doc> 字样误当成文档
_DOC_RE = re.compile(r"<doc\s+([^>]*)>(.*?)</doc>", re.S)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


class BaseLLM:
    async def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class BaseEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolChatResult:
    """带工具调用的模型回复：要么给出 tool_calls，要么给出最终文本。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


@runtime_checkable
class ToolCapableLLM(Protocol):
    """支持 function calling 的模型需实现此协议；抽取式模型不实现。"""

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ToolChatResult: ...


class ExtractiveLLM(BaseLLM):
    """离线抽取式回答：不生成新句子，只从检索到的原文里挑出能回答问题的那几行。

    相比"把第一个切块截断打印"，它做三件事：定位（按实词命中挑行）、
    裁剪（长句裁到关键词窗口）、组织（按文件/章节分组，标注出处）。
    因为不生成，所以不会编造成分、浓度或功效；最坏结果是答偏，不是答假。
    """

    def __init__(self, glossary: Glossary | None = None) -> None:
        self.glossary = glossary

    async def complete(self, system: str, user: str) -> str:
        docs = self._parse_docs(system)
        if not docs:
            return ""
        return compose(user, docs, self.glossary)

    @staticmethod
    def _parse_docs(system: str) -> list[DocBlock]:
        blocks: list[DocBlock] = []
        for raw_attrs, content in _DOC_RE.findall(system):
            attrs = {k: v for k, v in _ATTR_RE.findall(raw_attrs)}
            blocks.append(
                DocBlock(
                    title=attrs.get("title", ""),
                    section=attrs.get("section", ""),
                    source=attrs.get("source", ""),
                    content=content,
                )
            )
        return blocks


class MockEmbedder(BaseEmbedder):
    """字符二元组哈希向量，能反映字面相似度，离线即可跑通检索链路。"""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        clean = re.sub(r"\s+", "", text)
        grams = [clean[i : i + 2] for i in range(max(len(clean) - 1, 1))] or [clean]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
            slot = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[slot] += sign
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0 else vec).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class OpenAICompatLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=settings.llm_timeout_s,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """共用的带重试 POST：5xx/网络错误退避重试，4xx 直接失败。

        Agent 循环每一步都会调用模型，把重试集中在这里，避免两处逻辑漂移。
        """
        last_error = "unknown"
        for attempt in range(3):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code >= 500:
                    last_error = f"status={resp.status_code}"
                    await asyncio.sleep(0.4 * (2**attempt))
                    continue
                if resp.status_code >= 400:
                    # 4xx 属于配置或参数错误，重试没有意义
                    raise UpstreamError(
                        f"模型服务返回 {resp.status_code}", detail={"body": resp.text[:300]}
                    )
                return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = type(exc).__name__
                await asyncio.sleep(0.4 * (2**attempt))
        raise UpstreamError(f"模型服务连续失败：{last_error}")

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        data = await self._post_chat(payload)
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ToolChatResult:
        """OpenAI 兼容的 function calling：解析 model 返回的 tool_calls。

        智谱 GLM 的 OpenAI 兼容端点同样支持 tools 参数；若某个供应商不支持，
        会返回 4xx，由 _post_chat 抛 UpstreamError，启动时能力门控会提前拦下。
        """
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        data = await self._post_chat(payload)
        message = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("工具参数不是合法 JSON，按空参数处理", extra={"extra_fields": {"raw": args[:120]}})
                    parsed = {}
            else:
                parsed = args
            calls.append(
                ToolCall(
                    id=raw.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=parsed if isinstance(parsed, dict) else {},
                )
            )
        return ToolChatResult(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatEmbedder(BaseEmbedder):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.embed_base_url.rstrip("/"),
            timeout=settings.llm_timeout_s,
            headers={"Authorization": f"Bearer {settings.embed_api_key}"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = max(1, self.settings.embed_batch)
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            try:
                resp = await self._client.post(
                    "/embeddings",
                    json={"model": self.settings.embed_model, "input": chunk},
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise UpstreamError(f"向量模型不可用：{type(exc).__name__}") from exc
            if resp.status_code >= 400:
                raise UpstreamError(f"向量模型返回 {resp.status_code}")
            items = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
            for item in items:
                vec = item["embedding"]
                if len(vec) != self.settings.embed_dim:
                    raise UpstreamError(
                        f"向量维度不匹配：期望 {self.settings.embed_dim}，实际 {len(vec)}"
                    )
                vectors.append(vec)
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()


def build_llm(settings: Settings, glossary: Glossary | None = None) -> BaseLLM:
    # extractive = 离线抽取式，不需要任何密钥；线上换成真实模型只改这一个环境变量
    if settings.llm_provider in {"extractive", "mock"}:
        logger.info(
            "使用抽取式回答引擎（离线、不调用外部模型）",
            extra={"extra_fields": {"provider": settings.llm_provider}},
        )
        return ExtractiveLLM(glossary)
    return OpenAICompatLLM(settings)


def build_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embed_provider == "mock":
        return MockEmbedder(settings.embed_dim)
    return OpenAICompatEmbedder(settings)
