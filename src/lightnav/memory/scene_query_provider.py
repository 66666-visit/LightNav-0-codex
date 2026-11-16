"""OpenAI-compatible Chat Completions adapter, separate from LightNav inference."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lightnav.memory.scene_query_config import SceneQueryConfig


class QueryProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


class OpenAICompatibleQueryProvider:
    def __init__(self, config: SceneQueryConfig) -> None:
        self.config = config

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        config = self.config
        api_key = os.getenv("MEMORY_QUERY_API_KEY", "")
        if not (config.base_url and api_key and config.model):
            raise RuntimeError("memory query API configuration is incomplete")
        endpoint = config.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = json.dumps({
            "model": config.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "temperature": 0,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(endpoint, data=payload, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(request, timeout=config.timeout) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            # HTTPError may contain response text or headers with secrets. Log only type/status.
            status = f" status={exc.code}" if isinstance(exc, HTTPError) else ""
            raise RuntimeError(f"memory query API {type(exc).__name__}{status}") from None
        try:
            parsed = json.loads(body)
            message = parsed["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("memory query API returned an invalid chat completion") from exc
        if not isinstance(message, dict):
            raise ValueError("memory query API returned an invalid message")
        return message
