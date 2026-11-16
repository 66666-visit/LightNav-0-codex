"""CPU protocol checks for the independent OpenAI-compatible query provider."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError

import pytest

from lightnav.memory import scene_query_provider as provider_module
from lightnav.memory.scene_query_config import SceneQueryConfig
from lightnav.memory.scene_query_provider import OpenAICompatibleQueryProvider


class _Reply:
    def __init__(self, message: dict[str, Any]) -> None:
        self.body = json.dumps({"choices": [{"message": message}]}).encode()

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize("setting", [None, "enabled", "disabled"])
def test_thinking_switch_keeps_openai_tool_call_shape(
    monkeypatch: pytest.MonkeyPatch, setting: str | None,
) -> None:
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "private-test-credential")
    sent: list[dict[str, Any]] = []

    def urlopen(request: Any, *, timeout: float) -> _Reply:
        assert timeout == 7
        sent.append(json.loads(request.data))
        return _Reply({"role": "assistant", "content": "done"})

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
        timeout=7, thinking=setting,
    )
    messages = [{"role": "user", "content": "Find the sofa."}]
    tools = [{"type": "function", "function": {"name": "find_objects"}}]
    assert OpenAICompatibleQueryProvider(config).complete(messages, tools)["content"] == "done"

    assert sent[0]["model"] == "deepseek-v4-pro"
    assert sent[0]["messages"] == messages
    assert sent[0]["tools"] == tools
    assert "private-test-credential" not in json.dumps(sent[0])
    if setting is None:
        assert "thinking" not in sent[0]
        assert sent[0]["tool_choice"] == "auto"
        assert sent[0]["temperature"] == 0
    else:
        assert sent[0]["thinking"] == {"type": setting}
        if setting == "enabled":
            assert "temperature" not in sent[0]
            assert "tool_choice" not in sent[0]
        else:
            assert sent[0]["tool_choice"] == "auto"


def test_provider_returns_full_reasoning_message_for_multi_round_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "private-test-credential")
    tool_call = {
        "id": "call_1", "type": "function",
        "function": {"name": "find_objects", "arguments": '{"class_name":"sofa"}'},
    }
    first = {
        "role": "assistant", "content": "", "reasoning_content": "private model reasoning",
        "tool_calls": [tool_call],
    }
    second = {
        "role": "assistant", "content": '{"decision":"NO_MEMORY"}',
        "reasoning_content": "continued private reasoning",
    }
    replies = iter([_Reply(first), _Reply(second)])
    requests: list[dict[str, Any]] = []

    def urlopen(request: Any, *, timeout: float) -> _Reply:
        requests.append(json.loads(request.data))
        return next(replies)

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com/v1", model="deepseek-v4-pro",
        thinking="enabled",
    )
    provider = OpenAICompatibleQueryProvider(config)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Find the sofa."}]
    tools = [{"type": "function", "function": {"name": "find_objects"}}]
    returned = provider.complete(messages, tools)
    assert returned == first
    messages.append(returned)
    messages.append({"role": "tool", "tool_call_id": "call_1", "content": "[]"})
    assert provider.complete(messages, tools, tool_choice="none") == second
    assert requests[1]["messages"][1]["reasoning_content"] == first["reasoning_content"]
    assert requests[1]["messages"][1]["tool_calls"] == [tool_call]
    assert requests[1]["messages"][2]["tool_call_id"] == "call_1"
    assert requests[1]["thinking"] == {"type": "enabled"}
    assert requests[1]["tools"] == tools
    assert requests[1]["tool_choice"] == "none"


@pytest.mark.parametrize("setting", [None, "enabled", "disabled"])
def test_provider_omits_empty_tool_definitions_for_final_request(
    monkeypatch: pytest.MonkeyPatch, setting: str | None,
) -> None:
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "private-test-credential")
    sent: list[dict[str, Any]] = []

    def urlopen(request: Any, *, timeout: float) -> _Reply:
        sent.append(json.loads(request.data))
        return _Reply({"role": "assistant", "content": "final"})

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
        thinking=setting,
    )
    assert OpenAICompatibleQueryProvider(config).complete(
        [{"role": "user", "content": "Return Final."}], [],
    )["content"] == "final"
    assert "tools" not in sent[0]
    assert "tool_choice" not in sent[0]


@pytest.mark.parametrize("setting", [None, "enabled", "disabled"])
def test_final_request_keeps_tools_and_reasoning_with_tool_choice_none(
    monkeypatch: pytest.MonkeyPatch, setting: str | None,
) -> None:
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "private-test-credential")
    sent: list[dict[str, Any]] = []

    def urlopen(request: Any, *, timeout: float) -> _Reply:
        sent.append(json.loads(request.data))
        return _Reply({"role": "assistant", "content": "final"})

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
        thinking=setting,
    )
    messages = [
        {"role": "user", "content": "Find the sofa."},
        {"role": "assistant", "content": "", "reasoning_content": "private reasoning",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "find_objects", "arguments": "{}",
         }}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
    ]
    tools = [{"type": "function", "function": {"name": "find_objects"}}]
    assert OpenAICompatibleQueryProvider(config).complete(
        messages, tools, tool_choice="none",
    )["content"] == "final"
    assert sent[0]["tools"] == tools
    assert sent[0]["tool_choice"] == "none"
    assert sent[0]["messages"][1]["reasoning_content"] == "private reasoning"
    if setting == "enabled":
        assert "temperature" not in sent[0]


def test_config_parses_thinking_and_tool_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_QUERY_THINKING", "Enabled")
    monkeypatch.setenv("MEMORY_QUERY_MAX_TOOL_CALLS", "9")
    config = SceneQueryConfig.from_env(enabled=True)
    assert config.thinking == "enabled"
    assert config.max_tool_calls == 9
    monkeypatch.setenv("MEMORY_QUERY_THINKING", "disabled")
    assert SceneQueryConfig.from_env(enabled=True).thinking == "disabled"
    monkeypatch.setenv("MEMORY_QUERY_THINKING", "unsupported")
    with pytest.raises(ValueError, match="MEMORY_QUERY_THINKING"):
        SceneQueryConfig.from_env(enabled=True)
    # Disabled mode ignores query-only environment settings, as before.
    assert SceneQueryConfig.from_env(enabled=False) == SceneQueryConfig()
    with pytest.raises(ValueError, match="MEMORY_QUERY_MAX_TOOL_CALLS"):
        SceneQueryConfig(max_tool_calls=0)
    with pytest.raises(ValueError, match="MEMORY_QUERY_THINKING"):
        SceneQueryConfig(thinking=["enabled"])  # type: ignore[arg-type]


def test_provider_error_does_not_expose_api_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "private-test-credential"
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", key)

    def urlopen(_request: Any, *, timeout: float) -> _Reply:
        raise HTTPError("https://api.deepseek.com", 401, key, {}, None)

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
    )
    with pytest.raises(RuntimeError) as caught:
        OpenAICompatibleQueryProvider(config).complete([], [])
    assert key not in str(caught.value)
    assert "status=401" in str(caught.value)
