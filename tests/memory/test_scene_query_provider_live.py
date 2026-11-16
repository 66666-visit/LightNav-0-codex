"""Opt-in DeepSeek HTTP contract for thinking-mode SceneQuery tool turns.

Run with SCENE_QUERY_RUN_LIVE_DEEPSEEK=1 and MEMORY_QUERY_API_KEY set.
This test contacts the paid DeepSeek API and is skipped in the regular CPU suite.
"""

from __future__ import annotations

import json
import os

import pytest

from lightnav.memory.scene_query_config import SceneQueryConfig
from lightnav.memory.scene_query_controller import _assistant_message, _final
from lightnav.memory.scene_query_provider import OpenAICompatibleQueryProvider
from lightnav.memory.scene_query_tools import OPENAI_TOOLS


def test_live_deepseek_v4_pro_thinking_tool_call_and_final() -> None:
    if os.getenv("SCENE_QUERY_RUN_LIVE_DEEPSEEK") != "1":
        pytest.skip("set SCENE_QUERY_RUN_LIVE_DEEPSEEK=1 to call the real DeepSeek API")
    if not os.getenv("MEMORY_QUERY_API_KEY"):
        pytest.skip("MEMORY_QUERY_API_KEY is required for the live DeepSeek API test")

    provider = OpenAICompatibleQueryProvider(SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
        thinking="enabled", timeout=120,
    ))
    find_objects = next(tool for tool in OPENAI_TOOLS
                        if tool["function"]["name"] == "find_objects")
    tools = [find_objects]
    messages = [
        {"role": "system", "content": (
            "You are testing a read-only navigation memory tool. First call find_objects "
            "with class_name=sofa and history_eligible_only=true. After its empty result, "
            "return only this exact JSON object: "
            '{"decision":"NO_MEMORY","selected_l1_fact_ids":[],"selected_l3_object_ids":[],'
            '"selected_l3_evidence_ids":[],"reason_code":"NO_ELIGIBLE_HISTORY"}'
        )},
        {"role": "user", "content": "Find the sofa in navigation memory before Final."},
    ]

    # Thinking mode cannot force a tool via tool_choice=required. The prompt
    # requests it, and this assertion checks that the first HTTP response did.
    first = provider.complete(messages, tools)
    calls = first.get("tool_calls")
    assert isinstance(calls, list) and calls
    assert isinstance(first.get("reasoning_content"), str)
    assert all(call["function"]["name"] == "find_objects" for call in calls)
    messages.append(_assistant_message(first, tool_calls=calls))
    assert isinstance(messages[-1]["content"], str)
    for call in calls:
        assert isinstance(call["id"], str) and call["id"]
        messages.append({
            "role": "tool", "tool_call_id": call["id"],
            "content": json.dumps({"result": [], "cache_hit": False}),
        })

    # This is a real HTTP request with both tools and tool_choice=none. DeepSeek
    # must accept the replayed reasoning and return a message without tool calls.
    final = provider.complete(messages, tools, tool_choice="none")
    assert not final.get("tool_calls")
    parsed = json.loads(final["content"])
    assert _final(parsed) == parsed
    assert parsed["decision"] == "NO_MEMORY"
    assert parsed["selected_l3_object_ids"] == []
    assert parsed["selected_l3_evidence_ids"] == []
