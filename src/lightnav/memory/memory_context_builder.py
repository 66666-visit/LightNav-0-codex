"""Adapt episode memory to the existing LightNav processor sample format.

The checkpoint's prompt, video placeholders and output sentence are a trained
contract.  This adapter only appends bounded plain text after the instruction's
closing tag in the human message; it never changes the visual or action format.
"""

from __future__ import annotations

from html import escape

from lightnav.memory.schemas import MemoryContext

_TASK_END = "</navigation_task>"
_DEFAULT_MAX_CHARS = 600


def build_context_text(context: MemoryContext, *, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Return prompt-safe text within a character budget.

    Escaping angle brackets prevents remembered text from introducing a Qwen
    ``<video>`` placeholder or impersonating a navigation/action token.
    """
    if max_chars <= 0 or not context.text.strip():
        return ""
    safe = escape(context.text.strip(), quote=False)
    if len(safe) <= max_chars:
        return safe
    if max_chars < 2:
        return ""
    return safe[: max_chars - 1].rstrip() + "…"


def inject_memory_context(
    sample: dict,
    context: MemoryContext,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> dict:
    """Copy a VLN sample and insert memory into its human prompt.

    ``sample`` has the real inference shape: ``video_segments`` plus
    ``conversations`` containing a human prompt and a GPT placeholder.  The
    original sample and all video segment objects remain untouched.
    """
    copied = dict(sample)
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("sample['conversations'] must be a list")
    copied["conversations"] = [dict(message) for message in conversations]

    memory_text = build_context_text(context, max_chars=max_chars)
    if not memory_text:
        return copied

    human_messages = [m for m in copied["conversations"] if m.get("from") == "human"]
    if len(human_messages) != 1:
        raise ValueError("memory injection requires exactly one human message")
    human = human_messages[0]
    prompt = human.get("value")
    if not isinstance(prompt, str) or prompt.count(_TASK_END) != 1:
        raise ValueError("human prompt must contain one </navigation_task> marker")

    insert_at = prompt.index(_TASK_END) + len(_TASK_END)
    # The checked-in VLN prompts put a period immediately after the closing tag.
    # Keeping it with the original task sentence avoids changing that sentence.
    if prompt[insert_at : insert_at + 2] == ". ":
        insert_at += 2
    prefix = prompt[:insert_at]
    suffix = prompt[insert_at:]
    spacer = "" if prefix.endswith(" ") else " "
    human["value"] = f"{prefix}{spacer}{memory_text}\n{suffix}"
    return copied
