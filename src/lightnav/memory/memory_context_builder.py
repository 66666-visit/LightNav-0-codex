"""Adapt episode memory to the existing LightNav processor sample format.

The checkpoint's prompt, video placeholders and output sentence are a trained
contract.  This adapter only appends bounded plain text after the instruction's
closing tag in the human message; it never changes the visual or action format.
"""

from __future__ import annotations

from html import escape
from hashlib import sha256

from lightnav.memory.schemas import MemoryContext

_TASK_END = "</navigation_task>"
_DEFAULT_MAX_CHARS = 600


def build_context_text(context: MemoryContext, *, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Return prompt-safe text within the existing character limit.

    Escaping angle brackets prevents remembered text from introducing a Qwen
    ``<video>`` placeholder or impersonating a navigation/action token. If
    escaping expands a retrieved context beyond the limit, omit trailing
    records instead of cutting one record midway.
    """
    if max_chars <= 0 or not context.text.strip():
        return ""
    safe = escape(context.text.strip(), quote=False)
    if len(safe) <= max_chars:
        return safe
    lines = safe.splitlines()
    if len(lines) > 1:
        included: list[str] = []
        used = 0
        for line in lines:
            length = len(line) + (1 if included else 0)
            if used + length > max_chars:
                break
            included.append(line)
            used += length
        # A retrieved context starts with a heading. It is not useful to
        # inject that heading without any complete observation below it.
        if len(included) == 1 and included[0] == "Episode observations (may be uncertain):":
            return ""
        return "\n".join(included)
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


def audit_prompt_injection(base: dict, augmented: dict, context: MemoryContext,
                           *, max_chars: int) -> dict[str, object]:
    """Check exact original-prompt recovery and unchanged multimodal payload."""
    originals = base.get("conversations", [])
    changed = augmented.get("conversations", [])
    base_human = [message for message in originals if message.get("from") == "human"]
    new_human = [message for message in changed if message.get("from") == "human"]
    if len(base_human) != 1 or len(new_human) != 1:
        raise ValueError("prompt audit requires one human message")
    original_prompt = base_human[0]["value"]
    new_prompt = new_human[0]["value"]
    memory_text = build_context_text(context, max_chars=max_chars)
    if not memory_text:
        roundtrip = new_prompt == original_prompt and (
            augmented is base or (
                set(augmented) == set(base)
                and all(augmented[key] is value for key, value in base.items()
                        if key not in {"video_segments", "conversations"})
            )
        )
    elif isinstance(original_prompt, str) and original_prompt.count(_TASK_END) == 1:
        insert_at = original_prompt.index(_TASK_END) + len(_TASK_END)
        if original_prompt[insert_at:insert_at + 2] == ". ":
            insert_at += 2
        prefix, suffix = original_prompt[:insert_at], original_prompt[insert_at:]
        spacer = "" if prefix.endswith(" ") else " "
        memory_block = spacer + memory_text + "\n"
        roundtrip = (
            new_prompt == prefix + memory_block + suffix
            and new_prompt[:insert_at] + new_prompt[
                insert_at + len(memory_block):
            ] == original_prompt
        )
    else:
        roundtrip = False
    same_roles = [item.get("from") for item in originals] == [item.get("from") for item in changed]
    same_others = all(
        old == new for old, new in zip(originals, changed)
        if old.get("from") != "human"
    )
    same_video = base.get("video_segments") is augmented.get("video_segments")
    if not (roundtrip and same_roles and same_others and same_video):
        raise ValueError("memory prompt contract failed")
    return {
        "base_sha256": sha256(original_prompt.encode()).hexdigest(),
        "augmented_sha256": sha256(new_prompt.encode()).hexdigest(),
        "roundtrip_ok": True, "injected_chars": len(memory_text),
    }
