"""Prompt packing: keep the episode under a hard char budget.

Oldest tool results go first, because a stale search hit costs tokens on every
later turn. The system prompt is never dropped. An assistant message and the
tool results it asked for are dropped together, otherwise the tool messages
left behind would answer tool_call ids that no longer exist.
"""

from __future__ import annotations

import json

from . import policy


def _chars(message: dict) -> int:
    size = len(message.get("content") or "")
    if message.get("tool_calls"):
        size += len(json.dumps(message["tool_calls"]))
    return size


def _oldest_tool_block(messages: list[dict]) -> tuple[int, int] | None:
    """``[start, end)`` of the first assistant tool_calls block and its results."""
    for start, message in enumerate(messages):
        if start and message.get("role") == "assistant" and message.get("tool_calls"):
            end = start + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            return start, end
    return None


def trim(messages: list[dict], budget: int | None = None) -> list[dict]:
    """Drop oldest tool blocks until the prompt fits. System prompt stays."""
    budget = policy.CONTEXT_CHARS if budget is None else budget
    kept = list(messages)
    while sum(_chars(m) for m in kept) > budget:
        block = _oldest_tool_block(kept)
        if block is None:
            # shortcut: one oversized turn is kept whole; split it field-by-field
            # if a single result can ever approach the model window.
            break
        del kept[block[0]:block[1]]
    return kept
