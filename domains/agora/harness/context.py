"""Prompt packing: append-only between compactions, summarize at the budget.

The history is kept verbatim until the prompt approaches the token budget, so
each turn extends the previous prompt and the endpoint's prefix cache can reuse
it. At the trigger (``policy.CONTEXT_TOKENS``) the older part becomes one
summary and the newest ``policy.KEEP_RECENT_TOKENS`` stay verbatim, following
pi and OpenCode: the summary replaces the older turns, it is not a sliding
window that deletes single blocks.

Two invariants keep the split valid:

* the system prompt and the task are never dropped;
* a cut never lands on a tool result. The tail starts at an assistant message,
  so an assistant ``tool_calls`` message and its results stay together.

``usage.prompt_tokens`` is the primary count. The caller can inject an exact
prompt counter (vLLM ``/tokenize``); it drives the trigger and the summary
input limit. Chars remain the fallback for endpoints without that counter; the
ratio is calibrated from the last reported usage, because this corpus measures
2.3-2.6 chars/token, not 4.
"""

from __future__ import annotations

import json

from . import policy, prompts


SUMMARY_PREFIX = ("The conversation history before this point was compacted "
                  "into the following summary. Continue the task from the kept "
                  "messages.")


def message_chars(message: dict) -> int:
    """The endpoint-visible size of one message, excluding usage."""
    size = len(message.get("content") or "")
    if message.get("tool_calls"):
        size += len(json.dumps(message["tool_calls"]))
    return size


def total_chars(messages: list[dict], specs: list[dict] | None = None) -> int:
    """Message text plus the tool declaration the template renders first."""
    size = sum(message_chars(message) for message in messages)
    if specs:
        size += len(json.dumps(specs))
    return size


def chars_per_token(anchor: dict | None = None) -> float:
    """The episode's measured ratio, clamped against a bad provider count."""
    if anchor:
        chars = anchor.get("chars") or 0
        tokens = anchor.get("tokens") or 0
        if chars > 0 and tokens > 0:
            return min(6.0, max(1.2, chars / tokens))
    return policy.CHARS_PER_TOKEN


def estimate(messages: list[dict], specs: list[dict] | None = None,
             anchor: dict | None = None) -> int:
    """Estimated prompt tokens for ``messages``.

    ``anchor`` is the last request the endpoint measured: its exact prompt
    tokens and the char count of what was sent. The delta since then is a char
    estimate. Without an anchor the fallback ratio applies.
    """
    size = total_chars(messages, specs)
    ratio = chars_per_token(anchor)
    if anchor and anchor.get("tokens"):
        measured_chars = anchor.get("chars") or 0
        if size > measured_chars:
            return int(anchor["tokens"] + (size - measured_chars) / ratio)
    return int(size / ratio)


def cut_point(messages: list[dict], keep_tokens: int | None = None,
              ratio: float | None = None) -> int | None:
    """Index where the verbatim tail starts, or None when it cannot be cut.

    Walk back from the newest message until the tail holds ``keep_tokens`` and
    the boundary is an assistant message. If the whole history is smaller than
    the tail, there is nothing to summarize and None is returned.
    """
    keep = policy.KEEP_RECENT_TOKENS if keep_tokens is None else keep_tokens
    chars = policy.CHARS_PER_TOKEN if ratio is None else ratio
    target = keep * chars
    size = 0
    for index in range(len(messages) - 1, 1, -1):
        size += message_chars(messages[index])
        if size >= target and messages[index].get("role") == "assistant":
            return index
    # The budget is smaller than the tail: summarize from the oldest block that
    # leaves a non-empty older part, instead of refusing to compact.
    for index in range(3, len(messages)):
        if messages[index].get("role") == "assistant":
            return index
    return None


def split(messages: list[dict], cut: int) -> tuple[list[dict], list[dict], list[dict]]:
    """``(head, older, tail)``; the head is the system prompt and the task."""
    return messages[:2], messages[2:cut], messages[cut:]


def without_older(messages: list[dict], cut: int) -> list[dict]:
    """Deterministic fallback: drop the older part, keep the tail."""
    return messages[:2] + messages[cut:]


def drop_oldest_block(messages: list[dict]) -> list[dict]:
    """Drop the oldest assistant message and the tool results it asked for.

    Used only to fit the summary request in the window. Dropping a single
    message could leave an orphan tool result, so the call and its results go
    together.
    """
    for start, message in enumerate(messages):
        if message.get("role") == "assistant":
            end = start + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            return messages[:start] + messages[end:]
    return messages[1:] if messages else messages


def is_summary(message: dict) -> bool:
    return (message.get("role") == "user"
            and str(message.get("content") or "").startswith(SUMMARY_PREFIX))


def previous_summary(older: list[dict]) -> str | None:
    """The checkpoint the previous compaction inserted, if it is in ``older``."""
    for message in older:
        if is_summary(message):
            text = str(message.get("content") or "")[len(SUMMARY_PREFIX):].strip()
            if text.startswith("<summary>"):
                text = text[len("<summary>"):].lstrip()
            if text.endswith("</summary>"):
                text = text[:-len("</summary>")].rstrip()
            return text
    return None


def summary_message(text: str) -> dict:
    return {"role": "user",
            "content": f"{SUMMARY_PREFIX}\n<summary>\n{text.strip()}\n</summary>"}


def summary_messages(request: str) -> list[dict]:
    """The exact message list the summarization call sends."""
    return [{"role": "system", "content": prompts.SUMMARY_SYSTEM},
            {"role": "user", "content": request}]


def render_transcript(messages: list[dict]) -> str:
    """A readable transcript for the summarizer. Thinking is never present."""
    blocks = []
    for message in messages:
        if is_summary(message):
            continue
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "assistant":
            if content:
                blocks.append(f"[Assistant]\n{content}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append(f"[Tool call {function.get('name')}]\n"
                              + str(function.get("arguments") or ""))
        elif role == "tool":
            blocks.append(f"[Tool result]\n{content}")
        elif role == "user":
            blocks.append(f"[User]\n{content}")
        elif role == "system":
            blocks.append(f"[System]\n{content}")
    return "\n\n".join(blocks)


def summary_request(task: str, transcript: str, previous: str | None = None,
                    notes: str | None = None) -> str:
    """The one user message for the summarization call.

    The data comes first and the instructions last, as in pi and OpenCode:
    after a long transcript, the most recent text is the one the model follows.
    """
    parts = [f"<task>\n{task}\n</task>",
             f"<conversation>\n{transcript}\n</conversation>"]
    if previous:
        parts.append("Merge the previous summary below with the conversation. "
                     "Newer history wins; keep unresolved evidence and "
                     "questions.")
        parts.append(f"<previous-summary>\n{previous}\n</previous-summary>")
    if notes:
        parts.append(f"<notes>\n{notes}\n</notes>")
    parts.append(prompts.SUMMARY_TEMPLATE)
    return "\n\n".join(parts)
