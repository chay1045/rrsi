"""LangGraph loop: agent -> tools -> agent, until ``answer`` or MAX_TURNS.

The model client is injected: ``client(messages, tool_specs, force=None)``
returns an OpenAI-style assistant dict (``content`` and ``tool_calls``), with an
optional ``usage`` key that is summed into the trajectory and never sent back to
the model. ``force`` is the request's ``tool_choice``: None, ``"required"`` or a
tool name. Tests pass a scripted callable; cli.py passes one that wraps the
OpenAI client.

History is append-only until ``policy.CONTEXT_TOKENS``: the prompt cache then
matches the previous request except for the newest turn. At the budget,
``context`` replaces the older part with one summary and keeps
``policy.KEEP_RECENT_TOKENS`` verbatim. The summary is an injected callable, so
the graph stays model- and endpoint-agnostic; ``notes`` reads notes.md for the
summary input.

The last ``policy.FINAL_TURNS`` calls form a bounded final phase. The tool list
never changes, because the template renders it before the first message: a
changed list would invalidate the whole prefix. The phase is enforced with
``tool_choice`` and one nudge; the last call accepts only ``answer``.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import context, policy, prompts, tools


_RECENT_CALLS = 20       # exact-repeat detection window


class State(TypedDict, total=False):
    messages: list[dict]
    task: str
    turns: int
    empty_retries: int
    final_nudged: bool
    recent_calls: list[str]
    repeat_nudged: str | None
    answer: str
    usage: dict
    anchor: dict | None
    compactions: list[dict]
    summary_usage: dict


def _tool_message(call: dict, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call.get("id") or "",
            "content": content}


def _call_key(name: str, arguments: str | None) -> str:
    """Canonical ``name + arguments`` key for exact duplicate detection.

    Argument order and the provider's whitespace-padded keys are normalized so
    the same call hashes the same way across turns. Invalid JSON is kept as
    text: the call still failed identically, so it still counts as a repeat.
    """
    try:
        args = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        args = str(arguments or "")
    if isinstance(args, dict):
        args = {str(key).strip(): value for key, value in args.items()}
    return json.dumps([name, args], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _repeated_call_nudge(state: State) -> str | None:
    """The nudge when the latest call reaches REPEAT_LIMIT occurrences.

    The same key is nudged once. Another repeated call can trigger its own
    nudge, but a model that ignores the first one does not grow the prompt with
    an identical message every turn.
    """
    recent = state.get("recent_calls") or []
    if len(recent) < policy.REPEAT_LIMIT:
        return None
    key = recent[-1]
    if recent.count(key) < policy.REPEAT_LIMIT:
        return None
    if state.get("repeat_nudged") == key:
        return None
    return policy.REPEAT_NUDGE


def _empty_reply(message: dict) -> bool:
    """A reply with no content and no tool call cannot advance the episode."""
    return (not message.get("tool_calls")
            and not str(message.get("content") or "").strip())


def _last_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"].strip()
    return ""


def _sum_usage(target: dict, usage: dict | None) -> dict:
    """Add one usage report to ``target``; keep the nested cache count flat."""
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        elif key == "prompt_tokens_details" and isinstance(value, dict):
            cached = value.get("cached_tokens")
            if isinstance(cached, (int, float)):
                target["cached_tokens"] = target.get("cached_tokens", 0) + cached
    return target


def _add_usage(usage: dict, reply: dict) -> dict:
    return _sum_usage(dict(usage or {}), reply.pop("usage", None))


def _anchor(messages: list[dict], specs: list[dict], usage: dict | None) -> dict | None:
    """The exact prompt size the endpoint just measured, with its char count."""
    tokens = (usage or {}).get("prompt_tokens")
    if not isinstance(tokens, (int, float)) or tokens <= 0:
        return None
    return {"chars": context.total_chars(messages, specs), "tokens": int(tokens)}


def _final_phase(state: State) -> bool:
    """The bounded last turns: the client may only submit an answer."""
    return (policy.FINAL_TURNS > 0
            and state["turns"] >= policy.MAX_TURNS - policy.FINAL_TURNS)


def build(client=None, ctx: tools.Context | None = None, summarize=None,
          notes=None):
    """Compile the graph. ``client`` is only needed when it is invoked.

    ``summarize(request) -> {"content": str, "usage": dict, "truncated": bool}``
    is optional. Without it, the older part is dropped deterministically when
    the budget is crossed, so tests and offline runs never need an endpoint.
    ``notes`` is an optional callable that returns notes.md for the summary.

    ``run()`` sets the recursion limit to match MAX_TURNS; a direct ``invoke``
    keeps LangGraph's default and would stop after ~12 turns.
    """

    def _maybe_compact(state: State, messages: list[dict],
                       specs: list[dict]) -> tuple[list[dict], dict]:
        """Replace the older part with a summary when the prompt is over budget."""
        anchor = state.get("anchor")
        before = context.estimate(messages, specs, anchor)
        if before <= policy.CONTEXT_TOKENS:
            return messages, {}
        cut = context.cut_point(messages, policy.KEEP_RECENT_TOKENS,
                                context.chars_per_token(anchor))
        if cut is None:
            return messages, {}
        head, older, tail = context.split(messages, cut)
        if not older:
            return messages, {}
        event = {"turn": state["turns"] + 1, "before_tokens": before,
                 "dropped_messages": len(older)}
        update: dict = {}
        text, summary_usage, error = "", {}, None
        if summarize is None:
            error = "summary unavailable"
        else:
            try:
                task = (state.get("task")
                        or str((messages[1] if len(messages) > 1 else {})
                               .get("content") or ""))
                previous = context.previous_summary(older)
                body = [message for message in older
                        if not context.is_summary(message)]
                notes_text = notes() if notes is not None else None
                # The summary request must itself fit: its input plus the
                # output cap and a small reserve. A turn that adds several
                # large tool results can overshoot the agent trigger, so drop
                # the oldest blocks from the summary input if needed. The
                # dropped turns are recorded; the previous summary still covers
                # them when one exists.
                ratio = context.chars_per_token(anchor)
                # The summary prompt has a different mix from the agent prompt,
                # so keep a 10% char margin plus 4k tokens for the system
                # message and template. Overrunning this would reject the one
                # call that is supposed to save the episode.
                limit = int((policy.CONTEXT_WINDOW - policy.SUMMARY_MAX_TOKENS
                             - 4096) * ratio * 0.9)
                dropped = 0
                request = context.summary_request(
                    task, context.render_transcript(body), previous, notes_text)
                while len(request) > limit and body:
                    trimmed = context.drop_oldest_block(body)
                    if len(trimmed) == len(body):
                        break
                    dropped += len(body) - len(trimmed)
                    body = trimmed
                    request = context.summary_request(
                        task, context.render_transcript(body), previous,
                        notes_text)
                event["dropped_unsummarized"] = dropped
                result = summarize(request)
                if isinstance(result, str):
                    result = {"content": result}
                result = result or {}
                text = str(result.get("content") or "").strip()
                summary_usage = dict(result.get("usage") or {})
                if result.get("truncated"):
                    error = "summary reached the output limit"
                elif not text:
                    error = "summary was empty"
            except Exception as exc:  # a failed summary must not end the episode
                error = f"{type(exc).__name__}: {exc}"
        if error:
            messages = context.without_older(messages, cut)
            event["fallback"] = True
            event["error"] = error
        else:
            messages = head + [context.summary_message(text)] + tail
            event["summary_chars"] = len(text)
            event["summary"] = text
            event["fallback"] = False
        event["after_tokens"] = context.estimate(messages, specs, None)
        update["anchor"] = None
        update["compactions"] = list(state.get("compactions") or []) + [event]
        if summary_usage:
            update["summary_usage"] = _sum_usage(
                dict(state.get("summary_usage") or {}), summary_usage)
        return messages, update

    def agent(state: State) -> dict:
        if client is None:
            raise RuntimeError("graph.build() needs client=<callable> to run")
        messages = state["messages"]
        specs = tools.SPECS
        update: dict = {"turns": state["turns"] + 1}
        nudge = None
        force = None
        if _final_phase(state):
            if state["turns"] >= policy.MAX_TURNS - 1:
                # The last allowed call: commit. The tool list stays the same;
                # tool_choice is what revokes every route except answer.
                force = "answer"
                nudge = policy.FINAL_ANSWER_NUDGE
            else:
                # Earlier final calls: python is still available, and a call
                # other than answer is required.
                force = "required"
                if not state.get("final_nudged"):
                    nudge = policy.FINAL_NUDGE
                    update["final_nudged"] = True
        else:
            repeat = _repeated_call_nudge(state)
            if repeat:
                nudge = repeat
                update["repeat_nudged"] = state["recent_calls"][-1]
        messages, compacted = _maybe_compact(state, messages, specs)
        update.update(compacted)
        if nudge:
            messages = messages + [{"role": "user", "content": nudge}]
        reply = dict(client(messages, specs, force=force))
        update["messages"] = messages + [reply]
        anchor = _anchor(messages, specs, reply.get("usage"))
        if anchor:
            update["anchor"] = anchor
        if reply.get("usage"):
            update["usage"] = _add_usage(state.get("usage"), reply)
        return update

    def act(state: State) -> dict:
        call_message = state["messages"][-1]
        results = []
        answer = state.get("answer")
        recent = list(state.get("recent_calls") or [])
        for call in call_message.get("tool_calls") or []:
            name = call["function"]["name"]
            recent.append(_call_key(name, call["function"].get("arguments")))
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                if isinstance(args, dict):
                    # Some providers leak whitespace into argument keys, e.g.
                    # ``" path"``. The tool names are fixed, so normalize keys
                    # before dispatch instead of failing on the keyword.
                    args = {str(key).strip(): value for key, value in args.items()}
                if name == "answer":
                    answer = str(args.get("value") or "").strip()
                    output = "episode finished"
                else:
                    if ctx is None:
                        raise RuntimeError("graph.build() needs ctx=<Context> to run")
                    output = tools.dispatch(name, args, ctx)
            except Exception as exc:  # a tool bug is evidence, not a crash
                output = f"error: {type(exc).__name__}: {exc}"
            results.append(_tool_message(call, str(output)))
        update: dict = {"messages": state["messages"] + results,
                        "recent_calls": recent[-_RECENT_CALLS:]}
        if answer is not None:
            update["answer"] = answer
        return update

    def nudge(state: State) -> dict:
        # A reasoning model can spend its whole completion budget and return no
        # content and no call. One bounded re-prompt keeps the episode alive.
        return {"messages": state["messages"] + [
                    {"role": "user", "content": policy.EMPTY_TURN_NUDGE}],
                "empty_retries": state.get("empty_retries", 0) + 1}

    def after_agent(state: State) -> str:
        last = state["messages"][-1]
        # The cap is enforced in after_tools, so the last call still gets its
        # result: a trace that ends on an unanswered call says nothing.
        if last.get("tool_calls"):
            return "tools"
        if (not _empty_reply(last) or state["turns"] >= policy.MAX_TURNS
                or state.get("empty_retries", 0) >= policy.EMPTY_TURN_RETRIES):
            return END
        return "nudge"

    def after_tools(state: State) -> str:
        if state.get("answer") is not None:
            return END
        if state["turns"] >= policy.MAX_TURNS:
            return END
        return "agent"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", act)
    graph.add_node("nudge", nudge)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "nudge", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    graph.add_edge("nudge", "agent")
    return graph.compile()


def run(task: str, client, ctx: tools.Context, summarize=None, notes=None) -> dict:
    """One episode. Returns the final state plus a never-empty ``answer``."""
    state = build(client, ctx, summarize, notes).invoke(
        {"messages": [{"role": "system", "content": prompts.SYSTEM},
                      {"role": "user", "content": task}],
         "task": task, "turns": 0, "empty_retries": 0, "recent_calls": [],
         "repeat_nudged": None, "anchor": None, "compactions": [],
         "summary_usage": {}},
        {"recursion_limit": 2 * policy.MAX_TURNS + 5})
    state["answer"] = state.get("answer") or _last_text(state["messages"]) or "UNKNOWN"
    return state
