"""LangGraph loop: agent -> tools -> agent, until ``answer`` or MAX_TURNS.

The model client is injected: ``client(messages, tool_specs) -> message`` where
message is an OpenAI-style assistant dict (``content`` and ``tool_calls``), with
an optional ``usage`` key that is summed into the trajectory and never sent
back to the model. Tests pass a scripted callable; cli.py passes one that wraps
the OpenAI client.

The last ``policy.FINAL_TURNS`` calls form a bounded final phase: one nudge is
appended; earlier final calls may use python, and the last call receives only
the answer tool.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import context, policy, prompts, tools


_RECENT_CALLS = 20       # exact-repeat detection window


class State(TypedDict, total=False):
    messages: list[dict]
    turns: int
    empty_retries: int
    final_nudged: bool
    recent_calls: list[str]
    repeat_nudged: str | None
    answer: str
    usage: dict


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


def _add_usage(usage: dict, reply: dict) -> dict:
    used = dict(usage or {})
    for key, value in (reply.pop("usage", None) or {}).items():
        if isinstance(value, (int, float)):
            used[key] = used.get(key, 0) + value
    return used


def _final_phase(state: State) -> bool:
    """The bounded last turns: the client may only submit an answer."""
    return (policy.FINAL_TURNS > 0
            and state["turns"] >= policy.MAX_TURNS - policy.FINAL_TURNS)


def build(client=None, ctx: tools.Context | None = None):
    """Compile the graph. ``client`` is only needed when it is invoked.

    ``run()`` sets the recursion limit to match MAX_TURNS; a direct ``invoke``
    keeps LangGraph's default and would stop after ~12 turns.
    """

    def agent(state: State) -> dict:
        if client is None:
            raise RuntimeError("graph.build() needs client=<callable> to run")
        messages = context.trim(state["messages"])
        specs = tools.SPECS
        update: dict = {"turns": state["turns"] + 1}
        if _final_phase(state):
            if state["turns"] >= policy.MAX_TURNS - 1:
                # The last allowed call: commit. Revoke python explicitly; the
                # earlier nudge still permits it and the model may follow that.
                specs = [tools.ANSWER_SPEC]
                messages = messages + [{"role": "user",
                                        "content": policy.FINAL_ANSWER_NUDGE}]
            else:
                # Earlier final calls: allow python, but only answer or python.
                specs = tools.FINAL_SPECS
                if not state.get("final_nudged"):
                    messages = messages + [{"role": "user",
                                            "content": policy.FINAL_NUDGE}]
                    update["final_nudged"] = True
        else:
            repeat = _repeated_call_nudge(state)
            if repeat:
                messages = messages + [{"role": "user", "content": repeat}]
                update["repeat_nudged"] = state["recent_calls"][-1]
        reply = dict(client(messages, specs))
        update["messages"] = messages + [reply]
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


def run(task: str, client, ctx: tools.Context) -> dict:
    """One episode. Returns the final state plus a never-empty ``answer``."""
    state = build(client, ctx).invoke(
        {"messages": [{"role": "system", "content": prompts.SYSTEM},
                      {"role": "user", "content": task}],
         "turns": 0, "empty_retries": 0, "recent_calls": [],
         "repeat_nudged": None},
        {"recursion_limit": 2 * policy.MAX_TURNS + 5})
    state["answer"] = state.get("answer") or _last_text(state["messages"]) or "UNKNOWN"
    return state
