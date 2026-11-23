"""LangGraph loop: agent -> tools -> agent, until ``answer`` or MAX_TURNS.

The model client is injected: ``client(messages, tool_specs) -> message`` where
message is an OpenAI-style assistant dict (``content`` and ``tool_calls``), with
an optional ``usage`` key that is summed into the trajectory and never sent
back to the model. Tests pass a scripted callable; cli.py passes one that wraps
the OpenAI client.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import context, policy, prompts, tools


class State(TypedDict, total=False):
    messages: list[dict]
    turns: int
    answer: str
    usage: dict


def _tool_message(call: dict, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call.get("id") or "",
            "content": content}


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


def build(client=None, ctx: tools.Context | None = None):
    """Compile the graph. ``client`` is only needed when it is invoked.

    ``run()`` sets the recursion limit to match MAX_TURNS; a direct ``invoke``
    keeps LangGraph's default and would stop after ~12 turns.
    """

    def agent(state: State) -> dict:
        if client is None:
            raise RuntimeError("graph.build() needs client=<callable> to run")
        messages = context.trim(state["messages"])
        reply = dict(client(messages, tools.SPECS))
        update: dict = {"messages": messages + [reply],
                        "turns": state["turns"] + 1}
        if reply.get("usage"):
            update["usage"] = _add_usage(state.get("usage"), reply)
        return update

    def act(state: State) -> dict:
        call_message = state["messages"][-1]
        results = []
        answer = state.get("answer")
        for call in call_message.get("tool_calls") or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
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
        update: dict = {"messages": state["messages"] + results}
        if answer is not None:
            update["answer"] = answer
        return update

    def after_agent(state: State) -> str:
        # The cap is enforced in after_tools, so the last call still gets its
        # result: a trace that ends on an unanswered call says nothing.
        return "tools" if state["messages"][-1].get("tool_calls") else END

    def after_tools(state: State) -> str:
        if state.get("answer") is not None:
            return END
        if state["turns"] >= policy.MAX_TURNS:
            return END
        return "agent"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", act)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def run(task: str, client, ctx: tools.Context) -> dict:
    """One episode. Returns the final state plus a never-empty ``answer``."""
    state = build(client, ctx).invoke(
        {"messages": [{"role": "system", "content": prompts.SYSTEM},
                      {"role": "user", "content": task}],
         "turns": 0},
        {"recursion_limit": 2 * policy.MAX_TURNS + 5})
    state["answer"] = state.get("answer") or _last_text(state["messages"]) or "UNKNOWN"
    return state
