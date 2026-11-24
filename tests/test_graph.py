"""Issue #05: the graph loop and tool surface, with no model and no network.

    python3 -m pytest -q tests/test_graph.py

A scripted client is a plain callable ``(messages, tool_specs) -> message``, so
every test controls the episode turn by turn.
"""

import json
import sys
from pathlib import Path

import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"
sys.path.insert(0, str(AGORA))

import harness.context as context  # noqa: E402
import harness.graph as graph  # noqa: E402
import harness.policy as policy  # noqa: E402
import harness.tools as tools  # noqa: E402


def call(name, args, call_id="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def scripted(*replies):
    """A client that returns the scripted replies in order."""
    pending = list(replies)
    calls = []

    def client(messages, specs):
        calls.append(messages)
        return dict(pending.pop(0))

    client.calls = calls
    return client


CTX = tools.Context(domain="Finance & Economics")


def test_answer_tool_terminates():
    client = scripted(call("answer", {"value": "10.37 pp"}))
    state = graph.run("what is the margin?", client, CTX)
    assert state["answer"] == "10.37 pp"
    assert state["turns"] == 1
    assert state["messages"][-1]["role"] == "tool"


def test_stops_at_max_turns(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    client = scripted(*[call("nope", {}) for _ in range(10)])
    state = graph.run("never answered", client, CTX)
    assert state["turns"] == 3
    assert len(client.calls) == 3
    assert state["answer"] == "UNKNOWN"
    # the last call still gets its result, so the trace has no dangling call
    assert [m["role"] for m in state["messages"]][-2:] == ["assistant", "tool"]


def test_unknown_tool_is_an_error_string():
    client = scripted(call("teleport", {}), call("answer", {"value": "1"}))
    state = graph.run("q", client, CTX)
    errors = [m["content"] for m in state["messages"] if m["role"] == "tool"]
    assert errors and errors[0].startswith("error: ValueError: unknown tool")
    assert state["answer"] == "1"


def test_tool_exception_is_an_error_string(monkeypatch):
    def boom(name, args, ctx):
        raise RuntimeError("index is on fire")

    monkeypatch.setattr(tools, "dispatch", boom)
    client = scripted(call("grep", {"pattern": "x"}), call("answer", {"value": "2"}))
    state = graph.run("q", client, CTX)
    errors = [m["content"] for m in state["messages"] if m["role"] == "tool"]
    assert "RuntimeError: index is on fire" in errors[0]
    assert state["answer"] == "2"


def test_usage_is_summed_but_not_sent_back():
    client = scripted({**call("nope", {}), "usage": {"total_tokens": 11}},
                      {**call("answer", {"value": "3"}), "usage": {"total_tokens": 7}})
    state = graph.run("q", client, CTX)
    assert state["usage"] == {"total_tokens": 18}
    assert all("usage" not in m for m in state["messages"])


def test_answer_on_the_last_allowed_turn_is_captured(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 1)
    state = graph.run("q", scripted(call("answer", {"value": "42"})), CTX)
    assert state["answer"] == "42"


def test_content_reply_without_tools_ends_the_episode():
    state = graph.run("q", scripted({"role": "assistant", "content": " 42 "}), CTX)
    assert state["answer"] == "42"


def test_empty_turn_is_nudged_then_recovers():
    client = scripted({"role": "assistant", "content": None},
                      call("answer", {"value": "7"}))
    state = graph.run("q", client, CTX)
    assert state["answer"] == "7"
    assert state["turns"] == 2
    assert state["empty_retries"] == 1
    assert any(message["role"] == "user"
               and message["content"] == policy.EMPTY_TURN_NUDGE
               for message in client.calls[1])


def test_whitespace_reply_is_an_empty_turn():
    client = scripted({"role": "assistant", "content": "   "},
                      call("answer", {"value": "8"}))
    state = graph.run("q", client, CTX)
    assert state["answer"] == "8"
    assert state["empty_retries"] == 1


def test_empty_turns_stop_after_the_retry_bound(monkeypatch):
    monkeypatch.setattr(policy, "EMPTY_TURN_RETRIES", 2)
    client = scripted(*[{"role": "assistant", "content": None} for _ in range(5)])
    state = graph.run("q", client, CTX)
    assert state["answer"] == "UNKNOWN"
    assert state["turns"] == 3
    assert len(client.calls) == 3


def test_empty_turn_at_the_turn_cap_does_not_retry(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 1)
    client = scripted({"role": "assistant", "content": None})
    state = graph.run("q", client, CTX)
    assert state["answer"] == "UNKNOWN"
    assert state["turns"] == 1
    assert len(client.calls) == 1


def test_trim_drops_oldest_tool_block_and_keeps_system():
    big = {"role": "tool", "tool_call_id": "a", "content": "x" * 500}
    messages = [
        {"role": "system", "content": "S"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        big,
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "b", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "y"},
    ]
    kept = context.trim(messages, budget=600)
    assert kept[0] == messages[0]
    assert [m.get("tool_call_id") for m in kept if m["role"] == "tool"] == ["b"]


def test_read_file_stays_inside_the_domain_docs(tmp_path):
    docs = tmp_path / "documents" / "D"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "D.jsonl").write_text('{"answer": "42"}', encoding="utf-8")
    ctx = tools.Context("D", data_dir=tmp_path)
    assert "two" in tools.read_file("documents/D/a.md", 1, 2, ctx=ctx)
    assert "two" in tools.read_file("a.md", 1, 2, ctx=ctx)
    with pytest.raises((ValueError, FileNotFoundError)):
        tools.read_file("tasks/D.jsonl", 1, 2, ctx=ctx)
    with pytest.raises(ValueError):
        tools.read_file("../../etc/passwd", 1, 2, ctx=ctx)


def test_python_sees_documents_only(tmp_path):
    (tmp_path / "documents" / "D").mkdir(parents=True)
    ctx = tools.Context("D", data_dir=tmp_path)
    out = tools.python("import os; print(os.listdir(os.environ['AGORA_DATA']))", ctx=ctx)
    assert "documents" in out and "tasks" not in out


FORBIDDEN = ("reference_files", "agora_tasks", "task_id", "gold")


def test_harness_cannot_reach_gold_or_tasks():
    for path in (AGORA / "harness").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            assert word not in text, f"{word} reachable from {path.name}"
