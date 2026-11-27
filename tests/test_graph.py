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
    calls, specs_seen, forces = [], [], []

    def client(messages, specs, force=None):
        calls.append(messages)
        specs_seen.append(specs)
        forces.append(force)
        return dict(pending.pop(0))

    client.calls = calls
    client.specs = specs_seen
    client.forces = forces
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
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
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
    client = scripted(
        {**call("nope", {}),
         "usage": {"total_tokens": 11,
                   "prompt_tokens_details": {"cached_tokens": 4}}},
        {**call("answer", {"value": "3"}), "usage": {"total_tokens": 7}})
    state = graph.run("q", client, CTX)
    assert state["usage"] == {"total_tokens": 18, "cached_tokens": 4}
    assert all("usage" not in m for m in state["messages"])


def test_answer_on_the_last_allowed_turn_is_captured(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 1)
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
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
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
    client = scripted({"role": "assistant", "content": None})
    state = graph.run("q", client, CTX)
    assert state["answer"] == "UNKNOWN"
    assert state["turns"] == 1
    assert len(client.calls) == 1


def test_duplicate_call_appends_one_repeat_nudge(monkeypatch):
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "same")
    client = scripted(call("grep", {"pattern": "x"}, "c1"),
                      call("grep", {"pattern": "x"}, "c2"),
                      call("answer", {"value": "1"}, "c3"))
    state = graph.run("q", client, CTX)

    nudges = [m for m in state["messages"]
              if m.get("content") == policy.REPEAT_NUDGE]
    assert len(nudges) == 1
    assert client.calls[2][-1] == {"role": "user",
                                   "content": policy.REPEAT_NUDGE}
    assert state["answer"] == "1"


def test_repeated_identical_calls_produce_one_nudge(monkeypatch):
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "same")
    replies = [call("grep", {"pattern": "x"}, f"c{i}") for i in range(4)]
    replies.append(call("answer", {"value": "1"}, "c5"))
    state = graph.run("q", scripted(*replies), CTX)

    assert sum(m.get("content") == policy.REPEAT_NUDGE
               for m in state["messages"]) == 1


def test_distinct_calls_do_not_append_the_repeat_nudge(monkeypatch):
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    client = scripted(call("grep", {"pattern": "x"}, "c1"),
                      call("grep", {"pattern": "y"}, "c2"),
                      call("grep", {"pattern": "x", "path": "b.md"}, "c3"),
                      call("answer", {"value": "1"}, "c4"))
    state = graph.run("q", client, CTX)

    assert not any(m.get("content") == policy.REPEAT_NUDGE
                   for m in state["messages"])


def test_repeat_detection_normalizes_argument_order_and_keys(monkeypatch):
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    client = scripted(call("grep", {" pattern": "x", "path": "a.md"}, "c1"),
                      call("grep", {"path": "a.md", "pattern": "x"}, "c2"),
                      call("answer", {"value": "1"}, "c3"))
    state = graph.run("q", client, CTX)

    assert sum(m.get("content") == policy.REPEAT_NUDGE
               for m in state["messages"]) == 1


def test_recent_calls_keeps_the_last_twenty(monkeypatch):
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    replies = [call("grep", {"pattern": f"q{i}"}, f"c{i}") for i in range(25)]
    replies.append(call("answer", {"value": "1"}, "c26"))
    state = graph.run("q", scripted(*replies), CTX)

    assert len(state["recent_calls"]) == 20


def test_repeat_nudge_is_not_appended_in_the_final_phase(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    monkeypatch.setattr(policy, "FINAL_TURNS", 2)
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    client = scripted(call("grep", {"pattern": "x"}, "c1"),
                      call("grep", {"pattern": "x"}, "c2"),
                      call("answer", {"value": "1"}, "c3"))
    state = graph.run("q", client, CTX)

    contents = [m.get("content") for m in state["messages"]]
    assert policy.REPEAT_NUDGE not in contents
    assert contents.count(policy.FINAL_NUDGE) == 1
    assert client.calls[-1][-1] == {"role": "user",
                                    "content": policy.FINAL_ANSWER_NUDGE}


def test_repeated_calls_stay_bounded_by_max_turns(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    replies = [call("grep", {"pattern": "x"}, f"c{i}") for i in range(10)]
    client = scripted(*replies)
    state = graph.run("q", client, CTX)

    assert state["turns"] == 3
    assert len(client.calls) == 3
    assert state["answer"] == "UNKNOWN"


def test_final_phase_keeps_the_tool_list_and_restricts_the_choice(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    monkeypatch.setattr(policy, "FINAL_TURNS", 2)
    client = scripted(call("nope", {}, "c1"), call("answer", {"value": "4"}, "c2"))
    state = graph.run("q", client, CTX)

    # The declaration is rendered before the first message; a changed list would
    # invalidate the whole prompt cache. tool_choice does the restriction.
    assert client.specs[0] == tools.SPECS
    assert client.specs[1] == tools.SPECS
    assert client.forces[1] == "required"
    assert client.calls[1][-1] == {"role": "user",
                                   "content": policy.FINAL_NUDGE}
    assert state["turns"] == 2
    assert state["final_nudged"] is True
    assert state["answer"] == "4"
    assert len(client.calls) == 2  # no turn after the answer


def test_final_phase_allows_python_on_the_first_final_turn_then_forces_answer(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    monkeypatch.setattr(policy, "FINAL_TURNS", 2)
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "computed 4")
    client = scripted(call("python", {"code": "collect"}, "c1"),
                      call("python", {"code": "compute"}, "c2"),
                      call("answer", {"value": "4"}, "c3"))
    state = graph.run("q", client, CTX)

    assert client.forces[1] == "required"
    assert client.forces[2] == "answer"
    assert client.calls[2][-1] == {"role": "user",
                                   "content": policy.FINAL_ANSWER_NUDGE}
    results = [m for m in state["messages"] if m.get("role") == "tool"]
    assert results[-2]["content"] == "computed 4"
    assert state["turns"] == 3
    assert state["answer"] == "4"


def test_final_phase_allows_five_python_turns_before_the_forced_answer(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 7)
    monkeypatch.setattr(policy, "FINAL_TURNS", 6)
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "computed 4")
    replies = [call("python", {"code": "compute"}, f"c{i}") for i in range(1, 7)]
    replies.append(call("answer", {"value": "4"}, "c7"))
    client = scripted(*replies)
    state = graph.run("q", client, CTX)

    assert client.specs == [tools.SPECS] * 7
    assert client.forces[0] is None
    assert client.forces[1:6] == ["required"] * 5
    assert client.forces[6] == "answer"
    assert state["turns"] == 7
    assert state["answer"] == "4"


def test_final_phase_appends_the_nudge_once(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 3)
    monkeypatch.setattr(policy, "FINAL_TURNS", 2)
    client = scripted(call("nope", {}, "c1"), call("nope", {}, "c2"),
                      call("nope", {}, "c3"))
    state = graph.run("q", client, CTX)

    nudges = [m for m in state["messages"]
              if m.get("content") == policy.FINAL_NUDGE]
    assert len(nudges) == 1
    assert client.forces[1] == "required"
    assert client.forces[2] == "answer"
    assert client.calls[2][-1] == {"role": "user",
                                   "content": policy.FINAL_ANSWER_NUDGE}
    assert state["turns"] == 3
    assert state["answer"] == "UNKNOWN"


def test_final_phase_content_reply_is_the_text_fallback(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 2)
    monkeypatch.setattr(policy, "FINAL_TURNS", 2)
    client = scripted({"role": "assistant", "content": " 42 "})
    state = graph.run("q", client, CTX)

    assert client.specs[0] == tools.SPECS
    assert client.forces[0] == "required"
    assert state["turns"] == 1
    assert state["answer"] == "42"


def test_final_answer_nudge_forces_the_answer_and_keeps_the_text_fallback(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 1)
    monkeypatch.setattr(policy, "FINAL_TURNS", 1)
    client = scripted({"role": "assistant", "content": " 42 "})
    state = graph.run("q", client, CTX)

    assert client.specs[0] == tools.SPECS
    assert client.forces[0] == "answer"
    assert client.calls[0][-1] == {"role": "user",
                                   "content": policy.FINAL_ANSWER_NUDGE}
    assert "Only the answer tool" in policy.FINAL_ANSWER_NUDGE
    assert state["turns"] == 1
    assert state["answer"] == "42"


def test_cut_point_keeps_tool_results_with_their_call():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "x" * 500},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "b", "type": "function",
             "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "y"},
    ]
    cut = context.cut_point(messages, keep_tokens=1, ratio=1.0)
    head, older, tail = context.split(messages, cut)

    assert cut == 4
    assert head == messages[:2]
    assert len(older) == 2
    # The tail never starts with a tool result: the call and its result stay
    # together, or the provider rejects the transcript.
    assert tail[0]["role"] == "assistant"
    assert [m.get("tool_call_id") for m in tail if m["role"] == "tool"] == ["b"]


def test_estimate_anchors_on_the_measured_prompt_size():
    messages = [{"role": "system", "content": "x" * 100}]
    anchor = {"chars": 100, "tokens": 40}
    assert context.estimate(messages, None, anchor) == 40
    messages.append({"role": "user", "content": "y" * 100})
    # 100 new chars at the fallback 2.5 chars/token.
    assert context.estimate(messages, None, anchor) == 80


def test_previous_summary_is_reused_not_resummarized():
    message = context.summary_message("## Objective\n- q")
    assert message["role"] == "user"
    assert "## Objective" in context.previous_summary([message])
    assert context.render_transcript([message]) == ""


def test_prefix_is_append_only_between_turns(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 4)
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "ok")
    client = scripted(call("grep", {"pattern": "a"}, "c1"),
                      call("grep", {"pattern": "b"}, "c2"),
                      call("answer", {"value": "1"}, "c3"))
    graph.run("q", client, CTX)

    for previous, current in zip(client.calls, client.calls[1:]):
        assert current[:len(previous)] == previous


def test_budget_compacts_older_turns_into_one_summary(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 8)
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
    monkeypatch.setattr(policy, "CONTEXT_TOKENS", 3000)
    monkeypatch.setattr(policy, "KEEP_RECENT_TOKENS", 400)
    monkeypatch.setattr(policy, "CHARS_PER_TOKEN", 1.0)
    monkeypatch.setattr(tools, "SPECS", [{"type": "function", "function": {
        "name": "grep", "description": "", "parameters": {}}}])
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "x" * 900)

    requests = []

    def summarize(request):
        requests.append(request)
        return {"content": "## Objective\n- q\n\n## Evidence\n- x"}

    replies = [call("grep", {"pattern": f"q{i}"}, f"c{i}") for i in range(5)]
    replies.append(call("answer", {"value": "1"}, "c9"))
    state = graph.run("q", scripted(*replies), CTX, summarize=summarize)

    assert requests and "x" * 900 in requests[0]
    event = state["compactions"][0]
    assert event["fallback"] is False
    assert event["dropped_messages"] >= 2
    assert event["before_tokens"] > event["after_tokens"]
    messages = state["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert context.SUMMARY_PREFIX in messages[2]["content"]
    assert messages[3]["role"] == "assistant"
    assert state["answer"] == "1"


def test_failed_summary_falls_back_and_keeps_the_episode_alive(monkeypatch):
    monkeypatch.setattr(policy, "MAX_TURNS", 8)
    monkeypatch.setattr(policy, "FINAL_TURNS", 0)
    monkeypatch.setattr(policy, "CONTEXT_TOKENS", 3000)
    monkeypatch.setattr(policy, "KEEP_RECENT_TOKENS", 400)
    monkeypatch.setattr(policy, "CHARS_PER_TOKEN", 1.0)
    monkeypatch.setattr(tools, "SPECS", [{"type": "function", "function": {
        "name": "grep", "description": "", "parameters": {}}}])
    monkeypatch.setattr(tools, "dispatch", lambda name, args, ctx: "x" * 900)

    replies = [call("grep", {"pattern": f"q{i}"}, f"c{i}") for i in range(5)]
    replies.append(call("answer", {"value": "1"}, "c9"))
    state = graph.run("q", scripted(*replies), CTX,
                      summarize=lambda request: {"content": "partial",
                                                 "truncated": True})

    event = state["compactions"][0]
    assert event["fallback"] is True
    assert "output limit" in event["error"]
    assert not any(context.SUMMARY_PREFIX in str(m.get("content"))
                   for m in state["messages"])
    assert state["answer"] == "1"


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


@pytest.mark.parametrize("path", [
    "a.md",
    "D/a.md",
    "documents/a.md",
    "documents/D/a.md",
    "/data/documents/D/a.md",
    "D/D/a.md",
    "documents/D/D/a.md",
])
def test_read_file_accepts_every_document_path_form(tmp_path, path):
    docs = tmp_path / "documents" / "D"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("one\ntwo\n", encoding="utf-8")
    ctx = tools.Context("D", data_dir=tmp_path)
    assert "two" in tools.read_file(path, 1, 2, ctx=ctx)


@pytest.mark.parametrize("path", [
    "/other/D/a.md",
    "/data/documents/Other/a.md",
    "../../etc/passwd",
    "documents/../../etc/passwd",
    "documents/D/../../../etc/passwd",
])
def test_read_file_rejects_paths_outside_the_domain(tmp_path, path):
    (tmp_path / "documents" / "D").mkdir(parents=True)
    ctx = tools.Context("D", data_dir=tmp_path)
    with pytest.raises(ValueError):
        tools.read_file(path, 1, 2, ctx=ctx)


def test_grep_accepts_a_single_file_path(tmp_path):
    docs = tmp_path / "documents" / "D"
    (docs / "sub").mkdir(parents=True)
    (docs / "a.md").write_text("needle\n", encoding="utf-8")
    (docs / "b.csv").write_text("needle\n", encoding="utf-8")
    (docs / "sub" / "c.md").write_text("needle\n", encoding="utf-8")
    ctx = tools.Context("D", data_dir=tmp_path)

    out = tools.grep("needle", path="documents/D/a.md", ctx=ctx)

    assert "a.md:1:needle" in out
    assert "b.csv" not in out and "c.md" not in out


def test_grep_maps_a_non_file_path_to_glob(tmp_path):
    docs = tmp_path / "documents" / "D"
    (docs / "sub").mkdir(parents=True)
    (docs / "a.md").write_text("needle\n", encoding="utf-8")
    (docs / "b.csv").write_text("needle\n", encoding="utf-8")
    (docs / "sub" / "c.md").write_text("needle\n", encoding="utf-8")
    ctx = tools.Context("D", data_dir=tmp_path)

    out = tools.grep("needle", path="*.md", ctx=ctx)

    assert "a.md:1:needle" in out
    assert "sub/c.md:1:needle" in out
    assert "b.csv" not in out


def test_grep_path_cannot_escape_the_domain(tmp_path):
    (tmp_path / "documents" / "D").mkdir(parents=True)
    ctx = tools.Context("D", data_dir=tmp_path)
    with pytest.raises(ValueError):
        tools.grep("x", path="/etc/passwd", ctx=ctx)
    with pytest.raises(ValueError):
        tools.grep("x", path="../../etc/passwd", ctx=ctx)


def test_leading_space_argument_key_is_stripped(monkeypatch):
    seen = {}

    def capture(name, args, ctx):
        seen.update(args)
        return "ok"

    monkeypatch.setattr(tools, "dispatch", capture)
    client = scripted(call("read_file", {" path": "a.md", "start": 1}),
                      call("answer", {"value": "1"}))
    graph.run("q", client, CTX)
    assert seen == {"path": "a.md", "start": 1}


@pytest.mark.parametrize("name", ["bash", "shell"])
def test_unknown_shell_tools_remain_errors(name):
    client = scripted(call(name, {"command": "ls"}), call("answer", {"value": "1"}))
    state = graph.run("q", client, CTX)
    errors = [m["content"] for m in state["messages"] if m["role"] == "tool"]
    assert errors[0].startswith(f"error: ValueError: unknown tool: {name}")
    assert state["answer"] == "1"


def test_python_sees_documents_only(tmp_path, monkeypatch):
    monkeypatch.delenv("AGORA_SCRATCH", raising=False)
    (tmp_path / "documents" / "D").mkdir(parents=True)
    ctx = tools.Context("D", data_dir=tmp_path)
    out = tools.python("import os; print(os.listdir(os.environ['AGORA_DATA']))", ctx=ctx)
    assert "documents" in out and "tasks" not in out


def test_python_calls_share_one_scratchpad(tmp_path, monkeypatch):
    (tmp_path / "documents" / "D").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("AGORA_SCRATCH", str(scratch))
    ctx = tools.Context("D", data_dir=tmp_path)

    first = tools.python(
        "import os; from pathlib import Path; "
        "Path('notes.md').write_text('q -> 12 pp\\n'); "
        "print(os.getcwd()); print(Path('documents/D').is_dir())", ctx=ctx)
    second = tools.python(
        "import os; from pathlib import Path; "
        "print(Path('notes.md').read_text(), end=''); "
        "print(Path('documents').is_symlink()); "
        "print('AGORA_SCRATCH' in os.environ)", ctx=ctx)

    assert Path(first.splitlines()[0]) == scratch.resolve()
    assert first.splitlines()[1] == "True"
    assert second == "q -> 12 pp\nTrue\nFalse\n"


def test_python_repairs_a_stale_documents_link(tmp_path, monkeypatch):
    (tmp_path / "documents" / "D").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "documents").symlink_to(tmp_path / "gone")
    monkeypatch.setenv("AGORA_SCRATCH", str(scratch))
    ctx = tools.Context("D", data_dir=tmp_path)

    out = tools.python("from pathlib import Path; print(Path('documents/D').is_dir())",
                       ctx=ctx)

    assert out.strip() == "True"
    assert (scratch / "documents").resolve() == (tmp_path / "documents").resolve()


FORBIDDEN = ("reference_files", "agora_tasks", "task_id", "gold")


def test_harness_cannot_reach_gold_or_tasks():
    for path in (AGORA / "harness").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            assert word not in text, f"{word} reachable from {path.name}"
