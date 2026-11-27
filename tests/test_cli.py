"""Issue #06: the single-task runner, with no model and no network.

    python3 -m pytest -q tests/test_cli.py

A scripted client is the same callable the graph takes: ``(messages, specs) ->
message``. Token usage is per reply and must add up to the endpoint's report.
"""

import json
import os

import pytest
import sys
from pathlib import Path

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"
sys.path.insert(0, str(AGORA))

import cli  # noqa: E402

TASK = cli.tasks.Task("Finance & Economics", "q_0001", "what is the margin?",
                      "10.37 pp", [])
CTX = cli.tools.Context(domain=TASK.domain, data_dir=AGORA, index_dir=AGORA)


def call(name, args, call_id="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def scripted(*replies):
    pending = list(replies)

    def client(messages, specs, force=None):
        return dict(pending.pop(0))
    return client


def test_writes_both_files_and_sums_usage(tmp_path):
    replies = [
        {**call("search_docs", {"query": "margin"}), "usage": {"total_tokens": 7}},
        {**call("answer", {"value": "10.37 pp"}, "c2"), "usage": {"total_tokens": 5}},
    ]
    trajectory = cli.run_task(TASK, tmp_path, trial=2,
                              client=scripted(*replies), ctx=CTX)

    on_disk = json.loads((tmp_path / "trajectory.json").read_text())
    assert on_disk == trajectory
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"
    assert trajectory["status"] == "ok" and trajectory["error"] is None
    assert trajectory["trial"] == 2
    assert trajectory["turns"] == 2
    assert trajectory["usage"]["total_tokens"] == 12  # 7 + 5, as reported
    assert [c["name"] for c in trajectory["tool_calls"]] == ["search_docs", "answer"]
    assert trajectory["tool_calls"][1]["output"] == "episode finished"
    assert [m["role"] for m in trajectory["messages"]][:2] == ["system", "user"]


def test_existing_answer_is_skipped(tmp_path):
    (tmp_path / "answer.txt").write_text("10.37 pp\n")

    def never(messages, specs):
        raise AssertionError("the model must not be called on a finished task")

    trajectory = cli.run_task(TASK, tmp_path, client=never, ctx=CTX)
    assert trajectory["status"] == "skipped"
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"


def test_failure_writes_partial_trajectory_and_exits_nonzero(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "tasks").mkdir(parents=True)
    (data / "tasks" / "Finance.jsonl").write_text(json.dumps({
        "id": TASK.qid, "domain": TASK.domain, "query": TASK.query,
        "answer": TASK.gold}) + "\n")
    monkeypatch.setattr(cli.tasks, "TASKS_DIR", data / "tasks")

    replies = scripted({**call("grep", {"pattern": "margin"}),
                        "usage": {"total_tokens": 4}})
    calls = []

    def flaky(messages, specs, force=None):
        calls.append(messages)
        if len(calls) > 1:
            raise ConnectionError("endpoint down")
        return replies(messages, specs)

    monkeypatch.setattr(cli, "_client", lambda: flaky)
    out = tmp_path / "out"

    assert cli.main(["--task", TASK.id, "--out", str(out)]) == 1
    trajectory = json.loads((out / "trajectory.json").read_text())
    assert trajectory["status"] == "error"
    assert "endpoint down" in trajectory["error"]
    assert trajectory["turns"] == 1
    assert trajectory["usage"]["total_tokens"] == 4
    assert trajectory["messages"][-1]["role"] == "tool"  # the grep result
    assert not (out / "answer.txt").exists()


def test_compaction_does_not_shorten_the_trajectory(tmp_path, monkeypatch):
    # The harness replaces older turns with a summary in its own state. The
    # trajectory must still hold the whole episode, or the analyst reading it
    # never sees the loop that burned the tokens.
    monkeypatch.setattr(cli.policy, "MAX_TURNS", 6)
    monkeypatch.setattr(cli.policy, "FINAL_TURNS", 0)
    monkeypatch.setattr(cli.policy, "CONTEXT_TOKENS", 3000)
    monkeypatch.setattr(cli.policy, "KEEP_RECENT_TOKENS", 400)
    monkeypatch.setattr(cli.policy, "CHARS_PER_TOKEN", 1.0)
    monkeypatch.setattr(cli.tools, "SPECS", [{"type": "function", "function": {
        "name": "grep", "description": "", "parameters": {}}}])
    monkeypatch.setattr(cli.tools, "dispatch",
                        lambda name, args, ctx: "x" * 900)

    requests = []

    def summarize(request):
        requests.append(request)
        return {"content": "## Objective\n- q\n\n## Evidence\n- x",
                "usage": {"total_tokens": 3}}

    replies = [{**call("grep", {"pattern": "m"}, f"c{i}"), "usage": {"total_tokens": 1}}
               for i in range(4)]
    replies.append({**call("answer", {"value": "10.37 pp"}, "c9"),
                    "usage": {"total_tokens": 1}})
    trajectory = cli.run_task(TASK, tmp_path, client=scripted(*replies),
                              ctx=CTX, summarize=summarize,
                              notes=lambda: "notes evidence")

    assert trajectory["turns"] == 5
    assert trajectory["compactions"]
    assert "notes evidence" in requests[0]
    assert [c["name"] for c in trajectory["tool_calls"]] == ["grep"] * 4 + ["answer"]
    assert [len(c["output"]) for c in trajectory["tool_calls"]] == [900] * 4 + [16]
    assert [m["role"] for m in trajectory["messages"]].count("tool") == 5
    assert trajectory["usage"]["total_tokens"] == 5 + 3 * len(trajectory["compactions"])
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"


def test_python_scratch_is_per_episode(tmp_path, monkeypatch):
    monkeypatch.delenv("AGORA_SCRATCH", raising=False)
    data = tmp_path / "data"
    (data / "documents" / TASK.domain).mkdir(parents=True)
    ctx = cli.tools.Context(domain=TASK.domain, data_dir=data, index_dir=AGORA)

    first = scripted(
        call("python", {"code": "open('notes.md', 'w').write('first episode')"}, "c1"),
        call("answer", {"value": "10.37 pp"}, "c2"))
    cli.run_task(TASK, tmp_path / "one", client=first, ctx=ctx)
    assert (tmp_path / "one" / "scratch" / "notes.md").read_text() == "first episode"
    assert "AGORA_SCRATCH" not in os.environ

    second = scripted(
        call("python", {"code": "open('notes.md').read()"}, "c3"),
        call("answer", {"value": "10.37 pp"}, "c4"))
    trajectory = cli.run_task(TASK, tmp_path / "two", client=second, ctx=ctx)

    assert "FileNotFoundError" in trajectory["tool_calls"][0]["output"]
    assert not (tmp_path / "two" / "scratch" / "notes.md").exists()
    assert "AGORA_SCRATCH" not in os.environ


def test_main_resolves_the_task_id(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "tasks").mkdir(parents=True)
    (data / "tasks" / "Finance.jsonl").write_text(json.dumps({
        "id": TASK.qid, "domain": TASK.domain, "query": TASK.query,
        "answer": TASK.gold}) + "\n")
    monkeypatch.setattr(cli.tasks, "TASKS_DIR", data / "tasks")
    monkeypatch.setenv("AGORA_DOMAIN", TASK.domain)
    monkeypatch.setattr(cli, "_client",
                        lambda: scripted({**call("answer", {"value": "10.37 pp"})}))

    assert cli.main(["--task", TASK.qid, "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "answer.txt").read_text() == "10.37 pp\n"

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--task", "Nope/q_9", "--out", str(tmp_path / "x")])
    assert exit_info.value.code == 2


def test_interrupt_still_writes_a_partial_trajectory(tmp_path):
    def interrupted(messages, specs, force=None):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run_task(TASK, tmp_path, client=interrupted, ctx=CTX)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["status"] == "error"
    assert trajectory["error"].startswith("KeyboardInterrupt")
    assert not (tmp_path / "answer.txt").exists()


def _reply(content="10.37 pp", finish_reason="stop"):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason=finish_reason)],
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 1}))


def _fake_endpoint(monkeypatch, completions):
    """Replace the SDK endpoint; ``_client`` imports it at call time."""
    from types import SimpleNamespace

    import openai

    class Endpoint:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr(openai, "OpenAI", Endpoint)
    return openai


def test_client_maps_force_to_tool_choice(monkeypatch):
    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return _reply()

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("RRSI_API_KEY", "dummy")
    client = cli._client()

    client([{"role": "user", "content": "q"}], cli.tools.SPECS, force="required")
    client([{"role": "user", "content": "q"}], cli.tools.SPECS, force="answer")
    client([{"role": "user", "content": "q"}], cli.tools.SPECS)

    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "answer"}}
    assert "tool_choice" not in requests[2]
    assert requests[0]["tools"] == requests[1]["tools"] == cli.tools.SPECS


def test_client_reports_a_rejected_tool_choice_and_retries_once(monkeypatch, capsys):
    class Rejected(Exception):
        pass

    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            if "tool_choice" in kwargs:
                raise Rejected("tool_choice is not supported")
            return _reply()

    openai = _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(openai, "BadRequestError", Rejected)
    monkeypatch.setattr(openai, "UnprocessableEntityError", Rejected)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("RRSI_API_KEY", "dummy")

    reply = cli._client()([{"role": "user", "content": "q"}],
                          cli.tools.SPECS, force="answer")

    assert reply["content"] == "10.37 pp"
    assert len(requests) == 2
    assert "tool_choice" in requests[0]
    assert "tool_choice" not in requests[1]
    assert "rejected tool_choice" in capsys.readouterr().err


def test_summarizer_has_no_tools_and_keeps_thinking_on(monkeypatch):
    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return _reply("checkpoint")

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("AGORA_SUMMARY_REASONING_EFFORT", raising=False)

    result = cli._summarizer()("compress this")

    assert result == {"content": "checkpoint",
                      "usage": {"total_tokens": 1},
                      "truncated": False}
    assert "tools" not in requests[0]
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["max_tokens"] == cli.policy.SUMMARY_MAX_TOKENS
    assert "chat_template_kwargs" not in requests[0]


def test_summarizer_flags_a_truncated_checkpoint(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            return _reply("partial", finish_reason="length")

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    assert cli._summarizer()("compress")["truncated"] is True


def test_summarizer_passes_the_effort_override(monkeypatch):
    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return _reply("checkpoint")

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("AGORA_SUMMARY_REASONING_EFFORT", "none")

    cli._summarizer()("compress")

    assert requests[0]["chat_template_kwargs"] == {"reasoning_effort": "none"}
