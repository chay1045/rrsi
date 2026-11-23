"""Issue #06: the single-task runner, with no model and no network.

    python3 -m pytest -q tests/test_cli.py

A scripted client is the same callable the graph takes: ``(messages, specs) ->
message``. Token usage is per reply and must add up to the endpoint's report.
"""

import json

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

    def client(messages, specs):
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

    def flaky(messages, specs):
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


def test_context_trimming_does_not_shorten_the_trajectory(tmp_path, monkeypatch):
    # The harness trims old turns out of its state to stay under the char
    # budget. The trajectory must still hold the whole episode, or the analyst
    # reading it never sees the loop that burned the tokens.
    monkeypatch.setattr(cli.policy, "MAX_TURNS", 5)
    monkeypatch.setattr(cli.policy, "CONTEXT_CHARS", 4000)
    monkeypatch.setattr(cli.tools, "dispatch",
                        lambda name, args, ctx: "x" * 3000)

    replies = [{**call("grep", {"pattern": "m"}, f"c{i}"), "usage": {"total_tokens": 1}}
               for i in range(4)]
    replies.append({**call("answer", {"value": "10.37 pp"}, "c9"),
                    "usage": {"total_tokens": 1}})
    trajectory = cli.run_task(TASK, tmp_path, client=scripted(*replies), ctx=CTX)

    assert trajectory["turns"] == 5
    assert [c["name"] for c in trajectory["tool_calls"]] == ["grep"] * 4 + ["answer"]
    assert [len(c["output"]) for c in trajectory["tool_calls"]] == [3000] * 4 + [16]
    assert [m["role"] for m in trajectory["messages"]].count("tool") == 5
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"


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
    def interrupted(messages, specs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run_task(TASK, tmp_path, client=interrupted, ctx=CTX)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["status"] == "error"
    assert trajectory["error"].startswith("KeyboardInterrupt")
    assert not (tmp_path / "answer.txt").exists()
