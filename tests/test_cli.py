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
    assert trajectory["retrieval"]["rerank_calls"] == 0
    assert trajectory["retrieval"]["rerank_fallbacks"] == 0
    assert [c["name"] for c in trajectory["tool_calls"]] == ["search_docs", "answer"]
    assert trajectory["tool_calls"][1]["output"] == "episode finished"
    assert [m["role"] for m in trajectory["messages"]][:2] == ["system", "user"]


def test_usage_reaches_the_graph_and_stays_out_of_messages(tmp_path, monkeypatch):
    seen = {}

    def fake_run(task, client, ctx, summarize=None, notes=None, count=None):
        reply = client([{"role": "user", "content": "q"}], [], force=None)
        seen["reply"] = reply
        seen["count"] = count
        return {"messages": [reply], "turns": 1, "answer": "10.37 pp",
                "usage": {"total_tokens": 5, "cached_tokens": 2},
                "compactions": [], "summary_usage": {},
                "token_drift": [{"turn": 1, "predicted": 1,
                                  "actual": 1, "ratio": 0.0}]}

    monkeypatch.setattr(cli.graph, "run", fake_run)

    def marker(messages, specs=None):
        return 1

    trajectory = cli.run_task(
        TASK, tmp_path, ctx=CTX, count=marker,
        client=scripted({**call("answer", {"value": "10.37 pp"}),
                         "usage": {"total_tokens": 5,
                                   "prompt_tokens_details": {
                                       "cached_tokens": 2}}}))

    assert seen["reply"]["usage"]["total_tokens"] == 5
    assert seen["count"] is marker
    assert trajectory["usage"] == {"total_tokens": 5, "cached_tokens": 2}
    assert trajectory["token_drift"] == [{"turn": 1, "predicted": 1,
                                         "actual": 1, "ratio": 0.0}]
    assert all("usage" not in message for message in trajectory["messages"])


def test_tokenize_url_strips_the_v1_prefix():
    assert cli._tokenize_url("http://127.0.0.1:8000/v1") == (
        "http://127.0.0.1:8000/tokenize")
    assert cli._tokenize_url("http://127.0.0.1:8000/v1/") == (
        "http://127.0.0.1:8000/tokenize")
    assert cli._tokenize_url("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000/tokenize")


def test_token_counter_posts_the_chat_prompt(monkeypatch):
    seen = {}

    def fake_post(url, body):
        seen.update(url=url, body=body)
        return {"count": 321}

    monkeypatch.setattr(cli, "_post_json", fake_post)
    messages = [{"role": "user", "content": "q"}]
    specs = [{"type": "function", "function": {"name": "grep"}}]

    count = cli._token_counter("http://box:8000/v1", "inkling")

    assert count(messages, specs) == 321
    assert seen == {
        "url": "http://box:8000/tokenize",
        "body": {"model": "inkling", "messages": messages,
                 "add_generation_prompt": True, "tools": specs}}


def test_token_counter_normalizes_none_assistant_content(monkeypatch):
    seen = {}

    def fake_post(url, body):
        seen["body"] = body
        return {"count": 12}

    monkeypatch.setattr(cli, "_post_json", fake_post)
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "grep", "arguments": "{}"}}]}]

    assert cli._token_counter("http://box:8000/v1", "inkling")(
        messages) == 12
    # vLLM's /tokenize drops a None-content tool call; "" matches the chat route.
    assert seen["body"]["messages"][0]["content"] == ""
    assert messages[0]["content"] is None

    # A plain empty assistant message is not a tool call: leave it as None.
    cli._token_counter("http://box:8000/v1", "inkling")(
        [{"role": "assistant", "content": None}])
    assert seen["body"]["messages"][0]["content"] is None


def test_token_counter_disables_after_one_failure(monkeypatch, capsys):
    calls = []

    def fake_post(url, body):
        calls.append(url)
        raise OSError("no /tokenize")

    monkeypatch.setattr(cli, "_post_json", fake_post)
    count = cli._token_counter("http://box:8000/v1", "inkling")

    assert count([{"role": "user", "content": "q"}]) is None
    assert count([{"role": "user", "content": "q"}]) is None
    assert len(calls) == 1
    assert "chars fallback" in capsys.readouterr().err


def test_existing_answer_is_skipped(tmp_path):
    (tmp_path / "trajectory.json").write_text(json.dumps(
        {"status": "ok", "answered": True}))
    (tmp_path / "answer.txt").write_text("10.37 pp\n")

    def never(messages, specs):
        raise AssertionError("the model must not be called on a finished task")

    trajectory = cli.run_task(TASK, tmp_path, client=never, ctx=CTX)
    assert trajectory["status"] == "skipped"


def test_legacy_real_answer_is_skipped(tmp_path):
    (tmp_path / "trajectory.json").write_text(json.dumps({"status": "ok"}))
    (tmp_path / "answer.txt").write_text("10.37 pp\n")

    def never(messages, specs):
        raise AssertionError("the model must not be called on a finished task")

    trajectory = cli.run_task(TASK, tmp_path, client=never, ctx=CTX)
    assert trajectory["status"] == "skipped"


def test_legacy_unknown_answer_is_retried(tmp_path):
    (tmp_path / "trajectory.json").write_text(json.dumps({"status": "ok"}))
    (tmp_path / "answer.txt").write_text("UNKNOWN\n")
    client = scripted(call("answer", {"value": "10.37 pp"}))

    trajectory = cli.run_task(TASK, tmp_path, client=client, ctx=CTX)

    assert trajectory["answered"] is True
    assert trajectory["stop_reason"] == "answer"
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
    assert trajectory["answered"] is False
    assert trajectory["infra_error"] == "ConnectionError"
    assert trajectory["stop_reason"] == "error"
    assert trajectory["answer"] == "UNKNOWN"
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
    assert trajectory["answered"] is False
    assert trajectory["infra_error"] is None
    assert trajectory["error"].startswith("KeyboardInterrupt")
    assert not (tmp_path / "answer.txt").exists()


def test_text_reply_is_not_an_answer(tmp_path):
    client = scripted({"role": "assistant", "content": "the answer is 10.37 pp"})

    trajectory = cli.run_task(TASK, tmp_path, client=client, ctx=CTX)

    assert trajectory["status"] == "ok"
    assert trajectory["answered"] is False
    assert trajectory["stop_reason"] == "text_reply"
    assert trajectory["answer"] is None
    assert not (tmp_path / "answer.txt").exists()


def test_empty_final_turn_is_not_an_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.policy, "EMPTY_TURN_RETRIES", 0)
    client = scripted({"role": "assistant", "content": None})

    trajectory = cli.run_task(TASK, tmp_path, client=client, ctx=CTX)

    assert trajectory["answered"] is False
    assert trajectory["stop_reason"] == "empty_turn"
    assert not (tmp_path / "answer.txt").exists()


def test_unanswered_episode_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.policy, "MAX_TURNS", 1)
    monkeypatch.setattr(cli.policy, "FINAL_TURNS", 0)
    first = scripted(call("teleport", {}))

    trajectory = cli.run_task(TASK, tmp_path, client=first, ctx=CTX)

    assert trajectory["status"] == "ok"
    assert trajectory["answered"] is False
    assert trajectory["stop_reason"] == "max_turns"
    assert trajectory["answer"] is None
    assert not (tmp_path / "answer.txt").exists()

    second = scripted(call("answer", {"value": "10.37 pp"}))
    again = cli.run_task(TASK, tmp_path, client=second, ctx=CTX)

    assert again["answered"] is True
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"


def test_trajectory_records_the_retry_count(tmp_path):
    client = scripted(call("answer", {"value": "10.37 pp"}))
    client.stats = {"attempts": 2, "retries": 1, "errors": ["ServerError"]}

    trajectory = cli.run_task(TASK, tmp_path, client=client, ctx=CTX)

    assert trajectory["answered"] is True
    assert trajectory["retries"] == 1


def test_episode_completes_after_a_transient_500(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class ServerError(Exception):
        status_code = 500

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ServerError("upstream failed")
            tool_call = SimpleNamespace(model_dump=lambda: {
                "id": "c1", "type": "function",
                "function": {"name": "answer",
                             "arguments": json.dumps({"value": "10.37 pp"})}})
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=None,
                                            tool_calls=[tool_call]),
                    finish_reason="tool_calls")],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 1}))

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(cli, "_SLEEP", lambda _: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    trajectory = cli.run_task(TASK, tmp_path, client=cli._client(), ctx=CTX)

    assert trajectory["answered"] is True
    assert trajectory["retries"] == 1
    assert trajectory["infra_error"] is None
    assert (tmp_path / "answer.txt").read_text() == "10.37 pp\n"


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


def test_client_retries_a_transient_500_then_succeeds(monkeypatch):
    class ServerError(Exception):
        status_code = 500

    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise ServerError("upstream failed")
            return _reply()

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(cli, "_SLEEP", lambda _: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    client = cli._client()
    reply = client([{"role": "user", "content": "q"}], cli.tools.SPECS)

    assert reply["content"] == "10.37 pp"
    assert len(requests) == 2
    assert requests[0]["timeout"] == cli.REQUEST_TIMEOUT_S
    assert client.stats["retries"] == 1


def test_client_retries_a_timeout_to_the_bound(monkeypatch):
    import openai

    class Timeout(Exception):
        pass

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise Timeout("no reply")

    _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(openai, "APITimeoutError", Timeout)
    monkeypatch.setattr(cli, "_SLEEP", lambda _: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    client = cli._client()
    with pytest.raises(Timeout):
        client([{"role": "user", "content": "q"}], cli.tools.SPECS)

    assert len(calls) == cli.REQUEST_RETRIES + 1
    assert client.stats["retries"] == cli.REQUEST_RETRIES


def test_client_does_not_retry_a_400(monkeypatch):
    class BadRequest(Exception):
        status_code = 400

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise BadRequest("bad request")

    openai = _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(openai, "BadRequestError", BadRequest)
    monkeypatch.setattr(openai, "UnprocessableEntityError", BadRequest)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    client = cli._client()
    with pytest.raises(BadRequest):
        client([{"role": "user", "content": "q"}], cli.tools.SPECS)

    assert len(calls) == 1
    assert client.stats["retries"] == 0


def test_client_drops_a_rejected_seed(monkeypatch):
    class Rejected(Exception):
        pass

    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(dict(kwargs))
            if "seed" in kwargs:
                raise Rejected("seed is not supported")
            return _reply()

    openai = _fake_endpoint(monkeypatch, Completions())
    monkeypatch.setattr(openai, "BadRequestError", Rejected)
    monkeypatch.setattr(openai, "UnprocessableEntityError", Rejected)
    monkeypatch.setattr(cli, "SEED", 7)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    client = cli._client()
    client([{"role": "user", "content": "q"}], cli.tools.SPECS)
    client([{"role": "user", "content": "q"}], cli.tools.SPECS)

    assert ["seed" in request for request in requests] == [True, False, False]


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
    assert "extra_body" not in requests[0]


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

    assert requests[0]["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "none"}}
