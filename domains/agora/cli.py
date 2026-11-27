"""One task, one episode: write ``trajectory.json`` + ``answer.txt``.

Not evolvable. The adapter calls this from a worktree, so this file is the
stable I/O contract; RRSI may edit everything under ``harness/`` and nothing
here.

    python3 domains/agora/cli.py --task "Finance & Economics/q_0001" --out runs/t

Resume-safe: a trial whose ``trajectory.json`` says ``answered: true`` is
skipped and the exit code is 0. A finished episode without an answer leaves
``answer.txt`` absent, so a later run retries it. ``answer.txt`` is the
completion marker: it exists only after the ``answer`` tool call. A provider
failure writes ``"UNKNOWN"`` in the trajectory only, records ``infra_error``
and exits non-zero.

Env: ``AGORA_DATA``, ``AGORA_INDEX``, ``AGORA_EMBEDDER``, ``AGORA_DOMAIN``,
``AGORA_MODEL``, ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
``AGORA_REQUEST_RETRIES``, ``AGORA_REQUEST_TIMEOUT_S``, ``AGORA_SEED``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import tasks
from harness import context, graph, policy, prompts, tools

MODEL = os.environ.get("AGORA_MODEL", "Inkling-2-Mini")
LIVE = os.environ.get("AGORA_LIVE") == "1"

# Provider retry policy. Three retries means at most four attempts per turn.
# The timeout is per request, not per episode; the adapter still caps the task.
REQUEST_RETRIES = max(0, int(os.environ.get("AGORA_REQUEST_RETRIES", "3")))
REQUEST_TIMEOUT_S = max(1.0, float(os.environ.get(
    "AGORA_REQUEST_TIMEOUT_S", "600")))
SEED = int(os.environ["AGORA_SEED"]) if os.environ.get("AGORA_SEED") else None
RETRY_BASE_S = 2.0
RETRY_MAX_S = 30.0
_SLEEP = time.sleep


def _retry_delay(attempt: int) -> float:
    """Full-jitter exponential backoff for retry number ``attempt`` (0-based)."""
    ceiling = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    return random.uniform(0.0, ceiling)


def _transient(exc: Exception, openai) -> bool:
    """A provider error that a retry can fix."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError,
                        openai.RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or 500 <= status < 600)


def _clip(value, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + f" ...[+{len(text) - limit} chars]"


def _tokenize_url(base_url: str) -> str:
    """The vLLM ``/tokenize`` URL for an OpenAI-compatible base URL."""
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-len("/v1")]
    return urlunsplit((parts.scheme, parts.netloc, path + "/tokenize", "", ""))


def _post_json(url: str, body: dict) -> dict:
    request = Request(url, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    # A 262k-token prompt can take a few seconds to render and tokenize.
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _countable(messages: list[dict]) -> list[dict]:
    """Messages as the chat route renders them.

    vLLM's ``/tokenize`` drops an assistant tool-call message whose ``content``
    is None. The chat route renders the same message with an empty text block,
    so normalize it here to keep the count exact.
    """
    return [({**message, "content": ""}
             if message.get("role") == "assistant"
             and message.get("content") is None
             and message.get("tool_calls") else message)
            for message in messages]


def _token_counter(base_url: str | None = None, model: str | None = None):
    """Exact prompt count from vLLM ``/tokenize``; None for the chars fallback.

    The endpoint applies the served model's tokenizer and chat template, so the
    count matches the prompt the chat completion sees. A cloud endpoint without
    ``/tokenize`` disables the counter after the first failure; the graph then
    uses the chars estimate for the rest of the episode.
    """
    url = _tokenize_url(base_url or os.environ.get(
        "OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    name = model or MODEL
    available = True

    def count(messages: list[dict],
              specs: list[dict] | None = None) -> int | None:
        nonlocal available
        if not available:
            return None
        body = {"model": name, "messages": _countable(messages),
                "add_generation_prompt": True}
        if specs:
            body["tools"] = specs
        try:
            payload = _post_json(url, body)
        except Exception as exc:  # unreachable or unsupported endpoint
            available = False
            print(f"warning: exact token counter unavailable at {url}: {exc}; "
                  "using the chars fallback", file=sys.stderr)
            return None
        tokens = payload.get("count") if isinstance(payload, dict) else None
        if isinstance(tokens, int) and tokens > 0:
            return tokens
        available = False
        return None

    return count


class _Live:
    """Opt-in trace: full JSONL plus a compact text view in the output dir."""

    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.text_path = out / "live.log"
        self.json_path = out / "live.jsonl"

    def event(self, kind: str, **fields) -> None:
        stamp = time.strftime("%H:%M:%S")
        with open(self.json_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": stamp, "event": kind, **fields},
                                    ensure_ascii=False) + "\n")
        line = self._line(kind, fields)
        with open(self.text_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    @staticmethod
    def _line(kind: str, f: dict) -> str:
        if kind == "run_start":
            return (f"[live] RUN task={f['task']} model={f['model']} "
                    f"trial={f['trial']} max_turns={f['max_turns']} "
                    f"max_tokens={f['max_tokens']}\n"
                    f"[live] QUESTION: {f['query']}\n"
                    f"[live] SYSTEM: {f['system']}")
        if kind == "model_request":
            tail = f["messages"][-1] if f["messages"] else {}
            return (f"[live] MODEL turn={f['turn']} messages={len(f['messages'])} "
                    f"last={tail.get('role')}:{_clip(tail.get('content'), 300)} "
                    f"tools={f['tools']}")
        if kind == "model_reply":
            calls = "; ".join(
                f"{c.get('function', {}).get('name')}"
                f"({_clip(c.get('function', {}).get('arguments'), 240)})"
                for c in (f.get("tool_calls") or []))
            return (f"[live] REPLY turn={f['turn']} elapsed={f['elapsed_s']}s "
                    f"content={_clip(f.get('content'), 400)} calls=[{calls}] "
                    f"usage={f.get('usage')}")
        if kind == "model_error":
            return f"[live] MODEL ERROR turn={f['turn']} {f['error']}"
        if kind == "compaction":
            return (f"[live] COMPACT turn={f['turn']} "
                    f"before={f['before_tokens']} after={f['after_tokens']} "
                    f"dropped={f['dropped_messages']} "
                    f"fallback={f['fallback']} error={f.get('error')}")
        if kind == "token_drift":
            return (f"[live] DRIFT turn={f['turn']} "
                    f"predicted={f['predicted']} actual={f['actual']} "
                    f"ratio={f['ratio']}")
        if kind == "tool_call":
            return (f"[live] TOOL {f['name']} "
                    f"args={_clip(json.dumps(f.get('args'), ensure_ascii=False), 900)}")
        if kind == "tool_result":
            return (f"[live] RESULT {f['name']} chars={len(str(f.get('result')))} "
                    f"preview={_clip(f.get('result'), 700)}")
        if kind == "tool_error":
            return f"[live] TOOL ERROR {f['name']} {f['error']}"
        if kind == "run_error":
            return f"[live] RUN ERROR {f['error']}"
        if kind == "final":
            return (f"[live] FINAL status={f['status']} "
                    f"answered={f.get('answered')} turns={f['turns']} "
                    f"tokens={(f.get('usage') or {}).get('total_tokens')} "
                    f"retries={f.get('retries')} "
                    f"answer={_clip(f.get('answer'), 1200)} error={f.get('error')}")
        if kind == "skipped":
            return f"[live] SKIP task={f['task']} trial={f['trial']}"
        return f"[live] {kind} {json.dumps(f, ensure_ascii=False)[:500]}"


def _client():
    """OpenAI-compatible chat client for the policy endpoint (vLLM or Ollama).

    A transient provider error (429, 5xx, timeout, dropped connection) is
    retried with full-jitter exponential backoff. The SDK's own retries are
    off, so ``stats["retries"]`` is the exact count for the trajectory.
    """
    import openai
    from openai import (BadRequestError, OpenAI, UnprocessableEntityError)

    key = os.environ.get("OPENAI_API_KEY") or "dummy"
    endpoint = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=key, max_retries=0)
    # Keep the policy key out of the environment inherited by the python tool.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("RRSI_API_KEY", None)

    stats = {"attempts": 0, "retries": 0, "errors": []}
    seed = SEED

    def create(request: dict):
        """One request; drop an optional key the endpoint rejects."""
        nonlocal seed
        while True:
            try:
                return endpoint.chat.completions.create(
                    **request, timeout=REQUEST_TIMEOUT_S)
            except (BadRequestError, UnprocessableEntityError) as exc:
                dropped = "seed" if "seed" in request else (
                    "tool_choice" if "tool_choice" in request else None)
                if dropped is None:
                    raise
                del request[dropped]
                if dropped == "seed":
                    seed = None  # the endpoint does not support it
                print(f"warning: endpoint rejected {dropped}: {exc}; "
                      "retrying without it", file=sys.stderr)

    def call(messages: list[dict], specs: list[dict],
             force: str | None = None) -> dict:
        request = {"model": MODEL, "messages": messages, "tools": specs,
                   "temperature": policy.TEMPERATURE,
                   "max_tokens": policy.MAX_TOKENS}
        if force == "answer":
            # The forced answer: the provider must not let the model continue.
            request["tool_choice"] = {"type": "function",
                                      "function": {"name": "answer"}}
        elif force == "required":
            # Final-phase calls before the last: python or answer, not text.
            request["tool_choice"] = "required"
        if seed is not None:
            request["seed"] = seed
        for attempt in range(REQUEST_RETRIES + 1):
            stats["attempts"] += 1
            try:
                reply = create(request)
                break
            except Exception as exc:
                stats["errors"].append(type(exc).__name__)
                if attempt >= REQUEST_RETRIES or not _transient(exc, openai):
                    raise
                stats["retries"] += 1
                _SLEEP(_retry_delay(attempt))
        choice = reply.choices[0].message
        # Only the OpenAI message fields: the reply is sent back on every turn.
        message = {"role": "assistant", "content": choice.content}
        if choice.tool_calls:
            message["tool_calls"] = [c.model_dump() for c in choice.tool_calls]
        if reply.usage:
            message["usage"] = reply.usage.model_dump()
        return message

    call.stats = stats
    return call


def _summarizer():
    """The compaction client: one summary call, no tools, temperature 0.

    Thinking stays on by default, at the model's own setting (0.9 for Inkling);
    pi forwards its session thinking level to the summarization call for the
    same reason. ``AGORA_SUMMARY_REASONING_EFFORT`` can lower or disable it for
    speed. That flag is a vLLM template kwarg, so a rejected request retries
    without it, like tool_choice does.
    """
    from openai import BadRequestError, OpenAI, UnprocessableEntityError

    endpoint = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY") or "dummy", max_retries=0)
    effort = os.environ.get("AGORA_SUMMARY_REASONING_EFFORT")

    def summarize(request: str) -> dict:
        call = {"model": MODEL,
                "messages": context.summary_messages(request),
                "temperature": 0.0,
                "max_tokens": policy.SUMMARY_MAX_TOKENS}
        if effort:
            # vLLM extension; the OpenAI SDK sends it under extra_body.
            call["extra_body"] = {
                "chat_template_kwargs": {"reasoning_effort": effort}}
        try:
            reply = endpoint.chat.completions.create(
                **call, timeout=REQUEST_TIMEOUT_S)
        except (BadRequestError, UnprocessableEntityError) as exc:
            if "extra_body" not in call:
                raise
            print(f"warning: endpoint rejected chat_template_kwargs: {exc}; "
                  "retrying without it", file=sys.stderr)
            del call["extra_body"]
            reply = endpoint.chat.completions.create(
                **call, timeout=REQUEST_TIMEOUT_S)
        choice = reply.choices[0]
        # A checkpoint cut at the output cap omits its newest findings, so the
        # harness treats it as a failed summary and falls back.
        return {"content": choice.message.content or "",
                "usage": reply.usage.model_dump() if reply.usage else {},
                "truncated": choice.finish_reason == "length"}

    return summarize


def _notes() -> str | None:
    """The episode's notes.md, read at compaction time and bounded."""
    root = os.environ.get("AGORA_SCRATCH")
    if not root:
        return None
    try:
        text = (Path(root) / "notes.md").read_text(encoding="utf-8",
                                                   errors="replace")
    except OSError:
        return None
    return text[:policy.NOTES_CHARS] or None


def _context(task: tasks.Task) -> tools.Context:
    # AGORA_INDEX pins a snapshot without editing the evolvable harness, whose
    # Context defaults are fixed at import time.
    return tools.Context(
        domain=task.domain,
        data_dir=Path(os.environ.get("AGORA_DATA") or tasks.DATA_DIR),
        index_dir=Path(os.environ.get("AGORA_INDEX") or tools.INDEX_DIR))


def _tool_calls(messages: list[dict]) -> list[dict]:
    """Flat call list. Outputs pair in order: call ids repeat across turns."""
    calls, waiting = [], []
    for message in messages:
        if message.get("role") == "tool":
            if waiting:
                waiting.pop(0)["output"] = message.get("content")
            continue
        waiting = []  # results always follow their reply
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            record = {"name": function.get("name"),
                      "arguments": function.get("arguments"), "output": None}
            calls.append(record)
            waiting.append(record)
    return calls


def _add_tool_results(history: list[dict], seen: set, messages: list[dict]) -> None:
    """Append results the history lacks; compared by object, not by id."""
    for message in messages:
        if message.get("role") == "tool" and id(message) not in seen:
            seen.add(id(message))
            history.append(message)


def _write(path: Path, text: str) -> None:
    # Atomic: the adapter reads trajectory.json while a trial is still running.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _previous_trajectory(out: Path) -> dict | None:
    try:
        return json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _already_answered(out: Path) -> bool:
    """Whether a previous run of this trial submitted an answer.

    The explicit ``answered`` field wins. A trial from before the field exists
    only when its old completion marker is a non-UNKNOWN answer; ``UNKNOWN``
    was the old provider-failure placeholder and must be retried.
    """
    trajectory = _previous_trajectory(out) or {}
    if "answered" in trajectory:
        return trajectory.get("answered") is True
    try:
        answer = (out / "answer.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return (trajectory.get("status") == "ok" and bool(answer)
            and answer.upper() != "UNKNOWN")


def run_task(task: tasks.Task, out_dir, trial: int = 0, client=None,
             ctx: tools.Context | None = None, summarize=None,
             notes=None, count=None) -> dict:
    """One episode; always leaves ``trajectory.json`` (unless skipped)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = _Live(out) if LIVE else None

    def log(kind: str, **fields) -> None:
        if live is not None:
            live.event(kind, **fields)

    answer_path = out / "answer.txt"
    if _already_answered(out):
        log("skipped", task=task.id, trial=trial)
        return {"task": task.id, "trial": trial, "status": "skipped",
                "answered": True, "stop_reason": "answer", "infra_error": None,
                "retries": 0, "turns": 0, "usage": {"total_tokens": 0}}
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    previous_scratch = os.environ.get("AGORA_SCRATCH")

    usage: dict = {}
    usage_reports: list[dict] = []  # the error path has no graph state
    history: list[dict] = []  # the whole episode, unlike the compacted state
    seen: set = set()
    log("run_start", task=task.id, query=task.query, system=prompts.SYSTEM,
        model=MODEL, trial=trial, max_turns=policy.MAX_TURNS,
        max_tokens=policy.MAX_TOKENS, context_window=policy.CONTEXT_WINDOW,
        context_tokens=policy.CONTEXT_TOKENS,
        keep_recent_tokens=policy.KEEP_RECENT_TOKENS)

    turn = 0

    def model(messages: list[dict], specs: list[dict],
              force: str | None = None) -> dict:
        nonlocal turn
        turn += 1
        log("model_request", turn=turn, messages=messages,
            tools=[(spec.get("function") or {}).get("name") for spec in specs],
            force=force)
        if not history:
            history.extend(messages)  # system + task, once
        # A tool result reaches the model on the next call, compacted later.
        _add_tool_results(history, seen, messages)
        request_started = time.monotonic()
        try:
            reply = dict(client(messages, specs, force=force))
        except BaseException as exc:
            log("model_error", turn=turn,
                error=f"{type(exc).__name__}: {exc}")
            raise
        # Keep usage on the reply: graph._add_usage() reads it for the anchor
        # and the trajectory total, then removes it from the replayed message.
        report = reply.get("usage")
        if isinstance(report, dict) and report:
            usage_reports.append(dict(report))
        log("model_reply", turn=turn,
            elapsed_s=round(time.monotonic() - request_started, 3),
            content=reply.get("content"), tool_calls=reply.get("tool_calls"),
            usage=report)
        history.append({key: value for key, value in reply.items()
                        if key != "usage"})
        return reply

    original_dispatch = tools.dispatch

    def dispatch(name, args, ctx):
        log("tool_call", name=name, args=args)
        try:
            result = original_dispatch(name, args, ctx)
        except BaseException as exc:
            log("tool_error", name=name, error=f"{type(exc).__name__}: {exc}")
            raise
        log("tool_result", name=name, result=str(result))
        return result

    tools.dispatch = dispatch
    # One scratch directory per episode. The python tool reads it on every call,
    # so notes.md survives; restoring the old value keeps a reused process from
    # leaking one episode's scratch into the next.
    os.environ["AGORA_SCRATCH"] = str(scratch.resolve())
    started = time.monotonic()
    interrupted = None
    compactions: list[dict] = []
    drift: list[dict] = []
    try:
        state = graph.run(task.query, model, ctx or _context(task),
                          summarize=summarize, notes=notes, count=count)
        status, error, infra_error = "ok", None, None
        turns, answer = state["turns"], state.get("answer")
        compactions = state.get("compactions") or []
        drift = state.get("token_drift") or []
        # The summary call is part of the episode cost, but not a model turn.
        usage = dict(state.get("usage") or {})
        graph.sum_usage(usage, state.get("summary_usage") or {})
        for event in compactions:
            log("compaction", **event)
        for record in drift:
            log("token_drift", **record)
        # The final act's result never reaches another model call.
        _add_tool_results(history, seen, state["messages"])
    except BaseException as exc:  # Ctrl-C / SIGTERM must still leave a trace
        status, error = "error", f"{type(exc).__name__}: {exc}"
        # UNKNOWN is the provider-failure marker and never a completion.
        answer = "UNKNOWN" if isinstance(exc, Exception) else None
        infra_error = type(exc).__name__ if isinstance(exc, Exception) else None
        turns = sum(1 for m in history if m.get("role") == "assistant")
        # The graph state is not available, so sum the replies already seen.
        usage = {}
        for report in usage_reports:
            graph.sum_usage(usage, report)
        log("run_error", error=error)
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        tools.dispatch = original_dispatch
        if previous_scratch is None:
            os.environ.pop("AGORA_SCRATCH", None)
        else:
            os.environ["AGORA_SCRATCH"] = previous_scratch
    wall_time = round(time.monotonic() - started, 3)

    tool_calls = _tool_calls(history)
    answered = (status == "ok" and answer is not None
                and any(call["name"] == "answer" for call in tool_calls))
    if not answered:
        # No answer tool call means no submission, even if a candidate harness
        # left assistant text in the state.
        answer = None if status == "ok" else answer
    last = history[-1] if history else {}
    if answered:
        stop_reason = "answer"
    elif error is not None:
        stop_reason = "error"
    elif turns >= policy.MAX_TURNS:
        stop_reason = "max_turns"
    elif (last.get("role") == "assistant" and not last.get("tool_calls")
          and not str(last.get("content") or "").strip()):
        stop_reason = "empty_turn"
    elif last.get("role") == "assistant" and last.get("content"):
        stop_reason = "text_reply"
    else:
        stop_reason = "no_answer"
    stats = getattr(client, "stats", None)
    retries = int(stats.get("retries", 0)) if isinstance(stats, dict) else 0

    usage.setdefault("total_tokens", 0)
    log("final", status=status, answered=answered, turns=turns, answer=answer,
        usage=usage, retries=retries, error=error)
    trajectory = {
        "task": task.id,
        "domain": task.domain,
        "trial": trial,
        "model": MODEL,
        "status": status,
        "answered": answered,
        "answer": answer,
        "stop_reason": stop_reason,
        "infra_error": infra_error,
        "retries": retries,
        "error": error,
        "turns": turns,
        "wall_time_s": wall_time,
        "usage": usage,
        "compactions": compactions,
        "token_drift": drift,
        "tool_calls": tool_calls,
        "messages": history,
    }
    _write(out / "trajectory.json", json.dumps(trajectory, ensure_ascii=False))
    if answered and answer is not None:
        _write(answer_path, answer + "\n")
    if interrupted is not None:
        raise interrupted
    return trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="task id, <domain>/<qid>")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args(argv)

    task_id = args.task
    if "/" not in task_id:
        domain = os.environ.get("AGORA_DOMAIN")
        if not domain:
            parser.error("--task needs <domain>/<qid>, or set AGORA_DOMAIN")
        task_id = f"{domain}/{task_id}"
    domain, _, qid = task_id.rpartition("/")
    task = tasks.load_tasks().get((domain, qid))
    if task is None:
        parser.error(f"unknown task: {task_id}")

    # Build the summarizer first: _client() removes OPENAI_API_KEY from the
    # process environment so the python tool cannot read it.
    summarize = _summarizer()
    count = _token_counter()
    trajectory = run_task(task, args.out, args.trial, client=_client(),
                          ctx=_context(task), summarize=summarize,
                          notes=_notes, count=count)
    print(f"{trajectory['status']}: {task.id} -> {args.out} "
          f"({trajectory['turns']} turns, "
          f"{trajectory['usage']['total_tokens']} tokens)")
    return 0 if trajectory["status"] in {"ok", "skipped"} else 1


def _terminate(signum, frame):
    # Turn a SIGTERM into an exception so the partial trace gets written.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
