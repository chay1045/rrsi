"""One task, one episode: write ``trajectory.json`` + ``answer.txt``.

Not evolvable. The adapter calls this from a worktree, so this file is the
stable I/O contract; RRSI may edit everything under ``harness/`` and nothing
here.

    python3 domains/agora/cli.py --task "Finance & Economics/q_0001" --out runs/t

Resume-safe: an existing ``answer.txt`` means the episode finished, so it is
skipped and the exit code is 0. A failed episode writes a partial
``trajectory.json``, leaves ``answer.txt`` absent (so the adapter retries it)
and exits non-zero.

Env: ``AGORA_DATA``, ``AGORA_INDEX``, ``AGORA_EMBEDDER``, ``AGORA_DOMAIN``,
``AGORA_MODEL``, ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import tasks
from harness import graph, policy, prompts, tools

MODEL = os.environ.get("AGORA_MODEL", "Inkling-2-Mini")
LIVE = os.environ.get("AGORA_LIVE") == "1"


def _clip(value, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + f" ...[+{len(text) - limit} chars]"


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
            return (f"[live] FINAL status={f['status']} turns={f['turns']} "
                    f"tokens={(f.get('usage') or {}).get('total_tokens')} "
                    f"answer={_clip(f.get('answer'), 1200)} error={f.get('error')}")
        if kind == "skipped":
            return f"[live] SKIP task={f['task']} trial={f['trial']}"
        return f"[live] {kind} {json.dumps(f, ensure_ascii=False)[:500]}"


def _client():
    """OpenAI-compatible chat client for the policy endpoint (vLLM or Ollama)."""
    from openai import BadRequestError, OpenAI, UnprocessableEntityError

    key = os.environ.get("OPENAI_API_KEY") or "dummy"
    endpoint = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=key)
    # Keep the policy key out of the environment inherited by the python tool.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("RRSI_API_KEY", None)

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
        try:
            reply = endpoint.chat.completions.create(**request)
        except (BadRequestError, UnprocessableEntityError) as exc:
            if "tool_choice" not in request:
                raise
            print(f"warning: endpoint rejected tool_choice: {exc}; "
                  "retrying without it", file=sys.stderr)
            del request["tool_choice"]
            reply = endpoint.chat.completions.create(**request)
        choice = reply.choices[0].message
        # Only the OpenAI message fields: the reply is sent back on every turn.
        message = {"role": "assistant", "content": choice.content}
        if choice.tool_calls:
            message["tool_calls"] = [c.model_dump() for c in choice.tool_calls]
        if reply.usage:
            message["usage"] = reply.usage.model_dump()
        return message

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
        api_key=os.environ.get("OPENAI_API_KEY") or "dummy")
    effort = os.environ.get("AGORA_SUMMARY_REASONING_EFFORT")

    def summarize(request: str) -> dict:
        call = {"model": MODEL,
                "messages": [
                    {"role": "system", "content": prompts.SUMMARY_SYSTEM},
                    {"role": "user", "content": request}],
                "temperature": 0.0,
                "max_tokens": policy.SUMMARY_MAX_TOKENS}
        if effort:
            # vLLM extension; the OpenAI SDK sends it under extra_body.
            call["extra_body"] = {
                "chat_template_kwargs": {"reasoning_effort": effort}}
        try:
            reply = endpoint.chat.completions.create(**call)
        except (BadRequestError, UnprocessableEntityError) as exc:
            if "extra_body" not in call:
                raise
            print(f"warning: endpoint rejected chat_template_kwargs: {exc}; "
                  "retrying without it", file=sys.stderr)
            del call["extra_body"]
            reply = endpoint.chat.completions.create(**call)
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


def run_task(task: tasks.Task, out_dir, trial: int = 0, client=None,
             ctx: tools.Context | None = None, summarize=None,
             notes=None) -> dict:
    """One episode; always leaves ``trajectory.json`` (unless skipped)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = _Live(out) if LIVE else None

    def log(kind: str, **fields) -> None:
        if live is not None:
            live.event(kind, **fields)

    answer_path = out / "answer.txt"
    if answer_path.exists():
        log("skipped", task=task.id, trial=trial)
        return {"task": task.id, "trial": trial, "status": "skipped",
                "turns": 0, "usage": {"total_tokens": 0}}
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    previous_scratch = os.environ.get("AGORA_SCRATCH")

    usage: dict = {}
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
        log("model_reply", turn=turn,
            elapsed_s=round(time.monotonic() - request_started, 3),
            content=reply.get("content"), tool_calls=reply.get("tool_calls"),
            usage=reply.get("usage"))
        for key, value in (reply.pop("usage", None) or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            elif key == "prompt_tokens_details" and isinstance(value, dict):
                cached = value.get("cached_tokens")
                if isinstance(cached, (int, float)):
                    usage["cached_tokens"] = usage.get("cached_tokens", 0) + cached
        history.append(reply)
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
    try:
        state = graph.run(task.query, model, ctx or _context(task),
                          summarize=summarize, notes=notes)
        status, error = "ok", None
        turns, answer = state["turns"], state["answer"]
        compactions = state.get("compactions") or []
        # The summary call is part of the episode cost, but not a model turn.
        for key, value in (state.get("summary_usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        for event in compactions:
            log("compaction", **event)
        # The final act's result never reaches another model call.
        _add_tool_results(history, seen, state["messages"])
    except BaseException as exc:  # Ctrl-C / SIGTERM must still leave a trace
        status, error = "error", f"{type(exc).__name__}: {exc}"
        answer = "UNKNOWN"
        turns = sum(1 for m in history if m.get("role") == "assistant")
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

    usage.setdefault("total_tokens", 0)
    log("final", status=status, turns=turns, answer=answer, usage=usage,
        error=error)
    trajectory = {
        "task": task.id,
        "domain": task.domain,
        "trial": trial,
        "model": MODEL,
        "status": status,
        "error": error,
        "turns": turns,
        "wall_time_s": wall_time,
        "usage": usage,
        "compactions": compactions,
        "tool_calls": _tool_calls(history),
        "messages": history,
    }
    _write(out / "trajectory.json", json.dumps(trajectory, ensure_ascii=False))
    if status == "ok":
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
    trajectory = run_task(task, args.out, args.trial, client=_client(),
                          ctx=_context(task), summarize=summarize,
                          notes=_notes)
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
