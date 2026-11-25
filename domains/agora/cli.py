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
from harness import graph, policy, tools

MODEL = os.environ.get("AGORA_MODEL", "Inkling-2-Mini")


def _client():
    """OpenAI-compatible chat client for the policy endpoint (vLLM or Ollama)."""
    from openai import OpenAI

    endpoint = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY") or "dummy")

    def call(messages: list[dict], specs: list[dict]) -> dict:
        reply = endpoint.chat.completions.create(
            model=MODEL, messages=messages, tools=specs,
            temperature=policy.TEMPERATURE, max_tokens=policy.MAX_TOKENS)
        choice = reply.choices[0].message
        # Only the OpenAI message fields: the reply is sent back on every turn.
        message = {"role": "assistant", "content": choice.content}
        if choice.tool_calls:
            message["tool_calls"] = [c.model_dump() for c in choice.tool_calls]
        if reply.usage:
            message["usage"] = reply.usage.model_dump()
        return message

    return call


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
             ctx: tools.Context | None = None) -> dict:
    """One episode; always leaves ``trajectory.json`` (unless skipped)."""
    out = Path(out_dir)
    answer_path = out / "answer.txt"
    if answer_path.exists():
        return {"task": task.id, "trial": trial, "status": "skipped",
                "turns": 0, "usage": {"total_tokens": 0}}

    usage: dict = {}
    history: list[dict] = []  # the whole episode, unlike the trimmed state
    seen: set = set()

    def model(messages: list[dict], specs: list[dict]) -> dict:
        if not history:
            history.extend(messages)  # system + task, once
        # A tool result reaches the model on the next call, trimmed later.
        _add_tool_results(history, seen, messages)
        reply = dict(client(messages, specs))
        for key, value in (reply.pop("usage", None) or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        history.append(reply)
        return reply

    started = time.monotonic()
    interrupted = None
    try:
        state = graph.run(task.query, model, ctx or _context(task))
        status, error = "ok", None
        turns, answer = state["turns"], state["answer"]
        # The final act's result never reaches another model call.
        _add_tool_results(history, seen, state["messages"])
    except BaseException as exc:  # Ctrl-C / SIGTERM must still leave a trace
        status, error = "error", f"{type(exc).__name__}: {exc}"
        answer = "UNKNOWN"
        turns = sum(1 for m in history if m.get("role") == "assistant")
        if not isinstance(exc, Exception):
            interrupted = exc
    wall_time = round(time.monotonic() - started, 3)

    usage.setdefault("total_tokens", 0)
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
        "tool_calls": _tool_calls(history),
        "messages": history,
    }
    out.mkdir(parents=True, exist_ok=True)
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

    trajectory = run_task(task, args.out, args.trial, client=_client(),
                          ctx=_context(task))
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
