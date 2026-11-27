"""Agora domain adapter: run, score and explain one document-reasoning agent.

The harness under ``harness/`` is the only evolvable code. This adapter is the
frozen bridge between RRSI and that harness:

    run / score        one episode per (task, trial), k trials per task
    load_trial         trajectory + answer + gold for the analyst
    render_trace       what the analyst / digester / proposer read
    task_row           one line per trace in the task tables
    smoke              compile, import, then one real episode
    guards             non-compensatory checks after evaluation
    critic_patterns    leakage denylist before evaluation
    component_signals  diff regexes -> RRSI component tag
    briefs             domain paragraphs for the search roles

Layout of one evaluation::

    runs/agora/jobs/<job>/s<trial>/<domain>__<qid>/
        trajectory.json   written by cli.py
        answer.txt        written only on a clean finish

A missing trial is reward 0 with the full denominator, never a skipped slot.

Isolation
---------
``run()`` starts one container per episode from ``rrsi.json["sandbox_image"]``.
Only these things are visible inside it:

* the candidate's ``harness/`` directory, read-only at ``/app/agora/harness``;
* the frozen runtime ``cli.py``, ``tasks.py`` and ``index/``, read-only;
* the task's own domain corpus, read-only at ``/data/documents/<domain>``;
* one redacted task row (gold answer blanked) at ``/data/tasks``;
* the task's own hybrid index file, read-only at ``/index``;
* the EmbeddingGemma 2 text tower, read-only at ``/models/embeddinggemma2``;
* the trial output directory, writable at ``/out``.

The split, the scorer and the adapter are not mounted, so the agent cannot
read the evolve/heldout lists or the normalizer at runtime.

The container is non-root, read-only, without capabilities, with a tmpfs
``/tmp``. ``--network host`` is required because the policy endpoint runs on
the host loopback; the harness has no network tool, and guards reject network
imports and URLs in harness code. This is a filesystem and process sandbox,
not a network sandbox; see ``docs/sandbox.md`` in the umbrella repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
import briefs                            # noqa: E402
import score as agora_score              # noqa: E402
import split as agora_split              # noqa: E402
import tasks as agora_tasks              # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text())
DATA_DIR = Path(os.environ.get("AGORA_DATA", Path.home() / "work" / "agora-data"))
INDEX_DIR = Path(os.environ.get("AGORA_INDEX") or (
    Path(os.environ.get("RRSI_CACHE_DIR", Path.home() / ".cache" / "rrsi"))
    / "agora" / "index"))
EMBEDDER_DIR = Path(os.environ.get("AGORA_EMBEDDER") or (
    Path.home() / "Models" / "embeddinggemma2"))
AGENT_PY = os.environ.get("RRSI_AGENT_PYTHON", sys.executable)
SOURCE_EXTS = {".py", ".txt", ".md", ".json"}
_TRACE_CAP = 300_000
_LOG_LOCK = threading.Lock()

# A candidate may not add a network client, an endpoint or a credential to the
# harness. The task's corpus and the index are mounted read-only; a new client
# would escape the sandbox contract even when it runs.
_GUARD_PATTERNS = [
    (re.compile(r"https?://", re.IGNORECASE), "network URL"),
    (re.compile(r"\blocalhost\b", re.IGNORECASE), "localhost endpoint"),
    (re.compile(r"\bbase_url\b", re.IGNORECASE), "base_url endpoint"),
    (re.compile(r"\bOPENAI_[A-Za-z0-9_]*\b"), "OpenAI environment variable"),
    (re.compile(r"\bAPI_BASE\b"), "API_BASE endpoint"),
    (re.compile(r"\bapi[_-]?key\b", re.IGNORECASE), "credential reference"),
    (re.compile(r"\bAGORA_INDEX\b"), "index path override"),
    (re.compile(r"\bAGORA_EMBEDDER\b"), "embedder path override"),
    (re.compile(r"\bAGORA_RERANK[A-Z_]*\b"), "reranker endpoint override"),
    (re.compile(r"(?<![\w])/out(?:/|\b)"), "grader output path"),
    (re.compile(r"\banswer\.txt\b"), "grader answer file"),
    (re.compile(r"\btrajectory\.json\b"), "grader trajectory file"),
    (re.compile(r"\bgold\b", re.IGNORECASE), "gold answer reference"),
    (re.compile(r"^\s*(?:from|import)\s+"
                r"(?:openai|anthropic|requests|httpx|aiohttp|urllib|socket)\b",
                re.MULTILINE), "network client import"),
    (re.compile(r"^\s*(?:from|import)\s+(?:tasks|score|cli)\b",
                re.MULTILINE), "answer parser import"),
]


def _slug(domain: str) -> str:
    # Same formula as harness/tools.py and index/build.py: the index file name.
    return re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")


def _max_concurrent() -> int:
    """Episode concurrency: a cloud policy can exceed the local 4."""
    raw = os.environ.get("AGORA_MAX_CONCURRENT")
    if raw is None:
        return int(CFG["max_concurrent"])
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"AGORA_MAX_CONCURRENT must be an integer, got {raw!r}") from exc


MAX_CONCURRENT = _max_concurrent()


def _task_timeout() -> int:
    """Per-episode cap: the 96k-char budget needs more than 3600 s locally."""
    raw = os.environ.get("AGORA_TASK_TIMEOUT_S")
    if raw is None:
        return int(CFG["task_timeout_s"])
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"AGORA_TASK_TIMEOUT_S must be an integer, got {raw!r}") from exc


TASK_TIMEOUT_S = _task_timeout()


def _name(task_id: str) -> str:
    # The task id is "<domain>/<qid>"; one safe directory per task.
    return task_id.replace("/", "__")


def _clip(value, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + f" ...[+{len(text) - limit} chars]"


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_answer(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    print(message, flush=True)


@lru_cache(maxsize=2)
def _load_tasks(tasks_dir: str) -> dict:
    return agora_tasks.load_tasks(Path(tasks_dir))


def task_map() -> dict:
    """The task corpus, keyed by ``(domain, qid)``. Cached per directory."""
    return _load_tasks(str(agora_tasks.TASKS_DIR))


def _task(task_id: str):
    domain, _, qid = task_id.rpartition("/")
    return task_map().get((domain, qid))


def _gold(task_id: str) -> str:
    task = _task(task_id)
    return task.gold if task is not None else ""


def _split() -> dict:
    # load_split() verifies data/split.sha256 on every process that uses it.
    return agora_split.load_split()


def _scan_harness(root: Path) -> list[str]:
    """Reject a candidate harness that reaches outside its read-only mounts."""
    root = Path(root).resolve()
    harness = (root / "domains" / "agora" / "harness").resolve()
    if not harness.is_dir():
        return [f"harness directory missing: {harness}"]
    hits: list[str] = []
    for path in sorted(harness.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.parts
                or path.suffix not in SOURCE_EXTS):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            hits.append(f"{path.relative_to(root)}: unreadable: {exc}")
            continue
        for number, line in enumerate(lines, 1):
            for pattern, why in _GUARD_PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(root)}:{number}: {why}")
    return sorted(set(hits))


def _stage_task(stage: Path, task) -> Path:
    """Write a one-row task file with the gold answer removed.

    The CLI reads ``AGORA_DATA`` to load the task; it needs the query, not the
    gold answer. The real corpus stays mounted separately, read-only.
    """
    shutil.rmtree(stage, ignore_errors=True)
    target = stage / "tasks" / f"{task.domain}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"id": task.qid, "domain": task.domain, "query": task.query,
           "answer": ""}
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(target)
    return target


def _require_image(image: str) -> str:
    if not shutil.which("docker"):
        raise RuntimeError("docker is not on PATH; the Agora sandbox needs it")
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sandbox image {image!r} is missing; build it with "
            f"`docker build -t {image} {HERE / 'sandbox'}` "
            f"({(result.stderr or '').strip()[-300:]})")
    return result.stdout.strip()


def _env_args() -> list[str]:
    """Only the variables the episode needs. Never forward the host env."""
    names = ("AGORA_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "AGORA_LIVE",
             "AGORA_RERANK_URL", "AGORA_RERANK_MODEL", "AGORA_TEMPERATURE",
             "AGORA_CONTEXT_WINDOW", "AGORA_SUMMARY_MAX_TOKENS",
             "AGORA_SUMMARY_REASONING_EFFORT")
    args: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value:
            args += ["--env", f"{name}={value}"]
    return args


class AgoraDomain(Domain):
    name = "agora"
    harness_path = "harness"
    source_exts = SOURCE_EXTS
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}

    # Retrieval work is tagged client_tool because RRSI's fixed vocabulary has
    # no "retrieval" member; tool-output shaping is output_plumbing. The
    # proposer still sees the real failure family in the hypothesis.
    component_signals = [
        ("context_mgmt",    [r"context\.py", r"\btrim\(", r"CONTEXT_CHARS",
                             r"compact", r"keep_last", r"budget",
                             r"CONTEXT_TOKENS", r"CONTEXT_WINDOW",
                             r"KEEP_RECENT_TOKENS", r"SUMMARY_MAX_TOKENS"]),
        ("output_plumbing", [r"SNIPPET_CHARS", r"READ_CHARS", r"GREP_HITS",
                             r"PY_CHARS", r"truncat", r"snippet", r"_clip"]),
        ("client_tool",     [r"search_docs", r"index_search", r"match_expression",
                             r"\bbm25\b", r"retrieval", r"\bquery\b"]),
        ("control_flow",    [r"graph\.py", r"MAX_TURNS", r"after_tools",
                             r"after_agent", r"termination"]),
        ("config",          [r"policy\.py", r"=\s*\d+\s*#", r"\bthreshold\b",
                             r"\bTEMPERATURE\b", r"\bMAX_TOKENS\b"]),
        ("prompt",          [r"prompts\.py", r"\bSYSTEM\b", r"system prompt"]),
    ]

    def __init__(self):
        self._critic_patterns: list | None = None

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(_split()["evolve"])

    def heldout_ids(self) -> list[str]:
        return list(_split()["heldout"])

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        return list(_split()["smoke"])

    # ---- Evaluate ----------------------------------------------------------
    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        root, runs_dir = Path(root).resolve(), Path(runs_dir).resolve()
        image = CFG["sandbox_image"]
        image_id = _require_image(image)

        job_dir = runs_dir / "jobs" / job
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = runs_dir / "logs" / f"{job.replace('/', '_')}.agora.log"
        violations = _scan_harness(root)
        (job_dir / "guard.json").write_text(json.dumps(
            {"image": image, "image_id": image_id,
             "max_concurrent": MAX_CONCURRENT,
             "task_timeout_s": TASK_TIMEOUT_S,
             "violations": violations}, indent=1), encoding="utf-8")
        if violations:
            _log(log_path, f"{log_prefix}{job}: harness guard violations: "
                           f"{'; '.join(violations)}")

        embedder = EMBEDDER_DIR.resolve()
        if not embedder.is_dir():
            raise FileNotFoundError(
                f"missing embedder for hybrid retrieval: {embedder}; set "
                f"AGORA_EMBEDDER or install the model under ~/Models")

        episodes: list[tuple] = []
        for task_id in ids:
            task = _task(task_id)
            if task is None:
                raise KeyError(f"unknown Agora task: {task_id!r}")
            docs = (DATA_DIR / "documents" / task.domain).resolve()
            if not docs.is_dir():
                raise FileNotFoundError(
                    f"missing corpus for {task.domain!r}: {docs}")
            index = (Path(INDEX_DIR) / f"{_slug(task.domain)}.sqlite").resolve()
            if not index.is_file():
                raise FileNotFoundError(
                    f"missing index for {task.domain!r}: {index}; build it "
                    f"with domains/agora/index/build.py")
            stage = job_dir / "data" / _name(task_id)
            task_file = _stage_task(stage, task)
            for trial in range(k):
                out = job_dir / f"s{trial}" / _name(task_id)
                if (out / "answer.txt").exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                episodes.append((task_id, trial, out, task, docs, index,
                                 task_file))

        if not episodes:
            _log(log_path, f"{log_prefix}{job}: all {len(ids) * k} trials exist")
            return

        workers = max(1, min(MAX_CONCURRENT, len(episodes)))
        _log(log_path, f"{log_prefix}{job}: {len(episodes)} episodes, "
                       f"image {image}, concurrency {workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._episode, root, task_id, trial, out, task,
                            docs, index, task_file): (task_id, trial)
                for task_id, trial, out, task, docs, index, task_file in episodes
            }
            for future in as_completed(futures):
                task_id, trial = futures[future]
                try:
                    status = future.result()
                except Exception as exc:  # noqa: BLE001
                    status = f"exception {type(exc).__name__}: {exc}"
                _log(log_path, f"{log_prefix}{job}: {task_id} s{trial} {status}")

    def _episode(self, root, task_id, trial, out, task, docs, index,
                 task_file) -> str:
        image = CFG["sandbox_image"]
        name = f"rrsi-agora-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run", "--rm", "--init", "--pull", "never",
            "--name", name,
            "--read-only", "--network", "host",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory", str(CFG["container_memory"]),
            "--cpus", str(CFG["container_cpus"]),
            "--pids-limit", "512",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,size=2g,mode=1777",
            "--volume", f"{Path(root) / 'domains' / 'agora' / 'harness'}"
                        f":/app/agora/harness:ro",
            "--volume", f"{Path(root) / 'domains' / 'agora' / 'index'}"
                        f":/app/agora/index:ro",
            "--volume", f"{Path(root) / 'domains' / 'agora' / 'cli.py'}"
                        f":/app/agora/cli.py:ro",
            "--volume", f"{Path(root) / 'domains' / 'agora' / 'tasks.py'}"
                        f":/app/agora/tasks.py:ro",
            "--volume", f"{docs}:/data/documents/{task.domain}:ro",
            "--volume", f"{task_file}:/data/tasks/{task_file.name}:ro",
            "--volume", f"{index}:/index/{index.name}:ro",
            "--volume", f"{EMBEDDER_DIR.resolve()}:/models/embeddinggemma2:ro",
            "--volume", f"{Path(out)}:/out",
            "--env", "AGORA_DATA=/data",
            "--env", "AGORA_INDEX=/index",
            "--env", "AGORA_EMBEDDER=/models/embeddinggemma2",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "PYTHONUNBUFFERED=1",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "HOME=/tmp",
            "--env", "TMPDIR=/tmp",
            *_env_args(),
            "--workdir", "/app/agora",
            image, "python", "/app/agora/cli.py",
            "--task", task_id, "--out", "/out", "--trial", str(trial),
        ]
        log_file = Path(out) / "run.log"
        with open(log_file, "ab") as handle:
            process = subprocess.Popen(command, stdout=handle,
                                       stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=TASK_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60,
                               check=False)
                process.wait(timeout=60)
                code = -9
        if code == 0:
            return "ok"
        return f"exit {code} (see {log_file.name})"

    def score(self, runs_dir, job, ids, k):
        runs_dir = Path(runs_dir).resolve()
        job_dir = runs_dir / "jobs" / job
        guard_file = job_dir / "guard.json"
        guard = _read_json(guard_file) or {}
        violations = guard.get("violations") or []
        per: dict[str, TaskResult] = {}
        exact = parseable = answered = missing_total = trials = 0

        for task_id in ids:
            rewards, tokens = [], []
            missing = 0
            gold = _gold(task_id)
            for trial in range(k):
                out = job_dir / f"s{trial}" / _name(task_id)
                trajectory = _read_json(out / "trajectory.json")
                prediction = _read_answer(out / "answer.txt")
                status = (trajectory or {}).get("status")
                clean = prediction is not None and status == "ok"
                reward = (agora_score.score(gold, prediction) if clean else 0.0)
                is_parseable = bool(clean and prediction
                                    and prediction.strip().upper() != "UNKNOWN")
                usage = (trajectory or {}).get("usage") or {}
                token_count = usage.get("total_tokens")
                tokens.append(token_count if isinstance(token_count, (int, float))
                              and token_count > 0 else None)
                rewards.append(reward)
                trials += 1
                exact += reward
                parseable += int(is_parseable)
                answered += int(prediction is not None)
                if prediction is None:
                    missing += 1
                    missing_total += 1
            per[task_id] = TaskResult(rewards=rewards, tokens=tokens,
                                      missing=missing)

        denominator = max(1, trials)
        return per, {
            "n_trials": trials,
            "exact_matches": int(exact),
            "exact_rate": exact / denominator,
            "parseable": parseable,
            "parseable_rate": parseable / denominator,
            "answered": answered,
            "answered_rate": answered / denominator,
            "missing": missing_total,
            "missing_rate": missing_total / denominator,
            "guard_violations": list(violations),
        }

    def guards(self, incumbent, candidate) -> list[str]:
        out = list((candidate.extra or {}).get("guard_violations") or [])
        drop = ((incumbent.extra or {}).get("parseable_rate") or 0.0) - (
            (candidate.extra or {}).get("parseable_rate") or 0.0)
        if drop > CFG["max_parseable_drop"]:
            out.append(f"parseable-answer rate fell {drop:+.3f} "
                       f"(limit {CFG['max_parseable_drop']})")
        rise = ((candidate.extra or {}).get("missing_rate") or 0.0) - (
            (incumbent.extra or {}).get("missing_rate") or 0.0)
        if rise > CFG["max_missing_rise"]:
            out.append(f"missing-trial rate rose {rise:+.3f} "
                       f"(limit {CFG['max_missing_rise']})")
        return out

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        out = (Path(runs_dir).resolve() / "jobs" / job / f"s{trial}"
               / _name(task_id))
        trajectory = _read_json(out / "trajectory.json")
        if trajectory is None:
            return None
        prediction = _read_answer(out / "answer.txt")
        gold = _gold(task_id)
        return {
            "task_id": task_id,
            "domain": task_id.rpartition("/")[0],
            "trial": trial,
            "trajectory": trajectory,
            "answer": prediction,
            "gold": gold,
            "match": (agora_score.score(gold, prediction)
                      if prediction is not None else 0.0),
            "missing": prediction is None,
        }

    def render_trace(self, rec, detail=False):
        if not rec:
            return "(no trial)"
        trajectory = rec.get("trajectory") or {}
        messages = trajectory.get("messages") or []
        usage = trajectory.get("usage") or {}
        caps = ({"tool": 4000, "args": 1200, "assistant": 4000} if detail
                else {"tool": 700, "args": 400, "assistant": 1200})
        lines = [
            f"TASK: {rec.get('task_id')} | domain: {rec.get('domain')} | "
            f"trial: {rec.get('trial')}",
            f"STATUS: {trajectory.get('status')} | turns: "
            f"{trajectory.get('turns')} | total_tokens: "
            f"{usage.get('total_tokens')} | error: {trajectory.get('error')}",
            "", "=== TRAJECTORY ===",
        ]
        call_names = {}
        for message in messages:
            for call in message.get("tool_calls") or []:
                if call.get("id"):
                    call_names[call["id"]] = (call.get("function") or {}).get("name")
        step = 0
        for message in messages:
            role = message.get("role")
            if role == "system":
                lines.append(f"[SYSTEM PROMPT] {_clip(message.get('content'), 2000)}")
            elif role == "user":
                lines.append(f"[TASK PROMPT] {_clip(message.get('content'), 4000)}")
            elif role == "assistant":
                step += 1
                if message.get("content"):
                    lines.append(f"[step {step}] ASSISTANT: "
                                 f"{_clip(message['content'], caps['assistant'])}")
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    lines.append(f"[step {step}] TOOL_CALL {function.get('name')}"
                                 f"({_clip(function.get('arguments', ''), caps['args'])})")
            elif role == "tool":
                name = message.get("name") or call_names.get(
                    message.get("tool_call_id")) or "unknown_tool"
                lines.append(f"[step {step}] TOOL_RESULT {name}: "
                             f"{_clip(message.get('content'), caps['tool'])}")
        lines += [
            "", "=== RESULT ===",
            f"GOLD: {rec.get('gold')}",
            f"PREDICTION: {_clip(rec.get('answer'), 4000)}",
            f"EXACT MATCH: {float(rec.get('match') or 0):.0f}",
        ]
        text = "\n".join(lines)
        if len(text) > _TRACE_CAP:
            head, tail = int(_TRACE_CAP * 0.6), int(_TRACE_CAP * 0.4)
            text = (text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n"
                    + text[-tail:])
        return text

    def task_row(self, task_id, rec, result) -> str:
        trajectory = (rec or {}).get("trajectory") or {}
        usage = trajectory.get("usage") or {}
        mean = result.mean if result is not None else 0.0
        return (f"{task_id} | {rec.get('domain') if rec else '?'} | "
                f"exact={mean:.2f} | status={trajectory.get('status')} | "
                f"turns={trajectory.get('turns')} | "
                f"tokens={usage.get('total_tokens')} | "
                f"answer={_clip(rec.get('answer') if rec else None, 120)}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root).resolve(), Path(runs_dir).resolve()
        compiled = subprocess.run(
            [AGENT_PY, "-m", "compileall", "-q", str(self.harness_dir(root))],
            capture_output=True, text=True)
        if compiled.returncode != 0:
            return False, {"stage": "compile",
                           "detail": (compiled.stdout + compiled.stderr)[-1500:]}
        imported = subprocess.run(
            [AGENT_PY, "-c", "import harness.graph; harness.graph.build()"],
            cwd=str(root / "domains" / "agora"), capture_output=True, text=True)
        if imported.returncode != 0:
            return False, {"stage": "import",
                           "detail": (imported.stdout + imported.stderr)[-1500:]}
        if not ids:
            return False, {"stage": "smoke_run", "detail": "no smoke task"}

        shutil.rmtree(runs_dir / "jobs" / job, ignore_errors=True)
        try:
            self.run(root, runs_dir, job, ids[:1], 1, log_prefix="smoke ")
        except Exception as exc:  # noqa: BLE001
            return False, {"stage": "smoke_run",
                           "detail": f"{type(exc).__name__}: {exc}"}
        out = runs_dir / "jobs" / job / "s0" / _name(ids[0])
        trajectory = _read_json(out / "trajectory.json")
        if trajectory is None:
            return False, {"stage": "smoke_run", "detail": "no trajectory"}
        if trajectory.get("status") != "ok":
            return False, {"stage": "smoke_run",
                           "detail": f"status {trajectory.get('status')}: "
                                     f"{trajectory.get('error')}"}
        if not (out / "answer.txt").exists():
            return False, {"stage": "smoke_run", "detail": "no answer.txt"}
        return True, {"stage": "smoke_run", "task": ids[0],
                      "turns": trajectory.get("turns"),
                      "tokens": (trajectory.get("usage") or {}).get("total_tokens")}

    # ---- leakage denylist ----------------------------------------------------
    @property
    def critic_patterns(self):
        if self._critic_patterns is None:
            self._critic_patterns = _build_critic_patterns()
        return self._critic_patterns


def _build_critic_patterns() -> list:
    patterns: list[tuple[str, str]] = [
        (r"https?://", "network URL in diff"),
        (r"\blocalhost\b", "localhost endpoint in diff"),
        (r"\bbase_url\b", "base_url endpoint in diff"),
        (r"\bOPENAI_[A-Za-z0-9_]*\b", "OpenAI environment variable in diff"),
        (r"\bAPI_BASE\b", "API_BASE endpoint in diff"),
        (r"(?i)\bAGORA_INDEX\b", "index path override in diff"),
        (r"(?i)\bAGORA_EMBEDDER\b", "embedder path override in diff"),
        (r"(?i)\bAGORA_RERANK[A-Z_]*\b", "reranker endpoint override in diff"),
        (r"(?<![\w])/out(?:/|\b)", "grader output path in diff"),
        (r"(?i)\banswer\.txt\b", "grader answer file in diff"),
        (r"(?i)\btrajectory\.json\b", "grader trajectory file in diff"),
        (r"(?i)\bgold\b", "gold answer reference in diff"),
        (r"(?m)^\s*(?:from|import)\s+(?:tasks|score|cli)\b",
         "answer-parser import in diff"),
    ]
    seen = {pattern for pattern, _ in patterns}

    def add(pattern: str, why: str) -> None:
        if pattern not in seen:
            seen.add(pattern)
            patterns.append((pattern, why))

    tasks = task_map()
    for domain in sorted({task.domain for task in tasks.values()}):
        add(re.escape(domain), f"domain name {domain!r} in diff")
    for task in tasks.values():
        add(re.escape(task.id), f"task id {task.id!r} in diff")
        add(rf"(?<![\w]){re.escape(task.qid)}(?![\w])",
            f"task id {task.qid!r} in diff")
        answer = str(task.gold).strip()
        if len(answer) >= 5 or re.search(r"[.,%$]", answer):
            add(rf"(?<![\w.,]){re.escape(answer)}(?![\w.,])",
                f"gold answer for {task.id!r} in diff")
        for reference in task.refs:
            path = reference.get("path") if isinstance(reference, dict) else None
            if path:
                add(re.escape(path), f"reference file for {task.id!r} in diff")
    return patterns


DOMAIN = AgoraDomain()
