"""The 362 Agora tasks, keyed by ``(domain, qid)``.

Task ids travel through rrsi as one string, ``"<domain>/<qid>"`` (domain names
contain spaces and ``&`` but never ``/``), so ``gold()`` parses that form back.

The delivery is a symlink at the umbrella root and deliberately not inside a
rrsi worktree, so the directory comes from the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("AGORA_DATA", Path.home() / "work" / "agora-data"))
TASKS_DIR = DATA_DIR / "tasks"


@dataclass(frozen=True)
class Task:
    domain: str
    qid: str
    query: str
    gold: str
    refs: list[dict]  # reference_files: [{"path", "page_start", "page_end"}]

    @property
    def id(self) -> str:
        return f"{self.domain}/{self.qid}"


def load_tasks(tasks_dir: Path | None = None) -> dict[tuple[str, str], Task]:
    tasks: dict[tuple[str, str], Task] = {}
    root = Path(tasks_dir or TASKS_DIR)
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[(row["domain"], row["id"])] = Task(
                    row["domain"], row["id"], row["query"], str(row["answer"]),
                    row.get("reference_files") or [])
    if not tasks:
        # An unmounted AGORA_DATA must not read as "0 tasks, score 0.0".
        raise FileNotFoundError(f"no tasks/*.jsonl under {root}")
    return tasks


def gold(task_id: str, tasks: dict[tuple[str, str], Task]) -> str:
    """Gold answer of ``"<domain>/<qid>"`` (``Task.id``)."""
    domain, _, qid = task_id.rpartition("/")
    return tasks[(domain, qid)].gold
