"""The frozen Agora split: 8 smoke / 24 evolve / 24 heldout task ids.

The split is generated once from the task corpus and committed with its
SHA-256. The loader refuses a split whose bytes do not match the recorded
digest, so a rollout can never silently run on a different task set.

Selection is domain-stratified and deterministic without ``random``: within a
domain, task ids are ordered by ``sha256("<seed>:<id>")``. The same corpus and
seed therefore produce the same bytes on every machine and Python version.
Smoke is disjoint from evolve and heldout; a task is never reused.

Generating command, from the umbrella root::

    python3 vendor/rrsi/domains/agora/split.py --write --docs docs/split.sha256

``data/split.sha256`` sits beside the data because candidates run in git
worktrees that do not contain the umbrella repository's ``docs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import tasks

HERE = Path(__file__).resolve().parent
SPLIT_PATH = HERE / "data" / "split.json"
SHA_PATH = HERE / "data" / "split.sha256"
SEED = "agora-split-v1"
SETS = ("smoke", "evolve", "heldout")
COUNTS = {"smoke": 8, "evolve": 24, "heldout": 24}


class SplitError(ValueError):
    """The split cannot be built or does not satisfy the frozen contract."""


class SplitHashError(SplitError):
    """``split.json`` does not match the recorded SHA-256."""


def _rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()


def _quotas(total: int, domains: list[str]) -> dict[str, int]:
    """Equal per-domain quota, with the remainder on the first domains."""
    base, extra = divmod(total, len(domains))
    return {domain: base + (index < extra) for index, domain in enumerate(domains)}


def build_split(all_tasks: dict, seed: str = SEED,
                counts: dict[str, int] = COUNTS) -> dict:
    """Return the split for ``all_tasks`` (the mapping from ``tasks.load_tasks``)."""
    pools: dict[str, list[str]] = defaultdict(list)
    for task in all_tasks.values():
        pools[task.domain].append(task.id)
    domains = sorted(pools)
    if not domains:
        raise SplitError("no tasks to split")

    quotas = {name: _quotas(counts[name], domains) for name in SETS}
    picked: dict[str, list[str]] = {name: [] for name in SETS}
    for domain in domains:
        ids = sorted(pools[domain], key=lambda task_id: _rank(seed, task_id))
        need = sum(quotas[name][domain] for name in SETS)
        if len(ids) < need:
            raise SplitError(
                f"{domain}: {need} tasks needed for the split, {len(ids)} available")
        cursor = 0
        for name in SETS:
            size = quotas[name][domain]
            picked[name].extend(ids[cursor:cursor + size])
            cursor += size

    split = {"seed": seed, "counts": dict(counts)}
    for name in SETS:
        split[name] = sorted(picked[name])
    _validate(split)
    return split


def _validate(split: dict) -> None:
    """Reject duplicates, wrong sizes, and a domain missing from one set."""
    if not isinstance(split, dict):
        raise SplitError("expected a JSON object")
    if not isinstance(split.get("seed"), str):
        raise SplitError("seed: expected a string")
    counts = split.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(SETS):
        raise SplitError(f"counts: expected keys {', '.join(SETS)}")

    for name in SETS:
        ids = split.get(name)
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise SplitError(f"{name}: expected a list of task ids")
        if len(ids) != counts[name]:
            raise SplitError(f"{name}: {len(ids)} ids, expected {counts[name]}")
        if any("/" not in task_id for task_id in ids):
            raise SplitError(f"{name}: malformed task id")

    seen: set[str] = set()
    for name in SETS:
        for task_id in split[name]:
            if task_id in seen:
                raise SplitError(f"task id appears twice: {task_id}")
            seen.add(task_id)

    domains = {task_id.rpartition("/")[0] for task_id in seen}
    for name in SETS:
        present = {task_id.rpartition("/")[0] for task_id in split[name]}
        missing = sorted(domains - present)
        if missing:
            raise SplitError(f"{name}: missing domains: {', '.join(missing)}")


def _dumps(split: dict) -> bytes:
    """One canonical encoding, so the digest is stable across runs."""
    return (json.dumps(split, indent=1, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_split(all_tasks: dict | None = None, split_path: Path = SPLIT_PATH,
                sha_path: Path = SHA_PATH, docs_path: Path | None = None) -> dict:
    """Build, write and record the split; optionally write the umbrella copy."""
    split = build_split(all_tasks if all_tasks is not None else tasks.load_tasks())
    raw = _dumps(split)
    digest = hashlib.sha256(raw).hexdigest()

    _write(split_path, raw)
    _write(sha_path, f"{digest}  {os.path.relpath(split_path, sha_path.parent)}\n"
           .encode("utf-8"))
    if docs_path is not None:
        root = Path(docs_path).resolve().parent.parent
        line = f"{digest}  {os.path.relpath(Path(split_path).resolve(), root)}\n"
        _write(Path(docs_path), line.encode("utf-8"))
    return split


def load_split(split_path: Path = SPLIT_PATH,
               sha_path: Path = SHA_PATH) -> dict:
    """Load the frozen split, or raise when the bytes do not match the digest."""
    try:
        raw = Path(split_path).read_bytes()
        expected = Path(sha_path).read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError, UnicodeDecodeError) as exc:
        raise SplitHashError(f"cannot read the frozen split: {exc}") from exc

    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise SplitHashError(
            f"{split_path}: sha256 {actual} does not match {expected} "
            f"in {sha_path}")

    split = json.loads(raw)
    _validate(split)
    return split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="regenerate split.json and both hash files")
    mode.add_argument("--check", action="store_true",
                      help="verify the committed split (default)")
    parser.add_argument("--docs", type=Path, default=None,
                        help="also write the umbrella copy, e.g. docs/split.sha256")
    args = parser.parse_args(argv)

    if args.write:
        split = write_split(docs_path=args.docs)
        print(f"wrote {SPLIT_PATH}")
        print(json.dumps(split["counts"], sort_keys=True))
        return 0

    split = load_split()
    print(f"ok {SPLIT_PATH}")
    print(json.dumps(split["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
