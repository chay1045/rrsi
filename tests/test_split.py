"""Issue #10: the frozen Agora split, with no model, network or real corpus.

    python3 -m pytest -q tests/test_split.py

The real ``domains/agora/data/split.json`` is checked when present: its bytes
must match the committed digest, and its ids must satisfy the same contract.
"""

import hashlib
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"


def _load(filename, name=None):
    # By path, not sys.path: domains/agora would stay on sys.path for the whole
    # pytest session and shadow any other module named `tasks` or `split`.
    name = name or f"agora_{filename}"
    spec = spec_from_file_location(name, AGORA / f"{filename}.py")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "tasks" not in sys.modules:  # split.py imports its sibling
    _load("tasks", "tasks")
agora_split = _load("split")

DOMAINS = [
    "Agriculture, Resources & Energy",
    "Architecture, Construction, Real Estate & Facilities",
    "Business, Management, Marketing & Sales",
    "Education, Science & Academia",
    "Finance & Economics",
    "Healthcare & Medicine",
    "Law",
    "Technology, Software & Manufacturing",
]


@dataclass(frozen=True)
class FakeTask:
    domain: str
    qid: str

    @property
    def id(self) -> str:
        return f"{self.domain}/{self.qid}"


def corpus(per_domain: int = 10) -> dict:
    return {(domain, f"q_{index:04d}"): FakeTask(domain, f"q_{index:04d}")
            for domain in DOMAINS for index in range(per_domain)}


def test_counts_coverage_and_disjoint():
    split = agora_split.build_split(corpus())

    assert split["counts"] == {"smoke": 8, "evolve": 24, "heldout": 24}
    for name, size in split["counts"].items():
        assert len(split[name]) == size
        assert {task_id.rpartition("/")[0]
                for task_id in split[name]} == set(DOMAINS)

    all_ids = split["smoke"] + split["evolve"] + split["heldout"]
    assert len(all_ids) == len(set(all_ids)) == 56


def test_per_domain_strata():
    split = agora_split.build_split(corpus())
    for domain in DOMAINS:
        assert sum(task.startswith(domain + "/") for task in split["smoke"]) == 1
        assert sum(task.startswith(domain + "/") for task in split["evolve"]) == 3
        assert sum(task.startswith(domain + "/") for task in split["heldout"]) == 3


def test_deterministic_regardless_of_input_order():
    tasks = corpus()
    first = agora_split.build_split(tasks)
    second = agora_split.build_split(dict(reversed(list(tasks.items()))))
    assert first == second
    assert agora_split._dumps(first) == agora_split._dumps(second)


def test_small_domain_is_rejected():
    tasks = corpus(per_domain=6)  # needs 1 + 3 + 3
    with pytest.raises(agora_split.SplitError, match="7 tasks needed"):
        agora_split.build_split(tasks)


def test_write_and_load_roundtrip(tmp_path):
    path, sha = tmp_path / "split.json", tmp_path / "split.sha256"
    written = agora_split.write_split(corpus(), split_path=path, sha_path=sha)

    assert agora_split.load_split(path, sha) == written
    assert sha.read_text().split()[0] == hashlib.sha256(
        path.read_bytes()).hexdigest()


def test_tampered_split_is_refused(tmp_path):
    path, sha = tmp_path / "split.json", tmp_path / "split.sha256"
    agora_split.write_split(corpus(), split_path=path, sha_path=sha)
    path.write_bytes(path.read_bytes().replace(b'"smoke"', b'"smokf"', 1))

    with pytest.raises(agora_split.SplitHashError, match="does not match"):
        agora_split.load_split(path, sha)


def test_committed_split_is_frozen():
    if not agora_split.SPLIT_PATH.exists():
        pytest.skip("no committed split in this checkout")
    split = agora_split.load_split()
    assert split["counts"] == {"smoke": 8, "evolve": 24, "heldout": 24}


def test_committed_ids_exist_in_the_corpus():
    if not agora_split.SPLIT_PATH.exists():
        pytest.skip("no committed split in this checkout")
    try:
        all_tasks = agora_split.tasks.load_tasks()
    except FileNotFoundError:
        pytest.skip("no Agora task corpus in this checkout")

    known = {task.id for task in all_tasks.values()}
    split = agora_split.load_split()
    for name in agora_split.SETS:
        assert set(split[name]) <= known
