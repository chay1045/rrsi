"""Issue #02: Agora's exact-match score. It is strict by design, so most of
these tests pin down what does *not* match; loosening one is a scoring change.

    python3 -m pytest -q tests/test_score.py
"""
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"


def _load(filename):
    # By path, not sys.path: domains/agora would stay on sys.path for the whole
    # pytest session and shadow any other module named `score` or `tasks`.
    spec = spec_from_file_location(f"agora_{filename}", AGORA / f"{filename}.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks its own module up here
    spec.loader.exec_module(module)
    return module


agora_score = _load("score")
agora_tasks = _load("tasks")
score = agora_score.score


@pytest.mark.parametrize("gold,prediction", [
    ("10.37 pp", "10.37 pp"),
    ("-11.887%", "-11.887%"),
    ("  10.37   PP  ", "10.37 pp"),          # whitespace and case are normalized
    ("`10.37 pp`", "10.37 pp"),              # backticks / quotes stripped
    ('"1,234.5"', "1,234.5"),
    ("UNKNOWN", "unknown"),                  # casefold also folds UNKNOWN
    ("注文住宅: 10.36%", "注文住宅: 10.36%"),
])
def test_matches(gold, prediction):
    assert score(gold, prediction) == 1.0


@pytest.mark.parametrize("gold,prediction", [
    ("10.37 pp", "10.37"),                   # trailing unit dropped
    ("10.37 pp", "10.37 %"),                 # pp is not %
    ("10.37 percentage points", "10.37 pp"),  # spelled out vs abbreviated
    ("1,234", "1234"),                       # thousands separator
    ("-11.887%", "11.887%"),                 # sign
    ("-11.887%", "+11.887%"),
    ("10.37%", "10.4%"),                     # rounding
    ("10.37%", "10.36%"),
    ("5.517x", "5.517"),                     # ratio unit
    ("402.18 responses", "402.18 response"),
    ("UNKNOWN", "unknown value"),
])
def test_mismatches(gold, prediction):
    assert score(gold, prediction) == 0.0


@pytest.mark.parametrize("prediction", ["", "   ", None])
def test_empty_prediction(prediction):
    assert score("10.37 pp", prediction) == 0.0
    assert score("", prediction) == 0.0


@pytest.mark.skipif(not agora_tasks.TASKS_DIR.is_dir(), reason="data delivery not mounted")
def test_gold_answers_match_themselves():
    tasks = agora_tasks.load_tasks()
    assert len(tasks) == 362
    assert len({d for d, _ in tasks}) == 8
    assert sum(score(t.gold, t.gold) for t in tasks.values()) / len(tasks) == 1.0


@pytest.mark.skipif(not agora_tasks.TASKS_DIR.is_dir(), reason="data delivery not mounted")
def test_gold_lookup():
    tasks = agora_tasks.load_tasks()
    task = next(iter(tasks.values()))
    assert agora_tasks.gold(task.id, tasks) == task.gold
    assert tasks[(task.domain, task.qid)] is task


def test_unmounted_delivery_is_an_error_not_an_empty_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        agora_tasks.load_tasks(tmp_path)
