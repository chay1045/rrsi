"""Issue #09: the Agora Domain adapter, with no model, Docker or network.

The unit tests cover the frozen split, the score fold, the guard scan, the
gold-redacted task stage, the leakage denylist and every component family.
``run()`` itself is exercised by ``rrsi.py --domain agora smoke``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rrsi.components import K, normalize      # noqa: E402
from rrsi.critic import precheck              # noqa: E402
from rrsi.domain import load_domain           # noqa: E402

DOMAIN = load_domain("agora")
ADAPTER = sys.modules["domains.agora.adapter"]

ROWS = [
    {"id": "q_0001", "domain": "Finance & Economics",
     "query": "what is the margin?", "answer": "10.37 pp",
     "reference_files": [{"path": "documents/Finance & Economics/a.md"}]},
    {"id": "q_0002", "domain": "Finance & Economics",
     "query": "what is the rate?", "answer": "6.3",
     "reference_files": []},
]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "Finance & Economics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ROWS) + "\n", encoding="utf-8")
    monkeypatch.setattr(ADAPTER.agora_tasks, "TASKS_DIR", tasks_dir)
    ADAPTER._load_tasks.cache_clear()
    DOMAIN._critic_patterns = None
    yield tasks_dir
    ADAPTER._load_tasks.cache_clear()
    DOMAIN._critic_patterns = None


def test_frozen_split_is_stratified_and_disjoint():
    evolve, heldout, smoke = (set(DOMAIN.evolve_ids()), set(DOMAIN.heldout_ids()),
                              set(DOMAIN.smoke_ids()))
    assert (len(evolve), len(heldout), len(smoke)) == (24, 24, 8)
    assert not (evolve & heldout or evolve & smoke or heldout & smoke)
    domains = {task_id.rpartition("/")[0] for task_id in evolve | heldout | smoke}
    assert len(domains) == 8


def test_score_folds_exact_match_tokens_and_missing(tmp_path, corpus):
    runs = tmp_path / "runs"
    job = runs / "jobs" / "base"
    for trial, answer, tokens in ((0, "10.37 pp\n", 123), (1, None, None)):
        out = job / f"s{trial}" / "Finance & Economics__q_0001"
        out.mkdir(parents=True)
        (out / "trajectory.json").write_text(json.dumps(
            {"status": "ok", "usage": {"total_tokens": tokens or 0}}))
        if answer is not None:
            (out / "answer.txt").write_text(answer)
    out = job / "s0" / "Finance & Economics__q_0002"
    out.mkdir(parents=True)
    (out / "answer.txt").write_text("wrong\n")
    (out / "trajectory.json").write_text(json.dumps(
        {"status": "ok", "usage": {"total_tokens": 7}}))

    ids = ["Finance & Economics/q_0001", "Finance & Economics/q_0002"]
    per, extra = DOMAIN.score(runs, "base", ids, 2)

    first = per[ids[0]]
    assert first.rewards == [1.0, 0.0] and first.missing == 1
    assert first.tokens == [123, None]
    assert per[ids[1]].rewards == [0.0, 0.0] and per[ids[1]].missing == 1
    assert extra["n_trials"] == 4
    assert extra["exact_rate"] == 0.25
    assert extra["parseable_rate"] == 0.5
    assert extra["missing"] == 2 and extra["missing_rate"] == 0.5
    assert extra["guard_violations"] == []


def test_guards_reject_network_violation_and_rate_drop():
    incumbent = SimpleNamespace(extra={"parseable_rate": 0.9, "missing_rate": 0.05})
    candidate = SimpleNamespace(extra={
        "parseable_rate": 0.5, "missing_rate": 0.2,
        "guard_violations": ["domains/agora/harness/tools.py:1: network URL"]})

    problems = DOMAIN.guards(incumbent, candidate)
    assert any("network URL" in problem for problem in problems)
    assert any("parseable-answer rate fell" in problem for problem in problems)
    assert any("missing-trial rate rose" in problem for problem in problems)

    clean = SimpleNamespace(extra={"parseable_rate": 0.9, "missing_rate": 0.05})
    assert DOMAIN.guards(incumbent, clean) == []


def test_guard_scan_rejects_url_and_network_client(tmp_path):
    harness = tmp_path / "domains" / "agora" / "harness"
    harness.mkdir(parents=True)
    (harness / "tools.py").write_text(
        "import requests\nURL = 'https://example.com'\n"
        "open('/out/answer.txt', 'w')\n", encoding="utf-8")

    hits = ADAPTER._scan_harness(tmp_path)
    assert any("network client import" in hit for hit in hits)
    assert any("network URL" in hit for hit in hits)
    assert any("grader output path" in hit for hit in hits)
    assert any("grader answer file" in hit for hit in hits)
    assert ADAPTER._scan_harness(ROOT) == []


def test_stage_task_redacts_gold_and_keeps_one_domain(tmp_path, corpus):
    task = ADAPTER.task_map()[("Finance & Economics", "q_0001")]
    task_file = ADAPTER._stage_task(tmp_path / "stage", task)

    row = json.loads(task_file.read_text(encoding="utf-8"))
    assert row == {"id": "q_0001", "domain": "Finance & Economics",
                   "query": "what is the margin?", "answer": ""}
    assert [path.name for path in (tmp_path / "stage" / "tasks").iterdir()] == [
        "Finance & Economics.jsonl"]


def test_critic_patterns_cover_task_gold_reference_domain_and_url(corpus):
    patterns = DOMAIN.critic_patterns
    diff = ("+# Finance & Economics/q_0001 = 10.37 pp\n"
            "+path = 'documents/Finance & Economics/a.md'\n"
            "+url = 'http://example.com'\n"
            "+open('/out/answer.txt', 'w')\n")

    hits = precheck(diff, patterns)
    assert any("task id" in hit for hit in hits)
    assert any("gold answer" in hit for hit in hits)
    assert any("reference file" in hit for hit in hits)
    assert any("domain name" in hit for hit in hits)
    assert any("network URL" in hit for hit in hits)
    assert any("grader" in hit for hit in hits)


def test_constitution_and_briefs_are_present():
    skill, patterns = DOMAIN.constitution(ROOT)
    assert "Proposer Constitution" in skill
    assert "P1." in patterns
    assert set(DOMAIN.briefs) == {"analyst", "digester", "proposer", "critic"}
    for text in DOMAIN.briefs.values():
        assert len(text) > 200


def test_component_signals_map_every_diff_family():
    cases = {
        "prompt": '-SYSTEM = "old"\n+SYSTEM = "new"\n',
        "context_mgmt": '+\tif total > CONTEXT_CHARS:\n+\t\tkept = trim(kept)\n',
        "output_plumbing": '+\tsnippet = text[:SNIPPET_CHARS]\n',
        "client_tool": '+\thits = index_search.search(con, query, k)\n',
        "control_flow": '+\tif state["turns"] >= MAX_TURNS:\n',
        "config": '+\tTEMPERATURE = 0.6\n',
    }
    for expected, diff in cases.items():
        assert normalize(None, diff, DOMAIN.component_signals) == expected
    assert set(cases) <= set(K)
