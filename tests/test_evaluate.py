"""Issue #04: Gate-2 retrieval metrics (recall_file@10, recall_region@10, MRR@10).

    python3 -m pytest -q tests/test_evaluate.py
"""
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"

spec = spec_from_file_location("agora_index_evaluate", AGORA / "index" / "evaluate.py")
evaluate = module_from_spec(spec)
sys.modules[spec.name] = evaluate
spec.loader.exec_module(evaluate)

import search  # noqa: E402  (index/ is on sys.path after loading evaluate)


def test_evaluate_uses_the_shared_query_builder():
    # Gate 2 must measure the same query the harness's search_docs will send.
    assert evaluate.match_expression is search.match_expression
    assert evaluate.search is search.search


def test_match_expression_drops_glue_and_quotes_terms():
    expression = evaluate.match_expression("What is the total of 4.2 percent?")
    assert expression == '"total" OR "4" OR "2" OR "percent"'


def test_page_spans_run_until_the_next_marker():
    text = "intro\n<!-- pages 1-5 -->\none\ntwo\n<!-- pages 6-10 -->\nthree\n"
    spans = evaluate.page_spans(text)
    assert spans == [(1, 5, 2, 4), (6, 10, 5, 6)]


def test_ranks_find_file_and_region_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.build, "DATA_DIR", tmp_path)
    doc = tmp_path / "documents" / "Law" / "a.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("head\n<!-- pages 1-5 -->\nalpha\n<!-- pages 6-10 -->\nbeta\n")
    refs = [{"path": "documents/Law/a.md", "page_start": 6, "page_end": 10}]
    hits = [
        ("documents/Law/other.md", 1, 10),  # not gold
        ("documents/Law/a.md", 1, 2),       # gold file, page 1-5, not the region
        ("documents/Law/a.md", 4, 5),       # overlaps page 6-10
    ]
    assert evaluate.ranks(hits, refs, {}) == (2, 3)


def test_whole_document_reference_has_no_region_penalty(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.build, "DATA_DIR", tmp_path)
    refs = [{"path": "documents/Law/t.csv", "page_start": None, "page_end": None}]
    hits = [("documents/Law/other.md", 1, 10), ("documents/Law/t.csv", 1, 2)]
    assert evaluate.ranks(hits, refs, {}) == (2, 2)


def test_paged_reference_without_markers_falls_back_to_whole_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.build, "DATA_DIR", tmp_path)
    doc = tmp_path / "documents" / "Law" / "plain.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("no markers here\n")
    refs = [{"path": "documents/Law/plain.md", "page_start": 3, "page_end": 7}]
    hits = [("documents/Law/plain.md", 1, 1)]
    assert evaluate.ranks(hits, refs, {}) == (1, 1)


def test_report_scores_and_verdict():
    good = [(1, 1)] * 3
    bad = [(None, None)]
    markdown = evaluate.report({"A": good, "B": bad}, "bm25", 12.0)
    assert "| A | 3 | 1.000 | 1.000 | 1.000 |" in markdown
    assert "| **all** | 4 | 0.750 | 0.750 | 0.750 |" in markdown
    assert "PASS" in markdown and "Gate 2" in markdown
    assert "#14" in markdown  # the verdict names the next experiment


def test_rrf_prefers_a_chunk_that_both_rankers_found():
    both = (1, "documents/Law/a.md", 1, 1)
    bm25_only = (2, "documents/Law/b.md", 1, 1)
    dense_only = (3, "documents/Law/c.md", 1, 1)
    fused = search.reciprocal_rank_fusion(
        [[bm25_only, both], [dense_only, both]], k=3)
    assert fused[0] == both
    assert {hit[0] for hit in fused} == {1, 2, 3}


def test_ablation_report_records_the_expected_order_and_verdict():
    good = {"A": [(1, 1)] * 2}
    weak = {"A": [(None, None)] * 2}
    runs = {"bm25": (weak, 1.0), "dense": (good, 2.0), "hybrid": (good, 3.0)}
    markdown = evaluate.ablation_report(runs, 6.0)
    assert "| hybrid | 2 | 1.000 | 1.000 | 1.000 | 3 |" in markdown
    assert "Ablation ordering holds" in markdown
    assert "PASS" in markdown


def test_paired_delta_is_paired_and_bounded():
    base = {"A": [(None, None), (1, 1)], "B": [(None, None)]}
    tuned = {"A": [(1, 1), (1, 1)], "B": [(1, 1)]}
    n, delta, low, high = evaluate.paired_delta(base, tuned, resamples=2000)
    assert n == 3
    assert delta == pytest.approx(2 / 3)
    assert 0.0 <= low <= delta <= high <= 1.0


def test_paired_delta_orders_tasks_the_same_way():
    base = {"A": [(None, None), (1, 1)], "B": [(1, 1)]}
    tuned = {"A": [(1, 1), (None, None)], "B": [(1, 1)]}
    _, delta, _, _ = evaluate.paired_delta(base, tuned, resamples=100)
    assert delta == 0.0


def test_ablation_report_records_rerank_delta_and_ci():
    base = {"A": [(None, None), (1, 1), (2, 2), (None, None)]}
    tuned = {"A": [(1, 1), (1, 1), (2, 2), (None, None)]}
    runs = {"hybrid": (base, 1.0), "hybrid+rerank": (tuned, 1.5)}
    markdown = evaluate.ablation_report(
        runs, 2.0, rerank_seconds=0.5, rerank_calls=4)
    assert "Rerank delta: recall_file@10 +0.250" in markdown
    assert "95% CI" in markdown
    assert "Score latency: 0.5s over 4 calls (125.0 ms/task)" in markdown
    assert "Rerank clears both file and region recall at the gate" in markdown


def test_rerank_requires_the_hybrid_mode(capsys):
    with pytest.raises(SystemExit):
        evaluate.main(["--mode", "bm25", "--rerank"])
    assert "--mode hybrid" in capsys.readouterr().err


def test_ablation_report_marks_the_rerank_as_measured():
    scored = [(1, None)] * 5 + [(1, 1)] * 3 + [(None, None)] * 2
    runs = {"hybrid": ({"A": scored}, 1.0),
            "hybrid+rerank": ({"A": list(scored)}, 1.0)}
    markdown = evaluate.ablation_report(runs, 2.0)
    assert "measured below" in markdown
    assert "worth measuring" not in markdown
