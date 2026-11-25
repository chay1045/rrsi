"""Gate 2: retrieval quality with zero rollouts -- no GPU, no policy, no judge.

    python3 domains/agora/index/evaluate.py --mode bm25 --out ../../docs/retrieval.md

Modes: ``bm25``, ``dense`` and ``hybrid``. Repeat ``--mode`` to write one
ablation report. Dense and hybrid need ``build.py --embed`` and the local
EmbeddingGemma 2 text tower.

For every task, retrieve the top-10 chunks from the domain's index and score:

    recall_file@10    a retrieved chunk's doc_path is one of the task's reference_files
    recall_region@10  ... and the chunk's line range overlaps the gold page range
    MRR@10            mean reciprocal rank of the first gold-file chunk

Gold pages become line ranges through the ``<!-- pages a-b -->`` markers Agora's
parser left in the markdown. A reference with no page range (every xlsx/csv, some
md) is a whole-document hit; a paged reference to a file without markers falls
back to the same and the report says so.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # same directory, shares INDEX_DIR/DATA_DIR
from search import (DEPTH, K, bm25_chunks, dense_chunks, hybrid_chunks,  # noqa: F401
                    match_expression, search)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tasks as agora_tasks  # noqa: E402  (path insert must come first)

GATE = 0.70
MODES = ("bm25", "dense", "hybrid")
PAGE = re.compile(r"<!-- pages (\d+)-(\d+) -->")

WHOLE_DOC = [(1, 1 << 60)]


def page_spans(text: str) -> list[tuple[int, int, int, int]]:
    """``[(page_start, page_end, line_start, line_end)]`` from page markers."""
    lines = text.splitlines()
    spans: list[tuple[int, int, int, int]] = []
    for number, line in enumerate(lines, 1):
        found = PAGE.fullmatch(line.strip())
        if not found:
            continue
        if spans:
            start, end, first, _ = spans[-1]
            spans[-1] = (start, end, first, number - 1)
        spans.append((int(found[1]), int(found[2]), number, len(lines)))
    return spans


def region_lines(doc_path: str, page_start: int | None, page_end: int | None,
                 cache: dict[str, list[tuple[int, int, int, int]]]) -> list[tuple[int, int]]:
    """Line ranges of a gold page range; whole doc when pages cannot be mapped."""
    if page_start is None:
        return WHOLE_DOC
    if doc_path not in cache:
        path = build.DATA_DIR / doc_path
        text = path.read_text(encoding="utf-8", errors="replace") if path.suffix == ".md" else ""
        cache[doc_path] = page_spans(text)
    spans = cache[doc_path]
    if not spans:
        return WHOLE_DOC
    return [(first, last) for start, end, first, last in spans
            if start <= page_end and end >= page_start]


def ranks(hits: list[tuple[str, int, int]], refs: list[dict],
          cache: dict[str, list[tuple[int, int, int, int]]]) -> tuple[int | None, int | None]:
    """1-based rank of the first gold-file chunk and first gold-region chunk."""
    file_rank = region_rank = None
    for position, (doc_path, line_start, line_end) in enumerate(hits, 1):
        for ref in refs:
            if ref["path"] != doc_path:
                continue
            if file_rank is None:
                file_rank = position
            for first, last in region_lines(
                    doc_path, ref.get("page_start"), ref.get("page_end"), cache):
                if line_start <= last and first <= line_end:
                    region_rank = region_rank or position
                    break
        if file_rank and region_rank:
            break
    return file_rank, region_rank


def evaluate(domains: list[str],
             all_tasks: dict[tuple[str, str], agora_tasks.Task],
             mode: str = "bm25", embedder=None,
             depth: int = DEPTH) -> dict[str, list[tuple[int | None, int | None]]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    if mode != "bm25" and embedder is None:
        raise ValueError(f"mode {mode!r} needs an embedder")
    per_domain: dict[str, list[tuple[int | None, int | None]]] = {}
    for domain in domains:
        db_path = build.INDEX_DIR / f"{build._slug(domain)}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"{domain}: no index at {db_path} (run build.py)")
        rows = [task for task in all_tasks.values() if task.domain == domain]
        if not rows:
            raise FileNotFoundError(f"{domain}: no tasks in {agora_tasks.TASKS_DIR}")
        cache: dict[str, list[tuple[int, int, int, int]]] = {}
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            vectors = (embedder.encode_queries([task.query for task in rows])
                       if mode != "bm25" else None)
            scored = []
            for position, task in enumerate(rows):
                if mode == "bm25":
                    found = bm25_chunks(con, task.query, K)
                elif mode == "dense":
                    found = dense_chunks(con, vectors[position], K)
                else:
                    found = hybrid_chunks(con, task.query, vectors[position], K, depth)
                hits = [(doc_path, line_start, line_end)
                        for _, doc_path, line_start, line_end in found]
                scored.append(ranks(hits, task.refs, cache))
            per_domain[domain] = scored
        finally:
            con.close()
        print(f"{domain}: {len(rows)} tasks ({mode})", file=sys.stderr)
    return per_domain


def _row(label: str, scored: list[tuple[int | None, int | None]]) -> str:
    n = len(scored)
    files = sum(1 for file_rank, _ in scored if file_rank is not None) / n
    regions = sum(1 for _, region_rank in scored if region_rank is not None) / n
    mrr = sum(1 / file_rank for file_rank, _ in scored if file_rank is not None) / n
    return f"| {label} | {n} | {files:.3f} | {regions:.3f} | {mrr:.3f} |"


def _overall(per_domain: dict[str, list[tuple[int | None, int | None]]]
             ) -> tuple[float, float, float]:
    everything = [pair for scored in per_domain.values() for pair in scored]
    n = len(everything)
    files = sum(1 for file_rank, _ in everything if file_rank is not None) / n
    regions = sum(1 for _, region_rank in everything if region_rank is not None) / n
    mrr = sum(1 / file_rank for file_rank, _ in everything if file_rank is not None) / n
    return files, regions, mrr


def report(per_domain: dict[str, list[tuple[int | None, int | None]]], mode: str,
           elapsed: float) -> str:
    everything = [pair for scored in per_domain.values() for pair in scored]
    lines = [
        "# Retrieval quality (Gate 2)",
        "",
        f"Mode `{mode}`, top-{K} chunks, {len(everything)} tasks, "
        f"generated {date.today().isoformat()} in {elapsed:.0f}s.",
        "",
        "| domain | tasks | recall_file@10 | recall_region@10 | MRR@10 |",
        "|---|---|---|---|---|",
        *[_row(domain, scored) for domain, scored in per_domain.items()],
        _row("**all**", everything),
        "",
    ]
    recall_file = sum(1 for file_rank, _ in everything if file_rank is not None) / len(everything)
    recall_region = sum(1 for _, r in everything if r is not None) / len(everything)
    lines.append(f"Gate 2 (recall_file@10 >= {GATE:.2f}): "
                 f"**{'PASS' if recall_file >= GATE else 'FAIL'}** ({recall_file:.3f}).")
    if recall_file < GATE:
        if mode == "bm25":
            lines.append(
                "#14 (dense/hybrid) is needed: BM25 misses whole documents, and reranking "
                "cannot recover files it never retrieved.")
        else:
            lines.append(f"Mode `{mode}` is below the gate; record the plateau or "
                         "re-index once with a different embedder.")
    elif recall_region < GATE:
        lines.append(
            f"File recall clears the gate but region recall does not ({recall_region:.3f}); "
            f"#15 (reranker) is worth measuring before agents run.")
    else:
        if mode == "bm25":
            lines.append("#14 and #15 are not needed to clear Gate 2; "
                         "keep them as later ablations.")
        else:
            lines.append("File and region recall both clear the gate.")
    lines.append("")
    lines.append(
        "Caveat: gold pages are mapped to lines through `<!-- pages a-b -->` markers; "
        "paged references to markdown without markers (and every xlsx/csv reference) "
        "count as whole-document hits.")
    return "\n".join(lines) + "\n"


def ablation_report(runs: dict[str, tuple[dict[str, list[tuple[int | None, int | None]]],
                                          float]],
                    elapsed: float) -> str:
    """One report for repeated ``--mode`` flags: summary, per-domain tables, verdict."""
    first = next(iter(runs.values()))[0]
    tasks = sum(len(scored) for scored in first.values())
    lines = [
        "# Retrieval quality (Gate 2)",
        "",
        f"Top-{K} chunks, {tasks} tasks, generated {date.today().isoformat()} "
        f"in {elapsed:.0f}s.",
        "",
        "## Ablation",
        "",
        "| mode | tasks | recall_file@10 | recall_region@10 | MRR@10 | seconds |",
        "|---|---|---|---|---|---|",
    ]
    metrics: dict[str, tuple[float, float, float]] = {}
    for mode, (per_domain, seconds) in runs.items():
        files, regions, mrr = _overall(per_domain)
        metrics[mode] = (files, regions, mrr)
        lines.append(f"| {mode} | {tasks} | {files:.3f} | {regions:.3f} | "
                     f"{mrr:.3f} | {seconds:.0f} |")
    lines.append("")
    for mode, (per_domain, _) in runs.items():
        lines += [
            f"### {mode}",
            "",
            "| domain | tasks | recall_file@10 | recall_region@10 | MRR@10 |",
            "|---|---|---|---|---|",
            *[_row(domain, scored) for domain, scored in per_domain.items()],
            _row("**all**", [pair for scored in per_domain.values() for pair in scored]),
            "",
        ]

    hybrid = metrics.get("hybrid")
    if hybrid is not None:
        gate = hybrid[0] >= GATE
        lines.append(f"Gate 2 (recall_file@10 >= {GATE:.2f}): "
                     f"**{'PASS' if gate else 'FAIL'}** ({hybrid[0]:.3f}).")
        if not gate:
            lines.append(
                "Hybrid is below the gate. Re-index once with a hosted higher-dim "
                "embedder, or record the plateau; do not run agents against retrieval "
                "you do not trust.")
        elif hybrid[1] < GATE:
            lines.append(
                f"File recall clears the gate but region recall does not "
                f"({hybrid[1]:.3f}); #15 (reranker) is worth measuring.")
        else:
            lines.append("Hybrid clears both file and region recall at the gate.")
        expected = [m for m in ("bm25", "dense", "hybrid") if m in metrics]
        deviations = []
        names = ("recall_file@10", "recall_region@10", "MRR@10")
        for left, right in zip(expected, expected[1:]):
            for name, low, high in zip(names, metrics[left], metrics[right]):
                if high < low:
                    deviations.append(
                        f"{right} {name} {high:.3f} < {left} {low:.3f}")
        if deviations:
            lines.append("Deviation from hybrid >= dense-only >= BM25: "
                         + "; ".join(deviations) + ".")
        else:
            lines.append("Ablation ordering holds on all three metrics: "
                         "hybrid >= dense-only >= BM25.")
    lines.append("")
    lines.append(
        "Caveat: gold pages are mapped to lines through `<!-- pages a-b -->` markers; "
        "paged references to markdown without markers (and every xlsx/csv reference) "
        "count as whole-document hits.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="repeatable; default bm25. dense/hybrid need build.py --embed")
    parser.add_argument("--domain", action="append",
                        help="repeatable; default: every domain with tasks")
    parser.add_argument("--out", type=Path, help="write the markdown here, default stdout")
    args = parser.parse_args(argv)

    all_tasks = agora_tasks.load_tasks()
    known = sorted({task.domain for task in all_tasks.values()})
    domains = args.domain or known
    unknown = [d for d in domains if d not in known]
    if unknown:
        parser.error(f"unknown domain(s): {', '.join(unknown)}")

    modes = list(dict.fromkeys(args.mode or ["bm25"]))
    embedder = None
    if any(mode != "bm25" for mode in modes):
        from embed import EmbeddingGemma
        embedder = EmbeddingGemma()
        print(embedder.describe(), file=sys.stderr)

    started = time.perf_counter()
    runs: dict[str, tuple[dict[str, list[tuple[int | None, int | None]]], float]] = {}
    for mode in modes:
        mode_started = time.perf_counter()
        per_domain = evaluate(domains, all_tasks, mode=mode, embedder=embedder)
        runs[mode] = (per_domain, time.perf_counter() - mode_started)
    elapsed = time.perf_counter() - started
    if len(runs) == 1:
        mode, (per_domain, _) = next(iter(runs.items()))
        markdown = report(per_domain, mode, elapsed)
    else:
        markdown = ablation_report(runs, elapsed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
