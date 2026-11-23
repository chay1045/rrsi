"""Agora index builder: chunk markdown and tables, one SQLite FTS5 db per domain.

    python3 domains/agora/index/build.py --domain "Finance & Economics"

Not evolvable (PLAN.md appendix B): RRSI candidates run in git worktrees of this
repo and must never read or edit the index or its builder. The index lives in an
absolute cache dir for the same reason -- a file inside the worktree would be
rebuilt (or corrupted) by every candidate.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sqlite3
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

DATA_DIR = Path(os.environ.get("AGORA_DATA", Path.home() / "work" / "agora-data"))
DOCS_DIR = DATA_DIR / "documents"
INDEX_DIR = (
    Path(os.environ.get("RRSI_CACHE_DIR", Path.home() / ".cache" / "rrsi"))
    / "agora"
    / "index"
)

MIN_TOK, MAX_TOK, OVERLAP_TOK = 512, 800, 80
SUFFIXES = {".md": "md", ".xlsx": "xlsx", ".csv": "csv"}
HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
CJK = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

SCHEMA = """
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS chunks_fts;
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    doc_path TEXT NOT NULL,  -- "documents/<domain>/<file>", same form as tasks.reference_files
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    text TEXT NOT NULL
);
-- shortcut: unicode61 keeps a whole CJK run as one token (5614/5966 Finance docs
-- hold Chinese/Japanese text), so CJK queries must use prefix search ("*") or
-- exact runs; move to a trigram/second FTS table if recall@10 for CJK falls short.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='chunk_id', tokenize='porter unicode61');
"""


@dataclass(frozen=True)
class Unit:
    """One paragraph, heading or spreadsheet row, before packing."""

    heading_path: str
    line_start: int
    line_end: int
    text: str


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    line_start: int
    line_end: int
    text: str


def n_tokens(text: str) -> int:
    """Cheap token estimate: 1 per CJK character, 1 per 4 other characters.

    shortcut: not the policy tokenizer, just enough to hold the 512-800 window;
    swap in the real tokenizer if chunk budgets ever need to be exact.
    """
    cjk = len(CJK.findall(text))
    return cjk + (len(text) - cjk) // 4


def _md_units(text: str, title: str) -> Iterator[Unit]:
    """Split markdown on headings and blank lines; each unit keeps a breadcrumb."""
    lines = text.splitlines()
    path: list[tuple[int, str]] = []
    block: list[str] = []
    start = 0

    def crumbs() -> str:
        return " > ".join([title, *(heading for _, heading in path)])

    def take(end: int) -> Unit | None:
        nonlocal block
        if not block:
            return None
        unit = Unit(crumbs(), start, end, "\n".join(block))
        block = []
        return unit

    for lineno, line in enumerate(lines, 1):
        heading = HEADING.match(line)
        if heading or not line.strip():
            unit = take(lineno - 1)
            if unit:
                yield unit
            if heading:
                level, head = len(heading.group(1)), heading.group(2)
                path[:] = [p for p in path if p[0] < level] + [(level, head)]
                yield Unit(crumbs(), lineno, lineno, head)
            continue
        if not block:
            start = lineno
        block.append(line)
    unit = take(len(lines))
    if unit:
        yield unit


def _row(heading_path: str, line: int, cells: list[tuple[str, object]]) -> Unit | None:
    values = [(col, str(value).strip()) for col, value in cells
              if value is not None and str(value).strip()]
    if not values:
        return None
    return Unit(heading_path, line, line, " | ".join(f"{col}: {v}" for col, v in values))


def _csv_units(path: Path) -> Iterator[Unit]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            unit = _row(path.stem, lineno,
                        [(get_column_letter(i), v) for i, v in enumerate(row, 1)])
            if unit:
                yield unit


def _load_workbook(path: Path) -> openpyxl.Workbook:
    try:
        return openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        # One OECD export has a stylesheet openpyxl cannot parse (a Fill with no
        # patternType); styles carry no cell values, so read the zip without it.
        buf = io.BytesIO()
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(buf, "w") as out:
            for item in src.infolist():
                if item.filename != "xl/styles.xml":
                    out.writestr(item, src.read(item.filename))
        buf.seek(0)
        return openpyxl.load_workbook(buf, read_only=True, data_only=True)


def _xlsx_units(path: Path) -> Iterator[Unit]:
    wb = _load_workbook(path)
    for sheet in wb.sheetnames:
        for lineno, row in enumerate(wb[sheet].iter_rows(values_only=True), 1):
            unit = _row(f"{path.stem} > {sheet}", lineno,
                        [(get_column_letter(i), v) for i, v in enumerate(row, 1)])
            if unit:
                yield unit


def _units(path: Path) -> Iterator[Unit]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        yield from _md_units(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    elif suffix == ".csv":
        yield from _csv_units(path)
    elif suffix == ".xlsx":
        yield from _xlsx_units(path)


def _fit(unit: Unit) -> Iterator[Unit]:
    """Cut a unit that alone exceeds MAX_TOK into pieces (same text, real line spans)."""
    text = unit.text
    if n_tokens(text) <= MAX_TOK:
        yield unit
        return
    pos, line = 0, unit.line_start
    while pos < len(text):
        end = min(len(text), pos + 4 * MAX_TOK)  # 4 chars/token is the widest estimate
        while end > pos and n_tokens(text[pos:end]) > MAX_TOK:
            end = max(pos + 1, end - 200)
        if end < len(text):
            space = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
            if space > pos + (end - pos) // 2:
                end = space + 1
        piece = text[pos:end]
        newlines = piece.count("\n")
        yield Unit(unit.heading_path, line, line + newlines, piece.strip())
        line += newlines
        pos = end


def chunks(units: list[Unit]) -> list[Chunk]:
    """Pack units into 512-800 token chunks, flushing at headings once full."""
    groups: list[list[Unit]] = []
    group: list[Unit] = []
    size = 0
    for unit in [piece for unit in units for piece in _fit(unit)]:
        n = n_tokens(unit.text) + 1  # +1 for the "\n\n" join and per-unit rounding
        overflow = size + n > MAX_TOK
        if group and (overflow or (size >= MIN_TOK and unit.heading_path != group[0].heading_path)):
            groups.append(group)
            # The next unit may split a fact; when the split is for size, start
            # the new chunk with the tail of the old one (about 10% overlap).
            tail = group[-1]
            tail = tail if overflow and n_tokens(tail.text) + n <= MAX_TOK else None
            tail = tail if tail and n_tokens(tail.text) <= OVERLAP_TOK else None
            group = [tail] if tail else []
            size = n_tokens(tail.text) if tail else 0
        group.append(unit)
        size += n
    if group:
        groups.append(group)

    if len(groups) > 1:
        tail = sum(n_tokens(u.text) + 1 for u in groups[-1])
        prev = sum(n_tokens(u.text) + 1 for u in groups[-2])
        if tail < MIN_TOK and prev + tail <= MAX_TOK:
            groups[-2].extend(groups.pop())

    return [
        Chunk(g[0].heading_path, g[0].line_start, g[-1].line_end,
              "\n\n".join(u.text for u in g))
        for g in groups
    ]


def _slug(domain: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")


def build_domain(domain: str, docs_dir: Path = DOCS_DIR,
                 index_dir: Path = INDEX_DIR) -> tuple[dict[str, int], int, Path]:
    """(re)build one domain's index from scratch; return (file counts, chunks, db)."""
    root = Path(docs_dir) / domain
    if not root.is_dir():
        raise SystemExit(f"no domain directory: {root}")

    db_path = Path(index_dir) / f"{_slug(domain)}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)  # a killed build must not look like a good db

    counts = {"md": 0, "xlsx": 0, "csv": 0}
    total = 0
    con = sqlite3.connect(tmp_path)
    try:
        con.executescript(SCHEMA)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            kind = SUFFIXES.get(path.suffix.lower())
            if kind is None:
                continue
            doc_path = f"documents/{domain}/{path.relative_to(root).as_posix()}"
            rows = [(domain, doc_path, c.line_start, c.line_end, c.heading_path, c.text)
                    for c in chunks(list(_units(path)))]
            con.executemany(
                "INSERT INTO chunks (domain, doc_path, line_start, line_end, heading_path, text)"
                " VALUES (?, ?, ?, ?, ?, ?)", rows)
            counts[kind] += 1
            total += len(rows)
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        con.commit()
    finally:
        con.close()
    tmp_path.replace(db_path)
    return counts, total, db_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", action="append", metavar="NAME",
                    help="exact domain directory name; repeatable; default: all")
    ap.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = ap.parse_args(argv)

    if not args.docs_dir.is_dir():
        ap.error(f"no documents dir: {args.docs_dir} (set AGORA_DATA or --docs-dir)")
    all_domains = sorted(p.name for p in args.docs_dir.iterdir() if p.is_dir())
    domains = args.domain or all_domains
    for domain in domains:
        if domain not in all_domains:
            ap.error(f"unknown domain {domain!r}; choose from: {', '.join(all_domains)}")

    totals = {"md": 0, "xlsx": 0, "csv": 0}
    total_chunks = 0
    for domain in domains:
        started = time.perf_counter()
        counts, n_chunks, db_path = build_domain(domain, args.docs_dir, args.index_dir)
        elapsed = time.perf_counter() - started
        print(f"{domain}: {counts['md']} md + {counts['xlsx']} xlsx + {counts['csv']} csv"
              f" -> {n_chunks} chunks in {elapsed:.1f}s -> {db_path}")
        for kind in totals:
            totals[kind] += counts[kind]
        total_chunks += n_chunks
    if len(domains) > 1:
        print(f"total: {totals['md']} md + {totals['xlsx']} xlsx + {totals['csv']} csv"
              f" -> {total_chunks} chunks in {len(domains)} domains")


if __name__ == "__main__":
    main()
