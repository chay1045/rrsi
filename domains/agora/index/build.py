"""Agora index builder: chunk markdown and tables, one hybrid SQLite db per domain.

    python3 domains/agora/index/build.py --domain "Finance & Economics" --embed

The default build writes the FTS5 table only. ``--embed`` also writes 256d
EmbeddingGemma 2 vectors to the sqlite-vec ``chunk_vectors`` table, which the
hybrid query path needs.

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
import resource
import sqlite3
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

from embed import DIMS, Embedder, EmbeddingGemma, check_vectors

DATA_DIR = Path(os.environ.get("AGORA_DATA", Path.home() / "work" / "agora-data"))
DOCS_DIR = DATA_DIR / "documents"
INDEX_DIR = (
    Path(os.environ.get("RRSI_CACHE_DIR", Path.home() / ".cache" / "rrsi"))
    / "agora"
    / "index"
)

MIN_TOK, MAX_TOK, OVERLAP_TOK = 512, 800, 80
STRAY_TOK = 32  # footer/page-number leftovers, folded into a neighbour chunk
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
    text TEXT NOT NULL,  -- original text; what retrieval returns
    cjk TEXT NOT NULL  -- cjk_split(text): searchable copy for CJK phrase queries
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, cjk, content='chunks', content_rowid='chunk_id', tokenize='porter unicode61');
"""

VECTOR_SCHEMA = f"""
DROP TABLE IF EXISTS chunk_vectors;
CREATE VIRTUAL TABLE chunk_vectors USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[{DIMS}] distance_metric=cosine
);
"""

VECTOR_BATCH = 128  # chunks per encode call; keeps peak memory flat
PROGRESS_CHUNKS = 10_000  # one line every N chunks so a long embed is observable


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


def cjk_split(text: str) -> str:
    """Space-separate CJK characters into the `cjk` column.

    unicode61 keeps a whole CJK run as one token, so 动力电池储能 would never
    match a query for 储能. A spaced copy makes any substring a phrase match:
    query `cjk:"<cjk_split(term)>"`, e.g. cjk:"储 能". Latin words and numbers
    are already in `text`, so only the CJK characters are copied here.
    """
    return " ".join(CJK.findall(text))


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
    lineno = 0  # numbered across sheets so a chunk spanning two keeps a sane range
    for sheet in wb.sheetnames:
        for row in wb[sheet].iter_rows(values_only=True):
            lineno += 1
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
        # clamp to the unit's own span: a csv record index is not a line number
        newlines = min(piece.count("\n"), unit.line_end - line)
        yield Unit(unit.heading_path, line, line + newlines, piece.strip())
        line += newlines
        pos = end


def chunks(units: list[Unit]) -> list[Chunk]:
    """Pack units into 512-800 token chunks, flushing at headings once full."""
    groups: list[list[Unit]] = []
    group: list[Unit] = []
    size = 0
    for unit in [piece for unit in units for piece in _fit(unit)]:
        if not any(c.isalnum() for c in unit.text):  # "---" rules, page separators
            continue
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

    return [
        Chunk(max(g, key=lambda u: n_tokens(u.text)).heading_path,
              g[0].line_start, g[-1].line_end, "\n\n".join(u.text for u in g))
        for g in _fold_strays(groups)
    ]


def _fold_strays(groups: list[list[Unit]]) -> list[list[Unit]]:
    """Fold footer-sized chunks into a neighbour so page numbers get no chunk of their own."""
    def size(group: list[Unit]) -> int:
        return sum(n_tokens(u.text) + 1 for u in group)

    out: list[list[Unit]] = []
    for i, group in enumerate(groups):
        if size(group) < STRAY_TOK:
            if i + 1 < len(groups) and size(group) + size(groups[i + 1]) <= MAX_TOK:
                groups[i + 1] = group + groups[i + 1]
                continue
            if out and size(out[-1]) + size(group) <= MAX_TOK:
                out[-1].extend(group)
                continue
        out.append(group)
    return out


def _slug(domain: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")


def load_vec(con: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension on this connection, once."""
    try:
        con.execute("SELECT vec_version()").fetchone()
        return
    except sqlite3.OperationalError:
        pass
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError(
            "sqlite-vec is not installed; pip install sqlite-vec") from exc
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)


def _write_vectors(con: sqlite3.Connection, embedder: Embedder,
                   pending: list[tuple[int, str, str]]) -> None:
    """Embed one batch and insert it. Raises before a poisoned row is written."""
    if int(getattr(embedder, "dims", 0)) != DIMS:
        raise ValueError(f"embedder dims {getattr(embedder, 'dims', None)}, expected {DIMS}")
    texts = [text for _, _, text in pending]
    titles = [title for _, title, _ in pending]
    vectors = check_vectors(embedder.encode_documents(texts, titles), len(pending))
    con.executemany(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        [(chunk_id, vector.tobytes())
         for (chunk_id, _, _), vector in zip(pending, vectors)])


def build_domain(domain: str, docs_dir: Path = DOCS_DIR,
                 index_dir: Path = INDEX_DIR,
                 embedder: Embedder | None = None
                 ) -> tuple[dict[str, int], int, Path]:
    """(re)build one domain's index from scratch; return (file counts, chunks, db).

    With ``embedder``, also write 256d document vectors to the ``chunk_vectors``
    sqlite-vec table. The vector batch is finite-checked before insertion; a
    NaN or Inf aborts the build and leaves the previous database untouched.
    """
    root = Path(docs_dir) / domain
    if not root.is_dir():
        raise SystemExit(f"no domain directory: {root}")

    db_path = Path(index_dir) / f"{_slug(domain)}.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)  # a killed build must not look like a good db

    counts = {"md": 0, "xlsx": 0, "csv": 0}
    total = 0
    next_id = 1
    reported = 0
    started_domain = time.perf_counter()
    pending: list[tuple[int, str, str]] = []
    con = sqlite3.connect(tmp_path)
    try:
        con.executescript(SCHEMA)
        if embedder is not None:
            load_vec(con)
            con.executescript(VECTOR_SCHEMA)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            kind = SUFFIXES.get(path.suffix.lower())
            if kind is None:
                continue
            doc_path = f"documents/{domain}/{path.relative_to(root).as_posix()}"
            rows = []
            for chunk in chunks(list(_units(path))):
                rows.append((next_id, domain, doc_path, chunk.line_start,
                             chunk.line_end, chunk.heading_path, chunk.text,
                             cjk_split(chunk.text)))
                if embedder is not None:
                    pending.append((next_id, chunk.heading_path, chunk.text))
                next_id += 1
            con.executemany(
                "INSERT INTO chunks (chunk_id, domain, doc_path, line_start,"
                " line_end, heading_path, text, cjk)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            counts[kind] += 1
            total += len(rows)
            if embedder is not None and len(pending) >= VECTOR_BATCH:
                _write_vectors(con, embedder, pending)
                pending.clear()
            if embedder is not None and total // PROGRESS_CHUNKS > reported:
                reported = total // PROGRESS_CHUNKS
                print(f"{domain}: {total} chunks in "
                      f"{time.perf_counter() - started_domain:.0f}s", flush=True)
        if pending:
            _write_vectors(con, embedder, pending)
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        con.commit()
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
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
    ap.add_argument("--embed", action="store_true",
                    help="also write 256d EmbeddingGemma 2 vectors (sqlite-vec)")
    args = ap.parse_args(argv)

    if not args.docs_dir.is_dir():
        ap.error(f"no documents dir: {args.docs_dir} (set AGORA_DATA or --docs-dir)")
    all_domains = sorted(p.name for p in args.docs_dir.iterdir() if p.is_dir())
    domains = args.domain or all_domains
    for domain in domains:
        if domain not in all_domains:
            ap.error(f"unknown domain {domain!r}; choose from: {', '.join(all_domains)}")

    embedder = EmbeddingGemma() if args.embed else None
    if embedder is not None:
        print(embedder.describe(), flush=True)

    started_all = time.perf_counter()
    totals = {"md": 0, "xlsx": 0, "csv": 0}
    total_chunks = 0
    for domain in domains:
        started = time.perf_counter()
        counts, n_chunks, db_path = build_domain(
            domain, args.docs_dir, args.index_dir, embedder)
        elapsed = time.perf_counter() - started
        print(f"{domain}: {counts['md']} md + {counts['xlsx']} xlsx + {counts['csv']} csv"
              f" -> {n_chunks} chunks in {elapsed:.1f}s -> {db_path}", flush=True)
        for kind in totals:
            totals[kind] += counts[kind]
        total_chunks += n_chunks
    if len(domains) > 1:
        elapsed = time.perf_counter() - started_all
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"total: {totals['md']} md + {totals['xlsx']} xlsx + {totals['csv']} csv"
              f" -> {total_chunks} chunks in {len(domains)} domains,"
              f" {elapsed:.1f}s, peak RSS {peak_mb:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
