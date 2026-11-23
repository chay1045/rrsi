"""Issue #03: markdown/table chunker + SQLite FTS5 builder.

    python3 -m pytest -q tests/test_index.py
"""
import sqlite3
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"

spec = spec_from_file_location("agora_index_build", AGORA / "index" / "build.py")
build = module_from_spec(spec)
sys.modules[spec.name] = build  # dataclasses looks its own module up here
spec.loader.exec_module(build)


def test_build_domain_chunks_and_fts(tmp_path):
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    para = "alpha beta gamma delta " * 60  # ~360 estimated tokens
    (root / "a.md").write_text(
        "# Intro\n\n" + "\n\n".join([para] * 12) + "\n\n## Tail\n\nzebra 12345.678\n",
        encoding="utf-8")
    (root / "t.csv").write_text("year,value\n2024,12345.678\n", encoding="utf-8")

    counts, n_chunks, db = build.build_domain("Test Domain", docs, tmp_path / "index")

    assert counts == {"md": 1, "xlsx": 0, "csv": 1}
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT doc_path, line_start, line_end, heading_path, text FROM chunks").fetchall()

    md = [r for r in rows if r[0].endswith("/a.md")]
    assert md and all(build.MIN_TOK <= build.n_tokens(r[4]) <= build.MAX_TOK for r in md)
    assert md[0][1] == 1 and md[0][3].startswith("a > Intro")
    assert md[0][2] < md[-1][2]  # line ranges advance through the file

    hit = con.execute(
        "SELECT c.doc_path, c.heading_path FROM chunks_fts f JOIN chunks c"
        " ON c.chunk_id = f.rowid WHERE chunks_fts MATCH 'zebra'").fetchall()
    assert len(hit) == 1 and hit[0][0].endswith("/a.md")

    # table rows are flattened as "col: value" and searchable by number
    csv_rows = [r for r in rows if r[0].endswith("/t.csv")]
    assert len(csv_rows) == 1 and csv_rows[0][1:3] == (1, 2)
    assert "B: 12345.678" in csv_rows[0][4]


def test_oversized_paragraph_is_split_then_packed():
    units = list(build._md_units("# Big\n\n" + "word " * 4000, "big"))
    ch = build.chunks(units)
    # per-unit estimates are summed, the joined text re-rounds: allow a few tokens
    assert max(build.n_tokens(c.text) for c in ch) <= build.MAX_TOK + 8
    assert sum(c.text.count("word") for c in ch) >= 4000  # overlap may repeat, nothing lost
