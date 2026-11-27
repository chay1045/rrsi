"""Issue #03: markdown/table chunker + SQLite FTS5 builder.

    python3 -m pytest -q tests/test_index.py
"""
import os
import re
import sqlite3
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"
sys.path.insert(0, str(AGORA / "index"))

spec = spec_from_file_location("agora_index_build", AGORA / "index" / "build.py")
build = module_from_spec(spec)
sys.modules[spec.name] = build  # dataclasses looks its own module up here
spec.loader.exec_module(build)

import search  # noqa: E402  (index/ is on sys.path)


class WordTokenizer:
    """Deterministic tokenizer: one token per whitespace-separated word."""

    id = "word-test"

    @staticmethod
    def spans(text):
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


TOKENIZER = (Path(os.environ.get("AGORA_EMBEDDER") or
                  Path.home() / "Models" / "embeddinggemma2") / "tokenizer.json")


class FakeEmbedder:
    """Deterministic 768d vectors: dim 0 means 'zebra', dim 1 means 'alpha'."""

    dims = build.DIMS

    @staticmethod
    def vector(text: str) -> np.ndarray:
        vector = np.zeros(build.DIMS, dtype=np.float32)
        vector[0] = 1.0 if "zebra" in text else 0.0
        vector[1] = 1.0 if "alpha" in text else 0.0
        if not vector.any():
            vector[2] = 1.0
        norm = np.linalg.norm(vector)
        return vector / norm

    def encode_documents(self, texts, titles):
        return np.stack([self.vector(text) for text in texts])

    def encode_queries(self, texts):
        return np.stack([self.vector(text) for text in texts])


class NanEmbedder(FakeEmbedder):
    def encode_documents(self, texts, titles):
        return np.full((len(texts), build.DIMS), np.nan, dtype=np.float32)


def test_build_domain_chunks_and_fts(tmp_path):
    tokenizer = WordTokenizer()
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    para = "alpha beta gamma delta " * 60  # 240 word tokens
    (root / "a.md").write_text(
        "# Intro\n\n" + "\n\n".join([para] * 12) + "\n\n## Tail\n\nzebra 12345.678\n",
        encoding="utf-8")
    (root / "t.csv").write_text("year,value\n2024,12345.678\n", encoding="utf-8")
    (root / "c.md").write_text(
        "# 报告\n\n动力电池储能出货高增，同比+98.9%。\n\n---\n\n114\n",
        encoding="utf-8")

    counts, n_chunks, db, key = build.build_domain(
        "Test Domain", docs, tmp_path / "index", tokenizer=tokenizer)

    assert counts == {"md": 2, "xlsx": 0, "csv": 1}
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT doc_path, line_start, line_end, heading_path, text FROM chunks").fetchall()

    md = [r for r in rows if r[0].endswith("/a.md")]
    sizes = [build.n_tokens(r[4], tokenizer) for r in md]
    assert md and max(sizes) <= build.MAX_TOK
    assert max(sizes) >= build.MIN_TOK
    assert all(r[1] <= r[2] for r in md)
    assert md[0][1] == 1 and md[0][3].startswith("a > Intro")
    assert md[0][2] < md[-1][2]  # line ranges advance through the file
    assert con.execute(
        "SELECT value FROM meta WHERE key = 'build_key'").fetchone() == (key,)

    hit = con.execute(
        "SELECT c.doc_path, c.heading_path FROM chunks_fts f JOIN chunks c"
        " ON c.chunk_id = f.rowid WHERE chunks_fts MATCH 'zebra'").fetchall()
    assert len(hit) == 1 and hit[0][0].endswith("/a.md")

    # table rows are flattened as "col: value" and searchable by number
    csv_rows = [r for r in rows if r[0].endswith("/t.csv")]
    assert len(csv_rows) == 1 and csv_rows[0][1:3] == (1, 2)
    assert "B: 12345.678" in csv_rows[0][4]

    # a CJK term is findable mid-run, and separator/page-number text is not its
    # own chunk; no chunk is only punctuation
    cjk = con.execute(
        "SELECT c.text FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid"
        " WHERE chunks_fts MATCH ?", (f'cjk:"{build.cjk_split("储能")}"',)).fetchall()
    assert len(cjk) == 1 and "动力电池储能" in cjk[0][0]
    assert all(any(ch.isalnum() for ch in r[4]) for r in rows)
    assert not any(r[4].strip() in {"---", "114"} for r in rows)


def test_build_key_is_reproducible_and_tracks_the_corpus(tmp_path):
    tokenizer = WordTokenizer()
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nalpha alpha\n", encoding="utf-8")

    _, _, _, first = build.build_domain(
        "Test Domain", docs, tmp_path / "one", tokenizer=tokenizer)
    _, _, _, second = build.build_domain(
        "Test Domain", docs, tmp_path / "two", tokenizer=tokenizer)
    assert first == second

    (root / "a.md").write_text("# Alpha\n\nalpha alpha changed\n", encoding="utf-8")
    _, _, _, changed = build.build_domain(
        "Test Domain", docs, tmp_path / "three", tokenizer=tokenizer)
    assert changed != first


def test_oversized_paragraph_is_split_then_packed():
    tokenizer = WordTokenizer()
    units = list(build._md_units("# Big\n\n" + "word " * 4000, "big"))
    ch = build.chunks(units, tokenizer)
    assert max(build.n_tokens(c.text, tokenizer) for c in ch) <= build.MAX_TOK
    assert sum(c.text.count("word") for c in ch) >= 4000  # overlap may repeat, nothing lost


def test_size_split_has_overlap_and_valid_lines():
    tokenizer = WordTokenizer()
    unit = build.Unit("Doc", 1, 1,
                      "\n".join(["alpha beta gamma delta epsilon zeta eta theta"] * 120))
    pieces = list(build._fit(unit, tokenizer))

    assert len(pieces) > 1
    assert all(build.n_tokens(p.text, tokenizer) <= build.MAX_TOK for p in pieces)
    first_spans = tokenizer.spans(pieces[0].text)
    overlap = pieces[0].text[first_spans[-build.OVERLAP_TOK][0]:].strip()
    assert overlap and pieces[1].text.startswith(overlap)
    assert all(p.line_start <= p.line_end for p in pieces)
    assert pieces[1].line_start <= pieces[0].line_end  # the overlap keeps lines valid


def test_every_flush_starts_the_next_chunk_with_the_tail():
    tokenizer = WordTokenizer()
    units = [build.Unit(f"Doc > H{i}", i * 2 + 1, i * 2 + 2, "word " * 300)
             for i in range(4)]
    ch = build.chunks(units, tokenizer)

    assert len(ch) > 1
    tail = ch[0].text.split()[-build.OVERLAP_TOK:]
    assert tail and ch[1].text.split()[:len(tail)] == tail
    assert ch[1].line_start <= ch[0].line_end


def test_long_breadcrumb_keeps_the_deepest_section():
    capped = build.breadcrumb("Doc", "x" * 500, "Deepest")

    assert len(capped) == build.BREADCRUMB_CHARS
    assert capped.endswith("Deepest")
    assert build.breadcrumb("Doc", "Section") == "Doc > Section"


@pytest.mark.skipif(not TOKENIZER.is_file(), reason="embedder tokenizer not installed")
def test_dense_numeric_table_is_split_by_real_tokens():
    tokenizer = build.EmbeddingGemmaTokenizer()
    dense = " | ".join(f"{i}.{i * 7 % 100:02d}" for i in range(1, 351))

    # The old 4-chars estimate would have kept this whole.
    assert len(dense) // 4 <= build.MAX_TOK
    assert build.n_tokens(dense, tokenizer) > build.EMBED_MAX_TOKENS

    pieces = list(build._fit(build.Unit("Table", 1, 1, dense), tokenizer))
    assert len(pieces) > 1
    assert all(build.n_tokens(p.text, tokenizer) <= build.MAX_TOK for p in pieces)

    # A capped breadcrumb plus a full chunk still fits the window.
    title = build.breadcrumb("Doc", "x" * 500, "Table")
    ch = build.chunks([build.Unit(title, 1, 1, dense)], tokenizer)
    assert all(build.n_tokens(build.document_text(c.heading_path, c.text),
                              tokenizer) <= build.EMBED_MAX_TOKENS
               for c in ch)


def test_build_rejects_a_chunk_over_the_embedder_window(tmp_path, monkeypatch):
    tokenizer = WordTokenizer()
    monkeypatch.setattr(build, "MAX_TOK", 2000)  # let the packer keep a wide chunk
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    (root / "a.md").write_text("# Big\n\n" + "word " * 1500, encoding="utf-8")

    with pytest.raises(ValueError, match="embedder window"):
        build.build_domain("Test Domain", docs, tmp_path / "index",
                           tokenizer=tokenizer)
    assert not (tmp_path / "index" / "test-domain.sqlite").exists()


def test_embedded_domain_tables_and_knn(tmp_path):
    pytest.importorskip("sqlite_vec")
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nalpha alpha\n", encoding="utf-8")
    (root / "b.md").write_text("# Zebra\n\nzebra zebra\n", encoding="utf-8")

    counts, n_chunks, db, _ = build.build_domain(
        "Test Domain", docs, tmp_path / "index", FakeEmbedder(),
        tokenizer=WordTokenizer())

    assert counts == {"md": 2, "xlsx": 0, "csv": 0} and n_chunks == 2
    con = sqlite3.connect(db)
    try:
        # The FTS5 path still works without the extension; only vec0 needs it.
        assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
        build.load_vec(con)
        assert con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 2
        hits = search.dense_chunks(con, FakeEmbedder.vector("zebra"), k=2)
        assert hits and hits[0][1].endswith("/b.md")
        fused = search.reciprocal_rank_fusion(
            [search.bm25_chunks(con, "zebra", 2), hits], k=2)
        assert fused and fused[0][1].endswith("/b.md")

        # RRF alone ties, and chunk id puts a.md first; a cross-encoder that
        # prefers zebra must reverse that and still cap at k.
        class PreferZebra:
            def __init__(self):
                self.documents = []

            def score(self, query, documents):
                self.documents.append(list(documents))
                return [1.0 if "zebra" in text else 0.0 for text in documents]

        plain = search.hybrid_chunks(
            con, "alpha", FakeEmbedder.vector("zebra"), k=2)
        assert plain[0][1].endswith("/a.md")

        cross = PreferZebra()
        reranked = search.hybrid_chunks(
            con, "alpha", FakeEmbedder.vector("zebra"), k=1, reranker=cross)
        assert len(reranked) == 1 and reranked[0][1].endswith("/b.md")
        assert any("zebra" in text for text in cross.documents[0])

        class Tied:
            def score(self, query, documents):
                return [1.0] * len(documents)

        assert search.hybrid_chunks(
            con, "alpha", FakeEmbedder.vector("zebra"), k=2, reranker=Tied()
        ) == plain
    finally:
        con.close()


def test_nan_embedding_aborts_build_and_leaves_no_database(tmp_path):
    pytest.importorskip("sqlite_vec")
    docs = tmp_path / "docs"
    root = docs / "Test Domain"
    root.mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nalpha alpha\n", encoding="utf-8")
    index_dir = tmp_path / "index"

    # Deliberate trigger of the NaN landmine from PLAN §5.1: the build must
    # fail before the vector reaches sqlite-vec, and must not leave a db behind.
    with pytest.raises(ValueError, match="non-finite"):
        build.build_domain("Test Domain", docs, index_dir, NanEmbedder(),
                           tokenizer=WordTokenizer())
    assert not (index_dir / "test-domain.sqlite").exists()
    assert not (index_dir / "test-domain.sqlite.tmp").exists()
