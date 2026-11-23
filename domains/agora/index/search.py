"""The one BM25 query path: FTS5 MATCH expression and top-k chunk hits.

Both Gate-2 evaluation (evaluate.py) and the agent's ``search_docs`` tool call
this, so the retrieval the gate measures is the retrieval the harness uses.
Not evolvable, like the rest of index/: rewording the query would move the
gate without moving the agent.

``k`` is a parameter because the tool picks its own depth; Gate 2 uses 10.
"""

from __future__ import annotations

import re
import sqlite3

import build  # same directory: cjk_split

K = 10
WORD = re.compile(r"[A-Za-z0-9]+")
CJK_RUN = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")
STOP = frozenset(
    "a an and are as at be been by can does do for from had has have how if in into "
    "is it its may not of on or that the their then there these this to was were "
    "what when where which who whose why will with".split())


def match_expression(text: str) -> str:
    """FTS5 MATCH expression: every query term OR-ed, each quoted.

    Stopwords are dropped because Agora questions are long prose; keeping the
    glue turns the ranking into noise. ``bm25`` ranks the OR matches.
    """
    words = [w for w in WORD.findall(text.casefold()) if w not in STOP]
    parts = [f'"{w}"' for w in dict.fromkeys(words)]
    parts += [f'cjk:"{build.cjk_split(run)}"' for run in CJK_RUN.findall(text)]
    return " OR ".join(parts)


def search(con: sqlite3.Connection, query: str, k: int = K) -> list[tuple[str, int, int]]:
    """``[(doc_path, line_start, line_end)]``, best first."""
    expression = match_expression(query)
    if not expression:
        return []
    rows = con.execute(
        "SELECT c.doc_path, c.line_start, c.line_end"
        " FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid"
        " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (expression, k),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]
