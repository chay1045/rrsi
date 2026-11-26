"""The one retrieval path: BM25, dense KNN and RRF fusion over chunk ids.

Gate-2 evaluation (``evaluate.py``) and the agent's ``search_docs`` tool call
this module, so the retrieval the gate measures is the retrieval the harness
uses. Not evolvable, like the rest of index/: rewording the query would move
the gate without moving the agent.

The BM25 side needs no model. The dense side embeds the query with
EmbeddingGemma 2 (lazy import) and reads the 256d ``chunk_vectors`` table.
``k`` is a parameter because the tool picks its own depth; Gate 2 uses 10.

An optional cross-encoder reranker takes the RRF top-50 down to ``k``. The
reranker is a separate GPU service (``rerank.py``), so this module stays
importable in the CPU sandbox and the fusion path is unchanged when it is off.
"""

from __future__ import annotations

import re
import sqlite3
from array import array
from collections.abc import Callable, Sequence

import build  # same directory: cjk_split and the sqlite-vec loader
import rerank  # same directory: the /score client (stdlib only)

K = 10
DEPTH = 50       # candidates per ranker before fusion
RERANK_DEPTH = 50  # RRF candidates sent to the cross-encoder
RRF_K = 60       # the standard reciprocal-rank-fusion constant
WORD = re.compile(r"[A-Za-z0-9]+")
CJK_RUN = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")
STOP = frozenset(
    "a an and are as at be been by can does do for from had has have how if in into "
    "is it its may not of on or that the their then there these this to was were "
    "what when where which who whose why will with".split())

Hit = tuple[int, str, int, int]  # chunk_id, doc_path, line_start, line_end


def match_expression(text: str) -> str:
    """FTS5 MATCH expression: every query term OR-ed, each quoted.

    Stopwords are dropped because Agora questions are long prose; keeping the
    glue turns the ranking into noise. ``bm25`` ranks the OR matches.
    """
    words = [w for w in WORD.findall(text.casefold()) if w not in STOP]
    parts = [f'"{w}"' for w in dict.fromkeys(words)]
    parts += [f'cjk:"{build.cjk_split(run)}"' for run in CJK_RUN.findall(text)]
    return " OR ".join(parts)


def bm25_chunks(con: sqlite3.Connection, query: str, k: int = K) -> list[Hit]:
    """Top-k BM25 chunks with ids, best first."""
    expression = match_expression(query)
    if not expression:
        return []
    rows = con.execute(
        "SELECT c.chunk_id, c.doc_path, c.line_start, c.line_end"
        " FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid"
        " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (expression, k),
    ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def _chunk_hits(con: sqlite3.Connection, chunk_ids: Sequence[int]) -> list[Hit]:
    """Fetch chunk metadata in the given order, dropping missing ids."""
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT chunk_id, doc_path, line_start, line_end FROM chunks"
        f" WHERE chunk_id IN ({marks})", list(chunk_ids)).fetchall()
    found = {row[0]: row[1:] for row in rows}
    return [(chunk_id, *found[chunk_id])
            for chunk_id in chunk_ids if chunk_id in found]


def _chunk_texts(con: sqlite3.Connection, chunk_ids: Sequence[int]) -> list[str]:
    """The heading breadcrumb and text of every candidate, in the given order."""
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT chunk_id, heading_path, text FROM chunks"
        f" WHERE chunk_id IN ({marks})", list(chunk_ids)).fetchall()
    found = {row[0]: (row[1], row[2]) for row in rows}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in found]
    if missing:
        raise RuntimeError(f"chunks missing from the index: {missing[:3]}")
    return [f"{found[chunk_id][0]}\n{found[chunk_id][1]}"
            for chunk_id in chunk_ids]


def dense_chunks(con: sqlite3.Connection, vector: Sequence[float],
                 k: int = K) -> list[Hit]:
    """Top-k cosine neighbours with ids, best first."""
    from embed import DIMS

    if len(vector) != DIMS:
        raise ValueError(f"query vector has {len(vector)} dims, expected {DIMS}")
    build.load_vec(con)
    packed = array("f", (float(value) for value in vector)).tobytes()
    rows = con.execute(
        "SELECT chunk_id, distance FROM chunk_vectors"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (packed, k),
    ).fetchall()
    return _chunk_hits(con, [row[0] for row in rows])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = K,
                            rrf_k: int = RRF_K) -> list[Hit]:
    """Fuse ranked chunk lists with reciprocal rank fusion.

    score(d) = sum over lists of 1 / (rrf_k + rank), 1-based. Ties break by the
    best rank in any list, then by chunk id so the result is deterministic.
    """
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    hits: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            chunk_id = hit[0]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best[chunk_id] = min(best.get(chunk_id, rank), rank)
            hits.setdefault(chunk_id, hit)
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id],
                                                 best[chunk_id], chunk_id))
    return [hits[chunk_id] for chunk_id in order[:k]]


def _query_encoder() -> Callable[[str], Sequence[float]]:
    def encode(query: str):
        import embed
        return embed.encode_query(query)
    return encode


def hybrid_chunks(con: sqlite3.Connection, query: str, vector: Sequence[float],
                  k: int = K, depth: int = DEPTH,
                  reranker: rerank.Reranker | None = None) -> list[Hit]:
    """BM25 top-``depth`` + dense top-``depth`` -> RRF -> top-``k``.

    With a ``reranker``, the RRF order contributes the top ``RERANK_DEPTH``
    candidates, the cross-encoder rescores them and the best ``k`` come back.
    Ties keep the RRF order, so the off and on paths differ only by the score.
    """
    fused = reciprocal_rank_fusion(
        [bm25_chunks(con, query, depth), dense_chunks(con, vector, depth)],
        RERANK_DEPTH if reranker is not None else k)
    if reranker is None or not fused:
        return fused
    scores = reranker.score(query, _chunk_texts(con, [hit[0] for hit in fused]))
    order = sorted(range(len(fused)), key=lambda position: (-scores[position],
                                                            position))
    return [fused[position] for position in order[:k]]


def default_reranker() -> rerank.VllmReranker:
    """The client for the configured service; no network I/O happens here."""
    return rerank.from_env()


def dense_search(con: sqlite3.Connection, query: str, k: int = K, *,
                 encoder: Callable[[str], Sequence[float]] | None = None
                 ) -> list[tuple[str, int, int]]:
    """Dense-only path. The encoder defaults to the process-wide model."""
    vector = (encoder or _query_encoder())(query)
    return [(doc_path, line_start, line_end)
            for _, doc_path, line_start, line_end in dense_chunks(con, vector, k)]


def hybrid_search(con: sqlite3.Connection, query: str, k: int = K, *,
                  depth: int = DEPTH,
                  encoder: Callable[[str], Sequence[float]] | None = None,
                  reranker: rerank.Reranker | None = None
                  ) -> list[tuple[str, int, int]]:
    """Hybrid path used by both Gate 2 and the agent's search_docs."""
    vector = (encoder or _query_encoder())(query)
    return [(doc_path, line_start, line_end)
            for _, doc_path, line_start, line_end
            in hybrid_chunks(con, query, vector, k, depth, reranker)]


def search(con: sqlite3.Connection, query: str, k: int = K) -> list[tuple[str, int, int]]:
    """BM25-only ``[(doc_path, line_start, line_end)]``, best first.

    Kept as the ablation baseline and as the fallback for tests that do not
    install the embedder.
    """
    return [(doc_path, line_start, line_end)
            for _, doc_path, line_start, line_end in bm25_chunks(con, query, k)]
