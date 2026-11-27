"""The one retrieval path: BM25, dense KNN and RRF fusion over chunk ids.

Gate-2 evaluation (``evaluate.py``) and the agent's ``search_docs`` tool call
this module, so the retrieval the gate measures is the retrieval the harness
uses. Not evolvable, like the rest of index/: rewording the query would move
the gate without moving the agent.

The BM25 side needs no model. The dense side embeds the query with
EmbeddingGemma 2 (lazy import) and reads the 768d ``chunk_vectors`` table.
``k`` is a parameter because the tool picks its own depth; Gate 2 uses 10 by
default and can match the tool's depth with ``--k``.

An optional cross-encoder reranker takes the RRF top-50 down to ``k``. The
reranker is a separate GPU service (``rerank.py``), so this module stays
importable in the CPU sandbox and the fusion path is unchanged when it is off.
The tool-facing ``hybrid_search_ranked`` resolves the pinned reranker and, when
the service fails, returns the RRF order and records the fallback. Gate 2 calls
``hybrid_ranked`` directly and stays strict, so a dead service cannot make a
``hybrid+rerank`` report look like a clean reranked run.

The final result set is not just the top ``k``: at most ``PER_DOCUMENT`` chunks
come from one document, and a best rerank score below ``MIN_RERANK_SCORE``
returns nothing. The tool and Gate 2 share these rules, so the score is also
kept for the tool's confidence label.
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
PER_DOCUMENT = 2  # issue #31: at most two chunks per file, so the default five
                  # hits can cover three files when the corpus allows it
MIN_RERANK_SCORE = 0.005  # below this best score, return no hits. The score is
                          # a pair probability, not a calibrated "can answer"
                          # signal, so the cut only rejects the noise floor.
HIGH_RERANK_SCORE = 0.5   # high/low label boundary for the tool output
WORD = re.compile(r"[A-Za-z0-9]+")
CJK_RUN = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")
STOP = frozenset(
    "a an and are as at be been by can does do for from had has have how if in into "
    "is it its may not of on or that the their then there these this to was were "
    "what when where which who whose why will with".split())

Hit = tuple[int, str, int, int]  # chunk_id, doc_path, line_start, line_end
Ranked = tuple[Hit, float | None]  # hit and rerank score; None when reranking is off

MAX_ERROR_CHARS = 300
RETRIEVAL_STATS: dict[str, object] = {
    "rerank_enabled": rerank.enabled(),
    "rerank_calls": 0,
    "rerank_fallbacks": 0,
    "rerank_fallback": False,
    "last_rerank_error": None,
}


def reset_retrieval_stats() -> None:
    """Start an episode with a closed circuit and empty counters."""
    RETRIEVAL_STATS.update({
        "rerank_enabled": rerank.enabled(),
        "rerank_calls": 0,
        "rerank_fallbacks": 0,
        "rerank_fallback": False,
        "last_rerank_error": None,
    })
    rerank.reset_breaker()


def retrieval_stats() -> dict:
    """A copy of the counters for the trajectory and the run log."""
    return dict(RETRIEVAL_STATS)


def _record_fallback(exc: Exception) -> None:
    message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
    RETRIEVAL_STATS["rerank_fallbacks"] = int(
        RETRIEVAL_STATS["rerank_fallbacks"]) + 1
    RETRIEVAL_STATS["rerank_fallback"] = True
    RETRIEVAL_STATS["last_rerank_error"] = message


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


def confidence(score: float | None) -> str | None:
    """Coarse label for the tool; ``None`` when reranking is off."""
    if score is None:
        return None
    return "high" if score >= HIGH_RERANK_SCORE else "low"


def _diversify(ranked: Sequence[Ranked], k: int,
               per_document: int | None) -> list[Ranked]:
    """Keep the best ``k`` hits with at most ``per_document`` per file.

    ``per_document`` is ``None`` to disable the cap. The input is already in
    rank order, so the first kept chunk of a file is its best one.
    """
    if per_document is None:
        return list(ranked[:k])
    if per_document < 1:
        raise ValueError(f"per_document must be positive or None, got {per_document}")
    kept: list[Ranked] = []
    counts: dict[str, int] = {}
    for hit, score in ranked:
        doc_path = hit[1]
        if counts.get(doc_path, 0) >= per_document:
            continue
        counts[doc_path] = counts.get(doc_path, 0) + 1
        kept.append((hit, score))
        if len(kept) == k:
            break
    return kept


def hybrid_ranked(con: sqlite3.Connection, query: str,
                  vector: Sequence[float], k: int = K, depth: int = DEPTH,
                  reranker: rerank.Reranker | None = None,
                  per_document: int | None = PER_DOCUMENT,
                  min_score: float | None = MIN_RERANK_SCORE,
                  fallback_on_error: bool = False) -> list[Ranked]:
    """BM25 top-``depth`` + dense top-``depth`` -> RRF -> ranked top-``k``.

    The RRF top-``RERANK_DEPTH`` is the candidate pool. With a ``reranker``,
    the cross-encoder rescores the pool, the best ``k`` in score order come
    back, and ties keep the RRF order. A best score below ``min_score`` returns
    no hits. ``per_document`` caps near-duplicate chunks from one file.

    Without a reranker, every score is ``None`` and ``min_score`` does not
    apply; the RRF order is the ranking. With ``fallback_on_error``, a failed
    reranker also returns the RRF order and records the fallback. The tool sets
    it; Gate 2 leaves it false so a dead service fails loudly.
    """
    fused = reciprocal_rank_fusion(
        [bm25_chunks(con, query, depth), dense_chunks(con, vector, depth)],
        RERANK_DEPTH)
    if not fused:
        return []
    if reranker is None:
        ranked: list[Ranked] = [(hit, None) for hit in fused]
    else:
        try:
            scores = reranker.score(
                query, _chunk_texts(con, [hit[0] for hit in fused]))
        except Exception as exc:
            if not fallback_on_error:
                raise
            _record_fallback(exc)
            return _diversify([(hit, None) for hit in fused], k, per_document)
        ranked = sorted(zip(fused, scores), key=lambda item: -item[1])
        if min_score is not None and ranked[0][1] < min_score:
            return []
    return _diversify(ranked, k, per_document)


def hybrid_chunks(con: sqlite3.Connection, query: str, vector: Sequence[float],
                  k: int = K, depth: int = DEPTH,
                  reranker: rerank.Reranker | None = None,
                  per_document: int | None = PER_DOCUMENT,
                  min_score: float | None = MIN_RERANK_SCORE) -> list[Hit]:
    """Hit-only form of ``hybrid_ranked`` for callers that do not need scores."""
    return [hit for hit, _ in hybrid_ranked(
        con, query, vector, k, depth, reranker, per_document, min_score)]


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
                  reranker: rerank.Reranker | None = None,
                  per_document: int | None = PER_DOCUMENT,
                  min_score: float | None = MIN_RERANK_SCORE
                  ) -> list[tuple[str, int, int]]:
    """Hit-only path that resolves the pin and uses the RRF fallback."""
    return [(doc_path, line_start, line_end)
            for doc_path, line_start, line_end, _
            in hybrid_search_ranked(con, query, k, depth=depth, encoder=encoder,
                                    reranker=reranker, per_document=per_document,
                                    min_score=min_score)]


def hybrid_search_ranked(con: sqlite3.Connection, query: str, k: int = K, *,
                         depth: int = DEPTH,
                         encoder: Callable[[str], Sequence[float]] | None = None,
                         reranker: rerank.Reranker | None = None,
                         per_document: int | None = PER_DOCUMENT,
                         min_score: float | None = MIN_RERANK_SCORE
                         ) -> list[tuple[str, int, int, float | None]]:
    """Hybrid path with the rerank score kept for the tool's confidence label.

    The pinned switch is resolved here, not in the harness. A candidate that
    passes ``reranker=None`` therefore cannot disable it. A failed reranker
    returns the RRF order and increments the fallback counters.
    """
    if reranker is None:
        reranker = rerank.pinned_reranker()
    if reranker is not None:
        RETRIEVAL_STATS["rerank_calls"] = int(
            RETRIEVAL_STATS["rerank_calls"]) + 1
    vector = (encoder or _query_encoder())(query)
    return [(hit[1], hit[2], hit[3], score)
            for hit, score
            in hybrid_ranked(con, query, vector, k, depth, reranker=reranker,
                             per_document=per_document, min_score=min_score,
                             fallback_on_error=True)]


def search(con: sqlite3.Connection, query: str, k: int = K) -> list[tuple[str, int, int]]:
    """BM25-only ``[(doc_path, line_start, line_end)]``, best first.

    Kept as the ablation baseline and as the fallback for tests that do not
    install the embedder.
    """
    return [(doc_path, line_start, line_end)
            for _, doc_path, line_start, line_end in bm25_chunks(con, query, k)]
