"""Cross-encoder reranking through a vLLM ``/score`` service.

The 0.6B reranker is served as a separate GPU process (PLAN.md §5, Appendix A);
this module is only the client. It stays stdlib-only so the sandbox can import
it without a model, a tokenizer or a CUDA runtime.

``/score`` returns one relevance score per (query, document) pair. The score is
used for ordering only: the model's calibration (sigmoid or raw logit) differs
between checkpoints and does not matter to the top-k cut.

Requests are batched at 32 pairs, as issue #15 asks. The endpoint is
``AGORA_RERANK_URL`` (default ``http://127.0.0.1:8001/score``); the sandbox
reaches the host service through ``--network host``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

DEFAULT_URL = "http://127.0.0.1:8001/score"
DEFAULT_MODEL = "Qwen3-Reranker-0.6B"
BATCH_SIZE = 32
TIMEOUT_S = 120.0


class Reranker(Protocol):
    """The surface ``search.py`` needs: one score per document, in order."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class VllmReranker:
    """``/score`` client with bounded batches and deterministic ordering."""

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 batch_size: int = BATCH_SIZE, timeout: float = TIMEOUT_S):
        self.url = url
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)

    def describe(self) -> str:
        return f"{self.model} via {self.url} (batch {self.batch_size})"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            scores.extend(self._score_batch(
                query, list(documents[start:start + self.batch_size])))
        if len(scores) != len(documents):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(documents)} "
                f"documents")
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        body = json.dumps({
            "model": self.model,
            "queries": query,
            "documents": documents,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"reranker request failed at {self.url}: {exc}. "
                "Start it with scripts/serve_rerank.sh, or set RERANK=False."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"reranker returned non-JSON from {self.url}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"reranker returned {type(payload).__name__}, not an object")
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(documents):
            raise RuntimeError(
                f"reranker returned {type(rows).__name__} "
                f"{len(rows) if isinstance(rows, list) else '?'} rows for "
                f"{len(documents)} documents")
        ordered = sorted(rows, key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in ordered]
        if indices != list(range(len(documents))):
            raise RuntimeError(f"reranker returned indices {indices}")
        return [float(row["score"]) for row in ordered]


def from_env() -> VllmReranker:
    """The configured client. No network I/O happens here."""
    return VllmReranker(
        url=os.environ.get("AGORA_RERANK_URL") or DEFAULT_URL,
        model=os.environ.get("AGORA_RERANK_MODEL") or DEFAULT_MODEL,
    )
