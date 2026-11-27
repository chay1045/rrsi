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

The reranker is a quality-only stage. A dead or slow service must not stop
retrieval (issue #32), so a request has a short timeout, one retry and a
process-wide circuit breaker. ``search.py`` turns a failure into the RRF order;
this module only makes the failure bounded and visible.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

DEFAULT_URL = "http://127.0.0.1:8001/score"
DEFAULT_MODEL = "Qwen3-Reranker-0.6B"
BATCH_SIZE = 32
TIMEOUT_S = 5.0          # one /score request; the old 120 s could hang an episode
RETRIES = 1              # one retry for a transient network error
RETRY_DELAY_S = 0.25
BREAKER_COOLDOWN_S = 30.0
MAX_ERROR_CHARS = 300
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class Reranker(Protocol):
    """The surface ``search.py`` needs: one score per document, in order."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class RerankUnavailable(RuntimeError):
    """The circuit is open, so this search must use the RRF order."""


class _Breaker:
    """A bounded fuse for one reranker endpoint.

    A failed search costs at most ``RETRIES + 1`` requests. The breaker then
    rejects calls for ``cooldown_s`` and allows one probe when the cooldown
    expires. It stores one short error, not a growing history, so the state is
    bounded for an episode.
    """

    def __init__(self, cooldown_s: float = BREAKER_COOLDOWN_S):
        self.cooldown_s = float(cooldown_s)
        self._open_until = 0.0
        self._last_error = ""
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            remaining = self._open_until - time.monotonic()
            if remaining > 0:
                raise RerankUnavailable(
                    f"reranker circuit open for {remaining:.1f}s after: "
                    f"{self._last_error}")

    def failure(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        with self._lock:
            self._last_error = message
            self._open_until = time.monotonic() + self.cooldown_s

    def success(self) -> None:
        with self._lock:
            self._open_until = 0.0
            self._last_error = ""

    def reset(self) -> None:
        self.success()


_BREAKER = _Breaker()


def reset_breaker() -> None:
    """Start a new episode or test with a closed circuit."""
    _BREAKER.reset()


def enabled() -> bool:
    """The pinned per-run switch. Unset means on for direct host use."""
    raw = os.environ.get("AGORA_RERANK")
    return True if raw is None else raw.strip().lower() not in FALSE_VALUES


def _transient(exc: Exception) -> bool:
    """A network error that one retry can fix."""
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code < 600
    return isinstance(exc, (urllib.error.URLError, TimeoutError))


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
        _BREAKER.check()
        scores: list[float] = []
        try:
            for start in range(0, len(documents), self.batch_size):
                scores.extend(self._score_batch(
                    query, list(documents[start:start + self.batch_size])))
            if len(scores) != len(documents):
                raise RuntimeError(
                    f"reranker returned {len(scores)} scores for "
                    f"{len(documents)} documents")
        except Exception as exc:
            _BREAKER.failure(exc)
            raise
        _BREAKER.success()
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        body = json.dumps({
            "model": self.model,
            "queries": query,
            "documents": documents,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(RETRIES + 1):
            try:
                with urllib.request.urlopen(
                        request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= RETRIES or not _transient(exc):
                    raise RuntimeError(
                        f"reranker request failed at {self.url}: {exc}. "
                        "Search continues with the RRF order.") from exc
                time.sleep(RETRY_DELAY_S)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"reranker returned non-JSON from {self.url}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"reranker returned {type(payload).__name__}, not an object")
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


def pinned_reranker() -> VllmReranker | None:
    """The configured client when the run pins reranking on, else ``None``."""
    return from_env() if enabled() else None
