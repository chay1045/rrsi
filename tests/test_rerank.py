"""Issue #15 and #32: the /score client, batching, the pin and the fallback."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGORA = ROOT / "domains" / "agora"
sys.path.insert(0, str(AGORA))

import cli  # noqa: E402
import harness.tools as tools  # noqa: E402
import rerank  # noqa: E402  (tools.py put index/ on sys.path)
import search  # noqa: E402


class ScoreHandler(BaseHTTPRequestHandler):
    """Echo one score per document; record every request body."""

    calls: list[dict] = []
    bad = False

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        ScoreHandler.calls.append(body)
        documents = body["documents"]
        if ScoreHandler.bad:
            payload = {"data": []}
        else:
            payload = {"data": [
                {"index": index, "score": float(document.split()[-1])}
                for index, document in enumerate(documents)]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_retrieval_state():
    """Each test starts with a closed circuit and empty counters."""
    rerank.reset_breaker()
    search.reset_retrieval_stats()
    yield
    rerank.reset_breaker()
    search.reset_retrieval_stats()


@pytest.fixture
def score_server():
    ScoreHandler.calls = []
    ScoreHandler.bad = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/score"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_score_batches_at_32_and_keeps_input_order(score_server):
    client = rerank.VllmReranker(url=score_server)
    documents = [f"document {index}" for index in range(50)]
    scores = client.score("the query", documents)

    assert scores == [float(index) for index in range(50)]
    assert [len(call["documents"]) for call in ScoreHandler.calls] == [32, 18]
    assert all(call["queries"] == "the query" for call in ScoreHandler.calls)
    assert all(call["model"] == rerank.DEFAULT_MODEL for call in ScoreHandler.calls)


def test_score_rejects_a_short_response(score_server):
    ScoreHandler.bad = True
    with pytest.raises(RuntimeError, match="rows for 2 documents"):
        rerank.VllmReranker(url=score_server).score("q", ["a", "b"])


def test_missing_service_names_the_fallback():
    client = rerank.VllmReranker(url="http://127.0.0.1:1/score", timeout=0.2)
    with pytest.raises(RuntimeError, match="RRF"):
        client.score("q", ["a", "b"])


def test_from_env_reads_the_endpoint(monkeypatch):
    monkeypatch.setenv("AGORA_RERANK_URL", "http://example.invalid/score")
    monkeypatch.setenv("AGORA_RERANK_MODEL", "other-reranker")
    client = rerank.from_env()
    assert client.url == "http://example.invalid/score"
    assert client.model == "other-reranker"


def test_transient_failure_is_retried_once(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(
                {"data": [{"index": 0, "score": 0.5}]}).encode()

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        return Response()

    monkeypatch.setattr(rerank.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(rerank, "RETRY_DELAY_S", 0.0)

    assert rerank.VllmReranker(url="http://x/score").score("q", ["a"]) == [0.5]
    assert len(calls) == 2


def test_dead_service_opens_the_breaker(monkeypatch):
    calls = []

    def dead(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(rerank.urllib.request, "urlopen", dead)
    monkeypatch.setattr(rerank, "RETRY_DELAY_S", 0.0)
    client = rerank.VllmReranker(url="http://x/score")

    with pytest.raises(RuntimeError, match="RRF"):
        client.score("q", ["a"])
    assert len(calls) == rerank.RETRIES + 1

    # The breaker rejects the next search without a network call.
    with pytest.raises(rerank.RerankUnavailable, match="circuit open"):
        client.score("q", ["a"])
    assert len(calls) == rerank.RETRIES + 1


def test_search_uses_the_pinned_reranker(monkeypatch):
    sentinel = object()
    seen: list[object] = []

    def fake_hybrid(con, query, vector, k, depth, **kwargs):
        seen.append(kwargs["reranker"])
        return []

    monkeypatch.setattr(search, "hybrid_ranked", fake_hybrid)
    monkeypatch.setattr(rerank, "from_env", lambda: sentinel)

    monkeypatch.setenv("AGORA_RERANK", "1")
    assert search.hybrid_search_ranked(
        None, "q", 5, encoder=lambda query: [0.0]) == []
    assert seen[-1] is sentinel

    monkeypatch.setenv("AGORA_RERANK", "0")
    assert search.hybrid_search_ranked(
        None, "q", 5, encoder=lambda query: [0.0]) == []
    assert seen[-1] is None


def _context(tmp_path: Path) -> tools.Context:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    con = sqlite3.connect(index_dir / "test-domain.sqlite")
    con.execute(
        "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_path TEXT,"
        " line_start INTEGER, line_end INTEGER, heading_path TEXT, text TEXT)")
    con.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [(1, "doc1.md", 1, 1, "Heading 1", "snippet 1"),
         (2, "doc2.md", 2, 2, "Heading 2", "snippet 2")])
    con.commit()
    con.close()
    return tools.Context(domain="Test Domain", data_dir=tmp_path,
                         index_dir=index_dir)


def test_dead_endpoint_returns_rrf_hits_and_the_episode_completes(
        tmp_path, monkeypatch, capsys):
    ctx = _context(tmp_path)
    hits = [(1, "doc1.md", 1, 1), (2, "doc2.md", 2, 2)]

    monkeypatch.setattr(search, "bm25_chunks", lambda con, query, k: hits)
    monkeypatch.setattr(search, "dense_chunks", lambda con, vector, k: hits)
    monkeypatch.setattr(search, "_chunk_texts",
                        lambda con, ids: [f"text {index}" for index in ids])
    monkeypatch.setattr(search, "_query_encoder",
                        lambda: (lambda query: [0.0]))
    monkeypatch.setenv("AGORA_RERANK", "1")
    monkeypatch.setenv("AGORA_RERANK_URL", "http://127.0.0.1:1/score")

    task = cli.tasks.Task("Test Domain", "q_0001", "what?", "done", [])
    replies = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "search_docs",
                         "arguments": json.dumps({"query": "q"})}}]},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c2", "type": "function",
            "function": {"name": "answer",
                         "arguments": json.dumps({"value": "done"})}}]},
    ]

    def client(messages, specs, force=None):
        return dict(replies.pop(0))

    trajectory = cli.run_task(task, tmp_path / "out", client=client, ctx=ctx)

    assert trajectory["answered"] is True
    assert trajectory["retrieval"]["rerank_calls"] == 1
    assert trajectory["retrieval"]["rerank_fallbacks"] == 1
    assert trajectory["retrieval"]["rerank_fallback"] is True
    result = next(message["content"] for message in trajectory["messages"]
                  if message.get("role") == "tool")
    assert result.startswith("[1] ")
    assert "rerank fallback" in capsys.readouterr().err
