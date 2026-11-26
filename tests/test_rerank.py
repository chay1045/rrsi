"""Issue #15: the /score client, batching and the harness off switch."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGORA = ROOT / "domains" / "agora"
sys.path.insert(0, str(AGORA))

import harness.policy as policy  # noqa: E402
import harness.tools as tools  # noqa: E402
import rerank  # noqa: E402  (tools.py put index/ on sys.path)


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


def test_missing_service_names_the_switch():
    client = rerank.VllmReranker(url="http://127.0.0.1:1/score", timeout=0.2)
    with pytest.raises(RuntimeError, match="RERANK=False"):
        client.score("q", ["a", "b"])


def test_from_env_reads_the_endpoint(monkeypatch):
    monkeypatch.setenv("AGORA_RERANK_URL", "http://example.invalid/score")
    monkeypatch.setenv("AGORA_RERANK_MODEL", "other-reranker")
    client = rerank.from_env()
    assert client.url == "http://example.invalid/score"
    assert client.model == "other-reranker"


def test_search_docs_honours_the_off_switch(monkeypatch, tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    sqlite3.connect(index_dir / "test-domain.sqlite").close()
    ctx = tools.Context(domain="Test Domain", data_dir=tmp_path,
                        index_dir=index_dir)
    sentinel = object()
    seen: list[object] = []

    def fake_hybrid(con, query, k, **kwargs):
        seen.append(kwargs.get("reranker"))
        return []

    monkeypatch.setattr(tools.index_search, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(tools.index_search, "default_reranker", lambda: sentinel)

    monkeypatch.setattr(policy, "RERANK", False)
    assert tools.search_docs("q", ctx=ctx) == "(no hits)"
    assert seen[-1] is None

    monkeypatch.setattr(policy, "RERANK", True)
    assert tools.search_docs("q", ctx=ctx) == "(no hits)"
    assert seen[-1] is sentinel
