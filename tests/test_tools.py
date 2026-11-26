"""Issue #20: search_docs honours a requested depth up to policy.MAX_K."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "domains" / "agora"))

import harness.graph as graph  # noqa: E402
import harness.policy as policy  # noqa: E402
import harness.tools as tools  # noqa: E402


def _context(tmp_path: Path, rows: int = 10) -> tools.Context:
    """A tiny real index: search_docs rereads each hit from the chunks table."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    con = sqlite3.connect(index_dir / "test-domain.sqlite")
    con.execute(
        "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_path TEXT,"
        " line_start INTEGER, line_end INTEGER, heading_path TEXT, text TEXT)")
    for index in range(1, rows + 1):
        con.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            (index, f"doc{index}.md", index, index, f"Heading {index}",
             f"snippet {index}"))
    con.commit()
    con.close()
    return tools.Context(domain="Test Domain", data_dir=tmp_path,
                         index_dir=index_dir)


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    """Real chunks table plus a fake hybrid search that returns its first k."""
    ctx = _context(tmp_path)

    def fake_hybrid(con, query, k, **kwargs):
        return con.execute(
            "SELECT doc_path, line_start, line_end FROM chunks"
            " ORDER BY chunk_id LIMIT ?", (k,)).fetchall()

    monkeypatch.setattr(tools.index_search, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(tools.index_search, "default_reranker", lambda: None)
    monkeypatch.setattr(policy, "RERANK", False)
    return ctx


def _hits(output: str) -> int:
    return len(re.findall(r"^\[\d+\] ", output, re.MULTILINE))


def test_search_docs_returns_the_requested_depth(indexed):
    assert _hits(tools.search_docs("q", k=10, ctx=indexed)) == 10


def test_omitted_k_uses_the_default(indexed, monkeypatch):
    seen: list[int] = []

    def fake_hybrid(con, query, k, **kwargs):
        seen.append(k)
        return []

    monkeypatch.setattr(tools.index_search, "hybrid_search", fake_hybrid)
    assert tools.search_docs("q", ctx=indexed) == "(no hits)"
    assert seen == [policy.K]


@pytest.mark.parametrize("requested, effective",
                         [(99, policy.MAX_K), (0, 1), (-3, 1)])
def test_k_is_clamped(indexed, monkeypatch, requested, effective):
    seen: list[int] = []

    def fake_hybrid(con, query, k, **kwargs):
        seen.append(k)
        return []

    monkeypatch.setattr(tools.index_search, "hybrid_search", fake_hybrid)
    tools.search_docs("q", k=requested, ctx=indexed)
    assert seen == [effective]


def test_missing_index_names_the_build_command(tmp_path):
    ctx = tools.Context(domain="Test Domain", data_dir=tmp_path,
                        index_dir=tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="run index/build.py"):
        tools.search_docs("q", ctx=ctx)


def test_graph_passes_the_requested_depth_to_search_docs(indexed):
    def call(name, args, call_id):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}

    replies = iter([call("search_docs", {"query": "q", "k": 10}, "c1"),
                    call("answer", {"value": "done"}, "c2")])

    def client(messages, specs):
        return dict(next(replies))

    state = graph.run("q", client, indexed)
    result = next(message["content"] for message in state["messages"]
                  if message["role"] == "tool")
    assert result.startswith("[1] ")
    assert "\n\n[10] " in result
    assert state["answer"] == "done"


def test_search_spec_states_the_default_and_maximum():
    spec = next(item for item in tools.SPECS
                if item["function"]["name"] == "search_docs")
    prop = spec["function"]["parameters"]["properties"]["k"]
    assert prop["minimum"] == 1
    assert prop["maximum"] == policy.MAX_K
    assert str(policy.K) in prop["description"]
    assert str(policy.MAX_K) in prop["description"]
