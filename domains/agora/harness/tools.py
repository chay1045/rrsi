"""The five tools of Appendix C. Each returns a string, or raises.

Raises are not crashes: graph.py turns them into an ``error:`` tool message so
the model can recover. ``answer`` is handled by graph.py because it ends the
episode, not by a function here.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The index modules are siblings and are imported the way evaluate.py imports
# them: by path, so the tool queries the same hybrid retrieval path Gate 2 measures.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "index"))
import search as index_search  # noqa: E402

from . import policy  # noqa: E402

DATA_DIR = Path(os.environ.get("AGORA_DATA", Path.home() / "work" / "agora-data"))
INDEX_DIR = (
    Path(os.environ.get("RRSI_CACHE_DIR", Path.home() / ".cache" / "rrsi"))
    / "agora"
    / "index"
)


@dataclass(frozen=True)
class Context:
    domain: str
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "documents" / self.domain


def _slug(domain: str) -> str:
    # same formula as index/build.py's _slug; the index file is outside the tree
    return re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")


def search_docs(query: str, k: int | None = None, ctx: Context = None) -> str:
    db = Path(ctx.index_dir) / f"{_slug(ctx.domain)}.sqlite"
    if not db.is_file():
        raise FileNotFoundError(f"no index for {ctx.domain!r} at {db}; run index/build.py")
    k = min(max(int(k or policy.K), 1), policy.K)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        reranker = index_search.default_reranker() if policy.RERANK else None
        hits = index_search.hybrid_search(con, query, k, reranker=reranker)
        blocks = []
        for position, (doc_path, line_start, line_end) in enumerate(hits, 1):
            row = con.execute(
                "SELECT heading_path, text FROM chunks"
                " WHERE doc_path = ? AND line_start = ? AND line_end = ? LIMIT 1",
                (doc_path, line_start, line_end),
            ).fetchone()
            heading, text = row if row else ("", "")
            snippet = text[:policy.SNIPPET_CHARS]
            if len(text) > policy.SNIPPET_CHARS:
                snippet += " ..."
            blocks.append(
                f"[{position}] {doc_path}:{line_start}-{line_end}\n"
                f"{heading}\n{snippet}")
    finally:
        con.close()
    return "\n\n".join(blocks) or "(no hits)"


def read_file(path: str, start: int = 1, end: int | None = None,
              ctx: Context = None) -> str:
    # Only the domain's documents: DATA_DIR also holds tasks/*.jsonl with the
    # answers, so the root is docs_dir and the search form is stripped first.
    root = Path(ctx.docs_dir).resolve()
    prefix = f"documents/{ctx.domain}/"
    target = (root / (path[len(prefix):] if path.startswith(prefix) else path)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path outside the domain documents: {path}")
    if target.suffix.lower() not in {".md", ".txt", ".csv", ".json"}:
        raise ValueError(f"{path} is not text; use search_docs or python")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(int(start or 1), 1)
    stop = min(int(end), len(lines)) if end else len(lines)
    if start > stop:
        return f"{path}: no lines {start}-{stop} (file has {len(lines)})"
    text = "\n".join(lines[start - 1:stop])[:policy.READ_CHARS]
    return f"{path}:{start}-{stop}\n{text}"


def grep(pattern: str, glob: str = "*", ctx: Context = None) -> str:
    rg = shutil.which("rg")
    if not rg:
        raise RuntimeError("ripgrep (rg) is not installed")
    command = [rg, "-n", "--no-heading", "--color", "never",
               "-m", str(policy.GREP_PER_FILE)]
    if glob:
        command += ["-g", glob]
    command += ["--", pattern, str(ctx.docs_dir)]
    done = subprocess.run(command, capture_output=True, text=True,
                          errors="replace", timeout=policy.PY_TIMEOUT)
    if done.returncode not in (0, 1):
        raise RuntimeError((done.stderr or "rg failed").strip())
    lines = done.stdout.splitlines()
    if not lines:
        return "(no matches)"
    shown = lines[:policy.GREP_HITS]
    if len(lines) > len(shown):
        shown.append(f"... {len(lines) - len(shown)} more matching lines")
    return "\n".join(shown)


def python(code: str, ctx: Context = None) -> str:
    # shortcut: no sandbox here, so code runs with a writable host filesystem;
    # the Phase 3 docker shim owns real isolation. The scratch archive view at
    # least keeps documents/ the only visible path and the cwd disposable.
    with tempfile.TemporaryDirectory(prefix="agora-py-") as scratch:
        target = Path(ctx.data_dir).resolve() / "documents"
        (Path(scratch) / "documents").symlink_to(target)
        done = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, errors="replace",
                              cwd=scratch, timeout=policy.PY_TIMEOUT,
                              env={**os.environ, "AGORA_DATA": scratch})
    output = done.stdout or ""
    if done.stderr:
        output += ("\n[stderr]\n" if output else "") + done.stderr
    if not output:
        output = "(no output)"
    if done.returncode:
        output = f"exit {done.returncode}\n{output}"
    return output[:policy.PY_CHARS]


def dispatch(name: str, args: dict, ctx: Context) -> str:
    if name in {"search_docs", "read_file", "grep", "python"}:
        return globals()[name](**args, ctx=ctx)
    raise ValueError(f"unknown tool: {name}")


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required}}}


SPECS = [
    _spec("search_docs",
          "Hybrid keyword and semantic search over the domain's documents. "
          "Returns path, line range and a short snippet per hit.",
          {"query": {"type": "string", "description": "what to find, in the question's terms"},
           "k": {"type": "integer", "description": f"hits (default {policy.K})"}},
          ["query"]),
    _spec("read_file",
          "Read a line range of a text document from the archive. Paths come "
          "from search_docs and start with 'documents/'.",
          {"path": {"type": "string"},
           "start": {"type": "integer", "description": "first line, 1-based"},
           "end": {"type": "integer", "description": "last line, inclusive"}},
          ["path"]),
    _spec("grep",
          "Regex search for lines across the domain's documents, with paths.",
          {"pattern": {"type": "string", "description": "regular expression"},
           "glob": {"type": "string", "description": "e.g. '*.md' (default all)"}},
          ["pattern"]),
    _spec("python",
          "Run python (pandas, numpy, scipy available) and return stdout. "
          "The process cwd is a scratch directory; the archive is at the "
          "AGORA_DATA path.",
          {"code": {"type": "string"}},
          ["code"]),
    _spec("answer",
          "Finish the episode with the final value. Call once, last.",
          {"value": {"type": "string", "description": "the answer only"}},
          ["value"]),
]
