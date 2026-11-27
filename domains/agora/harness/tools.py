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
    requested = policy.K if k is None else int(k)
    k = min(max(requested, 1), policy.MAX_K)
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


def _relative_document_path(path: str, ctx: Context) -> str:
    """Normalize the path forms models send into a path under ``docs_dir``.

    Accepted: ``/data/documents/<domain>/...``, ``documents/<domain>/...``,
    ``documents/...``, ``<domain>/...`` and a bare relative path. The domain is
    redundant because ``docs_dir`` already ends with it, so strip it (even
    when repeated). Any other absolute path is outside the domain.
    """
    value = str(path)
    absolute_root = f"/data/documents/{ctx.domain}"
    if value == absolute_root or value.startswith(absolute_root + "/"):
        value = value[len(absolute_root):].lstrip("/")
    elif value.startswith("/"):
        raise ValueError(f"path outside the domain documents: {path}")
    if value.startswith("documents/"):
        value = value[len("documents/"):]
    while value.startswith(f"{ctx.domain}/"):
        value = value[len(ctx.domain) + 1:]
    if value in {"", ".", "documents", ctx.domain, f"documents/{ctx.domain}"}:
        value = ""
    return value


def _document_target(path: str, ctx: Context) -> Path:
    root = Path(ctx.docs_dir).resolve()
    target = (root / _relative_document_path(path, ctx)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path outside the domain documents: {path}")
    return target


def read_file(path: str, start: int = 1, end: int | None = None,
              ctx: Context = None) -> str:
    # Only the domain's documents: DATA_DIR also holds tasks/*.jsonl with the
    # answers, so the root is docs_dir and the search form is stripped first.
    target = _document_target(path, ctx)
    if target.suffix.lower() not in {".md", ".txt", ".csv", ".json"}:
        raise ValueError(f"{path} is not text; use search_docs or python")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(int(start or 1), 1)
    stop = min(int(end), len(lines)) if end else len(lines)
    if start > stop:
        return f"{path}: no lines {start}-{stop} (file has {len(lines)})"
    text = "\n".join(lines[start - 1:stop])[:policy.READ_CHARS]
    return f"{path}:{start}-{stop}\n{text}"


def grep(pattern: str, glob: str = "*", path: str | None = None,
         ctx: Context = None) -> str:
    rg = shutil.which("rg")
    if not rg:
        raise RuntimeError("ripgrep (rg) is not installed")
    root = Path(ctx.docs_dir).resolve()
    target = root
    if path:
        relative = _relative_document_path(path, ctx)
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"path outside the domain documents: {path}")
        if target.is_file():
            glob = None  # a single file needs no glob filter
        elif not target.is_dir():
            # Not a real path: the model meant a filename pattern.
            glob = relative or glob
            target = root
    # A single-file target makes rg omit the filename unless forced.
    command = [rg, "-n", "--with-filename", "--no-heading", "--color", "never",
               "-m", str(policy.GREP_PER_FILE)]
    if glob:
        command += ["-g", glob]
    command += ["--", pattern, str(target)]
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


def _link_documents(root: Path, ctx: Context) -> None:
    # The episode scratch directory outlives one call, so the archive link is
    # created once and repaired only when it is missing or points elsewhere.
    link = root / "documents"
    target = Path(ctx.data_dir).resolve() / "documents"
    if link.is_symlink():
        try:
            if link.resolve() == target:
                return
        except (OSError, RuntimeError):
            pass
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(target)


def python(code: str, ctx: Context = None) -> str:
    # shortcut: no sandbox here, so code runs with a writable host filesystem;
    # the Phase 3 docker shim owns real isolation. The scratch archive view at
    # least keeps documents/ the only visible path. cli.py pins AGORA_SCRATCH
    # for the episode so extracted data and notes.md survive between calls;
    # host tests without it keep the disposable fallback.
    keep = os.environ.get("AGORA_SCRATCH")
    disposable = None
    if keep:
        root = Path(keep).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
    else:
        disposable = tempfile.TemporaryDirectory(prefix="agora-py-")
        root = Path(disposable.name)
    # The child gets the stable archive path, not the private scratch switch.
    env = {**os.environ, "AGORA_DATA": str(root)}
    env.pop("AGORA_SCRATCH", None)
    try:
        _link_documents(root, ctx)
        done = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, errors="replace",
                              cwd=root, timeout=policy.PY_TIMEOUT, env=env)
    finally:
        if disposable is not None:
            disposable.cleanup()
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


ANSWER_SPEC = _spec(
    "answer",
    "Finish the episode with the final value. Call once, last. Format: submit "
    "the value only, with no label, explanation or surrounding text. Keep a "
    "unit, sign, separator or rounding only when the value itself needs it.",
    {"value": {"type": "string", "description": "the answer only"}},
    ["value"])

PYTHON_SPEC = _spec(
    "python",
    "Run python (pandas, numpy, scipy available) and return stdout. "
    "The process cwd is a scratch directory that lasts for the episode, so "
    "notes.md and other files survive between calls. The archive is at the "
    "AGORA_DATA path.",
    {"code": {"type": "string"}},
    ["code"])

# The tool list is fixed for the whole episode. The final phase restricts the
# choice with tool_choice instead: the template renders the declaration before
# the first message, so changing this list would invalidate the whole cache.
SPECS = [
    _spec("search_docs",
          "Hybrid keyword and semantic search over the domain's documents. "
          "Returns path, line range and a short snippet per hit.",
          {"query": {"type": "string", "description": "what to find, in the question's terms"},
           "k": {"type": "integer", "minimum": 1, "maximum": policy.MAX_K,
                 "description": (f"hits to return (default {policy.K}, "
                                 f"max {policy.MAX_K})")}},
          ["query"]),
    _spec("read_file",
          "Read a line range of a text document from the archive. Paths come "
          "from search_docs and start with 'documents/'.",
          {"path": {"type": "string"},
           "start": {"type": "integer", "description": "first line, 1-based"},
           "end": {"type": "integer", "description": "last line, inclusive"}},
          ["path"]),
    _spec("grep",
          "Regex search for lines across the domain's documents, with paths. "
          "Set path to search one file; a glob such as '*.md' is also accepted.",
          {"pattern": {"type": "string", "description": "regular expression"},
           "glob": {"type": "string", "description": "e.g. '*.md' (default all)"},
           "path": {"type": "string",
                    "description": "one file path or a glob; default all documents"}},
          ["pattern"]),
    PYTHON_SPEC,
    ANSWER_SPEC,
]
