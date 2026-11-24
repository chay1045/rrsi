# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load a ``.env`` file into ``os.environ`` for the command-line entry points.

Deliberately not imported by the role modules: ``rrsi/llm.py`` still reads only
``os.environ``, so importing the library (and running its tests) never touches a
key file. ``rrsi.py`` and ``python -m rrsi.llm`` call this before the roles run.

File format is the usual one::

    # comment
    RRSI_PROVIDER=openai
    export RRSI_API_KEY="..."
    RRSI_SEARCH_MODEL=deepseek-v4.1-flash:1017

Existing environment variables win unless ``override=True``. The file is found
from ``RRSI_ENV_FILE`` when set, otherwise by walking up from the CLI's repo
root looking for ``.env``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def find_env_file(start: str | os.PathLike | None = None) -> Path | None:
    """Return the ``.env`` to load: RRSI_ENV_FILE, else the nearest parent."""
    explicit = os.environ.get("RRSI_ENV_FILE")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"RRSI_ENV_FILE does not exist: {path}")
        return path
    here = Path(start or os.getcwd()).expanduser().resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _unescape(body: str) -> str:
    out = []
    i = 0
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body) and body[i + 1] in escapes:
            out.append(escapes[body[i + 1]])
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def _value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        body = raw[1:-1]
        return _unescape(body) if raw[0] == '"' else body
    # An inline comment starts at whitespace followed by '#'.
    comment = re.search(r"\s+#", raw)
    if comment:
        raw = raw[:comment.start()].rstrip()
    return raw


def parse_env(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; raise ``ValueError`` on a malformed line."""
    values: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise ValueError(f"line {lineno}: expected KEY=VALUE")
        key, raw = line.split("=", 1)
        key = key.strip()
        if not _KEY.match(key):
            raise ValueError(f"line {lineno}: invalid key {key!r}")
        values[key] = _value(raw)
    return values


def load_env(path: str | os.PathLike | None = None,
             start: str | os.PathLike | None = None,
             override: bool = False,
             environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Load a ``.env`` file and return the names actually set.

    ``environ`` is injectable for tests. Values already present in the
    environment win unless ``override=True``.
    """
    env = os.environ if environ is None else environ
    chosen = Path(path).expanduser() if path else find_env_file(start)
    if chosen is None:
        return []
    loaded = []
    for key, value in parse_env(chosen.read_text(encoding="utf-8")).items():
        if override or key not in env:
            env[key] = value
            loaded.append(key)
    return loaded
