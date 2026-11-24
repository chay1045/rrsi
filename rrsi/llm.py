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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client for the three search roles (proposer, analyst, critic).

Two providers share one `generate()` contract:

- `RRSI_PROVIDER=vertex` (default): Claude Opus 4.8 via AnthropicVertex,
  round-robin over the configured GCP projects with retry-and-rotate on
  failure. `cache_prefix` sends a large stable leading block (constitution +
  harness source) as its own ephemeral-cached content block so repeated turns
  pay to read it once.
- `RRSI_PROVIDER=openai`: any OpenAI-compatible endpoint (Ollama Cloud, vLLM)
  configured by `RRSI_BASE_URL` + `RRSI_API_KEY`. `cache_prefix` is prepended
  to the leading user block so prefix-caching endpoints still see one stable
  prefix.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
import time

MODEL = os.environ.get("RRSI_SEARCH_MODEL", "claude-opus-4-8")
_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")
    if p.strip()
]
MAX_TOKENS = 20_000

_clients: dict = {}
_openai_clients: dict = {}
_clients_lock = threading.Lock()
_rr = itertools.count()
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _provider() -> str:
    return (os.environ.get("RRSI_PROVIDER") or "vertex").strip().lower()


def _openai_client():
    """Client for the OpenAI-compatible endpoint; env only, never a key file."""
    base_url = (os.environ.get("RRSI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL") or "").strip()
    api_key = (os.environ.get("RRSI_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not base_url:
        raise RuntimeError("set RRSI_BASE_URL (or OPENAI_BASE_URL) for "
                           "RRSI_PROVIDER=openai")
    if not api_key:
        raise RuntimeError("set RRSI_API_KEY (or OPENAI_API_KEY) for "
                           "RRSI_PROVIDER=openai")
    with _clients_lock:
        c = _openai_clients.get(base_url)
        if c is None:
            from openai import OpenAI
            c = OpenAI(base_url=base_url, api_key=api_key)
            _openai_clients[base_url] = c
        return c


def _client_for(idx: int):
    from anthropic import AnthropicVertex
    if not _PROJECTS:
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                           "projects with Claude on Vertex AI enabled")
    with _clients_lock:
        c = _clients.get(idx)
        if c is None:
            c = AnthropicVertex(**_PROJECTS[idx])
            _clients[idx] = c
        return c


def extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{") and not t.startswith("["):
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
        if start != -1:
            t = t[start:]
    if t and t[0] == "{" and not t.endswith("}"):
        end = t.rfind("}")
        if end != -1:
            t = t[:end + 1]
    if t and t[0] == "[" and not t.endswith("]"):
        end = t.rfind("]")
        if end != -1:
            t = t[:end + 1]
    return t


def _openai_generate(mdl: str, prompt: str, sys_prompt: str,
                     cache_prefix: str | None, max_tokens: int,
                     max_retries: int, json_only: bool) -> str:
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    user = f"{cache_prefix}\n\n{prompt}" if cache_prefix else prompt
    messages.append({"role": "user", "content": user})
    client = _openai_client()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=mdl, max_tokens=max_tokens, messages=messages)
            text = resp.choices[0].message.content or ""
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - retry the same endpoint
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


def generate(prompt: str, system: str | None = None, max_retries: int = 6,
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None) -> str:
    mdl = model or MODEL
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")
    if _provider() == "openai":
        return _openai_generate(mdl, prompt, sys_prompt, cache_prefix,
                                max_tokens, max_retries, json_only)
    if not _PROJECTS:
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                           "projects with Claude on Vertex AI enabled")
    content = ([{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt}] if cache_prefix else prompt)
    n = len(_PROJECTS)
    start = next(_rr)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        idx = (start + attempt) % n
        try:
            client = _client_for(idx)
            kwargs = {"model": mdl, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": content}]}
            if sys_prompt:
                kwargs["system"] = sys_prompt
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - rotate to the other project
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(generate("Reply with exactly: OK", max_tokens=16))
