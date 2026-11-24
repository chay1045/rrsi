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
"""Issue #08: the OpenAI-compatible path in `rrsi/llm.py`, with a stubbed client.

    python3 -m pytest -q tests/test_llm_provider.py

No network and no key file: the client is a plain stub that records the
request kwargs and returns scripted content.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rrsi.llm as llm  # noqa: E402


class _Completions:
    def __init__(self, stub):
        self.stub = stub

    def create(self, **kwargs):
        self.stub.calls.append(kwargs)
        reply = self.stub.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


class _Stub:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=_Completions(self))


@pytest.fixture
def openai_env(monkeypatch):
    monkeypatch.setenv("RRSI_PROVIDER", "openai")
    monkeypatch.setenv("RRSI_BASE_URL", "http://stub/v1")
    monkeypatch.setenv("RRSI_API_KEY", "test-key")
    # The Vertex rotation must not be reached, even with no projects.
    monkeypatch.setattr(llm, "_PROJECTS", [])
    return monkeypatch


def _stub(monkeypatch, *replies):
    client = _Stub(*replies)
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    return client


def test_openai_path_skips_empty_vertex_rotation(openai_env):
    client = _stub(openai_env, "OK")
    assert llm.generate("Reply with exactly: OK", max_tokens=16,
                        model="deepseek-v4.1-flash:1017") == "OK"
    assert client.calls[0]["model"] == "deepseek-v4.1-flash:1017"
    assert client.calls[0]["max_tokens"] == 16


def test_cache_prefix_and_json_suffix_for_openai(openai_env):
    client = _stub(openai_env, '```json\n{"verdict": "accept"}\n```')
    out = llm.generate("Review the candidate.", system="You are the critic.",
                       json_only=True, cache_prefix="STABLE HARNESS",
                       model="m")
    assert json.loads(out) == {"verdict": "accept"}
    kwargs = client.calls[0]
    assert kwargs["messages"][0] == {
        "role": "system", "content": "You are the critic." + llm._JSON_SUFFIX}
    assert kwargs["messages"][1] == {
        "role": "user", "content": "STABLE HARNESS\n\nReview the candidate."}


@pytest.mark.parametrize("role,payload", [
    ("analyst", '{"report": {"failure_modes": []}}'),
    ("proposer", '{"action": "done", "edits": []}'),
    ("critic", '{"verdict": "accept", "reasons": []}'),
])
def test_json_only_responses_parse_for_all_three_roles(openai_env, role, payload):
    _stub(openai_env, f"Here is the object:\n```json\n{payload}\n```\n")
    out = llm.generate(f"{role} turn", system=role, json_only=True, model="m")
    assert json.loads(out) == json.loads(payload)


def test_empty_response_retries(openai_env):
    client = _stub(openai_env, "", "OK")
    assert llm.generate("x", max_retries=2, model="m") == "OK"
    assert len(client.calls) == 2


def test_failures_retry_then_raise(openai_env):
    client = _stub(openai_env, RuntimeError("boom"), RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="generate failed after 2 tries"):
        llm.generate("x", max_retries=2, model="m")
    assert len(client.calls) == 2


def test_vertex_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("RRSI_PROVIDER", raising=False)
    assert llm._provider() == "vertex"


def test_empty_vertex_projects_raise_a_clear_error(monkeypatch):
    monkeypatch.delenv("RRSI_PROVIDER", raising=False)
    monkeypatch.setattr(llm, "_PROJECTS", [])
    with pytest.raises(RuntimeError, match="RRSI_VERTEX_PROJECTS"):
        llm.generate("x")


def test_missing_base_url_and_key_fail_clearly(monkeypatch):
    for name in ("RRSI_BASE_URL", "OPENAI_BASE_URL",
                 "RRSI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="RRSI_BASE_URL"):
        llm._openai_client()
    monkeypatch.setenv("RRSI_BASE_URL", "http://stub/v1")
    with pytest.raises(RuntimeError, match="RRSI_API_KEY"):
        llm._openai_client()


def test_blank_rrsi_base_url_does_not_fall_back_to_policy_endpoint(monkeypatch):
    monkeypatch.setenv("RRSI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("RRSI_API_KEY", "cloud-key")
    with pytest.raises(RuntimeError, match="RRSI_BASE_URL"):
        llm._openai_client()


def test_openai_client_reads_base_url_and_key_from_env(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("RRSI_BASE_URL", "http://stub/v1/")
    monkeypatch.setenv("RRSI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_openai_clients", {})
    client = llm._openai_client()
    assert str(client.base_url).rstrip("/") == "http://stub/v1"
    assert client.api_key == "test-key"
