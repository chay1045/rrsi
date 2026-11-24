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
"""Unit tests for the optional .env loader. No network, no real keys."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rrsi.env as env  # noqa: E402


def test_parse_comments_export_and_quotes():
    parsed = env.parse_env(
        "# comment\n"
        "PLAIN=value\n"
        "export SPACED =  two words  \n"
        "SINGLE='literal $HOME'\n"
        'DOUBLE="line\\nbreak"\n'
        'PATHY="C:\\\\tmp"\n'
        "EMPTY=\n")
    assert parsed["PLAIN"] == "value"
    assert parsed["SPACED"] == "two words"
    assert parsed["SINGLE"] == "literal $HOME"
    assert parsed["DOUBLE"] == "line\nbreak"
    assert parsed["PATHY"] == "C:\\tmp"
    assert parsed["EMPTY"] == ""


def test_parse_inline_comment_only_outside_quotes():
    assert env.parse_env("KEY=value # note")["KEY"] == "value"
    assert env.parse_env("KEY='value # keep'")["KEY"] == "value # keep"


def test_parse_rejects_malformed_lines():
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        env.parse_env("not an assignment")
    with pytest.raises(ValueError, match="invalid key"):
        env.parse_env("1BAD=x")


def test_load_env_does_not_override_existing_by_default(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=from-file\nB=from-file\n")
    environ = {"A": "from-shell"}
    loaded = env.load_env(path, environ=environ)
    assert loaded == ["B"]
    assert environ == {"A": "from-shell", "B": "from-file"}


def test_load_env_override_true(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=from-file\n")
    environ = {"A": "from-shell"}
    loaded = env.load_env(path, override=True, environ=environ)
    assert loaded == ["A"]
    assert environ["A"] == "from-file"


def test_find_env_file_walks_up(tmp_path, monkeypatch):
    monkeypatch.delenv("RRSI_ENV_FILE", raising=False)
    root = tmp_path / "umbrella"
    nested = root / "vendor" / "rrsi"
    nested.mkdir(parents=True)
    (root / ".env").write_text("A=1\n")
    assert env.find_env_file(nested) == root / ".env"


def test_rrsi_env_file_override(tmp_path, monkeypatch):
    path = tmp_path / "custom.env"
    path.write_text("A=1\n")
    monkeypatch.setenv("RRSI_ENV_FILE", str(path))
    assert env.find_env_file(tmp_path) == path


def test_missing_rsri_env_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RRSI_ENV_FILE", str(tmp_path / "nope.env"))
    with pytest.raises(FileNotFoundError, match="RRSI_ENV_FILE"):
        env.find_env_file(tmp_path)


def test_no_env_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("RRSI_ENV_FILE", raising=False)
    environ = {}
    assert env.load_env(start=tmp_path, environ=environ) == []
    assert environ == {}
