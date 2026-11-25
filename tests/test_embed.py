"""Issue #14: EmbeddingGemma 2 prefixes and the pre-write finite check.

    python3 -m pytest -q tests/test_embed.py
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

AGORA = Path(__file__).resolve().parents[1] / "domains" / "agora"
sys.path.insert(0, str(AGORA / "index"))

spec = spec_from_file_location("agora_index_embed", AGORA / "index" / "embed.py")
embed = module_from_spec(spec)
sys.modules[spec.name] = embed
spec.loader.exec_module(embed)


def test_official_task_prefixes():
    assert embed.query_text("what is the margin?") == (
        "task: question answering | query: what is the margin?")
    assert embed.document_text("Report > Costs", "the margin was 10.37 pp") == (
        "title: Report > Costs | text: the margin was 10.37 pp")
    assert embed.document_text(None, "plain") == "title: none | text: plain"


def test_check_vectors_accepts_a_finite_batch():
    vectors = np.eye(embed.DIMS, dtype=np.float32)[:2]
    checked = embed.check_vectors(vectors, rows=2)
    assert checked.shape == (2, embed.DIMS)
    assert checked.dtype == np.float32


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_vectors_rejects_non_finite_values(bad):
    # Deliberate NaN/Inf trigger: a poisoned vector must never reach sqlite-vec.
    vectors = np.zeros((1, embed.DIMS), dtype=np.float32)
    vectors[0, embed.DIMS // 2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        embed.check_vectors(vectors, rows=1)


def test_check_vectors_rejects_wrong_dims_or_count():
    with pytest.raises(ValueError, match="shape"):
        embed.check_vectors(np.zeros((1, embed.DIMS - 1), dtype=np.float32))
    with pytest.raises(ValueError, match="rows"):
        embed.check_vectors(np.zeros((1, embed.DIMS), dtype=np.float32), rows=2)
