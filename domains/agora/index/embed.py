"""EmbeddingGemma 2 text embeddings for the Agora index and query path.

The document side runs once during ``build.py --embed``. The query side runs
inside the sandbox for every ``search_docs`` call, so this module must stay
light at import time: ``torch`` and ``sentence-transformers`` load only when an
``EmbeddingGemma`` is constructed.

PLAN.md §5.1 is mandatory here:

* text tower only (the vision and audio encoders are disabled);
* bf16 on CUDA when the device supports it, fp32 otherwise -- never fp16;
* the official task prefixes, not a hand-written paraphrase;
* mean pooling from the model's own sentence-transformers config;
* the full 768d output, then L2 normalization. The model supports Matryoshka
  truncation to 128/256/512d, but the index keeps the native dimension (#28);
* a finite-value check before any vector reaches the database.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

DIMS = 768  # native EmbeddingGemma 2 width; MRL would allow 128/256/512
QUERY_PREFIX = "task: question answering | query: "
DOCUMENT_PREFIX = "title: {title} | text: {text}"
DEFAULT_MODEL = Path.home() / "Models" / "embeddinggemma2"
MODEL_DIR = Path(os.environ.get("AGORA_EMBEDDER") or DEFAULT_MODEL)
BATCH_SIZE = int(os.environ.get("AGORA_EMBED_BATCH", "32"))

# The chunk budget targets the model's 1024-token local attention window
# (PLAN.md §5.1). The checkpoint's hard context is larger, but a chunk whose
# tail sits outside the local window is represented mostly by its head.
EMBED_MAX_TOKENS = 1024


class Embedder(Protocol):
    """The small surface ``build.py`` and ``search.py`` need."""

    dims: int
    id: str

    def encode_documents(self, texts: Sequence[str],
                         titles: Sequence[str]) -> np.ndarray:
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        ...


class ChunkTokenizer(Protocol):
    """The tokenizer surface the chunker needs: real token character spans."""

    id: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character ``(start, end)`` spans of the text's tokens, no specials."""
        ...


class EmbeddingGemmaTokenizer:
    """The EmbeddingGemma 2 text tokenizer, loaded without the model weights.

    Chunk sizes must use the same tokenizer the embedder applies. A 4-chars
    estimate overshoots badly: prose here measures ~2.9 chars/token and dense
    numeric tables ~1.1, so an "800-token" chunk can encode to thousands of
    real tokens.
    """

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        from tokenizers import Tokenizer

        path = Path(model_dir) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"no tokenizer at {path}; set AGORA_EMBEDDER or install the model")
        self._tokenizer = Tokenizer.from_file(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.id = f"embeddinggemma2:{digest}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return list(encoding.offsets)


def query_text(query: str) -> str:
    """The official question-answering query prefix."""
    return QUERY_PREFIX + query


def document_text(title: str | None, text: str) -> str:
    """The official document prefix, with the chunk breadcrumb as its title."""
    return f"title: {title or 'none'} | text: {text}"


def check_vectors(vectors, rows: int | None = None) -> np.ndarray:
    """Return float32 ``(rows, DIMS)`` vectors, or raise on a bad batch.

    A NaN or Inf vector is a silent retrieval failure: sqlite-vec would store
    it and every cosine distance becomes meaningless. Check before the write.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DIMS:
        raise ValueError(f"embedding shape {array.shape}, expected (n, {DIMS})")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"embedding batch has {array.shape[0]} rows, expected {rows}")
    finite = np.isfinite(array)
    if not finite.all():
        row, column = np.argwhere(~finite)[0]
        raise ValueError(
            f"non-finite embedding at row {row}, column {column}; "
            "refusing to write a poisoned index")
    return array


class EmbeddingGemma:
    """The text tower of EmbeddingGemma 2, loaded from a local directory."""

    def __init__(self, model_dir: Path | str = MODEL_DIR,
                 device: str | None = None, batch_size: int = BATCH_SIZE):
        import torch
        from sentence_transformers import SentenceTransformer

        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(
                f"no embedder at {path}; set AGORA_EMBEDDER or install the model")
        self.dims = DIMS
        self.batch_size = max(1, int(batch_size))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 is not a candidate. bf16 needs a device that supports it; fp32
        # is the documented fallback for CPU and older accelerators.
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        if dtype is torch.float16:  # defensive: never let a future edit pick fp16
            raise AssertionError("EmbeddingGemma 2 must not use fp16")
        self.dtype = dtype
        # dtype changes the vectors; the device does not. Keep the build key
        # stable across a CPU and a CUDA rebuild of the same precision.
        self.id = f"embeddinggemma2:{str(dtype).removeprefix('torch.')}/{self.dims}d"
        self.model = SentenceTransformer(
            str(path),
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        self.model.eval()

    def describe(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        return f"EmbeddingGemma 2 text tower, {self.dims}d, {dtype} on {self.device}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            truncate_dim=self.dims,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return check_vectors(vectors, len(texts))

    def encode_documents(self, texts: Sequence[str],
                         titles: Sequence[str]) -> np.ndarray:
        if len(texts) != len(titles):
            raise ValueError("documents and titles must have the same length")
        return self._encode([document_text(title, text)
                             for title, text in zip(titles, texts)])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([query_text(text) for text in texts])


_MODEL: EmbeddingGemma | None = None


def encode_query(text: str) -> np.ndarray:
    """Encode one query, reusing the process-wide model in the sandbox."""
    global _MODEL
    if _MODEL is None:
        _MODEL = EmbeddingGemma()
    return _MODEL.encode_queries([text])[0]
