# Agora episode sandbox. Build from the rrsi repository root:
#   docker build -t agora-sandbox:1 domains/agora/sandbox
#
# The image carries the frozen agent runtime only. Candidate code, the domain
# corpus, the hybrid SQLite index, the task row and the EmbeddingGemma 2 weights
# are bind-mounted read-only by domains/agora/adapter.py; nothing task-specific
# is baked in.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# CPU-only PyTorch: the sandbox embeds one query at a time and must not compete
# with vLLM for the GB10 or carry CUDA libraries it cannot use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.15.0+cpu torchvision==0.29.2+cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fixed mount points. The adapter mounts one domain corpus at
# /data/documents/<domain>, one redacted task row at /data/tasks, one hybrid
# index at /index, the EmbeddingGemma 2 text tower at /models/embeddinggemma2
# and the trial output at /out.
RUN mkdir -p /app/agora /data/documents /data/tasks /index /out /models

# Gate 0 checks: FTS5, sqlite-vec KNN, the text embedder, the agent runtime, the
# scientific Python the python tool advertises, and the PDF helpers the corpus
# may need.
RUN python -c "import sqlite3, sqlite_vec; con=sqlite3.connect(':memory:'); con.enable_load_extension(True); sqlite_vec.load(con); con.execute('CREATE VIRTUAL TABLE t USING vec0(embedding float[256] distance_metric=cosine)'); import bs4, fitz, langgraph, lxml, numpy, openai, openpyxl, pandas, pdfplumber, PIL, scipy, sentence_transformers, tabulate, torch, torchvision, yaml; print('agora sandbox runtime ok')"

RUN rg --version | head -1

WORKDIR /app/agora
CMD ["python", "--version"]
