# Agora episode sandbox. Build from the rrsi repository root:
#   docker build -t agora-sandbox:1 domains/agora/sandbox
#
# The image carries the frozen agent runtime only. Candidate code, the domain
# corpus, the FTS5 index and the task row are bind-mounted read-only by
# domains/agora/adapter.py; nothing task-specific is baked in.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fixed mount points. The adapter mounts one domain corpus at
# /data/documents/<domain>, one redacted task row at /data/tasks, one index
# file at /index and the trial output at /out.
RUN mkdir -p /app/agora /data/documents /data/tasks /index /out

# Gate 0 checks: FTS5, the agent runtime, the scientific Python the python tool
# advertises, and the PDF helpers the corpus may need.
RUN python -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); import bs4, fitz, langgraph, lxml, numpy, openai, openpyxl, pandas, pdfplumber, scipy, tabulate, yaml; print('agora sandbox runtime ok')"

RUN rg --version | head -1

WORKDIR /app/agora
CMD ["python", "--version"]
