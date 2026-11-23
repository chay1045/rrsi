"""Knobs, declarative only. RRSI tunes these; no logic belongs here.

Char budgets, not token budgets: the tool layer never sees the policy
tokenizer, and 4 chars/token keeps the estimate one line.
"""

K = 5                    # search_docs hits per call
MAX_TURNS = 60           # model calls per episode (PLAN §3.2: 60-turn cap)
TEMPERATURE = 1.0        # Agora's official setting; cli.py passes these two
MAX_TOKENS = 1024        # to the endpoint when it wraps the client

SNIPPET_CHARS = 1200     # ~300 tokens per search hit
READ_CHARS = 6000        # one read_file slice
GREP_HITS = 40           # total grep lines
GREP_PER_FILE = 3
PY_CHARS = 4000          # python stdout+stderr kept
PY_TIMEOUT = 60          # seconds

CONTEXT_CHARS = 48000    # ~12k tokens of prompt, trimmed oldest-tool-first
