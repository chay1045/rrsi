"""Knobs, declarative only. RRSI tunes these; no logic belongs here.

Char budgets, not token budgets: the tool layer never sees the policy
tokenizer, and 4 chars/token keeps the estimate one line.
"""

K = 5                    # search_docs hits per call
MAX_TURNS = 400          # model calls per episode. The cloud 100-turn smoke took
                         # 308 s, so 400 (~21 min) fits the 3600 s task timeout.
                         # The local 100-turn H0 is recorded at 42cfea4; local
                         # runs must lower this cap or expect timeouts.
TEMPERATURE = 1.0        # Agora's official setting; cli.py passes these two
MAX_TOKENS = 8192        # to the endpoint when it wraps the client. 1024 cut the
                         # reasoning short; 4096 was never saturated, 8192 gives
                         # a hard reasoning turn more room.
EMPTY_TURN_RETRIES = 2   # bounded re-prompts when an assistant turn has no content
                         # and no tool call, instead of ending the episode
EMPTY_TURN_NUDGE = ("Your last reply was empty. Call one tool now. "
                    "If you have the final value, call answer.")

SNIPPET_CHARS = 1200     # ~300 tokens per search hit
READ_CHARS = 12000       # one read_file slice (two 6k slices)
GREP_HITS = 40           # total grep lines
GREP_PER_FILE = 3
PY_CHARS = 4000          # python stdout+stderr kept
PY_TIMEOUT = 60          # seconds

CONTEXT_CHARS = 48000    # ~12k tokens of prompt, trimmed oldest-tool-first. The
                         # prompt budget stays fixed: with 100 turns, raising it
                         # risks the 3600 s timeout; the read slice is larger.
