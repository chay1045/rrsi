"""Knobs, declarative only. RRSI tunes these; no logic belongs here.

Char budgets, not token budgets: the tool layer never sees the policy
tokenizer, and 4 chars/token keeps the estimate one line.

``TEMPERATURE`` and ``CONTEXT_CHARS`` read a run-wide override so the same
harness serves the local and cloud policies. The defaults are the local H0
values; the adapter forwards the variables into the sandbox.
"""

import os

K = 5                    # search_docs default hits per call. Ten hits double
                         # the tool result, so the default stays at 5; a caller
                         # that needs breadth asks for up to MAX_K.
MAX_K = 10               # search_docs hard cap, equal to Gate 2's top-10 and
                         # the reranker output. The tool spec advertises it.
RERANK = True            # cross-encoder rescoring of the RRF top-50. Gate 2
                         # (issue #15): file recall +0.036, 95% CI [+0.006,
                         # +0.066]; region +0.102; MRR +0.109. Costs ~1.0 s per
                         # search, about 30 s per base14 episode. Requires
                         # scripts/serve_rerank.sh; set False for RRF only.
MAX_TURNS = 150          # model calls per episode. The cloud 100-turn smoke took
                         # 308 s, so 150 (~8 min) fits the 7200 s task timeout.
                         # The local 100-turn H0 is recorded at 42cfea4; the
                         # longer timeout leaves room for a 96k-char episode.
FINAL_TURNS = 6          # last calls of the episode: five may still use python
                         # to compute the value; the last call offers only the
                         # answer tool and the client forces it.
FINAL_NUDGE = ("You have reached the final turns of this episode. Do not search "
               "or read again. You may use the python tool if the value still "
               "needs computation. Call the answer tool when the value is ready; "
               "the last final turn accepts only answer. Submit the value only. "
               "If you are not certain, submit your best supported value; do not "
               "guess one.")
FINAL_ANSWER_NUDGE = ("This is the last final turn. The python tool is no longer "
                      "available. Call the answer tool now with the final value "
                      "only. If you are not certain, submit your best supported "
                      "value; do not guess one.")
# Agora's official setting is 1.0; the local and cloud runs use 0.6. cli.py
# passes this and MAX_TOKENS to the endpoint when it wraps the client.
TEMPERATURE = float(os.environ.get("AGORA_TEMPERATURE", "0.6"))
MAX_TOKENS = 8192        # 1024 cut the reasoning short; 4096 was never
                         # saturated, 8192 gives a hard reasoning turn more room.
EMPTY_TURN_RETRIES = 2   # bounded re-prompts when an assistant turn has no content
                         # and no tool call, instead of ending the episode
EMPTY_TURN_NUDGE = ("Your last reply was empty. Call one tool now. "
                    "If you have the final value, call answer.")
REPEAT_LIMIT = 2         # occurrences of one exact name+arguments key in the last
                         # 20 calls that trigger the repeat nudge
REPEAT_NUDGE = ("You repeated the same tool call with the same arguments. Do not "
                "repeat it; it adds no evidence. Read notes.md and reuse what you "
                "already have, change the query or path, compute with python, or "
                "call answer if the value is ready.")

SNIPPET_CHARS = 1200     # ~300 tokens per search hit
READ_CHARS = 12000       # one read_file slice (two 6k slices)
GREP_HITS = 40           # total grep lines
GREP_PER_FILE = 3
PY_CHARS = 4000          # python stdout+stderr kept
PY_TIMEOUT = 60          # seconds

# ~24k tokens of prompt at the 4 chars/token estimate, trimmed
# oldest-tool-first. The local and cloud runs use 96000; the 262k-token window
# leaves room for the completion and the next turn.
CONTEXT_CHARS = int(os.environ.get("AGORA_CONTEXT_CHARS", "96000"))
