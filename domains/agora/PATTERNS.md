# Pattern Library (agora-v1)

Observed in the base harness on 2026-11-24. Job:
`runs/agora/jobs/issue11-probe`, one trial on each of the eight smoke tasks,
model `Inkling-2-Mini`. The probe produced no correct answer. Seven episodes
ended with `UNKNOWN`; the eighth reached the 3600-second task timeout and left
no trajectory. The traces below are the evidence, not a prediction of every
future failure.

Every pattern cites a task id, the run directory that produced it, and a
redacted excerpt under `traces/`. Gold values are removed. Use the trace as
evidence for a mechanism; do not copy its entities, paths or wording into the
harness.

## P1. Empty assistant turn ends the episode

**Status:** fixed in H_0 before the baseline. `policy.MAX_TOKENS` is 4096 and
`graph` has a bounded `nudge` node (`policy.EMPTY_TURN_RETRIES`), so an empty
reply no longer ends the episode. Do not add a second retry loop; improve the
existing one only if the baseline still shows empty turns at `END`.

**Mechanism (old H_0).** `graph.after_agent` routed any assistant message
without tool calls to `END`. A model can spend its whole completion budget on
reasoning and return `content=None` with no tool call. The episode then ends
with no answer, and `graph.run` falls back to `UNKNOWN`.

**Evidence.**
- task: `Education, Science & Academia/q_0046`
- run: `runs/agora/jobs/issue11-probe/s0/Education, Science & Academia__q_0046`
- trace: `traces/01-empty-turn.md`

```text
[step 1] ASSISTANT: content=None, tool_calls=[]
usage: completion_tokens=1024
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

The same final empty assistant message appears in six other traces: after 50
turns in `Agriculture, Resources & Energy/q_0030`, after 8 turns in
`Healthcare & Medicine/q_0024`, after 21 turns in `Law/q_0004`, after 11 turns
in `Technology, Software & Manufacturing/q_0023`, after 310 turns in
`Business, Management, Marketing & Sales/q_0040`, and after 273 turns in
`Finance & Economics/q_0003`.

**Lever.** `control_flow` and `config`. A no-tool-call reply is not an answer
unless it carries one. Consider a bounded re-prompt that requires a tool call
or an explicit `answer` call, and a completion budget large enough for the
policy's reasoning. Do not add an unbounded retry loop.

## P2. Long query dilutes BM25

**Mechanism.** `search_docs` runs BM25 and dense retrieval, then fuses them with
reciprocal rank fusion. On the BM25 side, `search.match_expression` removes
stopwords and joins every remaining term with `OR`. A question-shaped query
therefore matches many documents through common terms, and the rare terms that
identify the evidence do not dominate the BM25 rank. The dense side can still
recover the right passage, but a diluted fusion is a weak starting point.

**Evidence.**
- task: `Agriculture, Resources & Energy/q_0030`
- run: `runs/agora/jobs/issue11-probe/s0/Agriculture, Resources & Energy__q_0030`
- trace: `traces/06-query-dilution.md`

```text
[step 2] TOOL_CALL search_docs({"k":5,"query":"viticulture harvest Germany light dark components quality-use shares"})
         -> [1] documents/.../SEIA-Consumer-Protection-Primer...
```

`Healthcare & Medicine/q_0024` shows the same shape with a short but generic
query: `search_docs({"query":"lab-only"})` returned a substance-abuse audit
report, not the infection-surveillance definition. `Finance & Economics/q_0003`
searched with a long English sentence against a Chinese-language collection and
received an OECD outlook chapter, then an insurance paper.

**Lever.** `client_tool`. The harness owns query construction, not the frozen
index. Extract the likely entity and measure terms, issue narrow searches, and
combine them. Keep the one frozen `index/search.py` path.

## P3. Search loop without reading a document

**Status:** the current cloud experiment uses a 400-turn cap. The 100-turn
local H₀ is recorded at commit `42cfea4`; its slowest episode took 1837 s. The
pattern is unchanged: the issue 12 smoke task was `UNKNOWN` at 400 turns, and
baseline job `base` timed out on 41/48 trials. The cap bounds the cost; it does
not fix the loop.

**Mechanism.** The agent repeats or rewords searches and treats snippets as the
document. It never calls `read_file`, so the line range and the table that
carry the value are never in context. Token cost grows while the evidence path
does not.

**Evidence.**
- task: `Agriculture, Resources & Energy/q_0030`
- run: `runs/agora/jobs/issue11-probe/s0/Agriculture, Resources & Energy__q_0030`
- trace: `traces/02-search-loop.md`

```text
TOOL COUNTS: search_docs=36, grep=24, python=9, read_file=0
REPEATED ARGS:
  3x grep {"pattern":"permanent crop"}
  3x grep {"glob":"*.md","pattern":"vine"}
  2x search_docs {"query":"permanent crop vine grapes pressing"}
turns=50, total_tokens=375345
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

`Healthcare & Medicine/q_0024` has 10 `search_docs`, 2 `grep` and 0
`read_file`. The loop scales with the corpus:
`Business, Management, Marketing & Sales/q_0040` ran 310 turns and 3,445,929
tokens with 131 `search_docs`, 267 `grep` and only 4 `read_file`;
`Finance & Economics/q_0003` ran 273 turns and 3,212,979 tokens with 119
`search_docs`, 197 `grep` and 0 `read_file`. The eighth task,
`Architecture, Construction, Real Estate & Facilities/q_0017`, hit the
3600-second timeout and produced no trajectory.

**Lever.** `control_flow` and `prompt`. Require a read of the best hit before
another search, and block an exact repeated tool call. Make the read step a
checkable action, not motivational text.

## P4. Grep output floods the context

**Mechanism.** `grep` caps the number of matching lines, not the number of
characters. One long line or forty long lines can consume a large share of the
prompt in a single result. The old trimmer dropped the oldest tool blocks, which
could include the search that found the right file. The current harness keeps
history append-only until 85% of the window, then replaces the older part with
one summary; a giant result still brings that compaction forward and costs a
re-prefill.

**Evidence.**
- task: `Healthcare & Medicine/q_0024`
- run: `runs/agora/jobs/issue11-probe/s0/Healthcare & Medicine__q_0024`
- trace: `traces/03-grep-flood.md`

```text
[step 8] TOOL_CALL grep({"pattern":"repeat.*posit|same care area|reset.*window"})
         -> 38,431 chars in one tool result
# at the time: CONTEXT_CHARS = 48000
```

`Agriculture, Resources & Energy/q_0030` returned 16,668 and 30,758 characters
from single `grep` calls. `Business, Management, Marketing & Sales/q_0040`
returned 43,660 characters in one result, which is larger than the whole context
budget; `Finance & Economics/q_0003` returned 42,117.

**Lever.** `output_plumbing`. Bound the total characters per tool result, clip
each line, and report how much was omitted. Keep `path:line` so the model can
request the exact range.

## P5. Python scratch hides the corpus

**Mechanism.** `tools.python` runs each call in a new temporary cwd and sets
`AGORA_DATA` to that cwd. It links `documents` into the cwd, but `os.walk` does
not follow the symlink by default, so enumeration returns zero files. The cwd
is deleted after the call, so an absolute scratch path from one call fails in
the next.

**Evidence.**
- task: `Law/q_0004`
- run: `runs/agora/jobs/issue11-probe/s0/Law__q_0004`
- trace: `traces/04-python-scratch.md`

```text
[step 1] python -> AGORA_DATA=/tmp/agora-py-3_3hqrgp
         -> os.walk(...) -> 0 files
[step 12] python -> FileNotFoundError: '/tmp/agora-py-3_3hqrgp/documents'
TOOL COUNTS: python=14, search_docs=5, read_file=1
```

`Agriculture, Resources & Energy/q_0030` shows the same new-scratch path and
zero-file enumeration across several `python` calls.
`Business, Management, Marketing & Sales/q_0040` found the files only after
five failed calls, when it walked the literal relative path `documents`. The
`AGORA_DATA` value still pointed at a scratch directory that no longer existed.

**Lever.** `client_tool` and `output_plumbing`. Expose a stable, read-only
archive root to `python`, or set `AGORA_DATA` to the real corpus root while the
cwd stays disposable. Tell the model which path is stable. Do not expose the
task file, the split or the scorer.

## P6. read_file truncates a table without a marker

**Mechanism.** `read_file` joins the requested lines and applies
`READ_CHARS=6000` with no truncation marker. A wide HTML table can be cut in
the middle of a row. The model cannot tell that more rows exist.

**Evidence.**
- task: `Technology, Software & Manufacturing/q_0023`
- run: `runs/agora/jobs/issue11-probe/s0/Technology, Software & Manufacturing__q_0023`
- trace: `traces/05-read-truncation.md`

```text
[step 11] read_file({"start":245,"end":295})
          -> 6,130 chars
          -> ends mid-row: "<td>22</td><td>CWE-426</td><td>Untrusted Search Path</td><t"
[step 11] ASSISTANT: content=None, tool_calls=[]
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

This trace did find and open the right file six times. Retrieval was not the
blocker; the missing rows and the missing continuation signal were.

**Lever.** `output_plumbing`. Return an explicit truncation marker and the next
line to request, or split a long range into bounded slices. Keep table rows
whole where possible.

## P7. Evidence was read but no answer was submitted

**Mechanism.** The agent can reach the right section and still end without an
`answer` call. This is the terminal half of P1: the loop has no bounded
finalization path when the model stops producing tool calls.

**Evidence.**
- task: `Technology, Software & Manufacturing/q_0023`
- run: `runs/agora/jobs/issue11-probe/s0/Technology, Software & Manufacturing__q_0023`
- trace: `traces/05-read-truncation.md`

```text
TOOL COUNTS: read_file=6, search_docs=3, grep=2
turns=11, total_tokens=110237
[step 11] ASSISTANT: content=None, tool_calls=[]
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

**Lever.** `control_flow` and `prompt`. When the agent has read and computed the
needed evidence, route to a bounded submission step. Do not let a silent empty
reply stand in for an answer, and do not add a check that can loop forever.

## What the probe did not show

The probe produced no clean success, so it gives no success habit that can be
kept as a regression guard. It also does not measure the heldout split. Treat
these patterns as failure mechanisms observed on eight smoke tasks, not as a
complete list or as a difficulty estimate.
