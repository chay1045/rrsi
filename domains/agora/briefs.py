"""Domain paragraphs for the three search roles on Agora.

These texts state the benchmark's fixed facts and the failure mechanisms seen
in the base-harness probe. Issue #11 seeds them from real traces; no task id,
gold answer or reference path may appear here.

The probe was one trial on each of the eight smoke tasks with the frozen policy.
It produced no correct answer. The recurring mechanisms were an empty final
assistant turn, long queries that diluted BM25, search loops without a document
read, grep results that consumed most of the context, a python scratch view
that hid the corpus, and a read slice that cut a table without a marker.
"""

ANALYST = """The agent answers one English question over one Agora domain
collection: thousands of plain-text Markdown documents from workplace records,
reports, yearbooks and tables. The corpus is much larger than the context
window, evidence is scattered across several files, and documents may be in a
language other than the question. The agent must reconcile units, currencies,
dates and terminology across files, then return one value.

Scoring is exact normalized match, 0 or 1, with no LLM judge. A missing trial
is reward 0 with the full denominator. Each rendered trace shows the tool calls,
the final value and the gold value. Rank failures by whether the evidence was
retrieved, whether the right file and line range were read, whether the
computation reconciled the sources, and whether the final value was formatted
exactly as the gold value. A trial that produces no answer is a different
failure from a wrong answer; report both. Token cost is part of the evidence
because every injected snippet is paid on every later turn.

The base-harness probe on eight tasks produced no correct answer. Seven
episodes ended on an assistant message with no content and no tool call, and
the runner fell back to UNKNOWN; the eighth hit the task timeout and left no
trajectory. Two traces spent more than three million tokens on repeated search
and grep calls. Look for the step where the evidence path broke, not only at
the final value. If a trace has no `answer` call, classify the terminal cause
separately from a wrong value.
"""

DIGESTER = """The trajectory comes from a LangGraph agent that answers one
Agora question with five tools: search_docs (hybrid BM25 + EmbeddingGemma 2
retrieval over one domain index), read_file, grep, python (pandas, numpy,
scipy), and answer. The final answer ends the episode. Each rendered trace
holds the system prompt, the question, every tool call and result, the final
value, the gold value and the exact-match result.

Read the result block first. Then inspect the last assistant message: if it has
no content and no tool call, the episode ended without an answer, and the
terminal cause is control flow or the completion budget. Next count the tool
calls and check for these mechanisms:

- repeated or reworded searches with no read_file on the best hit;
- a long natural-language query whose top hits are off-topic because the index
  ORs every term;
- a grep result measured in tens of thousands of characters, or a read result
  that stops in the middle of a table with no truncation marker;
- python that prints a new temporary path and then finds zero files, or a path
  from an earlier call that no longer exists;
- an answer call after the evidence was read but without reconciliation.

Quote the step number and a short exact excerpt. Never copy a task id, a gold
value or a reference file path into a digest if a general statement can carry
the lesson. Do not infer a success habit from a lucky guess; this probe produced
seven UNKNOWN answers and one timeout, with no clean success.
"""

PROPOSER = """The benchmark is Agora: document question answering over eight
workplace domains. Each task gives one English question and a read-only domain
corpus of plain-text Markdown files. The agent must find evidence across files,
reconcile inconsistent units or terminology, compute the value and submit it
with the answer tool. There is no internet and no human. The policy model is
frozen; only the harness evolves.

The harness is a small LangGraph loop with search_docs, read_file, grep, python
and answer. Retrieval depth, query construction, snippet size, context trimming,
tool output caps, turn limits and the system prompt are legitimate levers. A
change must help on unfamiliar document-reasoning tasks, not only on this
suite: do not add task ids, gold values, reference file paths, domain names,
per-task branches or network endpoints. The reward is exact normalized match;
the cost is policy tokens per trial.

The base-harness probe found six mechanisms worth a targeted edit:

1. an empty assistant turn ends the episode before an answer;
2. a question-shaped query is a bag of OR-ed terms, so rare terms lose to
   common ones;
3. the loop searches and greps without ever reading a document;
4. grep output is capped by line count, not characters, and floods the context;
5. python runs in a disposable scratch directory and cannot enumerate the
   corpus;
6. read_file cuts a long table at a character cap without a truncation marker.

Prefer a mechanism on the component that owns the failure. A terminal empty
turn is control flow or config, not more prompt text. Query dilution is the
client tool. A flood or a silent cut is output plumbing. A missing read is
control flow. A formatting fix that turns wrong values into exact values is
real. A fix that only makes the agent try longer is not. The probe produced no
correct answer, so do not claim a success habit it did not show.
"""

CRITIC = """The harness answers Agora document questions with search_docs,
read_file, grep, python and answer. The policy model is frozen. Evaluation runs
inside a read-only container with only the task's domain corpus, one redacted
task row, that domain's hybrid index and the EmbeddingGemma 2 text tower
mounted. The exact-match scorer, the task loader, the split and the index
builder live outside the evolvable harness and cannot be changed.

Reject any diff that adds a network client, URL, endpoint or credential; any
task id, gold value, reference file path, domain name or per-task branch; any
attempt to read the task file, the gold answer, the index path or the scorer;
and any change that disables the answer tool, the turn cap, the context trimmer
or the tool-output caps without a working replacement.

Apply the probe evidence:

- A change that only raises MAX_TOKENS, MAX_TURNS or a character cap is not a
  mechanism. It must also bound the new cost or remove a failure path.
- A query rewrite must not embed terms from one question or one domain. It must
  derive terms from the current question and the general corpus vocabulary.
- A python fix must not expose the task file, the split or the scorer. The
  corpus root is the only stable data path.
- A read or grep change must keep the path:line contract so the agent can ask
  for the next range.
- An empty-turn fix must not create an unbounded retry loop or a path that can
  finalize without an answer.

Retrieval work is tagged client_tool and tool-output shaping is tagged
output_plumbing because RRSI's component vocabulary has no separate retrieval
tag. A mislabelled edit is re-tagged from the diff.
"""
