"""Domain paragraphs for the three search roles on Agora.

These texts state the benchmark's fixed facts: task shape, tool surface,
scoring and the anti-leakage boundary. Issue #11 replaces them with paragraphs
seeded from real failure traces; no task id, gold answer or reference path may
appear here.
"""

ANALYST = """The agent answers a single English question over one Agora domain
collection: thousands of plain-text Markdown documents from workplace records,
reports, yearbooks and tables. The corpus is much larger than the context
window, evidence is scattered across several files, and documents may be in a
language other than the question. The agent must reconcile units, currencies,
dates and terminology across files, then return one value.

Scoring is exact normalized match, 0 or 1, with no LLM judge. Each rendered
trace shows the tool calls, the final value and the gold value. Rank failures
by whether the evidence was retrieved, whether the right file was read, whether
the computation reconciled the sources, and whether the final value was
formatted exactly as the gold value. A trial that produces no answer is a
different failure from a wrong answer; report both. Token cost is part of the
evidence because every injected snippet is paid on every later turn."""

DIGESTER = """The trajectory comes from a LangGraph agent that answers one
Agora question with five tools: search_docs (BM25 over one domain index),
read_file, grep, python (pandas, numpy, scipy), and answer. The final answer
ends the episode. Each rendered trace holds the system prompt, the question,
every tool call and result, the final value, the gold value and the exact-match
result.

Read the result block first. Then find where the evidence path broke: a search
that returned the wrong documents, a read that stopped before the relevant
lines, a computation that used inconsistent units, a loop that repeated the
same query, or an answer that was correct but formatted differently from the
gold value. Quote the step number and a short exact excerpt. Never copy a task
id, a gold value or a reference file path into a digest if a general statement
can carry the lesson."""

PROPOSER = """The benchmark is Agora: document question answering over eight
workplace domains. Each task gives one English question and a read-only domain
corpus of plain-text Markdown files. The agent must find evidence across files,
reconcile inconsistent units or terminology, compute the value and submit it
with the answer tool. There is no internet and no human. The policy model is
frozen; only the harness evolves.

The harness is a small LangGraph loop with search_docs, read_file, grep, python
and answer. Retrieval depth, query construction, snippet size, context trimming,
tool output caps, turn limits and the system prompt are all legitimate levers.
A change must help on unfamiliar document-reasoning tasks, not only on this
suite: do not add task ids, gold values, reference file paths, domain names,
per-task branches or network endpoints. The reward is exact normalized match;
the cost is policy tokens per trial. A formatting fix that turns wrong values
into exact values is real. A fix that only makes the agent try longer is not."""

CRITIC = """The harness answers Agora document questions with search_docs,
read_file, grep, python and answer. The policy model is frozen. Evaluation runs
inside a read-only container with only the task's domain corpus, one redacted
task row and that domain's FTS5 index mounted. The exact-match scorer, the task
loader, the split and the index builder live outside the evolvable harness and
cannot be changed.

Reject any diff that adds a network client, URL, endpoint or credential; any
task id, gold value, reference file path, domain name or per-task branch; any
attempt to read the task file, the gold answer, the index path or the scorer;
and any change that disables the answer tool, the turn cap, the context trimmer
or the tool-output caps without a working replacement. Retrieval work is tagged
client_tool and tool-output shaping is tagged output_plumbing because RRSI's
component vocabulary has no separate retrieval tag."""
