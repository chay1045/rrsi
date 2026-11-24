# Trace: long query dilutes BM25

- task: `Agriculture, Resources & Energy/q_0030`
- run: `runs/agora/jobs/issue11-probe/s0/Agriculture, Resources & Energy__q_0030`
- episode: base harness, `Inkling-2-Mini`, `k=1`
- result: `status=ok`, `turns=50`, `total_tokens=375345`, answer `UNKNOWN`, exact match 0

## Redacted excerpt

```text
[step 2] TOOL_CALL search_docs({"k":5,"query":"viticulture harvest Germany light dark components quality-use shares"})
         -> [1] documents/.../SEIA-Consumer-Protection-Primer...
```

`search.match_expression` removes stopwords and joins every remaining term with
`OR`. Long natural-language queries therefore match many documents through
common terms, and the rare terms that identify the evidence do not dominate the
BM25 rank.

`Healthcare & Medicine/q_0024` shows the same shape with a short but generic
query: `search_docs({"query":"lab-only"})` returned a substance-abuse audit
report, not the infection-surveillance definition.

## Reading

The harness, not the frozen index, owns query construction. The model passed a
question-shaped string to a search tool whose ranking treats it as a bag of
OR-ed words. The result set is plausible text, so the model keeps using it.
