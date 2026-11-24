# Trace: grep output floods the context

- task: `Healthcare & Medicine/q_0024`
- run: `runs/agora/jobs/issue11-probe/s0/Healthcare & Medicine__q_0024`
- episode: base harness, `Inkling-2-Mini`, `k=1`
- result: `status=ok`, `turns=8`, `total_tokens=70925`, answer `UNKNOWN`, exact match 0

## Redacted excerpt

```text
[step 8] TOOL_CALL grep({"pattern":"repeat.*posit|same care area|reset.*window"})
         -> 38,431 chars in one tool result
         -> first hits: unrelated provider manual, disability survey, counselling module
CONTEXT_CHARS = 48000
[step 8] ASSISTANT: content=None, tool_calls=[]
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

The base harness caps `grep` at 40 matching lines, not at a character count. A
single line can be long, so one call can consume most of the 48,000-character
context budget. The oldest tool blocks are then dropped by the trimmer,
including the searches that had found the right file.

`Agriculture, Resources & Energy/q_0030` shows the same shape: one `grep`
returned 16,668 characters and another returned 30,758 characters. Single
results in `Business, Management, Marketing & Sales/q_0040` and
`Finance & Economics/q_0003` reached 43,660 and 42,117 characters.

## Reading

The grep pattern was broad, and the output had no per-line cap and no summary
of omitted content. The model paid for the full result on every later turn,
then lost its early evidence to the trimmer.
