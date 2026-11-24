# Trace: python scratch hides the corpus

- task: `Law/q_0004`
- run: `runs/agora/jobs/issue11-probe/s0/Law__q_0004`
- episode: base harness, `Inkling-2-Mini`, `k=1`
- result: `status=ok`, `turns=21`, `total_tokens=201118`, answer `UNKNOWN`, exact match 0

## Redacted excerpt

```text
TOOL COUNTS: python=14, search_docs=5, read_file=1
[step 1] TOOL_CALL python({"code":"import os; print(os.environ.get('AGORA_DATA')); os.walk(...)"})
         -> /tmp/agora-py-3_3hqrgp
         -> /tmp/agora-py-3_3hqrgp 0
[step 7] TOOL_CALL read_file({"path":"documents/.../Burial-and-Cremation-Consultation-Paper.md","start":1,"end":120})
         -> works; returns the document slice
[step 12] TOOL_CALL python({"code":"... os.listdir('/tmp/agora-py-3_3hqrgp/documents') ..."})
         -> FileNotFoundError: '/tmp/agora-py-3_3hqrgp/documents'
[step 21] ASSISTANT: content=None, tool_calls=[]
RESULT: PREDICTION UNKNOWN, EXACT MATCH 0
```

## Reading

`tools.python` runs each call in a new temporary cwd and sets `AGORA_DATA` to
that cwd. It links `documents` into the cwd, but `os.walk` does not follow that
symlink by default, so the model sees zero files. The scratch directory is
deleted after the call, so a path printed by one call is invalid in the next.

The same failure appears in `Agriculture, Resources & Energy/q_0030`: several
`python` calls print a new `/tmp/agora-py-*` path and then find zero files.
`Business, Management, Marketing & Sales/q_0040` found the corpus only after
five failed calls, by walking the literal relative path `documents`; the
`AGORA_DATA` value still named a scratch directory that no longer existed.
