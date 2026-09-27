# scripts/

## `log_triage.sh` — a local-qwen first pass over a long log

**Use it when a log is too long to read and you only need to know what broke.**
A failing CI run, a Jetson bench log, a dead solver Job's pod log: paste the tail
in and get the failures listed, with the repeat counts, in a few lines. The model
is the local qwen2.5-32B, so it is free and the text never leaves the cluster.

```bash
scripts/log_triage.sh job.log                  # a file
pytest -q 2>&1 | scripts/log_triage.sh         # or a pipe
scripts/log_triage.sh --tail 800 build.log     # more context (default 300 lines)
TRIAGE_RAW=1 scripts/log_triage.sh job.log     # show what WOULD be sent, no model call
```

**Do not use it to decide anything.** It is a draft. The header line says so on
every run. On this rig's own logs it has produced confident, wrong detail — an
outcome and a repeat count right, a "progress" claim invented. Verify every cause
it names against the log itself, and never quote it as a finding.

Two properties are worth knowing before you trust its output:

- **It redacts before it sends.** DSNs, `*PASSWORD=`, `*_TOKEN=`, AWS/MinIO keys,
  `Bearer` tokens, signed-URL query strings and PEM blocks are replaced with
  `[REDACTED]`. `TRIAGE_RAW=1` shows the redacted text, which is the cheap way to
  check a specific log before sending it. Redaction is pattern-based, so a secret
  in a shape nobody anticipated still gets through — treat the log as leaving the
  process, not as safe to paste anywhere.
- **A failure to triage is never an all-clear.** A missing `qwen`, a model error,
  or an empty answer all exit non-zero with a message saying nothing was triaged.
  If it exits 0 you have a real answer; if it exits non-zero, read the log
  yourself.

The first call after an idle period can take ~4 minutes while the model loads
onto the shared GPUs. Calls are serialised for that reason.
