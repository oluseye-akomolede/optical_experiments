#!/usr/bin/env bash
# log_triage.sh -- a local-qwen FIRST PASS over a long log.
#
# WHAT THIS IS FOR. A failing pytest run, a Jetson bench log, a dead solver Job's
# pod log or a check-script report can be thousands of lines, and reading them
# costs the scarce thing: the operator's and the seats' attention. The local
# qwen2.5-32B (free, in-cluster, text never leaves it) is good at the mechanical
# part -- list the failures, quote the line that looks causal, count the repeats.
#
# WHAT IT IS NOT. A draft, never a verdict. The header line says so on every run,
# and it is printed even when the model fails, so a truncated or empty answer can
# never be mistaken for "nothing wrong". On a real log this model has produced
# confident, wrong detail -- an outcome and a repeat count right, a progress
# claim invented. Verify every cause it names against the log itself.
#
# Usage:
#     scripts/log_triage.sh path/to/log.txt
#     pytest ... 2>&1 | scripts/log_triage.sh
#     scripts/log_triage.sh --tail 500 job.log
#
# Environment:
#     TRIAGE_TAIL      lines to send (default 300)
#     TRIAGE_RAW=1     print the reduced, redacted text and DO NOT call qwen
#                      (for checking what would have been sent -- it is also the
#                       cheapest way to test the redaction)
#
# Exits non-zero, with a message, if qwen is unavailable: an empty "all clear" is
# the one output that must never be producible by a broken tool.

set -euo pipefail

TAIL_LINES="${TRIAGE_TAIL:-300}"
RAW="${TRIAGE_RAW:-0}"

usage() {
    sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --tail) shift; TAIL_LINES="${1:?--tail needs a number}"; shift ;;
        --raw) RAW=1; shift ;;
        -*) echo "log_triage: unknown option $1" >&2; usage 2 ;;
        *) break ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. read the log: from a file argument, or stdin
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    logfile="$1"
    if [ ! -f "$logfile" ]; then
        echo "log_triage: no such file: $logfile" >&2
        exit 2
    fi
    src="$logfile"
    total=$(wc -l < "$logfile")
    body=$(tail -n "$TAIL_LINES" "$logfile")
else
    if [ -t 0 ]; then
        echo "log_triage: nothing to read -- give a file or pipe a log in" >&2
        usage 2
    fi
    # A CLOSED stdin (`< /dev/null`) is not distinguished from an empty pipe
    # here, and deliberately: telling them apart would need a peek, and NO byte
    # may be peeked. The spool below reads stdin to EOF, and a `read(1)` on a
    # pipe CONSUMES that byte, so a peeking guard silently truncates the first
    # line of every piped log. (Measured: `printf abc | python -c '...
    # read(1)...'` leaves `bc`.) A peeking guard that also exits 0 on an empty
    # stream would be dead code that reads like a guard.
    #
    # Reading nothing costs nothing: both cases fall through to the spool and
    # are caught by the emptiness check below, which sees a 0-line spool and
    # says the input is empty. That is the correct message for both -- neither
    # one has anything to triage.
    src="<stdin>"
    # Spool stdin because it cannot be both counted and tailed in one pass, and
    # counting is what makes the "showing the last N of M" line honest.
    spool=$(mktemp)
    trap 'rm -f "$spool"' EXIT
    cat > "$spool"
    total=$(wc -l < "$spool")
    body=$(tail -n "$TAIL_LINES" "$spool")
fi

if [ -z "${body//[[:space:]]/}" ]; then
    echo "log_triage: the input is empty -- nothing to triage" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 2. REDACTS, then reduce. Redaction runs BEFORE anything is printed or sent,
#    and TRIAGE_RAW shows the redacted text for exactly that reason: a "raw"
#    mode that showed the unredacted log would be a footgun pointed at the
#    owner's secrets.
# ---------------------------------------------------------------------------
redact() {
    sed -E \
        -e 's#(postgres(ql)?(\+[a-z0-9_]+)?://[^:/@[:space:]]+):[^@[:space:]]+@#\1:[REDACTED]@#Ig' \
        -e 's/(password|passwd|pwd)([[:space:]]*[=:][[:space:]]*)("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:]"'"'"',;]+)/\1\2[REDACTED]/Ig' \
        -e 's/(PGPASSWORD|POSTGRES_PASSWORD|MSE_WORKER_TOKEN|LITELLM_MASTER_KEY|DEEPSEEK_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)([[:space:]]*[=:][[:space:]]*)[^[:space:]"'"'"',;]+/\1\2[REDACTED]/g' \
        -e 's/(AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|MINIO_ROOT_PASSWORD|MINIO_SECRET_KEY)([[:space:]]*[=:][[:space:]]*)[^[:space:]"'"'"',;]+/\1\2[REDACTED]/g' \
        -e 's/(Bearer[[:space:]]+)[A-Za-z0-9._~+/-]+=*/\1[REDACTED]/Ig' \
        -e 's/(AKIA|ASIA)[A-Z0-9]{16}/\1[REDACTED]/g' \
        -e 's/([?&](access_key|signature|sig|token|key)=)[^&[:space:]]+/\1[REDACTED]/Ig'
}

# Private keys are the one shape redaction cannot do line by line: the secret is
# the base64 AFTER the header, so a header-only rule leaves the key material in
# place while making the output look handled.
#
# The fence is anchored on LINE BOUNDARIES with a state machine, not on a run of
# non-'-' characters. A `[^-]*` fence (an earlier version) stops at the first
# '-' anywhere in the body, and a legacy RFC-1421 ENCRYPTED key carries
# hyphens on its second line:
#
#     -----BEGIN RSA PRIVATE KEY-----
#     Proc-Type: 4,ENCRYPTED
#     DEK-Info: AES-128-CBC,....
#
# so the fence closed immediately after "Proc" and the 25 base64 lines of the
# key went through VERBATIM while the BEGIN line looked redacted -- the worst
# shape a redaction failure can take. (Reproduce with `openssl genrsa
# -traditional -aes128 -passout pass:pw 2048`.)
#
# So: delete every line from a BEGIN ... PRIVATE KEY line through a following
# END ... PRIVATE KEY line. That covers the encrypted form, the plain form, a
# headerless body, and a key whose END line is missing (it runs to EOF, which
# costs a log that could not be read past the key anyway). It is a plain awk
# loop, so mawk and gawk behave identically -- no lookahead, no gawk extension
# on the path that has to work when a deploy log is unreadable.
#
# awk holds one boolean of state, so a multi-megabyte log cannot blow memory the
# way accumulating the whole text into a variable would.
reduced=$(printf '%s\n' "$body" | awk '
    BEGIN { inkey = 0 }
    /-----BEGIN [A-Z ]*PRIVATE KEY-----/ {
        if (!inkey) { print "-----BEGIN PRIVATE KEY----- [REDACTED]"; inkey = 1 }
        next
    }
    /-----END [A-Z ]*PRIVATE KEY-----/ {
        if (inkey) { inkey = 0; next }
    }
    { if (!inkey) print }
')
reduced=$(printf '%s\n' "$reduced" | redact)

if [ "$RAW" = "1" ]; then
    printf '%s\n' "$reduced"
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. ask qwen. The prompt is fixed: the failure mode to design against is a
#    model that answers confidently from a log it did not really read, so every
#    instruction either anchors the answer in the text or tells it to decline.
# ---------------------------------------------------------------------------
instruction='You are doing a FIRST PASS over the tail of a log, to save a human from reading all of it.
Rules, in order of importance:
1. ONLY report what the text supports. If the text does not say why something failed, write "cause unclear from this excerpt". Never guess a cause, a version, a file, or a line number that is not in the text.
2. List EACH distinct failure with a one-line likely cause, quoting the shortest decisive fragment. Group repeats and give the count, e.g. "[x23]".
3. State whether the run is TRUSTWORTHY: did every check actually execute, or did the log stop early / skip / abort? Say "unclear" if the text does not show it. A run whose checks did not all run is not a pass.
4. Say "unclear" rather than inferring. Short and specific beats fluent and vague.
Cite log lines verbatim where they matter; do not paraphrase them into something tidier.'

printf 'qwen first-pass triage: NOT a verdict, verify every cause.\n'
printf 'source: %s (%s lines, showing the last %s)\n' "$src" "$total" "$TAIL_LINES"

if ! command -v qwen >/dev/null 2>&1; then
    echo "log_triage: the 'qwen' CLI is not on PATH. It is how this machine" >&2
    echo "  reaches the local qwen2.5-32B. Nothing was triaged -- this is NOT" >&2
    echo "  an all-clear. Read the log yourself, or install/point at qwen." >&2
    exit 3
fi

# One call at a time (the GPUs are shared), and a cold start can take ~4 min, so
# the timeout is generous -- but it IS BOUNDED, enforced by `timeout` rather than
# merely intended. An earlier version's comment said the same thing with no
# `timeout` on the call at all, so a qwen that hung never returned and the script
# hung with it: indistinguishable from a log with nothing wrong, which is the one
# outcome this tool must never produce. `timeout` is coreutils, present wherever
# the shell is; if it is somehow missing the call proceeds UNBOUNDED rather than
# failing, and that is said out loud rather than silently.
QWEN_TIMEOUT_S="${TRIAGE_TIMEOUT:-360}"
timeout_cmd=()
if command -v timeout >/dev/null 2>&1; then
    timeout_cmd=(timeout "$QWEN_TIMEOUT_S")
fi

# The status is captured INSIDE the assignment, not read from `$?` afterwards:
# `if ! cmd` leaves `$?` as 0 (the negation succeeded), so testing it after the
# `if` would never match 124 and the timeout branch would be dead code that
# reads exactly like a working one -- and the operator would get "qwen failed"
# with no hint that a cold start or a busy GPU was the cause.
out_status=0
out=$(printf '%s\n' "$reduced" | "${timeout_cmd[@]}" qwen "$instruction" 2>/tmp/log_triage_err.$$) || out_status=$?
if [ "$out_status" != "0" ]; then
    err=$(tail -n 3 /tmp/log_triage_err.$$ 2>/dev/null || true)
    rm -f /tmp/log_triage_err.$$
    if [ "$out_status" = "124" ]; then
        echo "log_triage: qwen did not answer within ${QWEN_TIMEOUT_S}s; NOTHING WAS TRIAGED." >&2
        echo "  A cold start can take ~4 min -- raise TRIAGE_TIMEOUT if the GPUs" >&2
        echo "  are busy. This is NOT an all-clear." >&2
    else
        echo "log_triage: qwen failed to answer; NOTHING WAS TRIAGED." >&2
        [ -n "$err" ] && echo "  qwen said: $err" >&2
    fi
    exit 3
fi
rm -f /tmp/log_triage_err.$$

if [ -z "${out//[[:space:]]/}" ]; then
    echo "log_triage: qwen returned nothing; NOTHING WAS TRIAGED." >&2
    exit 3
fi

printf '%s\n' "$out"
