"""scripts/log_triage.sh -- the redaction, the tail cap, and the failure modes.

Two properties matter more than the prompt's wording, and both are the kind that
look right while being wrong:

* **The redaction.** This script pipes log text to another process. Logs are
  where credentials end up -- a DSN in a connection error, a token in a curl
  line, an env dump. A redactor that misses one spelling (``password: "x"`` with
  a quote, ``postgresql+psycopg2://`` with a driver suffix) leaks silently,
  because the output still looks redacted everywhere else.
* **The refusals.** A triage tool that prints an empty string when the model is
  down is worse than no tool: emptiness reads as "nothing wrong". Every path that
  cannot triage must exit non-zero with a message, and the header line -- which
  says the output is not a verdict -- must print even then.

The tests run the real script with a fake ``qwen`` first on PATH, so what is
exercised is the script's own plumbing. Where a test needs to see what WOULD be
sent, it uses ``TRIAGE_RAW=1``, which is the script's own supported way to do
that -- not a flag invented for the test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "log_triage.sh"


def run_triage(args=(), stdin="", env=None, qwen=None, tmp_path=None):
    """Run the real script, with ``qwen`` resolved from a directory we control.

    ``PATH`` is REPLACED for the fake case, so a real qwen on this machine cannot
    be reached by accident -- a test that silently called the 32B model would be
    slow, non-deterministic, and would hide a broken stub.
    """
    base = dict(os.environ)
    base.pop("TRIAGE_TAIL", None)
    base.pop("TRIAGE_RAW", None)
    if qwen is not None and tmp_path is not None:
        bindir = tmp_path / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        exe = bindir / "qwen"
        exe.write_text(qwen)
        exe.chmod(0o755)
        # Put our fake first, keeping the shell's own tools reachable. When the
        # caller also set PATH (to make qwen UNRESOLVABLE) that one wins, so a
        # test is never silently pointed at the real 32B model.
        if "PATH" not in (env or {}):
            base["PATH"] = f"{bindir}:{os.environ.get('PATH', '')}"
    if env:
        base.update(env)
    # An absolute binary path, because a test that narrows PATH to make `qwen`
    # unresolvable would otherwise take `bash` away from subprocess itself and
    # fail with `FileNotFoundError: 'bash'` -- a Python-side error that says
    # nothing about the script under test.
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args], input=stdin, capture_output=True,
        text=True, env=base, timeout=60,
    )


# ---------------------------------------------------------------------------
# the tail cap
# ---------------------------------------------------------------------------

def test_only_the_last_n_lines_are_taken(tmp_path):
    """The cap is what keeps a 500k-line log from being sent at all.

    Asserted on the RAW text the script would send, so the property is about
    what leaves the machine -- not about what the model later chose to mention.
    """
    log = tmp_path / "big.log"
    log.write_text("".join(f"line {n}\n" for n in range(1, 1001)))

    result = run_triage([str(log)], env={"TRIAGE_TAIL": "10", "TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    sent = result.stdout.splitlines()
    assert "line 1000" in sent, "the newest line was dropped"
    assert "line 991" in sent
    assert "line 990" not in sent, (
        "the cap sent more than TRIAGE_TAIL lines:\n" + result.stdout)


def test_the_default_cap_is_applied_without_being_asked(tmp_path):
    """A default that does not apply is not a default.

    The interesting assertion is the ABSENCE of the early lines, which is what
    distinguishes "capped at 300" from "capped at 300 only when told".
    """
    log = tmp_path / "d.log"
    log.write_text("".join(f"line {n}\n" for n in range(1, 501)))
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    sent = result.stdout.splitlines()
    assert len(sent) == 300, f"expected the default 300-line cap, got {len(sent)}"
    assert "line 500" in sent and "line 201" in sent and "line 200" not in sent


def test_the_manifest_line_counts_the_whole_log_not_the_tail(tmp_path):
    """The header must not imply the log was only 300 lines long.

    A reader who sees "300 lines" and knows the file is 40,000 will otherwise
    think the tool lied -- and they would be right to."""
    log = tmp_path / "c.log"
    log.write_text("".join(f"line {n}\n" for n in range(1, 501)))
    result = run_triage([str(log)], env={"TRIAGE_RAW": "0"},
                        qwen="#!/bin/sh\ncat >/dev/null\necho done\n", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "500 lines" in result.stdout, result.stdout
    assert "showing the last 300" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# the redaction
# ---------------------------------------------------------------------------

#: Every entry is (input line, a substring that must NOT survive).
#: The DSNs and prefixed variables are the shapes that actually occur in this
#: rig's logs; the quote/space variants are the ones a first-attempt regex
#: misses, which is why they are here rather than only the easy form.
SECRET_LINES = [
    ("connecting to postgresql://postgres:sup3rs3cret@127.0.0.1:55433/mse",
     "sup3rs3cret"),
    ("DATABASE_URL=postgresql://mse_app:hunter2@db:5432/mse", "hunter2"),
    ('DATABASE_URL="postgresql://postgres:pw123@127.0.0.1:55433/msecheck"', "pw123"),
    ("postgresql+psycopg2://user:driv3rpw@host:5432/db", "driv3rpw"),
    ("dsn = \"postgres://u:secretpw@h/db\"", "secretpw"),
    ("PGPASSWORD=abcdef123456", "abcdef123456"),
    ("POSTGRES_PASSWORD=pw", "=pw"),
    ("MSE_WORKER_TOKEN=tok-abc-123-def", "tok-abc-123-def"),
    ("LITELLM_MASTER_KEY=sk-1234567890abcdef", "sk-1234567890abcdef"),
    ("DEEPSEEK_API_KEY=dsk-abcdef1234", "dsk-abcdef1234"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENG", "wJalrXUtnFEMIK7MDENG"),
    ("MINIO_ROOT_PASSWORD=miniosecret", "miniosecret"),
    ("password: \"quoted secret\"", "quoted secret"),
    ("password='single-quoted'", "single-quoted"),
    ("  password = spaced-out", "spaced-out"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def", "eyJhbGciOiJIUzI1NiJ9"),
    ("curl -H \"Authorization: bearer lowercase-tok\" x", "lowercase-tok"),
    ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("?access_key=AKIAIOSFODNN7EXAMPLE&signature=deadbeef", "deadbeef"),
]

#: The private-key BODY sits on the lines AFTER the header, so a header-only
#: redaction leaves the key material in the log while looking redacted. Kept out
#: of the table above because the property is about a spilled multi-line block,
#: not about one line.
PRIVATE_KEY_BLOCK = """\
line before
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAsecretkeymaterial
morekeymaterial
-----END RSA PRIVATE KEY-----
line after
"""


@pytest.mark.parametrize("line,secret", SECRET_LINES,
                         ids=[s for _, s in SECRET_LINES])
def test_each_secret_shape_is_redacted(tmp_path, line, secret):
    """One test per shape, so a regression names the pattern that broke.

    A single test over all of them would report "redaction failed" and leave the
    next reader to find which spelling leaks.
    """
    log = tmp_path / "s.log"
    log.write_text(line + "\n")
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout, (
        f"the secret survived redaction:\n{result.stdout}")
    assert "[REDACTED]" in result.stdout, (
        "nothing was redacted -- the pattern did not match this shape at all")


def test_a_private_key_body_is_redacted_not_just_its_header(tmp_path):
    """Redacting the header and leaving the key is a redaction that looks done.

    A spilled PEM is the worst case for a line-oriented redactor: the header
    matches, so the output reads as handled, and the base64 -- the actual secret
    -- sits on the very next line, untouched.
    """
    log = tmp_path / "k.log"
    log.write_text(PRIVATE_KEY_BLOCK)
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    for secret in ("MIIEowIBAAKCAQEAsecretkeymaterial", "morekeymaterial"):
        assert secret not in result.stdout, (
            f"the key body survived redaction:\n{result.stdout}")
    # The marker proves the rule fired rather than that the text was dropped for
    # some other reason. The END line is deliberately NOT asserted: the rule
    # consumes it along with the body (an END-lookahead is a gawk-only
    # extension, and this script must not fail on mawk).
    assert "[REDACTED]" in result.stdout, result.stdout
    assert "line after" in result.stdout, (
        "the redaction ran past the key and ate the rest of the log")
    assert "line before" in result.stdout, result.stdout


#: A legacy RFC-1421 ENCRYPTED key. Its body carries hyphens on line two
#: ("Proc-Type: 4,ENCRYPTED"), which is what defeated the old `[^-]*` fence: the
#: fence closed at that hyphen and every base64 line went through verbatim while
#: the BEGIN line looked redacted. Reproduced from real openssl output.
ENCRYPTED_PEM_BLOCK = """\
before the key
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,E4C1BCC919C362175F032727E3B49670

eC0Grd53mqGgTmh4uTPwnv0zBo5ZBJxFiJx1aq6arLgUvVsf6/54MP/r3907oKEB
0+lHpA3zNYFZQNwrybyuC6/viaymgLEbB+YR4cvFFerpfaAtnmjqo+y9QRzcjedG
-----END RSA PRIVATE KEY-----
after the key
"""

#: A PEM whose body has no Proc-Type/DEK-Info preamble at all.
HEADERLESS_PEM_BLOCK = """\
-----BEGIN PRIVATE KEY-----
YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkwYWJjZGVm
-----END PRIVATE KEY-----
"""


def test_an_encrypted_pem_does_not_leak_its_body(tmp_path):
    """The encrypted form is the one the fence has to survive.

    A `[^-]*` fence stops at the hyphen in "Proc-Type", so the key body passes
    through untouched while the BEGIN line looks redacted -- a redaction that
    reads as done. Asserting on the whole body, not one line, so a partial fix
    that drops the preamble but keeps the base64 still fails.
    """
    log = tmp_path / "enc.pem"
    log.write_text(ENCRYPTED_PEM_BLOCK)
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    for leaked in ("eC0Grd53mqGgTmh4uTPwnv0zBo5ZBJxFiJx1aq6arLgUvVsf6",
                   "0+lHpA3zNYFZQNwrybyuC6/viaymgLEbB+YR4cvFFerpfaAtnmjq"):
        assert leaked not in result.stdout, (
            f"encrypted key material reached the output:\n{result.stdout}")
    assert "[REDACTED]" in result.stdout, result.stdout
    assert "after the key" in result.stdout, (
        "the fence ran past the END line and ate the rest of the log")
    assert "before the key" in result.stdout, result.stdout


def test_a_headerless_pem_body_is_redacted(tmp_path):
    """A bare body with no Proc-Type preamble, fenced only by BEGIN/END."""
    log = tmp_path / "h.pem"
    log.write_text(HEADERLESS_PEM_BLOCK)
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    assert "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw" not in result.stdout
    assert "[REDACTED]" in result.stdout, result.stdout


def test_no_key_material_reaches_the_model_even_from_a_truncated_tail(tmp_path):
    """What matters is what QWEN RECEIVES, not what the script prints.

    A redaction bug that only affects a branch (say, the non-RAW path, or a key
    that spans the tail cutoff) would pass a stdout test and still ship the key
    to another process. So this asserts on the stub's stdin directly.
    """
    log = tmp_path / "tail.pem"
    # The key starts well before the tail cutoff, so the tail contains a key
    # whose BEGIN line is NOT present -- a truncated fence.
    log.write_text("\n".join(f"filler {n}" for n in range(20)) + "\n"
                   + ENCRYPTED_PEM_BLOCK)
    capture = tmp_path / "stdin.txt"
    fake = f"#!/bin/sh\ncat > {capture}\necho ok\n"
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path,
                        env={"TRIAGE_TAIL": "40"})
    assert result.returncode == 0, result.stderr
    received = capture.read_text()
    for leaked in ("eC0Grd53mqGgTmh4uTPwnv0zBo5ZBJxFiJx1aq6arLgUvVsf6",
                   "0+lHpA3zNYFZQNwrybyuC6/viaymgLEbB+YR4cvFFerpfaAtnmjq"):
        assert leaked not in received, (
            f"KEY MATERIAL REACHED THE MODEL'S STDIN:\n{received}")
    assert "DEK-Info" not in received, (
        f"the encrypted key's preamble reached the model:\n{received}")


def test_no_base64_key_line_reaches_the_model_for_any_pem_shape(tmp_path):
    """The general form: no long base64 run may survive into stdin.

    The two tests above name the exact strings from one generated key. This one
    catches a key whose bytes we did not predict -- generated fresh here -- by
    refusing any long base64-looking line, which is what key material looks like
    regardless of how it was produced.
    """
    import subprocess as sp
    keygen = sp.run(["openssl", "genrsa", "-traditional", "-aes128",
                     "-passout", "pass:x", "2048"],
                    capture_output=True, text=True)
    if keygen.returncode != 0 or "BEGIN" not in keygen.stdout:
        pytest.skip("openssl cannot generate a key here")
    log = tmp_path / "fresh.pem"
    log.write_text(keygen.stdout + "trailing line\n")
    capture = tmp_path / "stdin.txt"
    fake = f"#!/bin/sh\ncat > {capture}\necho ok\n"
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    received = capture.read_text()
    base64_lines = [ln for ln in received.splitlines()
                    if len(ln) >= 20 and ln.strip("=+/").isalnum()]
    assert not base64_lines, (
        f"base64 key-body lines reached the model's stdin:\n" +
        "\n".join(base64_lines))
    assert "trailing line" in received, (
        "the fence consumed the rest of the log")


def test_a_harmless_line_is_not_mangled(tmp_path):
    """Redaction must not eat ordinary text.

    The password pattern is deliberately value-anchored (it needs ``=`` or ``:``
    and a value). A looser one that redacted the WORD everywhere would corrupt
    the log it is supposed to make readable -- and the corruption would be
    invisible, because redacted text looks intentional.
    """
    keep = [
        "the pwd builtin is not a secret",
        "password hashing uses argon2 here",
        "working directory /home/sigmastrain",
        "connecting to host db:5432",           # no credentials in the DSN
        "the token bucket refilled",
    ]
    log = tmp_path / "h.log"
    log.write_text("\n".join(keep) + "\n")
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert result.returncode == 0, result.stderr
    for line in keep:
        assert line in result.stdout, (
            f"redaction mangled an innocent line: {line!r}\n{result.stdout}")


def test_the_redaction_runs_before_the_text_is_printed(tmp_path):
    """RAW mode must show the REDACTED text, not the original.

    ``TRIAGE_RAW`` exists so a human can check what would be sent. If it printed
    the unredacted input it would be a footgun aimed at exactly the secrets it
    was added to protect.
    """
    log = tmp_path / "p.log"
    log.write_text("PGPASSWORD=topsecretvalue\n")
    result = run_triage([str(log)], env={"TRIAGE_RAW": "1"})
    assert "topsecretvalue" not in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# the refusals: an empty answer must never be producible by a broken tool
# ---------------------------------------------------------------------------

def test_qwen_missing_is_non_zero_with_a_message(tmp_path):
    """The whole point of the tool is to save reading; if it cannot read, say so.

    A silent success here would be the rig's signature defect: a could-not-ask
    reported as nothing-there.
    """
    log = tmp_path / "q.log"
    log.write_text("1 failed\n")
    # A bin dir with every tool the script needs EXCEPT qwen. Symlinked from the
    # real tool locations rather than an empty PATH, because an empty PATH makes
    # the script die on a missing `wc` before it ever looks for qwen -- a
    # failure that says nothing about the qwen-unavailable path.
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for tool in ("wc", "tail", "sed", "awk", "mktemp", "cat", "head", "rm",
                 "printf", "command", "sleep"):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    assert not (bindir / "qwen").exists()
    result = run_triage([str(log)], env={"PATH": str(bindir)})
    assert result.returncode != 0, (
        f"a missing qwen exited 0 -- an empty all-clear:\n{result.stdout}")
    assert "NOT" in result.stderr and "all-clear" in result.stderr, (
        f"the refusal does not say nothing was triaged:\n{result.stderr}")
    # The header (the "not a verdict" warning) must still be printed, so a log
    # captured into a file by a pipeline is still labelled.
    assert "NOT a verdict" in result.stdout, result.stdout


def test_qwen_failing_is_non_zero_with_its_own_words(tmp_path):
    """A model that errors is not a model that found nothing."""
    log = tmp_path / "q.log"
    log.write_text("1 failed\n")
    fake = "#!/bin/sh\ncat >/dev/null\necho 'model unavailable' >&2\nexit 1\n"
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path)
    assert result.returncode != 0
    assert "NOTHING WAS TRIAGED" in result.stderr, result.stderr
    assert "model unavailable" in result.stderr, (
        "qwen's own error was swallowed, so an operator cannot tell a dead "
        "endpoint from a cold start")


def test_qwen_returning_nothing_is_non_zero(tmp_path):
    """Exit 0 with empty output is the failure mode to design against.

    A model that cold-started, hung and was killed can exit 0 having produced no
    text; printing that as a successful triage is the bug.
    """
    log = tmp_path / "q.log"
    log.write_text("1 failed\n")
    result = run_triage([str(log)], qwen="#!/bin/sh\ncat >/dev/null\nexit 0\n",
                        tmp_path=tmp_path)
    assert result.returncode != 0, (
        f"an empty qwen answer exited 0:\n{result.stdout}")
    assert "NOTHING WAS TRIAGED" in result.stderr, result.stderr


def test_a_hanging_qwen_is_bounded_by_the_timeout(tmp_path):
    """The comment claimed the call was bounded; the call was not.

    A stub that sleeps forever used to hang the script forever -- and a hung
    triage is indistinguishable from a log with nothing wrong, which is the one
    outcome this tool must never produce. The timeout makes that a real
    non-zero exit instead.

    A SHORT timeout is set explicitly: the default (360s) is deliberately long
    enough for a cold start, so waiting for it would make this test slow and
    hide whether the mechanism works at all.
    """
    log = tmp_path / "h.log"
    log.write_text("1 failed\n")
    fake = "#!/bin/sh\ncat >/dev/null\nsleep 300\n"
    t0 = time.monotonic()
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path,
                        env={"TRIAGE_TIMEOUT": "2"})
    elapsed = time.monotonic() - t0
    assert result.returncode != 0, (
        f"a hanging qwen exited 0 -- an empty all-clear:\n{result.stdout}")
    assert elapsed < 60, (
        f"the script ran {elapsed:.0f}s despite a 2s timeout -- it is not bounded")
    assert "NOTHING WAS TRIAGED" in result.stderr, result.stderr
    # The timeout must be NAMED, so an operator can tell a slow-but-working
    # model from a dead endpoint. Collapsing both into "qwen failed" is how
    # someone concludes the model is broken when it was only cold-starting.
    assert "did not answer within 2s" in result.stderr, (
        f"the timeout branch did not name the timeout:\n{result.stderr}")
    assert "NOT a verdict" in result.stdout, result.stdout


def test_the_timeout_default_allows_a_cold_start(tmp_path):
    """The default must be generous: the model can take ~4 min to load.

    A default that is too short turns every cold start into a false failure,
    which trains an operator to ignore the message -- worse than no timeout.
    Asserted on the UNIT (minutes, not seconds), not by waiting, so the test
    stays fast while still pinning the decision.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'TRIAGE_TIMEOUT:-(\d+)\}', text)
    assert match, "the timeout default is no longer a literal in the script"
    default = int(match.group(1))
    assert default >= 240, (
        f"the default timeout is {default}s, below the ~4 min a cold start can "
        "take -- cold starts would be reported as failures")


def test_the_header_always_precedes_the_model_output(tmp_path):
    """The 'not a verdict' line is the contract of the whole tool.

    It must be the FIRST thing, before any model text, so a reader who stops
    after the first line has still been told not to trust it.
    """
    log = tmp_path / "h.log"
    log.write_text("1 failed\n")
    fake = "#!/bin/sh\ncat >/dev/null\necho 'CAUSE: everything is fine'\n"
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("qwen first-pass triage: NOT a verdict"), (
        f"the warning is not the first line: {lines[0]!r}")
    assert any("everything is fine" in line for line in lines[1:]), (
        "the model's answer was dropped")


def test_the_model_receives_the_log_on_stdin(tmp_path):
    """The handoff is stdin, which is how the culture's qwen wrapper reads it.

    Asserted by having the fake echo its stdin back: if the script passed the
    log some other way (an argument, a temp file), the fake would see nothing
    and this fails.
    """
    log = tmp_path / "s.log"
    log.write_text("DISTINCTIVE_MARKER_LINE\n")
    fake = "#!/bin/sh\ncat\n"          # echo stdin to stdout
    result = run_triage([str(log)], qwen=fake, tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "DISTINCTIVE_MARKER_LINE" in result.stdout, result.stdout


def test_a_pipe_input_works_and_is_capped(tmp_path):
    """The documented pipe form must behave like the file form."""
    text = "".join(f"line {n}\n" for n in range(1, 101))
    fake = "#!/bin/sh\ncat\n"
    result = run_triage(stdin=text, qwen=fake, tmp_path=tmp_path,
                        env={"TRIAGE_TAIL": "5"})
    assert result.returncode == 0, result.stderr
    assert "line 100" in result.stdout
    assert "line 95" not in result.stdout, result.stdout


def test_no_input_at_all_is_refused(tmp_path):
    """Called with no file and a CLOSED stdin: refuse, do not hang or print nothing.

    The message names the empty input rather than "nothing to read", and that is
    the honest one: telling a closed stdin from an empty pipe would need a peek,
    and peeking a pipe consumes a byte of the log (see the script's comment). So
    this is the same path an empty file takes, and it says the same thing.

    What matters is the exit code and that nothing is offered as a result -- an
    empty answer that exits 0 is the failure this guards.
    """
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "empty" in result.stderr, result.stderr
    assert result.stdout.strip() == "", (
        f"a refusal printed something that could be read as a result:\n{result.stdout}")


def test_an_empty_log_is_refused(tmp_path):
    """Empty input is not a clean run, and must not become one."""
    log = tmp_path / "empty.log"
    log.write_text("")
    result = run_triage([str(log)], qwen="#!/bin/sh\ncat\n", tmp_path=tmp_path)
    assert result.returncode != 0
    assert "empty" in result.stderr, result.stderr


def test_a_missing_file_is_refused_clearly(tmp_path):
    result = run_triage([str(tmp_path / "nope.log")])
    assert result.returncode != 0
    assert "no such file" in result.stderr, result.stderr


def test_the_script_is_executable_and_parses():
    """A shortcut for a broken edit, and a claim about the file mode.

    ``bash -n`` proves the file PARSES, not that it works -- the tests above are
    what establish behaviour. It is kept because a syntax error otherwise shows
    up as a confusing failure in the first behavioural test instead of here.
    """
    assert os.access(SCRIPT, os.X_OK), (
        "log_triage.sh is not executable -- it is documented as a command")
    result = subprocess.run(["bash", "-n", str(SCRIPT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
