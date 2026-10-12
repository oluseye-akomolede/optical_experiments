# Tasks

## 1. Derive every expectation from the job served

- [x] 1.1 `tdcheck.mjs` imports the job's own `static/tdplot.js` and takes the
      probe count, the line and swatch counts, the two axis units
      (`siPrefix` on the run's span and band), the trace's tick factor
      (`axisExponent` at the decoded values, both ends of the page's 5% padding),
      and the movie's presence, frame count and name width (`frames`) from the
      manifest. No one-run-shape constant remains.
      Verify: with the production mirror, `one trace and one spectrum line per
      probe (1)`, `time axis in fs`, `frequency axis in THz` and
      `trace y title states the factor its own values need (peak 1.43e-16):
      … ×10⁻¹⁶` all PASS.
- [x] 1.2 A job whose manifest says no movie is checked for the page's own
      "This run recorded no field movie." caption, and the movie cases are
      reported as not applicable to it.
      Verify: the no-movie path prints that line and does not FAIL.

## 2. Run against a mirrored real job

- [x] 2.1 `scripts/tdcheck_fetch.py` mirrors a real job (page, status, manifest,
      trace, every frame the manifest names, the `/static` assets the page loads
      by following module imports) into the `TD_DIR` layout, stdlib only.
- [x] 2.2 A job is named by a full id or a unique prefix; an ambiguous or
      unknown prefix exits with an error, and a fetch that does not answer 200
      names the URL and status.
      Verify: `tests/test_tdcheck_fetch.py` (tier 1, no network).
- [x] 2.3 `scripts/run_tdcheck.sh --dir DIR` runs against a mirror, refuses one
      missing a file the check reads, and never writes to it (`--rev` on a
      mirror works on a copy).
- [x] 2.4 The check refuses a mirror whose frames on disk are not exactly the
      ones the manifest names, so an incomplete mirror is named and not judged
      as a page defect.

## 3. Read the movie canvas without an animation frame

- [x] 3.1 The slider is set and fired in one `page.evaluate`; Play is an in-page
      click; the canvas is read with `toDataURL` after `addInitScript` forced
      `preserveDrawingBuffer: true` on every WebGL context.
- [x] 3.2 The check counts the page's own animation frames and reads only after
      the loop has drawn a further frame; a loop that draws nothing for 90 s is
      a named FAIL.
      Verify: against the production mirror's 120-frame movie, the run reaches
      the movie checks and reports `the movie canvas is drawn at a real size
      (834x331)` and `two frames draw two different pictures`.

## 4. Keep the fixture path green

- [x] 4.1 `scripts/run_tdcheck.sh` with no arguments (the tiny real solve) still
      reaches `ALL PASS` (36 checks, exit 0).
- [x] 4.2 `scripts/tdcheck_fixture.py` asserts its own window (120 frames, three
      digits), since the check no longer carries it.

## 5. Verification

- [x] 5.1 The check run against the mirrored production job reaches its end with
      every check PASS (36 checks, exit 0), on the real page and the real 69 MB
      of result files. The same fixture run also passes, so the derivation did
      not trade one job's constants for another's.
- [x] 5.2 `tests/test_tdcheck_fetch.py` passes at tier 1
      (`(ulimit -v 4194304; nice -n 19 .venv-mse/bin/python -B -m pytest …)`),
      and each of its rules is mutation-proved: the loose path pattern, a fixed
      two-digit frame width and a first-match prefix each redden exactly one
      test.
- [x] 5.3 `openspec validate tdcheck-real-job --strict` ("is valid") and
      `openspec validate --all --strict` (14 passed, 0 failed) pass. The delta
      is ADDED only: `time-domain` has no base spec yet, so a MODIFIED delta
      would be refused at archive.
- [x] 5.4 Commit locally on `fix-td-metal-job-page` and report the sha on row
      `qitem-20261002201204-20f019a0` (no push). Landed as `6891cd7`. The commit
      message's claim of "57 fetch tests" is wrong -- the file has 5 tests (6
      after 6.1); the correction is in the follow-up commit's message.

## 6. Review fixes (row `qitem-20261002205252-0cd57f5f`, on top of 6891cd7)

- [x] 6.1 The non-200 abort in `scripts/tdcheck_fetch.py` is driven through the
      module's real `get` with `urlopen` stubbed (a 200 answers the body first,
      as a positive control), instead of only through fakes of `get` itself.
      Verify: with the abort deleted, the test reddens with the `HTTPError`
      escaping, and the other five tests stay green.
- [x] 6.2 `scripts/run_tdcheck.sh`'s `--dir` pre-check names `files/trace.bin`
      among the files it refuses to run without, and `tests/test_run_tdcheck.py`
      derives the required list from `tdcheck.mjs`'s own
      `path.join(DIR, '...')` reads, so a read the runner does not cover fails
      the test rather than crashing the check.
      Verify: test 1 reddens against the old four-file list naming exactly
      `['files', 'files/trace.bin', 'static/tdmovie.js']`; test 2 forces the
      refusal (exit 2, "files/trace.bin is not readable", no `docker run`).
- [x] 6.3 The caption contract has one source: `tdcheck.mjs` reads
      `SCALE_PERCENTILE` and `MOVIE_THRESHOLD` from the job's own
      `static/tdmovie.js` (exiting with a named error if it cannot) instead of
      retyping "99th percentile" and "0.12".
