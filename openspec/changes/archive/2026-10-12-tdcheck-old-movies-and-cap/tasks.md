# Tasks

## 1. Name an old movie's frames the way the page does

- [x] 1.1 `scripts/tdcheck_fetch.py`: `FRAME_DIGITS_DEFAULT = 2` and a
      `frame_digits(frames)` that both sites use, so a manifest with no `digits`
      key (a movie recorded before three-digit names) names its files the width
      `viewer.js` defaults to, instead of raising `KeyError`.
      Verify: `tests/test_tdcheck_fetch.py`'s keyless-manifest case is red with
      the old `frames['digits']` (KeyError) and green with the default.

## 2. Cap the run by the movie, and name a cut run

- [x] 2.1 `scripts/run_tdcheck.sh`: read `frames.count` from the mirror's
      `files/spectrum.json`, cut at `180 + 12 x frames` seconds (the solver's
      120-frame maximum when there is no manifest to read), let
      `TDCHECK_TIMEOUT` override it, and print the cap chosen.
      Verify: `tests/test_run_tdcheck.py::test_the_cap_follows_the_movies_own_frame_count`
      and `::test_the_cap_defaults_to_the_solvers_maximum_and_can_be_overridden`
      are red against the old fixed `timeout 600` runner.
- [x] 2.2 `scripts/run_tdcheck.sh`: keep the run's status and report a 124 as
      `harness timed out after <cap>s (<frames> frames; ...)`, rather than
      passing `timeout(1)`'s own status through with nothing said.
      Verify: `::test_a_run_cut_by_the_cap_is_named_and_is_not_a_page_defect`
      (a stub that outlives a 1 s cap) exits 124 with that line on stderr.
- [x] 2.3 `dockerfiles/rendercheck/tdcheck.mjs`: on the signal that stops the
      container, print one plain `HARNESS TIMED OUT` line (written with
      `writeSync`, since `process.exit` does not flush a pipe) and exit 124; an
      unhandled rejection prints one `HARNESS ERROR` line rather than a stack
      trace about a closed page.
      Verify: read the two handlers; the cut test's forced case shows the
      runner's plain line and no stack trace in either stream.

## 3. Spec, then check it against what is already written

- [x] 3.1 This change's delta (`specs/time-domain/spec.md`), with a scenario per
      branch of the cap and one for the keyless manifest.
      Verify: `openspec validate --all --strict`.
- [x] 3.2 Grep the change dirs for a claim this supersedes (a fixed cap, or
      reading `frames['digits']`) so the delta does not leave an older
      requirement saying the opposite.
      Verify: no hit outside this change.

## 4. Affected tests only

- [x] 4.1 Run the two affected files, tier 1: `nice -n 19`,
      `(ulimit -v 4194304; ...)`. The full suite goes to a bounded `ci` Job, not
      this VM (the lead's correction, 2026-10-03).
      Verify: `pytest tests/test_run_tdcheck.py tests/test_tdcheck_fetch.py -q`.
