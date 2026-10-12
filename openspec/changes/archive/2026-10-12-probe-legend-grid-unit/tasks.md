# Tasks

## 1. Say which it is: (a) or (b)

- [x] 1.1 Read the stored request against the manifest for prod job `aa6f083b`,
      rather than inferring from the legend text. Verdict: **(a)**, cancellation
      residue of a probe at the frame origin.
      Verify: `frames.positions_m[a][16] == probes[0].position_m[a]` on all three
      axes, with 16 x 20 nm = 320 nm = |lo| (the probe sits on the frame origin);
      the neighbouring positions are 20 nm apart, so the grid is the requested
      nanometre grid; and the run's `dt_s` is the 3-D Courant step for
      `dx` = 20 nm, so no unit was applied twice ((b) would need ~1e9, not 1e-19).

## 2. The legend prints the grid's unit

- [x] 2.1 `probeLabel` takes the manifest, reads the grid step from
      `frames.positions_m` (`gridStep`, the finest gap), snaps a coordinate below
      `1e-6 x step` to 0, floors the prefix at the step, prints an exactly-zero
      coordinate as `0`, and keeps the old rule when there is no step.
- [x] 2.2 `viewer.js` passes the manifest into `probeLabel`.
- [x] 2.3 Tests in `tests/js/test_tdplot.mjs`, red on the old code: the
      production job's own three floats read `(0, 0, 0) nm`; a residue and a real
      node are told apart at the snap line; `gridStep` is the finest gap.
      Verify: 4 failures against `git show HEAD:web/static/tdplot.js`, one of
      them reproducing the reported string
      `probe 1: Ez at (0.0000400, -0.000320, 0.000100) fm`; 10 pass in the file
      and 146 pass / 0 fail for the whole JS suite on the fix.

## 3. tdcheck verifies the legend position

- [x] 3.1 `tdcheck.mjs` reads each legend line's position through the page's own
      prefix table, requires every axis to be the manifest's node within a grid
      step, requires an origin coordinate (below `1e-6` of the step) to read `0`,
      and says so on its output when the manifest carries no step.
- [x] 3.2 The mirror evidence, on the real 69 MB mirror of `aa6f083b`: with the
      page as prod serves it, the run FAILs exactly one check and it is the new
      origin rule -- `probe 1 x=4.002230675776569e-20 -> 0.0000400` (and y, z)
      against a `1.9999999999979903e-8 m` cell -- where the old check was
      `ALL PASS`; with `web/static/tdplot.js` and `viewer.js` from this change
      over the same mirror, `ALL PASS`.
- [x] 3.3 The fixture path (`scripts/run_tdcheck.sh`, no arguments) reaches
      `ALL PASS` (39 checks, exit 0), so the fixture job's probes -- whose
      manifest also carries positions -- are not reddened by a rule written for
      a grid they do not have.
      Verify: 38 PASS/1 FAIL old page, 39 PASS/0 FAIL fixed page and fixture;
      none of the three printed the no-step note, so all three took the grid
      branch.

## 4. OpenSpec and handoff

- [x] 4.1 `openspec validate --all --strict`: 15 passed, 0 failed, this change
      included. ADDED only: `time-domain` has no base spec, so a MODIFIED delta
      would be refused at archive.
- [x] 4.2 Commit locally on `fix-td-metal-job-page` (shared with the tdcheck
      review fixes) and report the sha on rows
      `qitem-20261002204427-9dd933c8` and `qitem-20261002205252-0cd57f5f`; hand
      both to review. No push.
