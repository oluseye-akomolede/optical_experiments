# Run the time-domain check against a real production job

## Why

`dockerfiles/rendercheck/tdcheck.mjs` is the check that says a finished
time-domain job's card draws what the run recorded. It had only ever been run
against its own fixture, and the lead ran it against a byte-for-byte mirror of a
real job (`aa6f083b`, 1 probe, 525 steps, a 120-frame movie, 69 MB). Three
things went wrong, and all three are measured in that run.

1. **The check asserted its fixture, not the job.** Five checks FAILed on a page
   that was right: "one trace and one spectrum line per probe" was hard-coded to
   2 (the job has 1), the swatch count likewise, the trace's tick factor was
   written as `×10⁻¹³` (the job's own peak is `1.43e-16`), the time axis was
   written as `time (ps|ns)` (the job's span picks `fs`), and the frequency axis
   as `frequency (GHz)` (its band picks `THz`). Those were the fixture's own
   numbers — its trace peaks at `3.33e-13`, its span picks `ps` and its band
   `GHz` — so the check had been asserting its fixture against itself and could
   only ever have been run there. A check that fails a correct page
   is as useless as one that passes a broken page: it cannot be pointed at
   production, which is exactly where it is supposed to run.

2. **The run never reached its end.** Both attempts aborted on the movie section:
   run A hung at `locator('#td-scene').screenshot()` (`tdcheck.mjs:163`), run B
   at `page.fill('#td-movie-frame')` actionability (`tdcheck.mjs:190`),
   `Timeout 30000ms exceeded`. The cause is one the cavity checks already
   document: a real movie's per-frame work starves `requestAnimationFrame`, and
   Playwright's `fill`/`screenshot` both wait on an animation frame. A run that
   hangs tells you nothing about the page.

3. **There was no way to point it at a real job.** The check reads a directory
   (`TD_DIR`) laid out like `scripts/tdcheck_fixture.py` writes it, but the only
   thing that wrote that layout was the fixture, and the openrig VM cannot
   resolve the tailnet, so a real job's page could not be fetched from where the
   check runs.

The requirement this serves is the rig's standing one: a check must be able to
run where it is needed, and must fail a page only for a defect it names.

## What changes

- **Every expectation is derived from the job served.** The check imports the
  page's own `static/tdplot.js` (the file the page was served) and uses the
  page's own rules: `siPrefix` on the run's span for the time axis and on its
  band for the frequency axis, `axisExponent` on the run's own trace values for
  the tick factor (bracketed over the page's 5% range padding), the probe count
  and the manifest's spectra for the line and swatch counts, and the manifest's
  `frames` for whether there is a movie at all, its count and its file-name
  width. No one-run-shape constant is left: a 1-probe fs/THz job and a 2-probe
  ns/GHz job are both checked as strictly as the fixture.

- **A real job can be mirrored into that layout.** `scripts/tdcheck_fetch.py`
  fetches the job page, its status, `spectrum.json`, `trace.bin` and every frame
  the manifest names, plus the `/static` assets the page loads (following its
  module imports), into `TD_DIR`. It is run where the app answers — the Jetson —
  and the directory is copied back; `scripts/run_tdcheck.sh --dir DIR` then runs
  the check against it without writing to it. A job named by a prefix that
  matches two runs is refused, not guessed; a fetch that does not answer 200
  aborts naming the URL and status, and the runner refuses a mirror whose
  `job.html`, `status.json`, `files/spectrum.json` or `static/tdplot.js` is
  missing — an incomplete mirror must be named as such, never read as a page
  defect. The check itself also verifies that the frames on disk are exactly the
  ones the manifest names.

- **The movie canvas is read without waiting on an animation frame.** The slider
  is set and its `input` event fired inside one `page.evaluate`; the canvas read
  is a single `toDataURL` after `addInitScript` has forced
  `preserveDrawingBuffer: true` on every WebGL context, and Play is a click
  dispatched in-page. The page's own loop counts its frames on `window`, and the
  check waits for one more of those before reading, so it never races the loop;
  a loop that draws nothing for 90 s is recorded as a FAIL naming the stalled
  loop, not a hang.

## Impact

- `dockerfiles/rendercheck/tdcheck.mjs` — derivations, capture, the movie
  section's strictness kept (caption, spacing, envelope verdict, two drawn
  frames differing, play, the all-zero control), plus a job with no movie.
- `scripts/run_tdcheck.sh` — `--dir` for a mirror (never written to), `--rev`
  for an older viewer, the same fixture path and the same bounded container.
- `scripts/tdcheck_fetch.py` (new) — the mirror; `tests/test_tdcheck_fetch.py`
  (new) — its three rules without network: which job a prefix names (never a
  guess), which frame names the manifest implies, and what counts as a request
  (a JS string that looks like a path is not one).
- `scripts/tdcheck_fixture.py` — the fixture asserts its own window, since the
  check no longer carries it.

Nothing about the page or the solver changes. The check's verdicts are not
relaxed: every one of the checks the lead saw FAIL now PASSes because it is
computed from the job, and each check that could still catch a real defect
(line counts, units, factor, caption, frames, play, the all-zero control) is
still there.
