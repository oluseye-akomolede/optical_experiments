# tdcheck: an old movie's frame names, and a cap that names itself

## Why

Running the served render check against real production jobs found two ways the
harness, not the page, fails a correct job -- both of them reading as page
defects:

* `scripts/tdcheck_fetch.py` reads `frames['digits']` at two sites. Movies
  recorded before three-digit names existed have no `digits` key at all and name
  their files two digits wide, and the page's viewer defaults to two
  (`viewer.js`). The fetch raises `KeyError` on such a job, so the mirror is
  never built -- "could not make a mirror" looks like "nothing to check".
* `scripts/run_tdcheck.sh` cuts the whole check at a fixed `timeout 600`. The
  check draws every movie frame through software WebGL in a 2-CPU container, so
  its cost follows the movie's length: a 120-frame open-boundary mirror of a
  real job needed ~740 s on this loaded VM and was killed at rc=124 after 45
  PASS, with Node's last words a stack trace about a closed page. A cut run read
  as a broken page.

## What changes

* The fetch derives the frame-name width from `frames.digits`, falling back to
  the page's own default of two digits when the key is absent.
* The runner derives its cap from the movie: `180 + 12 x frames` seconds, read
  from the mirror's own manifest (the solver's 120-frame maximum when there is
  no manifest to read), overridable with `TDCHECK_TIMEOUT`, and printed. 12 s a
  frame is ~2x the slowest per-frame cost measured here (740 s / 120 = 6.2 s,
  against 2.2 s on an idle VM).
* A run the cap cuts says so: the runner prints `harness timed out after Ns (M
  frames; ...)` and exits 124, and `tdcheck.mjs` prints a plain `HARNESS TIMED
  OUT` line and exits 124 on the signal that stops it, instead of exiting with a
  stack trace about a closed page.

## Impact

`scripts/tdcheck_fetch.py`, `scripts/run_tdcheck.sh`,
`dockerfiles/rendercheck/tdcheck.mjs`, and their tests. No page or solver code
changes. A mirror of a pre-three-digit job now builds; a long movie's run is
bounded by its own length rather than cut off; a cut run is legible.

## Out of scope

The check's own verdicts, the mirror layout, and the container's resources
(`--cpus=2 --memory=2g`) -- unchanged. Making the check itself faster (it is the
software renderer that costs 6 s a frame) is not part of this change.
