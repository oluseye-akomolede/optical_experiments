# time-domain

## ADDED Requirements

### Requirement: The mirror SHALL name a movie's frames the way the page does

`scripts/tdcheck_fetch.py` SHALL derive a movie's frame file names from the
served manifest's own `frames.count` and `frames.digits`, and SHALL use the
page's own default width -- two digits, as `viewer.js` defaults -- when the
manifest carries no `digits` key, which is what a movie recorded before
three-digit names looks like. A mirror SHALL name exactly the files the page
will ask for: reading the key outright raises `KeyError` on such a job, so no
mirror is written and the run reports a harness fault.

#### Scenario: a pre-three-digit movie is mirrored

- **WHEN** the served manifest is `{"frames": {"count": 9}}`, with no `digits`
- **THEN** the fetch asks for `frame_01.bin` through `frame_09.bin`, the files
  the page's viewer asks for, and the mirror is complete

#### Scenario: a three-digit movie is unchanged

- **WHEN** the served manifest is `{"frames": {"count": 120, "digits": 3}}`
- **THEN** the fetch asks for `frame_001.bin` through `frame_120.bin`

### Requirement: The runner's cap SHALL follow the movie, and a cut run SHALL say so

`scripts/run_tdcheck.sh` SHALL cut the check at a cap derived from the movie's
own length -- `180 + 12 x frames` seconds, where `frames` is the mirror
manifest's `frames.count` (the solver's own 120-frame maximum when there is no
manifest to read) -- SHALL print the cap it chose, and SHALL let
`TDCHECK_TIMEOUT` override it. A fixed cap is not enough: the check draws every
frame through the container's software renderer, so a correct 120-frame
open-boundary mirror of a real production job outlived a fixed 600 s on this
loaded VM. A run the cap cuts SHALL be reported as a cut run, never as a page
defect: the runner SHALL print `harness timed out after <cap>s (<frames> frames;
...)` and exit 124, the status `timeout(1)` itself reports, and `tdcheck.mjs`
SHALL print one plain `HARNESS TIMED OUT` line and exit 124 when the signal that
stops it arrives, in place of a stack trace about a closed page.

#### Scenario: a long movie gets a cap of its own

- **WHEN** the runner is pointed at a mirror whose manifest names 120 frames
- **THEN** it prints `cap: 1620s for 120 frames` and cuts the run there

#### Scenario: a manifest that will not read falls back to the solver's maximum

- **WHEN** the mirror holds no readable `files/spectrum.json`
- **THEN** the cap is the 120-frame one, and the run is bounded rather than
  unbounded or cut short

#### Scenario: an operator can raise the cap

- **WHEN** `TDCHECK_TIMEOUT=1234` is set
- **THEN** the runner cuts at 1234 s however long the movie is, and prints that
  number as the cap

#### Scenario: a run the cap cuts names itself and is not a page defect

- **WHEN** the container outlives the cap and is stopped
- **THEN** the runner prints the plain `harness timed out after ...` line naming
  the cap and the frame count, exits 124, and nothing in the output presents the
  run as a failed page
