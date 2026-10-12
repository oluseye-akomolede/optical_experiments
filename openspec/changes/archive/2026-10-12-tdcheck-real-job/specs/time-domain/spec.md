# time-domain

## ADDED Requirements

### Requirement: The render check SHALL derive every expectation from the job it is served

`tdcheck.mjs` SHALL compute what it expects of the page from the job it is
served and from the page's own drawing rules, and SHALL carry no constant that
holds for one run's shape only. The probe count and the line and swatch counts
SHALL come from the manifest's `probes`; the time and frequency axis units SHALL
come from the page's own `siPrefix` applied to the run's span and band; the
trace's tick factor SHALL come from the page's own `axisExponent` applied to the
values the check decodes with the page's own `decodeTraces` (accepting both ends
of the page's 5% range padding); whether the run recorded a movie, how many
frames it has and the width of its file names SHALL come from the manifest's
`frames`. The page's plotting module SHALL be imported from the job directory,
so the rules applied are the ones the page was served.

Why: run against a real production job (1 probe, 525 steps at
`dt_s 3.8e-17`, band 250–650 THz), five checks FAILed a correct page — a
fabricated 2-probe line count, a fabricated 2-swatch count, `×10⁻¹³` against the
job's `1.43e-16` peak, `time (ps|ns)` against the page's `fs`, and
`frequency (GHz)` against the page's `THz`. A check that fails a correct page
cannot be pointed at production, which is where it must run.

#### Scenario: a one-probe femtosecond/terahertz production job passes
- **WHEN** the check runs against a mirror of a real job whose manifest names 1
  probe, a femtosecond span, a terahertz band, a `1.43e-16` trace peak and a
  movie with 120 three-digit frames
- **THEN** the per-probe line and swatch counts are checked against 1, the trace
  factor against `×10⁻¹⁶`, the axes against `fs` and `THz`, and every check
  passes because the page is right

#### Scenario: a job with different units and more probes is checked just as strictly
- **WHEN** the check runs against a job whose span picks a different prefix, whose
  band picks a different prefix, or which names more than one probe
- **THEN** the expectations follow that job, and none of the production job's
  numbers is assumed

#### Scenario: a job that recorded no movie is not a failure
- **WHEN** the served manifest has no `frames`
- **THEN** the check verifies the page says "This run recorded no field movie."
  and reports the movie cases as not applicable, instead of checking frames a
  job never wrote

### Requirement: The check SHALL be runnable against a mirrored real job

The check SHALL read a `TD_DIR` laid out as a job page plus what it serves
(`job.html`, `status.json`, `files/spectrum.json`, `files/trace.bin`, the frames
the manifest names, `static/`), and `scripts/tdcheck_fetch.py` SHALL mirror a
real job into that layout from the running app. A job SHALL be named by a full
id or by a prefix that matches exactly one job; an ambiguous or unknown prefix
SHALL be an error, never a guess about which run was mirrored. A fetch that does
not answer 200 SHALL abort naming the URL and the status, and
`scripts/run_tdcheck.sh --dir DIR` SHALL refuse a mirror missing any file the
check reads, so an incomplete mirror is named as such and never read as a page
defect. The runner SHALL NOT write to a mirror it is pointed at.

#### Scenario: a real job is mirrored and checked
- **WHEN** `scripts/tdcheck_fetch.py` is run against the app for a finished
  time-domain job and `scripts/run_tdcheck.sh --dir` is pointed at the result
- **THEN** the check runs the real page with the real assets and result files,
  and reaches an end with every check PASS or a named page defect

#### Scenario: a prefix that names two runs is refused
- **WHEN** the job argument matches two jobs
- **THEN** the fetch exits with an error naming how many it matched, and nothing
  is written

#### Scenario: a mirror with a hole is named, not judged as a page defect
- **WHEN** a mirror is missing a frame the manifest names, or a file the check
  reads
- **THEN** the run stops naming the missing file, and the page is not reported as
  broken

### Requirement: The check SHALL read the movie canvas without waiting on an animation frame

The check SHALL move the movie without any Playwright actionability wait: the
slider's value SHALL be set and its `input` event dispatched inside one
`page.evaluate`, and Play SHALL be a click dispatched in-page. It SHALL read the
movie canvas with a `toDataURL` read of the page's own canvas after forcing
`preserveDrawingBuffer: true` on every WebGL context from the page's first
context, and SHALL count the page's own animation frames so that it reads only
after the loop has drawn a further frame. A render loop that draws no frame
within 90 s SHALL be recorded as a FAIL naming the stalled loop, and SHALL NOT
hang the run.

Why: measured — against a real 58x53x48 movie the run hung twice, at
`locator('#td-scene').screenshot()` and at `page.fill('#td-movie-frame')`
(`Timeout 30000ms exceeded`), because a loaded movie starves
`requestAnimationFrame` and both Playwright calls wait on one.

#### Scenario: a real movie is moved and read to the end of the run
- **WHEN** the check steps the slider, clicks Play, and reads two different
  frames of a real movie
- **THEN** each read is a canvas read that needed no animation frame of its own,
  the two frames differ, and the run reaches its end

#### Scenario: a stalled render loop is a named failure
- **WHEN** the page's loop draws no frame for 90 s
- **THEN** the check records a FAIL naming the loop that drew nothing, and the
  run ends instead of hanging
