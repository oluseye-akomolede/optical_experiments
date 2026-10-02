# time-domain

## ADDED Requirements

### Requirement: A time-domain request SHALL carry an optional movie window and frame count

The `td` request SHALL carry `movie_start_s`, `movie_end_s` (seconds, or null
for the whole run) and `frame_count` (a whole number, or null for the default
60). A frame count above the solver's cap of 120, a window end at or before its
start, a window reaching outside the run, or a frame count larger than the
number of time steps in the window SHALL be refused with a message that names
the value, the run's own numbers, and what to change — never silently narrowed.
A job written before these keys existed SHALL still solve: an absent window key
means the whole run and an absent frame count means the default, exactly as
that job solved before.

#### Scenario: the form's refusals, in words
- **WHEN** a `td` form is posted with a movie ending before it starts, with a
  window past the end of the run, or with 200 frames
- **THEN** the upload is refused with a message naming that value and the run's
  own duration, frame cap or step count, and no job row is created

#### Scenario: a window with fewer steps than frames is refused
- **WHEN** a 20 fs window on a run whose dt is 0.5 fs is asked for 60 frames
- **THEN** the refusal says the window holds 40 time steps and 60 frames were
  asked for

#### Scenario: a job from before the window still solves
- **WHEN** a stored `tds` request has no `movie_start_s`, `movie_end_s` or
  `frame_count` key
- **THEN** the worker solves it over the whole run with 60 frames, as it did
  before this change

### Requirement: The movie's frames SHALL spread over its window and end at its end

The recorded steps SHALL be spread evenly over the requested window, with the
first step at or after the window's start and the last at or before its end,
exactly as the whole-run movie spreads over the run and ends on its final step.
The last frame SHALL be the last step of the window, and the frame count SHALL
be the requested one unless the window holds fewer steps.

#### Scenario: a windowed plan ends at the window's end
- **WHEN** a 1,000-step run records 12 frames over steps 400 to 800
- **THEN** the frames are 12 steps, all between 400 and 800, the last is 800,
  and no two are the same step

#### Scenario: the whole-run default is today's frames exactly
- **WHEN** a movie with no window and no frame count is planned
- **THEN** its steps are the ones `frame_plan` produced before this change, in
  the same order, for every step count

### Requirement: The movie's caption SHALL state the frame spacing and whether it can show the wave

The caption SHALL state the time between consecutive frames in fs (or the
unit that fits) AND in cycles of the source's centre frequency, and when that
spacing is more than half a cycle it SHALL say plainly that the video shows the
envelope rather than the oscillation and that a shorter window shows the wave.
The spacing SHALL come from the job's own recorded frame times, not from a
constant, since a spread integer step list can differ by one step between
pairs.

#### Scenario: a coarse movie says so
- **WHEN** a job's frames sit 3 cycles of its source apart
- **THEN** the caption gives the spacing in fs and in cycles and says the video
  shows the envelope, not the oscillation

#### Scenario: a fine movie does not
- **WHEN** a job's frames sit a tenth of a cycle apart
- **THEN** the caption gives the same two numbers and claims nothing about an
  envelope

### Requirement: Every frame name the solver can write SHALL be accepted by every list of result names

A movie frame SHALL be named `frame_NNN.bin`, one per frame, 1-based, with
enough digits for the solver's 120-frame cap. The worker's upload list, the
app's file route and the worker's own classifier SHALL accept every name the
solver writes, and SHALL still accept a two-digit name, so a job recorded
before this change keeps loading. Near misses that are still refused SHALL
include a one-digit number and a four-digit one. The manifest SHALL state how
many digits its own names use, so a reader never guesses.

#### Scenario: a 120-frame job's names all pass
- **WHEN** a movie has 120 frames
- **THEN** `frame_100.bin`, `frame_120.bin` and `frame_001.bin` are all
  accepted by the worker's upload list and by the app's file route, and all of
  them are classified as movie frames by the worker

#### Scenario: an older job's names still pass
- **WHEN** a job recorded before this change has `frame_01.bin`..`frame_60.bin`
- **THEN** every one of them is accepted and served, and the page asks for
  two-digit names because that job's manifest says two

#### Scenario: a near miss is still refused
- **WHEN** the name is `frame_1.bin`, `frame_0001.bin`, `frame_01.vtu` or
  carries a path separator
- **THEN** the upload list, the file route and the worker's classifier all
  refuse it

### Requirement: The memory estimate SHALL count the frame count the run will record

The memory estimate behind the form's cap check and the worker's own check
SHALL count the requested number of frames (and their float64 |E| working
arrays), not a fixed 60, so a run that would not fit at the requested frame
count is refused before it is queued.

#### Scenario: the estimate grows with the frame count
- **WHEN** the same run is estimated with 60 and with 120 frames
- **THEN** the second estimate is larger by about 60 frames' worth of the
  movie's bytes, and both match the solver's own measured peak

### Requirement: The job page SHALL take its movie's shape from the manifest

The page SHALL read the frame count, the digits of the file names and the frame
times from the job's manifest, so a windowed or 120-frame job plays without any
change to the page, and it SHALL state the loading cost it imposes.

#### Scenario: a windowed 120-frame job plays
- **WHEN** the owner opens a finished job whose movie has 120 frames over a
  window in the middle of the run
- **THEN** the slider spans 120 frames, each frame loads, and the caption's
  time is the frame's own time in the run, not its index times the spacing

### Requirement: The render check SHALL read its numbers from the manifest

The headless render check (`tdcheck.mjs`) SHALL take the probe count, the
units of the object and the frame count from the job's manifest rather than
from constants in the script, so it can run against a windowed 120-frame job.

#### Scenario: the check runs on a 120-frame windowed job
- **WHEN** the render check is pointed at a fixture whose manifest is a
  120-frame windowed movie
- **THEN** it checks every frame the manifest names, captions the last frame
  with the manifest's own count, and passes

## MODIFIED Requirements

### Requirement: The solver SHALL record a bounded field movie of |E|

On request the solver SHALL record |E| at up to the requested number of steps
(default 60, at most 120) spread evenly over the requested window, the last
being the window's final step (the run's final step when no window is asked
for). The stride SHALL be the smallest whole number that brings each axis to at
most 64 samples. Each frame SHALL be taken after the step's E update, source
and conductors, so it shows no field inside a conductor. The three components
SHALL be combined where they are stored, without interpolation, and squared in
float64 so small fields keep their digits. The frames SHALL come back as host
float32 whatever array library ran the solve, and the memory estimate SHALL
count the requested number of frames.

#### Scenario: the frames match the probes
- **WHEN** a run records frames and probes at the same nodes
- **THEN** each frame's value at a probe node is the magnitude of that probe's
  three components at the frame's step, to float32 precision

#### Scenario: a long run is capped
- **WHEN** a 130 × 64 × 65-node grid runs for 1,000 steps with no window and no
  frame count
- **THEN** the movie has 60 frames ending on step 999, with strides (3, 1, 2)
  and frames of 44 × 64 × 33 samples

#### Scenario: a bad frame plan is refused
- **WHEN** a frame step is repeated or outside the run, or a stride is below 1
- **THEN** the solver refuses before stepping, naming the bad value

#### Scenario: a window inside the run
- **WHEN** the same grid records frames over a window that starts after step 0
- **THEN** every recorded step lies inside the window, the last is the window's
  last step, and the stride is unchanged from the whole-run movie
