# A time-domain movie window, and up to 120 frames

## Why

Today a `td` job's field movie is always 60 frames spread over the whole run
(`fdtd.frame_plan`). A run long enough to let a resonance build up is hundreds
of light cycles, so consecutive frames sit several cycles apart and the video
shows the envelope, not the oscillation. The owner asked for a window and more
frames (18:25Z, chose options 1 and 2): say *when* the movie looks, and allow
up to 120 frames.

There is also a silent-drop defect to close on the way: the frame names reach
three digits at 100+, and the upload allow-list (`web/worker_api.py`) and the
file route (`web/app.py`) accept only `frame_\d{2}.bin`, so a 100-frame movie
would lose frames 100+ with no error anywhere.

## What changes

- **A movie window on the `td` form.** Optionally, the movie starts and ends at
  a chosen time (fs); by default it is the whole run. Frames per movie can be
  set, default 60, at most 120. A window that ends at or before it starts,
  reaches outside the run, or asks for more frames than the window has time
  steps is refused in plain words before anything is queued.
- **Frames spread over the window, the last one at its end** — the same rule as
  today, which is what the whole-run default reduces to. A job or a form
  without the new fields produces exactly today's frames.
- **An honest caption.** The caption states the spacing between frames in fs
  *and* in cycles of the source's centre frequency, and says plainly when the
  spacing is more than half a cycle: this video shows the envelope, not the
  wave, and a shorter window shows the oscillation.
- **Three-digit frame names**, accepted by every list that accepts two-digit
  ones, so a 120-frame job's files all upload and all load; an older job's
  `frame_01..60` keep loading (the manifest says how many digits its names
  have).
- **The memory estimate follows the requested frame count**, so a 120-frame
  movie is refused by the cap where it would not fit, before it is queued.
- **The viewer takes the count and the name width from the manifest**, so a
  120-frame job plays without a code change, and the caption carries the
  spacing.
- **The render check folds in its owed generalisation** (`tdcheck.mjs`: probes,
  units and frame count read from the manifest rather than hardcoded), so it
  runs against a windowed 120-frame job.

## What this does not change

- The frame *sampling* rule (the per-axis stride to at most 64 samples per
  axis) and the |E| quantity.
- The default: a `td` job with no window and no frame count records 60 frames
  over the whole run, with the same steps as before this change.
- The 30-minute run limit and the other caps.
