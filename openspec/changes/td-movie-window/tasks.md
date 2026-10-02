# Tasks — td-movie-window

> Archive order: this change MODIFIES a requirement that the open change
> `time-domain-fdtd` introduces, and `openspec/specs/time-domain/` does not
> exist until that change archives. Archive `time-domain-fdtd` first, then this
> one (`openspec validate` reports this as INFO, and archive would refuse it).

## 1. Solver: the window and the frame count

- [x] 1.1 `dockerfiles/maxwell_solver/fdtd.py`: `frame_plan` takes `first_step`
      and `last_step` (default: the whole run) and spreads the frames over that
      range, keeping its old steps exactly when the range is the whole run.
      `memory_estimate`/`estimate_run` take `max_frames` instead of assuming
      `MAX_FRAMES`, and a `MAX_MOVIE_FRAMES = 120` cap is declared.
- [x] 1.2 `dockerfiles/maxwell_solver/fdtd_layout.py`: `movie_window(request,
      est)` turns the request's window in seconds into step indices and returns
      the frame plan, refusing (in words, naming the value and the run's own
      numbers) a count above the cap, an end at or before the start, a window
      outside the run, and more frames than the window has steps. Both the form
      and the worker call this one function, so their refusals cannot drift.
- [x] 1.3 `fdtd_layout.estimate` passes the requested frame count to
      `estimate_run`, so the cap check counts the frames the run will record.

## 2. Request and worker

- [x] 2.1 `worker/td_solve.py`: `REQUEST_KEYS` gains `movie_start_s`,
      `movie_end_s`, `frame_count`; `request_from` accepts an older job that
      has none of them (whole run, default count), and only those three may be
      absent. The plan comes from `fdtd_layout.movie_window`.
- [x] 2.2 `FRAME_NAME` widens to three digits; the manifest's `frames` records
      the digits, the frame times in seconds and the source frequency, so the
      page needs no constants.

## 3. Every name list accepts every name the solver writes

- [x] 3.1 `web/worker_api.py` (upload allow-list) and `web/app.py` (file route)
      accept `frame_\d{2,3}\.bin`; `worker/mse_worker.py`'s `_is_frame_file`
      accepts two or three digits. `frame_01..60` still load.
- [x] 3.2 The near-miss tests move `frame_001.bin` from the refused list to the
      accepted list and keep a genuine near miss in its place.

## 4. Form and page

- [x] 4.1 `web/td_form.py` + `web/templates/index.html`: movie start, movie
      end (fs, blank = whole run) and frames (blank = 60, max 120). The form
      refuses through `movie_window`, so the messages are the solver's own.
- [x] 4.2 `web/static/viewer.js`: the digits and the count come from the
      manifest; the caption states the spacing in fs and in cycles of the
      source's centre frequency, and says the movie shows the envelope when the
      spacing is more than half a cycle.

## 5. Render check

- [x] 5.1 `dockerfiles/rendercheck/tdcheck.mjs` (+ `scripts/tdcheck_fixture.py`):
      probes, units, frame count and name width come from the manifest, and the
      check passes against a 120-frame windowed fixture.

## 6. Proof

- [x] 6.1 Red-before-fix tests: the window's steps (fails on today's
      `frame_plan`), the 120-frame names through the allow-list and the route
      (fail on today's `frame_\d{2}`), the estimate growing with the count, the
      refusals' messages, and the old no-field form's exact steps.
- [x] 6.2 Mutants: revert each fix in turn and confirm the matching test is the
      one that goes red.
- [x] 6.3 The affected suites at tier 1 (tiny, `2` threads, `nice -n 19`).
