# paper1-nanowire-purcell — tasks

The lead's four steps (queue row, 2026-10-15). One commit each, local only; the lead merges.

## 0. Normalisation — what the paper measures, quoted

- [x] Paper identified from the arXiv page itself (title, authors, date, abstract verbatim).
      Abstract quoted in `proposal.md`.
- [x] Lumerical's own definition sought: the Ansys KB page for the dipole source answers
      **HTTP 403** to the fetch tool. Recorded in `proposal.md` as *not read*, so nothing is
      quoted from it, and the normalisation is derived in the spec delta instead.
- [x] The paper's geometry flagged as unverified: the fetched body text (wire 1375 nm by
      420 nm, 12 nm SiO2, dot 30 nm below the top) describes a different design from the
      abstract's quasi-BIC of two coupled modes. Every dimension is an argument.
- [x] Decision: `F = P / P0`, `P0 = μ0 n ω⁴|p̃|²/(12 π C0)` at the dipole's own index.
- [x] `openspec validate paper1-nanowire-purcell`.

## 1. The measurement, in `fdtd.py`, with its controls

- [x] `record_drive=` on `simulate`; `source_cell_volume`, `delivered_power`,
      `dipole_moment`, `larmor_power`, `radiated_power`, `purcell_factor`, and the shared
      `_windowed_at` / `_steady_tail` the CW measurements read their tail through.
- [x] Tier-1 controls in `tests/test_fdtd_purcell.py` (16 collected when this line was
      written, 22 after the second sitting's band, parallel and tail-window fixtures below,
      and 22 still after the third sitting's corrections — it fixed fixtures rather than
      adding them): the recorded
      drive against `Source.value` and the no-op check, the source cell volume on a graded
      axis (both
      spellings) with its guards, the closed-box energy identity with and without the
      trapezoid term, the source-against-field ladder, the windowed mean against the time
      domain's own, the moment against `ε0 V / ω`, the volume-law arithmetic, the
      homogeneous vacuum run on both instruments (delivered vs flux, box move, mesh sweep),
      the mirror law against its own pattern integral (both spellings), the PEC mirror sweep,
      the flux box's refusal inside the absorber, and the boundary layer that makes a flux
      surface the wrong instrument over a conductor.
- [x] The floors and tolerances the spec delta states, filled in from the measured run.
- [x] The measurements that changed the design, recorded in the spec delta and the module
      docstrings: the tail window (the record's own length is not part of the answer), the
      ledger identity (the old one-sided spelling missed it by 40x with the wrong sign, so
      the earlier rejection of the delivered power was a wrong-formula artifact), the flux
      instrument's boundary layer (F = 1.5882 against a law of 1.1629 at gap 7, and 346.4668
      against 1.8945 at gap 2, where the delivered power reads +2.8 per cent), and the vacuum
      tolerances (`pytest.approx`'s own absolute default is larger than everything in this
      module, so every assertion here is relative-only).
- [x] `tests/test_fdtd.py` still green after the refactor: 58 passed in 6.7 s (tier 1,
      `nice -n 19`, one thread), alongside `tests/test_fdtd_purcell.py`'s 16 in 85 s.
- [x] The lead's step-1 acceptance items, added in the second sitting (all tier 1, `nice -n 19`,
      one thread, one process at a time):
  - the **band arm**: `delivered_power_spectrum`, `purcell_spectrum` and the shared `_band`, so
    one pulse gives a whole band. The record-placement condition is measured, not argued:
    boxcar over a 10-sigma vacuum record reads 1.00143–1.00657 across the band; a Hann over the
    same record 0.98530–1.16101; the same pulse in a 20-sigma record reads −65.30 … −8.09 and
    in a 30-sigma record −94.34 … −8.21, negative, because the drive's delay is fixed at 5
    sigma and the record is what a caller moves.
  - the **cross-arm check**: one vacuum pulse's own bin against one CW run at the same
    frequency — 1.00443 against 1.00525, a ratio of 0.99919 — which is what makes the band arm
    the same measurement rather than a second one.
  - the **sweep in omega**: three carriers at a fixed 19-cell envelope, the delivered power
    following the sampled bins' own omega² (2.569484e-24, 1.716588e-24, 9.412558e-25 W at
    0.015751, 0.012601 and 0.009450 GHz: exponent 1.966 over 1.667x, not flat and not omega⁴).
  - the **parallel dipole**, the paper's own geometry: `image_dipole_factor_parallel` and the
    pattern integral it solves, the two laws pinned against each other at the limits neither
    closed form can evaluate (F(0) = 2 perpendicular, 0 parallel), the sweep F = 0.58549 …
    1.40227 over gaps 3 to 12 with the residual where every other fixture's is (+35.5 per cent
    of the law at gap 3, −1.0 at gap 12), and the discrimination from the perpendicular law
    carried as a property of the two laws (separations 1.342 and 0.933 at gaps 3 and 4 against
    the 0.34 the arm demands), so the arm cannot go vacuous unnoticed.
  - the **tail window stated as this arm's convention**, not the paper's, in the delta: the
    paper's number is a steady state at one wavelength and the window is this arm's spelling
    of one.
  - the **band arm's two edges**, on a 400-step record read by hand rather than by a
    simulation: a band holding no bin raises instead of returning an empty curve, and the DC
    bin returns moment/reference/F = 0 with the delivered power left as the record measured it
    (4.427528e-12 W on a drive carrying a DC offset), so neither guard is one nothing calls.
  - the **tail window made a fixture rather than a convention** (the lead's (A), answered where
    it can go red): a 12-period vacuum record and a 32-period one read F = 1.00616 and 1.00569
    over their last 8 periods — 0.0005 apart — while their WHOLE records read 1.70781 and
    1.07865, 0.63 apart and still falling with length. That is the record-length drift the
    probes found, and the arm that catches it is the window, not the simulator. The third
    sitting made the 16-cell-per-wavelength half of it a measurement too, and it moved the
    claim: a 1343-step record at that mesh reads 1.00937 windowed against 1.01854 whole, but a
    296-step record is only 10.7 periods there, so its 8-period window still reaches back into
    the 5-period ramp and BOTH its windows read high (1.33361 and 2.10286). The 1.088 / 1.029
    / 1.008–1.013 this line and four docstrings used to quote were not any fixture's numbers.
  - the **third sitting's corrections** (`edits1.log`, 4 passed): every number in the delta and
    the docstrings diffed against the run logs that produced them, which caught three that no
    instrument here makes — the 16 ppw drift figures above, an omega⁴ prediction written as
    7.718 where the bins say 7.716, and a "one cell closer is 1.2308" in the perpendicular
    fixture's docstring, where the law one cell either side of gap 7 reads 1.3040 and 1.0501,
    0.108 and 0.146 away against the 0.05 bound the fixture actually asserts. Two numbers that
    had no instrument at all now have one: the cell-volume fixture prints its three components'
    span (1.48524x, the ratio the delta quotes) and the old-formula comparison is stated with
    the mesh ladder it was measured on.
- [x] 22 collected in `tests/test_fdtd_purcell.py` (the second sitting's own earlier line said
      21 for a 20-def file: a count is a claim, and that one was one short; the third sitting
      re-counted 22 defs and 22 collected, and the whole tier-1 suite — 1379 passed, 14
      skipped — ran green on this tree in 653.57 s). The file's own last run on the committed
      tree: `22 passed in 148.25s`; on this branch rebased onto main `9a1bc55` the same file
      reads `22 passed in 148.21s`, so the step-2 work below sits on a file that is green on
      the tree it rides. The two shas the third sitting handed over are
      `bc833b1` (the solver and its fixtures) and `eb00ea1` (this delta and this file); the
      1.48524 above was written 1.48525 in `eb00ea1` and is corrected here rather than in the
      sha the lead has already read.

## 2. Bench: the paper's geometry

- [x] `scripts/paper1_nanowire.py`: the stack (a gold slab, the SiO2 buffer, a hexagonal InP
      prism), a point dipole on or off the axis, `--tiny` for tier 1, `--estimate` for the
      ladder, and the two measurement arms (CW, and pulse through `purcell_spectrum`). Every
      dimension is an argument -- the fetched paper's design does not match its own abstract,
      so nothing about it is asserted -- and `--hex` says what `--diameter` measures across.
- [x] The scene is the product's own machinery, not a second layout: `fdtd_layout.plan` /
      `layout` (the 16-cell open pad, the thin-layer rule, the Rakić gold's skin-depth
      refinement) and `fdtd.estimate_run` against the caps the web form enforces. The run
      prints that estimate before it starts and refuses over a cap; `refusals` is one function
      for both entry points, so the tier-1 scene cannot be one the Jetson arm would refuse.
- [x] The grid and the ladder, measured (`--estimate --spectrum --xp numpy`; the seconds are
      `fdtd_rate.py`'s cupy rates, which is why `main` warns when `--xp numpy` is asked for):

      dx(nm)  pad(nm)      cells    steps        s    GiB   nm/bin  verdict
       20.0     400    2,005,680   11,550    568.2   0.24    27.66  fits
       20.0     600    2,723,560   11,550    771.6   0.34    27.66  fits
       15.0     400    2,879,392   11,550    815.7   0.35    27.66  fits
       15.0     600    4,120,956   11,550   1167.5   0.51    27.66  fits
       12.0     400    3,973,800   11,550   1125.8   0.48    27.66  fits
       12.0     600    5,994,352   11,550   1698.2   0.73    27.66  fits
       10.0     400    5,532,780   11,550   1567.4   0.66    27.66  fits
       10.0     600    8,530,460        --       --     --       --  refused: over 8,000,000 cells

      Every rung's `steps` is 11,550 and every rung's `nm/bin` is 27.66: the timestep is the
      OXIDE's thin-layer cells, so `--dx` buys cells and seconds and never steps, and the
      record's resolution is the step count's rather than the cell size's. The last rung is
      the product's own cell cap refusing it (the message's own count: 8,530,460 cells against
      a limit of 8,000,000).
- [x] The resolution is printed, not implied: **27.66 nm per bin against the paper's 4 nm
      mode**. Resolving that mode takes 225 periods of record, i.e. 79,856 steps where
      `fdtd.MAX_STEPS` is 20,000 -- four times the product's own cap. Every run this envelope
      allows therefore measures the enhancement averaged over about 28 nm, and the run says so
      in its own output. This is the step-2 finding for the lead: the tier-2 envelope cannot
      resolve the paper's peak, only its neighbourhood.
- [x] The frame trap, measured and then made a control: `lay["nodes"]` are in the *owner*
      frame -- the origin at the parts' bounding-box min corner, as the live layout
      requirement and `fdtd_layout`'s module docstring both say -- so the pad is a negative-z
      overhang and the buffer sits at `z_oxide`, not at `pad + z_oxide`. The first spelling
      added the pad, looked 320 nm higher, and reported the tiny buffer as "40 nm over 2 cells
      of 20.00 nm" where the thin rule gives it cells of its own. Probe: `z[0] = -320 nm`,
      `z[-1] = 500 nm`, gold voxels 0..47.5 nm, `z_oxide` 40..80 nm, pad 16 cells. The tiny
      arm now reads the gold's voxel edges against the gold part's own bounds, so the same
      shift cannot pass quietly. At the ONE command's rung the same line reads `the SiO2 is
      12 nm over 4 cells of 3.00 nm` -- the paper's own buffer, resolved rather than asserted.
- [x] A record the hard-started CW drive has not settled, measured in one log
      (`/tmp/p1probe/settle.log`, the tiny vacuum scene at 900 nm):

      periods/tail   F            steps   note
      8 / 4          4.60760      1,068   the trap
      8 / 8          170.07210    1,068   same record, tail widened
      16 / 8         1.00104      2,136   settled (2x -- but not because of the ratio)
      32 / 8         1.00186      4,271   settled

      The settling is in PERIODS, not in the tail's share of the record: 8/4 and 16/8 are the
      same ratio and only the second is settled. So `report` warns below `SETTLED_PERIODS =
      16`, and the script's own default is 32. (The first spelling of the guard tested the
      ratio and the tier-1 fixture that runs 8/4 -- a 2x record -- went red, which is how the
      mistake was caught.)
- [x] `--tiny`: 15 controls, every one of them a claim that can go red (two of them are seen
      red on purpose in the tier-1 file): the prism mesh closed and its volume the hexagon's,
      the product's own clock the tier-2 timeout, the buffer keeping cells of its own and
      inside the thin rule's cap and finer than the coarse cell, the product's own
      `check_axis`, the gold a region and at the part's own z and clear of the absorber, the
      source in InP and clear of both the facet and the absorber, F = 1 in vacuum through
      the script's own path (band arm when `--spectrum`, the tail arm when `--ringdown`),
      the drive ending where its own envelope's algebra says it does, and the two readings
      of one record refusing to share it. The scene is deliberately small --
      20 nm cells, a 100 nm wire -- and exercises the same code path, not the same physics.
- [x] `tests/test_paper1_nanowire.py`, the step-2 file: 8 tests, `8 passed in 184.44s` (task
      `bvvw9vj3u`, exit 0; tier 1, `nice -n 19`, one worker, `(ulimit -v 4194304; ...)`). That
      run is on this branch rebased onto main `9a1bc55` and on exactly the two files committed
      below -- the commit adds the change docs only, not a line of the script or the test: the
      tiny scene's controls with their count pinned (13 of them at that commit; 15 after
      section 3a), the band arm through the same path, the facet
      control seen red (`--qd-below 25`), the prism control seen red (a mesh with a face
      removed), the three refusals/warnings -- a record shorter than its tail, a record over
      `fdtd.MAX_STEPS` (4,000 periods, through the same `refusals` the Jetson arm uses), and a
      record the drive has not settled -- and the ladder saying so when every rung is over a
      cap instead of reading the last rung it tried. The file stands at 15 tests after section
      3a: `14 passed in 308.58s` at step 2a (task `bw74qywga`), and the three ladder tests
      re-run green after 2b (section 3a's last bullet) before the full file was run again.
- [x] Bench scenarios stay here, not in the spec, until the lead's Orin run passes. Step 2
      changes nothing in the product: `fdtd.py`, `fdtd_grid.py` and `fdtd_layout.py` are read,
      not edited, and the script lives in `scripts/`.
- [ ] The dielectric as a function: still the deferred item below. Step 2's source sits in
      InP with a facet ~2 cells away, so the scalar `eps_r` the arm passes is the same
      approximation the delta describes; nothing here claims otherwise.

## 3. The Orin command

- [ ] The ONE tier-2 run for step 2, with its own `estimate_run` block printed before it
      starts (5,532,780 cells, 11,550 steps, 1,567.4 s, 0.66 GiB, 27.66 nm per bin; its margin
      under the 1,800 s timeout is 13 per cent). That estimate is `cells * steps /
      (RATE_SAFETY * rate)`, i.e. the step loop alone: `RATE_SAFETY = 0.9` makes it err long
      there, while the voxeliser and the frame recording are not in it at all. This is that
      block from the real path at this exact rung (a local build-and-estimate probe, no solve;
      every line below is the probe's own output):

          args      --spectrum --hex corners --dx 10 --mirror-pad 400
          grid      165 x 166 x 202  = 5,532,780 cells
          spacings  x, y 10.000 nm uniform;  z 3.000 nm finest, 10.000 nm coarsest
          z-axis    the product's own check_axis says: no problems
          thin      185 z cells below the coarse cell, finest 3.000 nm; the SiO2 is 12 nm over 4 cells of 3.00 nm
          stack     gold 150 nm = 5.5 skin depths (skin depth 27.2 nm at 333.1 THz), SiO2 12 nm, InP 1375 nm on a 1220 nm mirror; the wire is 420 nm across corners
          timestep  8.458600e-18 s  (354.9 steps per optical period)
          record    11,550 steps = 32.5 periods, 1567.4 s estimated on the Jetson (dispersive rates)
          memory    0.66 GiB estimated  (metals ['Au'], no pec, open boundary)
          resolves  27.66 nm per bin at 900 nm; the paper's mode is 4 nm and would need 79,856 steps (the cap is 20,000)
          caps      OK

      The copy and the run:

      rsync -a --relative scripts/paper1_nanowire.py dockerfiles/maxwell_solver/ \
          jetson:~/mse-bench/paper1/
      ssh jetson 'cd ~/mse-bench/paper1 && ulimit -v 5000000; nice -n 10 timeout 30m \
          ~/mse-worker/venv/bin/python scripts/paper1_nanowire.py --xp cupy --spectrum \
          --dx 10 --mirror-pad 400'

      Two things the lead should weigh before it starts. (1) The bench's python is
      `~/mse-worker/venv/bin/python` (the Orin's cupy lives there; `~/mse-bench/venv` has
      none), and the copy must keep the repo's paths -- the script resolves its imports
      relative to its own file. (2) The Orin is not free: another rig's realtime demo is
      holding 2.8 GB of ~7 GB, and the estimate's 0.66 GiB is on top of the CUDA runtime. If
      the bench stays busy, the dx 12 rung (3,973,800 cells, 1,125.8 s, 0.48 GiB, 22 ppw in
      InP) buys margin at a coarser mesh.
- [x] **The run came back** (the lead, tier 2, `--spectrum --dx 12 --mirror-pad 400`, not the
      dx 10 block above): exit 0, 1,458.6 s wall against 1,125.8 s estimated, `peak F = 2.2839
      at 976.3 nm`, and six pairs of `NvMapMemAlloc ... error 12` at the start. 976.3 nm is
      not a resonance: it is the band's own top bin (the band is centred at 900 nm in
      frequency, so its wavelength edges are 836.8 and 976.4 nm), and the shape string came
      back uniform -- every bin read F ~ 2. The paper's line is 4 nm wide at 900 nm and this
      record's bins are 27.66 nm wide, which is the wall section 3a names; what it does not
      answer is where in the band the cavity sits, and that is the CW arm's question
      (section 4), because a monochromatic drive reads F at exactly its own wavelength
      instead of averaging it over a bin.
- [x] **The estimate was 29 per cent short, and structurally so.** `seconds = cells * steps /
      (RATE_SAFETY * rate)` is the step loop alone -- the voxeliser and the spectrum's own
      pass are not in it -- and `RATE_SAFETY = 0.9` times the *slowest* recorded dispersive
      row (45.30 Mcell-steps/s, over two 300-step cupy runs) already errs long. The gap is
      the machine: 3,973,800 cells x 11,550 steps over 1,458.6 s is 31.5 Mcell-steps/s, 69
      per cent of that slowest row, and the Orin was holding another rig's realtime demo
      (5,540 MB of 7,619 MB used, 469 MB free, when the re-run below was launched). The six
      `NvMapMemAlloc` failures are the same contention seen from the allocator; the arm's own
      estimate is 0.48 GiB, so nothing in this scene asks for the ~1 GB those errors name.
      The re-run keeps stderr beside stdout to read the exact lines and the memory at the
      moment they appear, rather than to guess the allocation from here.
- [x] What was added instead of a new constant: the arm prints its own ratio, `run 1458.6 s
      wall  (1.29 x the 1125.8 s estimate)`, so every logged run says whether the table's
      rate held on that machine. `RATE_SAFETY` is left alone -- one contended run cannot set
      a constant the next machine has to live with.
- [ ] The consequence for choosing rungs, measured rather than assumed: at 1.29x, a rung must
      be estimated under about 1,400 s to survive the 1,800 s tier-2 clock. dx 10 at pad 400
      (1,567.7 s) is therefore out, and dx 10 at pad 200 (903.2 s, 3,222,828 cells) is in --
      which matters because the paper's own mesh is "10 nm in all three co-ordinates".

## 3a. The tail arm (the lead's ruling, 2026-10-15)

The step-2 report named the wall: the record's own bins are 27.66 nm wide at dx 10, the paper's
line is 4 nm, and reaching it by lengthening the record is 79,856 steps against a 20,000 cap.
The lead's ruling is not to lengthen records past the cap, but to read the *decay*: the drive
is over partway through a pulse record, and the tail after it is one decaying sinusoid --
`a_dip(t) = c + exp(-t/tau) (a cos wt + b sin wt)` -- giving Q = pi f tau with a covariance
error off the model's own residual. If the fit cannot pin Q from the short record, the arm
reports it as unresolved with the bound it gives. Both outcomes are accepted; a silent
linewidth is not.

- [x] `--tail-periods` puts whole drive periods of record after the pulse is over, and
      `--ringdown` reads them. `arm_conflicts` keeps the two readings off one another's
      records: the band arm needs its drive centred in its own record (its estimator is a
      cross-spectral density, and the delta records the -94 dB that a mostly-ring-down record
      costs it), and the tail arm needs the tail the band arm cannot have.
- [x] The window is a reading of the pulse, not a retyped constant: `pulse_end_sigma()` is
      `5 + sqrt(-ln floor)` sigma off the product's own envelope, and `drive_done()` finds the
      same cut in the run's recorded drive. `--estimate`/`report` prints the planned tail
      before anything runs; the arm prints the measured one.
- [x] Tier 1, both directions (the fixture is `test_the_fitter_recovers_a_decay_and_refuses_to_invent_one`):
      synthetic tails with a known tau come back within a few of their own statistical errors
      (10 fs over a 30 fs window: -0.00002%; 5 fs: -0.0033%; 20 fs: +0.0009%; 0.5 s each), and
      a sinusoid with no decay in its window returns `resolved = False` with `at_edge = True`.
      The first spelling of the fit stopped on its own scan grid -- 0.4% to 2.6% off, a hundred
      times its own error -- and the fix is a coordinate descent (golden-section per axis)
      rather than a fixed number of linspace rounds.
- [x] The fit's **frequency window follows the drive's band**, not a fixed +-5 per cent of the
      drive's centre: a 200 nm band around 1000 nm reaches -9 and +11 per cent, so a mode out there
      was being fitted at the search's own edge and printed as a line. `band=(lo_hz, hi_hz)` is an
      optional argument (omitted, the window is the old one, so nothing else moved), a transposed
      band is refused rather than silently swapped, and `f_edge` travels in the fit's dict while
      the arm prints `AT THE EDGE` when it lands on one. Found while preparing the lead's pulse
      run, whose whole point is to read f off a tail. Tier 1 fixture
      `the fit searches the band it is given, not a fixed +-5 per cent`: a synthetic tail 10 per
      cent below the drive comes back at 299.79 THz against 299.79 sought through the band window,
      while the fixed window stops at 313.70 (0.943 of the drive) and flags itself -- the two
      readings differ, so the control cannot pass by both windows agreeing.
- [ ] **What the fit's error is not.** It is the model's, not the geometry's: a tail holding
      two decays still fits one of them, and a fast CPML transient fits as well as a cavity
      mode. The arm prints that caveat in its own output, and `--vacuum` is the same arm on the
      same scene with no cavity, so the two taus read side by side say which is which. Neither
      has been run on the Orin.
- [x] The estimate path grew with it: `--estimate` overrode `--dx`/`--mirror-pad` on every row
      it printed, so `--estimate --dx 5` was silent about dx 5 -- which reads like the rung the
      caller named had been priced and fitted, and is the first command step 3's finer-grid run
      needs. The caller's own rung is now priced too and marked in the verdict column (a refused
      one as well, since that is the row a reader most needs marked), and "the finest fitting
      rung" is the minimum-dx rung that fits rather than the last row in the table that did.

## 4. Step 3's shape (the lead, 2026-10-15)

- [ ] The owner's step 3 as scoped in the row: the Purcell spectrum 850-950 nm at H=1375, a
      short H sweep (1300/1375/1450) to show the peak moves, and a run at a finer grid to
      show the peak is converged rather than asserted. The spectrum is the band arm's -- one
      pulse record covers the octave -- and the sweep is `--height`.
- [ ] The finer-grid run, measured rather than assumed: the ladder's own rungs stop at dx 10
      (5,593,868 cells at pad 400, 1,567.7 s), and `--estimate --dx 5` prices 26,873,280
      cells, 3.4x over the 8,000,000 cap, so dx 5 is not reachable at this pad. The pair
      that is reachable is dx 20 -> dx 10 (2,037,156 -> 5,593,868 cells). Reaching dx 5
      means shrinking the mirror pad, which is the scene's own physical geometry (what the
      mirror and the CPML need), so it is the owner's decision rather than a cost knob to
      turn quietly.
- [ ] CW Purcell at 900 nm and a few neighbouring wavelengths (`--wavelength`), the CW arm's
      own reading of the peak value. Nothing here lengthens a record past the cap to buy
      linewidth. The rungs, priced with the measured 1.29x in hand: dx 12 pad 400 (1,127.5 s
      estimated, the rung the lead ran) fits the 1,800 s clock; dx 10 pad 400 (1,567.7 s) does
      not, and dx 10 pad 200 (903.2 s, 3,222,828 cells) does -- and dx 10 is the paper's own
      mesh, so the pad's convergence run is what buys the comparable rung.
- [x] **The reading superseded one the scene had.** `--hex corners` (corner-to-corner = the
      quoted number) made the flats 363.7 nm, 12 per cent narrow, and was the default until
      the lead's decision below. It stays runnable, because run 1 is the comparison point for
      the reading.
- [x] The dipole's placement is the paper's, in its own words: "The QD emitter placed
      on-axis at 30 nm below the top." (Figure 2a) and "we placed the QD at the second
      antinode from the top" -- on a wire whose in-material half wave is 130.8 nm, 30 nm is
      that second antinode, so the two sentences agree. The scene puts the dipole on the axis
      30 nm under the top facet, and `report` now prints the node it actually chose and the
      gap to what was asked (tiny scene: 57.11 nm delivered against 60 asked, inside one
      finest cell; 3.33 nm off-axis on a 60 nm wire, well inside the paper's own +-100 nm
      displacement tolerance). The coarse grid misses the ask by more, and the arm says so:
      on the dx 20 Orin run the line reads "z 1497.01 nm, 39.99 nm below the top facet at
      1537.00 nm (--qd-below 30 asked); in plane +9.93, -9.57 nm off the wire's axis", i.e.
      10 nm deep and 13.8 nm off axis, because neither the depth nor the axis is a Yee node.
      Priced rather than waved through: over 10 nm of depth the in-material field falls by
      1 - cos(2*pi*3.44*10/900) = 2.9 per cent, and 13.8 nm is inside the central plateau of a
      210 nm-radius wire, so the two are worth a few per cent of F -- not the factor of eight
      under investigation, and inside the paper's own displacement scan. A finer z cell at the
      source is the fix, owed once the rungs are chosen. The paper's crown is not modelled, and the text says
      which F belongs to which geometry: the ~17 is `for this cavity`, the optimised
      one, defined by height, width and the 12 nm oxide alone, while the crown enters
      only as the growth's own deviation -- "It is difficult to achieve a perfectly
      flat top facet of the NW with the bottom-up growth process" -- whose sweep
      "provides a Purcell enhancement greater than 10 for crown heights smaller than
      90 nm". So a plain prism is the right thing to compare with 17, and no crown can
      explain reading ~2. (The crown figure's caption transposes the labels --
      "D=1375 nm and H= 420 nm" where the text says height 1375 and width 420 -- the
      numbers agree, the letters do not; this scene follows the text.)
- [x] **The 420 nm reading is the paper's own relation, and it is now the default**
      (the lead, 2026-10-15): 2s = 1.14 D with s the hexagon's side, so the quoted
      number fixes the corner-to-corner width at 1.14 x 420 = 478.8 nm and the
      flat-to-flat at 414.6 nm. `--hex paper` is that reading and the default;
      `--hex corners` (corners = the quoted number) and `--hex flats` (flats = the
      quoted number) remain, because run 1 below is the comparison point for the
      reading and has to stay runnable. `report` prints both widths and the flag it
      used, and the docstring cites the sentence.
- [ ] The pad is not a cost knob to turn quietly: it is the mirror's own lateral extent, and
      the paper's mode is a quasi-BIC, whose field is the kind that reaches the mirror's
      edge. Shrinking it to 200 nm to fit the dx 10 rung inside the clock therefore owes a
      convergence run (dx 20 at pad 200/400/600) rather than an assumption. The layout's own
      16-cell pad sits outside the box on every axis, so no `--mirror-pad` value can reach the
      absorber; what changes with it is how much gold the mode sees.
- [ ] The gold is Rakić Lorentz-Drude here and Palik's measurements in the paper ("For the
      Gold mirror, Palik's experimental permittivity data was used"). Both are metals at
      900 nm; the difference is a named one, to be read out of a run rather than argued.
- [ ] Q from the ring-down of the short record (section 3a), reported with its error -- or
      reported as unresolved with the bound it gives. Those are the two accepted outcomes.
- [ ] NA 0.8 collection waits for the far-field cloud job that is in flight; it is not an arm
      of this script today, and nothing here claims it.
- [ ] The 4 nm line, stated honestly: the record's bins resolve 27.66 nm, so the mode's width
      is not in them. The `--ringdown` arm says so in its own output, and the ladder's 4 nm
      line says so when `--ringdown` is set, rather than leaving a reader to conclude the mode
      was resolved.

## 3b. The tier-2 runs on the Orin (the lead's approval, 2026-10-15), one rung at a time

- [x] **Run 1** `--wavelength 900 --hex corners --dx 20 --mirror-pad 400`, the comparison
      point for the old reading: grid 121 x 122 x 138 = 2,037,156 cells, 11,426 steps = 32.0
      periods, **F = 3.2295 at 900 nm**, 727.6 s wall against a 570.9 s estimate (**1.27x**).
- [x] **Run 2** the paper's own reading (`--hex paper`: 478.8 nm across corners, 414.7 nm
      across flats): 123 x 124 x 138 = 2,104,776 cells, 11,426 steps, **F = 0.1971 at
      900 nm**, 753.0 s against 589.9 s (**1.28x**). The geometry moved F by 16x -- 3.2295 to
      0.1971 -- i.e. away from the paper's 17, not toward it.
- [x] **Run 3** (pad convergence at the smaller pad) `--mirror-pad 200`: 103 x 104 x 138 =
      1,478,256 cells, 11,426 steps, **F = 0.2288 at 900 nm**, 514.6 s against 414.3 s
      (**1.24x**, the quiet bench: load 1.66).
- [x] **The pad decision the lead's rule makes** (2026-10-15: "If either pad moves F by more
      than 10%, use the larger pad for the pulse run and say so"). Pad 200 reads 0.2288 and
      pad 400 reads 0.1971: 16.1 per cent apart, over the line, so **the pulse run takes the
      larger pad, 600 nm**. Priced there at 900 nm, dx 20, each row naming its arm -- "CW" is
      the point arm, "pulse" is `--ringdown --tail-periods 20`:

          arm    geometry  cells       steps    s est    nm per bin at 900 nm
          CW     paper     2,841,696   11,426     796.4  28.12
          pulse  paper     2,821,104   18,649   1,290.4  17.13
          CW     corners   2,763,036   11,426     774.4  28.12
          pulse  corners   2,723,560   18,649   1,245.8  17.13

      The arm is not a detail, and a row is only the row it names: the same rung priced as
      `--spectrum` reads 11,550 steps at 2,821,104 cells (paper) or 2,723,560 (corners), and
      `--band-half` moves the record too (dx 20 pad 400 at 900 nm: band-half 70 -> 11,550
      steps, band-half 100 -> 8,176, band-half 30 refused). dx 15 at pad 600 and every dx 12
      or finer rung are refused at that pad.

- [x] **The rungs that were run**, each read from its own log in
      `~/mse-bench/paper1/logs` (machines: the Orin bench, `/home/sigmastrain/mse-worker/venv`).
      **Rung 6** -- pulse, paper, dx 20, pad 600, drive 1000 nm with band 1000 +- 100:
      143 x 144 x 138 = 2,841,696 cells, dt 8.402479e-18 s, 17,948 steps = 45.2 periods,
      1,251.0 s estimated, tail 7,896 steps = 19.9 periods. **Killed at the 1,800 s cap**
      (exit 124, log ending at its `source` line) -- see the rate below.
      **Rung 7** -- the same arm at dx 30: 123 x 123 x 118 = 1,785,222 cells, the same 17,948
      steps, 785.9 s estimated, and it finished in **1,019.8 s wall = 1.30x**, at
      **31.4 Mcell-steps/s** (its own `progress` lines). That rate is what killed rung 6:
      2,841,696 cells x 17,948 steps / 31.4e6 = **1,624 s of stepping alone** against the
      1,800 s clock, so under 3 min of headroom for setup and recording -- and the kill says
      that is less than they take. The estimator prices those same cells and steps at
      1,251.0 s, i.e. it assumes **1.30x** more than this bench gives; treat every tier-2
      estimate here as 1.30x optimistic. **Rung 8** -- CW, paper, dx 20, pad 600, at the
      fitted 986.7 nm: 144 x 144 x 138 = 2,861,568 cells, dt 8.357141e-18 s, 12,603 steps =
      32.0 periods, 884.6 s estimated, **1,165.6 s wall = 1.32x**, **F = 3.2419**.
      **Rung 9** -- the same point 10 nm down-band, CW at 976.7 nm, same mesh class:
      144 x 144 x 138 = 2,861,568 cells, dt 8.362267e-18 s, 12,468 steps = 32.0 periods,
      875.1 s estimated, **1,152.4 s wall = 1.32x**, **F = 1.6687**, and no NvMap line at all
      (rung 8's log had 18). The two points are 10 nm apart in wavelength and a factor 1.94
      apart in F, which is one sharp line rather than a broad one -- but both are *40 nm* runs
      (see 3d) and neither stands beside the paper's 17.
- [ ] The two-point pad sweep is a rule applied, not a convergence proved: run 3b (pad 600,
      CW at 900 nm, about 16 min) is the run that would prove it. It is held back rather than
      spent, because the CW point at the fitted wavelength (also pad 600) may already supply
      the same 600-vs-400 comparison for free, and that is known once the pulse run lands.
      If the fitted wavelength is far from 900 nm, the pad's convergence at 900 nm stays
      unmeasured and the report says so.

## 3c. The tail arm's two faults, found and fixed in tier 1

- [x] **The fit's window was not the drive's band** (`12b674f`). `fit_decay` searched a fixed
      +-5 per cent of the drive's frequency, while the priced pulse run's band (900-1100 nm,
      `--band-half 100`) excites 9-11 per cent either side of the drive. A mode excited
      between 900 nm and the old window's own edge would have been fitted *at* that edge and
      printed as a line -- a window artefact that costs the run. The window now follows the
      band it is given, the fit returns `f_edge`, and the arm prints "AT THE EDGE of that
      band, so this f is the search's own edge and the mode is outside it" rather than
      leaving a reader to take the edge for a peak. The narrow window's failure is asserted
      in the controls, so the fix cannot pass by both windows agreeing.
- [x] **A tail could not outlive its run** (2f). One decaying sinusoid is the model the arm
      fits; a tail holding two of them needs a second model, and without the samples on disk
      that model costs the Orin again. `--save-tail PATH` writes the record with its `dt`,
      its start step, the drive's frequency and its band (`save_record`), and `load_record`
      reads back the argument the writer was handed -- including `np.savez`'s own `.npz`,
      which is why the arm prints the name it actually wrote. Control: a synthetic tail saved
      and read back re-fits to the line fitted in memory, bit for bit (`again["f"] ==
      wide["f"]`), off 2,972 samples on disk. 15 controls hold at this commit, 18 at step 2h,
      19 after 2i's own (`--save-tail` refused) -- each count being that revision's own
      assertion, `done.stdout.count("\n  ok   ")`, read off `git show <rev>:tests/...`:
      `tiny: all controls hold`, exit 0.
- [x] **A failed tail write could take the run's number with it** (2h). The write stood among
      the arm's own statements, so a path a run cannot write (a file where a directory
      belongs) raised `NotADirectoryError` out of the arm after the fit had been printed:
      the number the run was spent for died with the save. The write is now a `try` of its
      own, after the fit, and the `OSError` a path raises is caught and reported as a
      `warning` line naming the exception's type -- so the run keeps its fit and its `Q` and
      still returns 0. Two controls, each holding one half: `tiny`'s own asks the real raise
      site for the type of what it raises (`save_record` on a path whose parent is a regular
      file), and a tier-1 fixture drives the same command the bench uses,
      `--tiny --ringdown --save-tail <a file, not a directory>/tail`, asserting the child's
      exit status, the `fit` and `Q` lines, the `warning` naming `NotADirectoryError`, and
      the absence of the write's own success line -- so a fix that drops the number instead
      of the file cannot pass, and a check that cannot fail would be seen for one. The same
      flag on a run that never rings down is refused outright (`arm_conflicts`), which is
      why the flag has to be stated, not inherited, in the arm namespaces the controls
      build: a live `--save-tail` made the band arm read dirty and the run failed its own
      controls (measured: exit 1 as committed at 3a183d9, exit 0 after the fix).

## 3d. The emitter's depth is geometry, not resolution (2j)

- [x] **The mesh was moving the emitter, and nothing said so.** The source is snapped to the
      nearest node (`fdtd_grid._locate`), so the depth below the facet is a property of the
      node list, not of `--qd-below`. Read off the runs' own `dipole` lines: **rung 8**, dx 20,
      asked 30 nm -- node (71, 71, 119) at z 1497.01 nm, **39.99 nm** below the facet at
      1537.00 nm (so its F = 3.2419 is a *40 nm* run and does not stand beside the paper's 17);
      **rung 7**, dx 30, the same request -- 29.89 nm. The same 10.10 nm shift sits in rungs
      1 and 2 (dx 20, pad 400, also 39.99), so every 900 nm reading in section 3b is a 40 nm
      one. Two depths are two geometries: a dipole 10 nm further into a 1375 nm wire with a
      facet 40 nm above it is a different Purcell factor, and the difference between two dx
      values had been read as resolution.
- [x] **The fix** (this step's own sha). `fdtd_layout.plan` takes `request["pin_source_z"]`
      (default false) and, when it is set for a point source, adds the source's own z to the
      z axis' lines before `build_grid`, so the axis' own node is the depth asked for. The
      in-plane axes are left alone on purpose: x and y are uniform at `dx` for every part
      here, so a line there would grade an axis that is not graded today (the dipole is also
      9.89 nm and 9.87 nm off the wire's axis at dx 20, which is the same class of defect and
      is **not** fixed here -- on a uniform axis it costs a graded axis, not a node, and that
      is a second decision). `paper1_nanowire.py` sets the key on the one request all three
      arms share, so the pulse, band and CW arms are pinned alike.
- [x] **The refusal is the load-bearing half.** `build_grid` drops a line that lands outside
      the extent or within `LINE_SNAP` of an edge, and the source is then snapped anyway with
      nothing printed -- a silent geometry change, which is worse than a failed run. So
      `refusals` now takes the plan and refuses when the realised depth is more than 1 nm from
      `--qd-below`, naming the realised depth, the request and the difference; it is the same
      function at both entry points (`--tiny` and the bench run), so tier 1 cannot pass a
      scene the bench would refuse. The realised depth has one reader, `emitter_depth_nm`,
      used by both the `dipole` line and the refusal, so the two cannot drift.
- [x] **Controls** (tier 1, `--tiny`: 19 controls at 2i, **21 at 2j** -- the count is that
      revision's own assertion, `done.stdout.count("\n  ok   ")`). One plans two scenes at
      dx 20 and dx 30 **with and without the pin**: the control's own (`--qd-below 60`, where
      the un-pinned depth is **57.11 nm at both dx**, read off the pre-pin run's own `dipole`
      line in `tiny_2i.log` as well as off the plan) and the paper's, built by the parser's
      own defaults rather than by a retyped copy (`--qd-below 30`: 39.99 nm at dx 20, 29.89 at
      dx 30). The un-pinned column is not decoration -- an equality against the pinned plan
      alone would pass with the pin inert on any mesh whose nodes land on the request, and the
      paper's dx-30 mesh is 0.11 nm off, i.e. it would pass by luck. The assertion is that each
      scene has at least one dx past the 1 nm tolerance, and the message marks which readings
      are. The second control drives `refusals` with a plan whose depth is out of tolerance
      (the control's own scene, un-pinned) requiring the refusal to name the realised depth,
      the request and the difference, and with the run's own plan, requiring that call to pass.
      `test_the_tiny_scene_holds_every_control` also now pins the printed depth at the 0.01 nm
      the line prints (`< 0.005`, where it was only within one finest cell before): the
      predicate that let the defect through is the one that would have let the fix through.

## Deferred by the lead, recorded here so it is not dropped

- [ ] **The dielectric as a function, not a constant** (the lead's step-1 item (7), called a
      separate item in that verdict). `delivered_power` and `purcell_spectrum` take a scalar
      `eps_r`. That is exact for the vacuum controls and for a source that sits in a
      homogeneous region, because the field is read in the source's own cell; it is not exact
      for the paper's dipole, which sits inside InP with a facet a few cells away. What this
      needs is the permittivity at the source's own cell read from the grid's coefficient
      arrays rather than passed in, and a control at a dielectric interface where the scalar
      reading is measurably wrong. No fixture in this change claims it today.
- [x] Extensions kept out of this step (the lead's item (8)): nothing in step 1 depends on the
      paper's geometry, none of it is a far-field transform, and none of it has run anywhere
      but tier 1.

## Not verified here

- The paper's figures and its table's exact digits: what the fetch tool returned is the text
  of the design and method sections, quoted above where a number is used. It is the right
  paper -- the abstract page and the full text agree on the nanowire cavity, and the "quasi-BIC
  formed by strong coupling of two resonant optical modes" of the abstract is the mirror-backed
  wire of the body, not a different design (an earlier reading of this file said the opposite;
  the quotes above are what settled it). The figure numbers themselves (Figure 1b, Figure 2a,
  Figure 6) name where a reader should look in the PDF; the lead has the PDF.
- The Lumerical page: 403.
- Anything at tier 3 (the cluster) or beyond: not this change's. Tier 2 is the lead-approved
  Orin rungs in section 3b, run one at a time with the tier-2 wrapper (5 GiB, `nice -n 10`,
  a 30-minute clock); a rung that the clock refuses is named, not substituted silently.
