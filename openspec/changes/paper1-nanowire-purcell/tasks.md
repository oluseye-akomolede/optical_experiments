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
- [x] `--tiny`: 13 controls, every one of them a claim that can go red (two of them are seen
      red on purpose in the tier-1 file): the prism mesh closed and its volume the hexagon's,
      the product's own clock the tier-2 timeout, the buffer keeping cells of its own and
      inside the thin rule's cap and finer than the coarse cell, the product's own
      `check_axis`, the gold a region and at the part's own z and clear of the absorber, the
      source in InP and clear of both the facet and the absorber, and F = 1 in vacuum through
      the script's own path (band arm when `--spectrum`). The scene is deliberately small --
      20 nm cells, a 100 nm wire -- and exercises the same code path, not the same physics.
- [x] `tests/test_paper1_nanowire.py`: 8 tests, `8 passed in 184.44s` (task `bvvw9vj3u`, exit 0;
      tier 1, `nice -n 19`, one worker, `(ulimit -v 4194304; ...)`). The run is on this branch
      rebased onto main `9a1bc55` and on exactly the two files committed below -- the commit
      adds the change docs only, not a line of the script or the test: the 13-control
      verdict with its count pinned, the band arm through the same path, the facet control seen
      red (`--qd-below 25`), the prism control seen red (a mesh with a face removed), the three
      refusals/warnings -- a record shorter than its tail, a record over `fdtd.MAX_STEPS`
      (4,000 periods, through the same `refusals` the Jetson arm uses), and a record the drive
      has not settled -- and the ladder saying so when every rung is over a cap instead of
      reading the last rung it tried.
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

          args      --spectrum --dx 10 --mirror-pad 400
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
- [ ] The output to read: the printed grid and estimate, `measured F = ... at ... nm`, the
      spectrum's peak and its FWHM against the 27.66 nm bin, and whether the shape is the
      paper's table's. Step 3's ladder (`--height` over the paper's H) is one command per rung.

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

- The paper's method section and its table: what the fetch tool returned is summarised text
  from a page whose design does not match the abstract. The lead reads the PDF.
- The Lumerical page: 403.
- Anything at tier 2 or 3: the Orin run is the lead's.
