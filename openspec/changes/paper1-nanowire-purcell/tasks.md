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
      tree: `22 passed in 148.25s`. The two shas the third sitting handed over are
      `bc833b1` (the solver and its fixtures) and `eb00ea1` (this delta and this file); the
      1.48524 above was written 1.48525 in `eb00ea1` and is corrected here rather than in the
      sha the lead has already read.

## 2. Bench: the paper's geometry

- [ ] `scripts/paper1_nanowire.py`: the stack the lead described (nanowire, bottom oxide,
      top oxide, mirror), a point dipole on or off the cavity axis, `--tiny` for tier 1, the
      full ladder (mesh, offset) for the Orin.
- [ ] Reports peak `F`, its wavelength, the band's FWHM against the record's own resolution,
      and the shape next to the paper's table.
- [ ] `estimate_run` before it runs, so a grid that cannot fit the Orin's memory is refused
      rather than discovered.
- [ ] Test: `tests/test_fdtd_purcell.py` runs `--tiny` and checks its own verdict.
- [ ] Bench scenarios stay here, not in the spec, until the lead's Orin run passes.

## 3. The Orin command

- [ ] The exact command line(s), listed for the lead, with the expected size and runtime from
      `estimate_run`, and the output to compare against the paper's table.

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
