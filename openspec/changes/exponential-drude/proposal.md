# exponential-drude

## Why

Run K's Au-jc grew from round-off through its record while the shipped ADE
check (`lorentz_drude_growth`, which `stable_dt` consumes) reported the step
stable: it read `1 + 1.58e-8`, and `GROWTH_TOLERANCE = 1e-6` swallowed it.
The check cannot do better than that at `S = 0`. A uniform static field with
the pole state at its steady value is a fixed point of the coupled update, so
the exact spectral radius there is 1 at every finite dt, and every value the
check can print for it is "1, to within rounding". The tolerance's old
comment put that residue down to a near-defective Drude root returned up to
6e-8 high; measured now over six metals, 2-20 nm and 0.03-1 of Courant, it is
at most 1.4e-11. Nor is the Lorentz block behind Au-palik's `1 + 3.8e-9`:
that is the same rounding about the same root. Those two residues, and the
difference between them, were not Run K's mechanism.

What is a real defect is the pole's own step. Central-differencing the Drude
pole (`w0 = 0`) gives `P'' + G P' = eps0 f wp^2 E` a characteristic polynomial
whose roots are 1 and `(1 - G dt/2)/(1 + G dt/2)`: one root sits at **exactly
1** at every finite dt, so the scheme carries a current mode that never decays
and whose amplitude is whatever the initial state set. The exact solution of a
free current decays as `exp(-G dt)` instead.

## What Changes

- `lorentz_drude_coefficients` returns tagged records. A Lorentz pole keeps
  `("lorentz", alpha, xi, gamma)` and its central-difference recursion
  unchanged. The Drude pole becomes `("drude", a, b)` and steps by the exact
  solution of `J' = -G J + sigma E` over one step, with `sigma = eps0 f wp^2`,
  `a = exp(-G dt)` and `b = sigma (1 - a)/G`.
- The Drude pole's state is the current `J`, not a polarisation; its
  contribution to the E update is `dt J^{n+1}`. The immediate per-step
  coupling is unchanged to O(G dt), so the model's dispersion is untouched.
- `lorentz_drude_growth` models the new record, and its E row subtracts the
  pole's contribution to E **once per step**, `(d^{n+1} - d^n)/eps0`, because
  the curl update reads the E that already carried the previous step's
  subtraction -- the same thing `advance` does. Differencing that
  contribution a second time (the previous code, correct only because the old
  Drude step was itself a difference) describes a step the solver never
  takes: it refused Rakic gold at both paper grids, 1.0379 and 1.0634, when
  Rakic gold is the fit Run I settled with. `stable_dt`'s contract (return a
  step the check accepts, or the step it was given) is unchanged.
- The stepping is a small real change to every metal run, not a no-op: the
  current enters the E update at the step's end rather than its centre, so
  the metal's response moves by about `omega dt/2` in phase -- 0.49 deg at
  905 nm on the paper grids, 0.85 per cent of a radian.
- Tests: the Drude step's law over all seven library metals and five steps
  (`a == exp(-G dt)`, inside the circle, falling with dt); Rakic gold and both
  gold fits pass the check at both paper grids' dt and finest spacings, where
  the second-differenced E row refused Rakic at 1.0379 and 1.0634; and the
  2 nm comment that justified the tolerance is corrected (its residue is
  1.4e-11, not 6e-8).

## What this does not establish

Run K itself is not reproduced. A product-path fixture (a graded Au slab in
InP, open boundary, broadband pulse) grew at about 1.3e-3 per step in
amplitude for both gold fits -- but only in a box whose lateral interior was
10 cells wide against a 7-cell PML. Widening the interior to 23 and 39 cells
at the same spacings, metal and step turned the growth into a decay of
-9.8e-5 per step in both. The growth was a small-box artefact; neither gold
fit shows a growing mode at Run K's spacings in a normally sized box, and the
two golds never differed in any box tried. So this change is justified on its
own terms -- removing a structural unit root from the scheme, and making the
check's answer mean what it says -- and not as a confirmed cure for Run K.
Finding that mechanism needs Run K's own geometry.

The `GROWTH_TOLERANCE` constant stays, with the job it always had: the
coupled update's exact spectral radius at `S = 0` is 1, so a passing reading
is "1 to within rounding", and `eigvals` returns that root up to 1.4e-11 off
over the sweep measured. The check certifies "no mode grows". It cannot
certify "growth is below 1", and this change does not make it able to.
