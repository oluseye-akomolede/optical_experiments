# exponential-drude

## Why

Run K's Au-jc grew from round-off through its record while the shipped ADE
check (`lorentz_drude_growth`, which `stable_dt` consumes) reported the step
stable. The check did see something: `1 + 1.58e-8`. `GROWTH_TOLERANCE = 1e-6`
swallowed it, and the tolerance exists precisely because "eigvals returns the
Drude pole's static mode up to 6e-8 high".

That static mode is not a rounding accident. Central-differencing the Drude
pole (`w0 = 0`) gives `P'' + G P' = eps0 f wp^2 E` a characteristic polynomial
whose roots are 1 and `(1 - G dt/2)/(1 + G dt/2)`: one root sits at **exactly
1**, at every finite dt. In a homogeneous block that mode is neutral, so every
value the check can print for it is "1, to within rounding". A neutral mode is
not a stability guarantee, and a check that cannot distinguish neutral from
slightly growing cannot certify the step. The two golds land on opposite sides
of that residue: at their runs' steps Au-jc reads `1 + 1.6e-8` and grew,
Au-palik reads `1 + 3.8e-9` and did not.

## What Changes

- `lorentz_drude_coefficients` returns tagged records. A Lorentz pole keeps
  `("lorentz", alpha, xi, gamma)` and its central-difference recursion
  unchanged. The Drude pole becomes `("drude", a, b)` and steps by the exact
  solution of `J' = -G J + sigma E` over one step, with `sigma = eps0 f wp^2`,
  `a = exp(-G dt)` and `b = sigma (1 - a)/G`.
- The Drude pole's state is the current `J`, not a polarisation; its
  contribution to the E update is `dt J^{n+1}`. The immediate per-step
  coupling is unchanged to O(G dt), so the model's dispersion is untouched.
- `lorentz_drude_growth` models the new record: the static root is `a < 1`, so
  a Drude pole no longer contributes a unit-modulus mode. `stable_dt`'s
  contract (return a step the check accepts, or the step it was given) is
  unchanged.
- Tests: the Drude step's law over all seven library metals and five steps
  (`a == exp(-G dt)`, inside the circle, falling with dt); the two paper runs'
  own steps read below 1, with Au-jc strictly below; and the 2 nm comment that
  justified the tolerance is corrected (its residue is now about 2e-13, not
  6e-8).

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

The `GROWTH_TOLERANCE` constant stays. It still has a job: the drift-free
eigenvalues of the coupled E/P matrix carry an arithmetic floor of about
2e-13, and `eigvals` rounding on the Lorentz block is worth a few 1e-9. What
goes away is the need to cover a structural mode with it.
