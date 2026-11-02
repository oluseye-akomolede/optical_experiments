# optical-loss

## ADDED Requirements

### Requirement: The coupling SHALL report the optical loss any conductor part adds to the cavity
`scripts/contact_loss_bench.py` SHALL take a device (any multi-part `.obj`), a named
**absorber part** and a named source, and run the Purcell bench twice with the same grid and
source: once without the absorber, once with it. The absorber is a **dispersive metal** (through
the time-domain dispersive-metal path) **or** a weakly absorbing dielectric (a real ε′ with a
small ε″); any part of the upload may be named, not only a contact and not only the paper-1
gate. It SHALL print, for each run, the ring-down Q (with the ring-down fit's own validity
flag) and F at the cavity peak and at λ_X(0) when a dot is present, then ΔQ and ΔF. It SHALL
also print the free-carrier-absorption estimate Q_fca = 2πn_g/(λα) for any doped segments that
are present, labelled an estimate with its cited α source, and SHALL skip that line when the
device has no doped segment. It SHALL run at tier 2, one run at a time. Paper 1 is one
fixture, not the definition of the bench.

#### Scenario: Loss adds as inverse Qs
- **WHEN** a weakly absorbing dielectric film (small ε″, not a metal) is placed in the mode
- **THEN** 1/Q_with − 1/Q_without matches the first-order absorption 1/Q_abs =
  ∫ε″|E|²dV / ∫ε′|E|²dV over the unperturbed mode within 10 %

#### Scenario: A second conductor position is measured the same way
- **WHEN** the same cavity is run with the absorber placed at two different conductor parts,
  in either order, at the same grid and source
- **THEN** each ΔQ and ΔF is computed against its own without-absorber run, and the bench
  prints which part was the absorber for each pair

#### Scenario: A contact far from the mode costs nothing
- **WHEN** the contact is placed where the unperturbed mode's |E|² is below 1e-4 of its peak
- **THEN** |ΔQ|/Q is below 2 %

#### Scenario: A solver that drops the loss fails the loss check
- **WHEN** the time-domain run drops the film's ε″ (the material is passed to the solver as
  real ε′ only) while the first-order prediction still uses the configured ε″
- **THEN** the inverse-Q check fails (ΔQ ≈ 0 against a nonzero prediction) and the bench
  exits 1

#### Scenario: A device with no doped segment skips the free-carrier line
- **WHEN** the device has no part marked as doped
- **THEN** the bench omits Q_fca rather than printing a number computed from a division
  by a zero carrier density
