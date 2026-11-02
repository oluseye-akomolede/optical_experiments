# electrical-simulation

## Why

The owner (2026-11-01): "start speccing out electrical simulations so we can simulate things
like superconducting wires and electrically triggered quantum dots". Today mse solves only
optics: FEM cavity modes (`maxwell-solver`) and FDTD with dispersive metals and Purcell
benches (`time-domain`). It has no static fields and no carriers.

The owner then answered this change's questions (Slack, 2026-11-01; recorded in `tasks.md`
0.1). The first device is the paper-1 nanowire quantum dot (arXiv 2601.04440: an InP wire on
a gold mirror with a 12 nm SiO₂ buffer) with a gate or p-i-n contact. The owner wants the full
electrical analysis (electrostatics, Stark shift vs voltage, p-i-n injection and I-V), in full
3-D, with the emission energy, dipole and timing feeding the existing Purcell runs, and the
results shown in the web app. Superconductors are not needed for paper 1. They move to their
own not-started change, `superconductor-models`, for the "Majorana 2" paper the owner names
next.

## What changes

Four phases. Each model ends at a check against a published or exact answer, plus a mutation
that must make that check fail (see `design.md`).

- **Phase 1, the gate and the Stark shift (tier 1–2):**
  - `electrostatics` (new): a 3-D Poisson solver on the existing TetGen tet mesh, with
    per-part static permittivity and electrode parts held at fixed potentials. It returns the
    field at a point and a capacitance matrix. New: a graded mesh through TetGen's sizing
    field, and an iterative (AMG-preconditioned CG) solve above the direct-solve size.
  - `stark-shift` (new): a 3-D single-band effective-mass solve for the electron and the
    hole in the dot, in the gate field. It gives the exciton energy E_X(V), the
    electron-hole overlap and the dipole's orientation and strength.
- **Phase 2, p-i-n injection (tier 1–3):** `drift-diffusion` (new). Poisson plus electron and
  hole continuity through DEVSIM, in two forms: an axisymmetric (r, z) model that resolves
  the junctions, and a coarse 3-D model that fits the cluster (see the pricing in
  `design.md`). It gives I-V, the field at the dot under bias, and the carrier arrival
  profile for a voltage pulse.
- **Phase 3, coupling into the Purcell runs, and loss (tier 2):** `emitter-coupling` (new).
  It reads F at the Stark-tuned energy from an existing Purcell spectrum and scales the
  radiative rate by the overlap. It convolves injection with decay to give the photon-timing
  distribution. It also measures the optical loss the gate or contact metal adds to the
  cavity (Q and F with vs without the metal), which is how this change reads the owner's
  "try loss".
- **Phase 4, web app (tier 3):** a new job kind `elec` (`electrical-jobs`, new). It has an
  upload arm, a bounded worker run, flat result files and a result page with the voltage
  sweep, the I-V and the timing. Two live requirements change: the known kinds widen from
  (`cavity`, `pc`, `td`) to include `elec` (`time-domain` delta), and the cluster launcher
  builds a second Job shape for `elec` (`photonic-bands` delta).

## Impact

- New, when built: `dockerfiles/maxwell_solver/electrostatics.py`, `stark.py`,
  `drift_diffusion.py`, `emitter_coupling.py`, benches under `scripts/`, tests, and the web
  app's `elec` arm, worker branch and page. New dependencies: `pyamg` (Phase 1) and `devsim`
  (Phase 2).
- `mse_meshing.tetrahedralize` gains an opt-in sizing-field keyword that defaults to today's
  behaviour. The `maxwell-solver` and `mesh-upload` specs get no delta.
- Two live requirements are modified (the `time-domain` job-kind list and the
  `photonic-bands` launcher's Job shapes). Nothing is removed.
- Full 3-D drift-diffusion at junction-resolving resolution does not fit the namespace quota
  (estimate). `design.md` §5 prices it, and the owner's question 1 below asks about it.

## Questions for the owner (the few that change what gets built)

1. Full 3-D drift-diffusion with the junctions resolved needs roughly 100 GiB or more of
   memory (estimate), far above our 18 GiB. Is an axisymmetric junction model plus a coarse
   3-D check acceptable? Or do you want to request more memory?
2. Which contact? Paper 1 has none. (a) a gate only: the gold mirror as the back gate plus a
   top electrode. (b) an axial p-i-n with a contact through the oxide to the base and a top
   contact. And which contact metal or oxide, if you have a target device?
3. We read "try loss" as the light the added gate or contact metal absorbs: how much it
   lowers the cavity's Q and Purcell factor. Is that right, or did you mean something else?
</content>
</invoke>
