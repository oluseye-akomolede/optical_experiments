# electrical-simulation

## Why

The owner (2026-11-01): "start speccing out electrical simulations so we can simulate things
like superconducting wires and electrically triggered quantum dots". Today mse solves only
optics: FEM cavity modes (`maxwell-solver`) and FDTD with dispersive metals and Purcell
benches (`time-domain`). It has no static fields, no superconducting materials, no carriers.

## What changes

Four phases. Each ends at a check against a published or analytic answer (see `design.md`).

- **Phase 1 (this change's specs and tasks): two small libraries plus their benches, no web
  app, tier-1 compute.**
  - `superconductor-models` (new): BCS gap, London, two-fluid and Mattis-Bardeen
    conductivity, thin-film surface impedance and kinetic inductance.
  - `electrostatics` (new): a Poisson solver on the existing TetGen mesh (per-part static
    permittivity, electrode parts held at fixed potentials), giving the field at a point and
    a capacitance matrix.
- **Phase 2, superconducting wires:** superconductors inside the existing EM solvers (cavity
  Q and frequency shift by surface-impedance perturbation; two-fluid term in FDTD for THz),
  2-D cross-section kinetic inductance, pyTDGL critical current for short wire sections, and
  a lumped electrothermal detector model.
- **Phase 3, electrically triggered QDs:** gate field (Phase 1) into an effective-mass
  Schrödinger solve (axisymmetric) for Stark tuning; drift-diffusion for p-i-n injection;
  coupling into the FDTD/Purcell path (emitter energy, dipole orientation, timing).
- **Phase 4:** web-app upload, job kinds and result pages, only if the owner wants them.

## Impact

- New: `dockerfiles/maxwell_solver/superconductor.py`, `electrostatics.py`,
  `scripts/sc_film_bench.py`, `scripts/electrostatics_bench.py`, two test files.
- Unchanged in Phase 1: every existing spec, the web app, the solver images' job paths,
  and the quota.
- Some models will not fit the cluster: full SNSPD meanders in TDGL, 3-D 8-band k.p, and
  microwave-frequency FDTD. `design.md` gives the estimates and the reduced models used
  instead.

## Questions for the owner

1. Which device comes first: an NbN/NbTiN nanowire detector, a superconducting microwave
   resonator, or the nanowire QD from paper 1 with a gate or p-i-n contact? Which materials?
2. Which outputs matter most: critical current, kinetic inductance or reset time,
   resonator Q, Stark shift vs voltage, trigger timing or jitter, I-V curves?
3. Bench scripts only for now, or should results show in the web app?
4. Are 2-D, axisymmetric and analytic-limit models acceptable, or do you need 3-D? 3-D at
   useful resolution needs a GPU request or more quota.
5. Is there a target paper for each device, like paper 1, to use as the answer key?
