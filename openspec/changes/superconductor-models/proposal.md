# superconductor-models

**Status: not started.** Split out of `electrical-simulation` on 2026-11-01, after the owner
chose the paper-1 electrically triggered quantum dot as the first device. Paper 1 needs no
superconductor. The owner said "We'll be investigating the Majorana 2 paper next", and that work
will need superconducting materials, so this content is kept here, unbuilt, until then. No code
exists for it, so moving it removes no live behaviour.

## Why

The owner (2026-11-01): "start speccing out electrical simulations so we can simulate things
like superconducting wires and electrically triggered quantum dots". mse has no
superconducting materials today.

## What changes

- `superconductor-models` (new capability): BCS gap, London, two-fluid and Mattis-Bardeen
  conductivity, thin-film surface impedance and kinetic inductance, plus a bench that exits
  nonzero on a failed check (`specs/superconductor-models/spec.md`, unchanged from the
  original change).
- Later rungs (S4-S7 in `design.md`): surface impedance in the existing EM solvers, a 2-D
  cross-section kinetic-inductance solve, pyTDGL for short wire sections, and a lumped
  electrothermal detector model. Each needs its own proposal.

## Owner answers that bear on this change (Slack, 2026-11-01)

- Q6 "Is the first device limited by loss or by inductance?" → "I'm not sure, try loss".
  Read for this change: when it starts, build the film-loss path first (Mattis-Bardeen σ₁ →
  surface resistance R_s → cavity Q), and the kinetic-inductance rung after it.
- Q5 "just paper 1 … the Majorana 2 paper next". This change waits for that paper.

## Before this change starts

The Majorana paper may need different physics from what is specified here. Majorana devices
are usually semiconductor nanowires (InAs or InSb) with a thin superconducting shell (often Al),
described by a Bogoliubov-de Gennes model with the proximity-induced gap, spin-orbit coupling
and a Zeeman field (tools such as Kwant). The BCS gap and Mattis-Bardeen film models here are
inputs to that (the shell's Δ(T), its loss), not the model itself **(unsure: read the paper
first)**. Task 0.1 re-scopes this change against the paper before any code.

## Impact

- New, when built: `dockerfiles/maxwell_solver/superconductor.py`, `scripts/sc_film_bench.py`
  and a test file. Nothing else changes in this change's first phase.
