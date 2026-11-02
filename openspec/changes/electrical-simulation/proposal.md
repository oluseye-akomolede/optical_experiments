# electrical-simulation

## Why

The owner (2026-11-01): "start speccing out electrical simulations so we can simulate things
like superconducting wires and electrically triggered quantum dots". Today mse solves only
optics: FEM cavity modes (`maxwell-solver`) and FDTD with dispersive metals and Purcell
benches (`time-domain`). It has no static fields and no carriers.

The owner then answered this change's questions (Slack, 2026-11-01; recorded in `tasks.md`
0.1, **superseded 2026-11-02 by tasks 0.3**). That round picked the paper-1 nanowire quantum
dot (arXiv 2601.04440: an InP wire on a gold mirror with a 12 nm SiO₂ buffer) with a gate or
p-i-n contact as the *first device*. The owner wanted the full electrical analysis
(electrostatics, Stark shift vs voltage, p-i-n injection and I-V), in full 3-D, with the
emission energy, dipole and timing feeding the existing Purcell runs, and the results shown
in the web app. Superconductors are not needed for paper 1. They move to their own
not-started change, `superconductor-models`, for the "Majorana 2" paper the owner names
next.

**Revised 2026-11-02: this is a general-purpose electrical solver, not a paper-1 job.** The
owner's messages (Slack, 2026-11-02 13:19Z): *"Your questions about p-i-n diodes make me think
that you're not creating a general-purpose electrical simulation ability for our solver."*
Then, on this revision's shape (13:33Z): *"General ok, a ok, b yes"* — approving the general
request shape, the pre-run 3-D size check with a faster axisymmetric 2-D option, and
"try loss" as optical absorption by metal contact parts.

The owner's 13:33Z message answered the lead's three options and nothing more: **general** =
a general electrical job (any parts, materials, optional doping, multi-contact, with the QD
analysis as an optional add-on and paper 1 as validation); **a** = 3-D current flow ships with
a pre-run size check that refuses runs that will not fit the quota, plus a faster 2-D option
for round shapes, no quota increase; **b** = "try loss" is optical absorption by metal
contacts. The owner did not name the material list, did not ask for a 1-D form, and did not
say loss was out of scope — **b yes puts loss in this change.**

What is recorded here as design, not as an owner answer: the **shipped library** of common
electrical materials (that is how this revision makes "any device" true without the uploader
typing in physics), the axisymmetric form being a cylinder, and the pre-run estimate being
priced from the device's own mesh arrays. These are the lead's and builder's choices in
service of the owner's "general ok" and "a ok"; they are marked as design in `design.md`.

So the job is general: upload any device, any number of parts, any number of contacts. Each
part is a material from the shipped library — or the same material with its parameters
overridden, each value with its cited source — optionally doped, or a contact held at a
voltage. The library is what makes "any device" true without the uploader typing in physics;
every number in it is read from its source before it ships, never from memory. The output is potential, field and
charge-density maps on the mesh, contact currents, I-V curves and a capacitance matrix. The
quantum-dot analysis becomes an **optional add-on** to that job rather than the job's shape,
and paper 1 stays the first **validation bench**, not the limit of what the job can do.

## What changes

The shape is a general electrical job, with the quantum-dot analysis as an add-on on top of
it. Four phases. Each model ends at a check against a published or exact answer, plus a
mutation that must make that check fail (see `design.md`).

- **Phase 1, electrostatics (tier 1–2):** `electrostatics` (new). A 3-D Poisson solver on the
  existing TetGen tet mesh, general in its parts and contacts: per-part static permittivity,
  any number of contact parts each held at a voltage, field and potential sampled anywhere,
  a capacitance matrix over all contacts. New: a graded mesh through TetGen's sizing field,
  and an iterative (AMG-preconditioned CG) solve above the direct-solve size. The analysis
  kind follows the inputs: no doping anywhere means electrostatics only.
- **Phase 2, drift-diffusion (tier 1–3):** `drift-diffusion` (new). Poisson plus electron and
  hole continuity through DEVSIM, for any request that carries doping. Two forms: an
  axisymmetric (r, z) model for round devices that resolves the junctions, and a coarse 3-D
  model on the tet mesh (see the pricing in `design.md`). A pre-run size check refuses a run
  that will not fit the namespace quota, naming the estimate and the limit. It gives I-V,
  the field anywhere under bias, and the carrier arrival profile for a voltage pulse.
- **Phase 3, the optional quantum-dot add-on and loss (tier 2):** `stark-shift` (new) and
  `emitter-coupling` (new). When a request marks a dot, the add-on gives E_X(V), the
  electron-hole overlap and the dipole, reads F at the Stark-tuned energy from an existing
  Purcell spectrum and convolves injection with decay for photon timing. The loss requirement
  is general: the optical absorption that **any** conductor part adds to the cavity (Q and F
  with vs without it). The owner answered "b yes" to "try loss", so this ships **in this
  change** as its own `optical-loss` capability, with paper 1's gate as one fixture.
- **Phase 4, web app (tier 3):** a new job kind `elec` (`electrical-jobs`, new). It has an
  upload arm, a bounded worker run, flat result files and a result page with the voltage
  sweep, the I-V, the timing and the field/potential/charge maps (reusing the mode viewer).
  Two live requirements change: the known kinds widen from (`cavity`, `pc`, `td`) to include
  `elec` (`time-domain` delta), and the cluster launcher builds a second Job shape for `elec`
  (`photonic-bands` delta).

Paper 1 is the first validation bench for both generality and accuracy: the benches also
cover a MOS capacitor, parallel plates and a multi-terminal case, so generality is tested
rather than only paper 1.

## Impact

- New, when built: `dockerfiles/maxwell_solver/electrostatics.py`, `stark.py`,
  `drift_diffusion.py`, `emitter_coupling.py`, benches under `scripts/`, tests, and the web
  app's `elec` arm, worker branch and page. New dependencies: `pyamg` (Phase 1) and `devsim`
  (Phase 2).
- `mse_meshing.tetrahedralize` gains an opt-in sizing-field keyword that defaults to today's
  behaviour. The `maxwell-solver` and `mesh-upload` specs get no delta.
- A small cited materials library ships with the solver (Si, GaAs, InP, SiC, SiO₂, Al₂O₃,
  HfO₂, Au, Ag, air), each parameter carrying its source, so a general job does not require
  the uploader to know a material's numbers. A per-part override always wins.
- Two live requirements are modified (the `time-domain` job-kind list and the
  `photonic-bands` launcher's Job shapes). Nothing is removed.
- Full 3-D drift-diffusion at junction-resolving resolution does not fit the namespace quota
  (estimate). `design.md` §5 prices it; the pre-run size check refuses such a run and names
  the estimate and the limit, and the axisymmetric form is the faster option for round
  devices. No quota increase is requested.

## Questions the owner answered on 2026-11-02

The owner's words were *"General ok a ok b yes"* (13:33Z), answering the lead's three options.

1. Junction-resolving 3-D drift-diffusion needs about 100 GiB or more (estimate), far above
   our 18 GiB. **Answered "a":** ship the 3-D path with a pre-run size check that refuses
   runs that will not fit (naming the estimate and the limit), plus a faster axisymmetric 2-D
   option for round devices. No quota increase. The estimate being priced from the device's
   own mesh arrays, and a refused run never being retried coarser, are **our design choices**
   in service of that answer, not the owner's words.
2. Which contact? **Dissolved by "general ok":** the job accepts any number of contacts, so
   the question of one fixed contact no longer gates the build. Paper 1's gate and p-i-n
   variants are two validation fixtures among others.
3. "Try loss". **Answered "b yes":** optical absorption by metal contacts — Q and F with and
   without the metal, generalised to any conductor part rather than only a gate. **This is in
   scope for this change** (see the `optical-loss` capability under `specs/`), with paper 1 as
   one fixture among others.
4. Which materials? **Not an owner answer.** "General ok" asks for a general job; the shipped
   library (Si, GaAs, InP, SiC, SiO₂, Al₂O₃, HfO₂, Au, Ag, air) is **our design choice** for
   how to make "any device" usable, recorded as design in `design.md`. Every number in it is
   fetched from its cited source before it ships, never entered from memory.
