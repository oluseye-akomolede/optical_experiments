# paper1-nanowire-purcell

## Why

Paper 1 (the owner's call, 2026-10-15: papers before grading) is the reproduction of
arXiv:2601.04440, *A Broadband Nanowire Quantum Dot Cavity Design for the Efficient
Extraction of Entangled Photons* (Gangopadhyay, Grayli, Kokilathasan, Reimer; 7 Jan 2026).
Its figure of merit is a Purcell factor: a cavity mode of 4 nm bandwidth with an
enhancement of about 17.

The product's time-domain arm can run such a structure (`fdtd.py`, `Source`, probes,
`spectrum`) but can measure nothing about a dipole: it has no way to say what power a point
source delivered, and therefore no Purcell factor. A reproduction has no output without it.

## Normalisation — the decision this change turns on

The paper's own words (abstract, quoted verbatim, arXiv:2601.04440v1, fetched 2026-10-15):

> "we numerically predict this design to support a cavity mode with 4 nm bandwidth and a
> Purcell enhancement of $\sim$17"

**Decision: `F = P / P0`**, where `P0` is the analytic power of the same dipole in a
homogeneous medium of the dipole's own index `n`,

    P0(ω) = μ0 n ω⁴ |p̃(ω)|² / (12 π c)

the Larmor power in a medium. This is the standard FDTD normalisation (a dipole's emitted
power divided by the same dipole's power in a homogeneous medium), it is the convention under
which "Purcell enhancement" is a number about the cavity rather than about the mesh, and it
is the only one the paper's table can be read against. The dipole moment is derived, never
fitted: the source stands for the current `J_z = −ε0 ε_r S(t)` over its node's cell volume,
so `|p̃| = ε0 ε_r V |S̃| / ω` (the derivation is in the spec delta, and `P` is the power that
current handed the grid — radiated plus absorbed, which is why the numerator is
`delivered_power` and not the flux through a surface).

Not verified, and not used as fact here:

- **The Lumerical definition.** The Ansys KB page for the dipole source
  (`optics.ansys.com/hc/en-us/articles/360034915174-Dipole-source-Simulation-object`) answers
  **HTTP 403** to the fetch tool (2026-10-15), so nothing is quoted from it in this change.
  The normalisation above is derived from the physics, and the change *tests* it (a
  homogeneous medium must give F = 1) rather than citing it.
- **The paper's geometry.** The fetched body text gives a wire of 1375 nm by 420 nm, a 12 nm
  SiO2 buffer and a quantum dot 30 nm below the top facet — but the abstract describes a
  quasi-bound-state-in-the-continuum cavity from two coupled modes, and the fetched caption
  describes a gold-mirror cavity. Those are different designs, so none of those numbers is
  asserted. `scripts/paper1_nanowire.py` takes every dimension as an argument; the lead pins
  them from the PDF.

## What changes

- `fdtd.py`: `record_drive=` on `simulate` (the drive is recorded beside the trace, so a
  measurement never retypes the waveform the update used), and the measurement arm —
  `source_cell_volume`, `delivered_power`, `dipole_moment`, `larmor_power`,
  `radiated_power`, `purcell_factor`. A single-frequency CW number, not a spectrum: the
  paper's figure of merit is one frequency's enhancement, and the tail window is what makes
  the record's own length drop out of it.
- `openspec/specs/time-domain/`: six ADDED requirements (the recorded drive, the delivered
  power, the normalisation, the in-plane dipole scored against its mirror law, the broadband
  drive inside its own record, and the flux box with its refusal inside the absorber).
- `tests/test_fdtd_bench.py`: the bench's spec-path assertion follows the archived change to
  the live spec. (The lead's review nit on step 1: the file moved with the change and was
  missing from this list.)
- `tests/test_fdtd_purcell.py`: the measurement, the ladder, and the tier-1 controls.
- `scripts/paper1_nanowire.py` (step 2): the paper's stack as a scene this product's own
  layout builds -- a gold slab, the SiO2 buffer, a hexagonal InP prism, a dipole on or off
  the axis -- with the grid, the estimate and the record's resolution printed before anything
  runs, and three measurement arms (CW, one pulse through `purcell_spectrum`, and the same
  pulse read as a decay: `--ringdown` cuts the record at the drive's own end and fits one
  decaying sinusoid, reporting Q with the error it gets rather than a linewidth the record's
  bins cannot resolve). It is a bench script: it reads `fdtd.py`, `fdtd_grid.py` and
  `fdtd_layout.py` and edits none of them, so it adds no requirement to the spec. Its
  scenarios stay in `tasks.md`, by this change's own rule, until the lead's Orin run passes.
- `tests/test_paper1_nanowire.py` (step 2): the tiny scene's 15 controls, two of them seen
  red on purpose, the fitter checked both ways against synthetic tails (one with a known tau,
  one with no decay in its window at all), and the refusals the Jetson arm shares with the
  tier-1 arm.
- `scripts/paper1_nanowire.py` (step 2b): `--estimate` prices the rung the caller named when
  it is not one of the ladder's own pairs, marked in the verdict column -- before this it
  overrode `--dx`/`--mirror-pad` on every row it printed, so `--estimate --dx 5` was silent
  about dx 5 and read like the asked rung had been priced and fitted. Its "finest fitting
  rung" is now the minimum-dx fitting rung rather than the last row to fit, which was the
  same thing only by the luck of the table's order. Measured at dx 5 / pad 400:
  26,873,280 cells, 3.4x over the cap, so the owner's 2x-finer step-3 run is not reachable
  at this pad; the reachable convergence pair is dx 20 -> dx 10.

## Impact

- No product behaviour changes: no job can request a Purcell factor yet, and nothing in
  `web/`, `worker/` or the job schema moves.
- The bench's own scenarios stay in `tasks.md` until the lead's Orin run passes, the way the
  V6 bench scenarios stayed out of the spec until they did.
