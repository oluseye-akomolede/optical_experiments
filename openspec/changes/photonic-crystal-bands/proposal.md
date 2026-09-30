# Photonic crystal band diagrams

## Why

Feature 2 of the photonics roadmap: a second product on the same platform, the
2-D photonic-crystal band diagram. The owner uploads nothing; they fill a form
(lattice, `r/a`, refractive indices, bands, polarisation) and the cluster
computes the dispersion — the frequencies ω of the Bloch modes over the
irreducible-zone k-path, with the photonic band gaps shaded and the gap-midgap
ratio printed.

It reuses the platform end to end: the same `mse.jobs` table (a `kind`
discriminator, `cavity` | `pc`), the same claim → solve → store path, the same
MinIO/Postgres result, and the existing pages. No new JS library (the band plot
is a canvas/SVG line chart), no new Python dependency (`numpy`/`scipy` are
already in the solver image).

The physics is a **scalar** 2-D problem, not the 3-D vector one: with the
lattice in x–y and invariance along z, Maxwell splits into two scalar
eigenproblems, **TM** (`E_z`; plain stiffness, ε on the mass) and **TE** (`H_z`;
`1/ε` on the stiffness, plain mass). The Bloch condition is imposed by periodic
node pairing with a phase, giving a complex Hermitian generalised pair solved by
sparse shift-invert `eigsh`.

## What changes

This is a new capability, `photonic-bands`; no existing spec changes. It is
ADDED, so archiving it introduces `openspec/specs/photonic-bands/spec.md` and
touches nothing else.

The requirements cover: the request and its validation (every refusal naming its
own limit), the lattice/cell/mesh contract (conforming circle, periodic pairing
by coordinate, not index), the two scalar operators and their polarisation
labels, the k-path, the sparse solve, the band gap and its midgap ratio, the
storage shape, and the two validation controls the change is judged by — the
empty lattice (exact at Γ; O(h²) elsewhere) and the Joannopoulos alumina-rod TM
gap, each with its tolerance stated as a scenario.

## What this does NOT do

3-D slabs, dispersion/loss, defects and waveguides, non-circular rods, radius
optimisation, complex-ω quality factors, and finite-height effective-index slabs
are all out of scope for v1 (roadmap §6). `r/a` beyond where the circles still
fit is a refusal, not a clamp.
