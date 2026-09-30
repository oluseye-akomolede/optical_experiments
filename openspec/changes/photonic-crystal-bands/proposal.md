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
storage shape, and the three validation controls the change is judged by — the
empty lattice (Control A, exact at Γ; O(h²) elsewhere), the 1-D layered cell
against its transfer-matrix oracle (Control B), and the MPB tutorial's
triangular rod-lattice TM gap (Control C), each with its tolerance stated as a
scenario.

The benchmark originally named here — the Joannopoulos square lattice of alumina
rods, `eps = 8.9`, TM gap 0.302–0.443 — was **dropped**: the plan records that
source as unreachable, and its numbers as quoted from memory and never read. It
is replaced by the MPB Scheme tutorial's "Triangular Lattice" result, which was
read directly (`r = 0.2 a`, `eps = 12`, resolution 32, TM gap
`0.275065617068082 → 0.446289918847647`, 47.47%). One hazard that cost a
debugging round is recorded with the test: MPB prints the mirror's **permittivity**
(`eps = 12`) where `pc.py` takes **refractive index** (`eps_by_tri = n_rod**2`),
so the benchmark must be passed as `n = sqrt(12)`.

## Rollback

This change adds two columns to `mse.jobs`: `kind` (default `'cavity'`) and
`pcs` (nullable text). To roll the schema back, in order:

```sql
ALTER TABLE mse.jobs DROP COLUMN IF EXISTS pcs;
ALTER TABLE mse.jobs DROP COLUMN IF EXISTS kind;
```

The schema qualifier is `mse`, read from `db/001_schema.sql` — the same file that
creates these columns. `db/002_rollback_pc.sql` is that script, kept with
the schema so it can be run from this change's own directory. Dropping `kind`
loses the discriminator that separates a cavity row from a pc row, so any pc row
still present becomes indistinguishable from a cavity row; roll back only after
the pc rows are gone or drained. No data is moved; the cavity path uses neither
column.

## What this does NOT do

3-D slabs, dispersion/loss, defects and waveguides, non-circular rods, radius
optimisation, complex-ω quality factors, and finite-height effective-index slabs
are all out of scope for v1 (roadmap §6). `r/a` beyond where the circles still
fit is a refusal, not a clamp.
