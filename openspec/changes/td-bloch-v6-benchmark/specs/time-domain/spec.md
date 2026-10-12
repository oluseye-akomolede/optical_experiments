# time-domain

## ADDED Requirements

### Requirement: A Bloch-periodic run SHALL find the crystal's bands (benchmark V6)

A `td` run of one cell with Bloch-periodic faces SHALL give spectrum peaks at
the frequencies the photonic-crystal band solver gives for the same crystal at
the same k, for both polarisations, and the mapping from k to the face phase
SHALL be checked against the exact folded light lines of a homogeneous cell.

Moved here from `time-domain-fdtd` when that change was archived (2026-10-12):
the bench (`scripts/fdtd_bench_v6.py`) and its tier-1 `--tiny` test exist, but
the full Jetson run has not passed, so these scenarios are not yet behaviour
the live solver is shown to have.

#### Scenario: benchmark V6
- **WHEN** a pulse is run in one cell of a 2-D crystal extruded along z, with
  Bloch-periodic x and y faces at a chosen k, periodic z faces with kz = 0, for
  each polarisation (TM: E along z; TE: H along z)
- **THEN** the spectrum's peaks are within 1% of the photonic-crystal band
  solver's frequencies at that k for that polarisation, and every band in the
  pulse band appears
- **AND** peaks are paired with bands in frequency order (a grid shifts
  neighbouring bands alike, so closest-first pairing would swap two bands
  closer than twice that shift)
- **AND** the record is long enough that spectral resolution is not part of
  the error (at least 150 periods of the band's lowest frequency)

#### Scenario: benchmark V6 shows each crystal band converging
- **WHEN** the crystal runs at two resolutions, a coarse one and the full one
- **THEN** every band is found at both, and its error at the full resolution
  is smaller in magnitude than at the coarse one, unless both errors are
  within 0.25% (the reference's and the spectrum's own accuracy)
- **AND** a band within 1% at the full resolution that does not approach the
  pc value fails, since one resolution under 1% does not show convergence
- **AND** the crystal's eps is sampled at least 9 times per cell edge, where
  the slowest band's error is settled: at 48 cells, 9, 15, 18 and 24 samples
  read it as -1.17%, -1.10%, -1.14% and -1.13%, a spread of 0.064% against
  this gate's own 0.25%, so the sampling is no longer part of the error
- **AND** sampling below that (3 per cell edge) is rejected rather than
  accepted as coarser: it reads the same band as -0.93%, a smaller error that
  flatters the coarse line instead of converging it, so aliasing at the rod
  edge would hide a coarse line's true error

#### Scenario: benchmark V6 checks k against the exact light lines
- **WHEN** the same cell holds one homogeneous medium of index n and runs at
  several non-zero k, below and at the zone edge (X and M included)
- **THEN** every excitable folded light line f a / c = |k + G| / n in the
  pulse band has a peak within 1% of it, and no peak is 1% from every line
- **AND** a tiny form of this check (20 cells, four k) runs locally in tier 1,
  so a phase that ignored k or scaled it wrongly fails a local test
