# time-domain

## ADDED Requirements

### Requirement: A point source's drive SHALL be recorded beside its trace

`simulate` SHALL be able to return, in `"drive"`, the value the point source added at each
step, as recorded where the update read it (`fdtd.py`, the source line of the step loop:
`field[component][cell] += dt * source.value(t)`), not recomputed afterwards by a caller
from `Source.value`. A measurement that multiplies the trace by the drive SHALL use the
recorded one, so a change to the injection can never leave the measurement reading a
waveform the run did not use. The array SHALL be host numpy, float64, length `n_steps`; the
key SHALL be present and `None` when the run did not ask for it or has no point source to
record. Recording SHALL cost the run nothing else: probes and the energy record are the same
with and without it.

#### Scenario: the drive is the update's own waveform
- **WHEN** a point source with `profile="pulse"` runs 200 steps with `record_drive=True`
- **THEN** `result["drive"][n]` equals `source.value(n * result["dt"])` at every step
  checked, and the array is float64 of length 200
- **AND** without `record_drive` the key is `None`, and neither the traces nor the energy
  record of the run change

#### Scenario: a plane-wave run has no point drive
- **WHEN** `kind="plane"` runs with `record_drive=True`
- **THEN** the result's `"drive"` is `None` (a plane wave has no cell), rather than an array
  of zeros

### Requirement: The power a point dipole delivered SHALL be the work its own source did

`fdtd.delivered_power(drive, trace, dt, *, volume, eps_r, frequency, periods=8,
window="hann")` SHALL return (watts) the time-averaged power the soft source handed the
grid, and `fdtd.source_cell_volume(spacings, cell, component)` SHALL return the volume that
source's current fills: the dual thickness along the component's own axis, the node
thickness along the other two, the same weights `simulate`'s energy sum builds with.

The delivered power SHALL be the work the source did, not a point field sample. The source
adds `dt S` to one `E` element each step, so the energy that element gains during step `n`
is, exactly,

    W_n = ε0 ε_r V dt S_n (post_n + post_{n−1}) / 2

`post` and `pre` being the trace at the two samples either side of the injection; the
identity SHALL hold to the grid's own energy record (measured: 0.9999997 of it), and a
one-sided spelling (`post − dt S/2`) SHALL miss by 40 times, so the correction is shown to
be load bearing rather than a tidier spelling. Both series settle into tones, so the mean of
their product over whole periods is

    P = ε0 ε_r V Re[Ẽ S̃*] / 2

with the amplitudes read over the LAST `periods` drive periods (`periods=8` default) and a
Hann window by default. The mean SHALL be over the tail and not over the record: a CW record
carries its own 5-period turn-on, and the record's length must not be part of the answer.
Measured (16 cells per wavelength): a free dipole's tail-windowed F is 1.00937 on a
1343-step record where its whole record reads 1.01854; at 296 steps that mesh is still inside
its own 5-period ramp, so BOTH windows read high — 1.33361 windowed against 2.10286 whole.

The tail window SHALL be stated as this arm's convention and not the paper's. The paper's
number is a steady-state enhancement at one wavelength; the last `periods` drive periods of a
settled record is this arm's spelling of steady state, and the two are comparable because both
are steady states. The window, its 8-period default and its Hann taper are choices made here,
not read off the paper, and a reader comparing the two numbers is entitled to the difference.

The delivered power is the Purcell numerator for a structure that absorbs — radiated plus
absorbed — while a flux surface only sees what leaves the box, so it is the same number
wherever the field is later lost (PEC walls, absorber, a metal).

`delivered_power` SHALL raise `FDTDError` for a drive and trace that are not two
one-dimensional records of one length, for fewer than 2 steps, for a non-positive volume,
`eps_r` or frequency, for `periods < 1`, for a `dt` too coarse to sample the drive, and for
a window longer than the record (record longer, or ask for fewer periods).

#### Scenario: the delivered power is the ledger of the run's own energy
- **WHEN** a CW z-dipole drives a closed metal box — 17 nodes, 16 cells per wavelength, 30
  periods, no CPML and no dispersive term, so the source is the only thing that can move the
  energy — with `record_energy=True` and `record_drive=True`
- **THEN** `Σ W_n` equals `U(last) − U(first)` of the run's own energy record to within 1e-5
  relative (measured 0.9999997)
- **AND** the same sum with `post − dt S/2` in place of the midpoint misses by more than 10x
  (measured −40.2180 of the ledger), so the two samples either side are shown to be the ones

#### Scenario: the windowed power is the time domain's own mean
- **WHEN** the same dipole runs in an open box that radiates, and the per-step instantaneous
  power `ε0 V S_n mid_n` over the same tail is averaged with the same Hann window
- **THEN** the helper's number equals that mean to within the fixture's stated tolerance
  (measured 0.9985 of it, against a flat mean over the same 224 steps that is 0.9782 of it —
  the 2ω ripple the window exists to suppress)

#### Scenario: the reading is the steady state and not the record
- **WHEN** the same vacuum CW dipole is run for 12 drive periods and for 32, and each record is
  read over its last `WINDOW = 8` periods
- **THEN** the two agree (measured F = 1.00616 and 1.00569, 0.0005 apart), so the record's own
  length is not part of the answer and the window is what makes it so
- **AND** the same pair read over their WHOLE records does not agree (measured 1.70781 against
  1.07865, 0.63 apart and still falling with length): a CW record carries its own 5-period
  turn-on, so a whole-record mean is a mean over the ramp as well as the steady state. The
  fixture is therefore red if the window is ever widened back to the record — which is the
  failure this convention exists to prevent, and it was observed before it was written down

#### Scenario: the cell volume is the node's own cell
- **WHEN** `source_cell_volume` is asked for each component at a node on a graded axis
  (8:1 across the axis, so a wrong width is wrong by tens of per cent)
- **THEN** each is the product of the node's own widths — dual along the component's axis,
  node along the other two — as computed from the primal lengths, and it equals the module's
  own thickness arrays for the same node
- **AND** it is 1.0 for a uniform spacing of 1, it is not the primal cell on a graded grid
  (measured 2.935764 against 3.701389), the three components are not alike (1.485x between
  the extremes), and the run's own `"source_volume"` is this same number

### Requirement: The Purcell factor SHALL normalise by the same dipole in a homogeneous medium

`fdtd.purcell_factor(power, larmor)` SHALL return `F = P / P0` with `P` the power the dipole
delivered and

    P0(ω) = μ0 n ω⁴ |p̃(ω)|² / (12 π C0)

the analytic power of the same dipole in a homogeneous medium of the dipole's own index `n`
passed by the caller (`fdtd.larmor_power`). The dipole moment SHALL follow from the injected
current and SHALL NOT be a fitted constant: the source's current is `J = −ε0 εr S(t)` over
the source cell (from the update's own `+= dt S` on `E`, since `∂E/∂t = (∇×H)/ε − J/ε`), so
with the amplitude `_windowed_at` returns,

    |p̃(ω)| = ε0 ε_r V |S̃(ω)| / ω

(`fdtd.dipole_moment`), read over the same tail window as the power. The moment SHALL be
derived, never fitted: the same run's far field carries this moment's Larmor power, which is
what makes F a statement about the structure and not about the mesh. `purcell_factor` SHALL
raise `FDTDError` rather than divide by zero for a non-positive power or Larmor power.

#### Scenario: a homogeneous medium gives F = 1
- **WHEN** a dipole runs in a lossless homogeneous medium with `boundary="open"`, so the
  power radiates instead of being stored, and the delivered power is divided by that same
  dipole's Larmor power
- **THEN** F is 1 to within the fixture's measured floor: 1.0106 at 16 cells per wavelength
  (the normalisation is only as good as the grid's own dipole, which is why the number is
  quoted with the mesh)
- **AND** the flux instrument's own reading of the same medium converges rather than drifting
  with the mesh — 1.0661, 1.0455 and 1.0304 at 12, 16 and 24 cells per wavelength, each
  inside 0.2 of 1 and *not* the (wavelength/dx)² the old method's wrong formula gave, 4.66,
  11.74 and 26.30 on the same three meshes, which fits the 0.046 (wavelength/dx)² the retired
  probe recorded at 16 and 24 cells and overshoots the 12-cell point — and moving the flux box
  through the wave zone moves its quadrature error and not the answer: 1.0396, 1.0455, 1.0521
  for box half-extents 8, 12, 16 cells (a spread under 1.2 per cent, against the 15 the
  fixture allows)

#### Scenario: a PEC mirror gives the image-dipole factor
- **WHEN** the same dipole runs `gap` cells above a PEC half-space (a `pec` layer in an open
  box, at 24 cells per wavelength), and the delivered power is divided by the same dipole at
  the centre of the same box with no mirror, so the drive — hence the moment, hence the
  denominator — is identical in both
- **THEN**, for a dipole perpendicular to the plane, F follows

      1 + (3/4) [ sin(2a)/(2a³) − cos(2a)/a² ],   a = 2π gap / (cells per wavelength)

  — the same-sign image, half the space — which the fixture also checks against a numeric
  integral of the pattern, `(3/2) ∫ sin²t cos²(a cos t)`, so the closed form is not the only
  spelling of the oracle
- **AND** the fixture holds that oracle only where it can honestly fail, which the fixture
  states as its own measurement: at gap 7 of 24 cells per wavelength, F = 1.1956 against the
  law's 1.1629, and the sweep is printed rather than asserted past that point because the
  law's own excursion from 1 shrinks below the bound — at gaps 8, 9 and 10 that excursion is
  0.050, 0.029 and 0.073, so a bound loose enough to hold those would also pass a run with no
  mirror in it
- **AND** the measured residual is the source's own cell, not a mislabelled height: the
  excess over the law is +30.6 per cent at gap 4 and +2.8 at gap 7, falling with distance as
  a point source's near field does, while the same law one cell either side of the
  measurement reads 1.3040 and 1.0501, 0.108 and 0.146 away — each further than the whole
  0.05 the gap-7 point is held to, so one cell of height SHALL be told apart — and a
  12-cell mesh, where the near field is not resolved, has no trend at all, so the fixture is
  read on the finer mesh

### Requirement: A dipole lying IN the mirror's plane SHALL be scored against its own image, and the two laws SHALL be told apart

The paper's geometry is a dipole in the plane of the cavity mirror, whose image is
ANTI-parallel, so the test oracle `image_dipole_factor_parallel(cells, height)` in
`tests/test_fdtd_purcell.py` SHALL exist beside the perpendicular one and SHALL be the
quantity that fixture scores against:

    F = 1 − (3/4) [ sin(2a)/a + cos(2a)/(2a²) − sin(2a)/(4a³) ],   a = 2π height / cells

which is the same statement as the pattern integral it comes from,
`1 − (3/4) ∫₀¹ (1 + u²) cos(2 a u) du`, and which SHALL be pinned against that integral the
same way the perpendicular law is. The two laws SHALL be pinned against EACH OTHER at the
limits where neither closed form is evaluable, because that is where a sign error in the image
is visible and nowhere else: the perpendicular pair doubles the dipole and radiates half its
power (F(0) = 2), the parallel pair cancels it (F(0) = 0), and both fall back to 1 as the
mirror recedes.

#### Scenario: a parallel dipole follows the parallel law and not the other one
- **WHEN** the same dipole is driven ALONG the mirror's plane (`component="x"`) at the same
  heights, and the delivered power is divided by the Larmor power of that run's own drive's
  moment — no second run, because the drive is what the source was told to inject rather than
  what the field did, so one run already carries both halves
- **THEN** F follows the parallel law, and the run is printed rather than asserted past where
  the law's own excursion dies: F − law is +0.1534 at gap 3 (F = 0.58549 against 0.43209) and
  +0.1482, +0.0671, +0.0191, −0.0093 at gaps 4, 8, 10 and 12 — the same point-source near
  field as the perpendicular sweep's, +35.5 per cent of the law at gap 3 and −1.0 at gap 12
- **AND** the sweep SHALL be asserted not to be flat (measured 0.58549 to 1.40227 over gaps 3
  to 12, a span of 0.82), because a number that does not move with the height is the tell
- **AND** the fixture SHALL carry the non-vacuity of its discrimination arm as a property of
  the two laws rather than of the run: at gaps 3 and 4 the perpendicular law sits 1.342 and
  0.933 away from the parallel one, against the 0.34 the arm demands, which is where scoring
  the measurement against the wrong law is worth a whole unit of F and is therefore rejected
  by more than the bound

### Requirement: A broadband drive SHALL give the whole band in one run, with the pulse inside its own record

`fdtd.delivered_power_spectrum(drive, trace, dt, *, volume, eps_r, window=None, band=None)`
SHALL return `(frequencies_hz, watts)`: the same `ε0 ε_r V Re[Ẽ S̃*] / 2` as
`delivered_power`, bin by bin from ONE pulse rather than one CW run per frequency, so a
frequency ladder costs one run and one record. `fdtd.purcell_spectrum(...)` SHALL return
`(frequencies, power, reference, F)` over that same run's band, its `reference` built by
calling `larmor_power` bin by bin, so the denominator has one spelling and cannot drift from
the CW arm's. Each transform SHALL be normalised by twice the windowed sum over the window's
own sum — the CW arm's own `_windowed_at` convention — and the field SHALL be read at the two
samples either side of each injection, so the two arms share a phase convention (half a step
is `½ ω dt` = 4.3 degrees at 24 cells per wavelength, which a band-wide glance would not
notice and a single-frequency comparison would).

The default window SHALL be the boxcar, and the requirement SHALL state why: a pulse is over
at `t = 0`, so the record IS the signal and the drive's own spectrum is the band. Windowing is
then a choice with a cost, and the cost is measured rather than argued. The estimator SHALL be
stated as a **cross-spectral density**, valid only where the window weights the drive and the
field alike — so the pulse must be centred in the record it is measured over. The drive's own
delay is fixed at `start = 5σ`, so the record is what a caller moves, and a record that
pushes the pulse to its edge SHALL read as the failure it is rather than as a smaller error.
Measured in vacuum at 24 cells per wavelength, F over 0.7–1.3 f0 (5 bins):

    pulse at 0.53 of a 335-step record, boxcar:  F = 1.00143 .. 1.00657
    the same record under a Hann:                F = 0.98530 .. 1.16101
    the same pulse in a 20-sigma record (peak 0.26), Hann:  F = -65.30 .. -8.09
    the same pulse in a 30-sigma record (peak 0.18), Hann:  F = -94.34 .. -8.21

`band=(f_low, f_high)` in Hz SHALL keep the bins between the two, both ends included, for
every returned array together, and SHALL raise `FDTDError` when the band holds no bin — an
empty curve looks like a measurement that found nothing, which is not what happened. At DC the
moment, the reference and F SHALL be 0 rather than divided by (`p̃ = J̃ / (−iω)` at ω = 0 is not
a dipole, so F would be 0/0), while the delivered power SHALL be left as the record's own DC
bin measured it: that is a reading, not a definition. Both edges SHALL have a fixture on a
record small enough to read by hand — the refusal, and the DC bin — since a guard nothing
calls cannot fail.

The arm SHALL be stated as a PULSE arm and not a substitute for the CW one on a resonator: a
cavity's ring-down outlives any record the drive's fixed delay can be centred in, and the
ratio would then divide a windowed drive by a field that is mostly ring-down.

#### Scenario: the band-limited pulse reads vacuum across its own band
- **WHEN** a vacuum z-dipole is driven with `profile="pulse"` at 24 cells per wavelength (a
  10-sigma record, which the drive's own 5-sigma delay leaves at its middle), measured with
  `purcell_spectrum` over 0.7–1.3 f0 with the boxcar
- **THEN** every bin reads F = 1 to within 0.00657 (measured 1.00143 .. 1.00657)
- **AND** the same run under a Hann misses by more than 5x that (measured 0.98530 .. 1.16101),
  so the default is shown to be load-bearing rather than a tidier spelling
- **AND** the same pulse in a 20-sigma record and in a 30-sigma record reads NEGATIVE F
  (measured −65.30 .. −8.09 and −94.34 .. −8.21), which is what moves the pulse out of the
  window and not a small numerical error

#### Scenario: the pulse's own bin is the CW arm's number
- **WHEN** one vacuum pulse and one CW run are measured in the same box at 24 cells per
  wavelength, the first with `purcell_spectrum`'s bin nearest f0 and the second with the
  CW `delivered_power` / `dipole_moment` pair
- **THEN** the two agree within 3 per cent (measured 1.00443 at 0.012525 GHz against 1.00525,
  a ratio of 0.99919), which is what makes the band arm the same measurement rather than a
  second one — the two are not asked to agree exactly, since the CW arm reads its last 8
  periods of a record carrying a turn-on while the pulse arm reads a whole record that is its
  own band

#### Scenario: the band arm's power follows omega squared
- **WHEN** the same pulse is launched at three carriers — 20, 24 and 30 cells per wavelength —
  with the envelope held at 19 cells so the drive's own spectral amplitude is the same at each
- **THEN** each run's band F is 1 to within 0.0082, and the delivered power follows the
  sampled bins' own omega²: bins at 0.015751, 0.012601 and 0.009450 GHz carry 2.569484e-24,
  1.716588e-24 and 9.412558e-25 W, an exponent of 1.966 over 1.667x in omega — not flat, and
  not omega⁴, which over that span puts the extreme bins 7.716x apart where the run measured
  2.730x (omega² says 2.778x, 1.8 per cent out against a factor of 2.8)
- **AND** the exponent is read from the measured bins' own frequencies, not the carriers the
  runs were configured with: the sampled bin nearest a carrier is up to 6 per cent off it,
  because 10 sigmas is a different number of periods at each frequency

### Requirement: The radiated power SHALL be measurable through a box in the wave zone, and a box reaching the absorber SHALL be refused

`fdtd.radiated_power(flux, dt, *, index, frequency, periods=8)` SHALL return (watts) the power
`simulate(..., flux_box=...)`'s surface carried out: every step, `Σ |E|² · area` over the six
faces, read as `|E|² / η` with `η = η0 / index` — the locally travelling-wave statement the
far field was already checked against. The mean SHALL be over the LAST `periods` drive
periods, and `flux_faces` SHALL raise `FDTDError` for a face inside the absorbing layer
(where the surface measures the damped field and reads low), naming the axis and the CPML
depth. It is the RADIATED power, and the requirement SHALL state that plainly: a lossy
structure inside the box absorbs part of what the dipole delivers, so a Purcell factor for
one wants `delivered_power` rather than this.

#### Scenario: the flux box reads the same dipole the source did
- **WHEN** one vacuum run at 16 cells per wavelength is measured both ways
- **THEN** the flux box reads 1.0455 of the Larmor power where the source's own work reads
  1.0098 — the two agree within the gap the fixture states (measured +0.0357), neither being
  exact
- **AND** the flux instrument SHALL NOT be used to judge a field it cannot describe, and the
  fixture SHALL pin that with its own measurement: a box whose floor sits on the first node
  clear of a PEC mirror, with the whole reactive boundary layer on the surface, reads
  `F = 1.5882` where the image-dipole law says 1.1629 (`+36.6` per cent) for a dipole seven
  cells above it — the geometry whose delivered power is `+2.8` per cent — and
  `F = 346.4668` against 1.8945 for one two cells above it.
