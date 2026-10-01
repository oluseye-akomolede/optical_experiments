# time-domain

## ADDED Requirements

### Requirement: A time-domain run SHALL use a stable Yee time step

The solver SHALL step E and H on a graded tensor-product Yee grid with
dt = 0.99/(c·√(1/dx_min²+1/dy_min²+1/dz_min²)), so it never violates the 3-D
Courant limit of the smallest cell.

#### Scenario: a closed metal box conserves energy
- **WHEN** a pulse is launched in a lossless PEC box and the source has ended
- **THEN** the total electromagnetic energy stays within 1% of its value at
  source end for the rest of the run

### Requirement: The source SHALL be the one the user defined

A `td` request SHALL carry the source kind (`point` | `plane`), the time profile
(`cw` | `pulse`), the frequency, the bandwidth for a pulse, the polarisation, and
the position for `point` or the direction for `plane`. A `plane` source with
boundary `metal` SHALL be refused before it is queued, with a message that names
the conflict.

#### Scenario: a pulse sits at the requested frequency
- **WHEN** a `pulse` source with centre frequency f is run
- **THEN** the spectrum of the source's own time profile peaks at f within 1%

#### Scenario: plane wave in a metal cavity is refused
- **WHEN** the form asks for `plane` with boundary `metal`
- **THEN** the request is refused with a readable message and no job is queued

### Requirement: A metal-box pulse SHALL ring at the box's resonances

A pulse in a closed metal box SHALL excite only the box's own resonances, so
the time-domain solver agrees with the analytic cavity and with the cavity solver.

#### Scenario: benchmark V1
- **WHEN** a broadband pulse is run in an empty rectangular metal box
- **THEN** each spectrum peak in the pulse band is within 1% of an analytic
  rectangular-cavity frequency
- **AND** peaks are matched to modes one-to-one, and every analytic mode that
  both the source and the probe overlap has a peak within 1%; a missing mode
  fails the benchmark
- **AND** the recorded window is at least 4 / (the smallest gap between
  distinct analytic modes in the search band), so neighbouring modes don't merge
  into one peak; if that needs more than the configured maximum record, the
  benchmark refuses and names the pair rather than running
- **AND** the 1% tolerance is widened only to twice the grid's own dispersion
  error, and the output says so when it is

### Requirement: An open boundary SHALL absorb outgoing waves

With boundary `open`, the CPML SHALL absorb waves leaving the grid, so no
reflection from the grid edge is shown as part of the simulation.

#### Scenario: benchmark V3
- **WHEN** a plane-wave pulse crosses empty open space and leaves the grid
- **THEN** the residual field energy is below 1e-4 of the peak energy

### Requirement: A run SHALL be refused above its limits before it is queued

The estimate SHALL be computed from the form. A request above 8 M cells or
20 000 steps SHALL be refused with a message naming the limit and the value
that exceeded it.

#### Scenario: an oversized grid is refused readably
- **WHEN** the requested frequency and size give more than 8 M cells
- **THEN** the page shows the refusal with the cell count, and the server
  does not respond 500

### Requirement: Dispersive metals SHALL follow their published model

Gold, silver, aluminium, copper and chromium SHALL be modelled by Drude-Lorentz
fits from a cited source, and stepped with auxiliary differential equations.

#### Scenario: benchmark V4
- **WHEN** a pulse reflects at normal incidence from a gold slab
- **THEN** the reflectance across the pulse band is within 2% of the analytic
  Fresnel reflectance for the same Drude-Lorentz model

### Requirement: Thin layers SHALL be resolved, not lost

The grid SHALL place at least 4 cells across every layer thinner than the
coarse cell, or across the metal skin depth where that is thinner, with a
neighbour spacing ratio of at most 1.3.

#### Scenario: benchmark V5
- **WHEN** a dielectric film far thinner than the coarse cell sits between two
  metal plates
- **THEN** its resonances are within 1% of the analytic planar-cavity frequencies

### Requirement: A drawn waveform SHALL be checked against the grid

A `drawn` source SHALL be resampled to dt. A waveform with significant spectral
energy above the grid's resolvable frequency SHALL be refused or warned about
before it is queued, naming that frequency.

#### Scenario: an under-resolved drawing is caught before queueing
- **WHEN** a drawn waveform has a sharp edge whose spectrum extends past the
  grid's resolvable frequency
- **THEN** the form shows the warning, naming the frequency, before any job is
  queued

### Requirement: The scene's physical size SHALL come from the owner's units and scale

A `td` request SHALL take its physical size from the same `units` and `scale`
form fields the cavity tool uses (size × scale × unit), and the estimate SHALL
be computed from that physical size.

#### Scenario: a 10 nm flat film in a 1 µm scene fits
- **WHEN** a 1 µm scene holding an axis-aligned 10 nm gold film is requested at a
  visible-band pulse
- **THEN** the estimate is below 8 M cells on the graded grid, and the run is
  accepted

#### Scenario: a curved thin shell is estimated, not assumed
- **WHEN** a curved object of 1 µm with a 10 nm coating is requested
- **THEN** the estimate reports the cell count the graded grid actually needs, and
  a request above the cap is refused naming that count

### Requirement: A periodic cell SHALL reproduce the crystal's bands

A `td` request MAY declare periodic faces with a Bloch wave vector k. The run
SHALL then model an infinite crystal of that cell.

#### Scenario: benchmark V6
- **WHEN** a pulse is run in one cell of a 2-D crystal extruded along z, with
  Bloch-periodic faces at a chosen k
- **THEN** the spectrum's peaks are within 1% of the photonic-crystal band solver's
  frequencies at that k
