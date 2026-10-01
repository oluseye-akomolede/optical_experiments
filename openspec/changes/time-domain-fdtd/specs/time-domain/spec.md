# time-domain

## ADDED Requirements

### Requirement: A time-domain run SHALL use a stable Yee time step

The solver SHALL step E and H on a uniform Yee grid with
dt = 0.99·dx/(c·√3), so it never violates the 3-D Courant limit.

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
