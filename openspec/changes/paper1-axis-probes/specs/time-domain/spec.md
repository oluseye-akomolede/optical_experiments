# time-domain

## ADDED Requirements

### Requirement: The ring-down arm SHALL be able to record probes on the wire's axis, and their traces SHALL travel with the saved record

`scripts/paper1_nanowire.py --ringdown --save-tail PATH --axis-probes` SHALL add one probe per E component (`x`, `y`, `z` -- never a magnitude) on every z node of the wire's axis from the first node clear of the gold's top face (a millionth of a cell above it: the layout's point-in-part test reads a node on the face as on the mirror, whose field is zero) to three coarse cells above the wire's top facet, the in-plane position being the source's own node (which is the wire's axis: the layout builds the scene around the source's corner). The probes SHALL be placed from a `fdtd_layout.plan` taken before the voxeliser runs -- a probe must sit on a node, and only the plan's node list says where the nodes are -- and the run SHALL compare the plan's nodes with the layout's before stepping and refuse if they differ, so the line is on the nodes the run uses and not merely near them. The plan SHALL NOT pin a z line into the grid for the probes: a pinned line grades the axis it is added to, and probes placed after the grid would move it.

The traces SHALL be written into the `--save-tail` record beside the field: `probe_traces` (one row per probe), `probe_z_m` (each probe's node z read off the layout's nodes at the cell the layout placed it on, in the frame `--qd-below` uses), `probe_components`, the wire's own z span (`wire_z_m`) and the emitter's z (`emitter_z_m`). The writer SHALL take the layout the solver actually ran on -- whose probes are `(i, j, k, component)` four-tuples and which SHALL carry the parts' own boxes as `part_bounds`, passed through from its plan -- so the record's abscissa and its wire span come from the layout that was solved and not from a second plan of the same scene. `--axis-probes` SHALL be refused, before any scene is built, without `--ringdown` and without `--save-tail`, since the traces are read over the tail and would otherwise be recorded and thrown away -- except with `--estimate`, which prices the line and records nothing.

#### Scenario: a probe is on the node it names, on a graded axis
- **WHEN** the line is planned on Run C's own scene (dx 10 nm, the wire 162-1537 nm above the origin)
- **THEN** it holds 147 z nodes, 441 probes, from z 153 nm (the first node clear of the face at 150 nm) to z 1557 nm, and every probe's stored z is exactly the node coordinate it was built from (measured worst offset 0.0 m)

#### Scenario: the line ends where the geometry does
- **WHEN** the line is built for a scene whose wire top is at `hi` and whose gold top is at `lo`
- **THEN** its first z is the first node above `lo` by more than floating-point noise and its last is the last node at or below `hi + 3 dx`, and neither reaches into the absorber (`pml <= i <= len(nodes) - 1 - pml`), so a scene with no room is refused rather than probed inside a damped field

#### Scenario: the record is written from the layout that was solved
- **WHEN** the writer is given the layout the tiny (20 nm) scene's `solve_scene` returns, and its output is saved and read back
- **THEN** it writes one trace row per probe with that layout's own node z and component per row, the wire's span from the layout's `part_bounds` and the emitter's z from its source node, and `--axial-profile` on the written file prints one layer per z node (measured: 48 probes, 16 layers on that scene) -- the writer-to-reader path, not a hand-made record

### Requirement: The probes SHALL be priced in every estimate of the run that records them

`fdtd.memory_estimate` and `fdtd.estimate_run` SHALL take `probes` (and `probe_steps`, defaulting to the run's own step count) and count `4 x probes x probe_steps` bytes, the same four bytes a float32 trace is, because that is what the run holds in device memory and what `MAX_BYTES` must see. `fdtd_layout.estimate` SHALL pass the planned layout's own probe count, and the worker SHALL pass its layout's, so the product's memory check prices the traces it will record. The bench script's ladder SHALL price the same line the run will place, by building it from the plan through one function both paths call.

#### Scenario: a run's estimate grows by its traces
- **WHEN** `fdtd.estimate_run` is called on the same nodes with `probes=0` and with `probes=n`
- **THEN** the two `bytes` differ by exactly `4 n steps`, and `dt` and `steps` do not change: probes cost memory and no time

#### Scenario: the ladder prices the run's own line
- **WHEN** `--estimate --axis-probes` is read
- **THEN** each rung's bytes include `4 x 3 x layers x steps` for the line that rung's plan would place

### Requirement: The axial profile SHALL be read from the saved record, with no second run

`scripts/paper1_nanowire.py --axial-profile PATH [--axial-frequency-nm NM] [--axial-png OUT]` SHALL read a `--axis-probes` record and print, for every z layer, the power `|E|^2` of each component and their sum, normalised to the line's largest total, at the record's own drive frequency unless `--axial-frequency-nm` is given. Each probe's amplitude SHALL be the windowed DFT of its trace over the part of the record after the saved `start` step (the drive-free tail), not over the whole record. The profile SHALL also report the z of every interior local maximum of the summed power inside the wire's own span -- read from the record's `wire_z_m`, not retyped -- and a maximum at either end of the line SHALL NOT be reported as one, since it is the edge of the sampled range and not a maximum of the profile. A record without probe traces SHALL be refused with a message naming what it lacks. The PNG, which marks the wire's ends and the emitter's z from the record, SHALL be a convenience: a machine without matplotlib SHALL lose the picture and print the number.

#### Scenario: a standing wave's antinodes are where the physics puts them
- **WHEN** the record's traces are a standing wave of known wavelength and phase, sampled at the line's nodes
- **THEN** the reported maxima sit on the antinodes the wavelength and phase decide, within half a cell, and the count matches -- with the fixture's phase chosen so no antinode is at the centre of the line, so a reader that reported the centre would fail

#### Scenario: the profile reads the tail it was told to
- **WHEN** the same trace is read with the record's own `start` and with a later one
- **THEN** both are read from their own first step, and a `start` past the record's end is refused rather than silently reading nothing

#### Scenario: a record with no probes is refused, not read as empty
- **WHEN** `--axial-profile` is given a `--save-tail` record written without `--axis-probes`
- **THEN** it exits non-zero naming the missing traces, and `--axial-profile` together with any arm that would build a scene (`--ringdown`, `--estimate`, `--spectrum`, `--tiny`, `--save-tail`, `--save-series`, `--from-series`, `--collection-na`, `--locator-tau`, `--axis-probes`) SHALL be refused in `arm_conflicts`, so a flag that cannot be read is never accepted and ignored
