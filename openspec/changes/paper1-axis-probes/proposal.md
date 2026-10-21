# The 918 nm mode's antinodes, read off one ring-down

## Why

Run C (2026-10-21) measured the paper's nanowire cavity's Purcell factor as
F = 7.4 where the paper reports 17. Its Q and its wavelength match, so the
cavity is the paper's cavity; what is most likely wrong is where the emitter
sits. The emitter is a dipole 430 nm below the top facet (the paper's figure,
+-30 nm), and a dipole's Purcell factor is proportional to the local density of
states at its own position: an emitter parked on a node of the mode it drives
couples to very little of it, and one on an antinode to the most. Nothing in the
run so far says where those antinodes are -- the fit reads Q and frequency, not
a field profile -- so the next run cannot put the emitter on one except by
guessing.

The field itself is already in the record. A point probe costs four bytes a
step, and the run that produced the tail had every grid node in memory. What was
missing is where the wire's axis is (a grid line, not a scene coordinate) and a
reader that turns a trace into a profile.

## What changes

`scripts/paper1_nanowire.py --ringdown --save-tail PATH --axis-probes` places one
probe per E component (`Ex`, `Ey`, `Ez`, never a magnitude) on every z node from
the gold's top face to three coarse cells above the wire's top facet, at the
source's own in-plane node -- which is the wire's axis, since the layout builds
the scene around the source's corner. It stores the traces in the saved tail
record with each probe's node z, so the profile is read from the record and not
from a second run. The record is written from the layout the solver actually
ran on: its probes are `(i, j, k, component)` four-tuples, and `layout()` now
passes its plan's `part_bounds` through so the writer reads the wire's span from
the solved scene instead of planning it again.

`--axial-profile PATH [--axial-frequency-nm NM] [--axial-png OUT]` reads that
record: a windowed DFT of each trace over the drive-free tail at the drive's
frequency (or at NM), squared, split by component and summed; the z of every
interior local maximum inside the wire; and, given a path, a PNG marking the
wire's ends and the emitter's z -- the picture the owner asked for.

The probes are priced where the run is priced: `fdtd.estimate_run` and
`fdtd_layout.estimate` count `probes x steps x 4` bytes, the worker passes its
planned probe count, and the ladder prices the very line the run will place.

## Cost

Run C's line, planned on its own scene (its dx-10 flags; the wire 162-1537 nm
above the frame origin, the gold's top face at 150 nm), is 147 z nodes -- 441
probes with their three components -- from 153.000 nm, the first node above the
face (the face node itself is read as on the mirror, whose field is zero: see
the spec's "clear of" clause), to 1557.000 nm, three coarse cells
(30 nm) above the wire's top. All 441 probes land on the node they were placed
from: each requested z is a member of the plan's own z list and the
nearest-node lookup that places a probe returns that node (worst
|node z - requested z| = 0.0 m over the 441; 0 lookups chose a neighbour). The
run path additionally refuses a layout whose node lists differ from the
plan's, so the line the solver records is on the nodes it integrates. At Run
C's 121,121 steps that is 441 x 121,121 = 53,414,361 samples, 213.7 MB in the
solver's float32 traces, and the estimate reports those bytes so the next
run's memory check sees them: about a fourteenth of the bench's 3 GB cap,
against the 14.4M-cell grid's own share.

## What this does not do

It does not run on the cluster or the Jetson: no GPU solve is started here, and
the tier-1 tests read a saved record's arithmetic, not a device. It does not
choose the next emitter position either -- the profile measures where the
bright spots are and the owner decides whether to move the emitter onto one,
since moving the emitter is a different measurement from the one the paper's
figure 430 nm below the top describes.
