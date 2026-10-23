#!/usr/bin/env python3
"""Bench: paper 1's nanowire cavity -- the scene, the estimate, and the measurement.

Paper 1 is arXiv:2601.04440 (a hexagonal InP nanowire quantum-dot cavity on a
gold mirror). Its figure of merit is a Purcell factor at one band, and the
measurement arm for that lives in :mod:`fdtd` (``record_drive``,
``delivered_power``, ``dipole_moment``, ``purcell_factor``, and the band-wide
``purcell_spectrum``). This script is the scene those are pointed at.

The scene is built the way the product builds one, because the product already
has the machinery this needs and a second implementation would only be a second
set of rules: three closed triangle meshes (a gold slab, the SiO2 buffer, a
hexagonal InP prism) go to :func:`fdtd_layout.layout`, which pads the domain by
``OPEN_PAD_CELLS`` (16 cells) on every side, grades the grid, refines the thin
oxide to four cells and the gold to ``CELLS_PER_SKIN`` of a skin depth, and
returns the nodes, the voxelised permittivity, the metal masks and the source's
own node. The estimate is :func:`fdtd.estimate_run` over that same layout
against the same caps the web app enforces, so the grid this prints is a grid
the web form would refuse or accept -- not this script's opinion of one.

Geometry. Every dimension is an argument, and the defaults are the numbers the
row pins from the PDF -- a wire of 1375 nm by 420 nm, a 12 nm SiO2 buffer over
the gold, a dipole on the axis 30 nm below the top facet, and a 10 nm mesh in
the wire. The 420 nm is read through the paper's own relation for the wire --
"The diameter of a hexagonal nanowire is related to a cylindrical nanowire by
2s = 1.14 D", s the hexagon's side -- so it fixes the corner-to-corner width at
1.14 x 420 = 478.8 nm and the flat-to-flat at 414.6 nm, and ``report`` prints
both. ``--hex corners`` and ``--hex flats`` read the quoted number off that axis
directly instead, for the comparison runs. ``--hex-rotation`` turns the prism
about its axis. A grown wire's crown is NOT modelled: the paper's peak ~17 is
the idealised flat-facet cavity, and its own crown sweep keeps a real wire's
enhancement above 10 while the crown stays under 90 nm.

Why the grid is the shape it is. x and y are uniform at ``--dx``, and so is z
until a rule tightens it: the 12 nm oxide is thinner than four cells, so
``MIN_CELLS`` gives it four cells of 3 nm, and the gold is refined to a quarter
of a skin depth at the band's top frequency. The finest spacing anywhere sets
the timestep, so it is the OXIDE that sets the step count, not the wire: --dx
buys cells, the oxide's own splitting buys steps. Both appear in the printed
estimate, which is why the estimate is printed before anything runs.

The record's resolution, printed by every real run. N steps resolve a frequency
step of 1/(N dt), so dlambda = lambda^2/(c N dt), and the report also prints the
steps that resolving the paper's 4 nm mode would need. The paper's mode is
narrower than any record the tier-2 envelope allows (the caps are 8e6 cells,
20,000 steps, 3 GB and 1800 s), so a run must say in its own output which regime
it is in rather than quote the paper's number anyway.

What this measures, and what it does not. ``--spectrum`` runs a band-limited
pulse centred in its own record and reads the enhancement bin by bin
(``purcell_spectrum``): that is the arm the paper's 830-970 nm figure can be
read against, and its bins resolve tens of nanometres, so the paper's 4 nm mode
is measured through a window wider than the mode. The CW arm (``--periods``)
measures a settled steady state at one frequency per run. Neither substitutes
for the other; the printed resolution says which one a given run is.

``--ringdown`` is the third reading, and the one that answers the linewidth
question without a longer record: the same pulse, with ``--tail-periods`` of
record after its drive is over, fitted as one decaying sinusoid
(``fit_decay``). Its Q = pi f tau with a covariance error, and it says which of
the two outcomes it reached -- a Q with an error, or an unresolved tail with the
bound that is all the record can give. That is what the lead's ruling asks for:
the 4 nm line is read from the decay rather than bought with steps the cap
refuses. What its error is not: the model's covariance knows about the residual
and nothing about the geometry, so the same arm on ``--vacuum`` (no cavity at
all) is the reading to put beside it.

``--collection-na 0.8`` is the collection reading, an addition to the CW arm
and off by default: a phasor box (``fdtd.simulate``'s ``phasor_box``) around
the WHOLE scene -- the wire, the oxide and the entire gold pad, every face in
the open vacuum pad -- read over the CW tail, and ``farfield``'s fraction of
the upward far-field power inside that aperture. The mirror is inside the box
rather than modelled by a PEC image, which on a finite Lorentz-Drude pad was
measured 0.14 off (openspec change ``paper1-collection``,
``tests/test_phasor_mirror.py``). The fraction is the finite pad's, so it
owes the same pad convergence the F reading does.

The CW arm also says WHY its reading is or is not still moving (change
``cw-settling``): after the ``measured`` and ``settling`` lines, which are
printed as they always were, it prints what the window's phase alone does to
the reading (a 2f swing: the Hann's leakage of the -f image times the source
field's quadrature ratio, measured and predicted), the same reading at cuts
locked to the end's drive phase through the record, a fit of that curve
(F_inf, tau_s and their errors), and one class: settled, approaching (F_inf
then an extrapolation), growing without a limit (never a Purcell factor), or
unresolved. Where one time constant does not hold the whole curve, the
forecast is refit over the record's last half, and an F_inf the record has
already passed is called inconsistent rather than printed as the
extrapolation (change ``settling-tail-fit``). ``--locator-tau`` sets tau_s
against a ring-down's tau in words; ``--save-series`` keeps the record (off
by default) and ``--from-series`` re-reads it without a run.

Item (7) of the handoff (a dielectric that is a function, not a scalar) does not
bind the measurement this script makes: the dipole sits on the wire's axis, 30
nm below the top facet and clear of every graded interface, so the source cell's
own medium is homogeneous InP and ``sqrt(eps[cell])`` is the host index the
normalisation needs. The field update around it is still the product's scalar
path.

Exit codes: 0 measured, 1 a control failed, 2 refused (over a cap, or over the
tier-2 envelope), 3 measured but the record does not pin what was asked for (the
tail arm's unresolved reading).

The tier-2 commands (the Jetson bench, one run at a time; the lead runs them).
The bench's python is the worker venv -- that is where cupy lives, and the
script's own imports expect the repo's paths to be kept:

    rsync -a --relative scripts/paper1_nanowire.py dockerfiles/maxwell_solver/ \
        jetson:~/mse-bench/paper1/
    ssh jetson 'cd ~/mse-bench/paper1 && ulimit -v 5000000; nice -n 10 \
        timeout 30m ~/mse-worker/venv/bin/python scripts/paper1_nanowire.py \
        --xp cupy --spectrum --dx 10 --mirror-pad 400'

The ring-down command is the same with ``--ringdown`` in place of ``--spectrum``
(the two arms read one record two ways, so they are two runs) plus whatever
``--tail-periods`` the time budget allows: at dx 15 a 20-period tail puts the
pulse arm at 18,649 steps and 1,317 s estimated, still inside the 1,800 s
envelope, where dx 12 does not fit with a tail at all.

The stage-2 ladder (step 3) is ``--height`` over the paper's H, one command per
rung.
"""
from __future__ import annotations

import argparse
import inspect
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "dockerfiles" / "maxwell_solver"))
import fdtd         # noqa: E402
import fdtd_grid    # noqa: E402
import fdtd_layout  # noqa: E402

C = fdtd.C0
NM = 1e-9
N_INP = 3.44          # wurtzite InP at 900 nm
HEX_NW_RATIO = 1.14   # the paper's own relation for the wire: 2s = 1.14 D
N_SIO2 = 1.45         # thermal oxide at 900 nm
TIER2_SECONDS = 1800.0
SETTLED_PERIODS = 16.0  # measured: a hard-started CW drive's settling, in periods
PAPER_FWHM_NM = 4.0
DRIVE_FLOOR = 1e-4      # of the drive's own peak: where the pulse is over
MIN_TAIL_PERIODS = 6.0  # a decay needs at least this many periods to be fitted
TAU_EDGE = 20.0         # the fit's own tau grid reaches this far in windows
ERR_FLOOR = 1e-9        # of its own value: below this an error is the pinv's
DECAY_PAD = 8           # the decay scan's omega bins are 1/(8 windows) apart
RIVALS = 3              # separated minima of the scan refined against the fit
RIVAL_ITERATIONS = 20   # golden steps for each of them (the fit's own is 40)
AMBIGUITY_RATIO = 12.0  # a rival minimum under 12 x the fit's residual is a tie:
                        # measured (decay-fit-basin), 14 of the 16 fits >10% off
                        # in 105 tails had a rival at 11.1 x or less
UNRESOLVED = 3          # measured, but the record's tail leaves Q unpinned
DEFAULT_TAIL_PERIODS = 20.0   # asked for only by --ringdown
QD_DEPTH_TOL_NM = 1.0   # how far the emitter's realised depth may sit from --qd-below
# The CW arm's settling curve (change cw-settling).
SETTLING_STEP = 0.05    # the curve's cuts: one every 5% of the record
MIN_FIT_CUTS = 5        # three parameters and two degrees of freedom
SIGNIFICANCE = 3.0      # a difference counts when it is this many of its own errors
APPROACH_EDGE = 50.0    # the approach fit's rate grid reaches +-50 / (fitted span)
# The forecast (change settling-tail-fit), each value measured there on sums of
# two exponentials with known limits.
TAIL_SHARE = 0.5          # a misfit curve's forecast is refit over its last half
INCONSISTENT_SIGMA = 2.5  # F_inf behind a moving end by more than this many errors:
                          # correct forecasts reached 1.92 (1623 noisy records),
                          # the 600-period curve's old forecast 3.33
FORECAST_RESIDUAL = 3.0   # the forecast's residual over this x the floor: forecasts
                          # within 3 errors of their limit reached 2.74 (167 of
                          # them), the 10 + 60-period curve's tail misses by 3.7
                          # errors at 3.84
PHASE_CUTS = 32         # cuts across the last drive period for the phase scan
# the CW drive's own turn-on, read from the profile rather than retyped
CW_RAMP_PERIODS = float(
    inspect.signature(fdtd.cw_profile).parameters["ramp_periods"].default)


def paper_clock(args, request):
    """The paper's mode as the two numbers a decay can be read against.

    Both are DERIVED from the paper's own quoted pair -- 4 nm at 900 nm -- and
    neither is asserted: Q = lambda / dlambda = 225 and the amplitude decay time
    is Q / (pi f) = 215 fs. The ring-down arm prints them next to what it fits, so
    a reader can see how much of the decay the record's own tail holds.
    """
    q = args.wavelength / PAPER_FWHM_NM
    tau = q / (math.pi * float(request["frequency_hz"]))
    return q, tau


def part(vertices, faces):
    """A closed triangle mesh -- the form the layout's parts take.

    ``trimesh`` is the product's own part type (``mse_parts`` loads uploads into
    it), so the bench hands the layout exactly what an upload would be, and the
    mesh carries the ``bounds`` and ``volume`` the controls read.
    """
    return trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float).reshape(-1, 3),
                           faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
                           process=False)


def _box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return part(v, f)


def _hex_prism(z0: float, z1: float, radius: float, rotation_deg: float):
    """A regular hexagonal prism about the z axis: 12 vertices, 20 triangles.

    Its analytic volume is ``3*sqrt(3)/2 * radius**2 * (z1 - z0)``, which the
    tiny scene checks the mesh against.
    """
    angles = [math.radians(rotation_deg) + k * math.pi / 3.0 for k in range(6)]
    ring = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    v = [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]
    f = [(0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4),          # bottom, -z
         (6, 7, 8), (6, 8, 9), (6, 9, 10), (6, 10, 11)]        # top, +z
    for k in range(6):
        a, b = k, (k + 1) % 6
        f += [(a, b, b + 6), (a, b + 6, a + 6)]
    return part(v, f)


def band_hz(args) -> float:
    """Half the pulse's band, in Hz, from the band's two vacuum wavelengths."""
    lo = C / ((args.wavelength - args.band_half) * NM)
    hi = C / ((args.wavelength + args.band_half) * NM)
    return (lo - hi) / 2.0


def _nm_cells(nm, dx) -> int:
    """A bench boundary distance in nm as whole cells at this run's ``dx``."""
    return int(round(float(nm) / float(dx)))


def boundary_cells(args) -> dict:
    """This run's open boundary as cell counts: ``pad_cells``, ``pml_cells``.

    ``--pad-nm`` and ``--pml-nm`` name the boundary's DISTANCE in nanometres,
    which is the physical parameter here: ``fdtd_layout.OPEN_PAD_CELLS`` is a
    cell COUNT, so the product's own pad is 320 nm at dx 20 -- the run r20
    measured F = 4.38 on -- and 160 nm at dx 10, and the absorber with it. A
    dx-10 record run on those defaults would change the open boundary's
    distance at the same time as the wire's resolution, and a difference from
    r20 could then be either one (this scene's own record: 200 nm against 400
    nm of pad moved F by 16%). These two flags hold the distance, so dx is the
    only thing a dx-10 record changes.

    A distance that is not a whole number of cells is rounded to the nearest
    one and the run reports the distance it really held (``report``'s
    ``boundary`` line), so a log never claims a boundary the grid did not have.
    Below one cell is refused in ``arm_conflicts``: a pad of no cells is no
    boundary at all.

    ``--pml-nm`` and ``--pml`` set the same absorber. The nm flag is the answer
    when both are given; ``arm_conflicts`` refuses a run that names two
    different ones, so the rule is unreachable from the parser. Defaults are
    the product's own, read here at call time.
    """
    pad_nm = getattr(args, "pad_nm", None)
    pml_nm = getattr(args, "pml_nm", None)
    pml = getattr(args, "pml", None)
    if pml_nm is None:
        pml_cells = fdtd.DEFAULT_PML_CELLS if pml is None else int(pml)
    else:
        pml_cells = _nm_cells(pml_nm, args.dx)
    return {"pad_cells": (fdtd_layout.OPEN_PAD_CELLS if pad_nm is None
                          else _nm_cells(pad_nm, args.dx)),
            "pml_cells": pml_cells}


def gold_metal(args) -> str:
    """The library name of the gold this run uses (``--gold-data``).

    ``rakic`` is Rakić et al. 1998, the default the earlier runs used;
    ``palik`` is the second model, fitted to Palik's 1985 table over its
    653-1127 nm rows (``fdtd.PALIK_1985``).
    """
    return "Au" if args.gold_data == "rakic" else "Au-palik"


def scene(args):
    """The three parts, their materials, and the request the layout reads.

    Positions are in the layout's own frame, whose origin is the parts'
    bounding-box minimum corner. The mirror's footprint is the wire's
    circumradius plus ``--mirror-pad``; the layout then pads the whole extent by
    ``OPEN_PAD_CELLS`` cells, absorber included, so the parts never reach the
    absorber.
    """
    # --diameter names one width and --hex says which: by default the paper's own
    # relation for the wire (2s = 1.14 D), otherwise the quoted number taken
    # directly across the corners or across the flats.
    if args.hex == "paper":            # the paper's relation: 2s = 1.14 D
        radius = args.diameter * HEX_NW_RATIO * NM / 2.0
    elif args.hex == "flats":          # --diameter across flats -> circumradius
        radius = args.diameter * NM / (2.0 * math.cos(math.pi / 6.0))
    else:                              # --diameter across corners
        radius = args.diameter * NM / 2.0
    z_gold1 = args.gold * NM
    z_ox1 = z_gold1 + args.oxide * NM
    z_top = z_ox1 + args.height * NM
    half = radius + args.mirror_pad * NM
    names = ["mirror", "oxide", "wire"]
    parts = [_box((-half, -half, 0.0), (half, half, z_gold1)),
             _box((-half, -half, z_gold1), (half, half, z_ox1)),
             _hex_prism(z_ox1, z_top, radius, args.hex_rotation)]
    materials = {"mirror": gold_metal(args), "oxide": N_SIO2, "wire": N_INP}
    if args.vacuum:                    # the control: the same meshes, no material
        materials = {name: 1.0 for name in names}
    request = {
        "boundary": "open", "source_kind": "point", "dx_m": args.dx * NM,
        "position": (half, half, z_top - args.qd_below * NM),
        "probes": (), "profile": args.profile, "frequency_hz": None,
        "bandwidth": None, "polarisation": args.axis,
        "samples": None, "sample_duration_s": None,
        "duration_s": 1e-15, "coatings": None,
        # --qd-below is the paper's geometry, so the emitter's own z is pinned
        # as a grid node (see ``fdtd_layout.plan``) and the realised depth is
        # refused, not reported, when the mesh cannot put it there.
        "pin_source_z": True,
    }
    request["frequency_hz"] = C / (args.wavelength * NM)
    request["bandwidth"] = band_hz(args)
    # The product refuses a band a metal was not fitted over; this run drives
    # the same band, so it meets the same refusal here, before the scene is
    # built. ``source_band`` is that one definition -- the pulse's 1% spectrum,
    # not the nominal half-width below -- so this script and the app agree on
    # what a pulse asks for, and a cw or drawn source asks nothing. A
    # --vacuum control names no metal and asks nothing of one.
    band = fdtd.source_band(request["frequency_hz"], request["profile"],
                            request["bandwidth"])
    if band is not None and isinstance(materials["mirror"], str):
        fdtd.check_metal_band(materials["mirror"], *band)
    cells = boundary_cells(args)
    geom = {"z_top": z_top, "z_oxide": (z_gold1, z_ox1), "radius": radius,
            "half": half,
            # the open boundary the layout is built with: the cells, and the
            # distance they are (rounded to the nearest cell, so a log can name
            # the boundary it held rather than the one it asked for)
            "pad_cells": cells["pad_cells"], "pad_nm": cells["pad_cells"] * args.dx,
            "pml_cells": cells["pml_cells"], "pml_nm": cells["pml_cells"] * args.dx}
    return names, parts, materials, request, geom


def planned_scene(args):
    """The product's own plan, and its cheap estimate (no voxeliser).

    With ``--axis-probes`` the returned plan and request carry the axial line.
    The line is placed on the plan's own nodes -- a probe must sit on one --
    and the plan is built again with those points, so the plan the estimate
    prices is the plan whose probes the run will record. The two plans' node
    lists are compared: placing probes may not move the grid, and a grid that
    moved would put the line on nodes this plan does not have.
    """
    names, parts, materials, request, geom = scene(args)
    plan = fdtd_layout.plan(names, parts, request, length_scale=1.0,
                            materials=materials, pad_cells=geom["pad_cells"],
                            pml_cells=geom["pml_cells"])
    if getattr(args, "axis_probes", False):
        request = with_axis_probes(plan, request)
        lined = fdtd_layout.plan(names, parts, request, length_scale=1.0,
                                 materials=materials,
                                 pad_cells=geom["pad_cells"],
                                 pml_cells=geom["pml_cells"])
        for axis, (a, b) in enumerate(zip(plan["nodes"], lined["nodes"])):
            if not np.array_equal(a, b):
                raise SystemExit(
                    f"internal: the axial probes changed the grid on"
                    f" {'xyz'[axis]} ({len(a)} nodes without the line, {len(b)}"
                    f" with it); the line would not sit on the nodes this plan"
                    f" prices")
        plan = lined
    return plan, request, geom


def solve_scene(args):
    """The product's own layout: grid, voxelised scene, source node, probes."""
    names, parts, materials, request, geom = scene(args)
    planned = None
    series = getattr(args, "save_series", None)
    source = getattr(args, "from_series", None)
    locator = getattr(args, "locator_tau", None)
    if getattr(args, "axis_probes", False):
        # The axial line is placed from a PLAN, before the voxeliser runs: a
        # probe must sit on a grid node, and only the plan's node list says
        # where the nodes are. Probes do not enter the grid's construction, so
        # the layout below builds the same nodes -- and that is checked rather
        # than assumed (the test that pins a probe to its node is why).
        planned = fdtd_layout.plan(names, parts, request, length_scale=1.0,
                                   materials=materials,
                                   pad_cells=geom["pad_cells"],
                                   pml_cells=geom["pml_cells"])
        request = with_axis_probes(planned, request)
    lay = fdtd_layout.layout(names, parts, request, length_scale=1.0,
                             materials=materials, pad_cells=geom["pad_cells"],
                             pml_cells=geom["pml_cells"])
    if planned is not None:
        for axis, (a, b) in enumerate(zip(planned["nodes"], lay["nodes"])):
            if not np.array_equal(a, b):
                raise SystemExit(
                    f"internal: the axial probes changed the grid on {'xyz'[axis]}"
                    f" ({len(a)} nodes planned, {len(b)} laid out); the line would"
                    f" not sit on the nodes this run uses")
    return lay, request, geom


AXIS_PROBE_ABOVE_CELLS = 3   # how far above the wire's top facet the line ends
AXIS_COMPONENTS = fdtd.COMPONENTS   # every E component, never reduced to |E|


def axis_probe_points(planned, request) -> list:
    """The axial line: three probes per z node, from the mirror's top face up.

    The line runs on the source's own in-plane position (the wire's axis) from
    the first node clear of the gold's top face to ``AXIS_PROBE_ABOVE_CELLS``
    coarse cells above the wire's top facet, or as far as the absorber allows,
    whichever comes first. "Clear of" is not the same as "compared greater": a
    graded axis puts a node within float noise of the face, and the layout's
    point-in-part test reads that node as on the mirror -- a conductor, whose
    field is zero, so its trace would read as no response. The line therefore
    starts a millionth of a cell above the face. Each layer carries one probe
    per E component: a magnitude would throw away which way the field points,
    and the mode's E_z is what an emitter on an antinode couples to.

    The points are given in the layout's frame, and every one of them is a node
    coordinate, so ``plan``/``layout`` locate each probe on the node it names
    rather than snapping it to a neighbour.
    """
    nodes = planned["nodes"]
    z = np.asarray(nodes[2], dtype=float)
    lo, hi = float(planned["part_bounds"][-1][0][2]), \
        float(planned["part_bounds"][-1][1][2])        # the wire, the last part
    gold_top = float(planned["part_bounds"][0][1][2])   # the first part's top face
    dx = float(request["dx_m"])
    pml = int(planned["pml_cells"])
    start = int(np.searchsorted(z, gold_top + 1e-6 * dx, side="left"))
    stop = int(np.searchsorted(z, hi + AXIS_PROBE_ABOVE_CELLS * dx,
                               side="right"))
    start = max(start, pml)
    stop = min(stop, len(z) - pml)
    if not stop > start:
        raise SystemExit(
            f"refused: the axial line has no nodes between the mirror's top"
            f" face ({gold_top / NM:.3f} nm) and {AXIS_PROBE_ABOVE_CELLS} cells"
            f" above the wire top ({hi / NM:.3f} nm) outside the absorber")
    # the source's own node, both in plane: the line runs along the wire's axis
    x = float(nodes[0][planned["cell"][0]])
    y = float(nodes[1][planned["cell"][1]])
    # a list, not a generator: the request is read again by the estimate and
    # by the arm's own refusals, and a consumed generator would price nothing
    return [[x, y, float(z[i]), component]
            for i in range(start, stop) for component in AXIS_COMPONENTS]


def with_axis_probes(planned, request) -> dict:
    """``request`` with the axial line as its probes (a copy, not mutated).

    One function for the run path and the ladder, so a priced rung and the run
    it prices place the same line.
    """
    out = dict(request)
    out["probes"] = axis_probe_points(planned, request)
    return out


def steps_for(args, request, dt: float) -> int:
    """The record this arm wants: periods at CW, a sigma span plus tail for a pulse.

    The pulse spans ``--pulse-span`` sigma of its own envelope and the drive is
    centred at 5 sigma, so a span of 8 leaves about one period of tail. The
    ``--tail-periods`` after it are whole drive periods of record in which the
    pulse is over and the cavity is free to decay -- what the ring-down arm
    reads, and what the band arm must not have (see ``arm_conflicts``).
    """
    if args.profile == "pulse":
        sigma = 1.0 / (math.pi * request["bandwidth"])
        steps = int(math.ceil(args.pulse_span * sigma / dt))
        return steps + int(math.ceil(args.tail_periods / (request["frequency_hz"] * dt)))
    return int(math.ceil(args.periods / (request["frequency_hz"] * dt)))


def steps_for_4nm(dt: float, wavelength_nm: float) -> int:
    """The steps a record needs before it can resolve the paper's 4 nm mode."""
    periods = wavelength_nm / PAPER_FWHM_NM
    return int(math.ceil(periods / (C / (wavelength_nm * NM) * dt)))


def run_limits(args) -> dict:
    """The per-run ceilings this bench run is measured against.

    ``max_steps`` and ``max_seconds`` are the product's own unless the bench flags
    raised them. Those two exist for the one case the owner approved on
    2026-10-16: a settled CW record at 993 nm is 39,650 steps and about 47
    minutes, past both of the product's caps, and only a bench run may go longer.

    ``max_cells`` and ``max_bytes`` are ``None`` unless the bench flags set them,
    and ``None`` is not "no ceiling": it is the device's budget, read on the
    device by ``fdtd.check_limits`` (the 8,000,000-cell and 3.0 GB constants are
    gone; the owner, 2026-10-23: "a run should use all available resources").
    A flag sets an explicit ceiling instead, and it may now also set one below
    the device's: the product has no constant for it to raise from. ``--max-bytes``
    names GIGABYTES (``--max-bytes 14`` is 14e9 bytes).

    One function because the estimator (``ladder``) and the run path (``filled``)
    must read the SAME ceilings: a priced rung that disagreed with the run it
    priced would be a second opinion rather than an estimate.
    """
    cells = getattr(args, "max_cells", None)
    steps = getattr(args, "max_steps", None)
    seconds = getattr(args, "max_seconds", None)
    nbytes = getattr(args, "max_bytes", None)
    return {"max_cells": None if cells is None else int(cells),
            "max_steps": (fdtd.MAX_STEPS if steps is None
                          else max(fdtd.MAX_STEPS, int(steps))),
            "max_seconds": (fdtd.MAX_SECONDS if seconds is None
                            else max(fdtd.MAX_SECONDS, float(seconds))),
            "max_bytes": None if nbytes is None else float(nbytes)}


def bench_ceilings(lim) -> list:
    """The ceilings ``run_limits`` reports that are not the product's own, as
    the parenthetical the bench's caps lines print: a step or time ceiling the
    flags raised, and a cell or byte ceiling they set. Empty when the run is
    measured against the product's own two constants and the device's budget."""
    raised = []
    if lim["max_steps"] > fdtd.MAX_STEPS:
        raised.append(f"{lim['max_steps']:,} steps")
    if lim["max_seconds"] > fdtd.MAX_SECONDS:
        raised.append(f"{lim['max_seconds'] / 60:.0f} min")
    if lim["max_cells"] is not None:
        raised.append(f"{lim['max_cells']:,} cells")
    if lim["max_bytes"] is not None:
        raised.append(f"{lim['max_bytes'] / 1e9:.1f} GB")
    return raised


def phasor_estimate(plan) -> dict:
    """``fdtd.estimate_run``'s phasor-box arguments for a collection plan."""
    if plan is None:
        return {}
    return {"phasor_box": plan["box"], "phasor_frequencies": [plan["frequency"]],
            "phasor_steps": plan["steps"]}


def filled(args, lay, request, geom=None):
    """The layout's estimate, the caps' verdict, and the record's own reach.

    With ``--collection-na`` it also places the phasor box (``geom`` is then
    required) and prices it into the estimate the caps are checked against;
    the plan is ``"phasor"`` in the result, None without the flag.
    """
    est = fdtd.estimate_run(lay["nodes"], duration=request["duration_s"],
                            boundary=request["boundary"], metals=lay["metals"],
                            pec=lay["pec"], probes=len(lay["probes"]))
    steps = steps_for(args, request, est["dt"])
    request["duration_s"] = steps * est["dt"]
    phasor = None
    if getattr(args, "collection_na", None) is not None:
        if geom is None:
            raise ValueError("the collection arm places its box from the scene's"
                             " geometry; pass geom")
        phasor = collection_plan(args, lay["nodes"], geom, request, est["dt"],
                                 steps, eps=lay.get("eps"), metals=lay["metals"])
    est = fdtd.estimate_run(lay["nodes"], duration=request["duration_s"],
                            boundary=request["boundary"], metals=lay["metals"],
                            pec=lay["pec"], probes=len(lay["probes"]),
                            **phasor_estimate(phasor))
    exceeded = None
    try:
        fdtd.check_limits(est, **run_limits(args))
    except fdtd.EstimateError as exc:
        exceeded = str(exc)
    dlambda = (args.wavelength * NM) ** 2 / (C * est["steps"] * est["dt"]) / NM
    out = {"est": est, "steps": est["steps"], "exceeded": exceeded,
           "dlambda_nm": dlambda}
    if phasor is not None:
        out["phasor"] = phasor
    return out


def emitter_depth_nm(lay, geom) -> float:
    """How far below the top facet the source node really sits, in nm.

    One function because two readers ask it: the report prints this depth, and
    ``refusals`` measures the run against it. Read off the layout's own node
    list and source cell, so neither reader can retype the expression and drift
    from the plan they are describing.
    """
    z = np.asarray(lay["nodes"][2])
    return float((geom["z_top"] - z[int(lay["cell"][2])]) / NM)


COLLECTION_CLEAR_CELLS = 2   # node layers between a phasor face and the nearest part


def scene_extent(geom):
    """The parts' bounding box in the layout's frame, metres: ``(lo, hi)``.

    The layout's origin is the parts' bounding-box minimum corner (``scene``),
    so the box runs from 0 to the mirror's width in x and y and from the
    gold's underside (0) to the wire's top facet in z.
    """
    width = 2.0 * geom["half"]
    return (0.0, 0.0, 0.0), (width, width, geom["z_top"])


def _pick_layers(lo_range, hi_range):
    """Face layers ``(lo, hi)`` nearest the middle of each range, ``lo + hi`` even.

    ``simulate`` spells a box as ``centre +/- half_extent``, so the two faces
    on an axis must be an even number of layers apart. None if no pair is.
    """
    def by_middle(span):
        a, b = span
        mid = 0.5 * (a + b)
        return sorted(range(a, b + 1), key=lambda k: (abs(k - mid), k))
    best = None
    for i, lo in enumerate(by_middle(lo_range)):
        for j, hi in enumerate(by_middle(hi_range)):
            if (lo + hi) % 2 == 0 and (best is None or i + j < best[0]):
                best = (i + j, lo, hi)
    return None if best is None else best[1:]


def collection_box(nodes, geom, pml_cells, *, eps=None, metals=None):
    """The phasor box of the collection arm: around the WHOLE scene, in vacuum.

    Returns ``{"box": (centre, half_extent), "lo", "hi", "clear", "absorber"}``
    (node layers, per axis) or raises ``fdtd.FDTDError`` saying which axis has
    no room. The six faces enclose the wire, the oxide and the entire gold pad,
    so every source of the scattered field -- the dipole, the wire's and the
    oxide's polarisation, the gold's induced currents -- is inside, and outside
    is the open vacuum pad: there the free-space transform ``farfield.py``
    applies is exact, with no mirror model at all (openspec change
    ``paper1-collection``: a PEC image of a finite Lorentz-Drude pad was
    measured +0.14 off on a small one).

    Each face sits in the middle of the free gap on its side: at least
    ``COLLECTION_CLEAR_CELLS`` node layers outside the parts, so its samples
    and the H half a cell outside it are vacuum, and inside the layers
    ``pml + 1 .. n - 2 - pml`` ``fdtd.phasor_box_layers`` allows. ``nodes`` are
    the layout's (or the plan's) node coordinates. With ``eps`` and ``metals``
    (a voxelised layout) the faces' own slabs are checked to be vacuum, so a
    frame slip cannot put a face through the gold unnoticed.
    """
    lo_m, hi_m = scene_extent(geom)
    pml = int(pml_cells)
    lo, hi, clear, absorber = [], [], [], []
    for a in range(3):
        z = np.asarray(nodes[a], dtype=np.float64)
        n = z.size
        tol = 1e-6 * float(np.min(np.diff(z)))
        first = int(np.searchsorted(z, lo_m[a] + tol, "right")) - 1   # node at/below
        last = int(np.searchsorted(z, hi_m[a] - tol, "left"))         # node at/above
        lo_range = (pml + 1, first - COLLECTION_CLEAR_CELLS)
        hi_range = (last + COLLECTION_CLEAR_CELLS, n - 2 - pml)
        picked = (None if lo_range[0] > lo_range[1] or hi_range[0] > hi_range[1]
                  else _pick_layers(lo_range, hi_range))
        if picked is None:
            raise fdtd.FDTDError(
                f"the collection box has no room on axis {'xyz'[a]}: the parts"
                f" span node layers {first}..{last} of {n}, a face must be"
                f" {COLLECTION_CLEAR_CELLS} layers outside them and inside"
                f" {pml + 1}..{n - 2 - pml} (the {pml}-cell absorber and the H"
                f" half cell a face reads), which leaves layers"
                f" {lo_range[0]}..{lo_range[1]} below and"
                f" {hi_range[0]}..{hi_range[1]} above; widen --pad-nm or thin"
                f" the absorber")
        lo.append(picked[0])
        hi.append(picked[1])
        clear.append((first - picked[0], picked[1] - last))
        absorber.append((picked[0] - pml, n - 1 - pml - picked[1]))
    box = (tuple((l + h) // 2 for l, h in zip(lo, hi)),
           tuple((h - l) // 2 for l, h in zip(lo, hi)))
    if eps is not None:
        eps = np.asarray(eps)
        solid = np.zeros(eps.shape, dtype=bool)
        for region in dict(metals or {}).values():
            solid |= np.asarray(region, dtype=bool)
        for a in range(3):
            for layer in (lo[a], hi[a]):
                shell = [slice(lo[b] - 1, hi[b] + 2) for b in range(3)]
                shell[a] = slice(layer - 1, layer + 2)
                shell = tuple(shell)
                if np.any(eps[shell] != 1.0) or np.any(solid[shell]):
                    raise fdtd.FDTDError(
                        f"the collection box's face at node layer {layer} on"
                        f" axis {'xyz'[a]} touches material: the free-space"
                        f" transform needs vacuum outside every source")
    return {"box": box, "lo": tuple(lo), "hi": tuple(hi), "clear": clear,
            "absorber": absorber}


def tail_steps(args, request, dt: float) -> int:
    """The CW reading's window in steps: ``fdtd._steady_tail``'s own rounding."""
    return int(round(int(args.window_periods) / (float(request["frequency_hz"]) * dt)))


def collection_plan(args, nodes, geom, request, dt: float, steps: int, *,
                    eps=None, metals=None):
    """The phasor box and the steps it reads, or None without --collection-na.

    The box reads the same Hann-windowed tail the CW reading averages, so the
    surface power and the delivered power are two readings of one window.
    """
    if getattr(args, "collection_na", None) is None:
        return None
    placed = collection_box(nodes, geom, geom["pml_cells"], eps=eps, metals=metals)
    tail = min(tail_steps(args, request, dt), int(steps))
    return {**placed, "frequency": float(request["frequency_hz"]),
            "start": int(steps) - tail, "steps": tail}


def collection_line(plan, geom) -> str:
    """The box the arm will record, printed before the run."""
    lo, hi = plan["lo"], plan["hi"]
    return (f"box       phasor faces at node layers x {lo[0]}..{hi[0]},"
            f" y {lo[1]}..{hi[1]}, z {lo[2]}..{hi[2]}: the wire, the oxide and"
            f" the whole {2 * geom['half'] / NM:.0f} nm mirror inside, all six"
            f" faces in vacuum, {min(min(c) for c in plan['clear'])} or more layers"
            f" clear of the parts and {min(min(c) for c in plan['absorber'])} or"
            f" more of the absorber; read over the last {plan['steps']:,} steps")


def collection(args, faces, frequency: float, delivered: float) -> dict:
    """The collection arm's reading of one run's phasor faces, printed.

    The fraction is ``farfield.collection_fraction``: the far-field power in
    the cone ``theta <= asin(NA)`` about +z over the power in the upper
    hemisphere. The sanity lines are the box's surface power against the
    power the source delivered (below 1 by what the gold absorbs: the wire and
    the oxide are lossless) and the full-sphere far field against the surface
    power (the transform's own closure, 1 for a box in vacuum).
    """
    import farfield     # the far-field module imports fdtd; only this arm needs it
    na = float(args.collection_na)
    cone = farfield.cone_power(faces, frequency, math.asin(na))
    upward = farfield.hemisphere_power(faces, frequency)
    sphere = farfield.cone_power(faces, frequency, math.pi)
    surface = farfield.surface_power(faces)
    fraction = farfield.collection_fraction(faces, frequency, na)
    print(f"collect   NA {na:.2f}: {fraction:.4f} of the upward far-field power"
          f" (cone {cone:.6e} W of {upward:.6e} W upward), {cone / sphere:.4f} of"
          f" all the power radiated")
    print(f"surface   upward {upward:.6e} W, total {surface:.6e} W through the"
          f" box ({upward / surface:.4f} of it upward)")
    print(f"sanity    box surface power / delivered power = {surface / delivered:.4f}"
          f" (below 1 by what the gold absorbs: the wire and the oxide are"
          f" lossless); full-sphere far field / surface"
          f" power = {sphere / surface:.4f}")
    return {"fraction": fraction, "cone": cone, "upward": upward,
            "sphere": sphere, "surface": surface, "delivered": delivered}


def refusals(args, measured, lay=None, geom=None) -> str:
    """The refusal both entry points owe before anything runs.

    Returns the refusal line, or "" when the run may go ahead. It is one
    function on purpose: the tier-1 arm cannot pass a scene that the Jetson arm
    would refuse, which is what the ``--tiny --periods 4000`` test pins.

    The tier-2 clock needs no check of its own here: with no bench ceiling
    raised, the product's caps carry it (``fdtd.MAX_SECONDS`` is the same 30
    minutes), so a run over the timeout is already ``measured["exceeded"]``. That
    equality is itself a control -- see ``tiny``, which fails if someone moves one
    clock without the other. ``--max-seconds`` is what breaks it, deliberately:
    a flagged bench run is longer than the launcher's own clock (r19 ran 30.9
    minutes under a 45-minute one), so for that run the launcher's ``MSE_CLOCK``
    and not this line is what bounds the wall time.

    With a layout in hand it also refuses a source that is not at the depth
    that was asked for. The z line is pinned (``fdtd_layout.plan``'s
    ``pin_source_z``), but ``graded_axis`` drops a line that lands out of range
    or within ``LINE_SNAP`` of the extent's edges, and then the source is
    silently snapped to the nearest node: a run of a different geometry than
    the one asked for, which is worse than a refusal because nothing downstream
    can see it. Measured on this scene before the pin: 39.99 nm at dx 20 and
    29.89 nm at dx 30 for a requested 30.
    """
    if measured["exceeded"]:
        return f"refused: {measured['exceeded']}"
    if lay is not None and geom is not None:
        depth = emitter_depth_nm(lay, geom)
        out = abs(depth - float(args.qd_below))
        if out > QD_DEPTH_TOL_NM:
            return (f"refused: the emitter sits {depth:.2f} nm below the top facet"
                    f" where --qd-below {args.qd_below:g} was asked, {out:.2f} nm"
                    f" out (tolerance {QD_DEPTH_TOL_NM:g} nm): the source is"
                    f" snapped to the nearest grid node, so this run is a"
                    f" different geometry and not a coarser one")
    return ""


def arm_conflicts(args) -> str:
    """The refusal the arms owe each other, before any scene is built.

    Argument-level, so it runs first and costs nothing. Three flags would
    otherwise be inert or quietly wrong: ``--tail-periods`` on a CW arm is never
    read by ``steps_for``; ``--ringdown`` without a pulse has no tail to read;
    and a tail on the band arm moves the pulse off the centre of its own record.
    That last one is the reason ``fdtd.delivered_power_spectrum`` carries a
    centred-drive requirement -- its own docstring measures a mostly-ring-down
    record at -94 dB, which this check keeps out of reach rather than
    rediscovering.
    """
    if args.tail_periods < 0.0:
        return (f"refused: --tail-periods {args.tail_periods:g} is negative; a"
                f" record cannot end before it starts")
    if args.tail_periods and args.profile != "pulse":
        return (f"refused: --tail-periods belongs to a pulse record; this arm is"
                f" {args.profile} and its drive never stops")
    if args.tail_periods and args.spectrum:
        return ("refused: --tail-periods with --spectrum: the band arm reads a"
                " cross-spectral density against a record the pulse has to stay"
                " centred in, and a tail is exactly the record that costs it -94"
                " dB. Read --ringdown for the tail, or drop --tail-periods")
    if args.ringdown and args.spectrum:
        return ("refused: --ringdown and --spectrum read one record two ways, and"
                " the tail one needs is the tail the other cannot have; run them"
                " as two records")
    if args.ringdown and args.profile != "pulse":
        return (f"refused: --ringdown reads a pulse's own decay; a {args.profile}"
                f" record is a driven steady state with nothing to decay")
    if args.ringdown and args.tail_periods < MIN_TAIL_PERIODS:
        return (f"refused: --ringdown needs a drive-free tail of at least"
                f" {MIN_TAIL_PERIODS:g} periods to fit a decay to, got"
                f" {args.tail_periods:g}")
    if args.save_tail and not args.ringdown:
        return ("refused: --save-tail writes the pulse record the tail arm fits,"
                " so on a non-ring-down (CW or band) run the flag is silently"
                " ignored and the caller believes a record was written")
    series = getattr(args, "save_series", None)
    source = getattr(args, "from_series", None)
    locator = getattr(args, "locator_tau", None)
    # the read arm first: it builds nothing, so a flag it cannot read is the
    # conflict to name, even when that flag has its own rule below
    read = getattr(args, "axial_profile", None)
    if read is not None:
        clash = [flag for flag, on in (
            ("--tiny", args.tiny), ("--estimate", args.estimate),
            ("--spectrum", args.spectrum), ("--ringdown", args.ringdown),
            ("--axis-probes", getattr(args, "axis_probes", False)),
            ("--save-tail", args.save_tail is not None),
            ("--save-series", series is not None),
            ("--from-series", source is not None),
            ("--collection-na", getattr(args, "collection_na", None) is not None),
            ("--locator-tau", locator is not None)) if on]
        if clash:
            return (f"refused: --axial-profile reads a saved record and builds no"
                    f" scene, so {', '.join(clash)} would be inert")
    if getattr(args, "axis_probes", False) and not args.estimate:
        # --estimate only prices the line (the ladder must quote the run's own
        # memory); every other arm would record it and needs somewhere to put it
        if not args.ringdown:
            return ("refused: --axis-probes stores the line's traces in the tail"
                    " record, so it belongs to --ringdown (the traces are read"
                    " over the drive-free part of the record)")
        if not args.save_tail:
            return ("refused: --axis-probes' traces travel in the --save-tail"
                    " record; without it they would be recorded and thrown away,"
                    " and the antinode map needs a second run")
    if getattr(args, "axial_frequency_nm", None) is not None and read is None:
        return ("refused: --axial-frequency-nm chooses the wavelength an axial"
                " profile is read at; give it with --axial-profile")
    if getattr(args, "axial_png", None) is not None and read is None:
        return ("refused: --axial-png draws an axial profile; give it with"
                " --axial-profile")
    if source is not None:
        clash = [flag for flag, on in (
            ("--tiny", args.tiny), ("--estimate", args.estimate),
            ("--spectrum", args.spectrum), ("--ringdown", args.ringdown),
            ("--profile pulse", args.profile != "cw"),
            ("--collection-na", getattr(args, "collection_na", None) is not None),
            ("--save-series", series is not None),
            ("--save-tail", args.save_tail is not None)) if on]
        if clash:
            return (f"refused: --from-series re-reads a saved CW record and runs"
                    f" no scene, so {', '.join(clash)} would be inert")
    for flag, value in (("--save-series", series), ("--locator-tau", locator)):
        if value is None:
            continue
        if args.profile != "cw":
            return (f"refused: {flag} belongs to the CW arm's settling curve; a"
                    f" pulse record (--spectrum, --ringdown, --profile pulse) has"
                    f" none, so the flag would be silently ignored")
        if args.estimate or (args.tiny and source is None):
            return (f"refused: {flag} with {'--estimate' if args.estimate else '--tiny'}:"
                    f" {'nothing runs' if args.estimate else 'the tier-1 controls read only the vacuum control'},"
                    f" so the flag would be inert")
    na = getattr(args, "collection_na", None)
    if na is not None:
        if not 0.0 < na <= 1.0:
            return (f"refused: --collection-na {na:g} is not a numerical aperture"
                    f" in air, which is in (0, 1]")
        if args.profile != "cw":
            return ("refused: --collection-na reads the CW arm's settled steady"
                    " state at --wavelength; a pulse record (--spectrum,"
                    " --ringdown, --profile pulse) has no single frequency to read"
                    " it at")
        if args.tiny and not args.estimate:
            return ("refused: --collection-na with --tiny: the tier-1 controls do"
                    " not read the collection arm, so the flag would be inert;"
                    " run the arm on its own (--tiny --estimate prices it)")
    pad_nm = getattr(args, "pad_nm", None)
    pml_nm = getattr(args, "pml_nm", None)
    for flag, nm in (("--pad-nm", pad_nm), ("--pml-nm", pml_nm)):
        if nm is not None and nm < args.dx:
            return (f"refused: {flag} {nm:g} nm is under one cell at dx"
                    f" {args.dx:g} nm, so this run would have no boundary there"
                    f" at all")
    if pml_nm is not None and args.pml is not None \
            and _nm_cells(pml_nm, args.dx) != int(args.pml):
        return (f"refused: --pml {args.pml} cells and --pml-nm {pml_nm:g} nm name"
                f" different absorbers at dx {args.dx:g} nm"
                f" ({_nm_cells(pml_nm, args.dx)} cells); name one")
    cells = boundary_cells(args)
    if cells["pml_cells"] >= cells["pad_cells"]:
        return (f"refused: the {cells['pml_cells']}-cell absorber is not inside"
                f" the {cells['pad_cells']}-cell pad"
                f" ({cells['pml_cells'] * args.dx:.0f} nm in"
                f" {cells['pad_cells'] * args.dx:.0f} nm), so the parts would be"
                f" in the absorber")
    return ""


def pulse_end_sigma(floor: float = DRIVE_FLOOR) -> float:
    """Where the product's pulse falls under ``floor`` of its peak, in sigma.

    ``fdtd.pulse_profile`` is a Gaussian envelope centred at 5 sigma of its own
    width, so the drive is under ``floor`` past ``5 + sqrt(-ln floor)`` sigma.
    Derived from the envelope rather than picked, so the pre-run plan here and
    the measured cut in ``drive_done`` describe the same record.
    """
    return 5.0 + math.sqrt(-math.log(floor))


def drive_done(drive, floor: float = DRIVE_FLOOR) -> int:
    """The first step at which the drive is over: under ``floor`` of its peak.

    Read off the run's own recorded drive rather than recomputed from the
    profile's constants, so a moved ``pulse_profile`` moves the window with it
    instead of leaving the fit pointed at steps the drive still occupies.
    """
    drive = np.asarray(drive, dtype=np.float64).reshape(-1)
    if drive.size < 2:
        raise ValueError(f"a drive record is at least 2 steps, got {drive.size}")
    peak = float(np.max(np.abs(drive)))
    if not peak > 0.0:
        raise ValueError("the drive record is all zeros, so there is no pulse")
    above = np.flatnonzero(np.abs(drive) > floor * peak)
    return int(above[-1]) + 1 if above.size else 0


def _decay_scan(y, dt: float, taus, lo_w: float, hi_w: float):
    """The residual of one decaying sinusoid at every (tau, omega) of a fine grid.

    At fixed tau the model is linear in its amplitudes, and the energy it takes
    out of the (mean-removed) tail is, to O(1/(omega tau)), 2 |Y(omega)|^2 /
    sum(w^2) with Y the transform of the tail weighted by w = exp(-t/tau). So one
    zero-padded FFT per tau gives the residual at every omega at once, on bins
    1/(DECAY_PAD windows) apart -- finer than the basin of any mode the record
    can show (a mode's basin is at least about 1/window wide, and wider by
    1/(pi tau) for a short one), and never coarser than a twentieth of the band.
    Returns (taus, omegas, S): S[i, j] the approximate residual at taus[i],
    omegas[j]. The approximation only chooses the cell; every number the fit
    reports is from the exact least squares.
    """
    n = y.size
    window = n * dt
    step = min(2.0 * math.pi / (DECAY_PAD * window), (hi_w - lo_w) / 20.0)
    npad = 1 << int(math.ceil(math.log2(max(n, 2.0 * math.pi / (dt * step)))))
    w = 2.0 * math.pi * np.fft.rfftfreq(npad, dt)
    inside = (w >= lo_w) & (w <= hi_w)
    omegas = w[inside]
    t = np.arange(n) * dt
    yc = y - y.mean()
    energy = float(yc @ yc)
    S = np.empty((len(taus), omegas.size))
    for i, tau in enumerate(taus):
        env = np.exp(-t / tau)
        Y = np.fft.rfft(yc * env, npad)[inside]
        S[i] = energy - 2.0 * np.abs(Y) ** 2 / float(env @ env)
    return np.asarray(taus), omegas, S


def _local_minima(S):
    """Cells of S no higher than any of their eight neighbours, lowest first."""
    padded = np.pad(S, 1, constant_values=np.inf)
    rows, cols = S.shape
    low = np.ones(S.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                low &= S <= padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]
    cells = np.argwhere(low)
    return cells[np.argsort(S[low], kind="stable")]


def _golden(f, lo: float, hi: float, iterations: int = 40) -> float:
    """The minimum of a unimodal ``f`` on ``[lo, hi]``, by golden-section search."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def fit_decay(values, dt: float, start: int, *, frequency: float,
              band=None) -> dict:
    """(f, tau, Q) from one record's tail, or the bound that says it cannot.

    The model is a single decaying sinusoid over a constant: a_dip(t) =
    c + e^{-t/tau} (a cos wt + b sin wt), on the record's own sample times from
    ``start``. Linear in (a, b, c) at fixed (tau, w) and nonlinear in those two,
    so it is a scan (``_decay_scan``: 31 taus, and at each one every omega of
    the band on bins 1/(8 windows) apart -- finer than any mode's basin, so the
    descent cannot start in the wrong one because of the grid's spacing), then
    an exact descent in the best cell's basin, then a covariance. The frequency
    reported is the descent's, not a bin's: the bins only choose the basin.

    ``ambiguous`` is None, or the reason this tau is not a measurement: a
    second minimum of the scan, separated in frequency from the fit by more
    than its basin, whose residual is under ``AMBIGUITY_RATIO`` times the
    fit's (``rival`` holds it either way), or a degenerate tau direction in the
    covariance. A second minimum that refines LOWER than the first becomes the
    fit, so the answer does not depend on which cell the scan ranked first.

    ``resolved`` is the fit's own significance test, and it is the honest part:
    tau is reported with a covariance error from the model's own residual, and a
    decay shorter than a few of those errors is not a measurement. A tau at the
    grid's top edge means the data prefer "no decay", which bounds Q from below
    rather than measuring it. The error is the model's, not the geometry's --
    the arm says so beside the number, because a tail holding two decays still
    fits one of them.

    ``band`` is the frequency window the scan may search, as (lo_hz, hi_hz) --
    the band the drive actually excites, when the caller knows it. Omitted, the
    window is the drive's own centre +-5 per cent, which is narrower than a
    pulse's own band: a 200 nm band around 1000 nm reaches -9 and +11 per cent
    in frequency, so a fit that may only look +-5 per cent cannot see a mode at
    1080 nm the same drive put energy into. ``f_edge`` is set when the fit lands
    on the window's own edge -- that edge is the window talking, and a caller
    that prints ``f`` without it is printing its own search range as a line.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if start < 0 or start + 3 > values.size:
        raise ValueError(f"a tail from step {start} of {values.size} is too short")
    y = values[start:]
    n = y.size
    t = np.arange(n) * dt
    window = n * dt
    w0 = 2.0 * math.pi * frequency
    if band is None:
        lo_w, hi_w = 0.95 * w0, 1.05 * w0
    else:
        if not band[0] < band[1]:
            raise ValueError(f"a band of {band} is not (lo_hz, hi_hz)")
        lo_w = 2.0 * math.pi * float(band[0])
        hi_w = 2.0 * math.pi * float(band[1])

    def rss_at(tau, omega):
        env = np.exp(-t / tau)
        basis = np.column_stack((env * np.cos(omega * t),
                                 env * np.sin(omega * t), np.ones(n)))
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        resid = y - basis @ coef
        return float(resid @ resid), coef

    taus = np.logspace(math.log10(window / 100.0), math.log10(window * TAU_EDGE), 31)
    edge_w = (hi_w - lo_w) / 20.0
    # The scan's omega step is set by the record, not by a fixed count across
    # the band. A 21-column grid across a +-15% band is 4.97 THz at 331 THz,
    # wider than any line longer than 64 fs, and a mode between two columns was
    # fitted in the wrong basin: change decay-fit-basin measured 33 of 60 tiny
    # synthetic tails wrong (a 1060 fs mode at 108 fs, rms 0.38 of its peak),
    # and multi-starting that descent from the best 3 or 5 cells still 33 and
    # 31. With this scan and descent, bins 1/window apart left 5 of 60 long,
    # quiet lines 13-18% short; 1/(2, 4 or 8 windows) none, at the same cost,
    # so the scan uses 8, four times finer than the coarsest that worked.
    taus, omegas, S = _decay_scan(y, dt, taus, lo_w, hi_w)
    step_w = float(omegas[1] - omegas[0]) if omegas.size > 1 else edge_w

    def refine(i, j, iterations=40):
        """The exact minimum in cell (i, j)'s basin: tau within two grid cells
        of the cell's (a profile over log tau), and omega at each tau within
        the larger of two bins and the cell's own line, 1/tau rad/s either side.

        The line matters for a short tau. The scan's residual is good to
        O(1/(omega tau)), which is 25% for the --tiny vacuum tail's 1.9 fs, and
        there it chose a cell 22 THz off the exact minimum; two bins could not
        reach it, and the fit's rms came out 13 x the old search's (measured).
        A line that wide is one basin, so its whole width is safe to search."""
        w0 = float(omegas[j])
        half = max(2.0 * step_w, 1.0 / float(taus[i]))
        wlo, whi = max(lo_w, w0 - half), min(hi_w, w0 + half)

        def profile(log_tau):
            tau = math.exp(log_tau)
            om = _golden(lambda v: rss_at(tau, v)[0], wlo, whi, iterations)
            return rss_at(tau, om)[0], om

        lo_t = math.log(float(taus[max(i - 2, 0)]))
        hi_t = math.log(float(taus[min(i + 2, taus.size - 1)]))
        log_tau = _golden(lambda v: profile(v)[0], lo_t, hi_t, iterations)
        rss, om = profile(log_tau)
        return rss, math.exp(log_tau), om

    minima = _local_minima(S)
    i, j = (int(k) for k in minima[0])
    at_edge = i == taus.size - 1
    rss, tau, omega = refine(i, j)
    # The other basins: the lowest few local minima of the scan whose frequency
    # is separated from this fit's by more than its own basin (twice the larger
    # of the line's half width 1/(pi tau) and the record's 1/window). Each is
    # refined exactly; one that refines lower than the fit becomes the fit.
    rival = None
    sep = 2.0 * 2.0 * math.pi * max(1.0 / (math.pi * tau), 1.0 / window)
    others = [(int(ci), int(cj)) for ci, cj in minima[1:]
              if abs(float(omegas[cj]) - omega) > sep][:RIVALS]
    for ci, cj in others:
        other = refine(ci, cj, RIVAL_ITERATIONS)
        if other[0] < rss:
            other, (rss, tau, omega) = (rss, tau, omega), other
            at_edge = ci == taus.size - 1
        if rival is None or other[0] < rival[0]:
            rival = other
    rss, coef = rss_at(tau, omega)
    f_edge = bool(omega <= lo_w + edge_w or omega >= hi_w - edge_w)

    def model(p):
        a, b, c, tau, om = p
        return c + np.exp(-t / tau) * (a * np.cos(om * t) + b * np.sin(om * t))

    p = np.array([coef[0], coef[1], coef[2], tau, omega])
    jac = np.empty((n, 5))
    steps = np.array([abs(p[0]), abs(p[1]), abs(p[2]), tau, omega]) * 1e-5
    steps[3] = tau * 1e-5
    steps[4] = omega * 1e-7
    for k in range(5):
        if steps[k] <= 0.0:
            steps[k] = 1e-12 * (1.0 + abs(p[k]))
        hi, lo = p.copy(), p.copy()
        hi[k] += steps[k]
        lo[k] -= steps[k]
        jac[:, k] = (model(hi) - model(lo)) / (2.0 * steps[k])
    resid = y - model(p)
    rss = float(resid @ resid)
    sigma2 = rss / max(n - 5, 1)
    cov = sigma2 * np.linalg.pinv(jac.T @ jac)
    f = omega / (2.0 * math.pi)

    def _sigma(k, value):
        """Parameter ``k``'s own error, or None where the record does not fix it.

        A direction the record does not determine shows up in the covariance as
        an error that is not one, in either of two shapes. A non-positive
        diagonal is the pinv having dropped the direction, and the clamp this
        replaced called it zero: a flat tail -- the cavity never rose -- then
        reported f +- 0 and a resolved Q, because zero passes every "more than
        three of its own errors" test there is. A *positive* diagonal can be the
        pinv's just as surely -- Run A (2026-10-16, 898.3 nm, tau at the fit
        grid's top) printed 4.1e-65 THz against a 333.7 THz centre and 6.3e-14
        fs against a 2033 fs decay, both positive and both tens of orders below
        their own value (67 for the frequency, 16.5 for the decay), which is not a
        number any record of finite samples fixes.

        The test is the error beside its own parameter, not the rank of
        ``jac.T @ jac``: measured (2026-10-16) ``matrix_rank`` calls that matrix
        rank 1 at its default tolerance for *every* fixture tried, the flat tail
        whose other four singular values are exactly 0 included, and the noisy
        tail whose tau error is a real 6.5e-6 of tau. A rank verdict there
        would throw away a measurement the record does make; the size of the
        error separates the two, 6.5e-6 kept against ~1e-65 dropped, with this
        floor three decades below the kept error and fifty-five above the
        dropped one. That last digit is pinv noise -- 4.6237e-65 on the
        reviewer's VM, 7.4e-66 on an earlier run here -- so it is quoted as an
        order of magnitude, which is all the record fixes.
        """
        diag = float(cov[k, k])
        if diag <= 0.0:
            return None
        err = math.sqrt(diag)
        # An exactly-zero parameter keeps whatever error it has: the comparison
        # is against its own size, which is not a scale to divide by.
        return None if err < abs(value) * ERR_FLOOR else err

    tau_err = _sigma(3, tau)
    f_err = _sigma(4, f)
    q = math.pi * f * tau
    q_err = (math.pi * math.hypot(tau * f_err, f * tau_err)
             if f_err is not None and tau_err is not None else None)
    # Whether this tau is a measurement at all. Two separated minima whose
    # residuals are within AMBIGUITY_RATIO of each other are two answers the
    # record does not choose between (two modes, or a mode under the noise), and
    # a covariance whose tau direction is degenerate has no error to put beside
    # tau. Either way the arm prints the reason, not a tau.
    #
    # f_err is NOT read here. It is None on well-fitted tails too: the pinv
    # below sees Jacobian columns some 1e24 apart in scale (tau's ~1e13, omega's
    # ~1e-11) and drops omega's direction (measured, change decay-fit-basin, on
    # a 1060 fs tail it recovers to 0.03%). Flagging on it would flag every fit.
    ratio = (rival[0] / rss if rss > 0.0 else math.inf) if rival else None
    reasons = []
    if ratio is not None and ratio < AMBIGUITY_RATIO:
        reasons.append(f"a second minimum at {rival[2] / (2.0 * math.pi) / 1e12:.4f} THz,"
                       f" tau {rival[1] / 1e-15:.4g} fs, leaves {ratio:.3g} x this"
                       f" fit's residual (under {AMBIGUITY_RATIO:g} is a tie)")
    if tau_err is None:
        reasons.append("tau's own covariance direction is degenerate")
    return {"f": f, "f_err": f_err, "tau": tau, "tau_err": tau_err,
            "q": q, "q_err": q_err, "samples": n, "window": window,
            "rms": math.sqrt(rss / n), "peak": float(np.max(np.abs(y))),
            "at_edge": at_edge, "f_edge": f_edge,
            "rival": (None if rival is None else
                      {"f": rival[2] / (2.0 * math.pi), "tau": rival[1],
                       "rss_ratio": ratio}),
            "ambiguous": "; ".join(reasons) or None,
            "resolved": bool(tau_err is not None and tau > 3.0 * tau_err and not at_edge)}


UNDETERMINED = "undetermined (degenerate covariance)"


def _err(value, scale: float) -> str:
    """The ``+-`` half of a fitted error: a number, or why there is not one.

    ``fit_decay`` returns None for an error whose covariance diagonal is not
    positive, or which is orders of magnitude below its own parameter -- both
    are a direction the record does not determine. Formatting
    that as ``0`` is what this arm used to print, and it is the wrong number to
    print: a flat tail -- the cavity never rose -- passed the three-sigma test
    on a zero error and reported a centre and a Q with no error bars at all.
    """
    return UNDETERMINED if value is None else f"{value / scale:.2g}"


def _npz_path(path) -> str:
    """What np.savez will actually write -- it appends .npz, so we must too.

    Printing the path as given would name a file that is not there.
    """
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"


def save_record(path, values, dt: float, start: int, request, **extra) -> str:
    """Write one record so its tail can be re-fitted without another Orin run.

    One decaying sinusoid is the model ``fit_decay`` fits. A tail holding two of
    them needs a second model, and without the samples on disk that model costs
    the run again. So the dt and the drive travel with the values:
    ``load_record`` returns exactly what the arm fitted, which is the whole
    difference between a re-analysis and a repeat. Returns the path written.

    ``extra`` is written beside them unchanged -- the axial probe arm passes
    its traces, their node z, the wire's own span and the emitter's z, so the
    profile is read from this record rather than from a second run.
    """
    np.savez(path, values=np.asarray(values), dt=float(dt), start=int(start),
             frequency_hz=float(request["frequency_hz"]),
             bandwidth=float(request["bandwidth"]), **extra)
    return _npz_path(path)


def load_record(path) -> dict:
    """The other half of ``save_record``: values, dt, start, frequency, band.

    Reads the argument the writer was given, not the file it wrote.
    """
    z = np.load(_npz_path(path))
    f = float(z["frequency_hz"])
    half = float(z["bandwidth"])
    return {"values": z["values"], "dt": float(z["dt"]), "start": int(z["start"]),
            "frequency": f, "band": (f - half, f + half)}


def axial_profile(record, frequency=None, *, start=None, window="hann") -> dict:
    """|E(z)|^2 along the axial line, read from a saved record.

    ``record`` is the path of a ``--ringdown --save-tail --axis-probes`` npz.
    The profile at ``frequency`` is each probe's amplitude there -- the
    windowed DFT of its trace over the part of the record AFTER the source has
    decayed (``start``, the step ``save_record`` stored) -- squared and split by
    component; ``frequency`` defaults to the record's own drive.

    Returns the zs (metres), the per-component powers at each z, their sum, the
    wire's span and the emitter's z (both stored by the run), and the z of
    every interior local maximum of the summed power inside the wire -- the
    antinodes the emitter is being placed on. A maximum in the last or first
    sample of the range is not a local maximum of the profile but its edge;
    those are not listed.
    """
    with np.load(_npz_path(record)) as z:
        if "probe_traces" not in z.files:
            raise ValueError(
                f"{_npz_path(record)} has no probe traces: it was not written by"
                f" a --ringdown --save-tail run with --axis-probes")
        # float32 on disk, and read as it was written: a 441-probe line of
        # 121,121 steps is 213.7 MB, and a float64 copy of it is another
        # 213.7 MB the bench does not have to spend (each row is cast as it is
        # read)
        traces = np.asarray(z["probe_traces"])
        probe_z = np.asarray(z["probe_z_m"], dtype=float)
        comps = np.asarray(z["probe_components"]).astype(str)
        dt = float(z["dt"])
        begin = int(z["start"]) if start is None else int(start)
        f = float(z["frequency_hz"]) if frequency is None else float(frequency)
        wire = np.asarray(z["wire_z_m"], dtype=float)
        emitter = float(z["emitter_z_m"])
    if begin >= traces.shape[1]:
        raise ValueError(
            f"the tail starts at step {begin:,} of {traces.shape[1]:,}: there is"
            f" no decayed part of this record to read")
    hann = fdtd._window(traces.shape[1] - begin, window)
    powers, zs = {}, None
    for component in AXIS_COMPONENTS:
        rows = np.flatnonzero(comps == component)
        if rows.size == 0:
            continue
        # every component's probes were placed on the same z nodes, in line
        # order, so their own z column IS the abscissa -- exact equality, not
        # a mean or a rounding
        this_z = probe_z[rows]
        if zs is None:
            zs = this_z
        elif not np.array_equal(this_z, zs):
            raise ValueError(
                f"{_npz_path(record)} has no single abscissa: its {component}"
                f" probes sit on {rows.size} z nodes and the"
                f" {next(iter(powers))} probes on {zs.size}")
        amp = np.array([fdtd._windowed_at(traces[row, begin:], dt, f, hann)
                        for row in rows])
        powers[component] = np.abs(amp) ** 2
    if zs is None:
        raise ValueError(f"{_npz_path(record)} carries no E component among "
                         f"its probes, got {sorted(set(comps))}")
    total = sum(powers.values())
    inside = np.flatnonzero((zs >= wire[0] - 1e-12) & (zs <= wire[1] + 1e-12))
    maxima = [zs[i] for i in inside
              if 0 < i < total.size - 1
              and total[i] > total[i - 1] and total[i] > total[i + 1]]
    return {"z_m": zs, "power": powers, "total": total, "maxima_z_m": maxima,
            "wire_z_m": wire, "emitter_z_m": emitter, "frequency_hz": f,
            "tail_start": begin, "tail_steps": traces.shape[1] - begin,
            "dt": dt, "size": zs.size}


def axial_profile_command(args) -> int:
    """``--axial-profile``: print the saved record's profile, and its picture.

    The table is the measurement; the PNG is a convenience for the owner's
    eyes, so a machine without matplotlib loses the picture and not the
    numbers. Both markers the picture draws -- the wire's ends and the
    emitter's z -- come from the record itself.
    """
    profile = axial_profile(args.axial_profile,
                            None if args.axial_frequency_nm is None
                            else C / (args.axial_frequency_nm * NM))
    zs, total = profile["z_m"], profile["total"]
    peak = float(np.max(total)) if total.size else 0.0
    print(f"record    {_npz_path(args.axial_profile)}:"
          f" {profile['size']:,} z layers, {len(profile['power'])} components,"
          f" tail from step {profile['tail_start']:,}"
          f" ({profile['tail_steps']:,} steps)")
    print(f"frequency {profile['frequency_hz'] / 1e12:.4f} THz"
          f" ({C / profile['frequency_hz'] / NM:.1f} nm)")
    print(f"wire      {profile['wire_z_m'][0] / NM:.3f} to"
          f" {profile['wire_z_m'][1] / NM:.3f} nm above the frame origin;"
          f" the emitter is at {profile['emitter_z_m'] / NM:.3f} nm")
    print(f"{'z nm':>9} {'|Ex|^2':>10} {'|Ey|^2':>10} {'|Ez|^2':>10}"
          f" {'total':>10}  (each / the line's peak)")
    for i in range(zs.size):
        row = " ".join(f"{profile['power'][c][i] / peak:10.4f}"
                       if c in profile["power"] else f"{'-':>10}"
                       for c in AXIS_COMPONENTS)
        print(f"{zs[i] / NM:9.3f} {row} {total[i] / peak:10.4f}")
    if profile["maxima_z_m"]:
        spots = ", ".join(f"{m / NM:.2f}" for m in profile["maxima_z_m"])
        print(f"maxima    the summed profile peaks inside the wire at z ="
              f" {spots} nm")
    else:
        print("maxima    no interior local maximum inside the wire: the profile"
              " rises or falls across the whole of it")
    if args.axial_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:      # the picture, not the measurement
            print(f"warning   no PNG ({type(exc).__name__}: {exc}); the table"
                  f" above is the record's own")
            return 0
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for component in AXIS_COMPONENTS:
            if component in profile["power"]:
                ax.plot(zs / NM, profile["power"][component] / peak,
                        label=f"|E{component}|^2")
        ax.plot(zs / NM, total / peak, color="black", label="sum")
        for z, style in ((profile["wire_z_m"][0], ":"), (profile["wire_z_m"][1], ":"),
                         (profile["emitter_z_m"], "--")):
            ax.axvline(z / NM, color="grey", linestyle=style, linewidth=0.8)
        ax.set_xlabel("z above the frame origin (nm)")
        ax.set_ylabel("|E|^2 over the line's peak")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"axial mode profile at {C / profile['frequency_hz'] / NM:.1f} nm")
        fig.tight_layout()
        fig.savefig(args.axial_png, dpi=140)
        print(f"png       {args.axial_png}: the wire's ends dotted, the emitter"
              f" dashed")
    return 0


def axis_probe_record(sim, lay) -> dict:
    """The axial arm's half of a saved record: traces, their z, the spans.

    ``lay`` is the layout the solver ran on (``fdtd_layout.layout``), and its
    ``probes`` are four-tuples ``(i, j, k, component)`` -- the cell's node
    indices and the component -- so each probe's z is read off the layout's own
    node list at ``k``, and the profile's abscissa is the grid's, not a
    re-typed copy of the request's positions. The wire's span comes from the
    layout's own ``part_bounds`` (the last part by the scene's names order) and
    the emitter's z from the node its source cell sits on; the profile marker
    draws on them, and a reader must not have to reconstruct the scene to say
    where the wire begins.
    """
    nodes, probes = lay["nodes"], lay["probes"]
    z = np.array([float(nodes[2][probe[2]]) for probe in probes])
    comps = np.array([probe[3] for probe in probes])
    wire = lay["part_bounds"][-1]                  # names order: mirror, oxide, wire
    return {"probe_traces": np.asarray(sim["traces"]),
            "probe_z_m": z,
            "probe_components": comps,
            "wire_z_m": np.array([float(wire[0][2]), float(wire[1][2])]),
            "emitter_z_m": float(nodes[2][int(lay["cell"][2])])}


def ringdown(args, lay, request, measured):
    """The tail arm: one pulse record, its decay, and the Q it does or does not pin.

    Returns (fit, code). The code is 0 when the fit is significant, 3 when the
    record's own tail leaves Q unpinned -- not a failure to hide, and not a
    measurement to invent: the fit's own error is printed either way.
    """
    sim, volume, index, eps_applied = _run(args, lay, request, measured)
    dt = measured["est"]["dt"]
    f0 = float(request["frequency_hz"])
    start = drive_done(sim["drive"])
    tail_steps = int(measured["steps"]) - start
    window = tail_steps * dt
    periods = window * f0
    print(f"tail      the drive is under {DRIVE_FLOOR:g} of its own peak from step"
          f" {start:,} ({start * dt / 1e-15:.1f} fs); the tail is {tail_steps:,} steps"
          f" = {window / 1e-15:.1f} fs = {periods:.1f} periods")
    q_paper, tau_paper = paper_clock(args, request)
    print(f"paper     the paper's own pair -- {args.wavelength:.0f} nm at a"
          f" {PAPER_FWHM_NM:.0f} nm line -- is Q = {q_paper:.0f}, an amplitude"
          f" decay of {tau_paper / 1e-15:.0f} fs; it would hold"
          f" {window / tau_paper:.2f} of this tail")
    if periods < MIN_TAIL_PERIODS:
        print(f"refused: the drive is over only {periods:.1f} periods before the"
              f" record ends, short of the {MIN_TAIL_PERIODS:g} a decay needs."
              f" --pulse-span must clear the drive's own end: the pulse is centred"
              f" at 5 sigma and under {DRIVE_FLOOR:g} of its peak past"
              f" {pulse_end_sigma():.1f}")
        return None, 2
    half = float(request["bandwidth"])
    fit = fit_decay(sim["source_field"], dt, start, frequency=f0,
                    band=(f0 - half, f0 + half))
    mark = (f"; AT THE EDGE of that band, so this f is the search's own edge and"
            f" the mode is outside it") if fit["f_edge"] else ""
    ambiguous = fit.get("ambiguous")
    tau_text = (f"tau AMBIGUOUS ({fit['tau'] / 1e-15:.3f} fs is not a measurement:"
                f" {ambiguous})" if ambiguous else
                f"tau = {fit['tau'] / 1e-15:.3f} fs +- {_err(fit['tau_err'], 1e-15)}")
    print(f"fit       f = {fit['f'] / 1e12:.4f} THz +- {_err(fit['f_err'], 1e12)}"
          f" (the drive is {f0 / 1e12:.2f}, searched over"
          f" {C / (f0 + half) / NM:.0f}-{C / (f0 - half) / NM:.0f} nm){mark},"
          f" {tau_text},"
          f" rms {fit['rms']:.3e} against a {fit['peak']:.3e} peak")
    print(f"caveat    tau's error is the fit's own, not the geometry's: a tail"
          f" holding two decays still fits one of them. --vacuum is the same arm"
          f" on the transient this scene has with no cavity at all")
    if args.save_tail:
        # After the fit is printed, not before: the write is a convenience and the
        # fit is the measurement, so a write that fails must not be able to take the
        # number down with it -- and the only exception a path can raise here is an
        # OSError, which is caught rather than left to end the run.
        try:
            extra = (axis_probe_record(sim, lay)
                     if getattr(args, "axis_probes", False) else {})
            wrote = save_record(args.save_tail, sim["source_field"], dt, start,
                                request, **extra)
        except OSError as exc:
            print(f"warning   the tail was not written ({type(exc).__name__}:"
                  f" {exc}): the fit above is this run's own, and a second model on"
                  f" this record now needs another run")
        else:
            print(f"tail      {np.size(sim['source_field']):,} samples and the dt they"
                  f" were taken with in {wrote}: load_record plus fit_decay re-fits"
                  f" this tail without another run")
    if ambiguous:
        # Before the Q line: a tau the record does not choose is not one to
        # read a Q from, nor to bound one with.
        print(f"unresolved: the fit is ambiguous -- {ambiguous}. No Q is read"
              f" from this tail; a two-mode model on the saved record"
              f" (--save-tail) is the reading that can choose")
        return fit, UNRESOLVED
    if fit["resolved"]:
        print(f"Q         {fit['q']:.4g} +- {_err(fit['q_err'], 1.0)}, a line of"
              f" {args.wavelength / fit['q']:.2f} nm against the paper's"
              f" {PAPER_FWHM_NM:.0f} nm -- read from the decay, not from this"
              f" record's {measured['dlambda_nm']:.2f} nm bins")
        return fit, 0
    if fit["tau_err"] is None:
        print(f"unresolved: tau's own error is {UNDETERMINED}, so there is no"
              f" three-sigma floor to bound Q with: the covariance's tau direction"
              f" is one this record does not determine, not one it determines"
              f" badly")
        return fit, UNRESOLVED
    floor = max(fit["tau"] - 3.0 * fit["tau_err"], 0.0)
    tail = ("tau ran to the top of the fit's own grid"
            if fit["at_edge"] else "tau is not three of its own errors")
    print(f"unresolved: {tail}, so this tail does not pin Q. What it bounds is"
          f" Q > {math.pi * fit['f'] * floor:.1f} (the 3-sigma floor of the fit);"
          f" a decay slower than the {window / 1e-15:.0f} fs window stays inside the"
          f" bounds the two other arms already cover")
    return fit, UNRESOLVED


def _oxide_slice(z, geom):
    """The z spacings of the cells that lie inside the buffer.

    ``lay["nodes"]`` is in the parts' own frame: the parts keep the coordinates
    they were built with and the layout's pad is a negative-z overhang, so the
    gold's voxel edges sit at 0 and 40 nm for a 40 nm mirror while the grid
    starts at -pad (a probe read z[0] = -320 nm with the mirror at 0..40 nm).
    The buffer therefore sits at ``z_oxide`` here; adding the pad looks 320 nm
    further up and finds the wire's coarse cells instead, which is what this
    helper did until the probe was read. ``tiny`` checks the frame itself
    against the gold part's own bounds, so a shift cannot pass unnoticed.
    """
    dz = np.diff(z)
    mid = 0.5 * (z[:-1] + z[1:])
    inside = (mid > geom["z_oxide"][0]) & (mid < geom["z_oxide"][1])
    return dz[inside]


def _oxide_cell_nm(z, geom) -> float:
    cells = _oxide_slice(z, geom)
    return float(cells.mean()) / NM if cells.size else float("nan")


def report(args, lay, request, geom, measured):
    est = measured["est"]
    z = np.asarray(lay["nodes"][2])
    problems = fdtd_grid.check_axis(z, args.dx * NM)
    print(f"grid      {est['shape'][0]} x {est['shape'][1]} x {est['shape'][2]}"
          f"  = {est['cells']:,} cells")
    # The open boundary is the run's own two settings, in the cells the grid was
    # built with AND the distance those cells are. A cell count is not a
    # distance: the product's own pad is 160 nm at dx 10 against r20's 320, so a
    # dx-10 record's boundary has to be readable off its own log.
    asked_nm = " ".join(
        f"--{name} {getattr(args, name.replace('-', '_'), None):.0f}"
        for name in ("pad-nm", "pml-nm")
        if getattr(args, name.replace("-", "_"), None) is not None)
    print(f"boundary  open: pad {geom['pad_cells']} cells = {geom['pad_nm']:.0f} nm,"
          f" absorber {geom['pml_cells']} cells = {geom['pml_nm']:.0f} nm,"
          f" {geom['pad_nm'] - geom['pml_nm']:.0f} nm of open space"
          f" ({asked_nm or 'both at the product defaults'})")
    print(f"spacings  x, y {args.dx:.3f} nm uniform;  z {est['finest'] / NM:.3f} nm"
          f" finest, {np.diff(z).max() / NM:.3f} nm coarsest"
          f"  (the spec's neighbour ratio is <= {fdtd_grid.MAX_RATIO})")
    print(f"z-axis    the product's own check_axis says: "
          f"{problems if problems else 'no problems'}")
    thin = np.diff(z)[np.diff(z) < args.dx * NM * 0.999]
    print(f"thin      {thin.size} z cells below the coarse cell, finest"
          f" {thin.min() / NM if thin.size else float('nan'):.3f} nm; the SiO2 is"
          f" {(geom['z_oxide'][1] - geom['z_oxide'][0]) / NM:.0f} nm over"
          f" {_oxide_slice(z, geom).size} cells of"
          f" {_oxide_cell_nm(z, geom):.2f} nm")
    f0 = request["frequency_hz"]
    metal = gold_metal(args)
    depth = fdtd_grid.skin_depth(metal, f0) / NM
    print(f"gold      {metal} (--gold-data {args.gold_data})")
    print(f"stack     gold {args.gold:.0f} nm = {args.gold / depth:.1f} skin depths"
          f" (skin depth {depth:.1f} nm at {f0 / 1e12:.1f} THz, {metal}), SiO2"
          f" {args.oxide:.0f} nm, InP {args.height:.0f} nm on a"
          f" {2 * geom['half'] / NM:.0f} nm mirror; the wire is"
          f" {2 * geom['radius'] / NM:.1f} nm across corners,"
          f" {2 * geom['radius'] * math.cos(math.pi / 6.0) / NM:.1f} nm across flats"
          f" (--diameter {args.diameter:.0f} nm, --hex {args.hex})")
    cell = lay["cell"]
    x, y = np.asarray(lay["nodes"][0]), np.asarray(lay["nodes"][1])
    print(f"dipole    node ({cell[0]}, {cell[1]}, {cell[2]}): z"
          f" {z[cell[2]] / NM:.2f} nm, {emitter_depth_nm(lay, geom):.2f} nm"
          f" below the top facet at {geom['z_top'] / NM:.2f} nm"
          f" (--qd-below {args.qd_below:g} asked); in plane"
          f" {(x[cell[0]] - geom['half']) / NM:+.2f},"
          f" {(y[cell[1]] - geom['half']) / NM:+.2f} nm off the wire's axis,"
          f" {args.axis}-polarised")
    print(f"timestep  {est['dt']:.6e} s  ({1.0 / (f0 * est['dt']):.1f} steps per"
          f" optical period)")
    print(f"record    {est['steps']:,} steps = {est['steps'] * est['dt'] * f0:.1f}"
          f" periods, {est['seconds']:.1f} s estimated on the Jetson"
          f" ({est['setup']} rates)")
    if args.profile == "pulse" and args.tail_periods:
        sigma = 1.0 / (math.pi * request["bandwidth"])
        planned = int(math.ceil(pulse_end_sigma() * sigma / est["dt"]))
        print(f"tail      the drive is under {DRIVE_FLOOR:g} of its peak past"
              f" {pulse_end_sigma():.2f} sigma (the pulse is centred at 5), i.e."
              f" step {planned:,}, so the tail is {est['steps'] - planned:,} steps"
              f" = {(est['steps'] - planned) * est['dt'] * f0:.1f} periods"
              f" (--tail-periods {args.tail_periods:g} asked for"
              f" {args.tail_periods:g}); the mode is read from this decay, not from"
              f" the {measured['dlambda_nm']:.2f} nm bins")
    print(f"memory    {est['bytes'] / 2 ** 30:.2f} GiB estimated"
          f"  (metals {sorted(lay['metals']) or 'none'},"
          f" {'pec' if lay['pec'] is not None else 'no pec'},"
          f" {request['boundary']} boundary)")
    lim = run_limits(args)
    raised = bench_ceilings(lim)
    print(f"resolves  {measured['dlambda_nm']:.2f} nm per bin at"
          f" {args.wavelength:.0f} nm; the paper's mode is {PAPER_FWHM_NM:.0f} nm"
          f" and would need {steps_for_4nm(est['dt'], args.wavelength):,} steps"
          f" (the cap is {lim['max_steps']:,})")
    print(f"caps      {measured['exceeded'] or 'OK'}"
          + (f" (bench ceiling: {', '.join(raised)})" if raised else ""))
    if (args.profile == "cw"
            and est["steps"] * est["dt"] * f0 < args.window_periods):
        print(f"refused: the record is"
              f" {est['steps'] * est['dt'] * f0:.1f} periods, shorter than the"
              f" {args.window_periods:.1f}-period tail the reading averages")
        return 2
    if (args.profile == "cw"
            and est["steps"] * est["dt"] * f0 < SETTLED_PERIODS):
        print(f"warning: the record is {est['steps'] * est['dt'] * f0:.1f} periods."
              f" A hard-started CW drive is not settled that early, and what it"
              f" reads is not the steady state: measured on the tiny scene in"
              f" vacuum, 8 periods read F = 4.6076 with a 4-period tail and"
              f" 170.0721 with an 8-period tail, while 16 read 1.0010 and 32 read"
              f" 1.0019. The settling is in PERIODS, not in the tail's share of"
              f" the record -- 8 over 4 and 16 over 8 are the same ratio and only"
              f" the second is settled. Raise --periods.")
    return 0


def source_and_terms(args, lay, request):
    cell = lay["cell"]
    eps_cell = float(lay["eps"][cell])
    source = fdtd.Source(cell, frequency=float(request["frequency_hz"]),
                         profile=args.profile,
                         bandwidth=(request["bandwidth"]
                                    if args.profile == "pulse" else None),
                         component=args.axis)
    terms = [fdtd.LorentzDrude(name, region)
             for name, region in dict(lay["metals"]).items()]
    volume = fdtd.source_cell_volume(lay["spacings"], cell, args.axis)
    # The host index is the material's (the node value); the readings take the
    # permittivity the update applies at the source's own E sample, which a
    # metal node beside it along its axis sets to eps_inf = 1.
    eps_applied = fdtd.source_permittivity(lay["eps"], cell, args.axis,
                                           ade_terms=terms,
                                           boundary=request["boundary"])
    index = math.sqrt(eps_cell)
    print(f"source    node {tuple(int(c) for c in cell)}, eps_r {eps_cell:.4f}"
          f" (cell), {eps_applied:.4f} (applied at E{args.axis})"
          f" -> host index {index:.4f}, cell volume {volume:.6e} m^3")
    return source, terms, volume, index, eps_applied


def _run(args, lay, request, measured):
    source, terms, volume, index, eps_applied = source_and_terms(args, lay,
                                                                 request)
    # the collection arm's box; without --collection-na the call is unchanged
    phasor = measured.get("phasor")
    extra = ({} if phasor is None else
             {"phasor_box": phasor["box"], "phasor_frequencies": [phasor["frequency"]],
              "phasor_start": phasor["start"], "phasor_window": "hann"})
    started = time.monotonic()
    sim = fdtd.simulate(lay["eps"], source, measured["est"]["steps"],
                        lay["probes"], spacings=lay["spacings"],
                        boundary=request["boundary"], pec=lay["pec"],
                        pml_cells=lay["pml_cells"], ade_terms=terms, xp=args.xp,
                        record_drive=True, record_source_field=True, **extra)
    wall = time.monotonic() - started
    estimate = measured["est"]["seconds"]
    # The run reports its own prediction ratio, so a logged run carries the
    # evidence that the table's rate does or does not hold on this machine. It
    # is printed, not folded into a new constant: one contended run cannot set
    # a rate that the next machine has to live with.
    print(f"run       {wall:.1f} s wall  ({wall / estimate:.2f} x the"
          f" {estimate:.1f} s estimate)")
    if sim["source_eps"] != eps_applied:
        raise SystemExit(
            f"internal: the run applied eps_r {sim['source_eps']!r} at the"
            f" source, the printed line {eps_applied!r}")
    return sim, volume, index, eps_applied


def cw_powers(drive, field, dt: float, *, volume: float, eps_r: float,
              index: float, frequency: float, periods):
    """``(P, P0)``: the CW arm's two powers over the LAST ``periods`` of the series.

    ``fdtd.delivered_power`` and ``fdtd.dipole_moment`` both read the last
    ``periods`` drive periods of whatever series they are handed, so a reading
    at an earlier point of one record is this same call on the record cut
    there. One function, so the end reading, the two-thirds reading, the
    settling curve and a re-read of a saved series cannot drift apart.
    """
    power = fdtd.delivered_power(drive, field, dt, volume=volume, eps_r=eps_r,
                                 periods=periods, frequency=frequency)
    moment = fdtd.dipole_moment(drive, dt, volume=volume, eps_r=eps_r,
                                periods=periods, frequency=frequency)
    return power, fdtd.larmor_power(moment, frequency, index=index)


def cw_reading(drive, field, dt: float, **reading) -> float:
    """F = P/P0 from :func:`cw_powers` (``fdtd.purcell_factor``: P must be > 0)."""
    return fdtd.purcell_factor(*cw_powers(drive, field, dt, **reading))


def _period_steps(dt: float, frequency: float) -> float:
    return 1.0 / (float(frequency) * float(dt))


def window_steps(dt: float, frequency: float, periods) -> int:
    """The reading's window in steps: ``fdtd._steady_tail``'s own rounding."""
    return int(round(int(periods) * _period_steps(dt, frequency)))


def window_phase(drive, field, dt: float, **reading) -> dict | None:
    """How the reading moves with WHERE its window ends, measured on this record.

    The CW reading is ``Re[E~ S~*]`` over a Hann window, and the field at a
    point source's own cell is almost all reactive: its in-phase (power
    carrying) part is a small residue of the half-step average the delivered
    power takes (tiny vacuum scene: 1/4150 of the field). The Hann passes a
    little of each tone's -f image (3.65e-6 at 8 periods), which rotates at 2f
    against the drive, and the quadrature multiplies it into the reading. So
    the reading swings at TWICE the drive frequency as its window's end moves
    through one period -- measured on the tiny vacuum scene at +-3.03% of F
    with an 8-period window, where twice the leakage times the quadrature
    ratio predicts +-3.03%, and at +-25%/+-0.36% with 4/16 periods.

    This scans ``PHASE_CUTS + 1`` cuts across the record's last drive period
    and fits ``a + b x + c cos(4 pi x) + d sin(4 pi x)`` (x in periods from
    the end), so a reading that is still moving does not pass its trend off
    as phase. Returns ``mean`` (a: the end with the 2f term removed),
    ``swing`` (|c, d|), ``end_term`` (c: the 2f term's value at the end),
    ``slope`` (b, per period), the measured ``samples``, and the prediction
    ``predicted`` = 2 x ``leak`` x ``quadrature``. None when the record is
    too short to cut a window a period before its end.
    """
    drive = np.asarray(drive)
    field = np.asarray(field)
    n = int(np.size(field))
    dt = float(dt)
    frequency = float(reading["frequency"])
    per = _period_steps(dt, frequency)
    window = window_steps(dt, frequency, reading["periods"])
    cuts = sorted({n - int(round(i * per / PHASE_CUTS))
                   for i in range(PHASE_CUTS + 1)})
    if cuts[0] < window or window < 2:
        return None
    ratios = []
    for cut in cuts:
        power, reference = cw_powers(drive[:cut], field[:cut], dt, **reading)
        ratios.append(power / reference)       # P <= 0 kept: it is the swing
    ratios = np.asarray(ratios)
    x = (np.asarray(cuts, dtype=np.float64) - n) / per
    basis = np.column_stack((np.ones(x.size), x, np.cos(4.0 * math.pi * x),
                             np.sin(4.0 * math.pi * x)))
    coef, *_ = np.linalg.lstsq(basis, ratios, rcond=None)
    # the prediction, from the last window: the estimator's own half-step field
    trace = field.astype(np.float64)
    mid = 0.5 * (trace + np.concatenate((trace[:1], trace[:-1])))
    hann = fdtd._window(window, "hann")
    s = fdtd._windowed_at(drive[n - window:], dt, frequency, hann)
    e = fdtd._windowed_at(mid[n - window:], dt, frequency, hann)
    # the end window's in-phase part carries the end's own share of the swing, so
    # it is scaled to the phase mean (F is proportional to it) before it is used
    inphase = (abs(float(np.real(e * np.conj(s)))) / abs(s) if abs(s) > 0 else 0.0)
    inphase *= abs(float(coef[0]) / ratios[-1]) if ratios[-1] else 0.0
    quadrature = abs(e) / inphase if inphase > 0 else math.inf
    leak = float(abs(np.sum(hann * np.exp(-4j * math.pi * frequency * dt
                                          * np.arange(window))))
                 / np.sum(hann))
    return {"mean": float(coef[0]), "slope": float(coef[1]),
            "swing": float(math.hypot(coef[2], coef[3])), "end_term": float(coef[2]),
            "samples": ratios, "cuts": cuts, "leak": leak,
            "quadrature": quadrature, "predicted": 2.0 * leak * quadrature
            * abs(float(coef[0]))}


def settling_curve(drive, field, dt: float, *, step: float = SETTLING_STEP,
                   **reading) -> dict:
    """F through the record: the CW reading at cuts every ``step`` of it.

    Each cut is the same reading (``cw_reading``, the last ``periods`` of the
    record cut there). Every cut sits a WHOLE number of drive periods before
    the record's end -- the one nearest each 5% mark whose fractional period
    is smallest, searched over a third of the spacing -- so every window
    ends at the end's own drive phase and the 2f swing ``window_phase``
    measures is one constant offset along the curve, not cut-to-cut noise
    (5%-spaced cuts at free phase alias it into a sawtooth: measured on the
    tiny vacuum scene, a +-0.15% saw at 8 periods on a reading flat to 6e-5).
    ``mismatch`` is what is left of the phase, in periods.

    A cut is ``fitted`` once its window starts after the drive's own
    ``CW_RAMP_PERIODS`` ramp and the reading is a Purcell factor (P > 0).
    """
    n = int(np.size(field))
    frequency = float(reading["frequency"])
    per = _period_steps(dt, frequency)
    window = window_steps(dt, frequency, reading["periods"])
    ramp = int(math.ceil(CW_RAMP_PERIODS * per))
    marks = int(round(1.0 / step))
    reach = max(0, int(step * n / per / 3.0))
    rows, seen = [], set()
    for k in range(1, marks + 1):
        target = (marks - k) * step * n / per      # periods before the end
        best = None
        for j in range(max(0, int(round(target)) - reach),
                       int(round(target)) + reach + 1):
            cut = n - int(round(j * per))
            miss = abs((n - cut) - j * per) / per
            key = (round(miss, 12), abs(j - target))
            if best is None or key < best[0]:
                best = (key, cut, miss)
        _, cut, miss = best
        if cut < window or cut in seen:
            continue
        seen.add(cut)
        try:
            F = cw_reading(drive[:cut], field[:cut], dt, **reading)
            why = None
        except fdtd.FDTDError as exc:
            F, why = None, str(exc)
        rows.append({"cut": cut, "t": cut * dt, "periods": cut / per, "F": F,
                     "why": why, "mismatch": miss,
                     "fitted": F is not None and cut - window >= ramp})
    rows.sort(key=lambda r: r["cut"])
    return {"rows": rows, "window": window, "ramp": ramp, "steps": n, "per": per,
            "corr": hann_correlation([r["cut"] for r in rows if r["fitted"]],
                                     window, per)}


def _rise(u: float, x):
    """``(1 - exp(-u x)) / u``, and its limit ``x`` at ``u = 0``."""
    x = np.asarray(x, dtype=np.float64)
    return x if u == 0.0 else -np.expm1(-u * x) / u


def hann_correlation(cuts, window: int, period_steps: float):
    """How the readings at ``cuts`` correlate when the field's noise is white.

    Two cuts ``d`` steps apart share ``window - d`` samples of their Hann
    windows, and the reading projects each sample on the drive's phase, so
    their noise correlates as ``sum w(s) w(s + d) cos(2 pi d / P) / sum w^2``
    (P the period in steps), and not at all past one window. Measured against
    300 noisy synthetic records (cuts 3 periods apart, 8-period windows):
    0.38 here, 0.35 and 0.43 for the two neighbours of one cut there.
    """
    cuts = [int(c) for c in cuts]
    hann = fdtd._window(int(window), "hann")
    norm = float(np.sum(hann * hann))
    out = np.eye(len(cuts))
    for i, a in enumerate(cuts):
        for j in range(i + 1, len(cuts)):
            d = abs(cuts[j] - a)
            if d < window:
                out[i, j] = out[j, i] = (float(np.sum(hann[d:] * hann[:window - d]))
                                         * math.cos(2.0 * math.pi * d / period_steps)
                                         / norm)
    return out


def fit_approach(t, F, *, floor: float = 0.0, corr=None) -> dict:
    """``F(t) = F_inf - A exp(-t/tau_s)``, with the rate free to change sign.

    Written as ``c + D (1 - e^{-u x}) / u`` on ``x = (t - t0) / S`` (``t0``
    the first point, ``S`` the span), which is linear in (c, D) and smooth
    through ``u = 0`` (a straight line), so ONE fit holds all three shapes: an
    approach (u > 0, limit ``F_inf = c + D/u``), a straight rise (u = 0) and a
    growth with no limit (u < 0, a growth time of ``-S/u``). u is scanned on
    +-``APPROACH_EDGE`` and refined by golden section; (c, D) are linear least
    squares at each u. UNWEIGHTED: every point counts the same.

    The errors are one sigma, and they allow for the points' noise being
    CORRELATED (``corr``, from :func:`hann_correlation`; the identity when
    omitted): the least-squares sandwich ``s2 B J^T V J B``, ``B = (J^T
    J)^-1``, with ``s2 = rss / tr((I - H) V)`` -- the residual's own variance
    corrected for what correlated noise hides in a smooth fit -- floored at
    ``floor`` squared, since no fit knows the curve better than the reading's
    own floor. Fitted on F over its own scale, so a reading of 1e17 is as
    well-conditioned as one of 1 (measured: on raw values the covariance's
    pseudo-inverse dropped a direction and reported a tau error 4.7 x too
    small at 1e17).
    """
    t = np.asarray(t, dtype=np.float64)
    raw = np.asarray(F, dtype=np.float64)
    n = t.size
    if n < 3 or not t[-1] > t[0]:
        raise ValueError(f"a three-parameter fit needs 3 points over a span, got {n}")
    scale = float(np.max(np.abs(raw))) or 1.0
    F = raw / scale
    span = float(t[-1] - t[0])
    x = (t - t[0]) / span
    corr = np.eye(n) if corr is None else np.asarray(corr, dtype=np.float64)

    def solve(u):
        basis = np.column_stack((np.ones(n), _rise(u, x)))
        coef, *_ = np.linalg.lstsq(basis, F, rcond=None)
        resid = F - basis @ coef
        return float(resid @ resid), coef

    grid = np.concatenate((-np.geomspace(APPROACH_EDGE, 1e-3, 40), [0.0],
                           np.geomspace(1e-3, APPROACH_EDGE, 40)))
    scan = [solve(float(u))[0] for u in grid]
    i = int(np.argmin(scan))
    u = _golden(lambda v: solve(v)[0], float(grid[max(i - 1, 0)]),
                float(grid[min(i + 1, grid.size - 1)]), iterations=80)
    rss, (c, D) = solve(u)
    h = 1e-6 * max(abs(u), 1.0)
    jac = np.column_stack((np.ones(n), _rise(u, x),
                           D * (_rise(u + h, x) - _rise(u - h, x)) / (2.0 * h)))
    inner = np.linalg.pinv(jac.T @ jac)
    hat = jac @ inner @ jac.T
    dof = float(np.trace((np.eye(n) - hat) @ corr))
    s2 = max(rss / dof if dof > 0 else math.inf, (float(floor) / scale) ** 2)
    cov = s2 * inner @ jac.T @ corr @ jac @ inner
    u_err = math.sqrt(max(float(cov[2, 2]), 0.0))
    out = {"u": u, "u_err": u_err, "c": float(c) * scale, "D": float(D) * scale,
           "span": span, "t0": float(t[0]), "points": n,
           "rms": math.sqrt(rss / n) * scale,
           "edge": bool(i in (0, grid.size - 1)), "sigma": math.sqrt(s2) * scale,
           "rate": u / span, "rate_err": u_err / span}
    if u > 0.0:
        grad = np.array([1.0, 1.0 / u, -D / u ** 2])
        out.update(F_inf=float(c + D / u) * scale,
                   F_inf_err=math.sqrt(max(float(grad @ cov @ grad), 0.0)) * scale,
                   A=float(D / u) * scale, tau=span / u, tau_err=span * u_err / u ** 2)
    return out


def forecast_check(fit: dict, *, end: float, drift: float, drift_err: float,
                   floor: float) -> dict:
    """Whether an approach's F_inf can stand as the curve's extrapolation.

    ``inconsistent``: the curve is still moving at its end (the final third's
    line, ``drift``, beyond ``SIGNIFICANCE`` of its error) and F_inf sits on the
    wrong side of the end's own reading -- below it on a rise, above it on a
    fall -- by more than ``INCONSISTENT_SIGMA`` of the gap's error (F_inf's and
    the end's together). A limit the record has already passed is wrong,
    not uncertain. The end's error is the floor: the last third's line
    ``noise`` is no error of the end reading, since on a curve that still bends
    it holds the line's misfit (on the change's 600-period curve it was 0.0138
    against a floor of 2e-6). ``unreliable``: the forecast fit's residual is over
    ``FORECAST_RESIDUAL`` x the floor: what it was fitted to still holds more
    than one time constant, or noise the floor does not account for.
    """
    gap = float(fit["F_inf"]) - float(end)
    gap_err = math.hypot(float(fit["F_inf_err"]), float(floor))
    moving = abs(drift) > SIGNIFICANCE * drift_err
    wrong_side = gap * drift < 0.0
    ratio = fit["rms"] / floor if floor > 0 else math.inf
    return {"gap": gap, "gap_err": gap_err, "residual_ratio": ratio,
            "inconsistent": bool(moving and wrong_side
                                 and abs(gap) > INCONSISTENT_SIGMA * gap_err),
            "unreliable": bool(ratio > FORECAST_RESIDUAL)}


def classify_settling(curve: dict, *, floor: float = 0.0) -> dict:
    """Which of four things the curve is, with the numbers that say so.

    ``floor`` is the reading's own floor at the curve's cuts (what is left of
    the 2f swing at their phase mismatch, ``settling_report``); the noise the
    tests below use is the larger of it and the fit's residual rms, so every
    threshold is a measured error times ``SIGNIFICANCE``:

    (a) ``settled``: the final third moved less than its own noise, and the
        approach, if there is one, has less left in it than its error;
    (b) ``approaching``: the fitted rate is positive by more than
        ``SIGNIFICANCE`` errors -- ``F_inf`` is then an EXTRAPOLATION, checked
        against the record by :func:`forecast_check`;
    (c) ``growing``: the rate is negative by as much -- growth the approach
        model does not fit, never a Purcell factor;
    (d) ``unresolved``: too few cuts, or a rate within its errors of zero (a
        straight rise: the record cannot tell a slow approach from growth).

    ``whole`` is the fit over every fitted cut; ``fit`` is the forecast, which
    is ``whole`` unless its residual is over ``SIGNIFICANCE`` x the floor, when
    it is refit over the cuts in the record's last ``TAIL_SHARE``.
    """
    rows = [r for r in curve["rows"] if r["fitted"]]
    out = {"rows": len(rows), "floor": floor}
    if len(rows) < MIN_FIT_CUTS:
        out.update(kind="unresolved",
                   why=f"only {len(rows)} cuts read a window past the drive's"
                       f" {CW_RAMP_PERIODS:g}-period ramp, and the fit needs"
                       f" {MIN_FIT_CUTS}: record longer")
        return out
    t = [r["t"] for r in rows]
    F = [r["F"] for r in rows]
    scale = max(abs(v) for v in F)
    floor = max(float(floor), 1e-12 * scale)
    out["floor"] = floor
    corr = curve["corr"]
    whole = fit_approach(t, F, floor=floor, corr=corr)
    fit, tail = whole, None
    # One time constant that does not hold the whole curve (residual above the
    # floor) is pulled down by the early fast rise: its tau_s is short and its
    # F_inf low (change settling-tail-fit). The forecast is then refit over the
    # cuts in the last TAIL_SHARE of the record, where the slow part is left.
    if whole["rms"] > SIGNIFICANCE * floor:
        tail = [k for k, r in enumerate(rows)
                if r["cut"] >= (1.0 - TAIL_SHARE) * curve["steps"]]
        if len(tail) < MIN_FIT_CUTS:
            tail = list(range(len(rows) - MIN_FIT_CUTS, len(rows)))
        if tail[0] == 0:
            tail = None
        else:
            fit = fit_approach([t[k] for k in tail], [F[k] for k in tail],
                               floor=floor, corr=corr[np.ix_(tail, tail)])
    # The final third, read as a straight line: its slope against the line's own
    # residual (floored, with the same correlated-noise sandwich as the fit) is a
    # test of "the last cuts agree within the noise" that no model of the whole
    # curve enters -- a one-exponential misfit's residual is not noise.
    third = [k for k, r in enumerate(rows) if 3 * r["cut"] >= 2 * curve["steps"]]
    if len(third) < 3:
        third = list(range(len(rows) - 3, len(rows)))
    tt = np.array([rows[k]["t"] for k in third])
    ff = np.array([rows[k]["F"] for k in third])
    sub = corr[np.ix_(third, third)]
    width = float(tt[-1] - tt[0])
    basis = np.column_stack((np.ones(tt.size), (tt - tt[0]) / width))
    inner = np.linalg.pinv(basis.T @ basis)
    coef = inner @ basis.T @ ff
    resid = ff - basis @ coef
    dof = float(np.trace((np.eye(tt.size) - basis @ inner @ basis.T) @ sub))
    s2 = max(float(resid @ resid) / dof if dof > 0 else math.inf, floor ** 2)
    noise = math.sqrt(s2)
    drift = float(coef[1])
    drift_err = math.sqrt(max(float((s2 * inner @ basis.T @ sub @ basis @ inner)[1, 1]),
                              0.0))
    out.update(fit=fit, whole=whole, tail_from=(tail[0] if tail else 0),
               noise=noise, drift=drift, drift_err=drift_err,
               third_from=third[0], end=rows[-1]["F"])
    if abs(drift) <= SIGNIFICANCE * drift_err:
        out.update(kind="settled")
    elif fit["u"] - SIGNIFICANCE * fit["u_err"] > 0.0 and not fit["edge"]:
        out.update(kind="approaching")
        out.update(forecast_check(fit, end=rows[-1]["F"], drift=drift,
                                  drift_err=drift_err, floor=floor))
    elif fit["u"] + SIGNIFICANCE * fit["u_err"] < 0.0:
        out.update(kind="growing")
    else:
        bound = fit["u"] + SIGNIFICANCE * fit["u_err"]
        out.update(kind="unresolved", bound=(fit["span"] / bound if bound > 0 else None),
                   why="the fitted rate is within its errors of zero: a straight"
                       " rise over this record, which a slow approach and a"
                       " growth without a limit both make")
    return out


def parse_locator(text):
    """``--locator-tau``: ``TAU`` or ``TAU,ERR``, both in fs, TAU > 0."""
    try:
        parts = [float(p) for p in str(text).split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--locator-tau wants fs or fs,fs, got {text!r}")
    if len(parts) not in (1, 2) or not parts[0] > 0 or (len(parts) == 2 and parts[1] < 0):
        raise argparse.ArgumentTypeError(
            f"--locator-tau wants a positive tau in fs and an optional error, got {text!r}")
    return (parts[0], parts[1] if len(parts) == 2 else 0.0)


def _approach_text(fit: dict, verdict: dict, frequency: float, curve: dict,
                   dt: float) -> str:
    """One approach fit as the ``fit`` line prints it."""
    per_s = 1.0 / frequency
    return (f"F_inf = {fit['F_inf']:.6g} +- {fit['F_inf_err']:.3g},"
            f" A = {fit['A']:.4g} at {fit['t0'] * frequency:.1f} periods,"
            f" tau_s = {fit['tau'] / 1e-15:.4g} +- {fit['tau_err'] / 1e-15:.3g}"
            f" fs = {fit['tau'] / per_s:.4g} periods"
            f" ({fit['tau'] / (curve['steps'] * dt):.3g} of the record);"
            f" residual rms {fit['rms']:.3g}"
            f" ({fit['rms'] / verdict['floor']:.3g} x the floor)")


def settling_report(args, drive, field, dt: float, *, F: float, early: float,
                    **reading) -> dict:
    """The curve, the window's phase, the fit and the class, printed.

    Every line here is NEW and printed after the two existing ones, which are
    left exactly as they were; nothing here changes F.
    """
    frequency = float(reading["frequency"])
    per_s = 1.0 / frequency
    phase = window_phase(drive, field, dt, **reading)
    curve = settling_curve(drive, field, dt, **reading)
    rows = curve["rows"]
    fitted = [r for r in rows if r["fitted"]]
    floor = 0.0
    if phase is not None:
        miss = max((r["mismatch"] for r in fitted), default=0.0)
        floor = phase["swing"] * 4.0 * math.pi * miss
        rel = phase["swing"] / abs(phase["mean"]) if phase["mean"] else math.inf
        print(f"phase     the window's end moved through the last drive period"
              f" ({len(phase['cuts'])} cuts): F swings +-{phase['swing']:.4g}"
              f" ({100.0 * rel:.3g}% of F) at twice the drive frequency, where"
              f" the Hann's leakage of the -f image ({phase['leak']:.3g}), twice,"
              f" times the field's quadrature ratio ({phase['quadrature']:.4g})"
              f" predicts +-{phase['predicted']:.4g}; with that term taken out"
              f" the end reads {phase['mean']:.4f}")
        print(f"phase     so one reading can sit up to {phase['swing']:.4g} off by"
              f" its window's phase alone, and two up to {2 * phase['swing']:.4g}"
              f" apart: the two-thirds line's {F - early:+.4f} is"
              f" {abs(F - early) / (2 * phase['swing']) if phase['swing'] else math.inf:.3g}"
              f" of that")
    print(f"curve     the same reading at cuts a whole number of periods before the"
          f" end (so every window ends at the end's phase), one per"
          f" {100 * SETTLING_STEP:g}% of the record; * = fitted (the window starts"
          f" after the {CW_RAMP_PERIODS:g}-period ramp)")
    for r in rows:
        value = (f"{r['F']:.6g}" if r["F"] is not None
                 else "no reading (" + (r["why"] or "")[:40] + ")")
        print(f"curve     {100.0 * r['cut'] / curve['steps']:5.1f}%"
              f" {r['periods']:8.1f} periods  F = {value}"
              f"{'  *' if r['fitted'] else ''}")
    verdict = classify_settling(curve, floor=floor)
    fit = verdict.get("fit")
    whole = verdict.get("whole")
    if whole is not None:
        print(f"fit       F(t) = F_inf - A exp(-t/tau_s) over the {whole['points']}"
              f" fitted cuts, unweighted least squares; errors are 1 sigma,"
              f" allowing for neighbouring cuts sharing window samples (their"
              f" Hann overlap), from the residual or the phase floor"
              f" ({verdict['floor']:.3g}), whichever is larger")
        if "tau" in whole:
            print(f"fit       {_approach_text(whole, verdict, frequency, curve, dt)}")
        else:
            grow = (f"a growth time of {-whole['span'] / whole['u'] / 1e-15:.4g} fs ="
                    f" {-whole['span'] / whole['u'] / per_s:.4g} periods"
                    if whole["u"] < 0 else "no curvature at all")
            print(f"fit       no limit: the rate is {whole['rate'] * 1e-15:+.4g}"
                  f" +- {whole['rate_err'] * 1e-15:.3g} per fs, {grow};"
                  f" residual rms {whole['rms']:.3g}"
                  f" ({whole['rms'] / verdict['floor']:.3g} x the floor)")
    if fit is not None and fit is not whole:
        lead = (f"the residual is {whole['rms'] / verdict['floor']:.3g} x the"
                f" floor, so the curve holds more than this one time constant and"
                f" its early rise pulls tau_s short and F_inf low; the forecast is"
                f" refit over the last {100 * TAIL_SHARE:g}% of the record"
                f" ({fit['points']} cuts from {fit['t0'] * frequency:.1f} periods)")
        if "tau" in fit:
            print(f"fit       {lead}: {_approach_text(fit, verdict, frequency, curve, dt)}")
        else:
            print(f"fit       {lead}: no limit there (the rate is"
                  f" {fit['rate'] * 1e-15:+.4g} +- {fit['rate_err'] * 1e-15:.3g}"
                  f" per fs)")
    kind = verdict["kind"]
    if kind == "settled":
        print(f"class     (a) settled: a line through the last third moves"
              f" {verdict['drift']:+.4g} +- {verdict['drift_err']:.3g} across it,"
              f" within {SIGNIFICANCE:g} of its errors (the larger of the line's"
              f" residual and the floor): F = {verdict['end']:.6g} at the end is"
              f" the steady state as far as this record can tell")
    elif kind == "approaching" and verdict["inconsistent"]:
        way = "rising" if verdict["drift"] > 0 else "falling"
        side = "below" if verdict["drift"] > 0 else "above"
        print(f"class     (b) approaching, but the forecast is INCONSISTENT with"
              f" the record: the curve is still {way} at its end, which reads"
              f" {verdict['end']:.6g}, and the fit's F_inf = {fit['F_inf']:.6g} +-"
              f" {fit['F_inf_err']:.3g} lies {side} it by"
              f" {abs(verdict['gap']) / verdict['gap_err']:.3g} of their joint"
              f" error (over {INCONSISTENT_SIGMA:g}); no extrapolation is given --"
              f" the limit is at least as far as the end")
    elif kind == "approaching":
        trust = (f"; the forecast is UNRELIABLE: the fit's residual is"
                 f" {verdict['residual_ratio']:.3g} x the floor (over"
                 f" {FORECAST_RESIDUAL:g}): what it was fitted to still holds"
                 f" more than one time constant, or noise the floor does not"
                 f" account for"
                 if verdict["unreliable"] else "")
        print(f"class     (b) approaching: the rate is"
              f" {fit['u'] / fit['u_err'] if fit['u_err'] else math.inf:.3g} of its"
              f" errors above zero, so F_inf = {fit['F_inf']:.6g} +-"
              f" {fit['F_inf_err']:.3g} is an EXTRAPOLATION of this curve, not a"
              f" reading: the end reads {verdict['end']:.6g}{trust}")
    elif kind == "growing":
        print(f"class     (c) growing without a limit: the approach model does not"
              f" fit (its rate is {-fit['u'] / fit['u_err'] if fit['u_err'] else math.inf:.3g}"
              f" errors below zero). A likely numerical problem: the F above is"
              f" NOT a Purcell factor")
    else:
        extra = (f"; tau_s, if it is an approach, is above"
                 f" {verdict['bound'] / 1e-15:.4g} fs"
                 if verdict.get("bound") else "")
        print(f"class     (d) unresolved: {verdict['why']}{extra}")
    locator = getattr(args, "locator_tau", None)
    if locator is not None:
        locator_line(locator, verdict, frequency=frequency, early=early, F=F,
                     steps=curve["steps"], dt=dt)
    verdict.update(curve=curve, phase=phase)
    return verdict


def locator_line(locator, verdict: dict, *, frequency: float, early: float,
                 F: float, steps: int, dt: float) -> str:
    """The curve's tau_s against a ring-down locator's tau, in plain words."""
    tau_l, err_l = float(locator[0]) * 1e-15, float(locator[1]) * 1e-15
    fit = verdict.get("fit") or {}
    # what one mode with the locator's tau would leave between the two-thirds cut
    # and the end, as a share of the change the curve made from its first cut
    lines = []
    if verdict["kind"] == "approaching" and "tau" in fit:
        tau, err = fit["tau"], fit["tau_err"]
        ratio = tau / tau_l
        t1 = fit["t0"]
        t23 = (2 * steps // 3) * dt
        tend = steps * dt
        share = ((math.exp(-(t23 - t1) / tau_l) - math.exp(-(tend - t1) / tau_l))
                 / max(1.0 - math.exp(-(tend - t1) / tau_l), 1e-300))
        lines.append(
            f"locator   tau_s = {tau / 1e-15:.4g} +- {err / 1e-15:.3g} fs against the"
            f" locator's {tau_l / 1e-15:.4g} +- {err_l / 1e-15:.3g} fs: {ratio:.3g} x."
            f" One mode with the locator's tau would make {100 * share:.3g}% of"
            f" this curve's rise after the two-thirds cut")
        if tau - SIGNIFICANCE * err > tau_l + SIGNIFICANCE * err_l:
            lines.append(
                "locator   tau_s is longer than the locator's tau beyond both"
                " errors, so a single mode with the locator's tau cannot make"
                " this curve. Something slower than that mode is moving the"
                " reading -- or the locator's tau is not this mode's: a"
                " ring-down fit whose frequency grid is coarser than the line can"
                " stop on a short tau (change cw-settling measured fit_decay at"
                " 113 fs on a 252 fs mode); read its rms against its peak.")
        elif tau + SIGNIFICANCE * err < tau_l - SIGNIFICANCE * err_l:
            lines.append("locator   tau_s is shorter than the locator's tau beyond"
                         " both errors")
        else:
            lines.append("locator   tau_s agrees with the locator's tau within"
                         " their errors: the curve is that mode's own approach")
    else:
        lines.append(f"locator   the locator's tau is {tau_l / 1e-15:.4g} fs; this"
                     f" curve is {verdict['kind']}, so it has no tau_s to set"
                     f" against it")
    for line in lines:
        print(line)
    return "\n".join(lines)


SERIES_KEYS = ("drive", "source_field", "dt", "frequency_hz", "volume", "eps_r",
               "index", "window_periods", "wavelength_nm")


def save_series(path, drive, field, *, dt: float, frequency: float, volume: float,
                eps_r: float, index: float, window_periods: float,
                wavelength_nm: float) -> str:
    """Write the CW record the reading came from, compressed; returns the path.

    Everything ``cw_reading`` needs travels with the two series, at the dtype
    the run recorded them in, so a re-read of the file gives the identical F
    and the curve can be re-cut (another window, another fit) without a run.
    """
    np.savez_compressed(path, drive=np.asarray(drive), source_field=np.asarray(field),
                        dt=float(dt), frequency_hz=float(frequency), volume=float(volume),
                        eps_r=float(eps_r), index=float(index),
                        window_periods=float(window_periods),
                        wavelength_nm=float(wavelength_nm))
    return _npz_path(path)


def load_series(path) -> dict:
    """The other half of :func:`save_series`, read by the argument it was given."""
    with np.load(_npz_path(path)) as z:
        missing = [k for k in SERIES_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{_npz_path(path)} is not a --save-series record:"
                             f" it has no {', '.join(missing)}")
        out = {k: (z[k] if z[k].ndim else z[k].item()) for k in SERIES_KEYS}
    return out


def read_cw(args, drive, field, dt: float, *, volume: float, eps_r: float,
            index: float, frequency: float, wavelength_nm: float):
    """The CW reading of one record, printed: the two lines it always had, the
    ``--save-series`` write if asked, then the settling block. Returns (F, P)."""
    reading = {"volume": volume, "eps_r": eps_r, "index": index,
               "frequency": frequency, "periods": args.window_periods}
    power, reference = cw_powers(drive, field, dt, **reading)
    F = fdtd.purcell_factor(power, reference)
    print(f"measured  F = {F:.4f} at {wavelength_nm:.0f} nm  (P = {power:.6e} W,"
          f" P0 = {reference:.6e} W)")
    # One number cannot say whether the window it came from had stopped rising:
    # the same mean over the same window, ended at two places in one record, can.
    # delivered_power and dipole_moment both take the LAST `periods` of the series
    # they are handed (`fdtd.delivered_power` slices drive[size - steps:] itself),
    # so the earlier reading is those same functions on this record cut at the
    # two-thirds step: the same running scene, not a second run and not a
    # different estimator.
    cut = (2 * np.size(field)) // 3
    early = cw_reading(drive[:cut], field[:cut], dt, **reading)
    spread = (f", {100.0 * (F / early - 1.0):+.1f}% between them"
              if early > 0.0 else "")
    print(f"settling  F = {early:.4f} at two thirds of the record against"
          f" {F:.4f} at its end{spread}")
    path = getattr(args, "save_series", None)
    if path:
        # After the reading is printed and before the new analysis: a failed write
        # cannot take the number down, and a fault in the analysis cannot lose
        # the record a full-size run paid for.
        try:
            wrote = save_series(path, drive, field, dt=dt, frequency=frequency,
                                volume=volume, eps_r=eps_r, index=index,
                                window_periods=args.window_periods,
                                wavelength_nm=wavelength_nm)
        except OSError as exc:
            print(f"warning   the series was not written ({type(exc).__name__}:"
                  f" {exc}): the reading above is this run's own, and a re-read"
                  f" of this record now needs another run")
        else:
            print(f"series    {np.size(field):,} steps of drive and source field,"
                  f" with dt and the reading's constants, in {wrote}:"
                  f" --from-series re-reads it without a run")
    settling_report(args, drive, field, dt, F=F, early=early, **reading)
    return F, power


def from_series(args) -> int:
    """``--from-series``: the CW reading of a saved record, without a run."""
    try:
        rec = load_series(args.from_series)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    print(f"series    {np.size(rec['source_field']):,} steps from"
          f" {_npz_path(args.from_series)}: dt {rec['dt']:.6e} s, f"
          f" {rec['frequency_hz'] / 1e12:.4f} THz, window {args.window_periods:g}"
          f" periods (the run read {rec['window_periods']:g})")
    read_cw(args, rec["drive"], rec["source_field"], rec["dt"], volume=rec["volume"],
            eps_r=rec["eps_r"], index=rec["index"], frequency=rec["frequency_hz"],
            wavelength_nm=rec["wavelength_nm"])
    return 0


def measure(args, lay, request, measured):
    """The CW arm: one frequency, a settled tail, F = P/P0."""
    sim, volume, index, eps_applied = _run(args, lay, request, measured)
    dt = measured["est"]["dt"]
    F, power = read_cw(args, sim["drive"], sim["source_field"], dt, volume=volume,
                       eps_r=eps_applied, index=index,
                       frequency=request["frequency_hz"],
                       wavelength_nm=args.wavelength)
    if measured.get("phasor") is not None:
        measured["collection"] = collection(
            args, sim["phasor_faces"][0], float(request["frequency_hz"]), power)
    return F


def measure_spectrum(args, lay, request, measured):
    """The band arm: a pulse centred in its own record, read bin by bin."""
    sim, volume, index, eps_applied = _run(args, lay, request, measured)
    half = request["bandwidth"]
    f, power, reference, F = fdtd.purcell_spectrum(
        sim["drive"], sim["source_field"], measured["est"]["dt"], volume=volume,
        eps_r=eps_applied, index=index,
        band=(request["frequency_hz"] - half, request["frequency_hz"] + half))
    lam = C / np.asarray(f) / NM
    top = int(np.argmax(F))
    print(f"spectrum  {len(f)} bins, {lam[0]:.0f}-{lam[-1]:.0f} nm at"
          f" {measured['dlambda_nm']:.2f} nm per bin")
    # Every bin, not only the peak. A peak sitting on the band's edge, or a band
    # whose floor never comes back to 1, is itself the diagnosis, and the shape
    # string alone cannot say how far above 1 the floor is.
    for wavelength, value in zip(lam, F):
        print(f"bin       {wavelength:7.1f} nm   F = {value:9.4f}")
    print(f"peak      F = {F[top]:.4f} at {lam[top]:.1f} nm")
    span = max(float(np.max(F)), 1e-30)
    print("shape     " + "".join(
        " .:-=+*#%@"[int(min(9, max(0, round(9 * float(v) / span))))] for v in F))
    return f, F


def tiny(args) -> int:
    """The tier-1 verdict: the scene builds by the product's rules and reads 1.

    Scaled down so the record stays short: 20 nm cells and a 40 nm buffer, so
    the buffer is resolved by the thin-layer rule rather than by a 12 nm buffer
    split three ways. The record itself is the script's own 32 periods, not a
    shortened one: the vacuum reading settles only after about 16 (a hard
    started CW drive at 8 periods read 4.61 with a 4-period tail). What it
    checks is the same either way -- the buffer keeping cells of its own and
    inside the rule's cap, the gold at the part's own z and clear of the
    absorber, the source in the wire and clear of both the facet and the
    absorber, the scene this script builds inside the same
    caps a Jetson run answers to, and the normalisation reading F = 1 on a
    homogeneous scene through this script's own path. The geometry controls are
    read off the real scene's layout, which needs no solve. Two more are the
    tail arm's own: the cut through the record it fits is put against the
    product's pulse envelope's algebra, and the band and tail arms are asked to
    refuse each other's record. With --ringdown the homogeneous scene runs the
    tail arm instead of the band arm, so the fit itself is exercised here too.
    """
    base = dict(args.__dict__)
    ok = []

    unit = _hex_prism(0.0, 1.0, 100.0, 0.0)
    exact = 3.0 * math.sqrt(3.0) / 2.0 * 100.0 ** 2
    ok.append((f"the prism mesh is closed and its volume is the hexagon's"
               f" ({unit.volume:.4f} against {exact:.4f})",
               bool(unit.is_watertight)
               and abs(unit.volume - exact) / exact < 1e-9))

    ok.append((f"the product's own clock is the tier-2 timeout"
               f" ({fdtd.MAX_SECONDS:.0f} s, and refusals leans on it)",
               float(fdtd.MAX_SECONDS) == TIER2_SECONDS))

    real = args
    lay, request, geom = solve_scene(real)
    measured = filled(real, lay, request)
    code = report(real, lay, request, geom, measured)      # the run's own block
    if code:                                    # the pml guard, not a cap
        return code
    why = refusals(real, measured, lay, geom)
    if why:
        print(why)
        return 2

    # The tail arm's window is a cut through the record at the drive's own end.
    # The control drives a record with the product's own pulse profile and asks
    # the cut to land where that envelope's algebra says it should -- two
    # readings of one pulse, so a floor or a time base that drifts shows up here.
    dt = measured["est"]["dt"]
    sigma = 1.0 / (math.pi * request["bandwidth"])
    synth = argparse.Namespace(**{**base, "profile": "pulse", "spectrum": False,
                                  "ringdown": True, "tail_periods": 8.0})
    n_syn = steps_for(synth, request, dt)
    drive = np.array([fdtd.pulse_profile(k * dt, float(request["frequency_hz"]),
                                         float(request["bandwidth"]))
                      for k in range(n_syn)])
    cut = drive_done(drive)
    want = pulse_end_sigma() * sigma
    tail = (n_syn - cut) * dt * float(request["frequency_hz"])
    ok.append((f"the product's own pulse ends where its envelope says"
               f" ({cut * dt / sigma:.2f} sigma against {want / sigma:.2f}) and a"
               f" span of 8 plus a tail of 8 leaves {tail:.1f} periods of it",
               abs(cut * dt - want) < 0.25 * sigma and tail >= MIN_TAIL_PERIODS))

    # The fit's frequency window has to follow the band the drive excited. This
    # fixture is that case and only that case: a tail carrying a mode 10 per
    # cent BELOW the drive's centre, which a +-15 per cent band reaches and the
    # narrow default cannot. The narrow reading is asserted to FAIL on it, so
    # the control cannot pass by both windows agreeing.
    f_c = float(request["frequency_hz"])
    half_syn, f_off = 0.15 * f_c, 0.90 * f_c
    band_syn = (f_c - half_syn, f_c + half_syn)
    req_syn = {"frequency_hz": f_c, "bandwidth": half_syn}
    t_syn = np.arange(max(400, int(12.0 / (dt * f_c)))) * dt
    syn = np.exp(-t_syn / (6.0 / f_off)) * np.cos(2.0 * math.pi * f_off * t_syn)
    wide = fit_decay(syn, dt, 0, frequency=f_c, band=band_syn)
    narrow = fit_decay(syn, dt, 0, frequency=f_c)
    ok.append((f"the fit searches the band it is given, not a fixed +-5 per cent"
               f" ({wide['f'] / 1e12:.2f} THz against {f_off / 1e12:.2f} sought,"
               f" where the narrow window stops at {narrow['f'] / 1e12:.2f} and"
               f" flags itself at {narrow['f_edge']})",
               abs(wide["f"] - f_off) < 0.01 * f_off
               and not wide["f_edge"]
               and abs(narrow["f"] - f_off) > 0.03 * f_off
               and narrow["f_edge"] is True))

    # The record has to outlive the run. One decay is the model the arm fits and a
    # second one is a second model, so without the samples a two-mode fit means
    # another Orin run. The control is the round trip, and it is read back by the
    # same argument the writer was handed rather than by the path it wrote: what
    # comes off disk must re-fit to the line that was fitted in memory, or the
    # file is decoration.
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_record(tmp + "/tail", syn, dt, 0, req_syn)
        back = load_record(tmp + "/tail")
        again = fit_decay(back["values"], back["dt"], back["start"],
                          frequency=back["frequency"], band=back["band"])
    ok.append((f"a saved tail re-fits to the line it was fitted to"
               f" ({again['f'] / 1e12:.4f} THz against {wide['f'] / 1e12:.4f} in"
               f" memory, from {back['values'].size} samples written to"
               f" {saved.rsplit('/', 1)[-1]})",
               saved == tmp + "/tail.npz"
               and back["values"].size == syn.size
               and again["f"] == wide["f"] and again["tau"] == wide["tau"]))

    # The arm catches OSError around the write, so the control has to ask the real
    # raise site for its type: a path that cannot be written must produce the one
    # exception the arm catches, and not one that would escape the catch and end the
    # run after the fit had already been computed.
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "not-a-directory"
        blocker.write_text("x")
        try:
            save_record(str(blocker / "tail"), syn, dt, 0, req_syn)
            raised = None
        except OSError as exc:
            raised = type(exc).__name__
    ok.append((f"a tail that cannot be written raises what the arm catches"
               f" ({raised or 'nothing'} is an OSError)", raised is not None))

    # save_tail is STATED here, not inherited: a control that asks arm_conflicts
    # about an arm has to name every key that function reads, or the caller's own
    # flags leak into the arm the control is describing. It did: --tiny --ringdown
    # --save-tail <path> made --save-tail itself the reason the band arm read
    # dirty, so the run failed its own control (found by driving the flag end to
    # end, which is what the flag's own scenario asks for).
    as_tail = argparse.Namespace(**{**base, "profile": "pulse", "ringdown": True,
                                    "spectrum": True, "tail_periods": 20.0,
                                    "save_tail": None})
    as_band = argparse.Namespace(**{**base, "profile": "pulse", "ringdown": False,
                                    "spectrum": True, "tail_periods": 0.0,
                                    "save_tail": None})
    ok.append(("the band arm and the tail arm refuse to share one record"
               f" ({arm_conflicts(as_tail)[:40] or 'NOTHING'}), and the band arm"
               f" alone is clean",
               bool(arm_conflicts(as_band)) is False
               and bool(arm_conflicts(as_tail)) is True))

    # --save-tail is read in the tail arm and nowhere else, so on any other run
    # it would be inert: no record written, no warning, exit 0. The refusal is
    # asked of the same function the launch asks, on both of the arms that have
    # no tail to save, and the tail arm is asked to keep its write.
    as_cw_save = argparse.Namespace(**{**base, "profile": "cw", "ringdown": False,
                                       "spectrum": False, "tail_periods": 0.0,
                                       "save_tail": "logs/tail"})
    as_band_save = argparse.Namespace(**{**base, "profile": "pulse",
                                         "ringdown": False, "spectrum": True,
                                         "tail_periods": 0.0,
                                         "save_tail": "logs/tail"})
    as_tail_save = argparse.Namespace(**{**base, "profile": "pulse",
                                         "ringdown": True, "spectrum": False,
                                         "tail_periods": 20.0,
                                         "save_tail": "logs/tail"})
    ok.append(("--save-tail without the tail arm is refused"
               f" ({arm_conflicts(as_cw_save)[:40] or 'NOTHING'}), on the CW and"
               " the band arm alike, and the tail arm keeps its write",
               "--save-tail" in arm_conflicts(as_cw_save)
               and "--save-tail" in arm_conflicts(as_band_save)
               and arm_conflicts(as_tail_save) == ""))

    z = np.asarray(lay["nodes"][2])
    oxide = _oxide_slice(z, geom)
    cap = real.oxide * NM / fdtd_layout.THIN_CELLS
    ok.append((f"the buffer keeps >= MIN_CELLS cells ({oxide.size} at"
               f" {oxide.mean() / NM:.2f} nm)",
               oxide.size >= fdtd_grid.MIN_CELLS))
    ok.append((f"every buffer cell is inside the thin-layer rule's cap"
               f" (max {oxide.max() / NM:.2f} nm <= {cap / NM:.2f} nm)",
               oxide.size > 0 and float(oxide.max()) <= cap * 1.001))
    ok.append(("those cells are finer than the coarse cell",
               oxide.size > 0 and float(oxide.max()) < real.dx * NM * 0.999))
    ok.append(("the product's own check_axis passes",
               not fdtd_grid.check_axis(z, real.dx * NM)))
    gold = lay["metals"].get(gold_metal(args))
    ok.append(("the gold is a region", gold is not None and bool(gold.any())))
    if gold is not None and gold.any():
        kz = np.flatnonzero(gold.any(axis=(0, 1)))
        lo, hi = float(z[kz[0]]), float(z[kz[-1] + 1])
        ok.append((f"the gold's voxel edges sit at the part's own z, not a"
                   f" shifted frame ({lo / NM:.1f}..{hi / NM:.1f} nm against"
                   f" 0..{real.gold:.0f})",
                   abs(lo) <= real.dx * NM * 0.999
                   and abs(hi - real.gold * NM) <= real.dx * NM * 0.999))
        ok.append((f"the gold is clear of the absorber"
                   f" ({lo / NM - z[0] / NM:.0f} nm of pad against"
                   f" {real.pml * real.dx:.0f} nm)",
                   lo - float(z[0]) >= real.pml * real.dx * NM))
    cell = tuple(int(c) for c in lay["cell"])
    ok.append(("the source sits in InP",
               abs(float(lay["eps"][cell]) - N_INP ** 2) < 1e-4))
    ok.append(("the source is clear of the facet",
               (geom["z_top"] - float(z[cell[2]])) >= 2.0 * real.dx * NM))
    ok.append((f"the source is clear of the absorber"
               f" ({float(z[cell[2]] - z[0]) / NM:.0f} and"
               f" {float(z[-1] - z[cell[2]]) / NM:.0f} nm against"
               f" {real.pml * real.dx:.0f} nm)",
               float(z[cell[2]] - z[0]) > real.pml * real.dx * NM
               and float(z[-1] - z[cell[2]]) > real.pml * real.dx * NM))

    # --qd-below is the paper's geometry, and the source is snapped to the
    # nearest node unless its own z is a node, so a mesh change MOVES it. The
    # control plans two scenes -- this control's own, and the paper's as its
    # parser's own defaults build it rather than as a retyped copy -- at two dx
    # values, with and without the pin. Both columns are the point: an equality
    # against the pinned plan alone passes with the pin inert on any mesh whose
    # nodes happen to land on the request, and the depth the bench runs actually
    # got is the un-pinned one -- 39.99 nm at dx 20 and 29.89 at dx 30 for the
    # paper's 30, read off rung 8's own `dipole` line, which is how this was
    # found. One pitch is one dx, so the two dx values are the control.
    rows = []
    for label, ns in (("tiny", argparse.Namespace(**base)),
                      ("paper", parse_args([]))):
        for dx_nm in (20.0, 30.0):
            sub = argparse.Namespace(**{**ns.__dict__, "dx": dx_nm})
            names_s, parts_s, materials_s, req_s, geom_s = scene(sub)
            rows.append((f"{label} dx {dx_nm:g}", float(sub.qd_below),
                         emitter_depth_nm(fdtd_layout.plan(
                             names_s, parts_s, req_s, length_scale=1.0,
                             materials=materials_s), geom_s),
                         emitter_depth_nm(fdtd_layout.plan(
                             names_s, parts_s,
                             {k: v for k, v in req_s.items()
                              if k != "pin_source_z"},
                             length_scale=1.0, materials=materials_s), geom_s)))
    slides = {}
    for lab, want, _, snap in rows:
        slides.setdefault(lab.split()[0], []).append(abs(snap - want))

    # Built by a helper, not inline: the bench's python is 3.10, where an
    # f-string cannot contain the quote it is itself written in -- an inline
    # `{"*" if ...}` in this same double-quoted f-string is a SyntaxError there
    # (PEP 701 lifted that in 3.12, which is what this VM runs), and the first
    # version of this control was committed that way and died on the bench
    # before a single step. Tier 1 runs the VM's python, so it cannot see it.
    def label(lab, want, snap):
        over = "*" if abs(snap - want) > QD_DEPTH_TOL_NM else ""
        return f"{lab} {snap:.2f}{over}"

    ok.append((f"the emitter's own z is a node at both dx, so every mesh puts"
               f" it at --qd-below"
               f" ({', '.join(f'{lab} {pin:.2f}' for lab, _, pin, _ in rows)} nm)"
               f" where the nearest-node snap gives"
               f" ({', '.join(label(lab, want, snap) for lab, want, _, snap in rows)}"
               f" nm; * is more than the {QD_DEPTH_TOL_NM:g} nm the refusal allows)",
               all(abs(pin - want) <= 1e-6 for _, want, pin, _ in rows)
               and all(any(off > QD_DEPTH_TOL_NM for off in v)
                       for v in slides.values())))

    # The pin can be dropped -- a line out of range, or within LINE_SNAP of an
    # extent edge -- and then the source is snapped to the nearest node with
    # nothing said. So the refusal is asked of the same function the launch
    # calls, with a plan whose depth IS out of tolerance (the un-pinned plan
    # this scene produces) and with the run's own plan: the first has to be
    # refused, naming the depth it realised, and the second has to pass.
    names_s, parts_s, materials_s, req_s, geom_s = scene(
        argparse.Namespace(**{**base, "dx": 20.0}))
    off = refusals(argparse.Namespace(**{**base, "dx": 20.0}),
                   {"exceeded": None},
                   fdtd_layout.plan(names_s, parts_s,
                                    {k: v for k, v in req_s.items()
                                     if k != "pin_source_z"},
                                    length_scale=1.0, materials=materials_s),
                   geom_s)
    ok.append((f"a mesh that does not put the emitter at --qd-below is refused"
               f" ({off[:70] or 'NOTHING'})",
               f"{float(base['qd_below']):g}" in off and "nm out" in off
               and refusals(real, measured, lay, geom) == ""))

    vac = argparse.Namespace(**{**base, "vacuum": True})
    lay_v, request_v, geom_v = solve_scene(vac)
    measured_v = filled(vac, lay_v, request_v)
    if vac.ringdown:
        fit, code = ringdown(vac, lay_v, request_v, measured_v)
        ok.append((f"the tail arm runs on a homogeneous scene and answers either"
                   f" way (tau {fit['tau'] / 1e-15:.3f} fs +-"
                   f" {_err(fit['tau_err'], 1e-15)},"
                   f" {'pinned' if not code else 'unresolved'})"
                   if fit is not None else
                   "the tail arm answers on a homogeneous scene (it refused this"
                   " window's own drive instead)",
                   fit is not None and code in (0, 3)))
    elif vac.profile == "pulse":
        _, F_bins = measure_spectrum(vac, lay_v, request_v, measured_v)
        worst = float(np.max(np.abs(np.asarray(F_bins) - 1.0)))
        ok.append((f"vacuum F = 1 across the band (worst bin {worst:.4f})",
                   worst < 0.05))
    else:
        F = measure(vac, lay_v, request_v, measured_v)
        ok.append(("vacuum F = 1", abs(F - 1.0) < 0.02))

    for name, passed in ok:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    if not all(passed for _, passed in ok):
        return 1
    print("tiny: all controls hold")
    return 0


def ladder(args) -> int:
    """The grid/estimate table to choose a rung from, before any run."""
    print(f"{'dx':>5} {'pad':>6} {'cells':>12} {'steps':>8} {'s':>8} {'GiB':>6}"
          f" {'dlam':>6}  verdict")
    rungs = [(dx, pad) for dx in (20.0, 15.0, 12.0, 10.0)
             for pad in (400.0, 600.0)]
    asked = (float(args.dx), float(args.mirror_pad))
    extra = asked not in rungs          # the caller's own rung, if it is not one
    if extra:
        rungs.append(asked)
    held = boundary_cells(args)
    print(f"boundary  held at {held['pad_cells'] * args.dx:.0f} nm of pad and"
          f" {held['pml_cells'] * args.dx:.0f} nm of absorber at dx"
          f" {args.dx:.1f}; each rung's cell counts follow its own dx"
          f" ({held['pad_cells']}/{held['pml_cells']} cells at dx"
          f" {args.dx:.1f})\n")
    finest = None
    for dx, pad in rungs:
        sub = argparse.Namespace(**args.__dict__)
        sub.dx, sub.mirror_pad = dx, pad
        asked_row = extra and (dx, pad) == asked
        lim = run_limits(sub)        # the run path reads this same function
        try:
            plan, request, geom = planned_scene(sub)
            # the plan already carries the axial line (planned_scene places it
            # from the plan itself), so the estimate prices the probes the run
            # would record
            est = fdtd_layout.estimate(plan, request, record_frames=False)
            steps = steps_for(sub, request, est["dt"])
            request["duration_s"] = steps * est["dt"]
            est = fdtd_layout.estimate(plan, request, record_frames=False)
            # the collection arm's box, priced as the run path prices it
            phasor = collection_plan(sub, plan["nodes"], geom, request,
                                     est["dt"], est["steps"])
            if phasor is not None:
                est["bytes"] += fdtd.phasor_box_bytes(
                    phasor["box"], est["shape"], frequencies=[phasor["frequency"]],
                    steps=phasor["steps"])
            fdtd.check_limits(est, **lim)
        except fdtd.FDTDError as exc:
            print(f"{dx:5.1f} {pad:6.0f}   refused"
                  f"{' (as asked)' if asked_row else ''}: {str(exc)[:64]}")
            continue
        dlam = (sub.wavelength * NM) ** 2 / (C * est["steps"] * est["dt"]) / NM
        raised = bench_ceilings(lim)
        print(f"{dx:5.1f} {pad:6.0f} {est['cells']:12,} {est['steps']:8,}"
              f" {est['seconds']:8.1f} {est['bytes'] / 2 ** 30:6.2f}"
              f" {dlam:6.2f}  fits ("
              + ("as asked" if asked_row
                 else ("the bench ceiling, not the tier-2 clock" if raised
                       else "the caps carry the tier-2 clock"))
              + ")")
        if finest is None or dx < finest[0].dx:     # the finest spacing that fits
            finest = (sub, est)
    if finest is None:
        print("every rung above was refused")
        return 2
    sub, est = finest

    print(f"\nresolving the paper's {PAPER_FWHM_NM:.0f} nm mode needs"
          f" {sub.wavelength / PAPER_FWHM_NM:.0f} periods of record, i.e."
          f" {steps_for_4nm(est['dt'], sub.wavelength):,} steps at dx"
          f" {sub.dx:.1f} nm; the cap is {run_limits(sub)['max_steps']:,}"
          + ("  -- the tail arm does not wait for that: it reads the decay and"
             " reports the error it gets" if sub.ringdown else ""))
    return 0


def build_parser(argv=None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PARSER_ARGUMENTS(p)
    given = argv if argv is not None else sys.argv[1:]
    if "--tiny" in given:
        # the tiny scene's own numbers; anything the caller names still wins,
        # so the controls can be broken on purpose (see the tier-1 test). The
        # record stays at the script's own 32 periods: 8/4 read F = 4.61 in
        # vacuum and 8/8 read 170, i.e. the hard-started CW drive had not
        # settled (16/8 reads 1.0010, 32/8 1.0019).
        p.set_defaults(dx=20.0, diameter=60.0, height=100.0, oxide=40.0, gold=40.0,
                       mirror_pad=40.0, qd_below=60.0, pml=6)
    return p


def parse_args(argv=None):
    args = build_parser(argv).parse_args(argv)
    if args.spectrum or args.ringdown:
        args.profile = "pulse"
    if args.tail_periods is None:
        # only the ring-down arm wants a tail; the band arm's drive has to stay
        # centred in its own record, so an unasked-for tail is no tail
        args.tail_periods = DEFAULT_TAIL_PERIODS if args.ringdown else 0.0
    args.xp_name = args.xp           # the spelling, for run's own warning
    if args.xp == "cupy":            # pragma: no cover -- the Jetson arm
        import cupy as xp
    else:
        xp = np
    args.xp = xp                     # the bench idiom: xp is the module itself
    return args


def _gigabytes(text):
    """A bench flag's GB number as bytes: ``14`` is 14e9 bytes."""
    return int(round(float(text) * 1e9))


def _raised_cap(flag, product_name, unit, kind, show=None):
    """``type=`` for the bench flags that raise one of the product's caps.

    A value below the product's own is refused here, at the parser and before any
    scene is built: these flags raise a ceiling and never lower one (a shorter
    record is a smaller ``--periods``). The product's own is read from ``fdtd``
    when the flag is parsed, not when this factory is called, so a test that moves
    the constant moves the floor too -- the same late binding ``run_limits`` and
    ``fdtd.check_limits`` use. ``run_limits`` clamps with ``max()`` as well, so a
    namespace that never met this parser cannot lower a ceiling either. (Only the
    step and time ceilings have a product floor now; ``--max-cells`` and
    ``--max-bytes`` are explicit ceilings for a run and need none.) ``show``
    formats both numbers in the floor refusal, so a flag that names gigabytes
    reports gigabytes rather than the bytes its value is held in.
    """
    show = show or (lambda value: f"{value:,}")
    def parse(text):
        product = getattr(fdtd, product_name)
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} wants {unit}, got {text!r}")
        if value < product:
            raise argparse.ArgumentTypeError(
                f"{flag} {show(value)} is below the product's own {show(product)}, and the"
                f" flag may only raise a cap: a shorter record is a smaller"
                f" --periods")
        return value
    return parse


def _cap_ceiling(flag, unit, kind):
    """``type=`` for a bench flag that sets an explicit ceiling.

    Cells and bytes have no product constant to raise from any more (the device's
    budget is the default), so the flag names a ceiling and any positive value is
    a ceiling: it may sit above or below what the device offers. Non-positive is
    an argparse error here, at the parser and before any scene is built.
    """
    def parse(text):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} wants {unit}, got {text!r}")
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{flag} {text!r} is not a ceiling")
        return value
    return parse


def PARSER_ARGUMENTS(p) -> None:
    p.add_argument("--diameter", type=float, default=420.0,
                   help="nm, across what --hex says")
    p.add_argument("--height", type=float, default=1375.0, help="wire height, nm")
    p.add_argument("--oxide", type=float, default=12.0, help="SiO2 buffer, nm")
    p.add_argument("--gold", type=float, default=150.0, help="gold mirror, nm")
    p.add_argument("--gold-data", choices=("rakic", "palik"), default="rakic",
                   help="gold model: Rakić et al. 1998 (default, the earlier"
                        " runs' model) or Palik 1985 fitted at its 600-1300 nm"
                        " table points")
    p.add_argument("--qd-below", type=float, default=30.0,
                   help="dipole below the top facet, nm")
    p.add_argument("--dx", type=float, default=10.0, help="x-y cell, nm")
    p.add_argument("--mirror-pad", type=float, default=600.0,
                   help="gold beyond the wire's circumradius, nm")
    p.add_argument("--pml", type=int, default=None, metavar="N",
                   help=f"absorber cells (default {fdtd.DEFAULT_PML_CELLS}); the"
                        f" layout's own pad is"
                        f" {fdtd_layout.OPEN_PAD_CELLS} cells")
    p.add_argument("--pad-nm", type=float, default=None, metavar="NM",
                   help="open boundary's pad as a DISTANCE: the layout is padded"
                        f" by this and not by its own"
                        f" {fdtd_layout.OPEN_PAD_CELLS} cells, so a dx-10 record"
                        f" can hold r20's boundary (rounded to the nearest cell"
                        f" at this dx, and the run reports the distance it"
                        f" held)")
    p.add_argument("--pml-nm", type=float, default=None, metavar="NM",
                   help="absorber as a DISTANCE (same rounding and reporting as"
                        " --pad-nm); the answer when both it and --pml are given")
    p.add_argument("--hex", choices=("paper", "corners", "flats"),
                   default="paper",
                   help="what --diameter measures: paper (the paper's relation,"
                        " 2s = 1.14 D, so the corners are 1.14 x the quoted"
                        " number), or the quoted number across the corners or"
                        " across the flats")
    p.add_argument("--hex-rotation", type=float, default=0.0, help="degrees")
    p.add_argument("--wavelength", type=float, default=900.0, help="vacuum nm")
    p.add_argument("--band-half", type=float, default=70.0,
                   help="half the pulse's band, nm: the paper's figure spans 830-970")
    p.add_argument("--pulse-span", type=float, default=8.0,
                   help="pulse record, in sigma (the band arm needs it centred;"
                        " 10 sigma does not fit at --dx 10 -- see the docstring)")
    p.add_argument("--tail-periods", type=float, default=None,
                   help=f"drive periods of record after the pulse is over: what the"
                        f" ring-down arm reads its decay from (the product's own"
                        f" pulse is centred at 5 sigma and over by 8, so a span of"
                        f" 8 leaves almost none). Default {DEFAULT_TAIL_PERIODS:g}"
                        f" with --ringdown, 0 otherwise")
    p.add_argument("--ringdown", action="store_true",
                   help="a pulse run read as a decay: one decaying sinusoid fitted"
                        " to the tail, giving Q and its own error instead of bins")
    p.add_argument("--save-tail", metavar="PATH",
                   help="write the recorded field with its dt and drive, so its"
                        " tail can be re-fitted offline instead of by another run")
    p.add_argument("--axis-probes", action="store_true",
                   help="the ring-down also places one probe per E component on"
                        " every grid node of the wire's axis, from the gold's top"
                        " face to a few cells above the wire, and stores their"
                        " traces in the --save-tail record (it needs --ringdown"
                        " and --save-tail)")
    p.add_argument("--axial-profile", metavar="NPZ", default=None,
                   help="read a saved --axis-probes record: print |E(z)|^2 at its"
                        " drive (or --axial-frequency-nm) along the line and the"
                        " z of its interior maxima inside the wire; no scene, no"
                        " run")
    p.add_argument("--axial-frequency-nm", metavar="NM", type=float, default=None,
                   help="the wavelength to read the axial profile at; default is"
                        " the record's own drive (only with --axial-profile)")
    p.add_argument("--axial-png", metavar="PATH", default=None,
                   help="also draw the axial profile to PATH, marking the wire's"
                        " ends and the emitter's z (only with --axial-profile)")
    p.add_argument("--periods", type=float, default=32.0, help="CW record, periods")
    p.add_argument("--window-periods", type=float, default=8.0,
                   help="the CW reading's tail window, in drive periods")
    p.add_argument("--max-cells", default=None, metavar="N",
                   type=_cap_ceiling("--max-cells", "a whole number of cells",
                                     int),
                   help="set this bench run's own cell ceiling; with no flag the"
                        " device's memory budget decides, as it does for every"
                        " other run")
    p.add_argument("--max-steps", default=None, metavar="N",
                   type=_raised_cap("--max-steps", "MAX_STEPS",
                                    "a whole number of steps", int),
                   help="raise the product's step cap for this bench run; only"
                        " this bench script has the flag, so the web app and the"
                        " k8s jobs cannot reach a raised ceiling")
    p.add_argument("--max-seconds", default=None, metavar="S",
                   type=_raised_cap("--max-seconds", "MAX_SECONDS",
                                    "seconds", float),
                   help="raise the product's run-time cap, in seconds (distinct"
                        " from the launcher's own MSE_CLOCK)")
    p.add_argument("--max-bytes", default=None, metavar="GB",
                   type=_cap_ceiling("--max-bytes", "gigabytes of memory",
                                     _gigabytes),
                   help="set this bench run's own memory ceiling, in GB (e.g."
                        " 14 means 14e9 bytes); with no flag the device's budget"
                        " decides, as it does for every other run")
    p.add_argument("--axis", default="x", help="the dipole's component")
    p.add_argument("--profile", choices=("cw", "pulse"), default="cw")
    p.add_argument("--xp", choices=("numpy", "cupy"), default="numpy")
    p.add_argument("--vacuum", action="store_true",
                   help="strip every material (the F = 1 control)")
    p.add_argument("--estimate", action="store_true",
                   help="the table, no run; with --tiny, the same table at the control's"
                        " small dimensions, which is how a rung that cannot fit is seen")
    p.add_argument("--spectrum", action="store_true",
                   help="a pulse run, read bin by bin")
    p.add_argument("--tiny", action="store_true", help="the tier-1 control")
    p.add_argument("--collection-na", type=float, default=None, metavar="NA",
                   help="the CW arm also records a phasor box around the whole"
                        " scene (wire, oxide and gold pad) and prints the"
                        " fraction of the upward far-field power inside this"
                        " numerical aperture in air; off by default")
    p.add_argument("--save-series", metavar="PATH", default=None,
                   help="the CW arm also writes its drive and source-field series"
                        " (with dt, frequency, cell volume, eps_r and index) to"
                        " PATH as a compressed .npz, so the reading and its"
                        " settling curve can be re-read without a run; off by"
                        " default")
    p.add_argument("--from-series", metavar="PATH", default=None,
                   help="re-read a --save-series record: the CW reading, its"
                        " settling curve and class, with no scene and no run"
                        " (--window-periods and --locator-tau still apply)")
    p.add_argument("--locator-tau", metavar="FS[,ERR]", default=None,
                   type=parse_locator,
                   help="a ring-down locator's decay time (and error), fs: the"
                        " CW arm sets its settling time against it in plain words")


def main(argv=None) -> int:
    args = parse_args(argv)

    why = arm_conflicts(args)        # argument-level, so before any scene
    if why:
        print(why)
        return 2
    if getattr(args, "axial_profile", None) is not None:
        try:
            return axial_profile_command(args)
        except ValueError as exc:    # a record without the traces this arm reads
            print(f"refused: {exc}")
            return 2
    if getattr(args, "from_series", None) is not None:
        return from_series(args)
    if args.estimate:
        return ladder(args)          # --tiny --estimate is the table at the tiny dims
    if args.tiny:
        return tiny(args)

    lay, request, geom = solve_scene(args)
    try:
        measured = filled(args, lay, request, geom)
    except fdtd.FDTDError as exc:
        if args.collection_na is None:   # not this arm's: as it always was
            raise
        print(f"refused: {exc}")         # the collection box has no room
        return 2
    code = report(args, lay, request, geom, measured)
    if code:
        return code
    if measured.get("phasor") is not None:
        print(collection_line(measured["phasor"], geom))
    why = refusals(args, measured, lay, geom)
    if why:
        print(why)
        return 2
    if args.xp_name == "numpy":
        print("warning: --xp numpy; the estimate's seconds are fdtd_rate.py's cupy"
              " rates, so the real run takes a large factor longer")
    if args.ringdown:
        return ringdown(args, lay, request, measured)[1]
    if args.spectrum:
        measure_spectrum(args, lay, request, measured)
    else:
        measure(args, lay, request, measured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
