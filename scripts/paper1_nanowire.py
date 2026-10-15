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
import math
import sys
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
UNRESOLVED = 3          # measured, but the record's tail leaves Q unpinned
DEFAULT_TAIL_PERIODS = 20.0   # asked for only by --ringdown


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
    materials = {"mirror": "Au", "oxide": N_SIO2, "wire": N_INP}
    if args.vacuum:                    # the control: the same meshes, no material
        materials = {name: 1.0 for name in names}
    request = {
        "boundary": "open", "source_kind": "point", "dx_m": args.dx * NM,
        "position": (half, half, z_top - args.qd_below * NM),
        "probes": (), "profile": args.profile, "frequency_hz": None,
        "bandwidth": None, "polarisation": args.axis,
        "samples": None, "sample_duration_s": None,
        "duration_s": 1e-15, "coatings": None,
    }
    request["frequency_hz"] = C / (args.wavelength * NM)
    request["bandwidth"] = band_hz(args)
    geom = {"z_top": z_top, "z_oxide": (z_gold1, z_ox1), "radius": radius,
            "half": half, "pad_cells": fdtd_layout.OPEN_PAD_CELLS}
    return names, parts, materials, request, geom


def planned_scene(args):
    """The product's own plan, and its cheap estimate (no voxeliser)."""
    names, parts, materials, request, geom = scene(args)
    plan = fdtd_layout.plan(names, parts, request, length_scale=1.0,
                            materials=materials)
    return plan, request, geom


def solve_scene(args):
    """The product's own layout: grid, voxelised scene, source node, probes."""
    names, parts, materials, request, geom = scene(args)
    lay = fdtd_layout.layout(names, parts, request, length_scale=1.0,
                             materials=materials)
    return lay, request, geom


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


def filled(args, lay, request):
    """The layout's estimate, the caps' verdict, and the record's own reach."""
    est = fdtd.estimate_run(lay["nodes"], duration=request["duration_s"],
                            boundary=request["boundary"], metals=lay["metals"],
                            pec=lay["pec"])
    steps = steps_for(args, request, est["dt"])
    request["duration_s"] = steps * est["dt"]
    est = fdtd.estimate_run(lay["nodes"], duration=request["duration_s"],
                            boundary=request["boundary"], metals=lay["metals"],
                            pec=lay["pec"])
    exceeded = None
    try:
        fdtd.check_limits(est)
    except fdtd.EstimateError as exc:
        exceeded = str(exc)
    dlambda = (args.wavelength * NM) ** 2 / (C * est["steps"] * est["dt"]) / NM
    return {"est": est, "steps": est["steps"], "exceeded": exceeded,
            "dlambda_nm": dlambda}


def refusals(args, measured) -> str:
    """The refusal both entry points owe before anything runs.

    Returns the refusal line, or "" when the run may go ahead. It is one
    function on purpose: the tier-1 arm cannot pass a scene that the Jetson arm
    would refuse, which is what the ``--tiny --periods 4000`` test pins.

    The tier-2 clock needs no check of its own here: the product's caps carry
    it (``fdtd.MAX_SECONDS`` is the same 30 minutes), so a run over the timeout
    is already ``measured["exceeded"]``. That equality is itself a control --
    see ``tiny``, which fails if someone moves one clock without the other.
    """
    if measured["exceeded"]:
        return f"refused: {measured['exceeded']}"
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


def _decay_scan(t, y, taus, omegas):
    """Least squares over one decaying sinusoid plus a constant, on a grid."""
    best = None
    for tau in taus:
        env = np.exp(-t / tau)
        for omega in omegas:
            basis = np.column_stack((env * np.cos(omega * t),
                                     env * np.sin(omega * t), np.ones(t.size)))
            coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
            resid = y - basis @ coef
            rss = float(resid @ resid)
            if best is None or rss < best[0]:
                best = (rss, float(tau), float(omega), coef)
    return best


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
    so it is a grid scan, then a coordinate descent on the two of them, then a
    covariance. Nothing here reads the frequency from the DFT's bins: those are
    1/window apart, far too coarse for a mode the record's own length cannot
    resolve, and reading the decay is the way to beat them.

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
    omegas = np.linspace(lo_w, hi_w, 21)
    edge_w = (hi_w - lo_w) / (omegas.size - 1)
    rss, tau, omega, coef = _decay_scan(t, y, taus, omegas)
    at_edge = math.isclose(tau, float(taus[-1]), rel_tol=1e-9)
    # The scan's grid is coarse (a quarter of an octave in tau), far too coarse
    # to stop on: descend each axis in turn, which is exact to the search's own
    # tolerance rather than to a grid step. Measured against synthetic tails, a
    # fixed number of linspace rounds left tau on the grid -- 0.4% to 2.6% off at
    # a hundred times its own error -- where this recovers it to the noise.
    step_tau = (taus[-1] - taus[0]) / (taus.size - 1)
    step_w = (omegas[-1] - omegas[0]) / (omegas.size - 1)
    for sweep in range(3):
        tau = _golden(lambda v: rss_at(v, omega)[0],
                      max(tau - 1.5 * step_tau, 1e-3 * window), tau + 1.5 * step_tau)
        omega = _golden(lambda v: rss_at(tau, v)[0],
                        omega - 1.5 * step_w, omega + 1.5 * step_w)
        if sweep == 0:                 # the scan's step is used up: tighten
            step_tau, step_w = step_tau * 0.05, step_w * 0.05
    rss, coef = rss_at(tau, omega)
    f_edge = bool(omega <= omegas[0] + edge_w or omega >= omegas[-1] - edge_w)

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
    tau_err = math.sqrt(max(cov[3, 3], 0.0))
    f_err = math.sqrt(max(cov[4, 4], 0.0)) / (2.0 * math.pi)
    f = omega / (2.0 * math.pi)
    q = math.pi * f * tau
    q_err = math.pi * math.hypot(tau * f_err, f * tau_err)
    return {"f": f, "f_err": f_err, "tau": tau, "tau_err": tau_err,
            "q": q, "q_err": q_err, "samples": n, "window": window,
            "rms": math.sqrt(rss / n), "peak": float(np.max(np.abs(y))),
            "at_edge": at_edge, "f_edge": f_edge,
            "resolved": bool(tau > 3.0 * tau_err and not at_edge)}


def ringdown(args, lay, request, measured):
    """The tail arm: one pulse record, its decay, and the Q it does or does not pin.

    Returns (fit, code). The code is 0 when the fit is significant, 3 when the
    record's own tail leaves Q unpinned -- not a failure to hide, and not a
    measurement to invent: the fit's own error is printed either way.
    """
    sim, volume, index, eps_cell = _run(args, lay, request, measured)
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
    print(f"fit       f = {fit['f'] / 1e12:.4f} THz +- {fit['f_err'] / 1e12:.2g}"
          f" (the drive is {f0 / 1e12:.2f}, searched over"
          f" {C / (f0 + half) / NM:.0f}-{C / (f0 - half) / NM:.0f} nm){mark},"
          f" tau = {fit['tau'] / 1e-15:.3f} fs +- {fit['tau_err'] / 1e-15:.2g},"
          f" rms {fit['rms']:.3e} against a {fit['peak']:.3e} peak")
    print(f"caveat    tau's error is the fit's own, not the geometry's: a tail"
          f" holding two decays still fits one of them. --vacuum is the same arm"
          f" on the transient this scene has with no cavity at all")
    if fit["resolved"]:
        print(f"Q         {fit['q']:.4g} +- {fit['q_err']:.2g}, a line of"
              f" {args.wavelength / fit['q']:.2f} nm against the paper's"
              f" {PAPER_FWHM_NM:.0f} nm -- read from the decay, not from this"
              f" record's {measured['dlambda_nm']:.2f} nm bins")
        return fit, 0
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
    if args.pml >= geom["pad_cells"]:
        print(f"refused: --pml {args.pml} is not inside the layout's "
              f"{geom['pad_cells']}-cell pad; the parts would be in the absorber")
        return 2
    z = np.asarray(lay["nodes"][2])
    problems = fdtd_grid.check_axis(z, args.dx * NM)
    print(f"grid      {est['shape'][0]} x {est['shape'][1]} x {est['shape'][2]}"
          f"  = {est['cells']:,} cells")
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
    depth = fdtd_grid.skin_depth("Au", f0) / NM
    print(f"stack     gold {args.gold:.0f} nm = {args.gold / depth:.1f} skin depths"
          f" (skin depth {depth:.1f} nm at {f0 / 1e12:.1f} THz), SiO2"
          f" {args.oxide:.0f} nm, InP {args.height:.0f} nm on a"
          f" {2 * geom['half'] / NM:.0f} nm mirror; the wire is"
          f" {2 * geom['radius'] / NM:.1f} nm across corners,"
          f" {2 * geom['radius'] * math.cos(math.pi / 6.0) / NM:.1f} nm across flats"
          f" (--diameter {args.diameter:.0f} nm, --hex {args.hex})")
    cell = lay["cell"]
    x, y = np.asarray(lay["nodes"][0]), np.asarray(lay["nodes"][1])
    print(f"dipole    node ({cell[0]}, {cell[1]}, {cell[2]}): z"
          f" {z[cell[2]] / NM:.2f} nm, {(geom['z_top'] - z[cell[2]]) / NM:.2f} nm"
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
    print(f"resolves  {measured['dlambda_nm']:.2f} nm per bin at"
          f" {args.wavelength:.0f} nm; the paper's mode is {PAPER_FWHM_NM:.0f} nm"
          f" and would need {steps_for_4nm(est['dt'], args.wavelength):,} steps"
          f" (the cap is {fdtd.MAX_STEPS:,})")
    print(f"caps      {measured['exceeded'] or 'OK'}")
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
    index = math.sqrt(eps_cell)
    print(f"source    node {tuple(int(c) for c in cell)}, eps_r {eps_cell:.4f}"
          f" -> host index {index:.4f}, cell volume {volume:.6e} m^3")
    return source, terms, volume, index, eps_cell


def _run(args, lay, request, measured):
    source, terms, volume, index, eps_cell = source_and_terms(args, lay, request)
    started = time.monotonic()
    sim = fdtd.simulate(lay["eps"], source, measured["est"]["steps"],
                        lay["probes"], spacings=lay["spacings"],
                        boundary=request["boundary"], pec=lay["pec"],
                        pml_cells=args.pml, ade_terms=terms, xp=args.xp,
                        record_drive=True, record_source_field=True)
    wall = time.monotonic() - started
    estimate = measured["est"]["seconds"]
    # The run reports its own prediction ratio, so a logged run carries the
    # evidence that the table's rate does or does not hold on this machine. It
    # is printed, not folded into a new constant: one contended run cannot set
    # a rate that the next machine has to live with.
    print(f"run       {wall:.1f} s wall  ({wall / estimate:.2f} x the"
          f" {estimate:.1f} s estimate)")
    return sim, volume, index, eps_cell


def measure(args, lay, request, measured):
    """The CW arm: one frequency, a settled tail, F = P/P0."""
    sim, volume, index, eps_cell = _run(args, lay, request, measured)
    dt = measured["est"]["dt"]
    power = fdtd.delivered_power(sim["drive"], sim["source_field"], dt,
                                 volume=volume, eps_r=eps_cell, periods=args.window_periods,
                                 frequency=request["frequency_hz"])
    moment = fdtd.dipole_moment(sim["drive"], dt, volume=volume, eps_r=eps_cell,
                                periods=args.window_periods,
                                frequency=request["frequency_hz"])
    reference = fdtd.larmor_power(moment, request["frequency_hz"], index=index)
    F = fdtd.purcell_factor(power, reference)
    print(f"measured  F = {F:.4f} at {args.wavelength:.0f} nm  (P = {power:.6e} W,"
          f" P0 = {reference:.6e} W)")
    return F


def measure_spectrum(args, lay, request, measured):
    """The band arm: a pulse centred in its own record, read bin by bin."""
    sim, volume, index, eps_cell = _run(args, lay, request, measured)
    half = request["bandwidth"]
    f, power, reference, F = fdtd.purcell_spectrum(
        sim["drive"], sim["source_field"], measured["est"]["dt"], volume=volume,
        eps_r=eps_cell, index=index,
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
    why = refusals(real, measured)
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
    t_syn = np.arange(max(400, int(12.0 / (dt * f_c)))) * dt
    f_off = 0.90 * f_c
    syn = np.exp(-t_syn / (6.0 / f_off)) * np.cos(2.0 * math.pi * f_off * t_syn)
    wide = fit_decay(syn, dt, 0, frequency=f_c, band=(0.85 * f_c, 1.15 * f_c))
    narrow = fit_decay(syn, dt, 0, frequency=f_c)
    ok.append((f"the fit searches the band it is given, not a fixed +-5 per cent"
               f" ({wide['f'] / 1e12:.2f} THz against {f_off / 1e12:.2f} sought,"
               f" where the narrow window stops at {narrow['f'] / 1e12:.2f} and"
               f" flags itself at {narrow['f_edge']})",
               abs(wide["f"] - f_off) < 0.01 * f_off
               and not wide["f_edge"]
               and abs(narrow["f"] - f_off) > 0.03 * f_off
               and narrow["f_edge"] is True))

    as_tail = argparse.Namespace(**{**base, "profile": "pulse", "ringdown": True,
                                    "spectrum": True, "tail_periods": 20.0})
    as_band = argparse.Namespace(**{**base, "profile": "pulse", "ringdown": False,
                                    "spectrum": True, "tail_periods": 0.0})
    ok.append(("the band arm and the tail arm refuse to share one record"
               f" ({arm_conflicts(as_tail)[:40] or 'NOTHING'}), and the band arm"
               f" alone is clean",
               bool(arm_conflicts(as_band)) is False
               and bool(arm_conflicts(as_tail)) is True))

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
    gold = lay["metals"].get("Au")
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

    vac = argparse.Namespace(**{**base, "vacuum": True})
    lay_v, request_v, geom_v = solve_scene(vac)
    measured_v = filled(vac, lay_v, request_v)
    if vac.ringdown:
        fit, code = ringdown(vac, lay_v, request_v, measured_v)
        ok.append((f"the tail arm runs on a homogeneous scene and answers either"
                   f" way (tau {fit['tau'] / 1e-15:.3f} fs +-"
                   f" {fit['tau_err'] / 1e-15:.2g},"
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
    finest = None
    for dx, pad in rungs:
        sub = argparse.Namespace(**args.__dict__)
        sub.dx, sub.mirror_pad = dx, pad
        asked_row = extra and (dx, pad) == asked
        try:
            plan, request, geom = planned_scene(sub)
            est = fdtd_layout.estimate(plan, request, record_frames=False)
            steps = steps_for(sub, request, est["dt"])
            request["duration_s"] = steps * est["dt"]
            est = fdtd_layout.estimate(plan, request, record_frames=False)
            fdtd.check_limits(est)
        except fdtd.FDTDError as exc:
            print(f"{dx:5.1f} {pad:6.0f}   refused"
                  f"{' (as asked)' if asked_row else ''}: {str(exc)[:64]}")
            continue
        dlam = (sub.wavelength * NM) ** 2 / (C * est["steps"] * est["dt"]) / NM
        print(f"{dx:5.1f} {pad:6.0f} {est['cells']:12,} {est['steps']:8,}"
              f" {est['seconds']:8.1f} {est['bytes'] / 2 ** 30:6.2f}"
              f" {dlam:6.2f}  fits ("
              + ("as asked" if asked_row else "the caps carry the tier-2 clock")
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
          f" {sub.dx:.1f} nm; the product's cap is {fdtd.MAX_STEPS:,}"
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


def PARSER_ARGUMENTS(p) -> None:
    p.add_argument("--diameter", type=float, default=420.0,
                   help="nm, across what --hex says")
    p.add_argument("--height", type=float, default=1375.0, help="wire height, nm")
    p.add_argument("--oxide", type=float, default=12.0, help="SiO2 buffer, nm")
    p.add_argument("--gold", type=float, default=150.0, help="gold mirror, nm")
    p.add_argument("--qd-below", type=float, default=30.0,
                   help="dipole below the top facet, nm")
    p.add_argument("--dx", type=float, default=10.0, help="x-y cell, nm")
    p.add_argument("--mirror-pad", type=float, default=600.0,
                   help="gold beyond the wire's circumradius, nm")
    p.add_argument("--pml", type=int, default=fdtd.DEFAULT_PML_CELLS,
                   help=f"absorber cells (the layout's pad is"
                        f" {fdtd_layout.OPEN_PAD_CELLS})")
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
    p.add_argument("--periods", type=float, default=32.0, help="CW record, periods")
    p.add_argument("--window-periods", type=float, default=8.0,
                   help="the CW reading's tail window, in drive periods")
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


def main(argv=None) -> int:
    args = parse_args(argv)

    why = arm_conflicts(args)        # argument-level, so before any scene
    if why:
        print(why)
        return 2
    if args.estimate:
        return ladder(args)          # --tiny --estimate is the table at the tiny dims
    if args.tiny:
        return tiny(args)

    lay, request, geom = solve_scene(args)
    measured = filled(args, lay, request)
    code = report(args, lay, request, geom, measured)
    if code:
        return code
    why = refusals(args, measured)
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
