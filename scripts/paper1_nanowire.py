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
fetched text gives -- a wire of 1375 nm by 420 nm, a 12 nm SiO2 buffer over the
gold, and a dipole on the axis 30 nm below the top facet. NONE OF THEM IS
ASSERTED: the fetched abstract describes a quasi-BIC cavity from two coupled
modes while the fetched caption describes a gold-mirror cavity, so the lead pins
the real dimensions from the PDF and passes them here. ``--hex`` says what
--diameter measures across and ``--hex-rotation`` turns the prism about its
axis, because neither is in the fetched text.

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
read against, and it is a PULSE arm -- a pulse cannot ring a high-Q cavity up,
so a 4 nm-wide mode is measured through a window wider than the mode. The CW arm
(``--periods``) measures a settled steady state at one frequency per run.
Neither substitutes for the other; the printed resolution says which one a given
run is.

Item (7) of the handoff (a dielectric that is a function, not a scalar) does not
bind the measurement this script makes: the dipole sits on the wire's axis, 30
nm below the top facet and clear of every graded interface, so the source cell's
own medium is homogeneous InP and ``sqrt(eps[cell])`` is the host index the
normalisation needs. The field update around it is still the product's scalar
path.

Exit codes: 0 measured, 1 a control failed, 2 refused (over a cap, or over the
tier-2 envelope).

The tier-2 commands (the Jetson bench, one run at a time; the lead runs them).
The bench's python is the worker venv -- that is where cupy lives, and the
script's own imports expect the repo's paths to be kept:

    rsync -a --relative scripts/paper1_nanowire.py dockerfiles/maxwell_solver/ \
        jetson:~/mse-bench/paper1/
    ssh jetson 'cd ~/mse-bench/paper1 && ulimit -v 5000000; nice -n 10 \
        timeout 30m ~/mse-worker/venv/bin/python scripts/paper1_nanowire.py \
        --xp cupy --spectrum --dx 10 --mirror-pad 400'

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
N_SIO2 = 1.45         # thermal oxide at 900 nm
TIER2_SECONDS = 1800.0
SETTLED_PERIODS = 16.0  # measured: a hard-started CW drive's settling, in periods
PAPER_FWHM_NM = 4.0


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
    radius = args.diameter * NM / 2.0
    if args.hex == "flats":            # --diameter across flats -> circumradius
        radius /= math.cos(math.pi / 6.0)
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
    """The record this arm wants: periods at CW, a ten-sigma span for a pulse."""
    if args.profile == "pulse":
        sigma = 1.0 / (math.pi * request["bandwidth"])
        return int(math.ceil(args.pulse_span * sigma / dt))
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
          f" {args.diameter:.0f} nm across {args.hex}")
    print(f"timestep  {est['dt']:.6e} s  ({1.0 / (f0 * est['dt']):.1f} steps per"
          f" optical period)")
    print(f"record    {est['steps']:,} steps = {est['steps'] * est['dt'] * f0:.1f}"
          f" periods, {est['seconds']:.1f} s estimated on the Jetson"
          f" ({est['setup']} rates)")
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
    print(f"run       {time.monotonic() - started:.1f} s wall")
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
    read off the real scene's layout, which needs no solve.
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
    if vac.profile == "pulse":
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
    finest = None
    for dx in (20.0, 15.0, 12.0, 10.0):
        for pad in (400.0, 600.0):
            sub = argparse.Namespace(**args.__dict__)
            sub.dx, sub.mirror_pad = dx, pad
            try:
                plan, request, geom = planned_scene(sub)
                est = fdtd_layout.estimate(plan, request, record_frames=False)
                steps = steps_for(sub, request, est["dt"])
                request["duration_s"] = steps * est["dt"]
                est = fdtd_layout.estimate(plan, request, record_frames=False)
                fdtd.check_limits(est)
            except fdtd.FDTDError as exc:
                print(f"{dx:5.1f} {pad:6.0f}   refused: {str(exc)[:64]}")
                continue
            dlam = (sub.wavelength * NM) ** 2 / (C * est["steps"] * est["dt"]) / NM
            print(f"{dx:5.1f} {pad:6.0f} {est['cells']:12,} {est['steps']:8,}"
                  f" {est['seconds']:8.1f} {est['bytes'] / 2 ** 30:6.2f}"
                  f" {dlam:6.2f}  fits (the caps carry the tier-2 clock)")
            finest = (sub, est)
    if finest is None:
        print("every rung above was refused")
        return 2
    sub, est = finest           # the finest spacing that fits, not the last tried
    print(f"\nresolving the paper's {PAPER_FWHM_NM:.0f} nm mode needs"
          f" {sub.wavelength / PAPER_FWHM_NM:.0f} periods of record, i.e."
          f" {steps_for_4nm(est['dt'], sub.wavelength):,} steps at dx"
          f" {sub.dx:.1f} nm; the product's cap is {fdtd.MAX_STEPS:,}")
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
    if args.spectrum:
        args.profile = "pulse"
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
    p.add_argument("--hex", choices=("corners", "flats"), default="corners",
                   help="what --diameter measures across")
    p.add_argument("--hex-rotation", type=float, default=0.0, help="degrees")
    p.add_argument("--wavelength", type=float, default=900.0, help="vacuum nm")
    p.add_argument("--band-half", type=float, default=70.0,
                   help="half the pulse's band, nm: the paper's figure spans 830-970")
    p.add_argument("--pulse-span", type=float, default=8.0,
                   help="pulse record, in sigma (the band arm needs it centred;"
                        " 10 sigma does not fit at --dx 10 -- see the docstring)")
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
    if args.spectrum:
        measure_spectrum(args, lay, request, measured)
    else:
        measure(args, lay, request, measured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
