#!/usr/bin/env python3
"""Bench: the quantum-dot Stark shift, and what the well depth does to it (1.9).

The model is paper 1's InAsP dot in an InP nanowire: an electron and a heavy
hole in a finite square well along z, tilted by ``-e F z``, with a first-order
Coulomb term. The exciton energy is ``E_X(F) = E_e(F) + E_h(F) + E_C`` and the
Stark form the paper quotes is ``dE = -p F + beta F^2``.

What this script is FOR
-----------------------
It answers one question and refuses to hide the answer: **how much does the
reported ``beta`` depend on the confinement depth, and is the depth known?**

It is not. The dot's alloy band gap was never retrieved (no bowing parameter;
see :mod:`dot_material`), so the depth of the well is not a cited number. This
bench therefore SWEEPS it and prints ``beta`` at each depth alongside ``<z^2>``
of each carrier, which is the diagnostic that says whether the carrier is
confined at all. ``<z^2>`` is the honest test: a bare box has
``<z^2> = L^2 (1/12 - 1/(2 pi^2))``, a confined state is far narrower, and a
"dot" whose ``<z^2>`` equals the bare-box value is a free particle wearing the
word dot.

The output is deliberately not a single beta. Reporting one would require
four numbers nobody has measured: the well DEPTH, the well HEIGHT (4 nm is
assumed; the cited TEM range is 3-7 nm), the alloy masses (linear
interpolation, ASSUMED) and ``eps_s`` (the InP host's, ASSUMED for an InAsP
dot). ``beta`` is a property of all four, so a single value needs four
citations, not one.

Reading the table
-----------------
``<z2>e`` / ``<z2>h`` fall as the well deepens. Two very different ceilings
bracket the reading, and it is worth keeping them apart:

* the BARE-BOX value (printed above the table, ``8.364`` here) is the
  ceiling for this *box*. A carrier sitting at it is a FREE particle; the
  only level an unconfined carrier can occupy is that value, so proximity to
  it means "not a dot", never "confined".
* the TIGHT-WELL value ``h^2/12`` (``1.333`` nm^2 for a 4 nm well) is the
  ceiling for a carrier confined to the cited dot height. It depends on the
  well, not the box, which is why it is the criterion: confinement is a
  length.

So a carrier counts as confined when its ``<z^2>`` is below ``h^2/12``. Rows
where the ELECTRON is above that are electron-limited: the light carrier is
not held by the well, so the exciton's extent is set by the box and those
rows are not a dot's either, whatever beta does.

``beta`` still changes sign over the swept depths, which is the result. It is
not decoratable into a percentage: the swing is dominated entirely by the
deepest row.

Lateral confinement (the 1.9 follow-on)
---------------------------------------
By default the well fills the box in x and y, as in 1.9: the only lateral
confinement is the hard box wall. ``--lateral`` gives the dot its own lateral
size, a square box dot with the InP host as the barrier around it as well as
above and below. With no value it takes the equal-AREA square of the cited
18 nm diameter (``dot_material.DOT_DIAMETER_NM``), side ``D sqrt(pi) / 2`` =
15.95 nm, the same equal-area rule ``design.md`` uses for hexagon -> cylinder.
``--transverse`` must then exceed it, with room for the barrier tails.

A square, not a cylinder, because the solver is ``stark.py``'s uniform
Cartesian 7-point grid: a box dot's faces lie on grid planes, so its
cell-averaged step converges at second order against exact answers, while a
cylinder there would be a staircase. ``design.md`` section 1 fixes L3 as that
Cartesian 3-D grid (the hexagonal wire is not axisymmetric).

``--mask`` picks how the step is put on the grid (``exciton.WELL_MASKS``).
``node`` is 1.9's rule and its default; it makes the well up to one cell
wider than asked, an O(h) error. ``cell`` is the cell-averaged step, and is
the default when ``--lateral`` is given.

The second table, "electron binding", reports three tests side by side and
does not merge them: ``bound`` (``E_e`` below the continuum edge -- ``V0``
with a lateral dot, since hard outer walls only raise ``E``; ``V0`` plus the
box column's lateral energy without one), ``P_in`` (probability inside the dot), and the 1.9 length
criterion ``<z^2>e < h^2/12`` together with its lateral analogue
``<x^2>e < s^2/12``. The verdict lines that follow use the 1.9 criterion,
unchanged.

Cost and where it runs
----------------------
Tiny grids only (the default is a few thousand unknowns), so this is a tier-1
run: ``(ulimit -v 4194304; nice -n 19 python -I scripts/stark_bench.py)``.
It never builds a mesh and never solves a FEM problem.

Usage
-----
    python scripts/stark_bench.py                 # the default sweep
    python scripts/stark_bench.py --depths 0.1 0.2 0.3
    python scripts/stark_bench.py --height 7.0    # the top of the TEM range
    python scripts/stark_bench.py --lateral --transverse 32 --n-xy 16
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "dockerfiles" / "maxwell_solver"))

import dot_material as DM  # noqa: E402
import exciton as X  # noqa: E402
import stark as S  # noqa: E402

# The TEM range of dot heights, Haffouz et al. Nano Lett. 18, 3047 (2018):
# 3-7 nm at the same 18 nm diameter. 4 nm is the Cygorek et al. value.
DEFAULT_DEPTHS = (0.0, 0.10, 0.20, 0.30, 0.50, 0.80)
DEFAULT_FIELDS = (0.0, 0.1, 0.2, 0.4)
DEFAULT_HEIGHT_NM = 4.0

#: The equal-area square of the cited dot diameter: ``D sqrt(pi) / 2``. Derived
#: from ``dot_material``, never typed: the diameter is that module's number.
EQUAL_AREA_SIDE_NM = DM.DOT_DIAMETER_NM.value * math.sqrt(math.pi) / 2.0


def _material() -> X.DotMaterial:
    """The paper-1 dot's masses at x = 0.20, with the ASSUMED interpolation noted."""
    m_e, m_h = DM.paper1_dot_at()
    return X.DotMaterial(
        eps_s=float(DM.INP_EPS_S.value),
        m_e_ratio=float(m_e.value),
        m_h_ratio=float(m_h.value),
        source=(
            "ASSUMED linear interpolation between the InP and InAs cited "
            "endpoints; eps_s from the InP host (Ioffe/NSM InP/basic.html)"
        ),
    )


def _confinement(box, mass_ratio, depth, height_nm, lateral_nm=None, mask="node"):
    """Zero-field ground state of one carrier: energy, <z^2>, <x^2>, P_in.

    ``<x^2>`` is about the box's lateral centre; ``P_in`` weights each node by
    the dot fraction it carries (``exciton.dot_fill_fraction``, same mask as
    the potential), so it is the probability inside the dot. At depth 0 there
    is no dot and ``P_in`` is NaN.
    """
    xs, ys, zs = box.axes()
    Xg, _Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    well = depth if depth > 0 else None
    potential = X._potential_grid(box, 0.0, well, height_nm,
                                  lateral_nm if well else None, mask)
    evals, vec = S.lowest_states(box, mass_ratio, n_states=1,
                                 potential_eV=potential)
    w = vec[:, 0] ** 2
    w = w / w.sum()
    x_mid = 0.5 * (xs[0] + xs[-1])
    p_in = float("nan")
    if well:
        fill = X.dot_fill_fraction(box, height_nm, lateral_nm, mask)
        p_in = float(np.dot(fill.reshape(-1), w))
    return {
        "energy": float(evals[0]),
        "z2": float(np.dot(Z.reshape(-1) ** 2, w)),
        "x2": float(np.dot((Xg.reshape(-1) - x_mid) ** 2, w)),
        "p_in": p_in,
    }


def _z_squared(box, mass_ratio, depth, height_nm, lateral_nm=None, mask="node"):
    """<z^2> of the ground state at zero field, in nm^2."""
    return _confinement(box, mass_ratio, depth, height_nm, lateral_nm, mask)["z2"]


def sweep(depths, fields, height_nm, transverse_nm, z_extent_nm, n_xy, n_z,
          lateral_nm=None, mask="node"):
    """Run the depth sweep and return rows of (depth, E_X(0), beta, z2e, z2h, shifts).

    ``lateral_nm`` (None = the well fills the box laterally, as in 1.9) and
    ``mask`` are passed to :func:`exciton.exciton_energy_eV` unchanged.
    """
    material = _material()
    box = S.Box((0.0, 0.0, -z_extent_nm / 2.0),
                (transverse_nm, transverse_nm, z_extent_nm),
                (n_xy, n_xy, n_z))
    rows = []
    for depth in depths:
        well = depth if depth > 0 else None
        results = [
            X.exciton_energy_eV(material, box, f, well_depth_eV=well,
                                well_height_nm=height_nm,
                                well_lateral_nm=lateral_nm if well else None,
                                well_mask=mask)
            for f in fields
        ]
        e0 = results[0].e_x_eV
        try:
            _p, beta, resid, _design = X.fit_p_beta(results)
        except X.ExcitonError:
            beta, resid = float("nan"), float("nan")
        ce = _confinement(box, material.m_e_ratio, depth, height_nm, lateral_nm, mask)
        ch = _confinement(box, material.m_h_ratio, depth, height_nm, lateral_nm, mask)
        z2e, z2h = ce["z2"], ch["z2"]
        rows.append({
            "depth": depth,
            "e_x_0": e0,
            "beta": beta,
            "resid": resid,
            "z2e": z2e,
            "z2h": z2h,
            "shifts": [r.e_x_eV - e0 for r in results[1:]],
            "e_e0": ce["energy"],
            "x2e": ce["x2"],
            "p_in_e": ce["p_in"],
        })
    return rows, box


def report(rows, fields, box, height_nm, lateral_nm=None, mask="node"):
    """Print the table, then state plainly what it says about beta."""
    extent = box.size[2]
    bare = extent**2 * (1.0 / 12.0 - 1.0 / (2.0 * np.pi**2))
    print()
    print(f"paper-1 dot: x = {DM.DOT_X.value:g} (Cygorek 2020 Sec. V.1), "
          f"well height {height_nm:g} nm")
    print("  height sources: 4 nm is Cygorek 2020's quoted value; Haffouz 2018 "
          "(TEM) measured 3-7 nm at the same 18 nm diameter")
    print(f"box: {box.size[0]:g} x {box.size[1]:g} x {extent:g} nm, "
          f"grid {box.n[0]}x{box.n[1]}x{box.n[2]} cells")
    print(f"bare-box <z^2> for this extent, the ceiling: {bare:.3f} nm^2")
    if lateral_nm is None:
        print("lateral: NONE -- the well fills the box in x and y (the 1.9 "
              "model); the only lateral confinement is the hard box wall")
    else:
        print(f"lateral: box dot {lateral_nm:.3f} x {lateral_nm:.3f} nm, InP "
              f"barrier around it at the same depth (cited diameter "
              f"{DM.DOT_DIAMETER_NM.value:g} nm; equal-area square side "
              f"{EQUAL_AREA_SIDE_NM:.3f} nm)")
    print(f"step on the grid: '{mask}' mask (exciton.WELL_MASKS)")
    print()
    # Fields are eV/nm throughout the model; report them in kV/cm too, because
    # a reader needs to know whether these are laboratory fields and whether
    # the shift is small enough for the quadratic form to mean anything.
    # 1 eV/nm = 1e4 kV/cm (EV_PER_NM_TO_KV_PER_CM in exciton.py).
    print(f"fields (eV/nm -> kV/cm): "
          + ", ".join(f"{f:g} -> {f * X.EV_PER_NM_TO_KV_PER_CM:,.0f}"
                      for f in fields))
    # A beta from a row where the ELECTRON is still loose is not a dot's beta:
    # the well does not hold the carrier that sets the exciton extent, so the
    # number describes the box. The caveat has to sit IN THE CELL, because the
    # table is what gets screenshotted and the number lifted from; prose below
    # the table travels without it.
    tight = height_nm**2 / 12.0
    side = lateral_nm if lateral_nm is not None else box.size[0]
    lat_tight = side**2 / 12.0
    head = f"{'depth':>6} {'E_X(0)':>9} {'<z2>e':>7} {'<z2>h':>7} {'beta':>9}"
    head += "".join(f" {'dE(' + f'{f:g}' + ')':>11}" for f in fields[1:])
    print(head)
    flagged = []
    for r in rows:
        loose = r["z2e"] > tight
        cell = "  blank" if loose else f"{r['beta']:9.3f}"
        if loose:
            cell = f"{cell:>8}*"
            flagged.append(r["depth"])
        line = (f"{r['depth']:6.2f} {r['e_x_0']:9.5f} {r['z2e']:7.3f} "
                f"{r['z2h']:7.3f} {cell}")
        line += "".join(f" {s:11.3e}" for s in r["shifts"])
        print(line)
    print()
    if flagged:
        print(f"  * electron-limited: <z2>e is above the tight-well ceiling "
              f"{tight:.3f} nm^2, so the well does not hold the carrier that "
              f"sets the exciton extent. NO beta is reported for depth(s) "
              f"{', '.join(f'{d:g}' for d in flagged)}: any value there is the "
              f"box's, not the dot's.")
        print()
    biggest = max(abs(s) for r in rows for s in r["shifts"])
    tightest = min(r["e_x_0"] for r in rows)
    if biggest > abs(tightest):
        print(f"WARNING: the largest swept shift ({biggest:.3g} eV) exceeds the "
              f"smallest E_X(0) ({tightest:.3g} eV). At those fields both "
              f"carriers are being driven up the ramp past the well, which is "
              f"outside the regime the perturbation-theory fit assumes. The "
              f"quadratic p/beta form is being extrapolated there, not "
              f"measured. Treat large-field rows as a model probe.")
        print()

    # The binding table. Three different tests, printed apart on purpose:
    # merging them into one "bound" word is how a criterion gets shopped for.
    # The continuum edge the electron must sit below to be a bound state.
    # With a lateral dot it is V0: outside the dot the potential is V0 in an
    # unbounded host, and the hard outer box only RAISES E, so E < V0 in the
    # box proves a bound state exists. Without one (1.9) the box walls are part
    # of the model -- an infinite column of the box's cross-section -- so the
    # edge is V0 plus that column's lateral kinetic energy.
    edge_extra = 0.0
    if lateral_nm is None:
        edge_extra = S.hard_box_ground_eV(box.size[:2], _material().m_e_ratio)
    print("electron binding (zero field):")
    print(f"{'depth':>6} {'E_e':>8} {'edge':>8} {'bound':>5} {'P_in':>6} "
          f"{'<z2>e':>7} {'z:held':>6} {'<x2>e':>7} {'x:held':>6}")
    for r in rows:
        if r["depth"] <= 0:
            continue
        edge = r["depth"] + edge_extra
        r["bound_e"] = r["e_e0"] < edge
        print(f"{r['depth']:6.2f} {r['e_e0']:8.4f} {edge:8.4f} "
              f"{'yes' if r['bound_e'] else 'NO':>5} "
              f"{r['p_in_e']:6.3f} {r['z2e']:7.3f} "
              f"{'yes' if r['z2e'] <= tight else 'NO':>6} {r['x2e']:7.3f} "
              f"{'yes' if r['x2e'] <= lat_tight else 'NO':>6}")
    edge_txt = ("V0" if lateral_nm is not None else
                f"V0 + {edge_extra:.4f} eV (the box column's lateral energy)")
    print(f"  bound: E_e below the continuum edge = {edge_txt}, so the "
          f"electron is in a bound state of the dot. z:held: "
          f"<z2>e <= h^2/12 = {tight:.3f} nm^2 (the 1.9 criterion). x:held: "
          f"<x2>e <= s^2/12 = {lat_tight:.3f} nm^2 for lateral side "
          f"{side:.3f} nm" + (" (the BOX, no lateral dot)" if lateral_nm is None else ""))
    print()

    # Confinement is a LENGTH: the carrier is held by the well when its
    # <z^2> is below the tight-well ceiling h^2/12 (h = the well height),
    # not when it is below some fraction of the BOX's bare value. At a
    # fraction-of-box criterion the depth-0 row -- a free particle sitting
    # exactly at the box ceiling -- reads as "confined", which is backwards.
    # (``tight`` was computed above for the table's asterisk key; reuse it.)
    print(f"tight-well ceiling h^2/12 for a {height_nm:g} nm well: "
          f"{tight:.3f} nm^2; a carrier is confined when its <z^2> is below it")

    finite = [r for r in rows if np.isfinite(r["beta"])]
    if len(finite) >= 2:
        betas = [r["beta"] for r in finite]
        lo, hi = min(betas), max(betas)
        print(f"beta moves from {lo:.3f} to {hi:.3f} over the swept depths "
              f"(computed at every depth; only the unflagged rows above are "
              f"quotable, because the flagged ones have no confined electron).")
        if lo < 0.0 < hi:
            print("It changes SIGN, so beta is not even of one sign over the "
                  "range of depths a reader could reasonably choose.")

    for label, key in (("HOLE", "z2h"), ("ELECTRON", "z2e")):
        loose = [r["depth"] for r in rows if r[key] > tight]
        held = [r["depth"] for r in rows if r[key] <= tight]
        if loose:
            print(f"The {label} is NOT confined (its <z^2> is above "
                  f"{tight:.3f} nm^2) at depth(s) "
                  f"{', '.join(f'{d:g}' for d in loose)} eV.")
        if held:
            print(f"The {label} is confined at depth(s) "
                  f"{', '.join(f'{d:g}' for d in held)} eV.")
    electron_limited = [r["depth"] for r in rows
                        if r["z2e"] > tight and r["z2h"] <= tight]
    if electron_limited:
        print(f"At depth(s) {', '.join(f'{d:g}' for d in electron_limited)} eV "
              f"the hole is held but the ELECTRON is not: those rows are "
              f"electron-limited, so the exciton extent is set by the box and "
              f"they are not a dot's either, whatever beta does.")
    print()
    print("What this means for the paper's p/beta form: the fit is well-defined "
          "at every depth, but its value is a property of the ASSUMED depth, "
          "height, alloy masses and eps_s, none of which is a cited number for "
          "this dot. A single beta cannot be quoted until all four are. Until "
          "then, report the sweep.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS),
                        help="well depths in eV to sweep (0 means no well)")
    parser.add_argument("--fields", type=float, nargs="+", default=list(DEFAULT_FIELDS),
                        help="fields in eV/nm; must include 0 to anchor the shift")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_NM,
                        help="well height along z in nm (the dot height)")
    parser.add_argument("--transverse", type=float, default=12.0,
                        help="transverse box extent in nm")
    parser.add_argument("--z-extent", type=float, default=16.0,
                        help="box extent along z in nm; must exceed --height")
    parser.add_argument("--n-xy", type=int, default=6, help="transverse cells")
    parser.add_argument("--n-z", type=int, default=40, help="cells along z")
    parser.add_argument("--lateral", type=float, nargs="?", const=EQUAL_AREA_SIDE_NM,
                        default=None,
                        help="lateral side of a square box dot in nm; with no "
                             "value, the equal-area square of the cited "
                             f"{DM.DOT_DIAMETER_NM.value:g} nm diameter "
                             f"({EQUAL_AREA_SIDE_NM:.3f} nm). Omit for the 1.9 "
                             "well that fills the box laterally")
    parser.add_argument("--mask", choices=X.WELL_MASKS, default=None,
                        help="how the step sits on the grid: 'node' (1.9) or "
                             "'cell' (cell-averaged, O(h^2)); default 'node' "
                             "without --lateral and 'cell' with it")
    args = parser.parse_args(argv)
    mask = args.mask or ("node" if args.lateral is None else "cell")

    if 0.0 not in args.fields:
        parser.error("--fields must include 0.0 to anchor the Stark shift")
    if args.height >= args.z_extent:
        parser.error(
            f"well height {args.height} nm must be strictly less than the box "
            f"extent {args.z_extent} nm; with the height equal to the box the "
            f"barrier lands on the Dirichlet walls where its value is unused, "
            f"and the solve silently becomes the INFINITE-well problem"
        )

    if args.lateral is not None and args.lateral >= args.transverse:
        parser.error(
            f"lateral dot side {args.lateral:g} nm must be strictly less than "
            f"the transverse box {args.transverse:g} nm, with room for the "
            f"barrier tails (e.g. --transverse 32 --n-xy 16); at or past the "
            f"box the lateral barrier lands on the Dirichlet walls")

    rows, box = sweep(args.depths, args.fields, args.height, args.transverse,
                      args.z_extent, args.n_xy, args.n_z, args.lateral, mask)
    report(rows, args.fields, box, args.height, args.lateral, mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
