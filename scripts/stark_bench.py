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
"""

from __future__ import annotations

import argparse
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


def _z_squared(box, mass_ratio, depth, height_nm):
    """<z^2> of the ground state at zero field, in nm^2."""
    xs, ys, zs = box.axes()
    Z = np.meshgrid(xs, ys, zs, indexing="ij")[2]
    potential = X._potential_grid(box, 0.0, depth if depth > 0 else None,
                                  height_nm)
    _evals, vec = S.lowest_states(box, mass_ratio, n_states=1,
                                  potential_eV=potential)
    w = vec[:, 0] ** 2
    w = w / w.sum()
    return float(np.dot(Z.reshape(-1) ** 2, w))


def sweep(depths, fields, height_nm, transverse_nm, z_extent_nm, n_xy, n_z):
    """Run the depth sweep and return rows of (depth, E_X(0), beta, z2e, z2h, shifts)."""
    material = _material()
    box = S.Box((0.0, 0.0, -z_extent_nm / 2.0),
                (transverse_nm, transverse_nm, z_extent_nm),
                (n_xy, n_xy, n_z))
    rows = []
    for depth in depths:
        well = depth if depth > 0 else None
        results = [
            X.exciton_energy_eV(material, box, f, well_depth_eV=well,
                                well_height_nm=height_nm)
            for f in fields
        ]
        e0 = results[0].e_x_eV
        try:
            _p, beta, resid, _design = X.fit_p_beta(results)
        except X.ExcitonError:
            beta, resid = float("nan"), float("nan")
        z2e = _z_squared(box, material.m_e_ratio, depth, height_nm)
        z2h = _z_squared(box, material.m_h_ratio, depth, height_nm)
        rows.append({
            "depth": depth,
            "e_x_0": e0,
            "beta": beta,
            "resid": resid,
            "z2e": z2e,
            "z2h": z2h,
            "shifts": [r.e_x_eV - e0 for r in results[1:]],
        })
    return rows, box


def report(rows, fields, box, height_nm):
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
    print()
    # Fields are eV/nm throughout the model; report them in kV/cm too, because
    # a reader needs to know whether these are laboratory fields and whether
    # the shift is small enough for the quadratic form to mean anything.
    # 1 eV/nm = 1e4 kV/cm (EV_PER_NM_TO_KV_PER_CM in exciton.py).
    print(f"fields (eV/nm -> kV/cm): "
          + ", ".join(f"{f:g} -> {f * X.EV_PER_NM_TO_KV_PER_CM:,.0f}"
                      for f in fields))
    head = f"{'depth':>6} {'E_X(0)':>9} {'<z2>e':>7} {'<z2>h':>7} {'beta':>9}"
    head += "".join(f" {'dE(' + f'{f:g}' + ')':>11}" for f in fields[1:])
    print(head)
    for r in rows:
        line = (f"{r['depth']:6.2f} {r['e_x_0']:9.5f} {r['z2e']:7.3f} "
                f"{r['z2h']:7.3f} {r['beta']:9.3f}")
        line += "".join(f" {s:11.3e}" for s in r["shifts"])
        print(line)
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

    # Confinement is a LENGTH: the carrier is held by the well when its
    # <z^2> is below the tight-well ceiling h^2/12 (h = the well height),
    # not when it is below some fraction of the BOX's bare value. At a
    # fraction-of-box criterion the depth-0 row -- a free particle sitting
    # exactly at the box ceiling -- reads as "confined", which is backwards.
    tight = height_nm**2 / 12.0
    print(f"tight-well ceiling h^2/12 for a {height_nm:g} nm well: "
          f"{tight:.3f} nm^2; a carrier is confined when its <z^2> is below it")

    finite = [r for r in rows if np.isfinite(r["beta"])]
    if len(finite) >= 2:
        betas = [r["beta"] for r in finite]
        lo, hi = min(betas), max(betas)
        print(f"beta moves from {lo:.3f} to {hi:.3f} over the swept depths.")
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
    args = parser.parse_args(argv)

    if 0.0 not in args.fields:
        parser.error("--fields must include 0.0 to anchor the Stark shift")
    if args.height >= args.z_extent:
        parser.error(
            f"well height {args.height} nm must be strictly less than the box "
            f"extent {args.z_extent} nm; with the height equal to the box the "
            f"barrier lands on the Dirichlet walls where its value is unused, "
            f"and the solve silently becomes the INFINITE-well problem"
        )

    rows, box = sweep(args.depths, args.fields, args.height, args.transverse,
                      args.z_extent, args.n_xy, args.n_z)
    report(rows, args.fields, box, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
