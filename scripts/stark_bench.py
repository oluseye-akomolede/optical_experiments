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

The output is deliberately not a single beta. Reporting one would require a
depth nobody has measured.

Reading the table
-----------------
``<z2>e`` / ``<z2>h`` fall as the well deepens; the bare-box value printed above
the table is the ceiling. If the hole is still near that ceiling at the depth
you care about, the hole is not confined and the numbers are not a dot's.
``beta`` is a property of the ASSUMED depth, not of the dot: over the depths
swept here it changes sign.

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
    head = f"{'depth':>6} {'E_X(0)':>9} {'<z2>e':>7} {'<z2>h':>7} {'beta':>9}"
    head += "".join(f" {'dE(' + f'{f:g}' + ')':>11}" for f in fields[1:])
    print(head)
    for r in rows:
        line = (f"{r['depth']:6.2f} {r['e_x_0']:9.5f} {r['z2e']:7.3f} "
                f"{r['z2h']:7.3f} {r['beta']:9.3f}")
        line += "".join(f" {s:11.3e}" for s in r["shifts"])
        print(line)
    print()

    finite = [r for r in rows if np.isfinite(r["beta"])]
    if len(finite) >= 2:
        betas = [r["beta"] for r in finite]
        lo, hi = min(betas), max(betas)
        span = (hi - lo) / abs(lo) * 100.0 if lo else float("inf")
        print(f"beta moves from {lo:.3f} to {hi:.3f} over the swept depths "
              f"({span:.0f}% of the smaller magnitude).")
        if lo < 0.0 < hi:
            print("It changes SIGN, so beta is not even of one sign over the "
                  "range of depths a reader could reasonably choose.")
    confined = [r for r in rows if r["z2h"] < 0.5 * bare]
    shallow = [r for r in rows if r["z2h"] >= 0.5 * bare]
    if shallow:
        depths = ", ".join(f"{r['depth']:g}" for r in shallow)
        print(f"The HOLE is not confined (its <z^2> is within a factor 2 of the "
              f"bare box) at depth(s) {depths} eV. Those rows are a free "
              f"particle in a box, not a dot.")
    if confined:
        depths = ", ".join(f"{r['depth']:g}" for r in confined)
        print(f"The hole is confined at depth(s) {depths} eV.")
    print()
    print("What this means for the paper's p/beta form: the fit is well-defined "
          "at every depth, but its value is a property of a depth NOBODY HAS "
          "MEASURED. A single beta cannot be quoted until the dot's alloy gap "
          "is retrieved with a bowing parameter. Until then, report the sweep.")


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
