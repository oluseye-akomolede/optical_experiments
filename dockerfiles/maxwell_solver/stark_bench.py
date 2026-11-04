"""Stark-confinement bench: E_X(F), and the p/beta fit (task 1.9).

What this computes
------------------
For one InAsP/InP nanowire dot in an axial field F:

    E_X(F) = E_e(F) + E_h(F) + E_C(F)

each single-particle term being the ground state of a one-band effective-mass
Hamiltonian in the sawtooth potential ``-eFz`` (task 1.8's solver), and E_C the
first-order Coulomb term (``coulomb.py``).

Two things this bench does NOT claim, both because the sources do not support
them:

1. No absolute E_X. Without a band gap for the alloy (no bowing parameter was
   retrieved) there is no reference energy, so nothing here is an emission
   energy. What this bench reports is the SHIFT, dE_X(F) = E_X(F) - E_X(0),
   which is what p and beta describe.
2. No sourced p and beta. The dot's effective masses are a linear
   interpolation between cited InP and InAs endpoints; the rule is ours and is
   marked ASSUMED in ``dot_material``. Every number below inherits that.

Run it with:
    python stark_bench.py                  # the E_X(F) table
    python stark_bench.py --fit            # plus the p/beta fit
    python stark_bench.py --report-materials   # the citation table
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import dot_material as DM
import exciton as X

# The dot is modelled in a box a few times its own size, in each dimension.
BOX_PAD = 3.0
"""Box half-width, as a multiple of the dot half-extent."""
N_NODES = 17
"""Nodes per axis. Odd, so the centre node exists and the axis is sampled."""

FIELDS_EV_PER_NM = (0.0, 0.001, 0.002, 0.004, 0.008, 0.016)
"""Axial fields, eV/nm. 0.01 eV/nm is 10 MV/m (10 kV per micrometre)."""


def paper1_material() -> X.DotMaterial:
    """The paper-1 dot: sourced host constants, ASSUMED alloy masses."""
    m_e, m_hh = DM.paper1_dot_at()
    return X.DotMaterial(
        eps_s=DM.INP_EPS_S.value,
        m_e_ratio=m_e.value,
        m_h_ratio=m_hh.value,
        source=(
            f"eps_s: {DM.INP_EPS_S.basis} | m_e*, m_hh*: {m_e.basis} "
            f"(x={DM.DOT_X.value:g} from {DM.DOT_X.basis}). "
            "Alloy masses are ASSUMED interpolation; see dot_material.py."
        ),
    )


def paper1_box() -> X._stark.Box:
    """A cubic box centred on the dot, sized from its cited dimensions."""
    half = max(DM.DOT_HEIGHT_NM.value, DM.DOT_DIAMETER_NM.value) / 2.0
    extent_nm = BOX_PAD * half
    return X._stark.Box(
        origin=(-extent_nm, -extent_nm, -extent_nm),
        size=(2 * extent_nm,) * 3,
        n=(N_NODES, N_NODES, N_NODES),
    )


def run(fields=FIELDS_EV_PER_NM, verbose: bool = True):
    mat = paper1_material()
    box = paper1_box()
    if verbose:
        print(__doc__)
        print("Material (see --report-materials for citations):")
        for line in mat.source.split(" | "):
            print("   " + line)
        print()
        print("Caveat: " + DM.WURZITE_CAVEAT)
        print()
        h = box.size[0] / (box.n[0] - 1)
        print(f"Box: {box.size[0]:.1f} nm per side, {box.n[0]}^3 nodes, "
              f"h = {h:.3f} nm")
        print()

    results = []
    for F in fields:
        r = X.exciton_energy_eV(mat, box, F)
        results.append(r)

    if verbose:
        print(f"{'F (eV/nm)':>10} {'E_e':>10} {'E_h':>10} {'E_C':>10} "
              f"{'E_X':>10} {'dE_X':>11} {'|E_C|/E':>8}")
        print("-" * 82)
        base = results[0].e_x_eV
        for r in results:
            print(f"{r.field_eV_per_nm:>10.4f} {r.e_electron_eV:>10.5f} "
                  f"{r.e_hole_eV:>10.5f} {r.e_coulomb_eV:>10.5f} "
                  f"{r.e_x_eV:>10.5f} {r.e_x_eV - base:>11.3e} "
                  f"{r.coulomb_over_confinement:>8.3f}")
        print()
    return results


def report_materials():
    print(DM.report())


def report_fit(results):
    """Fit and report what the fit can and cannot say about this dot."""
    p, beta, resid, design = X.fit_p_beta(results)
    n_nonzero = sum(1 for r in results if r.field_eV_per_nm != 0.0)
    cond = float(np.linalg.cond(design))

    print("Fit: dE_X(F) = -p F + beta F^2")
    print(f"   p    = {p:.6g}  nm")
    print(f"   beta = {beta:.6g}")
    print(f"   residual norm = {resid:.3e} eV   (over {n_nonzero} non-zero fields)")
    print(f"   design condition number = {cond:.4g}")
    print()
    if n_nonzero <= 2:
        print("   WARNING: two parameters and two free points, so the system is")
        print("   exactly determined and the residual is NOT a fit diagnostic.")
        print()

    lbeta, lresid, drift = X.beta_low_field(results)
    print("Low-field pure-quadratic estimate (single parameter, residual is real):")
    print(f"   beta = {lbeta:.6g}")
    print(f"   residual norm = {lresid:.3e} eV")
    print(f"   per-point dE/F^2 drift across the sweep = {drift:.3e}"
          f"  ({abs(drift / lbeta) * 100:.1f}% of beta)" if lbeta else "")
    if lbeta and abs(drift / lbeta) > 0.25:
        print("   The drift is a large fraction of beta: the sweep has left the")
        print("   quadratic regime. Quote beta as the low-field value only.")
    print()

    print(f"Largest |E_C|/(E_e+E_h) over the sweep = "
          f"{X.perturbative_ratio(results):.3f}")
    print()

    print("Symmetry: the potential -eFz is odd in z, so the symmetric well has")
    print("the same spectrum at +F and -F and the LINEAR Stark coefficient is")
    print("ZERO by construction (exciton.sawtooth_potential refuses F < 0 for")
    print("exactly this reason). The nonzero p above is therefore not a dipole:")
    print("with only F and F^2 in the basis, the fit splits the F^4 content of")
    print("the shift into a spurious linear term. beta_low_field is the")
    print("estimator to quote.")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true", help="also fit p and beta")
    ap.add_argument("--report-materials", action="store_true")
    ap.add_argument("--fields", type=float, nargs="+", default=None)
    args = ap.parse_args(argv)

    if args.report_materials:
        report_materials()
        return 0

    fields = tuple(args.fields) if args.fields else FIELDS_EV_PER_NM
    results = run(fields)
    if args.fit:
        report_fit(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
