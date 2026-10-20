#!/usr/bin/env python3
"""Fit a Lorentz-Drude gold model to Palik's table (600-1300 nm).

Reads ``data/palik_au_nk.csv`` (E. D. Palik, Handbook of Optical Constants of
Solids, 1985, gold; via IMD) and fits the model ``fdtd.Metal`` steps,

    eps(w) = 1 - fp^2 / (w^2 + i w G0)
             + sum_j f_j fp^2 / (w_j^2 - w^2 - i w G_j)      (energies in eV),

to the table's complex permittivity over every point in 600-1300 nm. The Drude
pole carries f = 1 and the fit moves its plasma frequency; every parameter is
bounded positive (``f_j > 0`` and ``Gamma_j > 0`` keep every pole damped; the
solver's ``stable_dt`` still bounds the step at the poles' own frequencies),
and ``eps_inf`` stays 1, as the rest of the library and the ADE assume.

Prints the ``Metal`` literal for ``fdtd.PALIK_1985`` and the per-point error;
the worst point is what the test's tolerance is set above. Run:

    .venv-mse/bin/python scripts/fit_palik_gold.py

The fit is a seeded multi-start: the same seed and start count print the same
numbers on the same numpy/scipy. It is tier-1 arithmetic (a few thousand small
least-squares steps), not a solver run.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "palik_au_nk.csv"
#: h c in eV nm: photon energy is this over the wavelength in nm.
EV_NM = 1239.841984
BAND_NM = (600.0, 1300.0)


def load(path=DATA):
    """(wavelength nm, n, k) for every table row in the band."""
    with open(path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))
    lam = np.array([float(row["wavelength_A"]) / 10.0 for row in rows])
    n = np.array([float(row["n"]) for row in rows])
    k = np.array([float(row["k"]) for row in rows])
    keep = (lam >= BAND_NM[0]) & (lam <= BAND_NM[1])
    return lam[keep], n[keep], k[keep]


def model(p, energy_ev):
    """The library's eps(w) for a flat parameter vector (fp, G0, f, w0, G...)."""
    fp, g0 = p[0], p[1]
    eps = 1.0 - fp ** 2 / (energy_ev ** 2 + 1j * energy_ev * g0)
    for j in range((len(p) - 2) // 3):
        f, w0, gamma = p[2 + 3 * j:5 + 3 * j]
        eps = eps + f * fp ** 2 / (w0 ** 2 - energy_ev ** 2
                                   - 1j * energy_ev * gamma)
    return eps


def fit(energy_ev, target, poles, *, starts=64, seed=1,
        gamma_floor=0.01, max_nfev=8000):
    """The best of ``starts`` bounded relative-error fits, by cost."""
    rng = np.random.default_rng(seed)

    def residuals(p):
        got = model(p, energy_ev)
        return np.concatenate(((got.real - target.real) / np.abs(target),
                               (got.imag - target.imag) / np.abs(target)))

    best = None
    for _ in range(starts):
        p0 = [rng.uniform(4.0, 12.0), rng.uniform(gamma_floor, 0.5)]
        low = [1.0, gamma_floor]
        high = [30.0, 5.0]
        for _ in range(poles):
            p0 += [rng.uniform(1e-3, 2.0), rng.uniform(0.3, 4.0),
                   rng.uniform(gamma_floor, 2.0)]
            low += [1e-6, 1e-3, gamma_floor]
            high += [20.0, 10.0, 10.0]
        found = least_squares(residuals, p0, bounds=(low, high),
                              max_nfev=max_nfev)
        if best is None or found.cost < best.cost:
            best = found
    return best


def literal(p):
    """The ``Metal(...)`` literal the fit prints, to 10 significant digits.

    The digits matter: 4 of them moved the 2.11% fit to 3.13%, past the gate.
    """
    poles = [(1.0, 0.0, p[1])]
    for j in range((len(p) - 2) // 3):
        poles.append(tuple(p[2 + 3 * j:5 + 3 * j]))
    shown = ", ".join(f"({f:.10g}, {w0:.10g}, {g:.10g})" for f, w0, g in poles)
    return f"Metal(\"Au-palik\", {p[0]:.10g}, ({shown}), (0.96, 1.9))"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poles", type=int, default=4,
                        help="Lorentz poles besides the Drude one (default 4)")
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gamma-floor", type=float, default=1e-5,
                        help="lowest Lorentz/drude damping the fit may pick")
    args = parser.parse_args(argv)

    lam_nm, n, k = load()
    energy = EV_NM / lam_nm
    target = (n + 1j * k) ** 2
    found = fit(energy, target, args.poles, starts=args.starts,
                seed=args.seed, gamma_floor=args.gamma_floor)
    got = model(found.x, energy)
    error = np.abs(got - target) / np.abs(target)

    print(f"# {len(lam_nm)} Palik points, {lam_nm.min():.0f}-{lam_nm.max():.0f} nm,"
          f" {args.poles} Lorentz poles, {args.starts} starts, seed {args.seed}")
    print(f"# max |eps error| / |eps| = {error.max() * 100:.2f}%"
          f" at {lam_nm[error.argmax()]:.0f} nm")
    print(literal(found.x))
    print("# wavelength_nm  eps_target  eps_model  rel_error")
    for lam, want, has, err in zip(lam_nm, target, got, error):
        print(f"  {lam:9.1f}  {want.real:9.3f}{want.imag:+8.3f}i"
              f"  {has.real:9.3f}{has.imag:+8.3f}i  {err * 100:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
