"""Analytic Purcell-factor reference: a dipole above air / film / substrate.

The normalised decay rate (Purcell factor) ``F = P / P0`` of an electric
dipole at height ``d`` above a planar stack -- air, a lossless film of index
``n_f`` and thickness ``t``, then a semi-infinite substrate (a library metal
from ``fdtd.permittivity``, a perfect conductor, or any constant
permittivity) -- computed from the Sommerfeld-integral form of the reflected
field at the dipole. It exists to check the FDTD solver's Purcell
measurement against something exact, so nothing here depends on a grid.

Formulation (Novotny & Hecht, *Principles of Nano-Optics*, Cambridge 2006,
ch. 10 "Dipole emission near planar interfaces", pp. 335-360, the section on
decay rates; the same integrals are in sec. II of Chance, Prock & Silbey,
*Adv. Chem. Phys.* 37, 1 (1978)). With ``s = k_rho / k0`` the in-plane
wavenumber normalised to the
free-space one, ``s_z = sqrt(1 - s^2)`` (``Im s_z >= 0``), ``r_p(s)`` and
``r_s(s)`` the stack's Fresnel reflection coefficients seen from the air
side, and ``e = exp(2 i k0 d s_z)`` the round trip to the stack and back::

    F_perp = 1 + (3/2) Re  int_0^inf  (s^3 / s_z)  r_p  e  ds
    F_par  = 1 + (3/4) Re  int_0^inf  (s / s_z) [r_s - s_z^2 r_p]  e  ds

The time convention is ``exp(-i w t)``, the one ``fdtd.permittivity`` uses
(a lossy metal has ``Im eps > 0``), so every ``sqrt(eps - s^2)`` is taken with
a non-negative imaginary part and evanescent waves decay away from the
dipole. ``r_p = (eps_2 k_z1 - eps_1 k_z2) / (eps_2 k_z1 + eps_1 k_z2)``, so a
perfect conductor has ``r_p = +1`` and ``r_s = -1``: the image of a vertical
dipole points the same way and the one of a horizontal dipole is reversed.

The integrand has the surface-plasmon pole of a metal just above the real
``s`` axis (``Im s > 0`` for ``Im eps > 0``) and the ``1 / s_z`` branch point
at ``s = 1`` on it. The integral is therefore taken along a path deformed
into the LOWER half-plane, ``s(u) = u - i h sin(pi u / s_b)`` for
``0 <= u <= s_b``, which passes under every pole and branch point and also
damps the ``exp(2 i k0 d s_z)`` oscillation that makes the real-axis
integrand expensive at large ``d``; beyond ``s_b`` the integrand is real-axis
and decays like ``s^2 exp(-2 k0 d s)``, and it is followed to where that
factor is ``exp(-50)``. A lossless film on a perfect conductor (a grounded
slab, whose TM0 mode has a real pole for ``1 < s < n_f``) is handled by the
same path: passing under the pole is the vanishing-loss limit.

The power split. The power radiated into the air half-space is the
far-field integral (same source, the section on radiation patterns)::

    P_up,perp / P0 = (3/4) int_0^1 (s^3 / s_z) |1 + r_p e|^2 ds
    P_up,par  / P0 = (3/8) int_0^1 (s / s_z) [ |1 + r_s e|^2 + s_z^2 |1 - r_p e|^2 ] ds

and, for a LOSSLESS substrate of index ``n``, the power transmitted into it
is the flux of each plane wave of the dipole's angular spectrum through the
stack, ``|t|^2 Re(k_z3) / eps_3`` per unit incident amplitude, including the
``1 < s < n`` band that is evanescent in air but propagating in the
substrate ("forbidden light")::

    P_down,perp / P0   = (3/4) int_0^n s^3 Re(q3) |t_p|^2 / (eps_3 |s_z|^2) E ds
    P_down,par,TE / P0 = (3/8) int_0^n s   Re(q3) |t_s|^2 / |s_z|^2         E ds
    P_down,par,TM / P0 = (3/8) int_0^n s   Re(q3) |t_p|^2 / eps_3           E ds

with ``q3 = k_z3 / k0``, ``t_p`` the H-field and ``t_s`` the E-field stack
transmission coefficients and ``E = exp(-2 k0 d Im s_z)`` (1 for ``s < 1``).
The ``1 / |s_z|^2`` is the angular-spectrum weight of the dipole's own field
(its ``1 / k_z`` Weyl factor) and ``|t_p|^2`` vanishes like ``|s_z|^2`` at
``s = 1``, so the integrand is finite there. For a lossless substrate the
three add to ``F`` (``tests/test_purcell_reference.py`` holds this to 1e-6);
for a metal nothing escapes downward and ``F - P_up`` is the power absorbed
in it. ``nonradiative`` is always ``F - radiated - transmitted``: for a
lossless stack that supports a guided mode, it is the power the mode carries.

The perfect-conductor closed form the tests compare against is the image
dipole: the reflected field at the dipole is the field of the image at
distance ``2d`` (the dipole field of N&H ch. 8 "Light emission and optical
interactions in nanoscale environments", sec. "Classical lifetimes and decay
rates"), and ``F = 1 + (6 pi eps0 / |p|^2 k^3) Im(p* . E_image)`` (same
chapter, the "Inhomogeneous environment" subsection; also Chance, Prock &
Silbey sec. II for the mirror). With ``x = 2 k d``::

    F_perp = 1 + 3     [ sin x / x^3 - cos x / x^2 ]
    F_par  = 1 + (3/2) [ sin x / x^3 - cos x / x^2 - sin x / x ]

(``F_perp -> 2`` and ``F_par -> 0`` at the mirror, as they must.)

The normalisation ``P0`` is the dipole's power in free space (the dipole sits
in air, ``eps_1 = 1``). ``F`` is the radiative + non-radiative decay rate
normalised to the free-space radiative rate, i.e. the quantity an FDTD run
measures as emitted power over free-space emitted power.
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")   # before numpy loads its BLAS

from typing import NamedTuple

import numpy as np

import fdtd

ORIENTATIONS = ("perp", "par")

#: ``metal`` values that are not a library metal name.
PEC = "PEC"

#: The real-axis tail is followed until ``exp(-2 k0 d sqrt(s^2 - 1))`` is
#: ``exp(-TAIL_DECADES)``; the integrand there is ``s^2`` times that.
TAIL_DECADES = 50.0

#: Default quadrature: Gauss-Legendre points per panel and the depth ``h`` of
#: the complex detour, in units of ``k_rho / k0``. Doubling ``points`` or
#: halving/doubling ``detour`` moves F by under 1e-4 (test 6).
POINTS = 24
DETOUR = 0.5


class PurcellResult(NamedTuple):
    """``F`` and its split, each in units of the free-space dipole power."""

    F: float
    radiated: float          # into the air half-space (far field)
    transmitted: float       # into a lossless substrate's far field; 0 for a metal or PEC
    nonradiative: float      # F - radiated - transmitted: absorbed, or guided


def substrate_permittivity(metal, wavelength_m: float):
    """``eps`` of the substrate at ``wavelength_m``: a library metal by name
    (via ``fdtd.permittivity``), ``"PEC"`` (returned as ``math.inf``), or a
    number used as a constant permittivity (``2.25`` is glass, n = 1.5)."""
    if isinstance(metal, str):
        if metal.upper() == PEC:
            return math.inf
        return complex(fdtd.permittivity(metal, fdtd.C0 / wavelength_m))
    return complex(metal)


def _sqrt_up(w):
    """``sqrt`` with a non-negative imaginary part (the decaying wave)."""
    z = np.sqrt(np.asarray(w, dtype=complex))
    return np.where(z.imag < 0, -z, z)


def _stack(s, k0: float, eps_f, t: float, eps_sub):
    """Fresnel coefficients of air | film | substrate for in-plane ``s``.

    Returns ``(r_p, r_s, t_p, t_s)``: reflection seen from the air side and
    transmission into the substrate, ``p`` in terms of H and ``s`` in terms
    of E, with the Airy stack formulas ``r = (r12 + r23 E) / (1 + r12 r23 E)``
    and ``t = t12 t23 sqrt(E) / (1 + r12 r23 E)``, ``E = exp(2 i q_f k0 t)``.
    Both are even in ``q_f`` so the film's square-root branch is immaterial;
    ``Im q_f >= 0`` keeps ``|E| <= 1``. A perfect conductor has ``r23_p = 1``,
    ``r23_s = -1`` and transmits nothing.
    """
    s = np.asarray(s, dtype=complex)
    q1 = _sqrt_up(1.0 - s * s)
    qf = _sqrt_up(eps_f - s * s)
    r12_p = (eps_f * q1 - qf) / (eps_f * q1 + qf)
    r12_s = (q1 - qf) / (q1 + qf)
    if eps_sub == math.inf:
        r23_p = np.ones_like(s)
        r23_s = -np.ones_like(s)
        t23_p = t23_s = np.zeros_like(s)
    else:
        q3 = _sqrt_up(eps_sub - s * s)
        r23_p = (eps_sub * qf - eps_f * q3) / (eps_sub * qf + eps_f * q3)
        r23_s = (qf - q3) / (qf + q3)
        t23_p, t23_s = 1.0 + r23_p, 1.0 + r23_s
    phase = np.exp(1j * qf * k0 * t)
    E = phase * phase
    den_p, den_s = 1.0 + r12_p * r23_p * E, 1.0 + r12_s * r23_s * E
    r_p = (r12_p + r23_p * E) / den_p
    r_s = (r12_s + r23_s * E) / den_s
    t_p = (1.0 + r12_p) * t23_p * phase / den_p
    t_s = (1.0 + r12_s) * t23_s * phase / den_s
    return r_p, r_s, t_p, t_s


def _gauss(a: float, b: float, points: int):
    """Gauss-Legendre nodes and weights on ``[a, b]``."""
    x, w = np.polynomial.legendre.leggauss(points)
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def _panels(edges, points: int):
    """Composite Gauss-Legendre nodes and weights over consecutive ``edges``."""
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gauss(a, b, points)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _check(wavelength_m, height_m, film_thickness_m, film_index, orientation):
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, not {orientation!r}")
    if not wavelength_m > 0:
        raise ValueError("wavelength_m must be positive")
    if not height_m > 0:
        raise ValueError("height_m must be positive: on the surface the "
                         "non-radiative rate of a lossy substrate is infinite")
    if film_thickness_m < 0:
        raise ValueError("film_thickness_m must be non-negative")
    if not (np.isreal(film_index) and film_index > 0):
        raise ValueError("film_index must be a positive real number (a lossless film)")


def _total(k0, d, eps_f, t, eps_sub, orientation, points, detour):
    """``F`` from the Sommerfeld integral along the deformed path."""
    kd = k0 * d
    n_f = math.sqrt(eps_f)
    n_sub = 1.0 if eps_sub == math.inf else complex(np.sqrt(eps_sub)).real
    s_b = 1.0 + max(1.0, n_f, n_sub)
    # The detour: u in [0, s_b], s = u - i h sin(pi u / s_b). Its panel count
    # follows the phase 2 kd s_z accumulated along it.
    n_detour = 6 + int(math.ceil(kd / 3.0))
    u, wu = _panels(np.linspace(0.0, s_b, n_detour + 1), points)
    s = u - 1j * detour * np.sin(np.pi * u / s_b)
    ds = 1.0 - 1j * detour * (np.pi / s_b) * np.cos(np.pi * u / s_b)
    # The real tail: s_b to s_max in panels whose widths double, starting from
    # a fraction of the exp(-2 kd s) decay length.
    s_max = max(s_b + 1.0, math.sqrt(1.0 + (TAIL_DECADES / (2.0 * kd)) ** 2))
    width = 0.25 / kd
    edges = [s_b]
    while edges[-1] < s_max:
        edges.append(min(edges[-1] + width, s_max))
        width *= 2.0
    v, wv = _panels(np.asarray(edges), points)
    s_all = np.concatenate([s, v.astype(complex)])
    w_all = np.concatenate([wu * ds, wv.astype(complex)])
    s_z = _sqrt_up(1.0 - s_all * s_all)
    r_p, r_s, _, _ = _stack(s_all, k0, eps_f, t, eps_sub)
    e = np.exp(2j * kd * s_z)
    if orientation == "perp":
        integrand = 1.5 * s_all ** 3 / s_z * r_p * e
    else:
        integrand = 0.75 * s_all / s_z * (r_s - s_z * s_z * r_p) * e
    return 1.0 + float(np.sum(w_all * integrand).real)


def _radiated(k0, d, eps_f, t, eps_sub, orientation, points):
    """Power into the air half-space: the far-field integral over s < 1,
    with ``s = sin(theta)`` so the ``1 / s_z`` endpoint is smooth."""
    kd = k0 * d
    n_pan = 4 + int(math.ceil(kd / 2.0))
    theta, w = _panels(np.linspace(0.0, 0.5 * np.pi, n_pan + 1), points)
    s, s_z = np.sin(theta), np.cos(theta)
    r_p, r_s, _, _ = _stack(s, k0, eps_f, t, eps_sub)
    e = np.exp(2j * kd * s_z)
    if orientation == "perp":
        integrand = 0.75 * s ** 3 * np.abs(1.0 + r_p * e) ** 2
    else:
        integrand = 0.375 * s * (np.abs(1.0 + r_s * e) ** 2
                                 + s_z * s_z * np.abs(1.0 - r_p * e) ** 2)
    return float(np.sum(w * integrand))


def _transmitted(k0, d, eps_f, t, eps_sub, orientation, points):
    """Power into a lossless substrate's far field (0 for a metal or PEC)."""
    if eps_sub == math.inf or abs(complex(eps_sub).imag) > 0 or complex(eps_sub).real <= 0:
        return 0.0
    eps3 = complex(eps_sub).real
    n = math.sqrt(eps3)
    kd = k0 * d
    n_pan = 4 + int(math.ceil(kd / 2.0))

    def flux(s, s_abs2, q3_re, weight):
        _, _, t_p, t_s = _stack(s, k0, eps_f, t, eps_sub)
        s_z = _sqrt_up(1.0 - s * s)
        decay = np.exp(-2.0 * kd * s_z.imag)
        s = s.real
        if orientation == "perp":
            f = 0.75 * s ** 3 * q3_re * np.abs(t_p) ** 2 / (eps3 * s_abs2)
        else:
            f = 0.375 * s * q3_re * (np.abs(t_s) ** 2 / s_abs2
                                     + np.abs(t_p) ** 2 / eps3)
        return float(np.sum(weight * f * decay))

    # Propagating in air: s = s_c sin(theta) up to s_c = min(1, n); both square
    # roots are then smooth in theta.
    s_c = min(1.0, n)
    theta, w = _panels(np.linspace(0.0, 0.5 * np.pi, n_pan + 1), points)
    s = s_c * np.sin(theta)
    s_abs2 = np.abs(1.0 - s * s)
    q3_re = np.sqrt(n * n - s * s)
    total = flux(s.astype(complex), s_abs2, q3_re, w * s_c * np.cos(theta))
    if n > 1.0:
        # Forbidden light, 1 < s < n: s^2 = 1 + v^2 sin^2(phi) with
        # v = sqrt(n^2 - 1), so sqrt(s^2 - 1) = v sin(phi) and
        # sqrt(n^2 - s^2) = v cos(phi) are both smooth.
        v = math.sqrt(n * n - 1.0)
        phi, w = _panels(np.linspace(0.0, 0.5 * np.pi, 5), points)
        s = np.sqrt(1.0 + (v * np.sin(phi)) ** 2)
        ds = v * v * np.sin(phi) * np.cos(phi) / s
        total += flux(s.astype(complex), (v * np.sin(phi)) ** 2, v * np.cos(phi), w * ds)
    return total


def purcell_layered(wavelength_m: float, height_m: float, film_thickness_m: float,
                    film_index: float, metal="Au", orientation: str = "perp", *,
                    points: int = POINTS, detour: float = DETOUR) -> PurcellResult:
    """Purcell factor of a dipole ``height_m`` above a film on a substrate.

    ``wavelength_m`` is the free-space wavelength; ``film_thickness_m`` and
    ``film_index`` describe the lossless film (thickness 0, or index 1, means
    no film); ``metal`` is a library metal name, ``"PEC"`` or a constant
    permittivity; ``orientation`` is ``"perp"`` (dipole along the surface
    normal) or ``"par"``. ``points`` and ``detour`` are the quadrature knobs
    of the module docstring. Returns ``PurcellResult(F, radiated,
    transmitted, nonradiative)``; ``F`` is the first field, so
    ``F = purcell_layered(...).F`` or ``F, *split = purcell_layered(...)``.
    """
    _check(wavelength_m, height_m, film_thickness_m, film_index, orientation)
    k0 = 2.0 * math.pi / wavelength_m
    eps_f = float(film_index) ** 2
    eps_sub = substrate_permittivity(metal, wavelength_m)
    args = (k0, height_m, eps_f, film_thickness_m, eps_sub, orientation)
    F = _total(*args, points, detour)
    up = _radiated(*args, points)
    down = _transmitted(*args, points)
    return PurcellResult(F, up, down, F - up - down)


def image_dipole_pec(kd: float, orientation: str) -> float:
    """Closed form for a dipole ``d`` above a perfect conductor, ``x = 2kd``:
    ``F_perp = 1 + 3 (sin x / x^3 - cos x / x^2)``,
    ``F_par = 1 + (3/2)(sin x / x^3 - cos x / x^2 - sin x / x)``
    (image dipole; module docstring)."""
    x = 2.0 * kd
    near = math.sin(x) / x ** 3 - math.cos(x) / x ** 2
    if orientation == "perp":
        return 1.0 + 3.0 * near
    if orientation == "par":
        return 1.0 + 1.5 * (near - math.sin(x) / x)
    raise ValueError(f"orientation must be one of {ORIENTATIONS}, not {orientation!r}")


def main(argv=None) -> int:
    """Print F against wavelength for both orientations (defaults: the
    850-950 nm sweep for a dipole 6 nm above 12 nm of SiO2 on gold)."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--start-nm", type=float, default=850.0)
    p.add_argument("--stop-nm", type=float, default=950.0)
    p.add_argument("--steps", type=int, default=11)
    p.add_argument("--height-nm", type=float, default=6.0)
    p.add_argument("--film-nm", type=float, default=12.0)
    p.add_argument("--film-index", type=float, default=1.45)
    p.add_argument("--metal", default="Au",
                   help="a library metal name, PEC, or a constant permittivity")
    a = p.parse_args(argv)
    try:
        metal = complex(a.metal)
        metal = metal.real if metal.imag == 0 else metal
    except ValueError:
        metal = a.metal
    print(f"# dipole {a.height_nm:g} nm above {a.film_nm:g} nm of n={a.film_index:g} on {a.metal}")
    print(f"{'nm':>8} {'F_perp':>12} {'rad_perp':>10} {'F_par':>12} {'rad_par':>10}")
    for nm in np.linspace(a.start_nm, a.stop_nm, a.steps):
        rows = [purcell_layered(nm * 1e-9, a.height_nm * 1e-9, a.film_nm * 1e-9,
                                a.film_index, metal, o) for o in ORIENTATIONS]
        print(f"{nm:8.1f} {rows[0].F:12.4f} {rows[0].radiated:10.4f} "
              f"{rows[1].F:12.4f} {rows[1].radiated:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
