"""Dispersive metals for the time-domain solver: the Rakic 1998 Lorentz-Drude table.

SOURCE. Every parameter below is from

    A. D. Rakic, A. B. Djurisic, J. M. Elazar, M. L. Majewski,
    "Optical properties of metallic films for vertical-cavity optoelectronic
    devices", Appl. Opt. 37, 5271-5283 (1998), doi:10.1364/AO.37.005271,
    Table 1 (Lorentz-Drude model).

The model, with eps_inf = 1 and every parameter in eV (photon energy E = hbar w):

    eps(w) = 1 - Op^2 / (w (w + i G0))  +  sum_j  f_j wp^2 / (w_j^2 - w^2 - i w G_j)
    Op = sqrt(f0) wp

The sign convention is the physics one, fields ~ exp(-i w t), so a lossy metal
has Im(eps) > 0 and Im(n) > 0. The FDTD ADE terms (see :func:`ade_poles`) are
real-valued in time and do not depend on that choice; only the frequency-domain
helpers here do.

THE CHECK THAT THE TABLE WAS COPIED RIGHT. ``tests/test_td_metals.py`` compares
:func:`eps_ev` with an independently tabulated Rakic LD n, k dataset, the
refractiveindex.info database files ``data/main/<M>/nk/Rakic-LD.yml``
(M. N. Polyanskiy, Sci. Data 11, 94 (2024), doi:10.1038/s41597-023-02898-2;
CC0, https://github.com/polyanskiy/refractiveindex.info-database), at three
photon energies per metal, to 2 %.

VALID RANGE. A fit is only a fit inside the band it was fitted over. Outside
it the LD sum is an extrapolation with no measurement behind it, so every
function here that evaluates eps REFUSES an energy outside
``[e_min_ev, e_max_ev]`` with :class:`BandOutsideFit`. The ranges are the
spectral ranges of the published Rakic LD tables named above (Au 0.2-5 eV,
Ag 0.1-5 eV, Al 0.005-20 eV, Cu 0.1-6 eV, Cr 0.02-5 eV).

numpy only; no GPU. This module is the material table and the analytic
references (Fresnel, skin depth) used by benchmark V4; the ADE update itself
belongs to the FDTD core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# hbar / e in eV*s (CODATA 2018, exact): w [rad/s] = E [eV] / HBAR_EV_S.
HBAR_EV_S = 6.582119569e-16
# h / e in eV*s: f [Hz] = E [eV] / H_EV_S.
H_EV_S = 2.0 * np.pi * HBAR_EV_S
C0 = 299_792_458.0


class BandOutsideFit(ValueError):
    """An energy (or a source band) lies outside a metal's fitted range."""


@dataclass(frozen=True)
class LorentzDrude:
    """One metal's Rakic LD fit. Energies in eV."""

    name: str
    wp: float                 # plasma energy hbar*w_p
    f0: float                 # Drude oscillator strength
    gamma0: float             # Drude damping
    # Lorentz terms: (f_j, Gamma_j, omega_j), in the paper's order j = 1..k.
    lorentz: tuple[tuple[float, float, float], ...]
    e_min_ev: float           # fitted range, inclusive
    e_max_ev: float

    def check_ev(self, e_ev) -> None:
        """Refuse any energy outside the fit's range, naming metal, value and range."""
        e = np.asarray(e_ev, dtype=float)
        if e.size == 0:
            return
        lo, hi = float(np.min(e)), float(np.max(e))
        if not (np.all(np.isfinite(e)) and lo >= self.e_min_ev and hi <= self.e_max_ev):
            raise BandOutsideFit(
                f"{self.name}: {lo:.4g}-{hi:.4g} eV "
                f"({lo / H_EV_S:.4g}-{hi / H_EV_S:.4g} Hz) is outside the "
                f"Rakic 1998 Lorentz-Drude fit's range of "
                f"{self.e_min_ev:g}-{self.e_max_ev:g} eV "
                f"({self.e_min_ev / H_EV_S:.4g}-{self.e_max_ev / H_EV_S:.4g} Hz); "
                f"narrow the source band or choose another material"
            )


def _ld(name, wp, f0, g0, lorentz, e_min, e_max):
    return LorentzDrude(name, wp, f0, g0, tuple(lorentz), e_min, e_max)


# Rakic et al. 1998, Table 1 (LD). Each Lorentz row is (f_j, Gamma_j [eV], omega_j [eV]).
METALS: dict[str, LorentzDrude] = {
    "Au": _ld("Au", 9.03, 0.760, 0.053, [
        (0.024, 0.241, 0.415),
        (0.010, 0.345, 0.830),
        (0.071, 0.870, 2.969),
        (0.601, 2.494, 4.304),
        (4.384, 2.214, 13.32),
    ], 0.2, 5.0),
    "Ag": _ld("Ag", 9.01, 0.845, 0.048, [
        (0.065, 3.886, 0.816),
        (0.124, 0.452, 4.481),
        (0.011, 0.065, 8.185),
        (0.840, 0.916, 9.083),
        (5.646, 2.419, 20.29),
    ], 0.1, 5.0),
    "Al": _ld("Al", 14.98, 0.523, 0.047, [
        (0.227, 0.333, 0.162),
        (0.050, 0.312, 1.544),
        (0.166, 1.351, 1.808),
        (0.030, 3.382, 3.473),
    ], 0.005, 20.0),
    "Cu": _ld("Cu", 10.83, 0.575, 0.030, [
        (0.061, 0.378, 0.291),
        (0.104, 1.056, 2.957),
        (0.723, 3.213, 5.300),
        (0.638, 4.305, 11.18),
    ], 0.1, 6.0),
    "Cr": _ld("Cr", 10.75, 0.168, 0.047, [
        (0.151, 3.175, 0.121),
        (0.150, 1.305, 0.543),
        (1.149, 2.676, 1.970),
        (0.825, 1.335, 8.775),
    ], 0.02, 5.0),
}


def metal(name: str) -> LorentzDrude:
    """The library entry for ``name``; an unknown name is refused, listing the known ones."""
    try:
        return METALS[name]
    except KeyError:
        raise ValueError(
            f"unknown metal {name!r}; the library has {', '.join(sorted(METALS))}"
        ) from None


def _as_metal(m) -> LorentzDrude:
    return m if isinstance(m, LorentzDrude) else metal(m)


def eps_ev(m, e_ev):
    """Complex relative permittivity at photon energy ``e_ev`` (eV), exp(-iwt) convention.

    Refused with :class:`BandOutsideFit` outside the fit's range.
    """
    m = _as_metal(m)
    m.check_ev(e_ev)
    w = np.asarray(e_ev, dtype=float).astype(complex)
    wp2 = m.wp ** 2
    eps = 1.0 - m.f0 * wp2 / (w * (w + 1j * m.gamma0))
    for f, g, wj in m.lorentz:
        eps = eps + f * wp2 / (wj ** 2 - w ** 2 - 1j * w * g)
    return eps


def eps_hz(m, f_hz):
    """:func:`eps_ev` at frequency ``f_hz`` (Hz)."""
    return eps_ev(m, np.asarray(f_hz, dtype=float) * H_EV_S)


def require_band(m, f_min_hz: float, f_max_hz: float) -> None:
    """Refuse a source band [f_min, f_max] (Hz) that leaves ``m``'s fitted range.

    This is the pre-queue check: call it with the pulse band of every metal in
    the scene.
    """
    m = _as_metal(m)
    if not f_min_hz <= f_max_hz:
        raise ValueError(f"band is empty or reversed: {f_min_hz!r} > {f_max_hz!r} Hz")
    m.check_ev([f_min_hz * H_EV_S, f_max_hz * H_EV_S])


def refractive_index(eps):
    """n + i k = sqrt(eps) on the branch with k >= 0 (passive medium, exp(-iwt))."""
    n = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(n.imag < 0, -n, n)


def fresnel_reflectance(eps, eps_ambient=1.0):
    """Normal-incidence power reflectance |(n1 - n2)/(n1 + n2)|^2 of a half-space.

    ``eps`` is the half-space's relative permittivity, ``eps_ambient`` the
    incidence side's. This is the analytic reference of benchmark V4 (a slab
    of >= 10 skin depths is a half-space to 1e-8 in amplitude).
    """
    n1 = refractive_index(eps_ambient)
    n2 = refractive_index(eps)
    return np.abs((n1 - n2) / (n1 + n2)) ** 2


def metal_reflectance(m, f_hz, eps_ambient=1.0):
    """Fresnel reflectance of a ``m`` half-space at ``f_hz`` (Hz), same LD model."""
    return fresnel_reflectance(eps_hz(m, f_hz), eps_ambient)


def skin_depth_m(m, f_hz):
    """Field 1/e decay length c / (w k) in metres (field ~ exp(-z/delta))."""
    k = refractive_index(eps_hz(m, f_hz)).imag
    return C0 / (2.0 * np.pi * np.asarray(f_hz, dtype=float) * k)


def ade_poles(m):
    """The model as SI pole terms for an ADE update, in rad/s.

    Returns ``(drude, lorentz)``:

    * ``drude = (w_D, g_D)``: eps_D(w) = -w_D^2 / (w (w + i g_D)), w_D = sqrt(f0) w_p;
    * ``lorentz = [(d_eps_j, w_j, g_j), ...]``:
      eps_j(w) = d_eps_j w_j^2 / (w_j^2 - w^2 - i w g_j), d_eps_j = f_j w_p^2 / w_j^2.

    eps = 1 + eps_D + sum eps_j, i.e. exactly :func:`eps_ev`. eps_inf is 1, so the
    ADE terms are the whole dispersion and the core's eps0 is unchanged.
    """
    m = _as_metal(m)
    s = 1.0 / HBAR_EV_S
    drude = (np.sqrt(m.f0) * m.wp * s, m.gamma0 * s)
    lorentz = [(f * m.wp ** 2 / wj ** 2, wj * s, g * s) for f, g, wj in m.lorentz]
    return drude, lorentz
