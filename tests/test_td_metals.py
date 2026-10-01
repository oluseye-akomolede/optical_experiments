"""The time-domain metal library: the Rakic 1998 LD table, its ranges, and Fresnel.

WHAT THE TABLE IS CHECKED AGAINST. The spec scenario "the material table
matches the paper" asks for an INDEPENDENTLY published Rakic LD n, k dataset.
The one used is the refractiveindex.info database (M. N. Polyanskiy, Sci. Data
11, 94 (2024), doi:10.1038/s41597-023-02898-2, CC0), files

    database/data/main/{Au,Ag,Al,Cu,Cr}/nk/Rakic-LD.yml
    https://github.com/polyanskiy/refractiveindex.info-database (master, fetched 2026-10-01)

whose REFERENCES line is Rakic et al., Appl. Opt. 37, 5271 (1998). The rows
below are copied verbatim from those files (wavelength in um, n, k), the rows
nearest 1, 2 and 3 eV. They are NOT recomputed from ``td_metals`` -- a test that
compares the module with itself cannot fail.

The tolerance is the spec's 2 % on complex eps = (n + i k)^2; the copied table
actually agrees to ~1e-4 (its 5 significant figures). The control below shows
the 2 % check is tight enough to catch a wrong parameter set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import td_metals as T  # noqa: E402

# h c in eV*um, from the module's own constants (exact SI values).
HC_EV_UM = T.H_EV_S * T.C0 * 1e6

# refractiveindex.info Rakic-LD.yml rows: (wavelength um, n, k).
RAKIC_LD_NK = {
    "Au": [(1.2499, 0.37924, 7.7361), (0.62346, 0.32432, 3.0630), (0.41609, 1.4122, 1.7441)],
    "Ag": [(1.2430, 0.28978, 8.0929), (0.62468, 0.14257, 3.7474), (0.41340, 0.14871, 1.9714)],
    "Al": [(1.2416, 1.2767, 12.027), (0.61817, 1.2723, 7.1563), (0.41499, 0.49088, 4.8882)],
    "Cu": [(1.2376, 0.32537, 8.2260), (0.61488, 0.36943, 3.3746), (0.41593, 1.3033, 2.2558)],
    "Cr": [(1.2385, 4.4573, 4.3973), (0.62066, 3.3668, 4.3328), (0.41231, 1.8558, 3.6405)],
}


def _table_errors(params, rows):
    lam, n, k = np.array(rows).T
    ref = (n + 1j * k) ** 2
    return np.abs(T.eps_ev(params, HC_EV_UM / lam) - ref) / np.abs(ref)


def test_the_library_has_exactly_the_five_metals():
    assert sorted(T.METALS) == sorted(RAKIC_LD_NK) == ["Ag", "Al", "Au", "Cr", "Cu"]


@pytest.mark.parametrize("name", sorted(RAKIC_LD_NK))
def test_the_material_table_matches_the_paper(name):
    rows = RAKIC_LD_NK[name]
    assert len(rows) == 3
    err = _table_errors(name, rows)
    assert np.all(err < 0.02), f"{name}: relative eps error {err}"


@pytest.mark.parametrize("name, wrong", [("Au", "Cu"), ("Cu", "Au"), ("Ag", "Au"),
                                         ("Al", "Ag"), ("Cr", "Al")])
def test_the_table_check_can_fail_on_the_wrong_parameters(name, wrong):
    """Control: another metal's fit (all ranges cover 1-3 eV) misses by far more than 2 %."""
    assert np.max(_table_errors(T.metal(wrong), RAKIC_LD_NK[name])) > 0.02


def test_the_table_check_can_fail_on_a_two_percent_plasma_error():
    """Control: a 2 % error in one parameter (w_p) is already caught."""
    au = T.metal("Au")
    off = T.LorentzDrude(au.name, au.wp * 1.02, au.f0, au.gamma0, au.lorentz,
                         au.e_min_ev, au.e_max_ev)
    assert np.max(_table_errors(off, RAKIC_LD_NK["Au"])) > 0.02


def test_a_lossy_metal_has_positive_imaginary_parts():
    for name in T.METALS:
        e = np.linspace(1.0, 3.0, 21)
        eps = T.eps_ev(name, e)
        assert np.all(eps.imag > 0)
        assert np.all(T.refractive_index(eps).imag > 0)


# ---------------------------------------------------------------------------
# Refusal outside the fitted range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RAKIC_LD_NK))
def test_an_energy_outside_the_fit_is_refused_naming_metal_and_range(name):
    m = T.metal(name)
    for bad in (m.e_min_ev * 0.9, m.e_max_ev * 1.1):
        with pytest.raises(T.BandOutsideFit) as exc:
            T.eps_ev(name, [1.0, bad])
        msg = str(exc.value)
        assert name in msg and "Rakic" in msg
        assert f"{m.e_min_ev:g}-{m.e_max_ev:g} eV" in msg
    # The ends themselves are inside.
    T.eps_ev(name, [m.e_min_ev, m.e_max_ev])


def test_a_source_band_leaving_the_fit_is_refused_in_hz():
    au = T.metal("Au")
    f_lo, f_hi = 1.0 / T.H_EV_S, 2.5 / T.H_EV_S   # 1-2.5 eV: inside
    T.require_band("Au", f_lo, f_hi)
    with pytest.raises(T.BandOutsideFit, match=r"Au: .* Hz\) is outside"):
        T.require_band("Au", f_lo, (au.e_max_ev + 0.5) / T.H_EV_S)
    with pytest.raises(T.BandOutsideFit):
        T.require_band("Au", 0.1 / T.H_EV_S, f_hi)    # Au's fit starts at 0.2 eV
    T.require_band("Al", 0.1 / T.H_EV_S, f_hi)        # Al's starts at 0.005 eV
    with pytest.raises(ValueError, match="reversed"):
        T.require_band("Au", f_hi, f_lo)


def test_nonfinite_energies_are_refused():
    with pytest.raises(T.BandOutsideFit):
        T.eps_ev("Ag", [1.0, np.nan])


def test_an_unknown_metal_is_refused_listing_the_known_ones():
    with pytest.raises(ValueError, match="unknown metal 'Pt'.*Ag, Al, Au, Cr, Cu"):
        T.metal("Pt")
    with pytest.raises(ValueError, match="unknown metal"):
        T.eps_ev("gold", 2.0)


# ---------------------------------------------------------------------------
# Analytic Fresnel reflectance (benchmark V4's reference)
# ---------------------------------------------------------------------------

def test_fresnel_reproduces_known_dielectric_and_conductor_limits():
    assert T.fresnel_reflectance(2.25) == pytest.approx(0.04)          # glass, n = 1.5
    assert T.fresnel_reflectance(1.0) == pytest.approx(0.0)
    assert T.fresnel_reflectance(-1e8 + 1j) == pytest.approx(1.0, abs=1e-3)
    # From glass into vacuum is the same 4 % (reciprocity at normal incidence).
    assert T.fresnel_reflectance(1.0, eps_ambient=2.25) == pytest.approx(0.04)


def test_fresnel_never_exceeds_one_for_a_passive_metal():
    """The branch of sqrt matters: the k < 0 root gives R > 1 for a lossy metal."""
    eps = np.array([-10 + 0.1j, -50 + 5j, -1 + 3j, 0.5 + 0.01j])
    r = T.fresnel_reflectance(eps)
    assert np.all((r >= 0) & (r <= 1))
    n_wrong = -T.refractive_index(eps)
    r_wrong = np.abs((1 - n_wrong) / (1 + n_wrong)) ** 2
    assert np.all(r_wrong > 1)   # control: the other branch is unphysical


@pytest.mark.parametrize("name", sorted(RAKIC_LD_NK))
def test_metal_reflectance_matches_the_n_k_formula_on_the_dataset(name):
    """R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2), from the dataset's own n, k."""
    lam, n, k = np.array(RAKIC_LD_NK[name]).T
    f = T.C0 / (lam * 1e-6)
    r_ref = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
    np.testing.assert_allclose(T.metal_reflectance(name, f), r_ref, atol=2e-4)


def test_gold_reflects_red_light_more_than_blue():
    f_red, f_blue = T.C0 / 650e-9, T.C0 / 450e-9
    assert T.metal_reflectance("Au", f_red) > 0.9 > 0.5 > T.metal_reflectance("Au", f_blue)


def test_metal_reflectance_refuses_outside_the_fit():
    with pytest.raises(T.BandOutsideFit):
        T.metal_reflectance("Au", 6.0 / T.H_EV_S)


def test_gold_skin_depth_at_1_55_um_is_about_25_nm():
    d = T.skin_depth_m("Au", T.C0 / 1.55e-6)
    assert 20e-9 < d < 30e-9
    lam, n, k = RAKIC_LD_NK["Au"][0]
    assert T.skin_depth_m("Au", T.C0 / (lam * 1e-6)) == pytest.approx(
        lam * 1e-6 / (2 * np.pi * k), rel=1e-3)


# ---------------------------------------------------------------------------
# ADE pole form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RAKIC_LD_NK))
def test_the_ade_poles_rebuild_eps(name):
    """The SI pole terms an ADE update would step sum back to eps_ev exactly."""
    (wd, gd), lorentz = T.ade_poles(name)
    e = np.linspace(0.5, 4.5, 9)
    w = e / T.HBAR_EV_S
    eps = 1 - wd ** 2 / (w * (w + 1j * gd))
    for d_eps, wj, gj in lorentz:
        eps = eps + d_eps * wj ** 2 / (wj ** 2 - w ** 2 - 1j * w * gj)
    np.testing.assert_allclose(eps, T.eps_ev(name, e), rtol=1e-12)
    assert len(lorentz) == len(T.metal(name).lorentz)
