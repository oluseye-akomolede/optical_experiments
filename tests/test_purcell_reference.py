"""The analytic Purcell reference (``purcell_reference.py``) against closed forms.

Every test is tier 1: a few hundred vectorised integrand evaluations per
call, no grid, no solve, under a second each. Each docstring names the
mutation of the module that turns it red; ``docs/purcell-reference.md``
records the run in which each mutation was applied and the test failed.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "dockerfiles" / "maxwell_solver"
sys.path.insert(0, str(SOLVER))

import fdtd  # noqa: E402
import purcell_reference as pr  # noqa: E402

LAM = 900e-9
K0 = 2.0 * math.pi / LAM
GLASS = 2.25            # a lossless n = 1.5 half-space, as a constant permittivity


def gold_r_normal():
    """``(r_p, r_s)`` of bare gold at normal incidence, from the module's own
    Fresnel coefficients (``r_p = -r_s`` there)."""
    eps = pr.substrate_permittivity("Au", LAM)
    r_p, r_s, _, _ = pr._stack(np.array([0.0]), K0, 1.0, 0.0, eps)
    return complex(r_p[0]), complex(r_s[0])


# --------------------------------------------------------------------------
# 1. perfect conductor, no film: the image dipole
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_pec_matches_the_image_dipole(orientation):
    """``F`` for ``metal="PEC"`` equals the image-dipole closed form to 1e-4
    for ``x = 2kd`` from 0.1 to 20 (it agrees to 1e-13; the bound is the
    job's). Catches: the round-trip phase written as ``exp(i k d s_z)``
    instead of ``exp(2 i k d s_z)`` (wrong period: off by O(1) at
    ``x = 20``); the ``3/2`` and ``3/4`` prefactors swapped; the
    ``- s_z^2 r_p`` term of the parallel integrand with the wrong sign
    (``F_par`` would not vanish at the mirror).
    """
    worst = 0.0
    for x in np.geomspace(0.1, 20.0, 40):
        kd = 0.5 * x
        got = pr.purcell_layered(LAM, kd / K0, 0.0, 1.45, "PEC", orientation).F
        want = pr.image_dipole_pec(kd, orientation)
        worst = max(worst, abs(got - want))
    assert worst < 1e-4, worst


def test_image_dipole_limits_at_the_mirror():
    """The closed form itself: ``F_perp -> 2`` and ``F_par -> 0`` as
    ``x -> 0``, ``F -> 1`` as ``x -> inf``. Catches a transcription error in
    ``image_dipole_pec`` (the test above would then agree with nothing)."""
    assert pr.image_dipole_pec(1e-3, "perp") == pytest.approx(2.0, abs=1e-5)
    assert pr.image_dipole_pec(1e-3, "par") == pytest.approx(0.0, abs=1e-5)
    assert pr.image_dipole_pec(1e4, "perp") == pytest.approx(1.0, abs=1e-7)
    assert pr.image_dipole_pec(1e4, "par") == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_pec_radiates_everything(orientation):
    """A perfect conductor absorbs nothing, so the far-field integral over
    the air half-space must reproduce ``F`` on its own: ``nonradiative`` is
    0 to 1e-9. Catches an error in the ``|1 + r e|^2`` far-field integrand
    (the ``s_z^2`` weight on its p part, or the sign in ``1 - r_p e``)
    that leaves the Sommerfeld integral untouched.
    """
    for kd in (0.1, 1.0, 7.0):
        r = pr.purcell_layered(LAM, kd / K0, 0.0, 1.0, "PEC", orientation)
        assert r.transmitted == 0.0
        assert abs(r.nonradiative) < 1e-9, r
        assert r.radiated == pytest.approx(r.F, abs=1e-9)


# --------------------------------------------------------------------------
# 2. large height: F -> 1 with the far-field oscillation
# --------------------------------------------------------------------------

def test_large_height_perpendicular_envelope():
    """Gold, ``x = 2kd`` from 30 to 200: ``F_perp - 1`` follows the
    stationary-phase asymptote ``-3 Re[r_p(0) e^{ix}] / x^2`` to within
    ``6 / x^3`` (the next term of the expansion is ``3 sin x / x^3`` for a
    mirror), so it oscillates with period ``lambda / 2`` in ``d`` and its
    envelope decays as ``1 / x^2``. Catches a wrong phase factor (the
    asymptote's period changes), a wrong sign of ``r_p`` (the oscillation
    inverts) and a missing ``3/2`` (its amplitude halves).
    """
    r_p, _ = gold_r_normal()
    for x in (30.0, 45.0, 70.0, 100.0, 200.0):
        F = pr.purcell_layered(LAM, 0.5 * x / K0, 0.0, 1.0, "Au", "perp").F
        asymptote = -3.0 * (r_p * np.exp(1j * x)).real / x ** 2
        assert abs(F - 1.0 - asymptote) < 6.0 / x ** 3, (x, F - 1.0, asymptote)
        assert abs(F - 1.0) < 3.3 / x ** 2


def test_large_height_parallel_envelope():
    """Gold, ``x = 2kd`` from 30 to 200: ``F_par - 1`` follows
    ``(3/2) Im[r_s(0) e^{ix}] / x`` to within ``3 / x^2``, so its envelope
    decays as ``1 / x`` (slower than the perpendicular dipole's: the
    parallel dipole radiates along the normal). Catches the same mutations
    as the perpendicular case in the ``r_s`` half of the integrand.
    """
    _, r_s = gold_r_normal()
    for x in (30.0, 45.0, 70.0, 100.0, 200.0):
        F = pr.purcell_layered(LAM, 0.5 * x / K0, 0.0, 1.0, "Au", "par").F
        asymptote = 1.5 * (r_s * np.exp(1j * x)).imag / x
        assert abs(F - 1.0 - asymptote) < 3.0 / x ** 2, (x, F - 1.0, asymptote)
        assert abs(F - 1.0) < 1.6 / x


# --------------------------------------------------------------------------
# 3. a film of index 1 is no film
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_unit_index_film_is_no_film(orientation):
    """12 nm of index-1 film under a dipole at 6 nm is the bare substrate
    under a dipole at 18 nm: every field of the result agrees to 1e-12,
    for gold and for glass (which also exercises ``transmitted``). Catches
    the film's round-trip phase written as ``exp(i q_f k0 t)`` (the
    equivalent height would be 12 nm), a stack denominator ``1 - r12 r23 E``
    and a transmission phase that forgets the single pass.
    """
    for metal in ("Au", GLASS):
        with_film = pr.purcell_layered(LAM, 6e-9, 12e-9, 1.0, metal, orientation)
        bare = pr.purcell_layered(LAM, 18e-9, 0.0, 1.45, metal, orientation)
        assert np.allclose(with_film, bare, rtol=1e-12, atol=1e-12), (with_film, bare)


def test_zero_thickness_film_is_no_film():
    """A film of index 1.45 and thickness 0 is also no film (``E = 1``
    collapses the Airy formula to ``r23``). Catches a stack formula that
    keeps the ``r12`` of an interface that is not there."""
    a = pr.purcell_layered(LAM, 6e-9, 0.0, 1.45, "Au", "perp")
    b = pr.purcell_layered(LAM, 6e-9, 0.0, 1.0, "Au", "perp")
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 4. a lossless dielectric half-space: power is conserved
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_lossless_dielectric_conserves_power(orientation):
    """n = 1.5 glass, bare and under 100 nm of n = 1.45 film (no guided
    mode, since the film is below the substrate's index), heights from
    ``lambda / 10^4`` to ``2 lambda``: the Sommerfeld total equals the
    far-field power into the air plus the power into the glass, including
    the ``1 < s < n`` band that is evanescent in air, to 1e-6 (it holds to
    1e-14). The two sides are independent formulas (reflection along a
    complex path against ``|t|^2`` fluxes on the real axis). Catches:
    dropping the forbidden-light band (10 % of the power at small height);
    the ``1 / |s_z|^2`` Weyl weight written as ``1 / s_z^2``; the parallel
    dipole's TE/TM transmission weights swapped; a wrong ``r_p``
    convention (the total then changes while the fluxes do not); an Airy
    denominator ``1 - r12 r23 E`` (the film case).
    """
    for thickness, index in ((0.0, 1.0), (100e-9, 1.45)):
        for d in (1e-4 * LAM, 0.02 * LAM, 0.1 * LAM, 0.5 * LAM, 2.0 * LAM):
            r = pr.purcell_layered(LAM, d, thickness, index, GLASS, orientation)
            assert r.transmitted > 0.3
            assert abs(r.nonradiative) < 1e-6, (thickness, d, r)
            assert r.F == pytest.approx(r.radiated + r.transmitted, abs=1e-6)


def glass_interface_closed_form(orientation):
    """``F(d -> 0)`` of a dipole in air at an n = 3/2 half-space, exactly.

    The real-axis integrals are elementary once the two square roots are
    rationalised (``docs/purcell-reference.md``: ``s_z = v``, Euler
    substitution ``sqrt(v^2 + n^2 - 1) = v + z`` on ``0 < s < 1``;
    ``w = sqrt(n^2 - s^2)`` on the forbidden band ``1 < s < n``, where
    only ``Im r_p`` and ``Im r_s`` survive the real part). Both
    orientations share one log combination.
    """
    r5, r13 = math.sqrt(5.0), math.sqrt(13.0)
    log_sum = (2.0 * math.log((1.0 + r13) * (r13 - r5) / ((r13 + r5) * (r13 - 1.0)))
               + math.log((9.0 * r13 + 13.0 * r5) / (9.0 * r13 - 13.0 * r5)))
    if orientation == "perp":
        return 2511.0 / 1690.0 + 2187.0 * r13 / 10985.0 * log_sum
    return 2653.0 / 1690.0 - 486.0 * r13 / 10985.0 * log_sum


@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_lossless_dielectric_known_limits(orientation):
    """Known limits of the glass half-space: a substrate of index 1 is free
    space (``F = 1`` and exactly half the power goes each way, to 1e-12);
    a dipole far above n = 1.5 tends to ``F = 1`` within the ``1 / x``
    envelope of test 2; and at ``d = lambda / 10^8`` the module reproduces
    the exact ``d -> 0`` closed form, ``F_perp = 2.30357446575872`` and
    ``F_par = 1.38809456316473`` (``glass_interface_closed_form``; an
    independent 30-digit real-axis quadrature gives the same digits), to
    1e-6: finite, no quenching without loss, and more power into the glass
    than into the air. Catches a lossless substrate treated as absorbing
    (``nonradiative`` non-zero), the ``s^3`` weight written as ``s``, and a
    path that mishandles the real branch point at ``s = n``.
    """
    assert glass_interface_closed_form("perp") == pytest.approx(2.30357446575872, abs=1e-12)
    assert glass_interface_closed_form("par") == pytest.approx(1.38809456316473, abs=1e-12)
    free = pr.purcell_layered(LAM, 0.3 * LAM, 0.0, 1.0, 1.0, orientation)
    assert free.F == pytest.approx(1.0, abs=1e-12)
    assert free.radiated == pytest.approx(0.5, abs=1e-12)
    assert free.transmitted == pytest.approx(0.5, abs=1e-12)
    far = pr.purcell_layered(LAM, 50.0 * LAM, 0.0, 1.0, GLASS, orientation)
    assert abs(far.F - 1.0) < 1.6 / (2.0 * K0 * 50.0 * LAM)
    near = pr.purcell_layered(LAM, 1e-8 * LAM, 0.0, 1.0, GLASS, orientation)
    assert near.transmitted > near.radiated
    assert near.F == pytest.approx(glass_interface_closed_form(orientation), abs=1e-6)


# --------------------------------------------------------------------------
# 5. gold at small height: quenching as 1 / d^3
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation, prefactor", [("perp", 3.0 / 8.0), ("par", 3.0 / 16.0)])
def test_gold_quenching_goes_as_inverse_cube(orientation, prefactor):
    """Bare gold at 900 nm, ``d`` from 2 to 5 nm: the log-log slope of the
    non-radiative rate is -3 within 0.06 (the fit gives -2.96; the ``1/d``
    correction is what keeps it from -3), and at 2 nm the rate is within
    1 % of the quasi-static image ``(3/8) Im[(eps - 1)/(eps + 1)] / (kd)^3``
    (half that for a parallel dipole), ``eps`` from ``fdtd.permittivity``.
    Catches: the substrate Fresnel coefficient with its ``eps`` factors
    swapped (the image strength is then ``(1 - eps)/(1 + eps)``, wrong
    sign); the real-axis tail cut short (``TAIL_DECADES`` of 2 loses half
    the rate at 2 nm); a conjugated ``eps`` (negative absorption).
    """
    eps = pr.substrate_permittivity("Au", LAM)
    assert eps == fdtd.permittivity("Au", fdtd.C0 / LAM)
    heights = np.array([2e-9, 3e-9, 4e-9, 5e-9])
    rates = np.array([pr.purcell_layered(LAM, d, 0.0, 1.0, "Au", orientation).nonradiative
                      for d in heights])
    slope = np.polyfit(np.log(heights), np.log(rates), 1)[0]
    assert abs(slope + 3.0) < 0.06, slope
    image = prefactor * ((eps - 1.0) / (eps + 1.0)).imag / (K0 * heights[0]) ** 3
    assert rates[0] == pytest.approx(image, rel=1e-2), (rates[0], image)
    assert rates[0] > 400.0


# --------------------------------------------------------------------------
# 6. the quadrature is converged and the path does not matter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", pr.ORIENTATIONS)
def test_quadrature_and_path_are_converged(orientation):
    """The job's stack (6 nm above 12 nm SiO2 on gold) and a mirror at
    ``kd = 5``: doubling the Gauss points per panel, and halving or doubling
    the depth of the complex detour, each move ``F`` by less than 1e-4
    (they move it by 1e-12). Catches a path integral without its Jacobian
    ``ds/du`` (``F`` then depends on the detour), a detour that returns to
    the real axis before the branch point at ``s = 1`` (``F`` then depends
    on the point count), and a tail that stops before the integrand has
    decayed.
    """
    cases = [((LAM, 6e-9, 12e-9, 1.45, "Au", orientation), {}),
             ((LAM, 5.0 / K0, 0.0, 1.0, "PEC", orientation), {})]
    for args, _ in cases:
        base = pr.purcell_layered(*args).F
        for knobs in ({"points": 2 * pr.POINTS}, {"detour": 0.5 * pr.DETOUR},
                      {"detour": 2.0 * pr.DETOUR}):
            other = pr.purcell_layered(*args, **knobs).F
            assert abs(other - base) < 1e-4, (args, knobs, base, other)


def test_grounded_slab_real_pole_is_passed_below():
    """A lossless 100 nm film of index 1.45 on a perfect conductor guides a
    TM0 mode with a REAL pole in ``1 < s < 1.45``. The detour passes below
    it, so ``F`` is the same for three detour depths to 1e-9 and the
    guided power shows up as ``nonradiative``. Catches a path that runs
    along the real axis (the pole is then on it) and one that passes above.
    """
    values = [pr.purcell_layered(LAM, 50e-9, 100e-9, 1.45, "PEC", "perp", detour=h)
              for h in (0.25, 0.5, 1.0)]
    for v in values[1:]:
        assert v.F == pytest.approx(values[0].F, abs=1e-9)
    assert values[0].nonradiative > 1.0


# --------------------------------------------------------------------------
# the API and the CLI
# --------------------------------------------------------------------------

def test_api_shape_and_errors():
    """``F`` is the first field and ``metal`` accepts the three spellings."""
    r = pr.purcell_layered(LAM, 6e-9, 12e-9, 1.45, "Au", "perp")
    F, radiated, transmitted, nonradiative = r
    assert F == r.F and F > radiated > 0 and transmitted == 0.0
    assert nonradiative == pytest.approx(F - radiated)
    assert pr.purcell_layered(LAM, 6e-9, 0.0, 1.0, 2.25, "par").transmitted > 0
    with pytest.raises(ValueError):
        pr.purcell_layered(LAM, 6e-9, 12e-9, 1.45, "Au", "diagonal")
    with pytest.raises(ValueError):
        pr.purcell_layered(LAM, 0.0, 12e-9, 1.45, "Au", "perp")
    with pytest.raises(fdtd.FDTDError):
        pr.purcell_layered(LAM, 6e-9, 12e-9, 1.45, "Unobtainium", "perp")


def test_cli_prints_the_sweep():
    """The CLI with no arguments prints 11 wavelengths from 850 to 950 nm
    with both orientations, and the 900 nm row is the API's number."""
    out = subprocess.run([sys.executable, str(SOLVER / "purcell_reference.py")],
                         capture_output=True, text=True, check=True, timeout=60).stdout
    rows = [line.split() for line in out.splitlines() if line and not line.startswith("#")]
    assert rows[0] == ["nm", "F_perp", "rad_perp", "F_par", "rad_par"]
    data = rows[1:]
    assert len(data) == 11 and data[0][0] == "850.0" and data[-1][0] == "950.0"
    want = pr.purcell_layered(900e-9, 6e-9, 12e-9, 1.45, "Au", "perp").F
    assert float(data[5][1]) == pytest.approx(want, abs=1e-4)
