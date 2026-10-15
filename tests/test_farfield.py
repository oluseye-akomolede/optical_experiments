"""The near-to-far-field transform against closed-form dipole results.

Every test samples the EXACT field of a Hertzian dipole (Jackson eq. 9.18) on
a box and asks the transform for something a textbook gives in closed form:
the sin^2(theta) pattern, Jackson's radiated power, the cone fraction of a
dipole, the image pattern over a perfect mirror. Each docstring names the
mutation of farfield.py that turns it red; tests/check_discriminating.py
carries the same mutations so a reviewer can re-run them in one command.

Sizes are small on purpose (a 16^3-cell box, a 19 x 24 angle grid): the whole
file runs in a few seconds under OMP_NUM_THREADS=2, and the point is the
formula, not resolution. test_halving_the_spacing_converges is what says the
resolution is enough.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import farfield as ff  # noqa: E402
from fdtd import C0, ETA0  # noqa: E402

LAM = 1.0e-6                      # 1 um: a round wavelength; nothing depends on it
F0 = C0 / LAM
K0 = 2.0 * math.pi / LAM
P_Z = np.array([0.0, 0.0, 1.0e-22])   # C m: radiates 14 W at 1 um, so no power is below approx's 1e-12
P_X = np.array([1.0e-22, 0.0, 0.0])   # absolute floor (every power comparison also passes abs=0)
#: a cube one wavelength on a side, 16 cells per axis: 1536 samples
HALF = 0.5 * LAM
N = 16
THETA = np.linspace(0.0, math.pi / 2.0, 19)
PHI = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
ALL_FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
OPEN_BOTTOM = ("+x", "-x", "+y", "-y", "+z")


def dipole(moment, position):
    return lambda pts: ff.hertzian_dipole(pts, moment, position, F0)


def cube(moment, n=N, position=(0.0, 0.0, 0.0)):
    return ff.sample_box([-HALF] * 3, [HALF] * 3, (n, n, n), dipole(moment, position))


def z_dipole_intensity(theta):
    """Jackson eq. 9.23: dP/dOmega = c^2 eta0 k^4 |p|^2 sin^2(theta) / (32 pi^2)."""
    return C0 ** 2 * ETA0 * K0 ** 4 * np.sum(np.abs(P_Z) ** 2) * np.sin(theta) ** 2 / (32.0 * math.pi ** 2)


def z_dipole_cone_fraction(na):
    """Closed form of int sin^3 over theta < asin(NA) / int over the hemisphere.

    int_0^T sin^3 dtheta = 2/3 - cos T + cos^3 T / 3, hemisphere = 2/3, so the
    fraction is 1 - (3 cos T - cos^3 T) / 2: 0.208 at NA 0.8.
    """
    c = math.sqrt(1.0 - na ** 2)
    return 1.0 - (3.0 * c - c ** 3) / 2.0


def x_dipole_cone_fraction(na):
    """Same for U = 1 - sin^2 cos^2 phi: cone 2 pi (1 - c) - pi (2/3 - c + c^3/3), hemisphere 4 pi / 3."""
    c = math.sqrt(1.0 - na ** 2)
    cone = 2.0 * math.pi * (1.0 - c) - math.pi * (2.0 / 3.0 - c + c ** 3 / 3.0)
    return cone / (4.0 * math.pi / 3.0)


# --------------------------------------------------------------------------
# check 1: a z-dipole in vacuum
# --------------------------------------------------------------------------

def test_z_dipole_intensity_is_sin_squared():
    """U(theta, phi) equals Jackson's dP/dOmega to 1% of its peak (measured 0.41%).

    Absolute, not normalised, so the i k / (4 pi) scale is checked too.
    Mutation it catches: the sign of M_s (``-np.cross(n, e)`` -> ``+``), which
    makes the J and M halves of the equivalence interfere the wrong way and
    turns sin^2 into a lobed pattern (max error 0.5 of the peak).
    """
    fld = ff.far_field(cube(P_Z), F0, THETA, PHI)
    ref = z_dipole_intensity(THETA)[:, None] * np.ones((1, PHI.size))
    assert np.max(np.abs(fld.intensity - ref)) / ref.max() < 0.01


def test_z_dipole_surface_power_matches_jackson():
    """1/2 Re int (E x H*).n dS equals c^2 eta0 k^4 |p|^2 / (12 pi) to 1% (measured 0.096%).

    Mutation it catches: dropping the 1/2 of the time average (``0.5 * np.real``
    -> ``np.real``), which doubles the power.
    """
    assert ff.surface_power(cube(P_Z)) == pytest.approx(ff.dipole_power(P_Z, F0), rel=0.01, abs=0.0)


def test_far_field_power_equals_surface_power():
    """int U dOmega over the full sphere equals the Poynting flux to 1% (measured 0.1%), and Jackson's power.

    Mutation it catches: a wrong far-field scale, e.g. ``1j * k / (4 pi)`` ->
    ``1j / (4 pi)``, which the normalised pattern cannot see.
    """
    faces = cube(P_Z)
    sphere = ff.cone_power(faces, F0, math.pi)
    assert sphere == pytest.approx(ff.surface_power(faces), rel=0.01, abs=0.0)
    assert sphere == pytest.approx(ff.dipole_power(P_Z, F0), rel=0.01, abs=0.0)


# --------------------------------------------------------------------------
# check 2: an x-dipole
# --------------------------------------------------------------------------

def test_x_dipole_pattern():
    """U / max U equals 1 - sin^2(theta) cos^2(phi) to 1% (measured 0.34%).

    A z-dipole has E_phi = 0 everywhere, so only this test sees the E_phi
    line. Mutation it catches: the sign in E_phi (``ETA0 * N_p - L_t`` ->
    ``+ L_t``), which the z-dipole tests pass.
    """
    pattern = ff.radiation_pattern(cube(P_X), F0, THETA, PHI)
    ref = 1.0 - np.sin(THETA)[:, None] ** 2 * np.cos(PHI)[None, :] ** 2
    assert np.max(np.abs(pattern - ref)) < 0.01


# --------------------------------------------------------------------------
# check 3: collection fraction against the closed-form cone integral
# --------------------------------------------------------------------------

def test_collection_fraction_matches_closed_form():
    """NA 0.8 collects 0.208 of a z-dipole's upward power and 0.496 of an x-dipole's, within 1%.

    Measured 0.2076 and 0.4961. Mutation it catches: using the NA as the cone
    angle (``math.asin(na)`` -> ``na``), which reads 0.124 for the z-dipole.
    """
    assert z_dipole_cone_fraction(0.8) == pytest.approx(0.208)
    assert ff.collection_fraction(cube(P_Z), F0, 0.8) == pytest.approx(0.208, rel=0.01)
    assert ff.collection_fraction(cube(P_X), F0, 0.8) == pytest.approx(x_dipole_cone_fraction(0.8), rel=0.01)


def test_collection_fraction_is_one_at_full_aperture():
    """NA 1 is the whole hemisphere, so the fraction is exactly 1 (same quadrature twice)."""
    assert ff.collection_fraction(cube(P_Z, n=8), F0, 1.0) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# check 4: sampling convergence
# --------------------------------------------------------------------------

def test_halving_the_spacing_converges():
    """8 -> 16 cells per side moves the pattern < 2% of peak, the cone power < 0.5%, towards the closed form.

    Measured: pattern 1.0e-2 (errors 1.4e-2 -> 3.4e-3: the midpoint rule is
    O(h^2), so the 8-cell error is about 4x the 16-cell one), cone power
    2.1e-4 (errors 2.4e-4 -> 2.1e-5). Mutation it catches: a
    quadrature weight that is not the cell area (``J * dS[:, None]`` ->
    ``J``), which the normalised pattern and the fraction, both ratios,
    cannot see: the cone power then grows 16x per halving.
    """
    coarse, fine = cube(P_Z, n=8), cube(P_Z, n=16)
    p_coarse = ff.radiation_pattern(coarse, F0, THETA, PHI)
    p_fine = ff.radiation_pattern(fine, F0, THETA, PHI)
    ref = np.sin(THETA)[:, None] ** 2 * np.ones((1, PHI.size))
    assert np.max(np.abs(p_fine - p_coarse)) < 0.02
    assert np.max(np.abs(p_fine - ref)) < np.max(np.abs(p_coarse - ref))
    w_coarse = ff.cone_power(coarse, F0, math.asin(0.8))
    w_fine = ff.cone_power(fine, F0, math.asin(0.8))
    assert abs(w_fine / w_coarse - 1.0) < 0.005
    analytic = 0.208 * 0.5 * ff.dipole_power(P_Z, F0)   # half the power goes up, 0.208 of that in the cone
    assert abs(w_fine - analytic) < abs(w_coarse - analytic)


def test_graded_edges_use_each_cells_own_area():
    """A face with non-uniform cells gives the same power as a uniform one (within 0.2%).

    Mutation it catches: every cell weighted by the first cell's width
    (``np.diff(self.u_edges)`` -> ``np.full(n_u, u_edges[1] - u_edges[0])``).
    """
    uniform = np.linspace(-HALF, HALF, N + 1)
    graded = -HALF + LAM * (0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, N + 1))))  # Chebyshev-like
    assert graded[0] == pytest.approx(-HALF) and graded[-1] == pytest.approx(HALF)
    faces = []
    for normal in ALL_FACES:
        axis = "xyz".index(normal[1])
        position = HALF if normal[0] == "+" else -HALF
        ua, va = [a for a in range(3) if a != axis]
        edges = [uniform] * 3
        edges[ua] = graded            # grade one in-plane axis of every face
        pts = ff.face_points(normal, position, edges[ua], edges[va])
        e, h = dipole(P_Z, (0.0, 0.0, 0.0))(pts)
        faces.append(ff.Face(normal, position, edges[ua], edges[va],
                             np.moveaxis(e, -1, 0), np.moveaxis(h, -1, 0)))
    assert ff.surface_power(faces) == pytest.approx(ff.surface_power(cube(P_Z)), rel=0.002, abs=0.0)


# --------------------------------------------------------------------------
# check 5: unit / frame check
# --------------------------------------------------------------------------

def test_off_centre_dipole_in_non_cubic_box_has_the_same_pattern():
    """A z-dipole at d in a 1.1 x 1.1 x 1.1 um box with unequal cells per axis matches the centred cube to 1%.

    The complex amplitudes differ only by exp(-i k r^.d) (checked to 1% of
    peak; the conjugate phase would be 191% off). Mutations it catches: a
    cell area taken as du^2 instead of du dv (``np.diff(self.v_edges)[None, :]``
    -> ``np.diff(self.u_edges)[None, :]``), invisible wherever the cells are
    square, and the sign of the exponent (``np.exp(-1j * k`` ->
    ``np.exp(1j * k``), which the symmetric cube's pattern cannot see but the
    phase relation does.
    """
    d = np.array([0.1, -0.15, 0.2]) * LAM
    low = np.array([-0.4, -0.6, -0.3]) * LAM
    high = low + 1.1 * LAM
    box = ff.sample_box(low, high, (18, 16, 20), dipole(P_Z, d))
    centred = ff.far_field(cube(P_Z), F0, THETA, PHI)
    shifted = ff.far_field(box, F0, THETA, PHI)
    assert np.max(np.abs(shifted.pattern - centred.pattern)) < 0.01
    r_hat, _, _ = ff._direction_frame(THETA, PHI)
    phase = np.exp(-1j * K0 * (r_hat @ d))
    peak = np.max(np.abs(centred.E_theta))
    assert np.max(np.abs(shifted.E_theta - centred.E_theta * phase)) / peak < 0.01


# --------------------------------------------------------------------------
# the mirror: a surface open at the bottom, with and without the PEC image
# --------------------------------------------------------------------------

H_MIRROR = 0.3 * LAM


def over_pec(pts):
    """A vertical dipole at height h over z = 0 PEC: the dipole plus its same-sign image."""
    e1, h1 = ff.hertzian_dipole(pts, P_Z, [0.0, 0.0, H_MIRROR], F0)
    e2, h2 = ff.hertzian_dipole(pts, P_Z, [0.0, 0.0, -H_MIRROR], F0)
    return e1 + e2, h1 + h2


def mirror_box(faces=OPEN_BOTTOM, n=N):
    return ff.sample_box([-HALF, -HALF, 0.0], [HALF, HALF, 2.0 * HALF], (n, n, n), over_pec, faces=faces)


def pec_reference_pattern():
    """sin^2(theta) cos^2(k h cos theta): the two-element array of dipole and image, normalised."""
    ref = (np.sin(THETA) ** 2 * np.cos(K0 * H_MIRROR * np.cos(THETA)) ** 2)[:, None] * np.ones((1, PHI.size))
    return ref / ref.max()


def test_pec_image_reproduces_the_dipole_over_a_mirror():
    """Top and sides with ``pec_mirror_z=0`` give sin^2 cos^2(k h cos) to 1% (measured 0.65%),
    and the upward far-field power equals the flux through those faces to 0.5% (measured 0.05%).

    Mutation it catches: the image sign of a horizontal electric current
    (``[-1.0, -1.0, 1.0]`` -> ``[1.0, 1.0, -1.0]``), i.e. a magnetic-wall image.
    """
    faces = mirror_box()
    fld = ff.far_field(faces, F0, THETA, PHI, pec_mirror_z=0.0)
    assert np.max(np.abs(fld.pattern - pec_reference_pattern())) < 0.01
    assert ff.hemisphere_power(faces, F0, pec_mirror_z=0.0) == pytest.approx(ff.surface_power(faces), rel=0.005, abs=0.0)


def test_bottom_face_on_the_mirror_plane_is_redundant_with_the_image():
    """Adding the "-z" face on z = 0 changes the image-mode far field by < 1e-9 (it cancels its own image)."""
    without = ff.far_field(mirror_box(OPEN_BOTTOM, n=8), F0, THETA, PHI, pec_mirror_z=0.0)
    with_bottom = ff.far_field(mirror_box(ALL_FACES, n=8), F0, THETA, PHI, pec_mirror_z=0.0)
    peak = np.max(np.abs(without.E_theta))
    assert np.max(np.abs(with_bottom.E_theta - without.E_theta)) / peak < 1e-9


def test_open_bottom_without_image_misses_the_reflection():
    """The documented approximation, measured: top and sides alone, no image, for this emitter
    0.3 wavelengths over the mirror in a 1-wavelength box, put the upward far-field power
    40-45% below the surface flux and the pattern 0.6-0.7 of peak off.

    This pins the numbers farfield.py quotes, so the docstring cannot drift
    from the code. It is not a tolerance to tighten: the error is the
    reflection from the mirror outside the box, which the free-space
    transform cannot contain.
    """
    faces = mirror_box()
    ratio = ff.hemisphere_power(faces, F0) / ff.surface_power(faces)
    assert 0.55 < ratio < 0.60
    assert 0.6 < np.max(np.abs(ff.radiation_pattern(faces, F0, THETA, PHI) - pec_reference_pattern())) < 0.7


def test_samples_below_the_mirror_are_refused():
    with pytest.raises(ff.FarFieldError, match="mirror plane"):
        ff.far_field(mirror_box(n=4), F0, THETA, PHI, pec_mirror_z=0.5 * LAM)


# --------------------------------------------------------------------------
# the layout contract
# --------------------------------------------------------------------------

def test_face_layout_is_checked():
    edges = np.linspace(0.0, 1.0, 5)
    good = np.zeros((3, 4, 4), dtype=complex)
    ff.Face("+z", 1.0, edges, edges, good, good)
    with pytest.raises(ff.FarFieldError, match="normal"):
        ff.Face("z", 1.0, edges, edges, good, good)
    with pytest.raises(ff.FarFieldError, match="shape"):
        ff.Face("+z", 1.0, edges, edges, good[:, :3], good)
    with pytest.raises(ff.FarFieldError, match="increasing"):
        ff.Face("+z", 1.0, edges[::-1], edges, good, good)
    with pytest.raises(ff.FarFieldError, match="no faces"):
        ff.surface_power([])


def test_in_plane_axes_are_the_non_normal_axes_in_increasing_order():
    """The layout rule: "+-x" -> (y, z), "+-y" -> (x, z), "+-z" -> (x, y), samples at cell centres.

    Mutation it catches: ``1: (0, 2)`` -> ``1: (2, 0)`` in ``_IN_PLANE``. Only
    this test sees it, because ``sample_box`` and ``face_points`` share the
    table and stay consistent with each other; a monitor built to the
    documented order would not.
    """
    u = np.array([0.0, 1.0, 3.0])
    v = np.array([10.0, 12.0])
    pts = ff.face_points("-y", 5.0, u, v)
    assert pts.shape == (2, 1, 3)
    np.testing.assert_allclose(pts[:, 0, 0], [0.5, 2.0])     # x from u
    np.testing.assert_allclose(pts[:, 0, 1], [5.0, 5.0])     # y is the position
    np.testing.assert_allclose(pts[:, 0, 2], [11.0, 11.0])   # z from v
    face = ff.Face("-y", 5.0, u, v, np.zeros((3, 2, 1), complex), np.zeros((3, 2, 1), complex))
    np.testing.assert_allclose(face.areas()[:, 0], [2.0, 4.0])
    np.testing.assert_allclose(face.unit_normal, [0.0, -1.0, 0.0])


def test_bad_parameters_are_refused():
    faces = cube(P_Z, n=4)
    with pytest.raises(ff.FarFieldError, match="numerical aperture"):
        ff.collection_fraction(faces, F0, 1.2)
    with pytest.raises(ff.FarFieldError, match="frequency"):
        ff.far_field(faces, 0.0, THETA, PHI)
    with pytest.raises(ff.FarFieldError, match="theta_max"):
        ff.cone_quadrature(0.0, 4, 4)
    with pytest.raises(ff.FarFieldError, match="empty"):
        ff.sample_box([0, 0, 0], [1, 1, 0], (2, 2, 2), dipole(P_Z, (5, 5, 5)))


# --------------------------------------------------------------------------
# every number the docstrings and the spec quote, in one command
# --------------------------------------------------------------------------

def report():
    """``OMP_NUM_THREADS=2 python tests/test_farfield.py`` prints the measured values the tests bound."""
    z_cube, x_cube = cube(P_Z), cube(P_X)
    fld = ff.far_field(z_cube, F0, THETA, PHI)
    ref = z_dipole_intensity(THETA)[:, None] * np.ones((1, PHI.size))
    p0, ps = ff.dipole_power(P_Z, F0), ff.surface_power(z_cube)
    sphere = ff.cone_power(z_cube, F0, math.pi)
    refx = 1.0 - np.sin(THETA)[:, None] ** 2 * np.cos(PHI)[None, :] ** 2
    coarse = cube(P_Z, n=8)
    refz = np.sin(THETA)[:, None] ** 2 * np.ones((1, PHI.size))
    p_coarse, p_fine = (ff.radiation_pattern(b, F0, THETA, PHI) for b in (coarse, z_cube))
    w_coarse, w_fine = (ff.cone_power(b, F0, math.asin(0.8)) for b in (coarse, z_cube))
    w_exact = 0.208 * 0.5 * p0
    d = np.array([0.1, -0.15, 0.2]) * LAM
    low = np.array([-0.4, -0.6, -0.3]) * LAM
    shifted = ff.far_field(ff.sample_box(low, low + 1.1 * LAM, (18, 16, 20), dipole(P_Z, d)), F0, THETA, PHI)
    r_hat, _, _ = ff._direction_frame(THETA, PHI)
    phase = np.exp(-1j * K0 * (r_hat @ d))
    peak = np.max(np.abs(fld.E_theta))
    mirror = mirror_box()
    image = ff.far_field(mirror, F0, THETA, PHI, pec_mirror_z=0.0)
    open_pattern = ff.radiation_pattern(mirror, F0, THETA, PHI)
    rows = [
        ("check 1  z-dipole |U - Jackson| / peak", np.max(np.abs(fld.intensity - ref)) / ref.max()),
        ("check 1  surface power / Jackson - 1", ps / p0 - 1.0),
        ("check 1  sphere far-field power / surface power - 1", sphere / ps - 1.0),
        ("check 2  x-dipole pattern error", np.max(np.abs(ff.radiation_pattern(x_cube, F0, THETA, PHI) - refx))),
        ("check 3  z-dipole fraction in NA 0.8 (closed form 0.208)", ff.collection_fraction(z_cube, F0, 0.8)),
        ("check 3  x-dipole fraction in NA 0.8 (closed form 0.496)", ff.collection_fraction(x_cube, F0, 0.8)),
        ("check 4  pattern change 8 -> 16 cells", np.max(np.abs(p_fine - p_coarse))),
        ("check 4  pattern error at 8 cells", np.max(np.abs(p_coarse - refz))),
        ("check 4  pattern error at 16 cells", np.max(np.abs(p_fine - refz))),
        ("check 4  NA 0.8 cone power change 8 -> 16 cells", w_fine / w_coarse - 1.0),
        ("check 4  cone power error at 8 cells", w_coarse / w_exact - 1.0),
        ("check 4  cone power error at 16 cells", w_fine / w_exact - 1.0),
        ("check 5  off-centre non-cubic pattern - centred cube", np.max(np.abs(shifted.pattern - fld.pattern))),
        ("check 5  phase relation error / peak", np.max(np.abs(shifted.E_theta - fld.E_theta * phase)) / peak),
        ("check 5  same with the conjugate phase", np.max(np.abs(shifted.E_theta - fld.E_theta * np.conj(phase))) / peak),
        ("mirror   image-mode pattern error", np.max(np.abs(image.pattern - pec_reference_pattern()))),
        ("mirror   image-mode upward power / surface flux - 1", ff.hemisphere_power(mirror, F0, pec_mirror_z=0.0) / ff.surface_power(mirror) - 1.0),
        ("mirror   open-bottom upward power / surface flux", ff.hemisphere_power(mirror, F0) / ff.surface_power(mirror)),
        ("mirror   open-bottom pattern error", np.max(np.abs(open_pattern - pec_reference_pattern()))),
        ("mirror   NA 0.8 fraction, image mode", ff.collection_fraction(mirror, F0, 0.8, pec_mirror_z=0.0)),
        ("mirror   NA 0.8 fraction, open bottom", ff.collection_fraction(mirror, F0, 0.8)),
    ]
    for label, value in rows:
        print(f"{label:58s} {value:.4g}")


if __name__ == "__main__":
    report()
