"""Lateral confinement of the dot (1.9 follow-on): a box dot, proven on exact answers.

The 1.9 well fills the box in x and y, so its only lateral confinement is the
hard box wall. ``exciton.dot_fill_fraction`` gives the dot its own lateral
size, with the host as the barrier around it. A box dot, not a cylinder,
because ``stark.py`` is a Cartesian 7-point grid: a box's faces lie on grid
planes, a cylinder's would be a staircase.

What is proven, each by MEASURED convergence order rather than a tolerance:

* the hard-wall box (the ``V0 -> inf`` limit of the dot), exact
  ``K pi^2 sum 1/L^2``;
* the separable finite box well ``V0 (3 - fx - fy - fz)``, exact as the sum of
  three 1-D finite-well-in-a-box roots (``stark.finite_well_in_box``);
* the dot's other limit, lateral side -> the box (the 1.9 model), exact as the
  1-D z well plus the hard lateral box;
* the non-separable dot itself, bracketed EXACTLY by two separable wells
  (``V0/3`` and ``V0``) and self-converging at second order.

All on the ``"cell"`` mask with faces on nodes. The 1.9 ``"node"`` mask is O(h)
and is pinned as such, because it is what moved the 1.9 verdict.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import dot_material as DM  # noqa: E402
import exciton as X  # noqa: E402
import stark as S  # noqa: E402

K = S.HBAR2_OVER_2ME_EV_NM2
M = 0.08                      # a plain test mass, not the dot's
L_BOX = 8.0                   # nm; with the 6 x 6 x 4 dot below, h = 1, 0.5, 0.25
GRIDS = (8, 16, 32)           # cells per axis, so every face lies on a node


def _orders(errs):
    return [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def _cube(n, L=L_BOX):
    return S.Box((-L / 2,) * 3, (L,) * 3, (n,) * 3)


def _ground(box, V, mass=M):
    e, v = S.lowest_states(box, mass, n_states=1, potential_eV=np.ravel(V),
                           maxiter=5000)
    return float(e[0]), v[:, 0]


# ------------------------------------------------------------ the primitives

def test_cell_fraction_preserves_the_well_width_on_any_grid():
    """sum(h f) is the well width exactly, however the faces sit on the grid.

    That is the property node sampling lacks (it is off by up to one cell) and
    the reason the cell mask is second order.
    """
    rng = np.random.default_rng(1)
    for _ in range(20):
        h = rng.uniform(0.1, 1.0)
        nodes = -10.0 + h * np.arange(int(20 / h) + 1)
        lo = rng.uniform(-6, -1)
        hi = rng.uniform(1, 6)
        f = X.axis_inside_fraction(nodes, h, lo, hi)
        assert h * f.sum() == pytest.approx(hi - lo, rel=1e-12)
        assert f.min() >= 0.0 and f.max() <= 1.0
    # a node exactly on a face is half in, half out
    f = X.axis_inside_fraction(np.array([-2.0, 0.0, 2.0]), 0.5, -2.0, 2.0)
    assert f.tolist() == [0.5, 1.0, 0.5]


def test_node_mask_without_lateral_is_exactly_the_1_9_potential():
    """The default path must not move a single 1.9 number."""
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    got = X._potential_grid(box, 0.1, 0.5, 4.0)
    xs, ys, zs = box.axes()
    Z = np.meshgrid(xs, ys, zs, indexing="ij")[2]
    z_mid = 0.5 * (zs[0] + zs[-1])
    outside = (Z < z_mid - 2.0) | (Z > z_mid + 2.0)
    want = -0.1 * Z + np.where(outside, 0.5, 0.0)
    assert np.array_equal(got, want)


def test_the_exact_1d_helper_agrees_with_the_infinite_barrier_root():
    """``finite_well_in_box`` with a far box is the 1.9 test's ``u tan u`` root.

    An independent formula (no box, no coth), so this checks the helper rather
    than restating it. And its two limits: a huge depth is the hard-wall well,
    and a vanishing well leaves the barrier everywhere, the bare box plus V0.
    """
    from scipy.optimize import brentq

    a, V0 = 2.0, 0.5
    v0 = V0 / (K / M) * a * a
    u = brentq(lambda u: u * math.tan(u) - math.sqrt(v0 - u * u), 1e-9, math.pi / 2 - 1e-9)
    E_inf = u * u * K / M / (a * a)
    E, _ = S.finite_well_in_box(V0, 2 * a, 60.0, M)
    assert E == pytest.approx(E_inf, rel=1e-10)

    E_deep, _ = S.finite_well_in_box(1e7, 4.0, 12.0, M)
    assert E_deep == pytest.approx(S.hard_box_ground_eV([4.0], M), rel=2e-3)
    E_thin, _ = S.finite_well_in_box(0.5, 1e-4, 12.0, M)
    assert E_thin == pytest.approx(S.hard_box_ground_eV([12.0], M) + 0.5, rel=1e-3)


# ------------------------------------------------------ exact limits, by order

def test_hard_wall_limit_of_the_dot_converges_at_second_order():
    """V0 -> inf: the dot becomes a hard-walled 6 x 6 x 4 nm box.

    The face nodes carry V0/2 and the outside V0; at V0 = 1e6 eV both are far
    above the hopping (t <= 0.9 eV at the finest grid), so they act as walls
    at the faces and the exact answer is ``K pi^2 (2/36 + 1/16) / m``.
    Measured orders 1.981 / 1.994 (errors 2.07e-2, 5.24e-3, 1.32e-3 eV).
    """
    exact = S.hard_box_ground_eV((6.0, 6.0, 4.0), M)
    errs = []
    for n in GRIDS:
        box = _cube(n)
        fill = X.dot_fill_fraction(box, 4.0, 6.0, "cell")
        E, _ = _ground(box, 1e6 * (1.0 - fill))
        errs.append(abs(E - exact))
    o = _orders(errs)
    assert 1.8 < min(o) and max(o) < 2.2, (o, errs)
    assert errs[-1] / exact < 1e-2, errs


def _separable_exact(V0, widths, L=L_BOX, mass=M):
    return sum(S.finite_well_in_box(V0, w, L, mass)[0] for w in widths)


def _separable_V(box, V0, widths):
    fs = []
    for nodes, h, w in zip(box.axes(), box.h(), widths):
        fs.append(X.axis_inside_fraction(nodes, h, -w / 2, w / 2))
    FX, FY, FZ = np.meshgrid(*fs, indexing="ij")
    return V0 * (3.0 - FX - FY - FZ)


def test_separable_finite_box_well_converges_at_second_order():
    """V0 (3 - fx - fy - fz) is exact as a sum of three 1-D roots.

    The 1-D roots carry the hard outer walls, so there is no tail floor: the
    error is the discretisation's alone. Measured orders 1.857 / 1.962 (errors
    1.34e-2, 3.71e-3, 9.52e-4 eV against the exact 0.29914 eV).
    """
    widths, V0 = (6.0, 6.0, 4.0), 0.5
    exact = _separable_exact(V0, widths)
    errs = []
    for n in GRIDS:
        box = _cube(n)
        E, _ = _ground(box, _separable_V(box, V0, widths))
        errs.append(abs(E - exact))
    o = _orders(errs)
    assert 1.8 < min(o) and max(o) < 2.2, (o, errs)
    assert errs[-1] / exact < 1e-2, errs


def test_lateral_fill_limit_is_the_1d_well_plus_the_hard_box():
    """Lateral side -> the box: the dot is the 1.9 model, exactly separable.

    ``well_lateral_nm=None`` fills x and y, so the exact ground is the 1-D z
    root plus the hard lateral box. Second order on the cell mask; measured
    1.847 / 1.960 (errors 5.08e-3, 1.41e-3, 3.63e-4 eV).
    """
    V0, h = 0.5, 4.0
    exact = (S.finite_well_in_box(V0, h, L_BOX, M)[0]
             + S.hard_box_ground_eV((L_BOX, L_BOX), M))
    errs = []
    for n in GRIDS:
        box = _cube(n)
        fill = X.dot_fill_fraction(box, h, None, "cell")
        E, _ = _ground(box, V0 * (1.0 - fill))
        errs.append(abs(E - exact))
    o = _orders(errs)
    assert 1.8 < min(o) and max(o) < 2.2, (o, errs)


# ------------------------------------------------ the non-separable dot itself

@pytest.fixture(scope="module")
def dot_levels():
    """The 6 x 6 x 4 nm, 0.5 eV dot's ground energy at 16, 32 and 64 cells.

    Shared, because the 64-cell solve is the slowest thing in this file
    (~12 s). The 8-cell grid (h = 1 nm, four cells across the dot height) is
    pre-asymptotic for the NON-separable dot -- measured p = 1.82 on 8/16/32
    against 1.95 on 16/32/64 -- so the dot's own order is taken one level up.
    """
    out = {}
    for n in (16, 32, 64):
        box = _cube(n)
        fill = X.dot_fill_fraction(box, 4.0, 6.0, "cell")
        out[n] = _ground(box, 0.5 * (1.0 - fill))[0]
    return out


def test_the_dot_is_bracketed_by_its_two_separable_wells(dot_levels):
    """E_sep(V0/3) <= E_dot <= E_sep(V0), on the grid and in the continuum.

    Outside a box dot V_dot = V0 max(ox, oy, oz) for the per-axis "outside"
    fractions, and ``max`` lies between the mean and the sum. Ordering of
    diagonal potentials orders the ground states (min-max), so the bracket is
    exact on the same grid AND its two ends are exactly known in the
    continuum. Checked both ways, so a dot mask that dropped its lateral faces
    (V_dot = V0 oz) falls below the lower end.

    The continuum check uses the Richardson value of the dot (32/64 cells,
    order 2), not a raw grid value: the dot sits only 1.4 meV under the upper
    end, and the raw 32-cell value is 0.9 meV above its own limit. Measured:
    grid 0.25398 < 0.29858 < 0.30010; continuum 0.25376 < 0.29770 < 0.29914.
    """
    widths, V0 = (6.0, 6.0, 4.0), 0.5
    box = _cube(32)
    E_dot = dot_levels[32]
    E_lo, _ = _ground(box, _separable_V(box, V0 / 3.0, widths))
    E_hi, _ = _ground(box, _separable_V(box, V0, widths))
    assert E_lo < E_dot < E_hi, (E_lo, E_dot, E_hi)
    E_inf = dot_levels[64] - (dot_levels[32] - dot_levels[64]) / 3.0
    exact_lo = _separable_exact(V0 / 3.0, widths)
    exact_hi = _separable_exact(V0, widths)
    assert exact_lo < E_inf < exact_hi, (exact_lo, E_inf, exact_hi)


def test_the_dot_self_converges_at_second_order(dot_levels):
    """No closed form for the dot, so its order is measured on itself.

    ``(E_16 - E_32) / (E_32 - E_64)`` is 2^p for an order-p scheme. Measured
    p = 1.95: the edges and corners of the box dot do not spoil the order.
    """
    E = [dot_levels[n] for n in (16, 32, 64)]
    p = math.log2((E[0] - E[1]) / (E[1] - E[2]))
    assert 1.8 < p < 2.2, (p, E)


# --------------------------------------------- the 1.9 mask, and what it moved

def _electron_z2(nz, mask, depth=0.8, height=4.0):
    m_e = DM.paper1_dot_at()[0].value
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (2, 2, nz))
    V = X._potential_grid(box, 0.0, depth, height, None, mask)
    _E, v = _ground(box, V, m_e)
    Z = np.meshgrid(*box.axes(), indexing="ij")[2]
    w = v ** 2 / np.sum(v ** 2)
    return float(np.dot(Z.ravel() ** 2, w))


def test_the_node_mask_widens_the_well_and_the_cell_mask_does_not():
    """The 1.9 verdict "electron not held at 0.8 eV" was the mask's width error.

    The 1.9 bench well: 4 nm, 0.8 eV, the dot electron, h_z = 0.4 nm. The
    exact ``<z^2>`` is 1.2574 nm^2, BELOW the 1.333 ceiling. The node mask
    counts both face nodes inside, a 4.4 nm well, and reads 1.391 (above the
    ceiling); it converges at first order (1.322 at h_z = 0.2). The cell mask
    reads 1.246 at h_z = 0.4.
    """
    m_e = DM.paper1_dot_at()[0].value
    _E, exact = S.finite_well_in_box(0.8, 4.0, 16.0, m_e)
    assert exact == pytest.approx(1.2574, abs=1e-4)
    node40, node80 = _electron_z2(40, "node"), _electron_z2(80, "node")
    cell40 = _electron_z2(40, "cell")
    assert node40 > 16.0 / 12.0 > exact          # the 1.9 reading, and the truth
    assert node40 - exact > node80 - exact > 0   # wider, converging from above
    assert 1.8 < (node40 - exact) / (node80 - exact) < 2.2   # first order
    assert abs(cell40 - exact) < 0.15 * (node40 - exact)


def test_lateral_confinement_does_not_tighten_the_electron_along_z():
    """The physics the follow-on asked about: lateral size does not bind along z.

    With a finite lateral barrier the part of the state outside the dot sees no
    z-well, so the effective z-depth is lower, not higher. Same cell mask, same
    z grid, 32 nm transverse box: the equal-area 15.95 nm dot raises the
    electron's energy and its ``<z^2>`` against the well that fills the box.
    """
    m_e = DM.paper1_dot_at()[0].value
    side = DM.DOT_DIAMETER_NM.value * math.sqrt(math.pi) / 2.0
    box = S.Box((-16.0, -16.0, -8.0), (32.0, 32.0, 16.0), (16, 16, 40))
    Z = np.meshgrid(*box.axes(), indexing="ij")[2].ravel()
    out = {}
    for lat in (None, side):
        V = X._potential_grid(box, 0.0, 0.8, 4.0, lat, "cell")
        E, v = _ground(box, V, m_e)
        w = v ** 2 / np.sum(v ** 2)
        out[lat] = (E, float(np.dot(Z ** 2, w)))
    assert out[side][0] > out[None][0]
    assert out[side][1] > out[None][1], out


# ------------------------------------------------------------------ refusals

def test_the_dot_refuses_what_it_cannot_represent():
    box = _cube(8)
    with pytest.raises(X.ExcitonError, match="strictly less"):
        X.dot_fill_fraction(box, 4.0, 12.0)       # the barrier would be the wall
    with pytest.raises(X.ExcitonError, match="positive"):
        X.dot_fill_fraction(box, 4.0, -1.0)
    with pytest.raises(X.ExcitonError, match="well_mask"):
        X.dot_fill_fraction(box, 4.0, 6.0, "nearest")
    with pytest.raises(X.ExcitonError, match="needs a well depth"):
        X._potential_grid(box, 0.0, None, 4.0, 6.0)
    with pytest.raises(S.StarkError, match="strictly inside"):
        S.finite_well_in_box(0.5, 12.0, 12.0, M)
