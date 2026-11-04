"""Tests for the effective-mass Stark solver (elec 1.8).

Every number here is checked against an analytic answer or an exact property
of the discretisation, never against a previously recorded output. Two cases
carry the proof:

* the harmonic oscillator, exact spectrum ``hbar*omega*(n + 1/2)``;
* the Airy problem, set by the zeros of the Airy function.

Both carry a CONVERGENCE test: the measured order is the evidence, so no
tolerance can be tuned to pass a wrong operator.

The electrostatic coupling is checked at two levels -- node self-consistency,
which a P1 interpolant must reproduce to rounding, and the parallel-plate
field on the axis, where the analytic value applies. Off-axis departure is
real fringing and is asserted as such, not tolerated away.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dockerfiles", "maxwell_solver"))

import mse_electrostatics as E  # noqa: E402
import mse_meshing as M  # noqa: E402
import stark as S  # noqa: E402


# ---------------------------------------------------------------- constants

def test_kinetic_constant_matches_the_physical_value():
    """hbar^2/2m_e in eV nm^2, computed from SI, is the known constant."""
    assert S.HBAR2_OVER_2ME_EV_NM2 == pytest.approx(0.0380998, rel=1e-5)


def test_airy_first_zero_matches_dlmf():
    assert S.AIRY_A1 == pytest.approx(2.33811, rel=1e-5)


def test_omega_convention_is_hbar_omega():
    assert S.hbar_omega_eV(20.0) == pytest.approx(0.020, rel=1e-15)


# ------------------------------------------------------- analytic: harmonic

def test_harmonic_spectrum_matches_hbar_omega_times_n_plus_half():
    """Levels sit within 2% of (n + 1/2) hbar omega.

    A box 7 oscillator lengths wide keeps the wall's push small, and the step
    resolves the ground state; the residual is the wall plus O(h^2).
    """
    n, h, V, exact = S.harmonic(n_states=3, half_width_nm=20.0,
                                omega_meV=20.0, n_cells=80)
    got, _ = S.lowest_states_1d(n, h, potential_eV=V, n_states=3)
    want = (np.arange(3) + 0.5) * 0.020
    assert np.allclose(got, want, rtol=2e-2), (got, want)


def test_harmonic_converges_at_second_order():
    """The MEASURED order is 2 at fixed physical extent.

    A tolerance cannot tell a correct second-order operator from a wrong one
    that happens to be close; the order can. Measured ~2.00 on this setup; the
    bound is 1.8 so the test pins the ORDER, not a run's third digit.
    """
    errors = []
    for n_cells in (160, 320, 640):
        n, h, V, _ = S.harmonic(n_states=1, half_width_nm=20.0,
                                omega_meV=20.0, n_cells=n_cells)
        got, _ = S.lowest_states_1d(n, h, potential_eV=V, n_states=1)
        errors.append(abs(got[0] - 0.5 * 0.020))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.8, orders


def test_finite_square_well_recovers_the_bound_state():
    """A finite square well's ground state matches the transcendental root.

    This is the "case with a known answer" the confinement validation stands
    on. The bound-state energy solves, in dimensionless ``u = k a`` with
    ``v0 = V0 a^2 / (K/m)``,

        u tan u = sqrt(v0 - u^2)        (even / ground state)

    and ``E = u^2 K / (m a^2)``. The well is narrower than the box so that the
    outer barrier is a real wall, not the Dirichlet boundary (with ``a`` equal
    to the box half-width the mask covers the whole grid, the barrier lands on
    the walls where its value is unused, and the solver silently reports the
    INFINITE-well energy -- measured 0.0734434 against the true 0.0471547 --
    which is how that mistake was caught).

    The convergence is O(h), not O(h^2), and that is correct for this problem:
    a node-sampled step sits between grid points, so the effective well width
    carries an O(h) uncertainty. Measured orders 1.16/1.09/1.05/1.04.
    """
    import math
    from scipy.optimize import brentq

    K = S.HBAR2_OVER_2ME_EV_NM2
    mass_ratio, a, V0 = 0.08, 2.0, 0.5
    v0 = V0 / (K / mass_ratio) * a**2
    u = brentq(lambda u: u * math.tan(u) - math.sqrt(max(v0 - u * u, 0.0)),
               1e-9, math.pi / 2 - 1e-9)
    exact = u * u * K / mass_ratio / a**2

    def solve(n_cells):
        h = 16.0 / n_cells
        zs = np.linspace(-8.0, 8.0, n_cells + 1)
        V = np.where(np.abs(zs) < a, 0.0, V0)
        e, _ = S.lowest_states_1d(n_cells + 1, h, mass_ratio=mass_ratio,
                                  potential_eV=V, n_states=1)
        return float(e[0])

    errs = [abs(solve(n) - exact) for n in (127, 255, 511)]
    assert errs[-1] < 2e-3, errs
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]

    # FIRST order, not second, and that is the correct answer here, not a
    # defect. The potential step is sampled AT THE NODES: whether the node at
    # |z| = a falls on the near or the far side of the step flips with h, so
    # the well the discrete problem actually solves has a width uncertain by
    # up to one cell, and dE/da is nonzero. That costs one order. The smooth
    # (harmonic) and hard-wall cases in this file converge at 2 and pin the
    # difference; see test_laplacian... and the infinite-well test.
    #
    # Measured here: 1.09, 1.05, 1.04. The bound is set to catch drift
    # (towards O(1), or a sudden jump), NOT to pass on a technicality -- the
    # previous bound of 0.9 would have passed 0.91, which pins nothing.
    assert 1.0 < min(orders) and max(orders) < 1.5, (orders, errs)


# ----------------------------------------------------------- analytic: Airy

def test_airy_ground_state_matches_the_airy_zero():
    """E_0 = (K F^2)^(1/3) |a_1| to better than 1%.

    A different analytic family from the oscillator: linear potential against a
    hard wall, with a turning point the grid must resolve.
    """
    n, h, V, exact = S.airy(field_eV_per_nm=0.05, extent_nm=120.0, n_cells=480)
    got, _ = S.lowest_states_1d(n, h, potential_eV=V, n_states=1)
    assert exact[0] == pytest.approx(0.1067789, rel=1e-5)
    assert got[0] == pytest.approx(exact[0], rel=1e-2), (got[0], exact[0])


def test_airy_converges_at_second_order():
    errors = []
    for n_cells in (240, 480, 960):
        n, h, V, exact = S.airy(field_eV_per_nm=0.05, extent_nm=120.0,
                                n_cells=n_cells)
        got, _ = S.lowest_states_1d(n, h, potential_eV=V, n_states=1)
        errors.append(abs(got[0] - exact[0]))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.8, orders


# ------------------------------------------------------- operator properties

def test_laplacian_is_positive_semidefinite():
    """A Laplacian must have NO negative eigenvalue.

    The first version of this module had the DIAGONAL's sign flipped, giving a
    spectrum from -2t to 0: every energy negative, which is impossible for a
    kinetic operator alone.

    Reproduced 2026-11-04 against THIS commit: setting ``main = -2.0 * t`` in
    ``laplacian_1d_eV_nm2`` (stark.py:366, the 1-D operator these tests use --
    NOT the 3-D ``build_laplacian_eV_nm2``) fails 7 of 16, including all four
    analytic tests. Note the 1-D operator is a separate function from the 3-D
    builder; mutating the 3-D diagonal does nothing to these tests, which is why
    the first three attempts at this mutation appeared to change nothing.

    (An earlier version of this docstring said the analytic tests' tolerances
    were loose enough to let it through. That is false at this commit. What I
    had seen was a looser intermediate ``harmonic`` on a coarse box, not these
    tests. Corrected 2026-11-04.)
    """
    L = S.laplacian_1d_eV_nm2(21, 0.5)
    ev = np.linalg.eigvalsh(L.toarray())
    assert ev.min() > -1e-12, ev.min()
    assert ev.max() > 0


def test_zero_potential_recovers_the_particle_in_a_box():
    """With V = 0 the ground state is K (pi / L)^2.

    NOT pi^2 K / (2 L^2): ``K`` is already ``hbar^2 / 2m``, so the half is
    inside it. Writing the extra half gives exactly one half of the true
    energy -- a ratio of 2.0000 that does not move with the grid, which is how
    this was caught: a discretisation error would converge away, a wrong
    formula will not. Measured ratio 0.99998.
    """
    extent, n_cells = 20.0, 200
    n = n_cells + 1
    h = extent / n_cells
    got, _ = S.lowest_states_1d(n, h, potential_eV=np.zeros(n), n_states=1)
    want = S.HBAR2_OVER_2ME_EV_NM2 * (np.pi / extent) ** 2
    assert got[0] == pytest.approx(want, rel=2e-3), (got[0], want)


def test_a_deep_well_actually_confines_the_state():
    """A deep well must localise the state; a bare box must not.

    The trap this pins: ``exciton.sawtooth_potential`` used to return ``-F z``
    only, so the "--well" case was a free particle in the box and its
    ``<z^2>`` was the bare-box ``L^2 (1/12 - 1/(2 pi^2))`` to five digits
    (95.27 nm^2 for L = 40 nm) while the docstrings called it a quantum dot.
    An energy comparison alone does not catch that: the bare box has a
    perfectly finite ground state. Only the WIDTH does.

    So this test asserts the confined state is much narrower than the bare
    box, and that the bare box reproduces the analytic width. If the well is
    silently dropped, the first assertion fails with <z^2> == 95.27, which is
    the exact value the second assertion pins.
    """
    n_cells = 40
    extent = 40.0
    box = S.Box((0.0, 0.0, -extent / 2), (12.0, 12.0, extent),
                (8, 8, n_cells))
    xs, ys, zs = box.axes()
    _x, _y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    def z_squared(potential):
        _e, vec = S.lowest_states(box, 0.08, n_states=1, potential_eV=potential)
        dens = vec[:, 0] ** 2
        dens = dens / dens.sum()
        return float(np.dot(Z.reshape(-1) ** 2, dens))

    bare = z_squared(None)
    widest = extent**2 * (1.0 / 12.0 - 1.0 / (2.0 * np.pi**2))
    assert bare == pytest.approx(widest, rel=5e-3), (bare, widest)

    well = np.where(np.abs(Z) > 2.0, 0.6, 0.0)
    confined = z_squared(well)
    assert confined < 0.05 * bare, (confined, bare)


def test_walls_are_dirichlet():
    """The returned vectors are zero at both ends by construction."""
    n, h, V, _ = S.harmonic(n_states=2, half_width_nm=10.0, n_cells=40)
    _e, vec = S.lowest_states_1d(n, h, potential_eV=V, n_states=2)
    assert np.allclose(vec[0], 0.0)
    assert np.allclose(vec[-1], 0.0)


# ------------------------------------------------- electrostatics coupling

def _slab_solution(extent_m=4e-7, n_modes=8):
    """A box with the top and bottom 0.6 um held at 1 V and 0 V."""
    import trimesh
    box = trimesh.creation.box(extents=(extent_m, extent_m, extent_m))
    pts, tets, _ = M.tetrahedralize(box, "m", n_modes=n_modes, verbose=False)
    zc = pts[tets].mean(axis=1)[:, 2]
    zb = pts[:, 2].min() + 0.6e-7
    zt = pts[:, 2].max() - 0.6e-7
    rid = np.where(zc < zb, 0, np.where(zc > zt, 2, 1))
    sol = E.solve(pts, tets, ["bot", "body", "top"], rid,
                  {"bot": 1.0, "body": 4.0, "top": 1.0},
                  {"bot": 0.0, "top": 1.0})
    return sol, zb, zt


def test_interpolation_reproduces_the_nodal_potential_exactly():
    """At a mesh node the P1 interpolant returns that node's own value.

    This is the self-consistency the coupling rests on: pick the wrong tet
    anywhere and some node comes back with a neighbour's potential. Measured
    2.3e-15 relative.
    """
    sol, _zb, _zt = _slab_solution()
    got = S.interpolate_potential(sol, sol["points_m"] * 1e9)
    assert np.abs(got - sol["phi"]).max() < 1e-12


def test_interpolation_refuses_points_outside_the_mesh():
    """Outside is an ERROR, not a nearest-tet answer.

    A silent nearest-tet value is a wrong number with no warning, and the
    uniform-field check would then be testing an artifact.
    """
    sol, _zb, _zt = _slab_solution()
    with pytest.raises(S.StarkError, match="outside the mesh"):
        S.interpolate_potential(sol, np.array([[0.0, 0.0, 500.0]]))


def test_field_approaches_the_parallel_plate_value_under_refinement():
    """The axis field converges TOWARD -1/d as the mesh refines.

    It does NOT sit on the 1-D value, and the gap is not solver error: a 4 um
    box with a 0.6 um gap is not a parallel-plate capacitor, and the fringing
    field converges inward as the mesh resolves it. Measured error 14.28% ->
    9.96% -> 8.79% at n_modes 8/16/30.

    The evidence that this is fringing and not a bad operator is that the error
    FALLS MONOTONICALLY UNDER REFINEMENT at fixed geometry. A defective operator
    would not improve as the mesh resolves.

    Do NOT claim "a narrower box is worse" -- and do not claim the reverse.
    Neither is measured here, and both have been asserted in this docstring at
    different times.

    Note what _slab_solution actually does before comparing across box sizes:
    it holds the top and bottom 0.6 um at fixed potential (see its docstring),
    so the gap is extent_m - 1.2 um and it CHANGES WITH THE BOX. Measured
    gaps: 1.2 um box -> 1.08 um, 4 um box -> 3.88 um. Any "fixed gap" cross-box
    comparison is therefore measuring two different geometries, not two box
    sizes. An earlier pair here (1.2 um 8.98% vs 4 um 10.54%) was labelled
    "fixed 0.2 um gap" and was wrong on both counts -- the electrode was 0.6 um
    -- and has been removed rather than replaced with another unverified pair.

    Nothing is claimed here about box size in either direction, and no refusal
    is quoted to prop that up. An earlier version of this note said the helper
    refuses an overlapping configuration at extent_m=1.2e-7 and called that "the
    1.2 um box"; 1.2e-7 m is 0.12 um, and the refusal is gone by 0.20 um. The
    sentence was mis-sized by a factor of ten in the same way as the pair it
    replaced. The honest position needs no boundary: box size is simply not
    measured here.

    The test therefore asserts the convergence DIRECTION and a bound, because
    a tolerance on a single value cannot distinguish the two explanations.
    """
    errors = []
    for n_modes in (8, 16, 30):
        sol, zb, zt = _slab_solution(n_modes=n_modes)
        zm = 0.5 * (zb + zt) * 1e9
        grid = np.stack(np.meshgrid([0.0], [0.0], [zm], indexing="ij"), axis=-1)
        ez = S.field_on_grid(sol, grid)[0, 0, 0, 2]
        want = -1.0 / (zt - zb)
        errors.append(abs(ez - want) / abs(want))
    assert errors[1] < errors[0], errors
    assert errors[2] < errors[1], errors
    assert errors[2] < 0.12, errors


def test_transverse_field_is_small_on_the_axis():
    """Rotational symmetry forces E_x = E_y = 0 on the axis."""
    sol, zb, zt = _slab_solution()
    zm = 0.5 * (zb + zt) * 1e9
    grid = np.stack(np.meshgrid([0.0], [0.0], [zm], indexing="ij"), axis=-1)
    Efield = S.field_on_grid(sol, grid)
    Ez = abs(Efield[0, 0, 0, 2])
    assert abs(Efield[0, 0, 0, 0]) < 0.05 * Ez
    assert abs(Efield[0, 0, 0, 1]) < 0.05 * Ez


def test_field_grid_refuses_points_outside_the_mesh():
    sol, _zb, _zt = _slab_solution()
    grid = np.stack(np.meshgrid([0.0], [0.0], [500.0], indexing="ij"), axis=-1)
    with pytest.raises(S.StarkError, match="outside the mesh"):
        S.field_on_grid(sol, grid)


def test_field_from_a_tet_solve_feeds_airy_levels_end_to_end():
    """The coupling 1.8 exists for: electrostatics in, spectrum out.

    The field comes FROM the solve rather than being supplied, so a sign or
    unit error in the extraction shows up as a spectrum that is not Airy.

    The conversion is ``F[eV/nm] = F[V/m] * 1e-9`` and nothing else: a potential
    difference of 1 V across 1 nm IS 1 eV/nm, because the electron charge
    cancels between the volt and the electron-volt. Dividing by ``e`` as well
    (the obvious-looking mistake) multiplies the field by 6.2e18 and produces a
    spectrum in the millions of eV, which is how this was caught.
    """
    sol, zb, zt = _slab_solution()
    zm = 0.5 * (zb + zt) * 1e9
    grid = np.stack(np.meshgrid([0.0], [0.0], [zm], indexing="ij"), axis=-1)
    # Keep the SIGN: E_z must point DOWN (top held at +1 V), so this is
    # negative, and the Airy case needs a positive confining field. Taking abs()
    # here would let a sign error in field_on_grid through -- a mutation that
    # flipped the field's sign was caught by only ONE test until this line
    # stopped discarding it.
    E_z_v_m = S.field_on_grid(sol, grid)[0, 0, 0, 2]
    assert E_z_v_m < 0.0, E_z_v_m
    field_eV_per_nm = -E_z_v_m * 1e-9
    n, h, V, exact = S.airy(field_eV_per_nm=field_eV_per_nm,
                            extent_nm=120.0, n_cells=960)
    got, _ = S.lowest_states_1d(n, h, potential_eV=V, n_states=1)
    assert got[0] == pytest.approx(exact[0], rel=2e-3), (got[0], exact[0])


# ------------------------------------------------ silent-failure guardrails


def test_unconverged_lobpcg_raises_instead_of_returning():
    """A solve that hits the iteration cap must NOT hand back a number.

    scipy's lobpcg reports a missed tolerance only through a UserWarning and
    then returns whatever it has; a caller that does not read warnings cannot
    tell that result from a converged one. That happened on this change's own
    sweep (12 of 159 electron solves), which is why lowest_states now measures
    the residual itself.

    ``maxiter=3`` cannot converge a matrix of this size, so this is a forced
    failure, not a lucky one.
    """
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    xs, ys, zs = box.axes()
    Z = np.meshgrid(xs, ys, zs, indexing="ij")[2]
    V = np.where(np.abs(Z) < 2.0, 0.0, 0.2)

    with pytest.raises(S.StarkError, match="did not converge"):
        S.lowest_states(box, 0.0684, n_states=1, potential_eV=V, maxiter=3)

    # The reporting form returns the same verdict instead of raising, so a
    # caller can record it rather than abort.
    _e, _v, converged = S.lowest_states(box, 0.0684, n_states=1,
                                        potential_eV=V, maxiter=3,
                                        return_converged=True)
    assert converged.tolist() == [False]


def test_converged_lobpcg_is_reported_converged():
    """The guard must not fire on a good solve -- otherwise it is just a bug
    that rejects correct answers, which is the worse failure mode."""
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    xs, ys, zs = box.axes()
    Z = np.meshgrid(xs, ys, zs, indexing="ij")[2]
    V = np.where(np.abs(Z) < 2.0, 0.0, 0.2)
    _e, _v, converged = S.lowest_states(box, 0.0684, n_states=1,
                                        potential_eV=V, return_converged=True)
    assert converged.tolist() == [True]


def test_lowest_states_emits_no_lobpcg_warning_to_callers():
    """The residual check replaces the warning; it must not leak the raw one
    to a caller that would reasonably mistake it for the check."""
    import warnings as _w
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    xs, ys, zs = box.axes()
    Z = np.meshgrid(xs, ys, zs, indexing="ij")[2]
    V = np.where(np.abs(Z) < 2.0, 0.0, 0.2)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        S.lowest_states(box, 0.0684, n_states=1, potential_eV=V, maxiter=3,
                        return_converged=True)
    msgs = [str(c.message) for c in caught]
    assert not any("tolerance" in m for m in msgs), msgs
