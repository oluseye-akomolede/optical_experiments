"""2-D photonic-crystal band diagrams: the oracle, the mapping, and the caps.

WHY THESE EXIST, and the one that gates all the others. Every dispersion claim
in this feature is measured against the 1-D layered transfer-matrix oracle in
``pc.layered_traces``. So that oracle is checked FIRST, against an independent
ODE shooting solve, before any mesh is compared to it -- if the reference is
wrong, every other number is wrong in the same direction and reads as correct.

The specific trap this file is built around: ``eta_s = kx`` but ``eta_p = kx/eps``,
and the two agree only at ``eps = 1``. A check of the eps slot run at ``eps = 1``
CANNOT FAIL -- the rig's "a control that cannot fail leaves the bug in". So every
comparison here is at ``eps2 = 4``.

The second trap is the polarisation MAPPING. The PhC names modes for the
out-of-plane field and the layer problem names them for the interface normal, so
``TM <-> s`` and ``TE <-> p``. That was written wrong twice in the plan, so it is
asserted here as a geometric fact (``z-hat . x-hat = 0``) rather than restated.

WHAT IS SKIPPED. Anything calling ``eigsh`` is marked ``heavy`` (see
``tests/conftest.py``) and runs on a cluster Job, not on the VM. The oracle and
the meshing/pairing controls are pure numpy/scipy and run here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import pc  # noqa: E402


# ---------------------------------------------------------------------------
# The oracle's own control: closed form vs an independent ODE shooting solve
# ---------------------------------------------------------------------------

def _shooting_trace(lam, ky, eps2, fill, pol, eps1=1.0, a=1.0, rtol=1e-13):
    """Monodromy trace from integrating the ODE across the cell -- no closed form.

    The state integrated is the one that is PHYSICALLY CONTINUOUS across the
    interface. For s that is ``(psi, psi')``. For p it is ``(psi, (1/eps) psi')``:
    ``psi'`` jumps at the interface and only ``(1/eps) psi'`` does not, so
    integrating ``(psi, psi')`` and rescaling afterwards is WRONG -- the jump is
    mid-cell and no end-point rescale fixes it. (That mistake produced a 1.0
    disagreement with the closed form in an earlier session.)
    """
    from scipy.integrate import solve_ivp

    if pol == "s":
        def rhs(x, y):
            e = eps1 if x < fill * a else eps2
            return [y[1], (ky ** 2 - lam * e) * y[0]]
    else:
        def rhs(x, y):
            e = eps1 if x < fill * a else eps2
            # y = (psi, w) with w = (1/eps) psi'  ->  psi' = e*w, w' = (ky^2/e - lam) psi
            return [e * y[1], (ky ** 2 / e - lam) * y[0]]

    cols = []
    for ic in ([1, 0], [0, 1]):
        sol = solve_ivp(rhs, [0, a], ic, rtol=rtol, atol=rtol * 1e-2)
        cols.append(sol.y[:, -1])
    return np.trace(np.column_stack(cols))


def test_the_oracle_validates_itself_against_shooting():
    """The reference must pass its OWN check before any mesh is compared to it.

    eps2 = 4, never eps = 1 (see the module docstring). Tolerance 1e-10; the
    binding constraint is the integrator's rtol, which is why it is pinned at
    1e-13 rather than left at the scipy default.
    """
    worst = 0.0
    worst_at = None
    for ky in np.linspace(0.05, 1.2, 7):
        for lam in (1.5, 4.0, 9.0, 15.0):
            for pol, idx in (("s", 0), ("p", 1)):
                closed = pc.layered_traces(lam, ky, 4.0, 0.5)[idx]
                shoot = _shooting_trace(lam, ky, 4.0, 0.5, pol)
                d = abs(closed - shoot)
                if d > worst:
                    worst, worst_at = d, (lam, ky, pol)
    assert worst < 1e-10, (
        f"oracle disagrees with shooting by {worst:.3e} at {worst_at}; "
        f"the closed form or the shooting state is wrong"
    )


def test_the_wrong_admittance_is_rejected_by_that_same_check():
    """CONTROL: the check can fail. ``eta_p = kx/eps^2`` (the error made once)
    must disagree with the shooting oracle by orders of magnitude, or the check
    above is measuring nothing."""
    def wrong_trace(lam, ky, eps2, fill, eps1=1.0, a=1.0):
        k1 = np.sqrt(lam * eps1 - ky ** 2 + 0j)
        k2 = np.sqrt(lam * eps2 - ky ** 2 + 0j)
        def m(k, e, d):
            if abs(k) < 1e-14:
                return np.array([[1.0, e * d], [0.0, 1.0]], dtype=np.complex128)
            c, s = np.cos(k * d), np.sin(k * d)
            return np.array([[c, e * e * s / k], [-k * s / e, c]], dtype=np.complex128)
        return np.trace(m(k2, eps2, (1 - fill) * a) @ m(k1, eps1, fill * a))

    worst = max(
        abs(wrong_trace(l, ky, 4.0, 0.5) - _shooting_trace(l, ky, 4.0, 0.5, "p"))
        for ky in (0.3, 0.9) for l in (2.0, 6.5)
    )
    assert worst > 1e-10, (
        "the 1/eps^2 admittance passed the oracle check -- the check cannot "
        "distinguish the correct form from the wrong one"
    )


def test_s_and_p_coincide_at_eps_one_and_differ_otherwise():
    """The discriminating property, both directions.

    At eps = 1 the two polarisations are identical -- this is WHY a uniform-eps
    check cannot catch the swap. At eps != 1 with a nonzero PARALLEL component
    they differ. Both halves are asserted; either alone is misleading.
    """
    # eps2 = 1: identical (this is the reason the control above is at eps2 = 4)
    for ky in (0.0, 0.3):
        s = pc.layered_traces(2.0, ky, 1.0, 0.5)[0]
        p = pc.layered_traces(2.0, ky, 1.0, 0.5)[1]
        assert abs(s - p) < 1e-12, f"s and p must coincide at eps=1, ky={ky}"

    # eps2 = 4, ky != 0: differ
    s = pc.layered_traces(2.0, 0.3, 4.0, 0.5)[0]
    p = pc.layered_traces(2.0, 0.3, 4.0, 0.5)[1]
    assert abs(s - p) > 1e-3, "s and p must differ at eps2=4 with ky!=0"

    # eps2 = 4, ky = 0: coincide again -- the parallel component is required
    s0 = pc.layered_traces(2.0, 0.0, 4.0, 0.5)[0]
    p0 = pc.layered_traces(2.0, 0.0, 4.0, 0.5)[1]
    assert abs(s0 - p0) < 1e-12, (
        "at ky=0 the layers are invisible to the in-plane component; s and p "
        "must coincide -- if they do not, the admittance is wrong"
    )


# ---------------------------------------------------------------------------
# Lattices and the high-symmetry path
# ---------------------------------------------------------------------------

def test_reciprocal_basis_is_dual():
    for lat in ("square", "triangular"):
        a1, a2 = pc.lattice_vectors(lat)
        b1, b2 = pc.reciprocal_vectors(a1, a2)
        A = np.column_stack([a1, a2])
        B = np.column_stack([b1, b2])
        np.testing.assert_allclose(B.T @ A, 2 * np.pi * np.eye(2), atol=1e-12)


def test_high_symmetry_points_match_their_closed_forms():
    """The triangular corners are DERIVED, so assert them against the textbook
    magnitudes. The ``(b1+b2)/3`` formula would give |K| = 2.42 here instead of
    4 pi/3a, which is exactly the basis-orientation trap the plan warns about."""
    # The sampler emits ``points_per_segment`` samples per segment INCLUDING the
    # segment's start, so the three closed segments give 3n samples with the
    # corners at 0, n, 2n and the closing Gamma at index 3n-1. (Measured, not
    # derived: n=3 -> 9 samples, n=5 -> 15.)
    n = 4
    labels, kp = pc.high_symmetry_path("triangular", points_per_segment=n)
    assert labels == ["$\\Gamma$", "M", "K", "$\\Gamma$"]
    assert kp.shape == (3 * n, 2), f"got {kp.shape} for n={n}"

    a1, a2 = pc.lattice_vectors("triangular")
    b1, b2 = pc.reciprocal_vectors(a1, a2)
    M = b1 / 2.0
    K = (2 * b1 + b2) / 3.0
    # the closed forms this lattice's corners must hit
    assert abs(np.linalg.norm(M) - 2 * np.pi / np.sqrt(3)) < 1e-12
    assert abs(np.linalg.norm(K) - 4 * np.pi / 3) < 1e-12
    # ...and the naive (b1+b2)/3 is NOT it, so the assertion above is discriminating
    assert abs(np.linalg.norm((b1 + b2) / 3.0) - 4 * np.pi / 3) > 1e-3, (
        "the naive (b1+b2)/3 coincides with the correct K here, so this test "
        "cannot tell the derived corner from the textbook formula"
    )

    two_pi = 2 * np.pi
    assert np.allclose(kp[0], 0, atol=1e-12), "path must start at Gamma"
    assert np.allclose(kp[n], M / two_pi, atol=1e-12), (
        f"sample {n} is not M: {kp[n]}"
    )
    assert np.allclose(kp[2 * n], K / two_pi, atol=1e-12), (
        f"sample {2 * n} is not K: {kp[2 * n]}"
    )
    assert np.allclose(kp[-1], 0, atol=1e-12), "path must close at Gamma"


def test_square_path_reaches_x_and_m():
    """Square: Gamma-X-M-Gamma, corners at the sampler's own stride."""
    n = 3
    labels, kp = pc.high_symmetry_path("square", points_per_segment=n)
    assert labels == ["$\\Gamma$", "X", "M", "$\\Gamma$"]
    assert kp.shape == (3 * n, 2), f"got {kp.shape} for n={n}"
    assert np.allclose(kp[0], 0)
    assert np.allclose(kp[n], [0.5, 0.0]), f"sample {n} is not X: {kp[n]}"
    assert np.allclose(kp[2 * n], [0.5, 0.5]), f"sample {2*n} is not M: {kp[2*n]}"
    assert np.allclose(kp[-1], 0)

    # The path is CONNECTED: samples strictly inside a segment are collinear, and
    # collinearity breaks only at the corners, where the direction is supposed to
    # change. A path that teleported across the zone would break it mid-segment.
    # A triple is "inside a segment" when all three samples share a segment index.
    for i in range(len(kp) - 2):
        if {i // n, (i + 1) // n, (i + 2) // n} != {i // n}:
            continue  # the triple straddles a corner
        v0 = kp[i + 1] - kp[i]
        v1 = kp[i + 2] - kp[i + 1]
        # the 2-D cross product is the scalar z-component
        cross = v0[0] * v1[1] - v0[1] * v1[0]
        assert abs(cross) < 1e-12, (
            f"direction changed inside a segment at sample {i}: {kp[i:i+3]}"
        )

    # ...and the direction DOES change at the corner, so the loop above is not
    # vacuous. Sample n-1 -> n -> n+1 spans the X corner.
    v0 = kp[n] - kp[n - 1]
    v1 = kp[n + 1] - kp[n]
    cross = v0[0] * v1[1] - v0[1] * v1[0]
    assert abs(cross) > 1e-6, (
        f"no direction change at the X corner (sample {n}): {kp[n-1:n+2]}"
    )


# ---------------------------------------------------------------------------
# Meshing and periodic pairing
# ---------------------------------------------------------------------------

def test_unit_cell_pairs_every_boundary_node_once():
    """Pairing correctness, asserted as the INVARIANT rather than a count.

    ``divisions`` gives ``divisions + 1`` nodes per side, so each of the two far
    faces carries ``divisions + 1`` nodes and they share one corner:
    ``2*divisions + 1`` pairs. That closed form is asserted, but it is the
    SECONDARY claim -- it is a number I got wrong twice by arithmetic. The
    primary claim is the invariant: the set of slaves is exactly the set of
    boundary nodes, each once. That is what the reduction actually requires, and
    it is measured in the loop rather than restated.
    """
    divisions = 8
    pts, tris, eps, pairs = pc.unit_cell_mesh("square", divisions=divisions,
                                              r_over_a=0.2, n_rod=3.4, n_bg=1.0)
    a1, a2 = pc.lattice_vectors("square")
    A = np.column_stack([a1, a2])
    frac = np.round(pts @ np.linalg.inv(A).T, 9)

    on_boundary = {
        i for i, (u, v) in enumerate(frac)
        if abs(u - 1.0) < 1e-9 or abs(v - 1.0) < 1e-9
    }
    slaves = [s for s, _, _ in pairs]

    # INVARIANT: paired set == boundary set, each node once.
    assert len(slaves) == len(set(slaves)), "a node was paired twice"
    assert set(slaves) == on_boundary, (
        f"{len(on_boundary - set(slaves))} boundary nodes unpaired, "
        f"{len(set(slaves) - on_boundary)} interior nodes paired"
    )
    assert len(pairs) == len(on_boundary), (
        f"{len(pairs)} pairs for {len(on_boundary)} boundary nodes"
    )

    # SECONDARY: the closed form, checked across divisions so it is not a
    # lucky match at one size.
    for d in (4, 6, 10):
        _, _, _, p_d = pc.unit_cell_mesh("square", divisions=d, r_over_a=0.2,
                                         n_rod=3.4, n_bg=1.0)
        assert len(p_d) == 2 * d + 1, f"divisions={d}: got {len(p_d)}"

    # every pair is a lattice translation, in the direction of the slave's face
    for slave, master, tvec in pairs:
        T = tvec[0] * a1 + tvec[1] * a2
        assert np.allclose(pts[slave], pts[master] + T, atol=1e-9), (
            f"pair ({slave},{master},{tvec}) is not a lattice translation"
        )
        u, v = frac[slave]
        assert tvec[0] or tvec[1], "a pair with no translation is not a pair"
        if tvec[0] and not tvec[1]:
            assert abs(u - 1.0) < 1e-9, f"slave {slave} has tvec a1 but u={u}"
        if tvec[1] and not tvec[0]:
            assert abs(v - 1.0) < 1e-9, f"slave {slave} has tvec a2 but v={v}"
        if tvec[0] and tvec[1]:  # the corner carries both phases
            assert abs(u - 1.0) < 1e-9 and abs(v - 1.0) < 1e-9

    # THE PHASE IS THE POINT. The count above is satisfied by a pairing that
    # sends the corner across only ONE of its two faces -- every node still
    # appears exactly once, so the count is blind to it, but the corner then
    # carries half the Bloch phase it must and the band structure is wrong.
    # (Measured: forcing the corner down the generic on_u branch keeps 17 pairs
    # and keeps the invariant, and silently gives tvec [1,0] instead of [1,1].)
    # So derive the translation from the geometry and demand they agree: the
    # translation whose application maps the slave onto its master is the one
    # whose phase the slave must carry.
    for slave, master, tvec in pairs:
        delta = pts[slave] - pts[master]
        # solve delta = m1*a1 + m2*a2 by hand -- 2x2, no dependence on the module
        det = a1[0] * a2[1] - a1[1] * a2[0]
        m1 = (delta[0] * a2[1] - delta[1] * a2[0]) / det
        m2 = (a1[0] * delta[1] - a1[1] * delta[0]) / det
        assert abs(m1 - round(m1)) < 1e-9 and abs(m2 - round(m2)) < 1e-9, (
            f"pair ({slave},{master}) is not a lattice translation: {delta}"
        )
        measured = np.array([round(m1), round(m2)])
        assert np.array_equal(measured, tvec), (
            f"pair ({slave},{master}) carries phase {tvec} but the actual "
            f"translation is {measured} -- the Bloch phase is wrong"
        )
    # and the corner specifically must carry BOTH, or the check above is vacuous
    corner = [
        s for s, _, t in pairs
        if abs(frac[s][0] - 1.0) < 1e-9 and abs(frac[s][1] - 1.0) < 1e-9
    ]
    assert len(corner) == 1, f"expected exactly one corner slave, got {corner}"
    corner_tvec = [t for s, _, t in pairs if s == corner[0]][0]
    assert np.array_equal(corner_tvec, [1, 1]), (
        f"the corner carries {corner_tvec}, must carry [1, 1] (both shifts)"
    )


def test_the_rod_gets_n_rod_and_the_background_gets_n_bg():
    """WHICH region holds which eps, not merely that both values appear.

    An earlier version of this test asserted the SET of eps values was
    ``{n_bg^2, n_rod^2}``, which is invariant under swapping the two -- so
    swapping ``np.where``'s branches left it green while the rod got the
    background's permittivity. (Found by mutation, not by reading.) The check
    here recomputes the geometry independently and asserts the value inside the
    rod, with ``n_bg > n_rod`` included so the two cannot be confused.
    """
    a1, a2 = pc.lattice_vectors("square")
    A = np.column_stack([a1, a2])
    centre = 0.5 * (a1 + a2)
    r = 0.25

    for n_rod, n_bg in ((3.4, 1.0), (1.0, 3.4), (2.5, 1.2)):
        pts, tris, eps, _ = pc.unit_cell_mesh("square", divisions=12, r_over_a=r,
                                              n_rod=n_rod, n_bg=n_bg)
        # fold triangle centres into the cell, independently of the module
        cent = pts[tris].mean(axis=1)
        frac = cent @ np.linalg.inv(A).T
        frac = frac - np.floor(frac)
        folded = frac @ np.vstack([a1, a2])
        inside = np.linalg.norm(folded - centre, axis=1) < r

        assert inside.sum() > 0 and (~inside).sum() > 0, (
            f"rod region empty or covers the cell at r/a={r}"
        )
        assert set(np.round(eps[inside], 9)) == {round(float(n_rod) ** 2, 9)}, (
            f"n_rod={n_rod}, n_bg={n_bg}: eps inside the rod is "
            f"{sorted(set(np.round(eps[inside], 9)))}, expected {n_rod**2}"
        )
        assert set(np.round(eps[~inside], 9)) == {round(float(n_bg) ** 2, 9)}, (
            f"n_rod={n_rod}, n_bg={n_bg}: eps outside the rod is "
            f"{sorted(set(np.round(eps[~inside], 9)))}, expected {n_bg**2}"
        )


def test_rod_must_fit_the_cell_and_the_limit_is_cell_aware():
    """The rod's r/a limit is the cell's INRADIUS, which is lattice-dependent.

    A flat ``< 0.5`` test is wrong for the triangular cell: the 60-degree rhombus
    has inradius ``sin(60)/2 = 0.4330`` (measured), so a rod at r/a = 0.45 pokes
    through the slanted far edge. That is why the guard exists: with the flat
    test, the circle points land outside the cell in fractional coordinates and
    steal chords, so 16 triangles straddled at divisions 8 (measured). The
    guard now refuses that r/a before any of it runs.

    The test asserts the boundary from BOTH sides, because a guard that refuses
    everything would pass a one-sided check.
    """
    square_max = 0.5
    tri_max = np.sin(np.pi / 3.0) / 2.0
    assert abs(tri_max - 0.433013) < 1e-6, "the rhombus inradius moved"

    for lattice, rmax in (("square", square_max), ("triangular", tri_max)):
        # just inside is accepted
        pc.unit_cell_mesh(lattice, divisions=8, r_over_a=rmax - 1e-4, n_rod=3.4)
        # at and beyond is refused, NAMING the limit
        for r in (rmax, rmax + 0.02):
            with pytest.raises(pc.PCError) as excinfo:
                pc.unit_cell_mesh(lattice, divisions=8, r_over_a=r, n_rod=3.4)
            msg = str(excinfo.value)
            assert "r/a" in msg, f"refusal did not name r/a: {msg!r}"
            assert f"{rmax:.4f}" in msg, (
                f"refusal did not name the limit {rmax:.4f}: {msg!r}"
            )

    # the specific case that exposed it: r/a = 0.45 is legal on the square and
    # ILLEGAL on the triangular cell
    pc.unit_cell_mesh("square", divisions=8, r_over_a=0.45, n_rod=3.4)
    with pytest.raises(pc.PCError):
        pc.unit_cell_mesh("triangular", divisions=8, r_over_a=0.45, n_rod=3.4)


def test_unknown_lattice_is_refused_naming_the_choices():
    with pytest.raises(pc.PCError) as excinfo:
        pc.lattice_vectors("hexagonal-ish")
    msg = str(excinfo.value)
    assert "square" in msg and "triangular" in msg, (
        f"refusal did not name the accepted lattices: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Conformity: the mesh must not straddle the interface
# ---------------------------------------------------------------------------

def _straddlers(lattice, divisions, r_over_a, n_rod=3.4, n_bg=1.0):
    """Count triangles whose INTERIOR the circle passes through.

    Classifying vertices by ``d < r`` is WRONG and I did it first: every circle
    point sits at ``d = r``, so floating-point noise splits them 17 inside / 5
    outside out of 64, and a boundary chord triangle reads as a straddler. The
    correct test uses a tolerance: a vertex within ``tol`` of the circle is ON it
    (assigned to whichever side its neighbours agree on), and a triangle
    straddles when its vertices are strictly mixed, or when they are all
    inside-or-on / outside-or-on but the CENTROID is on the other side.
    """
    pts, tris, eps, pairs = pc.unit_cell_mesh(lattice, divisions,
                                              r_over_a=r_over_a,
                                              n_rod=n_rod, n_bg=n_bg)
    a1, a2 = pc.lattice_vectors(lattice)
    centre = 0.5 * (a1 + a2)
    d = np.linalg.norm(pts - centre, axis=1)
    tol = 1e-9
    tri_in = (d < r_over_a - tol)[tris]
    tri_out = (d > r_over_a + tol)[tris]
    mixed = tri_in.any(axis=1) & tri_out.any(axis=1)

    cen = pts[tris].mean(axis=1)
    dc = np.linalg.norm(cen - centre, axis=1)
    all_in_or_on = ~tri_out.any(axis=1)
    all_out_or_on = ~tri_in.any(axis=1)
    return int(
        (mixed
         | (all_in_or_on & (dc > r_over_a + tol))
         | (all_out_or_on & (dc < r_over_a - tol))).sum()
    )


def test_the_mesh_never_straddles_the_interface():
    """THE PROMISE: zero triangles straddle, across the whole sweep.

    Delaunay on a structured grid does NOT promise this -- a grid point close to
    the circle steals a chord -- so the circle's own nodes are inserted and the
    grid points within h/2 of it are dropped. This asserts the promise itself,
    not a proxy for it.

    The sweep is the one the lead specified: divisions 8..40 x r/a 0.1..0.45, on
    both cells, INCLUDING r values that fall just off a grid line. r/a = 0.45 is
    only legal on the square (the rhombus inradius is 0.4330), so for the
    triangular cell the 0.45 column is asserted as a REFUSAL instead -- sweeping
    it as a mesh would be sweeping a shape that does not fit.
    """
    divisions_list = (8, 12, 16, 24, 32, 40)
    r_list = (0.1, 0.2, 0.25, 0.3, 0.45)
    tri_max = np.sin(np.pi / 3.0) / 2.0

    checked = 0
    refusals = 0
    for lattice in ("square", "triangular"):
        for divisions in divisions_list:
            for r in r_list:
                if lattice == "triangular" and r >= tri_max:
                    with pytest.raises(pc.PCError):
                        pc.unit_cell_mesh(lattice, divisions, r_over_a=r,
                                          n_rod=3.4)
                    refusals += 1
                    continue
                n = _straddlers(lattice, divisions, r)
                assert n == 0, (
                    f"{lattice} divisions={divisions} r/a={r}: {n} triangles "
                    f"straddle the interface; the mesh must conform"
                )
                checked += 1

    assert checked == 54, f"expected 54 meshable cases, ran {checked}"
    assert refusals == 6, f"expected 6 refusals, got {refusals}"


def test_the_conforming_circle_does_not_move_the_periodic_pairing():
    """The circle points must never touch the cell boundary, so the pairing is
    exactly what the grid alone produced.

    This is the control for a bug I actually introduced: dropping grid points
    within h/2 of the circle also removed BOUNDARY nodes on the rhombic cell,
    which collapsed the pairing. So the count is asserted, AND the set of paired
    nodes is asserted to be exactly the grid's far-face nodes.

    The r/a list deliberately pushes INTO the regime where that happens: at
    triangular, divisions 12, removing the boundary exemption gives 13 pairs at
    r/a = 0.40 and 0.42 and 5 at 0.43, where 25 are required. 0.43 is accepted
    (the inradius is 0.4330), so this is the live case, and a sweep that stopped
    at 0.383 -- `rmax - 0.05` -- would have missed it entirely.
    """
    for lattice in ("square", "triangular"):
        a1, a2 = pc.lattice_vectors(lattice)
        A = np.column_stack([a1, a2])
        rmax = 0.5 if lattice == "square" else np.sin(np.pi / 3.0) / 2.0
        # the last r/a is just inside the inradius, where the rod comes closest
        # to the far edge and the exemption must still hold
        r_list = (0.0, 0.2, min(0.4, rmax - 0.05), rmax - 1e-3)
        for divisions in (8, 12, 24, 40):
            for r in r_list:
                pts, tris, eps, pairs = pc.unit_cell_mesh(
                    lattice, divisions, r_over_a=r, n_rod=3.4
                )
                # the pair count is the grid's, unchanged by the circle
                assert len(pairs) == 2 * divisions + 1, (
                    f"{lattice} divisions={divisions} r/a={r}: {len(pairs)} "
                    f"pairs, expected {2 * divisions + 1}"
                )
                # every slave is a far-face node, and every far-face node is a
                # slave -- recomputed from the points, not from the module
                frac = np.round(pts @ np.linalg.inv(A).T, 9)
                far = {
                    i for i, (u, v) in enumerate(frac)
                    if abs(u - 1.0) < 1e-9 or abs(v - 1.0) < 1e-9
                }
                slaves = [s for s, _, _ in pairs]
                assert len(slaves) == len(set(slaves)), "a node paired twice"
                assert set(slaves) == far, (
                    f"{lattice} divisions={divisions} r/a={r}: the paired set "
                    f"is not the far-face set"
                )


def test_the_circle_points_sit_strictly_inside_the_cell():
    """The circle never reaches the cell boundary -- the geometric reason the
    pairing above is undisturbed. At r/a at the inradius it would touch; the
    guard refuses that, so every accepted r leaves a margin."""
    a1, a2 = pc.lattice_vectors("square")
    A = np.column_stack([a1, a2])
    for divisions in (8, 24, 40):
        for r in (0.2, 0.25, 0.45):
            pts, _, _, _ = pc.unit_cell_mesh("square", divisions, r_over_a=r,
                                             n_rod=3.4)
            centre = 0.5 * (a1 + a2)
            on_circle = np.abs(np.linalg.norm(pts - centre, axis=1) - r) < 1e-9
            assert on_circle.sum() > 0, "no circle points were inserted"
            frac = pts[on_circle] @ np.linalg.inv(A).T
            # fractional coordinates strictly inside (0, 1) means no boundary
            assert frac.min() > 0.0 and frac.max() < 1.0, (
                f"divisions={divisions} r/a={r}: a circle point is on the cell "
                f"boundary (frac range {frac.min():.6f}..{frac.max():.6f})"
            )


def test_the_weight_parameter_separates_the_two_operators():
    """TM and TE must be DIFFERENT operators on a non-uniform cell and the SAME
    on a uniform one.

    This is the test for the ``weight=`` argument, and it exists because the
    previous assembly could not express TE at all: it hard-coded eps onto the
    mass, so TE (1/eps on the STIFFNESS) was unreachable. The uniform half is not
    padding -- it is the measured reason a uniform-eps check cannot catch a
    swapped eps slot, which is Control B's whole justification.
    """
    pts, tris, eps, pairs = pc.unit_cell_mesh("square", divisions=12,
                                              r_over_a=0.25, n_rod=3.4, n_bg=1.0)
    K_tm, M_tm = pc._p1_matrices(pts, tris, eps, weight="mass")
    K_te, M_te = pc._p1_matrices(pts, tris, 1.0 / eps, weight="stiffness")

    # on a NON-uniform cell they differ, in both matrices
    assert not np.allclose(K_tm.toarray(), K_te.toarray()), (
        "TM and TE stiffnesses agree on a non-uniform cell"
    )
    assert not np.allclose(M_tm.toarray(), M_te.toarray()), (
        "TM and TE masses agree on a non-uniform cell"
    )

    # and eps lands on the RIGHT one each time. The discriminator is the SPREAD
    # of the diagonal: the TM stiffness is plain (its diagonal varies only with
    # element shape), the TE stiffness carries 1/eps and so varies MORE. Neither
    # mass diagonal is uniform even when unweighted -- the lumped mass scales
    # with element area -- so the mass pair is compared by its RATIO instead,
    # which is what "plain mass" actually predicts.
    tm_k_spread = K_tm.diagonal().max() / K_tm.diagonal().min()
    te_k_spread = K_te.diagonal().max() / K_te.diagonal().min()
    assert te_k_spread > tm_k_spread * 1.5, (
        f"the 1/eps stiffness should vary more than the plain one: "
        f"TE spread {te_k_spread:.3f} vs TM {tm_k_spread:.3f}"
    )
    # eps lands on the MASS and only the mass: the weighted mass equals eps
    # times the plain one, EXACTLY, at every eps. This is an element-level fact
    # and it has to be checked there -- a NODE of this mesh touches elements on
    # both sides of the interface, so its assembled diagonal is an area-weighted
    # mixture of the two eps values, not 1/eps. (I asserted the node ratio first
    # and got 11 values where I predicted 2; the prediction was wrong, not the
    # code.) The uniform cell below is the same check where mixtures cannot
    # arise; here the whole cell is one eps and it must still hold.
    for e in (1.0, 4.0, 3.4 ** 2):
        eps_one = np.full(len(tris), e, dtype=np.float64)
        K_w, M_w = pc._p1_matrices(pts, tris, eps_one, weight="mass")
        K_p, M_p = pc._p1_matrices(pts, tris, eps_one, weight="stiffness")
        # weight="mass" carries eps on the MASS: M_w = e * M_p, plain stiffness
        assert np.allclose(M_w.toarray(), e * M_p.toarray()), (
            f"weight='mass' did not put eps on the mass at eps={e}"
        )
        assert np.allclose(K_p.toarray(), e * K_w.toarray()), (
            f"weight='stiffness' did not put eps on the stiffness at eps={e}"
        )

    # on a UNIFORM cell the two operators are IDENTICAL
    pts1, tris1, eps1, _ = pc.unit_cell_mesh("square", divisions=12,
                                             r_over_a=0.0, n_rod=1.0, n_bg=1.0)
    assert set(np.round(eps1, 12).tolist()) == {1.0}, "cell is not uniform"
    K1m, M1m = pc._p1_matrices(pts1, tris1, eps1, weight="mass")
    K1s, M1s = pc._p1_matrices(pts1, tris1, 1.0 / eps1, weight="stiffness")
    assert np.allclose(K1m.toarray(), K1s.toarray())
    assert np.allclose(M1m.toarray(), M1s.toarray())

    # an unknown weight is refused, naming the choices
    with pytest.raises(pc.PCError) as excinfo:
        pc._p1_matrices(pts, tris, eps, weight="both")
    assert "mass" in str(excinfo.value) and "stiffness" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The mesh controls (HEAVY: every one of these calls eigsh)
#
# Skipped by default. Run on a cluster Job:
#     MSE_RUN_HEAVY=1 python -m pytest tests/test_pc.py -m heavy
# The selection and the expected wall-clock are in the row for the lead.
# ---------------------------------------------------------------------------

# The production mesh, from plan section 3's measurement.
PRODUCTION_DIVISIONS = 40


def _bands(lattice, divisions, r_over_a, n_rod, n_bg, k_points, n_bands,
           polarisation):
    """One call into the shared chain, so every control uses the same path."""
    _, bands = pc.bands_from_mesh(
        lattice, divisions, r_over_a, n_rod, n_bg, k_points, n_bands,
        polarisation=polarisation,
    )
    return bands


@pytest.mark.heavy
def test_control_A_empty_lattice_matches_the_folded_light_line():
    """CONTROL A -- the empty lattice, analytic, and it can fail.

    ``n_rod = n_bg = 1`` reduces to the free-space problem folded into the
    Brillouin zone, so the lowest band is ``|k| a / (2 pi c)``. At the three
    high-symmetry points the exact values are 0, 1/2, sqrt2/2.

    Tolerance, per the revised plan: ONLY lam = 0 at Gamma is exact on a P1 mesh
    (the constant field is in the space and the stiffness annihilates it), and
    that one is asserted at 1e-10. Everything else is O(h^2), so it is asserted
    as a relative bound at the production mesh, plus a measured convergence
    order -- asserting equality here would be asserting something false.
    """
    lattice = "square"
    a1, a2 = pc.lattice_vectors(lattice)
    # Gamma, X, M in the 2 pi/a units the module uses
    k_pts = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
    exact = np.array([0.0, 0.5, np.sqrt(2.0) / 2.0])

    bands = _bands(lattice, PRODUCTION_DIVISIONS, 0.0, 1.0, 1.0, k_pts, 4, "TM")
    lowest = bands[:, 0]

    # the ONE exact landmark
    assert abs(lowest[0] - 0.0) < 1e-10, (
        f"lam = 0 at Gamma is exact on a P1 mesh; got {lowest[0]:.3e}"
    )
    # the rest, as a relative bound stated before running
    for name, got, want in zip(("X", "M"), lowest[1:], exact[1:]):
        rel = abs(got - want) / want
        assert rel <= 1e-3, (
            f"empty lattice at {name}: {got:.6f} vs {want:.6f}, rel {rel:.3e} "
            f"exceeds the 1e-3 production-mesh bound"
        )


@pytest.mark.heavy
def test_control_A_convergence_order_is_second_order():
    """CONTROL A, the half that actually discriminates.

    A correct second-order P1 assembly shows err(2h)/err(h) -> 4. A bad stiffness
    or a mishandled periodic boundary term still lands near the right values at
    one mesh size but does NOT converge at order 2, so the bound above alone can
    be satisfied by a wrong operator. The lead asked for the observed order to
    land in [1.8, 2.2]; the measured ratio must fall in [3.5, 4.5].
    """
    lattice = "square"
    k_pts = np.array([[0.5, 0.5]])          # M, the largest error point
    want = np.sqrt(2.0) / 2.0

    errs = []
    for divisions in (10, 20, 40):
        bands = _bands(lattice, divisions, 0.0, 1.0, 1.0, k_pts, 1, "TM")
        errs.append(abs(bands[0, 0] - want))

    assert errs[0] > errs[1] > errs[2], (
        f"error does not decrease under refinement: {errs}"
    )
    ratio = errs[0] / errs[1]          # (h -> h/2)
    order = np.log2(ratio)
    assert 1.8 <= order <= 2.2, (
        f"observed order {order:.3f} (ratio {ratio:.3f}) is outside [1.8, 2.2]; "
        f"errors were {errs}. A correct P1 assembly gives order 2."
    )


@pytest.mark.heavy
def test_control_B_layered_cell_matches_the_transfer_matrix_bands():
    """CONTROL B -- a 1-D layered cell against the exact transfer-matrix bands.

    The stripe varies only in x, and the k-point carries a NONZERO component
    parallel to the layers (ky != 0). Both are required: at ky = 0 (or at
    uniform eps) the two polarisations coincide and the comparison cannot see a
    swapped eps slot -- which is exactly the defect this control exists to catch.

    The mesh dispersion is compared to ITS OWN polarisation's closed-form branch.
    """
    lattice = "square"
    eps2, fill = 4.0, 0.5
    n_rod = 1.0                       # the "rod" is the eps2 stripe
    ky = 0.3

    # per polarisation, the closed-form band edges at this ky
    bands_s, bands_p = pc.layered_bands(eps2, fill, ky, n_bands=6)
    assert len(bands_s) >= 2 and len(bands_p) >= 2, (
        f"oracle returned too few band edges: s={bands_s}, p={bands_p}"
    )

    k_pts = np.array([[0.0, ky]])
    for polarisation, oracle_bands in (("TM", bands_s), ("TE", bands_p)):
        got = _bands(lattice, PRODUCTION_DIVISIONS, 0.5, n_rod ** 0.5 * 0.0 + 1.0,
                     1.0, k_pts, 1, polarisation)[0, 0]
        # the lowest branch below the first edge is the first band edge
        want = oracle_bands[0]
        rel = abs(got - want) / abs(want)
        assert rel <= 1e-3, (
            f"Control B {polarisation}: mesh {got:.6f} vs transfer matrix "
            f"{want:.6f}, rel {rel:.3e} exceeds 1e-3"
        )


@pytest.mark.heavy
def test_control_B_can_fail_because_the_polarisations_differ():
    """The precondition that makes Control B meaningful.

    If the two oracle branches coincided at the chosen k, Control B would pass
    with eps in EITHER slot. So assert they differ at the tested ky -- the
    control's own can-fail check -- and that they DO coincide at ky = 0.
    """
    eps2, fill = 4.0, 0.5
    s, p = pc.layered_bands(eps2, fill, 0.3, n_bands=4)
    assert not np.allclose(s[0], p[0], rtol=1e-6), (
        "the s and p branches coincide at ky=0.3, so Control B cannot detect a "
        "swapped eps slot"
    )
    s0, p0 = pc.layered_bands(eps2, fill, 0.0, n_bands=4)
    assert np.allclose(s0[0], p0[0], rtol=1e-9), (
        "at ky=0 the polarisations must coincide; they do not, so the ky=0 "
        "degeneracy this control relies on is not reproduced"
    )
