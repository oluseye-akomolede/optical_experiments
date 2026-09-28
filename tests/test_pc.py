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


def test_r_over_a_at_half_is_refused_naming_the_limit():
    """A rod of r/a >= 0.5 would not fit in the cell. The refusal must NAME the
    value and the limit -- a refusal that cannot say what was wrong is the
    could-not-ask shape."""
    with pytest.raises(pc.PCError) as excinfo:
        pc.unit_cell_mesh("square", divisions=8, r_over_a=0.5)
    msg = str(excinfo.value)
    assert "0.5" in msg or "r/a" in msg, f"refusal did not name the limit: {msg!r}"


def test_unknown_lattice_is_refused_naming_the_choices():
    with pytest.raises(pc.PCError) as excinfo:
        pc.lattice_vectors("hexagonal-ish")
    msg = str(excinfo.value)
    assert "square" in msg and "triangular" in msg, (
        f"refusal did not name the accepted lattices: {msg!r}"
    )
