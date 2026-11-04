"""P1 scalar assembly for -div(eps_static eps0 grad phi) = rho (task 1.2).

The checks, in the order the task lists them:

(a) one-tet element matrix against the closed form, on a tet that is NOT
    right-angled -- a right-angled tet has J = I (or a permutation of it) and
    cannot see the ``inv(J).T`` transpose trap;
(b) every assembled row sums to zero (a constant potential has no flux);
(c) the assembled matrix is symmetric;
(d) a two-part fixture whose parts differ, where a mutation that ignores
    ``eps_static`` (uses 1 everywhere) turns the check red.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dockerfiles", "maxwell_solver"))

import mse_electrostatics as E  # noqa: E402
import mse_fem as F  # noqa: E402


# A general tet: no right angles, so J is not a permutation of the identity.
GENERAL_TET = np.array(
    [[0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [0.2, 1.1, 0.05], [0.1, 0.3, 1.4]]
)


def _closed_form_element(points, tet, eps_static):
    """The 4x4 P1 matrix from the definition, computed independently.

    Deliberately does NOT reuse ``tet_geometry``: it solves for the barycentric
    gradients as the solution of ``grad(lambda_i) . (x_j - x_0) = delta_ij``,
    which is the definition, so a transpose error in either place shows up as a
    disagreement rather than cancelling.
    """
    x = np.asarray(points, dtype=np.float64)[np.asarray(tet)]
    vol = abs(np.linalg.det(np.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]], axis=1))) / 6.0

    # grad(lambda_i) = 1/3 * sum over faces ... too easy to get wrong; instead
    # solve the linear system: lambda_i is affine, lambda_i(x_j) = delta_ij,
    # so writing lambda_i(x) = a_i . x + b_i, the a_i solve
    #     [x_1 - x_0; x_2 - x_0; x_3 - x_0] @ a_i = e_i
    # for i = 1,2,3, and a_0 = -(a_1 + a_2 + a_3).
    Bm = np.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]], axis=0)   # (3, 3), rows
    a = np.linalg.solve(Bm, np.eye(3))                                # columns are a_1..a_3
    grads = np.zeros((4, 3))
    grads[1:] = a.T
    grads[0] = -grads[1:].sum(axis=0)

    return eps_static * E.EPS0 * vol * (grads @ grads.T), vol


def test_a_one_tet_element_matrix_matches_the_closed_form():
    tets = np.array([[0, 1, 2, 3]])
    eps_static = 12.5

    K_elem, volumes = E.element_stiffness(
        GENERAL_TET, tets, np.array([eps_static])
    )
    expected, vol = _closed_form_element(GENERAL_TET, tets[0], eps_static)

    # the reference volume is computed a different way here, so compare it too
    assert volumes[0] == pytest.approx(vol, rel=1e-12)
    assert np.allclose(K_elem[0], expected, rtol=1e-12, atol=0.0), (
        "element matrix disagrees with the closed form; the usual cause is "
        "grad(lambda) = J @ ... instead of inv(J).T @ ..."
    )


def test_the_closed_form_check_would_see_a_transpose_error():
    """(a) is only worth its pass if the wrong transpose fails it.

    Guards against writing a reference that reuses the implementation's own
    convention and therefore agrees with a bug.
    """
    tets = np.array([[0, 1, 2, 3]])
    K_elem, _ = E.element_stiffness(GENERAL_TET, tets, np.array([12.5]))

    det_J, grads_right = F.tet_geometry(GENERAL_TET, tets)
    x = GENERAL_TET[tets][0]
    J = np.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]], axis=1)
    grad_ref = np.array(
        [-1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ).reshape(4, 3)
    # J is one 3x3 here, so no batch index: "ij,aj->ai" applies inv(J) on the
    # wrong side. (Copying the batched "nij,aj->nai" spelling straight from the
    # vector-path test passes a 3-axis subscript to a 2-D operand.)
    grads_wrong = np.einsum("ij,aj->ai", np.linalg.inv(J), grad_ref)

    assert not np.allclose(grads_right[0], grads_wrong), "the fixture cannot tell them apart"
    wrong = 12.5 * E.EPS0 * abs(det_J[0]) / 6.0 * (grads_wrong @ grads_wrong.T)
    # Relative, no absolute floor. EPS0 is ~8.9e-12, so every entry here is
    # ~1e-11, far below np.allclose's default atol=1e-8: an absolute floor that
    # large would call ANY two entries equal and the guard could never fail. The
    # default is a trap for future mismatches at this scale, not a live 3.5%
    # error -- on this fixture the two matrices differ by ~1e-12 absolute (measured 1.7e-12).
    assert not np.allclose(K_elem[0], wrong, rtol=1e-12, atol=0.0), (
        "a transposed gradient gives the same element matrix, so (a) would pass "
        "on the bug"
    )


def _two_tet_fixture():
    """Two CONNECTED tets sharing a face, in different parts with different eps.

    Connected on purpose: this is what the mesher produces, and it is the case a
    node-derived material would reject (the shared face's three nodes are in
    both tets). The mesh's own ``region_of_tet`` assigns each tet wholly to a
    part, so the boundary running through the shared face is not a problem.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.0],
            [0.2, 1.1, 0.05],
            [0.1, 0.3, 1.4],       # apex of tet A (part "core")
            [0.6, 0.55, -1.1],     # apex of tet B (part "shell")
        ]
    )
    tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])   # B's orientation is flipped
    parts = ["core", "shell"]
    region_of_tet = np.array([0, 1])
    eps_static = {"core": 12.5, "shell": 3.9}
    return points, tets, parts, region_of_tet, eps_static


def test_b_every_assembled_row_sums_to_zero():
    """A constant potential has no flux, so K @ ones == 0."""
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    K, _ = E.assemble_stiffness(points, tets, parts, eps_static, region_of_tet)

    row_sums = np.asarray(K.sum(axis=1)).ravel()
    scale = np.asarray(abs(K).sum(axis=1)).ravel().max()
    assert np.allclose(row_sums, 0.0, atol=1e-12 * scale), (
        f"row sums are not zero (max {abs(row_sums).max():.3e}); a constant "
        "potential would carry flux"
    )


def test_c_the_assembled_matrix_is_symmetric():
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    K, _ = E.assemble_stiffness(points, tets, parts, eps_static, region_of_tet)

    diff = abs(K - K.T)
    scale = abs(K).max()
    assert diff.max() <= 1e-12 * scale, (
        f"K is not symmetric (max asymmetry {diff.max():.3e})"
    )


def test_d_the_parts_permittivity_reaches_the_matrix():
    """(d) Mutation target: ignoring eps_static (using 1 everywhere) fails here.

    Rather than assert "eps is used", this asserts the VALUE the two-part
    matrix must have for the stated eps, so a build that quietly uses one
    material for both parts cannot pass.
    """
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    K, tet_part = E.assemble_stiffness(points, tets, parts, eps_static, region_of_tet)

    assert list(tet_part) == [0, 1], "fixture should give one tet per part"

    # Rebuild each element by hand with its own part's eps and scatter.
    expected = np.zeros((points.shape[0], points.shape[0]))
    for n, tet in enumerate(tets):
        eps_n = eps_static[parts[tet_part[n]]]
        k_n, _ = E.element_stiffness(points, np.array([tet], dtype=np.int64),
                                     np.array([eps_n]))
        for i_local, i_global in enumerate(tet):
            for j_local, j_global in enumerate(tet):
                expected[i_global, j_global] += k_n[0, i_local, j_local]

    # atol=0.0 is load-bearing: with the default atol=1e-8 every entry (~1e-11)
    # compares equal and this assertion cannot fail.
    assert np.allclose(K.toarray(), expected, rtol=1e-12, atol=0.0), (
        "assembled matrix is not the sum of per-part element matrices"
    )

    # and the uniform-eps matrix is measurably different, so the check above is
    # not satisfied by a build that ignores eps_static
    K_uniform, _ = E.assemble_stiffness(
        points, tets, parts, {"core": 12.5, "shell": 12.5}, region_of_tet
    )
    rel = abs(K - K_uniform).max() / abs(K).max()
    assert rel > 1e-3, (
        f"the two-part matrix is within {rel:.2e} of the uniform one, so an "
        "eps-ignoring mutation would not be caught"
    )


def test_a_part_with_no_permittivity_is_refused():
    points, tets, parts, region_of_tet, _ = _two_tet_fixture()
    with pytest.raises(ValueError, match="no eps_static for part"):
        E.assemble_stiffness(points, tets, parts, {"core": 12.5}, region_of_tet)


def test_a_permittivity_for_a_part_that_does_not_exist_is_refused():
    points, tets, parts, region_of_tet, _ = _two_tet_fixture()
    with pytest.raises(ValueError, match="do not exist"):
        E.assemble_stiffness(
            points, tets, parts, {"core": 12.5, "shell": 3.9, "coret": 1.0}, region_of_tet
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_permittivity_is_refused(bad):
    points, tets, parts, region_of_tet, _ = _two_tet_fixture()
    with pytest.raises(ValueError, match="eps_static for part"):
        E.assemble_stiffness(
            points, tets, parts, {"core": 12.5, "shell": bad}, region_of_tet
        )


def test_a_mesh_whose_nodes_straddle_a_part_boundary_still_assembles():
    """The material is the TET's, not its nodes', so a shared boundary face is fine.

    This is the regression that a node-derived material would fail: tet 0 and
    tet 1 share nodes 0,1,2, so those nodes sit on both sides of the boundary.
    The mesh's region ids say tet 0 is "core" and tet 1 is "shell", which is all
    the assembly needs.
    """
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    K, tet_part = E.assemble_stiffness(points, tets, parts, eps_static, region_of_tet)
    assert list(tet_part) == [0, 1]
    assert K.shape == (5, 5)


def test_a_region_id_with_no_part_is_refused():
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    bad = region_of_tet.copy()
    bad[1] = 7                                       # a tet in a part nobody declared
    with pytest.raises(ValueError, match="no part"):
        E.assemble_stiffness(points, tets, parts, eps_static, bad)


def test_a_part_with_no_tets_is_refused():
    """A part the caller believes is in the device but the mesh does not contain."""
    points, tets, parts, region_of_tet, eps_static = _two_tet_fixture()
    all_core = np.zeros(2, dtype=np.int64)           # "shell" has no tets
    with pytest.raises(ValueError, match="have no tets"):
        E.assemble_stiffness(
            points, tets, parts, {"core": 12.5, "shell": 3.9}, all_core
        )


# --- the interface physics ---------------------------------------------------
#
# A layered block is the one case where the exact answer is known in closed form,
# so it is the only test that pins the ASSEMBLY rather than an invariant (row
# sums, symmetry) the assembly could satisfy while still being wrong. Two layers
# in series, eps_a below x=0.5 and eps_b above, 0 V at x=0 and 1 V at x=1.
#
# Flux continuity eps_a*E_a = eps_b*E_b with E_a*d_a + E_b*d_b = 1 over equal
# thicknesses gives E_a = 2*E_b, so phi jumps to eps_b/(eps_a+eps_b) at the
# interface (NOT eps_a/(eps_a+eps_b) -- that is the layers-swapped answer).
#
# The mesh must be STRUCTURED, with the interface on a node plane. A Delaunay
# tetrahedralization of a regular grid produces flat (zero-volume) tets there
# and the matrix goes singular; and any tet SPANNING the interface would carry
# one eps for two materials, which this assembly does not model (see the module
# docstring). Zero tets cross the plane here -- asserted below, because that
# property is what makes the exact answer exact rather than approximate.

_CUBE_TETS = np.array(
    [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
)


def _structured_cube_chain(nx=8, ny=2, nz=2):
    """An nx x ny x nz block of cubes, each split into 6 tets about one diagonal."""
    xs, ys, zs = (np.linspace(0.0, 1.0, n + 1) for n in (nx, ny, nz))

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    points = np.array(
        [[xs[i], ys[j], zs[k]] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)]
    )
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corner = {
                    (a, b, c): node(i + a, j + b, k + c)
                    for a in (0, 1) for b in (0, 1) for c in (0, 1)
                }
                for t in _CUBE_TETS:
                    bits = [tuple(int(v) for v in np.unravel_index(u, (2, 2, 2))) for u in t]
                    tets.append([corner[b] for b in bits])
    return points, np.array(tets)


def _solve_layers(eps_a, eps_b):
    """Solve the layered block; return (points, phi, region_of_tet)."""
    points, tets = _structured_cube_chain()
    centroids_x = points[tets][:, :, 0].mean(axis=1)
    region_of_tet = (centroids_x > 0.5).astype(np.int64)      # 0 = "a", 1 = "b"
    K, _ = E.assemble_stiffness(
        points, tets, ["a", "b"], {"a": eps_a, "b": eps_b}, region_of_tet
    )
    K = K.toarray()
    fixed = {}
    for n, (x, _, _) in enumerate(points):
        if abs(x) < 1e-12:
            fixed[n] = 0.0
        elif abs(x - 1.0) < 1e-12:
            fixed[n] = 1.0
    free = [n for n in range(len(points)) if n not in fixed]
    phi = np.zeros(len(points))
    for n, v in fixed.items():
        phi[n] = v
    phi[free] = np.linalg.solve(
        K[np.ix_(free, free)], -K[np.ix_(free, list(fixed))] @ np.array(list(fixed.values()))
    )
    return points, phi, region_of_tet, tets


def _exact_phi(x, interface):
    """Piecewise-linear solution: linear to the interface, linear after it."""
    return interface * (x / 0.5) if x <= 0.5 else interface + (1.0 - interface) * ((x - 0.5) / 0.5)


@pytest.mark.parametrize("eps_a,eps_b", [(1.0, 2.0), (2.0, 1.0), (1.0, 4.0), (4.0, 1.0)])
def test_the_layered_interface_sits_at_eps_b_over_the_sum(eps_a, eps_b):
    """The assembly reproduces the exact layered solution, not just its invariants.

    The contrast is asymmetric under swapping (1 vs 2, 1 vs 4), so a fixture that
    swapped the layers would land on 1/3 or 1/5 and this would see it. A
    symmetric contrast (2 vs 2) would give 0.5 either way and prove nothing about
    the interface.
    """
    points, phi, region_of_tet, tets = _solve_layers(eps_a, eps_b)

    # The exactness claim only holds for a conforming interface. If this ever
    # trips, the fixture stopped being the case the module documents.
    lo, hi = points[tets][:, :, 0].min(axis=1), points[tets][:, :, 0].max(axis=1)
    assert not np.any((lo < 0.5) & (hi > 0.5)), (
        "a tet spans the interface; the exact layered answer does not apply here"
    )

    interface = eps_b / (eps_a + eps_b)
    assert interface != 0.5, "a symmetric contrast cannot distinguish the two layers"

    plane = np.isclose(points[:, 0], 0.5)
    # No absolute floor: atol=1e-8 would make every assertion below vacuous.
    assert np.allclose(phi[plane], interface, rtol=1e-12, atol=0.0), (
        f"interface phi is {np.unique(phi[plane])}, expected eps_b/(eps_a+eps_b)={interface}"
    )

    for x in np.unique(points[:, 0]):
        at_x = np.isclose(points[:, 0], x)
        expected = _exact_phi(x, interface)
        assert np.allclose(phi[at_x], expected, rtol=1e-12, atol=0.0), (
            f"x={x}: phi {np.unique(phi[at_x])} != exact {expected}"
        )


# --- task 1.3: electrode Dirichlet reduction and the direct solve ------------
#
# The layered chain above pins the ASSEMBLY with voltages imposed by x-plane.
# These pin SOLVE: the same physics reached through parts, which is how the
# product states a contact -- an electrode is a part, and every node of that
# part is held. The known answer is therefore the same closed form: three
# layers in series (electrode, gap, electrode), phi linear between the contacts.

def _contact_chain(nx=4, ny=3, nz=3, gap_eps=2.0):
    """A bar of ``nx`` cubes: the first and last slabs are electrodes.

    Returns (points, tets, parts, region_of_tet). Part 0 "left" and part 2
    "right" are the contacts; part 1 "gap" is the dielectric between them. The
    contact is a whole slab of elements, so its nodes are a face, which is what
    the mesher produces for a part boundary.
    """
    points, tets = _structured_cube_chain(nx, ny, nz)
    cx = points[tets][:, :, 0].mean(axis=1)
    # The contacts are the OUTER QUARTER at each end, not a single slab: with
    # nx=4 that puts each contact face on a node plane (x=0.25, 0.75) and makes
    # the gap contain real interior nodes. A one-slab contact leaves the
    # electrode tets far from the interface, and then a mutation that wrongly
    # assembles them is numerically invisible -- measured: it passed 29/29.
    region_of_tet = np.where(cx < 0.25, 0, np.where(cx > 0.75, 2, 1))
    return points, tets, ["left", "gap", "right"], region_of_tet, gap_eps


def test_solve_holds_each_electrode_part_and_solves_the_gap():
    """Two contacts at 0 V and 1 V: the gap's potential is linear in x.

    Known answer: with a single uniform dielectric between two plane contacts
    the potential is exactly linear, so every node at x must read x -- not
    merely be bounded by [0, 1]. A solve that dropped the Dirichlet terms, or
    reduced the wrong block, would bend this curve.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    result = E.solve(points, tets, parts, region_of_tet,
                     {"gap": gap_eps}, {"left": 0.0, "right": 1.0})

    # The contacts are SLABS, not planes: every node in x in [0, 1/nx] is held
    # at 0 V and every node in [1 - 1/nx, 1] at 1 V. The field therefore runs
    # only across the gap between the INNER faces, and the potential is
    # constant (not linear) inside each contact slab. Asserting globally
    # linear-in-x would be testing the wrong device -- and did, until the
    # fixture was read: it caught x-quantised values, not a solver error.
    x = points[:, 0]
    x_lo, x_hi = 0.25, 0.75
    expected = np.clip((x - x_lo) / (x_hi - x_lo), 0.0, 1.0)
    phi = result["phi"]
    assert np.allclose(phi, expected, rtol=0.0, atol=1e-12), (
        f"worst deviation from the piecewise answer: "
        f"{np.abs(phi - expected).max():.3e}"
    )
    # The contact slabs really are held, not merely close.
    assert np.allclose(phi[x <= x_lo], 0.0, rtol=0.0, atol=0.0)
    assert np.allclose(phi[x >= x_hi], 1.0, rtol=0.0, atol=0.0)
    assert result["electrodes"]["left"].size and result["electrodes"]["right"].size
    assert result["n_free"] == int((~result["free"]).size - (~result["free"]).sum())


def test_solve_imposes_the_voltage_not_a_fitted_one():
    """The same bar at 0 V and 3 V gives 3x the potential everywhere.

    A solve that imposed the electrodes only approximately (or let them float)
    would not scale exactly, because the boundary values are the only source
    of the field.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    one = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                  {"left": 0.0, "right": 1.0})["phi"]
    three = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                    {"left": 0.0, "right": 3.0})["phi"]
    assert np.allclose(three, 3.0 * one, rtol=1e-13, atol=0.0)


def test_the_maximum_principle_holds():
    """No interior node exceeds the contacts' voltage range.

    The discrete maximum principle follows from K being an M-matrix (positive
    diagonal, non-positive off-diagonal, zero row sums); it is checked rather
    than assumed, because an assembly with a sign error can still be SPD and
    yet oscillate.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    phi = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                  {"left": -2.0, "right": 5.0})["phi"]
    assert phi.min() >= -2.0 - 1e-9 * 7.0
    assert phi.max() <= 5.0 + 1e-9 * 7.0


def test_a_floating_part_is_solved_not_refused():
    """A dielectric touching no electrode is a valid device (its potential is
    set by its neighbours), so it must solve rather than be refused."""
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    result = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                     {"left": 0.0, "right": 1.0})
    assert np.isfinite(result["phi"]).all()
    assert result["free"].any()


def test_one_electrode_is_enough():
    """One contact at 1 V and a natural outer boundary is well posed."""
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    result = E.solve(points, tets, parts, region_of_tet,
                     {"gap": gap_eps, "right": gap_eps}, {"left": 1.0})
    assert np.allclose(result["phi"], 1.0, rtol=0.0, atol=1e-12), (
        "with one contact and no other source the whole device is at that potential"
    )


def test_an_electrode_needs_no_permittivity():
    """A held part is not assembled, so it needs no eps_static.

    Requiring one would refuse a device that is perfectly well posed, and would
    also tempt a caller to invent a number for a conductor.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
            {"left": 0.0, "right": 1.0})     # no eps_static for "left"/"right"


@pytest.mark.parametrize("electrodes, match", [
    ({}, "no electrode"),
    ({"nowhere": 0.0}, "not parts of this mesh"),
    ({"left": float("nan")}, "non-finite"),
])
def test_a_bad_electrode_set_is_refused(electrodes, match):
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    # The electrode voltage is read before any permittivity is, so a NaN is
    # reported as a NaN rather than as the gap's missing eps_static.
    with pytest.raises(ValueError, match=match):
        E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps}, electrodes)


def test_two_electrodes_sharing_a_node_at_different_volts_are_refused():
    """No single value can be imposed on a shared node, and picking one would
    silently short the two contacts together."""
    points, tets = _structured_cube_chain(2, 1, 1)
    parts = ["a", "b"]
    # Both parts must OCCUPY tets, or check_tet_regions refuses first and the
    # shared-node guard is never reached. Split by cube column: the shared face
    # at x=0.5 then has nodes belonging to both.
    cx = points[tets][:, :, 0].mean(axis=1)
    region_of_tet = (cx > 0.5).astype(np.int64)
    with pytest.raises(ValueError, match="share"):
        E.solve(points, tets, parts, region_of_tet, {}, {"a": 0.0, "b": 1.0})


def test_a_permittivity_below_vacuum_is_refused():
    """eps_s < 1 is the shape of an optical index passed as a permittivity."""
    points, tets, parts, region_of_tet, _ = _contact_chain()
    with pytest.raises(ValueError, match="below 1"):
        E.solve(points, tets, parts, region_of_tet, {"gap": 0.5},
                {"left": 0.0, "right": 1.0})


def test_skipping_a_held_part_changes_nothing_the_solve_can_see():
    """``skip`` exists for the REFUSAL contract, not to change the answer.

    A held part's rows and columns are either dropped from the reduced system
    or overwritten by the imposed value, so the two assemblies give the same
    potential -- measured: the matrices differ by 8.4e-12 over 352 entries and
    phi agrees to 0.0. This test says so explicitly, because a future reader
    will otherwise assume ``skip`` is load-bearing for correctness and either
    remove it (losing the "a contact needs no permittivity" contract) or
    defend it with a test that cannot fail.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    without = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                      {"left": 0.0, "right": 1.0})["phi"]
    with_eps = E.solve(points, tets, parts, region_of_tet,
                       {"gap": gap_eps, "left": 1.0, "right": 1.0},
                       {"left": 0.0, "right": 1.0})["phi"]
    assert np.array_equal(without, with_eps)


# --- AMG-CG (task 1.4) -------------------------------------------------------
#
# The switch is AMG_NODE_CROSSOVER, a named estimate in design.md §5. These
# tests force each path with method= rather than build a 5e5-node mesh: the
# point is that the two paths agree, and a mesh large enough to cross the
# constant naturally cannot run inside tier 1.


def _exact_contact_bar(points):
    """The known answer for _contact_chain: linear between the contact faces.

    The contacts occupy x < 0.25 and x > 0.75, so the driven region is the gap
    between those two planes, where a uniform dielectric gives exactly
    phi = (x - 0.25) / 0.5.
    """
    return (points[:, 0] - 0.25) / 0.5


def test_the_amg_and_direct_paths_agree_on_the_exact_fixture():
    """AMG-CG and the direct LU give the same potential, to AMG's own tolerance.

    The bound is 1e-12 and it comes from AMG_RTOL, not from taste: the direct
    path is exact to machine precision (measured 1e-15 on this fixture), so the
    whole discrepancy is AMG's and asking for more than its rtol would be
    asking the iterative solver for accuracy it was never told to reach.
    Measured at nx=12: direct 1.2e-15, AMG 1.9e-13 against a reference with no
    discretisation error to hide behind (the answer is exactly linear).
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(12, 6, 6)
    volts = {"left": 0.0, "right": 1.0}
    eps = {"gap": gap_eps}
    direct = E.solve(points, tets, parts, region_of_tet, eps, volts, method="direct")
    amg = E.solve(points, tets, parts, region_of_tet, eps, volts, method="amg")

    assert direct["method"] == "direct" and amg["method"] == "amg"
    assert direct["n_free"] == amg["n_free"] > 100

    exact = _exact_contact_bar(points)
    free = direct["free"]
    # Both paths carry the same error against the known answer; neither is
    # merely agreeing with the other's mistake.
    assert np.abs(direct["phi"][free] - exact[free]).max() < 1e-12, "direct"
    assert np.abs(amg["phi"][free] - exact[free]).max() < 1e-12, "amg"
    assert np.abs(direct["phi"] - amg["phi"]).max() < 1e-12


def test_the_amg_residual_is_checked_against_the_true_system():
    """The reported residual is measured on A and b, and is small.

    pyamg's own ``residuals`` list is relative to its internal initial residual,
    which answers a different question. This pins that the number ``solve``
    reports is the real one: recomputed here from the returned phi.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(8, 4, 4)
    volts = {"left": 0.0, "right": 1.0}
    result = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps}, volts,
                     method="amg")
    assert result["method"] == "amg"
    assert result["residual"] is not None and result["residual"] < E.AMG_RTOL * 10

    # Recompute the residual independently from the assembled system.
    K, _ = E.assemble_stiffness(points, tets, parts, {"gap": gap_eps,
                                                      "left": 1.0, "right": 1.0},
                                region_of_tet, skip={"left", "right"})
    free = result["free"]
    fixed = ~free
    phi = result["phi"]
    rhs = -(K[free][:, fixed] @ phi[fixed])
    res = K[free][:, free] @ phi[free] - rhs
    rel = np.linalg.norm(res) / np.linalg.norm(rhs)
    assert rel < 1e-10, f"recomputed relative residual {rel:.2e}"


def test_a_non_converging_amg_solve_raises_rather_than_returning_half_an_answer(monkeypatch):
    """A solver that stops early must produce an error, not a quiet answer.

    The row's requirement is that a half-converged potential is never returned
    silently. The mutation is a pyamg solver that returns its starting guess:
    the checked residual on the true system then fails the tolerance and
    :func:`amg_solve` raises.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(6, 4, 4)

    class _Lazy:
        def solve(self, b, **kw):
            return np.zeros_like(b)

    import pyamg

    monkeypatch.setattr(pyamg, "smoothed_aggregation_solver", lambda A, **kw: _Lazy())
    # A tiny maxiter is irrelevant here: _Lazy never iterates. Build the real
    # reduced operator so the residual is measured on a real matrix.
    K, _ = E.assemble_stiffness(points, tets, parts, {"gap": gap_eps,
                                                      "left": 1.0, "right": 1.0},
                                region_of_tet, skip={"left", "right"})
    free = np.ones(K.shape[0], dtype=bool)
    free[:1] = False
    K_ff = K[free][:, free].tocsc()
    b = np.ones(int(free.sum()))
    with pytest.raises(RuntimeError, match="did not converge"):
        E.amg_solve(K_ff, b)


def test_the_crossover_sends_a_small_mesh_to_the_direct_path():
    """Below AMG_NODE_CROSSOVER, method="auto" takes the exact path.

    The direct solve is exact to machine precision and needs no iteration, so
    the default must not spend it on a mesh that fits easily.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(6, 4, 4)
    result = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                     {"left": 0.0, "right": 1.0})
    assert result["n_free"] < E.AMG_NODE_CROSSOVER
    assert result["method"] == "direct"
    assert result["residual"] is None


def test_a_mesh_over_the_crossover_takes_the_amg_path_unasked(monkeypatch):
    """method="auto" switches by NODE COUNT, and the switch is exercised.

    The previous test proves auto stays DIRECT below the crossover; nothing
    proved it ever goes the other way, so the comparison could have been
    written ``<`` instead of ``>=``, or dropped entirely, with every test still
    green. The crossover is lowered below this fixture's free-node count rather
    than building a 5 x 10**5-node mesh, which tier 1 cannot.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(6, 4, 4)
    volts, eps = {"left": 0.0, "right": 1.0}, {"gap": gap_eps}

    def _nodes_free(monkeypatch_crossover):
        monkeypatch.setattr(E, "AMG_NODE_CROSSOVER", monkeypatch_crossover)
        return E.solve(points, tets, parts, region_of_tet, eps, volts)

    always = _nodes_free(E.AMG_NODE_CROSSOVER)      # the real constant: direct
    assert always["method"] == "direct"
    eager = _nodes_free(always["n_free"])           # exactly at the line: >= means AMG
    assert eager["method"] == "amg"
    assert eager["residual"] is not None
    assert eager["n_free"] == always["n_free"]

    # The two paths must still agree where the switch happened, so this is a
    # routing test and not a licence for AMG to be worse above the line.
    assert np.abs(eager["phi"] - always["phi"]).max() < 1e-12


def test_an_unknown_method_is_refused(monkeypatch):
    """A typo must not be silently read as "auto".

    Without this, ``method="amg "`` (a stray space, an f-string, a config
    default) would fall through to the direct path and behave plausibly while
    the caller believed it had asked for AMG -- the same class of quiet setting
    mismatch as MSE_SOLVER_BACKEND=cuda on the A4000 Job.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(4, 3, 3)

    # Any fall-through would call the AMG path (crossover forced to 0) AND the
    # factorisation (crossover forced huge); forbidding both makes "it was
    # treated as auto" impossible to pass by luck.
    def _forbidden(*a, **kw):
        raise AssertionError("an unknown method reached a solver")

    monkeypatch.setattr(E, "AMG_NODE_CROSSOVER", 0)
    monkeypatch.setattr(E, "amg_solve", _forbidden)
    monkeypatch.setattr(E.BACKEND, "factor", _forbidden)
    with pytest.raises(ValueError, match="not a known solve method"):
        E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                {"left": 0.0, "right": 1.0}, method="AMG")


def test_the_amg_path_actually_uses_the_amg_preconditioner():
    """AMG is used, not merely the CG it accelerates.

    Measured why this test has to exist: replacing the AMG setup with plain
    scipy CG leaves EVERY other AMG test green, because plain CG converges on
    these fixtures too. Agreement and residual tests therefore cannot tell a
    preconditioned solve from an unpreconditioned one, and without this test
    deleting the preconditioner would break nothing.

    It asserts the call, not the iterations: on a system small enough for tier
    1 the iteration counts do not separate the two, and a test that quoted them
    would be quoting the fixture's shape.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain(8, 4, 4)
    calls = []

    import pyamg

    real = pyamg.smoothed_aggregation_solver

    def spy(A, **kw):
        calls.append(A.shape)
        return real(A, **kw)

    pyamg.smoothed_aggregation_solver = spy
    try:
        result = E.solve(points, tets, parts, region_of_tet, {"gap": gap_eps},
                         {"left": 0.0, "right": 1.0}, method="amg")
    finally:
        pyamg.smoothed_aggregation_solver = real

    assert result["method"] == "amg"
    assert len(calls) == 1, f"expected one AMG setup, saw {calls}"
    assert calls[0][0] == result["n_free"], (
        "AMG was built for the wrong system: "
        f"{calls[0][0]} rows for {result['n_free']} free nodes"
    )


def test_a_pyamg_that_aborts_still_fails_the_checked_residual():
    """pyamg can return a zero vector after detecting an indefinite matrix.

    Measured: on a reduced operator it dislikes, pyamg warns "Indefinite matrix
    detected in CG, aborting" and returns x = 0 with relative residual 1.0 --
    a silent wrong answer. The checked residual is what turns that into a
    refused solve, so this pins the behaviour rather than trusting the solver.
    """
    import pyamg

    class _Aborting:
        def solve(self, b, **kw):
            return np.zeros_like(b)

    points, tets, parts, region_of_tet, gap_eps = _contact_chain(6, 4, 4)
    K, _ = E.assemble_stiffness(points, tets, parts,
                                {"gap": gap_eps, "left": 1.0, "right": 1.0},
                                region_of_tet, skip={"left", "right"})
    free = np.ones(K.shape[0], dtype=bool)
    free[:1] = False
    K_ff = K[free][:, free].tocsc()

    real = pyamg.smoothed_aggregation_solver
    pyamg.smoothed_aggregation_solver = lambda A, **kw: _Aborting()
    try:
        with pytest.raises(RuntimeError, match="did not converge"):
            E.amg_solve(K_ff, np.ones(int(free.sum())))
    finally:
        pyamg.smoothed_aggregation_solver = real


def test_skip_naming_a_part_outside_the_mesh_is_refused():
    """The skip guard is unreachable through ``solve``, so test it directly.

    ``solve`` checks every electrode is a part before it calls the assembler, so
    ``skip`` can never name an unknown part through that door -- the guard is
    dead code from ``solve``'s point of view. The reviewer's note on 1.3 asked
    for it to be either tested directly or dropped; this is the direct test,
    chosen because the assembler is a public entry point that a caller can
    reach without going through ``solve``.
    """
    points, tets = _structured_cube_chain(2, 2, 2)
    region_of_tet = np.zeros(tets.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="skip names part"):
        E.assemble_stiffness(points, tets, ["a"], {"a": 2.0}, region_of_tet,
                             skip={"nope"})


# ---------------------------------------------------------------------------
# 1.5 field_at and capacitance_matrix
#
# Units: the fixtures above are 1 m cubes so their potentials read cleanly.
# The capacitor fixtures below are built at DEVICE scale instead -- a plate pair
# 200 µm on a side with a 20 µm gap. That is not decoration: the spec's accuracy
# bounds assume fringing is small, and fringing is measured in ABSOLUTE length,
# so the same geometry at 1 m is a different (much worse) device. Sampling is
# deliberately inside materials and away from faces, never on one.
# ---------------------------------------------------------------------------

_PLATE = 200e-6      # plate side, m
_GAP_H = 20e-6       # plate separation, m


def _graded_z_chain(widths_z, side, n_side):
    """A box split into cubes in X/Y and into ``widths_z`` layers in Z.

    Reuses ``_CUBE_TETS``' six-tet split, so interfaces land on element faces
    exactly as the mesher places a part boundary.
    """
    zs = np.concatenate([[0.0], np.cumsum(widths_z)])
    xs = np.linspace(0.0, side, n_side + 1)
    nz = zs.size

    def node(i, j, k):
        return (i * (n_side + 1) + j) * nz + k

    points = np.array(
        [[xs[i], xs[j], zs[k]] for i in range(n_side + 1)
         for j in range(n_side + 1) for k in range(nz)]
    )
    tets = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(nz - 1):
                corner = {(a, b, c): node(i + a, j + b, k + c)
                          for a in (0, 1) for b in (0, 1) for c in (0, 1)}
                for t in _CUBE_TETS:
                    bits = [tuple(int(v) for v in np.unravel_index(u, (2, 2, 2))) for u in t]
                    tets.append([corner[b] for b in bits])
    return points, np.array(tets), zs


def _plate_capacitor(eps, n_gap=6, n_plate=1, n_side=6):
    """Bottom plate | ``eps`` dielectric | top plate, with open air at the sides.

    The lateral box is 2 * _PLATE so the fringing field has somewhere to go. The
    plates are ONE layer thick on purpose: with a thicker plate the fringing
    that leaves through the box's open sides still has to cross plate material,
    and the measured error stops being a clean function of the lateral extent.
    """
    widths_z = [_PLATE / n_plate, _GAP_H / n_gap, _PLATE / n_plate]
    points, tets, zs = _graded_z_chain(widths_z, 2 * _PLATE, 2 * n_side)
    cz = points[tets][:, :, 2].mean(axis=1)
    region_of_tet = np.where(cz < zs[1], 0, np.where(cz > zs[2], 2, 1)).astype(np.int64)
    return points, tets, ["bottom", "gap", "top"], region_of_tet, zs[1], zs[2]


def _three_plate_stack(eps, widths, n_side=6):
    """plate | dielectric | plate | dielectric | plate, of unequal thickness.

    Three electrodes and two gaps of DIFFERENT thickness, so the exact
    capacitance matrix is asymmetric: C(bottom, middle) and C(middle, top) are
    the two gap capacitances, and they differ. A two-electrode fixture cannot
    see a transposed matrix at all -- C and C^T are the same object there --
    which is how an earlier ``capacitance_matrix`` returned C^T unnoticed.
    """
    points, tets, zs = _graded_z_chain(widths, 2 * _PLATE, 2 * n_side)
    cz = points[tets][:, :, 2].mean(axis=1)
    layer = np.searchsorted(zs[1:-1], cz, side="right")
    return points, tets, ["p0", "d0", "p1", "d1", "p2"], layer.astype(np.int64), zs


def test_assemble_capacitance_puts_the_drive_in_the_column():
    """C[i, j] is the charge on i with j driven -- pinned on a NON-symmetric input.

    WHY THIS EXISTS. The obvious way to catch a transposed matrix is an
    asymmetric mesh, but these fixtures cannot produce one: they are series
    chains, and a Maxwell capacitance matrix is symmetric by reciprocity unless
    the electrodes form a loop. Measured max |C - C^T| is exactly 0.0, so
    writing C[j, i] instead of C[i, j] yields a bit-identical matrix and
    survived the entire suite four times. Recomputing charges through
    ``solve`` does not help either: the recomputation is symmetric the same way.

    So the convention is pinned on charge maps that are deliberately NOT
    symmetric -- which is what they would be for a device whose electrodes form
    a loop, the case the real solver must also get right.
    """
    names = ["A", "B", "C"]
    columns = [
        {"A": 10.0, "B": -7.0, "C": -3.0},
        {"A": -2.0, "B": 5.0, "C": -3.0},
        {"A": -1.0, "B": -4.0, "C": 5.0},
    ]
    C = E.assemble_capacitance(columns, names)
    for j, driven in enumerate(names):
        for i, name in enumerate(names):
            assert C[i, j] == columns[j][name], (
                f"C[{i}][{j}] must be the charge on {name!r} with {driven!r} "
                f"driven ({columns[j][name]!r}), got {C[i, j]!r}"
            )
    assert not np.array_equal(C, C.T)


def test_capacitance_matrix_columns_match_a_driven_solve():
    """End to end: each matrix column is the charge map a driven solve reports."""
    eps = 4.0
    gaps = (_GAP_H, 3 * _GAP_H)
    widths = [_PLATE, gaps[0], _PLATE, gaps[1], _PLATE]
    points, tets, parts, region_of_tet, _ = _three_plate_stack(eps, widths)
    eps_static = {"d0": eps, "d1": eps, "p0": eps, "p1": eps, "p2": eps}
    C, names = E.capacitance_matrix(points, tets, parts, region_of_tet,
                                    eps_static, {"p0": 0.0, "p1": 0.0, "p2": 0.0},
                                    method="direct")
    assert names == ["p0", "p1", "p2"]
    for j, driven in enumerate(names):
        drive = {name: (1.0 if name == driven else 0.0) for name in names}
        result = E.solve(points, tets, parts, region_of_tet, eps_static, drive,
                         method="direct")
        charge = E._electrode_charge(result)
        for i, name in enumerate(names):
            assert C[i, j] == pytest.approx(charge[name], rel=1e-9, abs=1e-24)
    # The two gaps differ by 3x, so the couplings must too.
    assert C[0, 1] / C[1, 2] == pytest.approx(3.0, rel=0.05)
    assert C[0, 0] > 0 > C[0, 1]


def test_capacitance_matrix_two_plates_is_stable_and_signed():
    """The measured pair capacitance, its sign, and that refinement does not move it.

    WHAT THIS PINS, and what it deliberately does not. The fixture is a 400 um
    box holding two 200 um-square plates across a 3.33 um gap (_GAP_H / n_gap,
    NOT _GAP_H -- the gap is subdivided so interfaces land on element faces). At
    that aspect ratio the field is mostly fringing, so the measured pair
    capacitance is 1.700e-12 F, roughly 36x the idealised eps0*eps*A/d of the
    nominal gap. That ratio is a property of the fixture, not of the solver, and
    asserting an analytic value here would be pinning my own geometry twice --
    which is exactly how an earlier version of this test ended up "confirming"
    a plate area that was off by 4x and a gap that was off by 6x.

    So the physics is checked where it IS analytic: row sums vanish for an
    earthed pair, the diagonal is positive and the coupling negative, and the
    two plates agree. What is pinned numerically is MESH INDEPENDENCE, which is
    the property that would break if the assembly were subtly wrong.
    """
    eps = 4.0
    first = None
    for n_side in (6, 12):
        points, tets, parts, region_of_tet, z0, z1 = _plate_capacitor(eps, n_side=n_side)
        eps_static = {"gap": eps, "bottom": eps, "top": eps}
        C, names = E.capacitance_matrix(points, tets, parts, region_of_tet,
                                        eps_static, {"bottom": 0.0, "top": 0.0},
                                        method="direct")
        assert names == ["bottom", "top"]
        assert C[0, 0] > 0 > C[0, 1]            # a capacitor, not an inductor
        assert C[0, 0] == pytest.approx(C[1, 1], rel=1e-9)
        assert C[0, 1] == pytest.approx(C[1, 0], rel=1e-9)
        assert abs(C.sum(axis=1)).max() < 1e-20  # earthed pair: rows cancel
        assert z1 > z0
        if first is None:
            first = C[0, 1]
            # Measured on this fixture at 200 um plates / 3.33 um gap.
            assert -C[0, 1] == pytest.approx(1.7000e-12, rel=1e-3)
        else:
            # Refining every cell edge together leaves the fringing fraction
            # alone, so the value must not move. It does not: this is the
            # assertion that would fail on a cell-size-dependent assembly.
            assert C[0, 1] == pytest.approx(first, rel=1e-6)


def test_field_at_on_an_electrode_face_reads_the_gap_not_the_electrode():
    """A point ON the electrode/gap interface must answer, not refuse.

    The reviewer's reproducer. A point lying exactly on the bottom plate's
    gap-side face is contained by BOTH plate tets and gap tets, so taking the
    lowest-index containing tet (the old code) made the answer depend on the
    mesh's numbering: sometimes a field, sometimes a ValueError -- two kinds of
    answer to one question. Reverting to ``np.flatnonzero(inside)[0]`` turns
    this red.
    """
    points, tets, parts, region_of_tet, z0, z1 = _plate_capacitor(1.0)
    sol = E.solve(points, tets, parts, region_of_tet,
                  {"gap": 1.0, "bottom": 1.0, "top": 1.0},
                  {"bottom": 0.0, "top": 1.0}, method="direct")
    # On the bottom plate's upper face == the gap's lower node plane.
    on_face = np.array([0.0, 0.0, z0])
    E_z = E.field_at(sol, on_face)[2]
    # The gap field, -dV/dz; the plate itself holds no field at all.
    assert E_z == pytest.approx(-1.0 / (z1 - z0), rel=1e-6)

    # Strictly inside the plate, every containing tet is an electrode: refused.
    with pytest.raises(ValueError, match="inside electrode 'bottom'"):
        E.field_at(sol, np.array([0.0, 0.0, z0 / 2]))


def test_capacitance_matrix_needs_an_electrode():
    points, tets, parts, region_of_tet, _, _ = _plate_capacitor(1.0)
    with pytest.raises(ValueError, match="at least one electrode"):
        E.capacitance_matrix(points, tets, parts, region_of_tet,
                             {"gap": 1.0, "bottom": 1.0, "top": 1.0}, {})


def test_field_at_matches_a_uniform_field():
    """E = -grad(phi) in the gap, against -dV/d: the sign is the whole test.

    The gap's field is V/d pointing from the top plate (z high) down to the
    bottom, so E_z is NEGATIVE, ~-V/d. Reversing the gradient's sign would make
    every component positive and the solver would still report a plausible
    magnitude, which is why the direction is asserted and not just |E|.

    Sampled at several points, one ``field_at`` call each (it takes a single
    3-vector), and all off the box's lateral centre line so the fringing field
    is not what is being measured.
    """
    eps = 1.0
    points, tets, parts, region_of_tet, z0, z1 = _plate_capacitor(eps)
    volts = {"bottom": 0.0, "top": 1.0}
    eps_static = {"gap": eps, "bottom": eps, "top": eps}
    result = E.solve(points, tets, parts, region_of_tet, eps_static, volts,
                     method="direct")
    mid = 0.5 * (z0 + z1)
    # Coordinates are METRES: the box is 2*_PLATE = 400 um wide, so these sit
    # well inside it and away from the open sides where fringing dominates.
    probes = [[1e-4, 1e-4, mid], [2e-4, 1.5e-4, mid], [1.5e-4, 2.5e-4, mid]]
    got = np.array([E.field_at(result, p) for p in probes])

    assert got.shape == (3, 3)
    # Away from the fringing edges the field is the uniform V/d, pointing down.
    assert got[:, 2] == pytest.approx(-1.0 / (z1 - z0), rel=0.02)
    assert np.abs(got[:, :2]).max() < 0.02 * abs(got[0, 2])


def test_field_at_refuses_a_point_outside_the_mesh():
    """A point in no element has no field; extrapolating would invent one."""
    points, tets, parts, region_of_tet, _, _ = _plate_capacitor(1.0)
    eps_static = {"gap": 1.0, "bottom": 1.0, "top": 1.0}
    result = E.solve(points, tets, parts, region_of_tet, eps_static,
                     {"bottom": 0.0, "top": 1.0}, method="direct")
    with pytest.raises(ValueError, match="outside"):
        E.field_at(result, [-5.0, -5.0, -5.0])  # metres: far outside a 400 um box


def test_field_at_refuses_a_point_inside_an_electrode():
    """In the held part grad(phi) is 0, and E=0 would be a zero disguised as data."""
    points, tets, parts, region_of_tet, z0, _ = _plate_capacitor(1.0)
    eps_static = {"gap": 1.0, "bottom": 1.0, "top": 1.0}
    result = E.solve(points, tets, parts, region_of_tet, eps_static,
                     {"bottom": 0.0, "top": 1.0}, method="direct")
    # Half a plate thickness in from the bottom face, at the centre of the box.
    with pytest.raises(ValueError, match="electrode"):
        E.field_at(result, [1e-4, 1e-4, 0.5 * z0])


# ---------------------------------------------------------------------------
# Task 1.7: the memory estimate and the budget refusal
# ---------------------------------------------------------------------------

def test_estimate_picks_the_path_solve_would_take():
    """The estimate must model the path that will actually run.

    Both branches are checked and they must pick DIFFERENT paths, so a mutation
    that hardcodes one path fails on the other rather than passing on both.

    ``AMG_NODE_CROSSOVER`` is 5e5, which is ABOVE the top of the measured direct
    table (112,847), so ``auto`` takes AMG there -- the path that has a model
    that wide. Forcing direct at that count is a different question and is
    covered by the refusal test: the table does not reach, so it will not answer.
    """
    below = E.estimate_memory(10**4)
    above = E.estimate_memory(E.AMG_NODE_CROSSOVER + 1)
    assert below["path"] == "direct"
    assert above["path"] == "amg"
    # Forcing a path overrides the crossover, as it does in solve().
    assert E.estimate_memory(10, method="amg")["path"] == "amg"
    assert E.estimate_memory(10**4, method="direct")["path"] == "direct"


def test_auto_chooses_amg_in_the_band_above_the_direct_table():
    """Between the table's top and the crossover, ``auto`` must still pick AMG.

    This is the product's actual mesh size. The crossover (5e5) sits ABOVE the
    direct table's top (112,847), so a size in that band has: direct chosen by
    the crossover rule, direct unpriceable by the table. An earlier version used
    the crossover alone and raised the direct refusal for every mesh in the band
    -- a refusal of a solve AMG could have done. The test above passed anyway
    because it sampled at ``crossover + 1``, on the far side of both limits.
    """
    top = E.DIRECT_MEMORY_TABLE[-1][0]
    assert top < E.AMG_NODE_CROSSOVER, "the band only exists while this holds"
    for n in (top + 1, 240_000, E.AMG_NODE_CROSSOVER - 1):
        est = E.estimate_memory(n, method="auto")
        assert est["path"] == "amg", n
        assert est["bytes"] > 0, n
    # Forcing direct up there is still refused -- the band changes what AUTO
    # picks, not what direct may claim to know.
    with pytest.raises(E.EstimateError):
        E.estimate_memory(top + 1, method="direct")


def test_estimate_scales_with_the_free_node_count():
    """The estimate rises with the unknowns, and held nodes cost nothing.

    The second half is the reason the check takes ``n_free`` rather than the node
    count: a mesh whose electrodes are most of its nodes would otherwise be
    refused for memory it never allocates.

    NOT asserted: that doubling ``n_free`` doubles the estimate. The measured
    table is not linear in ``n`` (bytes per unknown climb 5.86k -> 42.5k across
    it), so an exact 2x belongs to the old flat-rate model and asserting it here
    would pin the estimator to the shape the measurement disproved.
    """
    one = E.estimate_memory(10**4, method="direct")["bytes"]
    two = E.estimate_memory(2 * 10**4, method="direct")["bytes"]
    assert two > one
    assert E.estimate_memory(0)["bytes"] == 0.0


def test_the_model_is_measured_and_marks_its_own_extrapolation():
    """Every estimate now derives from the cluster measurement, not a guess.

    This is the flag that keeps the refusal honest. In range the figure is
    ``measured`` and not ``extrapolated``; above the AMG law's measured range it
    is still measured-derived but must SAY it is extrapolated, because a reader
    deciding whether to trust a 4 GiB number needs to know which it is.
    """
    in_range = E.estimate_memory(10**4, method="amg")
    assert in_range["measured"] is True
    assert in_range["extrapolated"] is False
    beyond = E.estimate_memory(E.AMG_MEASURED_MAX_UNKNOWNS * 4, method="amg")
    assert beyond["measured"] is True
    assert beyond["extrapolated"] is True
    # A caller-supplied rate is a test seam, not a measurement, and is never
    # allowed to claim otherwise.
    seam = E.estimate_memory(10**4, method="direct",
                             bytes_per_unknown={"direct": 1e3, "amg": 1e3})
    assert seam["measured"] is False
    assert seam["extrapolated"] is False


def test_amg_in_range_follows_the_measured_power_law():
    """AMG bytes == coefficient * n**exponent inside the fitted range, no margin."""
    law = E.AMG_MEMORY_POWER_LAW
    for n in (1_859, 10_000, E.AMG_MEASURED_MAX_UNKNOWNS):
        expect = law["coefficient"] * n ** law["exponent"]
        got = E.estimate_memory(n, method="amg")["bytes"]
        assert got == pytest.approx(expect, rel=1e-12), n
    # ~5 kB per unknown at the top of the range, as the measurement reported.
    top = E.estimate_memory(E.AMG_MEASURED_MAX_UNKNOWNS, method="amg")
    assert top["bytes_per_unknown"] / 1e3 == pytest.approx(5.0, rel=0.3)


def test_amg_above_range_applies_the_stated_margin():
    """Above the measured range the law is scaled by AMG_EXTRAPOLATION_MARGIN.

    The margin must be a real factor, not a comment: an under-predicting memory
    estimate is the one that gets a pod OOM-killed, so the test compares the
    extrapolated figure against the bare law it was grown from.
    """
    law = E.AMG_MEMORY_POWER_LAW
    n = E.AMG_MEASURED_MAX_UNKNOWNS * 10
    bare = law["coefficient"] * n ** law["exponent"]
    got = E.estimate_memory(n, method="amg")["bytes"]
    assert got == pytest.approx(bare * E.AMG_EXTRAPOLATION_MARGIN, rel=1e-12)
    assert got > bare


def test_direct_hits_every_measured_table_point_exactly():
    """A table point estimates to its own measured bytes, not near it.

    This is the check that the interpolation is anchored to the measurements:
    an off-by-a-gap error in the bracket search, or an extrapolation leaking
    into the in-range path, shows up as a point missing its own value.
    """
    for n, b in E.DIRECT_MEMORY_TABLE:
        got = E.estimate_memory(n, method="direct")["bytes"]
        assert got == pytest.approx(b, rel=1e-9), n


def test_direct_table_is_stored_in_bytes_not_mebibytes():
    """The table holds BYTES, pinned absolutely so a units slip cannot hide.

    Every other table test reads ``DIRECT_MEMORY_TABLE`` and so passes whatever
    unit it is stored in. The bench prints growth in MiB (``grow_kb / 1024``,
    since ``VmHWM`` is KiB); storing those figures as ``e6`` made every direct
    estimate 4.63 % LOW, which is the unsafe direction for an OOM guard. These
    literals are from the cluster run and must not be re-derived from the table.
    """
    top_n, top_b = E.DIRECT_MEMORY_TABLE[-1]
    assert top_n == 112_847
    # The printed figure was 4,797 MiB; in bytes that is 5.030e9, not 4.797e9.
    assert top_b == pytest.approx(5.030e9, rel=1e-3)
    assert top_b / top_n == pytest.approx(44_570, rel=2e-3)

    first_n, first_b = E.DIRECT_MEMORY_TABLE[0]
    assert first_n == 1_859
    assert first_b == pytest.approx(10.9 * 1_048_576, rel=1e-3)

    # A MiB-as-MB slip is 4.63 % low; assert the table is NOT that reading.
    assert top_b > 4_797e6 * 1.04


def test_direct_interpolates_between_measured_points():
    """Between two table entries the estimate lands between their byte counts.

    Log-log interpolation is monotone, so the midpoint of a gap sits strictly
    inside the gap's endpoints. A model that ignored the table -- the old flat
    rate -- would fail this against the measured spread.
    """
    table = E.DIRECT_MEMORY_TABLE
    for (n0, b0), (n1, b1) in zip(table, table[1:]):
        mid = E.estimate_memory((n0 + n1) // 2, method="direct")["bytes"]
        assert min(b0, b1) < mid < max(b0, b1), (n0, n1)


def test_direct_below_the_table_charges_the_smallest_measured_rate():
    """Below the table, cost is the smallest point's OWN rate, scaled linearly.

    The rate is not monotonic -- it falls 5.86 -> 5.25 kB/unknown across the
    first two points before climbing -- so "the rate climbs with size" is false
    at the bottom and a below-table estimate can read LOW by up to 11 %. That
    direction is pinned here so a future edit cannot silently claim
    conservatism the data does not support; the docstring says the same.
    """
    n0, b0 = E.DIRECT_MEMORY_TABLE[0]
    rate = b0 / n0
    for n in (1, 37, 500, n0 - 1):
        got = E.estimate_memory(n, method="direct")["bytes"]
        assert got == pytest.approx(rate * n, rel=1e-9), n
    # The rate at the second point is genuinely lower, which is why the
    # below-table branch is flagged as possibly-low rather than conservative.
    n1, b1 = E.DIRECT_MEMORY_TABLE[1]
    assert b1 / n1 < rate


def test_direct_interpolation_is_log_log_not_linear():
    """The gap is interpolated in LOG space, which is a different number.

    THIS IS THE CURVATURE CHECK. The midpoint-in-the-gap test above passes for a
    straight line between the endpoints as well as for a log-log curve, so on
    its own it cannot tell the two apart -- a mutation swapping in linear
    interpolation survived it. The two schemes disagree at the geometric mean of
    the counts, where log-log returns ``sqrt(b0*b1)``; at the log-midpoint the
    linear reading is strictly LARGER than the log-log one, so asserting the
    log-log value with a tight tolerance rejects the linear model.
    """
    table = E.DIRECT_MEMORY_TABLE
    for (n0, b0), (n1, b1) in zip(table, table[1:]):
        # Geometric mean of the counts: log(n) is exactly midway, t = 0.5.
        n_mid = math.sqrt(n0 * n1)
        n = int(round(n_mid))
        # Recompute t for the INTEGER count actually passed, so the expected
        # value is the law's own, not a rounded approximation of it.
        t = (math.log(n) - math.log(n0)) / (math.log(n1) - math.log(n0))
        want_loglog = math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
        want_linear = b0 + t * (b1 - b0)
        got = E.estimate_memory(n, method="direct")["bytes"]
        assert got == pytest.approx(want_loglog, rel=1e-9), (n0, n1)
        # And the two readings must actually be distinguishable at this point,
        # or the assertion above would not be testing curvature.
        assert want_linear > want_loglog * 1.001, (n0, n1, want_linear, want_loglog)


def test_direct_beyond_the_table_is_refused_with_a_reason_not_a_number():
    """Above the measured table direct refuses and points at AMG.

    Extrapolating a curve already 27 % off its own worst point would produce a
    confidently wrong figure; "too big to predict, use AMG" is something a
    caller can act on. The refusal is an EstimateError so it travels the same
    route as a budget refusal.
    """
    largest = E.DIRECT_MEMORY_TABLE[-1][0]
    with pytest.raises(E.EstimateError) as caught:
        E.estimate_memory(largest + 1, method="direct")
    msg = str(caught.value)
    assert f"{largest:,}" in msg
    assert "amg" in msg.lower()
    # At the largest measured point it still answers: the refusal is beyond the
    # table, not at its edge.
    assert E.estimate_memory(largest, method="direct")["bytes"] > 0


def test_budget_refusal_says_extrapolated_when_the_margin_was_applied():
    """A refusal on an extrapolated figure must not present it as measured."""
    est = E.estimate_memory(E.AMG_MEASURED_MAX_UNKNOWNS * 10, method="amg")
    assert est["extrapolated"] is True
    with pytest.raises(E.EstimateError) as caught:
        E.check_budget(est, budget=est["bytes"] / 2.0)
    assert "extrapolated" in str(caught.value)


def test_budget_refusal_names_both_the_estimate_and_the_budget():
    """The message has to carry the numbers, or a person cannot act on it.

    The count stays inside the measured direct table: past 112,847 unknowns the
    estimate itself refuses (it will not extrapolate), so a larger n here would
    test that refusal instead of this message.
    """
    est = E.estimate_memory(100_000, method="direct")
    budget = est["bytes"] / 2.0
    with pytest.raises(E.EstimateError) as caught:
        E.check_budget(est, budget=budget)
    msg = str(caught.value)
    # Both figures, in the same units, and the mesh lever that would fix it.
    assert "GiB" in msg and msg.count("GiB") >= 2
    assert "coarser mesh" in msg
    # It is a MemoryError, because that is the failure it is standing in for.
    assert isinstance(caught.value, MemoryError)


def test_budget_under_the_estimate_passes_through_untouched():
    """No refusal when the run fits, and the estimate comes back unchanged."""
    est = E.estimate_memory(10**4, method="direct")
    assert E.check_budget(est, budget=est["bytes"] * 1.5) is est
    assert E.check_budget(est, budget=est["bytes"]) is est  # exactly at the line


def test_no_device_means_no_check():
    """``budget=NO_DEVICE`` skips the check: the caller is not on the run's device.

    The web app asks on the control plane while the solve runs in a Job, so a
    figure read here would be the wrong machine's. That caller says so with
    ``NO_DEVICE`` (the same sentinel ``fdtd.check_limits`` uses) rather than
    relying on a ``None`` that would now read the wrong device.

    The estimate comes from the AMG path, which will answer at any size (its law
    extrapolates with a stated margin). The direct path refuses past its table,
    so it cannot produce the "far too big" figure this test needs.
    """
    huge = E.estimate_memory(10**12, method="amg")
    assert huge["extrapolated"] is True
    assert E.check_budget(huge, budget=E.NO_DEVICE) is huge
    # The sentinel is the time-domain one, shared rather than redefined, so the
    # two solvers cannot drift apart on what "not on the device" means.
    import fdtd
    assert E.NO_DEVICE == fdtd.NO_DEVICE


def test_budget_none_reads_this_process_device_at_call_time():
    """``budget=None`` reads the calling process's admission budget, not a constant.

    This is the spec's requirement (``admission_budget()``, as ``check_limits``
    does). Proved by asking for a budget that is unambiguously larger than any
    ceiling this process could report, and which therefore must pass: a hardcoded
    constant would have refused it.
    """
    est = E.estimate_memory(100, method="direct")  # a tiny estimate
    # Whatever the ceiling is, a 100-unknown solve at ~6 kB is a few hundred kB.
    assert E.check_budget(est, budget=None) is est
    # And the ceiling it read is real: a solve far over it is refused.
    huge = E.estimate_memory(10**5, method="direct")
    with pytest.raises(E.EstimateError):
        E.check_budget(huge, budget=1.0)


def test_solve_refuses_an_over_budget_mesh_before_assembling():
    """The end-to-end refusal, through ``solve`` itself.

    THE MUTATION TEST. With the budget check removed from ``solve`` this raises
    nothing and the solve succeeds: the mesh is tiny and would otherwise solve
    fine. So a green run here is only possible if the check is present and
    reached -- it is not a test that passes either way.
    """
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    eps_static = {"left": 1.0, "gap": gap_eps, "right": 1.0}
    electrodes = {"left": 0.0, "right": 1.0}

    # A one-byte budget refuses ANY solve, which is the cleanest proof the check
    # runs at all -- it does not depend on the estimate's rate being right.
    with pytest.raises(E.EstimateError, match="coarser mesh"):
        E.solve(points, tets, parts, region_of_tet, eps_static, electrodes,
                method="direct", budget=1)

    # And with no budget the same mesh solves, so the refusal above was the
    # budget's doing and not the mesh's.
    result = E.solve(points, tets, parts, region_of_tet, eps_static, electrodes,
                     method="direct")
    assert result["n_free"] > 0
    assert "memory_estimate" in result


def test_solve_reports_the_estimate_it_ran_against():
    """A run that was allowed should record what it was checked against."""
    points, tets, parts, region_of_tet, gap_eps = _contact_chain()
    result = E.solve(points, tets, parts, region_of_tet,
                     {"left": 1.0, "gap": gap_eps, "right": 1.0},
                     {"left": 0.0, "right": 1.0}, method="direct")
    est = result["memory_estimate"]
    assert est["n_free"] == result["n_free"]
    assert est["path"] == "direct"
    assert est["bytes"] > 0
