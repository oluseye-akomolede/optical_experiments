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
    # ~1e-11 and np.allclose's default atol=1e-8 calls a 3.5% difference equal --
    # the guard would pass on the bug it exists to catch.
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
