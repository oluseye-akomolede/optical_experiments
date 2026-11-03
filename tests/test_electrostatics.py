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
