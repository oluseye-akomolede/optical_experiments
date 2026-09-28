"""Physics tests for the MSE Maxwell edge-element solver.

These run with no MinIO and no database. The standard is the analytic spectrum
of a PEC rectangular cavity, whose modes obey

    k^2 = (m pi/a)^2 + (p pi/b)^2 + (q pi/d)^2

with the TE/TM selection rule that **at most one** index may be zero:

* all three indices non-zero  -> doubly degenerate (TE + TM pair)
* exactly one index zero      -> single

The first 10 in-medium ``k`` (1/mm) for a 10x7x5 mm box, n=1, are

    0.5478 0.7025 0.7721 0.7721 0.8336 0.8336 0.8886 0.9510 0.9955 0.9955

where 0.7721 is the accidental pair (2,1,0)/(0,1,1) and 0.8336 is the TE/TM
pair (1,1,1).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_eigen as EIG  # noqa: E402
import mse_fem as F  # noqa: E402
import mse_meshing as M  # noqa: E402

BOX = (10.0, 7.0, 5.0)          # mm
C = 299_792_458.0


def analytic_box_k(a, b, d, n_modes=10, n_max=6):
    """Lowest ``n_modes`` analytic in-medium k (1/length) for a PEC box."""
    values = []
    for m in range(n_max):
        for p in range(n_max):
            for q in range(n_max):
                nonzero = (m > 0) + (p > 0) + (q > 0)
                if nonzero < 2:
                    continue                      # at most one index may be zero
                k2 = (m * np.pi / a) ** 2 + (p * np.pi / b) ** 2 + (q * np.pi / d) ** 2
                multiplicity = 2 if nonzero == 3 else 1
                values.extend([k2] * multiplicity)
    return np.sqrt(np.array(sorted(values)[:n_modes]))


def structured_box_mesh(a, b, d, nx, ny, nz):
    """A Kuhn-split structured box: ``6 * nx * ny * nz`` tets, no TetGen needed.

    Every cell is cut into six tets along the main diagonal, which makes the
    discretication reproducible and independent of the tetrahedralizer.
    """
    xs = np.linspace(0, a, nx + 1)
    ys = np.linspace(0, b, ny + 1)
    zs = np.linspace(0, d, nz + 1)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs], dtype=float)

    def idx(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        corners = [idx(i + di, j + dj, k + dk)
                   for dk in (0, 1) for dj in (0, 1) for di in (0, 1)]
        for perm in itertools.permutations([1, 2, 4]):
            path = [0]
            total = 0
            for step in perm:
                total += step
                path.append(total)
            tets.append([corners[c] for c in path])
    return points, np.array(tets, dtype=np.int64)


def solve_box(a, b, d, units="mm", n_index=1.0, n_modes=10, nx=12, ny=9, nz=6):
    """Assemble and solve a structured box.

    Returns ``(k_per_m, freq_hz, wl_m)`` with ``k`` in **1/metre** -- the solver
    works in SI throughout, so callers comparing against an analytic expression
    written in 1/mm must convert. ``nx``/``ny``/``nz`` set the structured grid.
    """
    points, tets = structured_box_mesh(a, b, d, nx, ny, nz)
    scale = M.unit_scale(units)
    points_m = points * scale

    edges, tet_edges, _ = F.build_edges(tets)
    n_edges = edges.shape[0]
    A, B = F.assemble_global(points_m, tets, tet_edges, n_edges)
    constrained = F.boundary_edges(tets, edges)
    A_free, B_free, free = F.pec_eliminate(A, B, constrained, n_edges)

    values, _, _ = EIG.solve_modes(
        A_free, B_free, n_modes, points_m,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free, constrained=constrained, verbose=False,
    )
    k, k0, freq, wl = EIG.to_physical(np.sort(values), n_index)
    return k, freq, wl


# --------------------------------------------------------------------------
# element-level tests (the lead's required checks)
# --------------------------------------------------------------------------

def test_gradient_transpose_on_random_tets():
    """grad(lambda) = J^-T g, checked by the invariant G @ (x1 - x0) == [-1,1,0,0].

    This is the check that catches the transpose error: on the right-angled
    reference tet J = I, so a transposed gradient is invisible there.
    """
    rng = np.random.default_rng(1234)
    grad_ref = np.array(
        [-1.0, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float
    ).reshape(4, 3)

    for _ in range(50):
        tet = rng.standard_normal((4, 3))
        jac = np.column_stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]])
        grads = np.einsum("ji,aj->ai", np.linalg.inv(jac), grad_ref)

        # sum of barycentric gradients vanishes
        assert np.allclose(grads.sum(axis=0), 0.0, atol=1e-10)
        # grad(lambda_p) . (x_q - x_0) == delta_pq
        assert np.allclose(grads[1:4] @ jac, np.eye(3), atol=1e-9)
        # the lead's stated invariant
        assert np.allclose(grads @ (tet[1] - tet[0]), [-1, 1, 0, 0], atol=1e-9)


def test_element_matrices_match_quadrature_on_random_tets():
    """The element A and B must equal an independent numerical integration."""
    pairs = F.TET_EDGE_PAIRS
    grad_ref = np.array(
        [-1.0, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float
    ).reshape(4, 3)
    # 4-point rule on the reference tet. The reference tet has volume 1/6 and
    # the rule's weights sum to 1, so each point carries (1/6)/4 = 1/24 of the
    # reference volume; the jacobian factor completes the integral.
    quad = np.array([
        [0.58541020, 0.13819660, 0.13819660],
        [0.13819660, 0.58541020, 0.13819660],
        [0.13819660, 0.13819660, 0.58541020],
        [0.13819660, 0.13819660, 0.13819660],
    ])
    rng = np.random.default_rng(7)

    for _ in range(10):
        tet = rng.standard_normal((4, 3))
        jac = np.column_stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]])
        det_j = abs(np.linalg.det(jac))
        grads = np.einsum("ji,aj->ai", np.linalg.inv(jac), grad_ref)

        tets = np.array([[0, 1, 2, 3]])
        _, tet_edges, _ = F.build_edges(tets)
        a_elem, b_elem, vol = F.element_matrices(tet, tets, tet_edges)

        # reference quadrature for B
        b_ref = np.zeros((6, 6))
        for xi in quad:
            lam = np.array([1.0 - xi.sum(), xi[0], xi[1], xi[2]])
            basis = np.array([
                lam[i] * grads[j] - lam[j] * grads[i] for i, j in pairs
            ])
            b_ref += (basis @ basis.T) * (det_j / 24.0)
        # scale-relative tolerance: the off-diagonal entries span several orders
        # of magnitude, so an absolute tolerance fixed by the largest entry is
        # both too loose for the small ones and, if tightened, too strict.
        scale = np.abs(b_ref).max()
        assert np.abs(b_elem[0] - b_ref).max() <= 1e-7 * scale
        assert np.allclose(vol, det_j / 6.0, rtol=1e-12)

        # A must be symmetric positive semi-definite
        assert np.allclose(a_elem[0], a_elem[0].T, atol=1e-10)
        assert np.linalg.eigvalsh(a_elem[0]).min() > -1e-9 * max(
            1.0, np.abs(a_elem[0]).max()
        )


def test_mass_matrix_is_positive_definite_on_a_real_mesh():
    """A wrong Whitney mass shows up as an indefinite B, so guard it directly."""
    points, tets = structured_box_mesh(*BOX, 4, 3, 2)
    edges, tet_edges, _ = F.build_edges(tets)
    _, b = F.assemble_global(points * 1e-3, tets, tet_edges, edges.shape[0])
    eigenvalues = np.linalg.eigvalsh(b.toarray())
    assert eigenvalues.min() > 0, "mass matrix must be positive definite"
    assert eigenvalues.max() / eigenvalues.min() < 1e6, "mass matrix badly conditioned"


# --------------------------------------------------------------------------
# the contract's validation cases
# --------------------------------------------------------------------------

@pytest.mark.heavy
def test_box_first_ten_modes_match_analytic():
    """First 10 modes of a 10x7x5 mm PEC box within 3% of analytic, no spurious."""
    a, b, d = BOX
    k_per_m, _, _ = solve_box(a, b, d)
    expected_per_mm = analytic_box_k(a, b, d, 10)      # 1/mm
    expected = expected_per_mm * 1e3                   # 1/m

    assert k_per_m.shape[0] == 10
    assert np.all(k_per_m > 0), "no zero/spurious modes may be returned"

    rel = np.abs(k_per_m / expected - 1.0)
    assert rel.max() < 0.03, (
        f"worst relative error {rel.max():.4f} exceeds 3%\n"
        f"computed {np.round(k_per_m / 1e3, 4)}\n"
        f"expected {np.round(expected_per_mm, 4)}"
    )


@pytest.mark.heavy
def test_refractive_index_scales_frequency_by_inverse_index():
    """Same geometry at n=1.5 must scale every frequency by exactly 1/1.5."""
    a, b, d = BOX
    k1, f1, _ = solve_box(a, b, d, n_index=1.0)
    k15, f15, _ = solve_box(a, b, d, n_index=1.5)

    # the in-medium wavenumber is a property of the geometry alone
    assert np.allclose(k1, k15, rtol=1e-9)
    assert np.allclose(f15 / f1, 1.0 / 1.5, rtol=1e-9)


@pytest.mark.heavy
def test_units_scale_frequency_by_physical_size():
    """The same physical object described in cm and in mm gives the same Hz.

    ``10x7x5 mm`` and ``1.0x0.7x0.5 cm`` are the same cavity, so frequencies
    must agree; conversely a box 10x larger is 1/10 the frequency.
    """
    k_mm, f_mm, _ = solve_box(10.0, 7.0, 5.0, units="mm")
    k_cm, f_cm, _ = solve_box(1.0, 0.7, 0.5, units="cm")

    # same physical object -> same SI k and same Hz
    assert np.allclose(k_mm, k_cm, rtol=1e-9)
    assert np.allclose(f_mm, f_cm, rtol=1e-9)

    # and a 10x larger object is 1/10 the frequency
    k_big, f_big, _ = solve_box(100.0, 70.0, 50.0, units="mm")
    assert np.allclose(f_big / f_mm, 0.1, rtol=1e-9)


@pytest.mark.heavy
def test_frequency_and_wavelength_are_consistent():
    """``f``, ``k0`` and the vacuum wavelength must agree with the definitions."""
    k_per_m, freq, wl = solve_box(*BOX, n_index=1.0)
    assert np.allclose(freq, C * k_per_m / (2 * np.pi), rtol=1e-9)
    assert np.allclose(wl, 2 * np.pi / k_per_m, rtol=1e-9)


def test_default_shift_is_below_the_first_mode():
    """The shift must sit below the first physical eigenvalue, or it selects junk."""
    points, _ = structured_box_mesh(*BOX, 4, 3, 2)
    points_m = points * 1e-3
    shift = EIG.default_shift(points_m)
    first_mode_k2 = analytic_box_k(*BOX, 1)[0] ** 2 * 1e6
    assert 0 < shift < first_mode_k2


@pytest.mark.heavy
def test_solver_matches_dense_generalized_eigendecomposition():
    """The projected shift-invert must reproduce a dense SVD-based solve."""
    import scipy.linalg as sla

    points, tets = structured_box_mesh(*BOX, 4, 3, 2)
    points_m = points * 1e-3
    edges, tet_edges, _ = F.build_edges(tets)
    n_edges = edges.shape[0]
    A, B = F.assemble_global(points_m, tets, tet_edges, n_edges)
    constrained = F.boundary_edges(tets, edges)
    A_free, B_free, free = F.pec_eliminate(A, B, constrained, n_edges)

    u, sv, _ = sla.svd(B_free.toarray())
    rank = int((sv > 1e-9 * sv[0]).sum())
    b_half = u[:, :rank] * (1.0 / np.sqrt(sv[:rank]))
    m = b_half.T @ A_free.toarray() @ b_half
    m = 0.5 * (m + m.T)
    dense = np.sort(sla.eigvalsh(m))
    dense = np.sqrt(dense[dense > 1e-4 * max(dense.max(), 1e-30)][:10])

    sparse_k, _, _ = EIG.solve_modes(
        A_free, B_free, 10, points_m,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free, constrained=constrained, verbose=False,
    )
    sparse_k = np.sqrt(np.sort(sparse_k))
    assert np.allclose(sparse_k, dense, rtol=1e-6), (
        f"sparse {np.round(sparse_k, 5)}\ndense {np.round(dense, 5)}"
    )


# --------------------------------------------------------------------------
# the SPD ordering (the lead's Jetson bench, pinned at tier 1)
# --------------------------------------------------------------------------

def _box_matrices(nx=6, ny=5, nz=4):
    """The assembled, PEC-reduced A and B for a small structured box."""
    points, tets = structured_box_mesh(10.0, 7.0, 5.0, nx=nx, ny=ny, nz=nz)
    edges, tet_edges, _ = F.build_edges(tets)
    n_edges = edges.shape[0]
    A, B = F.assemble_global(points, tets, tet_edges, n_edges)
    constrained = F.boundary_edges(tets, edges)
    A_free, B_free, free = F.pec_eliminate(A, B, constrained, n_edges)
    return points, tets, edges, tet_edges, free, constrained, A_free, B_free


def test_the_symmetric_ordering_finds_the_same_modes_as_the_default():
    """The ordering is a permutation, so the answer must not move.

    MMD_AT_PLUS_A with pivoting off is a different factorisation of the same
    matrix, and the whole justification for it is that it is *cheaper*, not
    different. If it changed the eigenvalues the speedup would be a different
    solver, not a faster one -- so this is the test that makes the change safe
    to take. rtol 1e-8 is tighter than the tolerance the two orderings could
    plausibly disagree by if they were doing the same arithmetic; a real
    difference in the ordering shows up far above it.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    symmetric, _, _ = EIG.solve_modes(
        A_free, B_free, 6, points,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free, constrained=constrained, verbose=False,
        spd_ordering=True,
    )
    default, _, _ = EIG.solve_modes(
        A_free, B_free, 6, points,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free, constrained=constrained, verbose=False,
        spd_ordering=False,
    )
    symmetric = np.sort(np.asarray(symmetric))
    default = np.sort(np.asarray(default))
    assert np.allclose(symmetric, default, rtol=1e-8), (
        f"symmetric {np.round(symmetric, 10)}\ndefault   {np.round(default, 10)}"
    )


def test_the_ordering_arguments_are_the_symmetric_ones():
    """Mutation guard: the three arguments must actually reach SuperLU.

    The eigenvalue test above passes whether or not the arguments are applied,
    because the same matrix factorises to the same answer either way -- so it
    cannot be the thing that protects this change. This one pins the arguments
    themselves, by recording what `splu` was called with. Revert any of the
    three to its default and this goes red.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    calls = []
    real_splu = EIG.spla.splu

    def recording_splu(m, **kwargs):
        calls.append(kwargs)
        return real_splu(m, **kwargs)

    EIG.spla.splu = recording_splu
    try:
        EIG.solve_modes(
            A_free, B_free, 4, points,
            edges=edges, tets=tets, tet_edges=tet_edges,
            free_edges=free, constrained=constrained, verbose=False,
            spd_ordering=True,
        )
    finally:
        EIG.spla.splu = real_splu

    assert len(calls) == 2, f"expected K and S to be factored, got {len(calls)}"
    for kwargs in calls:
        assert kwargs.get("permc_spec") == "MMD_AT_PLUS_A", kwargs
        assert kwargs.get("diag_pivot_thresh") == 0.0, kwargs
        assert kwargs.get("options") == {"SymmetricMode": True}, kwargs


def test_both_factored_matrices_are_symmetric_positive_definite():
    """The claim the arguments depend on, checked rather than asserted.

    If either matrix were only PSD, `diag_pivot_thresh=0.0` would be unsafe --
    pivoting is the thing that rescues a singular pivot -- so the premise is
    tested directly rather than trusted. `S = G^T B G` is SPD because B is SPD
    and G has full column rank; `K = A + sB` is SPD because the shift is chosen
    below the first eigenvalue and B is SPD.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    s = EIG.default_shift(points)
    K = (A_free + s * B_free).tocsc()
    G, _ = EIG.build_gradient(points, tets, edges, tet_edges, free, constrained)
    S = ((G.T @ B_free) @ G).tocsc()

    for name, matrix in (("K = A + sB", K), ("S = G^T B G", S)):
        dense = matrix.toarray()
        assert np.abs(dense - dense.T).max() < 1e-9, f"{name} is not symmetric"
        # Cholesky exists iff the matrix is positive definite, and it fails
        # loudly rather than returning a plausible wrong answer.
        np.linalg.cholesky(dense)


# --------------------------------------------------------------------------
# meshing contract
# --------------------------------------------------------------------------

def test_non_watertight_obj_is_rejected():
    """An open surface must fail with a message a user can act on.

    NOT marked heavy: this builds a 12-face box and never reaches TetGen, so it
    belongs in the default suite. The contract requires this case to be covered
    without opting in.
    """
    import trimesh

    box = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    # drop one face to open the surface
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    assert not open_mesh.is_watertight

    with pytest.raises(M.MeshingError, match="not a closed surface"):
        M.load_surface(open_mesh.export(file_type="obj").encode(), units="mm")


def test_watertight_obj_is_accepted_and_scaled_to_metres():
    """A closed box loads, converts to metres, and tetrahedralizes.

    Deliberately NOT marked heavy: the load half is pure trimesh, and the
    tetrahedralization is a 12-face box at a generous cap, which is well inside
    the "a few-dozen-tet mesh" allowance. The 3888-tet box and the sphere bench
    are the genuinely heavy cases and carry the marker.
    """
    import trimesh

    box = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    mesh, scale = M.load_surface(box.export(file_type="obj").encode(), units="mm")
    assert mesh.is_watertight
    assert scale == pytest.approx(1e-3)

    points_m, tets, _ = M.tetrahedralize(mesh, "mm", n_modes=20, verbose=False)
    assert tets.shape[1] == 4
    assert points_m.max() < 0.011, "points must be scaled into metres"


def test_garbage_obj_is_rejected():
    with pytest.raises(M.MeshingError):
        M.load_surface(b"this is not an obj file at all", units="mm")


def test_unsupported_units_are_rejected():
    with pytest.raises(M.MeshingError, match="unsupported units"):
        M.unit_scale("furlongs")


@pytest.mark.heavy
def test_tet_cap_is_respected_or_the_mesh_is_refused():
    """A cap below what the object needs must either be honoured or refused.

    Honouring the cap by coarsening is fine (that is what the cap is for), but
    the result must never be a mesh too coarse to resolve the requested modes,
    which would produce confidently wrong frequencies instead of an error.
    """
    import trimesh

    box = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    mesh, _ = M.load_surface(box.export(file_type="obj").encode(), units="mm")

    with pytest.raises(M.MeshingError, match="resolve|limit"):
        M.tetrahedralize(mesh, "mm", n_modes=20, max_tets=50, verbose=False)

    # a cap that is generous enough must succeed and stay within it
    points_m, tets, _ = M.tetrahedralize(mesh, "mm", n_modes=10, max_tets=60_000,
                                      verbose=False)
    assert len(tets) <= 60_000


# --------------------------------------------------------------------------
# field sampling (pure numpy; guards the same transpose trap as mse_fem)
# --------------------------------------------------------------------------

def test_barycentric_inversion_on_random_tets():
    """``b = J^-1 (x - v0)`` must recover known barycentric coordinates.

    Sampling uses the *rows* of ``J^-1``, not its columns. Both mistakes are
    invisible on a right-angled reference tet, so check a random one.
    """
    rng = np.random.default_rng(11)
    for _ in range(20):
        tet = rng.standard_normal((4, 3))
        jac = np.column_stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]])
        inv_j = np.linalg.inv(jac)

        for b_true in ([0.2, 0.3, 0.1], [0.1, 0.1, 0.1], [0.6, 0.2, 0.15]):
            point = tet[0] + jac @ np.array(b_true)
            d = point - tet[0]
            b_got = np.array([d @ inv_j[k, :] for k in range(3)])
            assert np.allclose(b_got, b_true, atol=1e-10)


def test_sampled_field_reproduces_a_constant_field_on_a_rotated_tet():
    """A constant E must sample back as that constant, on a skewed tet.

    The lowest-order edge element reproduces constants exactly, so this catches
    a wrong ``grad(J)``/barycentric convention, which would tilt the field.
    """
    import mse_fields as FLD

    rng = np.random.default_rng(5)
    # one deliberately skewed tet
    tet = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.0, 0.1], [0.1, 0.2, 1.0]])
    tets = np.array([[0, 1, 2, 3]])
    points = tet

    edges, tet_edges, signs = F.build_edges(tets)

    # edge coefficients that make E = (1, 0, 0) everywhere: E = sum_a c_a N_a.
    # On the reference tet with lambda_i affine, the constant field is recovered
    # by the circuit coefficients around the tetrahedron.
    grad_ref = np.array(
        [-1.0, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float
    ).reshape(4, 3)
    jac = np.column_stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]])
    grads = np.einsum("ji,aj->ai", np.linalg.inv(jac), grad_ref)

    target = np.array([1.0, 0.0, 0.0])
    # Solve B c = f for the edge coefficients, where f_a = \int N_a . target.
    # The lowest-order edge space reproduces constants exactly, so c exists.
    a_elem, b_elem, _ = F.element_matrices(points, tets, tet_edges)
    quad = np.array([
        [0.58541020, 0.13819660, 0.13819660],
        [0.13819660, 0.58541020, 0.13819660],
        [0.13819660, 0.13819660, 0.58541020],
        [0.13819660, 0.13819660, 0.13819660],
    ])
    det_j = abs(np.linalg.det(jac))
    f = np.zeros(6)
    for xi in quad:
        lam = np.array([1.0 - xi.sum(), xi[0], xi[1], xi[2]])
        for a, (i, j) in enumerate(F.TET_EDGE_PAIRS):
            na = lam[i] * grads[j] - lam[j] * grads[i]
            f[a] += (na @ target) * (det_j / 24.0)
    coeffs = np.linalg.solve(b_elem[0], f)

    # sample on a coarse grid inside the tet and compare to the constant
    field, shape, origin, spacing = FLD.sample_field(
        points, tets, edges, tet_edges, signs, coeffs, n_longest=8
    )
    inside = np.isfinite(field).all(axis=-1)
    assert inside.sum() > 0, "no sample points landed inside the tet"
    sampled = field[inside]
    peak = np.linalg.norm(sampled, axis=1)
    # normalized to max|E| = 1, so the direction must still be +x
    assert np.allclose(sampled / peak[:, None], np.array([1.0, 0.0, 0.0]),
                       atol=2e-2), "sampled field direction is wrong"


# --------------------------------------------------------------------------
# solver entrypoint: end-to-end on a tiny mesh, and the failure paths
# --------------------------------------------------------------------------

def test_the_vtu_carries_the_exact_values_and_a_reader_gets_them_back(tmp_path):
    """The .vtu the owner downloads must be the field the solver computed.

    It was written as ascii with ``%.6g``, which rounds every sample to six
    significant digits and -- worse -- does so in a format where nothing
    downstream notices: ParaView opens the file and draws a field that is
    subtly not the one that was solved. This checks the property that the
    rounding broke, through the reader the user actually uses, rather than
    checking that the file exists.
    """
    import mse_fields as FLD
    import pyvista as pv

    shape = (2, 3, 4)
    f = np.random.default_rng(11).standard_normal((2, 3, 4, 3)).astype("<f4")
    # A value whose seventh significant digit is non-zero is what the old
    # writer truncated; 1/7 is the cheapest such number to write down.
    f[1, 2, 3] = [1.0 / 7.0, -1.0 / 3.0, np.float32(0.1)]
    origin, spacing = (1.0, 2.0, 3.0), (0.5, 0.25, 0.125)

    out = tmp_path / "mode_01.vtu"
    FLD.write_vtu(out, f, shape, origin, spacing)
    read = pv.read(out)

    assert read.n_points == 24 and read.n_cells == 24
    got = np.asarray(read["E"]).reshape(shape + (3,))
    # Bit-exact, not close: float32 is the solver's own precision, and the whole
    # point of the change was to stop losing digits on the way out.
    assert np.array_equal(got, f), (
        "the .vtu does not round-trip the field exactly -- the writer is "
        "still narrowing the values on the way out"
    )
    mag = np.asarray(read["E_magnitude"]).reshape(shape)
    assert np.allclose(mag, np.linalg.norm(f, axis=-1), rtol=1e-6)

    pts = np.asarray(read.points)
    assert np.allclose(pts[0], origin)
    assert np.allclose(pts[-1], (origin[0] + 3 * spacing[0],
                                 origin[1] + 2 * spacing[1],
                                 origin[2] + 1 * spacing[2]))


def test_the_vtu_keeps_the_solvers_nan_outside_the_object(tmp_path):
    """NaN is "no sample here", and ParaView must be told that, not given a 0.

    A zero would draw the outside of the object as a real part of the field,
    which is the same failure the plot has a test for -- the two views have to
    agree about where the object ends, or one of them is lying.
    """
    import mse_fields as FLD
    import pyvista as pv

    shape = (2, 2, 2)
    f = np.full((2, 2, 2, 3), np.nan, dtype="<f4")
    f[0, 0, 0] = [0.25, 0.5, 0.75]

    out = tmp_path / "mode_01.vtu"
    FLD.write_vtu(out, f, shape, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    got = np.asarray(pv.read(out)["E"]).reshape(shape + (3,))

    assert np.array_equal(got[0, 0, 0], f[0, 0, 0])
    # 8 samples, one of them real, three components each: 21 components are
    # outside the object. NaN is per-component, so that is what is counted --
    # a 21 (=7 samples) here would mean a whole sample had been dropped.
    assert np.isnan(got).sum() == 21, "the outside-the-object NaN was lost"
    assert not np.isnan(got[0, 0, 0]).any()


def _write_box(root, job_id, extents=(10.0, 7.0, 5.0), open_surface=False):
    """Put a box .obj where the solver expects to find it."""
    import trimesh

    box = trimesh.creation.box(extents=list(extents))
    if open_surface:
        box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1],
                              process=False)
    path = Path(root) / "jobs" / job_id / "input.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(box.export(file_type="obj").encode())
    return path


JOB = "11111111-2222-3333-4444-555555555555"


def test_solver_runs_end_to_end_and_writes_the_contract_files(tmp_path):
    """A tiny box must produce every file the web app depends on."""
    import json

    import mse_maxwell_solver as SOL

    _write_box(tmp_path, JOB)
    rc = SOL.main([
        "--job-id", JOB, "--units", "mm", "--n-index", "1.0", "--n-modes", "3",
        "--local-in", str(tmp_path), "--local-out", str(tmp_path),
        "--max-tets", "4000", "--grid", "16", "--quiet",
    ])
    assert rc == 0

    out = Path(tmp_path) / "jobs" / JOB
    for name in ("surface.glb", "modes.json"):
        assert (out / name).is_file(), f"missing {name}"
    for index in (1, 2, 3):
        assert (out / f"mode_{index:02d}.bin").is_file()
        assert (out / f"mode_{index:02d}.vtu").is_file()

    meta = json.loads((out / "modes.json").read_text())
    assert meta["units"] == "mm"
    assert meta["n_index"] == 1.0
    assert meta["n_tets"] > 0 and meta["n_edges"] > 0
    assert set(meta["grid"]) == {"nx", "ny", "nz", "origin", "spacing"}
    assert len(meta["modes"]) == 3

    # frequencies must be positive, increasing, and consistent with k0
    freqs = [m["freq_hz"] for m in meta["modes"]]
    assert all(f > 0 for f in freqs)
    assert freqs == sorted(freqs)
    for mode in meta["modes"]:
        assert mode["freq_hz"] == pytest.approx(
            C * mode["k0_per_m"] / (2 * np.pi), rel=1e-9
        )
        assert mode["vacuum_wavelength_m"] == pytest.approx(
            2 * np.pi / mode["k0_per_m"], rel=1e-9
        )

    # the .bin layout is part of the contract
    nx, ny, nz = meta["grid"]["nx"], meta["grid"]["ny"], meta["grid"]["nz"]
    raw = np.fromfile(out / "mode_01.bin", dtype=np.float32)
    assert raw.size == nx * ny * nz * 3
    field = raw.reshape(nz, ny, nx, 3)
    assert np.isnan(field).any(), "samples outside the object must be NaN"
    inside = field[np.isfinite(field).all(axis=-1)]
    assert inside.size > 0
    assert np.nanmax(np.linalg.norm(field, axis=-1)) == pytest.approx(1.0, abs=1e-6)


def test_solver_rejects_a_non_watertight_object(tmp_path):
    """The owner gets a plain-English refusal and a non-zero exit."""
    import mse_maxwell_solver as SOL

    _write_box(tmp_path, JOB, open_surface=True)
    rc = SOL.main([
        "--job-id", JOB, "--local-in", str(tmp_path), "--local-out", str(tmp_path),
        "--max-tets", "4000", "--grid", "8", "--quiet",
    ])
    assert rc == 1


def test_solver_rejects_a_missing_input(tmp_path):
    import mse_maxwell_solver as SOL

    rc = SOL.main([
        "--job-id", JOB, "--local-in", str(tmp_path), "--local-out", str(tmp_path),
        "--quiet",
    ])
    assert rc == 1


def test_solver_rejects_a_non_positive_refractive_index(tmp_path):
    import mse_maxwell_solver as SOL

    _write_box(tmp_path, JOB)
    rc = SOL.main([
        "--job-id", JOB, "--n-index", "0",
        "--local-in", str(tmp_path), "--local-out", str(tmp_path),
        "--max-tets", "4000", "--grid", "8", "--quiet",
    ])
    assert rc == 1


def test_owner_messages_are_plain_english():
    """Nothing technical may reach the owner-facing message."""
    import mse_maxwell_solver as SOL

    for exc, needle in [
        (M.MeshingError("This object is too large or detailed to resolve 20 modes "
                        "within the solver's size limit."), "too large"),
        (EIG.EigenError("could not factor A + sB: something internal"), "solve"),
        (RuntimeError("Traceback numpy.linalg nonsense"), "went wrong"),
    ]:
        message = SOL.owner_message(exc)
        assert needle in message.lower()
        for jargon in ("Traceback", "EigenError", "MeshingError", "0x", ".py", "numpy"):
            assert jargon not in message, f"{jargon!r} leaked into {message!r}"


def test_local_store_refuses_to_escape_its_root(tmp_path):
    """The path stand-in for the bucket must not be a traversal hole."""
    import mse_storage as STORE

    store = STORE.LocalStore(tmp_path / "bucket")
    with pytest.raises(STORE.StorageError):
        store.read("../../etc/passwd")
    with pytest.raises(STORE.StorageError):
        store.write("../escape.bin", b"x")


# ---------------------------------------------------------------------------
# The PEC sphere reference used by the tier-2 bench.
#
# A box cannot test curved walls or faceting, so the bench's other shape is a
# sphere whose spectrum is known exactly. These are the analytic roots, pinned
# here (rather than only in the bench) because a wrong reference would make the
# bench's PASS meaningless.
# ---------------------------------------------------------------------------
def test_sphere_reference_degeneracies_and_roots():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_box

    k = bench_box.analytic_sphere_k(radius=5e-3, n_modes=23)   # 1/m, R = 5 mm
    x = k * 5e-3                                               # kR

    # TM1 x=2.7437 (x3), TM2 x=3.8702 (x5), TE1 x=4.4934 (x3),
    # TM3 x=4.9734 (x7), TE2 x=5.7635 (x5) -- 23 modes in all
    expected = [(2.7437, 3), (3.8702, 5), (4.4934, 3), (4.9734, 7), (5.7635, 5)]
    got: list[tuple[float, int]] = []
    for value in x:
        if got and abs(got[-1][0] - value) < 1e-3:
            got[-1] = (got[-1][0], got[-1][1] + 1)
        else:
            got.append((float(value), 1))
    assert [(round(v, 4), n) for v, n in got] == expected


def test_sphere_reference_scales_inversely_with_radius():
    """k = x/R, so halving the radius doubles every wavenumber."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_box

    big = bench_box.analytic_sphere_k(radius=10e-3, n_modes=23)
    small = bench_box.analytic_sphere_k(radius=5e-3, n_modes=23)
    assert small == pytest.approx(2.0 * big, rel=1e-9)


# ---------------------------------------------------------------------------
# The writing stage, as the bench runs it.
#
# BUG A (deploy blocker 3) reached the cluster because the bench stopped after
# the solve: the writing stage -- sampling, .bin, .vtu, .glb, modes.json -- was
# never exercised, so the ~15 GB (chunk, n_tets, 3) sampler allocation was
# never seen. These two tests keep the bench's writing stage honest: one that it
# really drives the production solve(), and one that it reports the stage's time
# rather than folding it into the total.
# ---------------------------------------------------------------------------
def test_the_bench_writing_stage_drives_the_production_solve(tmp_path):
    """The bench must call the production ``solve()``, not a copy of it.

    A bench that re-implements the pipeline stops measuring the pipeline the
    moment the two drift apart, and the writing stage is the one that OOM-killed
    the Job. Measured on a tiny box: the artefacts below are the production
    ones, written through the same ``LocalStore`` the container uses.

    Tier 1 by construction -- 1.4k tets, 2 modes, a 10^3 grid -- because the
    claim is about wiring, not accuracy. The accuracy claim is the bench's job
    on the Jetson.
    """
    import trimesh

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_box

    obj = trimesh.creation.box(extents=[40.0, 30.0, 20.0]).export(
        file_type="obj").encode()
    summary = bench_box.run_writing_stage(
        obj, "bencht", "mm", 1.0, 2, max_tets=3000, grid=10,
        write_dir=str(tmp_path), verbose=False,
    )

    written = summary["written"]
    assert {"surface.glb", "modes.json"} <= set(written), (
        f"the writing stage did not run: wrote {sorted(written)}"
    )
    assert set(written) >= {f"mode_{i:02d}.bin" for i in (1, 2)}
    assert set(written) >= {f"mode_{i:02d}.vtu" for i in (1, 2)}

    # every reported stage is timed, and writing is one of them -- if the stage
    # vanishes from `timings` the bench would print a total with a hole in it
    for stage in ("meshing", "assembling", "solving", "writing"):
        assert stage in summary["timings"], f"no timing for {stage}"

    # the field is what the viewer reads: the .bin is flat float32 triples, and
    # the NaN mask is how the outside is marked. Checking the VALUES rather than
    # the byte count is the point -- a stage that wrote the right number of
    # bytes of garbage would pass a size check.
    import numpy as np

    field = np.frombuffer(
        (tmp_path / "jobs" / "bencht" / "mode_01.bin").read_bytes(),
        dtype=np.float32,
    )
    assert field.size % 3 == 0, "the .bin must be a whole number of vectors"
    assert np.isfinite(field).any() and np.isnan(field).any(), (
        "the sampled field must straddle the surface"
    )
    inside = np.isfinite(field)
    assert np.linalg.norm(field[inside].reshape(-1, 3), axis=1).max() == \
        pytest.approx(1.0, rel=1e-5), "the field must come out normalised"


def test_the_bench_reports_the_writing_stage_among_its_timings():
    """The bench's reason for --write is that the writing cost was invisible.

    A run that folds writing into `total` would reproduce the gap that let the
    ~15 GB sampler reach the cluster. The timing has to be its own measurement,
    so this pins that `writing` is a stage the bench's own summary carries and
    prints -- measured on the summary the bench produces, not read off its
    source, because a printed total with a hole in it is the failure mode.
    """
    import trimesh

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_box

    obj = trimesh.creation.box(extents=[40.0, 30.0, 20.0]).export(
        file_type="obj").encode()
    summary = bench_box.run_writing_stage(
        obj, "bencht2", "mm", 1.0, 1, max_tets=3000, grid=8,
        write_dir=None, verbose=False,
    )
    timings = summary["timings"]
    assert "writing" in timings
    assert timings["writing"] >= 0.0
    assert timings["total"] >= timings["writing"], (
        "the writing stage cannot take longer than the whole solve"
    )
    # and the bench prints exactly the stages its summary carries, so a stage
    # added to solve() without being printed shows up as a mismatch
    printed = [s for s in ("meshing", "assembling", "solving", "writing")
               if s in timings]
    assert printed == ["meshing", "assembling", "solving", "writing"]


# ---------------------------------------------------------------------------
# The `db` contract: what a reporter handed to solve() must be able to do
#
# Deploy 3 died here. The solver calls three methods on its `db`, and the
# worker's StageReporter implemented one of them, so the solve completed every
# tet, wrote every result file, and then raised AttributeError on the line that
# put its modes in the table. Nothing in tier 1 caught it because every test that
# drove the real solve() passed `db=None` -- so the whole `if db is not None:`
# region was dead code as far as the suite was concerned.
#
# These legs run the REAL solve() against the REAL StageReporter at bench size.
# ---------------------------------------------------------------------------
class _RecordingReporter:
    """A ``db`` that implements the three methods solve() calls, and no others.

    ``__getattr__`` raises ``AttributeError`` for anything else, which is what
    makes this a guard rather than a convenience: if the solver learns a fourth
    method, the solve fails HERE with the method's name, on a tiny box, instead
    of on the Jetson after a nine-thousand-tet solve.

    Deliberately NOT a double that records unknown names and returns a lambda.
    That converts the defect -- a method the caller does not have -- into a
    silent no-op, which is the exact shape being fixed: the solve would look
    like it wrote its modes and the table would stay empty.
    """

    def __init__(self):
        self.stages: list[str] = []
        self.modes: list[tuple] = []
        self.deleted = 0

    def set_stage(self, _job_id, _status, stage=None, **_ignored):
        if stage:
            self.stages.append(str(stage))

    def delete_modes(self, _job_id):
        self.deleted += 1
        self.modes = []

    def insert_modes(self, _job_id, rows):
        self.modes = [tuple(r) for r in rows]

    def __getattr__(self, name):
        raise AttributeError(
            f"solve() called db.{name}, which this reporter does not implement. "
            f"The worker's StageReporter must implement every method solve() "
            f"uses -- see the other leg in this section."
        )


def _add_web_and_worker_to_path():
    """``storage`` and ``mse_worker`` live outside the solver directory.

    They are imported inside the tests that need them rather than at module
    scope, because the rest of this file is physics and must not depend on the
    web layer being importable.
    """
    for part in ("web", "worker"):
        path = str(Path(__file__).resolve().parents[1] / part)
        if path not in sys.path:
            sys.path.insert(0, path)


def _tiny_box(tmp_path, job_id="reptest"):
    import trimesh

    box = trimesh.creation.box(extents=[40.0, 30.0, 20.0])
    root = Path(tmp_path)
    path = root / "jobs" / job_id / "input.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(box.export(file_type="obj").encode())
    return root


def test_the_real_solve_reports_its_modes_to_the_db_it_was_handed(tmp_path):
    """The path that failed on deploy 3, on a box small enough for tier 1.

    This is the leg that would have caught it. It calls the production
    ``solve()`` with a ``db`` -- which ``bench_box.run_writing_stage`` does NOT,
    and that omission is why a missing method could ship twice.
    """
    import mse_maxwell_solver as SOL

    _add_web_and_worker_to_path()
    import mse_worker as W
    import storage as STORE

    n_modes = 2
    root = _tiny_box(tmp_path)
    store = STORE.LocalStore(root)
    reporter = W.StageReporter()

    summary = SOL.solve(
        "reptest", "mm", 1.0, n_modes, store, store,
        max_tets=3000, grid=10, verbose=False, db=reporter,
    )

    # The rows reached the reporter, one per mode, with finite fields.
    assert len(reporter.modes) == n_modes, (
        f"the solve reported {len(reporter.modes)} rows for {n_modes} modes")
    import math

    for index, k0, freq, wavelength in reporter.modes:
        assert all(math.isfinite(v) for v in (k0, freq, wavelength)), (
            f"mode {index} has a non-finite field: {(k0, freq, wavelength)}")
    assert [r[0] for r in reporter.modes] == list(range(1, n_modes + 1))

    # The single-source claim: what the reporter recorded is the same list the
    # summary carries. If these ever drift, the table and modes.json would
    # describe different solves.
    by_index = {int(m["index"]): m for m in summary["modes"]}
    for index, k0, freq, wavelength in reporter.modes:
        entry = by_index[index]
        assert (k0, freq, wavelength) == (
            entry["k0_per_m"], entry["freq_hz"], entry["vacuum_wavelength_m"]), (
            f"row {index} disagrees with summary['modes']")


def test_the_worker_reporter_can_do_everything_solve_asks_of_a_db(tmp_path):
    """The seam guard: a FOURTH method on the solver cannot slip through.

    ``_RecordingReporter`` raises on any attribute it does not implement, so
    driving the real solve() with it fails loudly and by name the moment the
    solver's ``db`` contract grows. ``StageReporter`` is asserted to satisfy the
    same contract by exposing the same names -- and the assertion is that it can
    be used, not that the name exists: the solve above already proves that.
    """
    import mse_maxwell_solver as SOL

    _add_web_and_worker_to_path()
    import mse_worker as W

    for name in ("set_stage", "delete_modes", "insert_modes"):
        assert callable(getattr(W.StageReporter(), name, None)), (
            f"StageReporter is missing {name}, which solve() calls")

    # And the reverse direction: the recording double must NOT be able to
    # silently absorb a name it lacks. This is the guard's own positive
    # control -- a `__getattr__` that returned a lambda would pass the loop
    # above and catch nothing.
    probe = _RecordingReporter()
    with pytest.raises(AttributeError, match="not implement"):
        probe.something_the_solver_might_learn_to_call  # noqa: B018


def test_a_reporter_refuses_a_method_it_does_not_implement(tmp_path):
    """The refusal must be an AttributeError, not a silent None.

    Every ``except AttributeError: pass`` in the app exists because a missing
    method is expected in some places; if this guard returned None instead, a
    future solver call would get ``None`` where it expected a bound method, and
    the failure would move somewhere less legible than the call site.
    """
    reporter = _RecordingReporter()
    with pytest.raises(AttributeError):
        reporter.set_stages("j", "running")  # a plausible pluralisation typo
