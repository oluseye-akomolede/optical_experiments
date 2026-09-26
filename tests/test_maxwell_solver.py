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

    points_m, tets = M.tetrahedralize(mesh, "mm", n_modes=20, verbose=False)
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
    points_m, tets = M.tetrahedralize(mesh, "mm", n_modes=10, max_tets=60_000,
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
