"""Per-tet permittivity: the mass matrix, and the eps-weighted projector.

WHY THESE EXIST. ``eps_r = n^2`` enters the weak form through the MASS matrix
only -- the weak form is

    integral of curl(E).curl(v)  =  k^2 * integral of eps_r * E.v

-- so a per-tet eps is a scaling of ``B`` and nothing else. That much is easy.
The hard part is the divergence-free projector, which must constrain
``div(eps E) = 0`` and therefore has to be built from the eps-weighted mass
matrix, ``G^T M_eps G``, not from ``B``.

WHERE THE WEIGHT COMES FROM, and why there is only one of it.
``mse_maxwell_solver.solve`` passes ``eps_by_tet`` to
:func:`mse_fem.assemble_global`, so the ``B`` that reaches ``solve_modes`` is
ALREADY the eps-weighted mass. That single matrix is both the eigenproblem's M
and the projector's weight, and there is deliberately no separate ``M_eps``
parameter: two parameters that must agree are two that can drift.

That moves the real risk to the CALLER -- a solver that forgets ``eps_by_tet``
assembles the vacuum mass and silently projects ``div(E) = 0`` instead of
``div(eps E) = 0``. So the gate drives ``mse_maxwell_solver.solve`` end to end
and checks the mass that came back equals eps-per-tet times the vacuum mass,
independently rebuilt.

A FINDING THAT CHANGES HOW THIS IS TESTED. On the meshes available at tier 1,
using the WRONG weight in the projector does NOT visibly change the reported
eigenvalues: measured on a 6x4x3 box with a 12:1 eps step, the vacuum weight and
the eps weight give eigenvalues differing by about **3e-3 RELATIVE**, with
converged eigenvectors that are close but NOT identical. (An earlier version of
this docstring -- and of the comment in ``mse_eigen`` -- claimed "byte-identical
eigenvalues and eigenvectors". That is false; the reviewer reproduced the box.
Byte identity holds only for UNIFORM eps, where the two matrices are
proportional and the constant cancels in the projector.) The ~3e-3 is far too
little for a "no spurious modes" test to redden.

That is the rig's signature defect -- a check that cannot fail -- so the
assertions here are on the OPERATOR's defining property instead:
``G^T B P = 0``, which the vacuum weight violates by 3.3e-2 against 1.3e-16 for
the correct matrix. That gap is what makes the test able to redden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_eigen as EIG  # noqa: E402
import mse_fem as FEM  # noqa: E402
import mse_meshing as MESH  # noqa: E402

trimesh = pytest.importorskip("trimesh")
sparse = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")


# ---------------------------------------------------------------------------
# Fixtures: small enough for tier 1, big enough to have an interior.
# ---------------------------------------------------------------------------
def _box_mesh(extents=(6.0, 4.0, 3.0), max_tets=40_000):
    box = trimesh.creation.box(extents=extents)
    points_m, tets, _ = MESH.tetrahedralize(
        box, "mm", n_modes=6, max_tets=max_tets, verbose=False
    )
    edges, tet_edges, _ = FEM.build_edges(tets)
    return box, points_m, tets, edges, tet_edges


def _split_eps(points_m, tets, high=4.0, axis=0):
    """A two-material assignment: everything past the mid-plane gets ``high``."""
    centres = points_m[tets].mean(axis=1)
    cut = 0.5 * (centres[:, axis].min() + centres[:, axis].max())
    return np.where(centres[:, axis] > cut, float(high), 1.0)


# ---------------------------------------------------------------------------
# The mass matrix is the whole physics change, and None must be free.
# ---------------------------------------------------------------------------
def test_eps_none_is_bit_identical_to_the_path_before_materials():
    """The one-part upload's regression rests on this.

    ``None`` must not multiply by a ones array: it must skip the scaling, so the
    arithmetic is the same operations in the same order and the bytes match.
    """
    _box, points_m, tets, _edges, tet_edges = _box_mesh()
    A0, B0, _ = FEM.element_matrices(points_m, tets, tet_edges)
    A1, B1, _ = FEM.element_matrices(points_m, tets, tet_edges, None)
    assert np.array_equal(A0, A1), "A must be untouched by the eps parameter"
    assert np.array_equal(B0, B1), "None must reproduce the old B exactly"


def test_a_uniform_eps_is_a_exact_scaling_of_the_mass_matrix():
    """A uniform eps scales B and leaves A exactly alone.

    Both halves are asserted. ``A`` being untouched is the statement that the
    curl-curl has no material in it; ``B`` scaling by exactly 4.0 (not close to
    it) is the statement that the per-tet factor reaches every one of the four
    Mn blocks -- scaling three of the four is the failure this catches.
    """
    _box, points_m, tets, _edges, tet_edges = _box_mesh()
    A0, B0, _ = FEM.element_matrices(points_m, tets, tet_edges)
    eps = np.full(len(tets), 4.0)
    A1, B1, _ = FEM.element_matrices(points_m, tets, tet_edges, eps)
    assert np.array_equal(A0, A1), "the curl-curl must not see the material"
    assert np.array_equal(B1, 4.0 * B0), "B must scale by exactly eps_r"


def test_a_per_tet_eps_reaches_every_tet_not_an_average():
    """A half-and-half assignment must not read as a uniform 2.5.

    The failure mode this guards: applying eps through a scalar (a mean, or one
    region's value) gives a plausible matrix for the wrong problem.
    """
    _box, points_m, tets, _edges, tet_edges = _box_mesh()
    eps = _split_eps(points_m, tets, high=4.0)
    assert set(np.unique(eps)) == {1.0, 4.0}
    _, B_uneven, _ = FEM.element_matrices(points_m, tets, tet_edges, eps)
    _, B_flat4, _ = FEM.element_matrices(
        points_m, tets, tet_edges, np.full(len(tets), 4.0)
    )
    assert not np.array_equal(B_uneven, B_flat4), (
        "a two-material B must differ from any single-material B"
    )


def test_a_wrong_length_eps_is_refused():
    """A mismatched eps array must fail loudly, not broadcast or truncate."""
    _box, points_m, tets, _edges, tet_edges = _box_mesh()
    with pytest.raises(ValueError):
        FEM.element_matrices(points_m, tets, tet_edges, np.ones(len(tets) - 1))


# ---------------------------------------------------------------------------
# The projector: the defining property, not the spectrum.
# ---------------------------------------------------------------------------
def _projector(points_m, tets, edges, tet_edges, free, constrained, mass):
    G, _ = EIG.build_gradient(points_m, tets, edges, tet_edges, free, constrained)
    GtM = (G.T @ mass).tocsr()
    S = (GtM @ G).tocsc()
    lu = spla.splu(S)
    return GtM, (lambda x: x - G @ lu.solve(GtM @ x))


def test_the_projector_annihilates_the_eps_weighted_divergence():
    """THE assertion that can fail: ``G^T M_eps P x`` must be zero.

    Measured on a 12:1 step: the eps-weighted projector leaves 1.3e-16, while a
    projector built from the vacuum mass leaves 3.3e-2. Fourteen orders of
    magnitude apart, so a tolerance anywhere between them discriminates -- and
    this is the property the spurious modes would come from, asserted directly
    rather than inferred from a spectrum that turns out not to show them.
    """
    _box, points_m, tets, edges, tet_edges = _box_mesh()
    constrained = FEM.boundary_edges(tets, edges)
    eps = _split_eps(points_m, tets, high=12.0)
    n_edges = len(edges)

    A_eps, B_eps = FEM.assemble_global(points_m, tets, tet_edges, n_edges, eps_by_tet=eps)
    _A_free, M_eps, free = FEM.pec_eliminate(A_eps, B_eps, constrained, n_edges)

    GtM, project = _projector(points_m, tets, edges, tet_edges, free, constrained, M_eps)

    x = np.random.default_rng(0).standard_normal(M_eps.shape[0])
    residual = float(np.linalg.norm(GtM @ project(x)) / max(np.linalg.norm(x), 1e-30))
    assert residual < 1e-10, (
        f"the eps-weighted projector leaves a divergence of {residual:.3e}; "
        "it is not projecting onto div(eps E) = 0"
    )


def test_the_vacuum_projector_would_fail_that_same_assertion():
    """The control: the wrong matrix must actually fail the test above.

    Without this, an assertion with a tolerance nobody has tested is just a
    number. This builds the projector from the VACUUM mass on the same
    two-material problem and shows it leaves a divergence orders of magnitude
    above the tolerance -- so the assertion above is known to be able to redden.
    """
    _box, points_m, tets, edges, tet_edges = _box_mesh()
    constrained = FEM.boundary_edges(tets, edges)
    eps = _split_eps(points_m, tets, high=12.0)
    n_edges = len(edges)

    A_eps, B_eps = FEM.assemble_global(points_m, tets, tet_edges, n_edges, eps_by_tet=eps)
    A_vac, B_vac = FEM.assemble_global(points_m, tets, tet_edges, n_edges)
    _Af, M_eps, free = FEM.pec_eliminate(A_eps, B_eps, constrained, n_edges)
    _Av, B_vac_free, _ = FEM.pec_eliminate(A_vac, B_vac, constrained, n_edges)

    # The discriminating measurement is a CROSS product, and getting this wrong
    # is how the control itself becomes a check that cannot fail: the vacuum
    # projector annihilates the vacuum divergence perfectly, so measuring
    # ``G^T M_vac @ P_vac x`` returns zero however wrong the projector is.
    # What must be measured is the EPS divergence of what the vacuum projector
    # leaves behind -- ``G^T M_eps @ P_vac x``.
    GtM_eps, _project_eps = _projector(
        points_m, tets, edges, tet_edges, free, constrained, M_eps
    )
    _g, project_bad = _projector(
        points_m, tets, edges, tet_edges, free, constrained, B_vac_free
    )
    x = np.random.default_rng(0).standard_normal(M_eps.shape[0])
    residual_bad = float(
        np.linalg.norm(GtM_eps @ project_bad(x)) / max(np.linalg.norm(x), 1e-30)
    )
    assert residual_bad > 1e-6, (
        "the vacuum-matrix projector was expected to leave a large divergence; "
        f"it left {residual_bad:.3e}, so the discriminating test is not "
        "discriminating"
    )


def test_uniform_eps_makes_the_two_projectors_identical():
    """With uniform eps the constraint reduces to div(E) = 0, so P is unchanged.

    This is why the one-material regression is meaningful: a constant factor
    cancels inside the projector, so any difference there is a misapplication
    rather than a rearrangement.
    """
    _box, points_m, tets, edges, tet_edges = _box_mesh()
    constrained = FEM.boundary_edges(tets, edges)
    n_edges = len(edges)
    A_eps, B_eps = FEM.assemble_global(
        points_m, tets, tet_edges, n_edges, eps_by_tet=np.full(len(tets), 4.0)
    )
    A_vac, B_vac = FEM.assemble_global(points_m, tets, tet_edges, n_edges)
    _Ae, M_eps, free = FEM.pec_eliminate(A_eps, B_eps, constrained, n_edges)
    _Av, B_vac_free, _ = FEM.pec_eliminate(A_vac, B_vac, constrained, n_edges)

    _g1, p_eps = _projector(points_m, tets, edges, tet_edges, free, constrained, M_eps)
    _g2, p_vac = _projector(points_m, tets, edges, tet_edges, free, constrained, B_vac_free)
    x = np.random.default_rng(0).standard_normal(M_eps.shape[0])
    assert np.allclose(p_eps(x), p_vac(x), atol=1e-10), (
        "with uniform eps the mass matrices are proportional and the projectors "
        "must agree"
    )


# ---------------------------------------------------------------------------
# The end-to-end physics: uniform eps via materials == n_index^2.
# ---------------------------------------------------------------------------
# The end-to-end physics: uniform eps via materials == n_index^2.
# ---------------------------------------------------------------------------
def test_uniform_eps_r_equals_the_n_index_the_old_path_used():
    """A cavity filled with eps_r = 4 must give 1/4 the vacuum k^2.

    The row's acceptance says a uniform eps_r = 4 via materials must equal the
    old path at n_index = 2. Those are the same statement: n_index enters as
    eps_r = n^2, so n = 2 IS eps_r = 4, and filling a cavity with a medium of
    index n divides every k^2 by n^2.

    Measured on a 4538-tet sphere: every eigenvalue at eps = 4 is exactly a
    quarter of the vacuum value, to twelve digits.

    ``solve_modes`` now takes the weight from ``B`` alone -- the caller's
    ``eps_by_tet`` is what made ``B`` eps-weighted -- so this builds ``B`` the
    same way the solver does, by passing ``eps_by_tet`` to ``assemble_global``.
    """
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    points_m, tets, _ = MESH.tetrahedralize(
        sphere, "mm", n_modes=6, max_tets=40_000, verbose=False
    )
    edges, tet_edges, _ = FEM.build_edges(tets)
    constrained = FEM.boundary_edges(tets, edges)
    n_edges = len(edges)

    def modes(eps):
        # ONE mass matrix, assembled with eps, exactly as the solver does.
        A, B = FEM.assemble_global(points_m, tets, tet_edges, n_edges, eps_by_tet=eps)
        A_free, B_free, free = FEM.pec_eliminate(A, B, constrained, n_edges)
        vals, _vecs, _sigma = EIG.solve_modes(
            A_free, B_free, 5, points_m, edges, tets, tet_edges, free, constrained,
            verbose=False,
        )
        return np.asarray(vals)

    vacuum = modes(None)
    filled = modes(np.full(len(tets), 4.0))

    assert vacuum.shape == filled.shape
    ratio = filled / vacuum
    # control: the two spectra must not be trivially equal, or the ratio below
    # is asserting 1.0 == 0.25 and would fail for the wrong reason
    assert not np.allclose(ratio, 1.0)
    assert np.allclose(ratio, 0.25, rtol=1e-9), (
        f"eps_r = 4 must quarter every k^2; got ratios {np.round(ratio, 6)}"
    )


# ---------------------------------------------------------------------------
# The end-to-end gate: solve() must actually apply the materials.
# ---------------------------------------------------------------------------
def test_solve_applies_eps_per_tet_to_the_mass_and_the_projector(tmp_path):
    """The gate the lead asked for, and the only shape that can fail.

    REMOVING THE SEPARATE ``M_eps`` MOVED THE RISK HERE. There is one mass
    matrix now: ``solve_modes`` projects with ``B``, and ``B`` is eps-weighted
    only because :func:`mse_maxwell_solver.solve` passes ``eps_by_tet`` to
    ``assemble_global``. A solver that forgets the argument assembles the VACUUM
    mass and silently projects ``div(E) = 0`` instead of ``div(eps E) = 0`` --
    and the eigenvalues barely move (~3e-3 relative), so nothing else notices.

    So this test drives the REAL ``solve()`` on a two-material upload and checks
    two independent things against matrices it rebuilds ITSELF:

      (a) the assembled mass equals eps-per-tet times the vacuum mass; and
      (b) the projector weight the solver recorded is ``G^T`` of that same mass.

    The discriminator is the VACUUM control, not a self-comparison: the test
    asserts first that ``G^T`` of the eps-weighted mass DIFFERS from ``G^T`` of
    the vacuum one on this fixture. Without that precondition the whole thing
    collapses back into ``A == A`` -- which is how the earlier version of this
    test stayed green with the bug present, twice.

    The mutation is "solve() forgets eps_by_tet": it must redden this test by
    name, on assertion (a) or (b).
    """
    import json

    import mse_maxwell_solver as SOL
    import mse_parts as PARTS
    import mse_storage as STORE

    job = "0d338445-0000-0000-0000-000000000001"
    units = "mm"
    n_wall, n_core = 1.0, 2.0

    # A wall box with a smaller dielectric core inside it, as two named groups.
    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    obj_bytes = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj"
    ).encode("utf-8")

    store_in = STORE.LocalStore(tmp_path)
    store_out = STORE.LocalStore(tmp_path)
    store_in.write(STORE.job_key(job, "input.obj"), obj_bytes)
    materials = {"wall": n_wall, "core": n_core}
    store_in.write(STORE.job_key(job, "input.materials.json"),
                   json.dumps(materials).encode("utf-8"))

    summary = SOL.solve(
        job, units, n_wall, 3, store_in, store_out,
        max_tets=60_000, grid=12, verbose=False,
        materials=materials,
    )
    assert summary["n_tets"] > 0
    assert summary["n_edges"] > 0, (
        "the summary must report the free edge count; without it this gate "
        "cannot prove it rebuilt the solver's own system"
    )

    # The mesh the solver built, rebuilt here from the same upload so the test
    # owns its own geometry rather than reaching into the solver's internals.
    parts = PARTS.load_parts(obj_bytes, units=units)
    assert len(parts) == 2, "this gate needs a two-material upload"
    points_m, tets, region_of_tet = MESH.tetrahedralize(
        PARTS.merged_surface(parts), units, n_modes=3, n_index=n_wall,
        max_tets=60_000, verbose=False,
        materials=materials, parts=parts,
    )
    # The solver meshed the same upload with the same arguments, and the mesher
    # is deterministic (the canonical frame exists precisely so it is), so the
    # rebuilt mesh must have the solver's tet count. Asserting it is what lets
    # the checks below speak about the mesh the SOLVER built -- without it they
    # would be a second opinion about a different object.
    assert region_of_tet.shape[0] == summary["n_tets"], (
        f"the rebuilt mesh has {region_of_tet.shape[0]} tets but the solver "
        f"reported {summary['n_tets']}; this gate cannot speak for the solve"
    )

    # eps per tet, from the regions the mesh reported. n^2, defined once.
    eps_by_tet = np.where(region_of_tet == 0, float(n_wall) ** 2,
                          float(n_core) ** 2)
    assert len(np.unique(eps_by_tet)) == 2, "fixture must be two-material"
    edges, tet_edges, _ = FEM.build_edges(tets)
    n_edges = len(edges)
    constrained = FEM.boundary_edges(tets, edges)

    # (a) the assembled mass equals eps-per-tet times the vacuum mass,
    #     INDEPENDENTLY rebuilt here from the vacuum call.
    A_vac, B_vac = FEM.assemble_global(points_m, tets, tet_edges, n_edges)
    A_eps, B_eps = FEM.assemble_global(points_m, tets, tet_edges, n_edges,
                                       eps_by_tet=eps_by_tet)

    # (a) THE MASS CARRIES eps PER TET. Checked at the ELEMENT level, where eps
    # is a per-tet multiplier by construction: assembled global rows cannot be
    # read that way because an edge is shared between tets, so a per-row ratio
    # would be an average over neighbours and could be right for the wrong
    # reason. Every nonzero entry of B_elem must scale by exactly its tet's eps.
    A_e_v, B_e_v, _ = FEM.element_matrices(points_m, tets, tet_edges)
    A_e_e, B_e_e, _ = FEM.element_matrices(points_m, tets, tet_edges, eps_by_tet)
    ratio = B_e_e / np.where(B_e_v == 0, np.nan, B_e_v)
    # every nonzero entry of B_elem scales by that tet's eps -- one value per tet
    per_tet = np.nanmax(np.abs(ratio - eps_by_tet[:, None, None]), axis=(1, 2))
    assert np.allclose(per_tet, 0.0, atol=1e-12), (
        "element_matrices did not scale the mass by eps per tet"
    )

    # (b) the recorded projector weight is G^T of that same mass.
    free = None
    A_free, B_free_sol, free = FEM.pec_eliminate(A_eps, B_eps, constrained, n_edges)
    G, _ = EIG.build_gradient(points_m, tets, edges, tet_edges, free, constrained)
    want = (G.T @ B_free_sol).tocsr()

    _Av, B_free_vac, _ = FEM.pec_eliminate(A_vac, B_vac, constrained, n_edges)
    vac = (G.T @ B_free_vac).tocsr()

    def same(x, y):
        dd = (x - y).tocsr()
        dd.eliminate_zeros()
        return dd.nnz == 0

    # PRECONDITION: the two weights must differ, or (b) cannot discriminate --
    # this is the self-comparison the lead warned about, made impossible to
    # fall back into silently.
    assert not same(want, vac), (
        "G^T(eps-weighted mass) and G^T(vacuum mass) are identical on this "
        "fixture, so this test cannot detect a solver that forgot eps_by_tet"
    )

    weight = EIG.LAST_PROJECTOR_WEIGHT
    assert weight is not None, "solve() must record the projector it used"
    assert same(weight, want), (
        "the recorded projector weight is not G^T of the eps-weighted mass -- "
        "solve() projected with the VACUUM mass, so it enforced div(E) = 0 "
        "instead of div(eps E) = 0"
    )
    assert not same(weight, vac), (
        "the recorded projector weight IS G^T of the vacuum mass, so "
        "eps_by_tet never reached the assembly (this is the mutation)"
    )
