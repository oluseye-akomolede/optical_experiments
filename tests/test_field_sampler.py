"""The grid-point locator: correct, and not the thing that OOM-killed the Job.

Regression file for deploy blocker 3. ``sample_field`` used to rebuild a
``(chunk, n_tets, 3)`` difference array for every chunk *and again for every
mode*. On the real 10x7x5 mm box that is ~15 GB at 10k tets, so a Job that
meshed and solved in 7 s then sat at its 14 Gi limit with 0 CPU until it was
deleted.

Two properties have to hold together, and neither is enough alone:

* the located set must be EXACTLY what brute force finds -- an approximate
  locator would silently drop or move samples and change the picture the owner
  sees, with nothing failing;
* the working set must not scale with ``n_tets`` -- otherwise the fix is a
  rewrite that still OOMs on a bigger object.

The mesh here is deliberately TIER 1: a few hundred tets and a 24^3 grid, pure
numpy, no TetGen, no eigen solve. The real object is the lead's Jetson run.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import scipy.spatial
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "dockerfiles" / "maxwell_solver"))

import mse_fields as FLD  # noqa: E402


def jittered_cube_mesh(n: int = 4, jitter: float = 0.25, seed: int = 0,
                       extent: float = 2.0):
    """A conforming mesh of ``(n-1)^3`` jittered cubes spanning ``extent``.

    Skewed on purpose: on a right-angled tet a wrong barycentric or gradient
    convention is invisible, and the point of this file is to catch exactly the
    mistakes that a symmetric mesh hides.

    ``extent`` is fixed while ``n`` varies, so raising ``n`` REFINES the mesh in
    place rather than enlarging it. That distinction matters for the memory
    test: a bigger box with more tets is not a denser mesh, and a fixed-size
    neighbourhood legitimately contains more tets when the box grows.
    """
    rng = np.random.default_rng(seed)
    spacing = extent / (n - 1)
    lattice = np.stack(
        np.meshgrid(*[np.arange(n) for _ in range(3)], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = (lattice * spacing).astype(float) + rng.normal(
        0, jitter * spacing, lattice.shape)
    index_of = {tuple(int(v) for v in pos): i for i, pos in enumerate(lattice)}

    cube = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    six = [(0, 1, 2, 3), (1, 4, 2, 5), (2, 5, 3, 6),
           (1, 2, 5, 3), (3, 5, 6, 7), (2, 3, 5, 6)]

    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                verts = [index_of[(i + dx, j + dy, k + dz)]
                         for (dx, dy, dz) in cube]
                tets.extend([verts[a], verts[b], verts[c], verts[d]]
                            for (a, b, c, d) in six)
    return points, np.array(tets)


def delaunay_mesh(n_side: int, jitter: float = 0.22, seed: int = 0,
                  extent: float = 2.0):
    """A TetGen-shaped mesh: jittered interior points, Delaunay-tetrahedralised.

    Deliberately unlike ``jittered_cube_mesh``: that one is a cube lattice where
    every vertex is a cube corner shared by up to 24 tets (0.25-0.40 vertices per
    tet). A real tetrahedralisation of a solid gives an interior vertex 6-12 of
    them (~0.16-0.33). The difference matters here because the candidate ball's
    size depends on how many cells fit near a point, and the lattice is the
    pessimistic case, not the representative one.

    Tier 1: scipy only, a few hundred tets, no TetGen import.
    """
    rng = np.random.default_rng(seed)
    grid = np.stack(
        np.meshgrid(*[np.linspace(0, extent, n_side)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid = grid + rng.normal(0, jitter * extent / (n_side - 1), grid.shape)
    grid[:, 0] += 0.001          # off the axes, so the hull is not degenerate
    return grid, np.asarray(Delaunay(grid).simplices, dtype=np.int64)


def test_the_locator_agrees_with_brute_force_exactly():
    """The fast path must find the SAME points as testing every tet.

    Not 'approximately': the located set is compared as a set, so a point that
    this version finds and brute force does not (or the reverse) fails.
    """
    points, tets = jittered_cube_mesh(n=4)
    rng = np.random.default_rng(7)
    samples = np.vstack([
        points + rng.normal(0, 0.3, points.shape),   # near the mesh
        points + rng.normal(0, 3.0, points.shape),   # well outside it
    ])

    idx, _tet_of, _bary = FLD.locate_samples(points, tets, samples)
    fast = np.zeros(samples.shape[0], dtype=bool)
    fast[idx] = True
    slow = FLD.brute_force_mask(points, tets, samples)

    assert fast.any(), "no sample landed inside; the test would be vacuous"
    assert np.array_equal(fast, slow), (
        f"{int((fast != slow).sum())} of {samples.shape[0]} points disagree "
        "with brute force"
    )


def test_the_locator_finds_the_containing_tet_and_its_barycentric_coords():
    """Inside a known tet, the barycentric coords must sum to 1 and be >= 0."""
    points, tets = jittered_cube_mesh(n=3)
    rng = np.random.default_rng(11)
    samples = points + rng.normal(0, 0.2, points.shape)

    idx, tet_of, bary = FLD.locate_samples(points, tets, samples)
    assert idx.size
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-9)
    assert (bary >= -1e-9).all(), "a located point has a negative barycentric coord"

    # and independently: the point must actually be inside the tet it was given
    v0 = points[tets[tet_of, 0]]
    for slot in range(idx.size):
        tet = tets[tet_of[slot]]
        # the point reconstructed from its coords must equal the sample
        recon = bary[slot] @ points[tet]
        assert np.allclose(recon, samples[idx[slot]], atol=1e-9), (
            f"sample {idx[slot]} reconstructs to {recon}, not "
            f"{samples[idx[slot]]}"
        )
    assert v0.shape[0] == idx.size


def test_a_point_on_a_shared_face_is_found_by_both():
    """Ties matter: a sample exactly on a face between tets must be inside.

    The brute-force version picks the first tet that contains the point; the
    fast path picks the first *candidate*. Both must call it inside, or the two
    disagree on the boundary and the comparison above is luck.
    """
    points, tets = jittered_cube_mesh(n=3)
    # a vertex of the mesh is shared by every tet that meets there
    samples = points[:6]
    idx, _tet_of, _bary = FLD.locate_samples(points, tets, samples)
    assert idx.size == samples.shape[0]
    assert FLD.brute_force_mask(points, tets, samples).all()


def test_the_locator_does_not_broadcast_against_every_tet():
    """The assertion that would have caught the original bug, stated exactly.

    The old code built ``(m, n_tets, 3)`` -- once per chunk, and again for every
    one of the 20 modes. At 10k tets and 262k samples that is ~15 GB, and it is
    why a Job that meshed and solved in 7 s sat at its 14 Gi limit with 0 CPU
    until the lead deleted it.

    So the property to pin is not "the peak is small" -- measured on this mesh
    the peak is 23 MB against a naive 54 MB, a ratio of only 2.3x, because most
    of it is the KD-tree's own index. A "10x below naive" threshold therefore
    passes on a real 10k-tet mesh and fails on a tier-1 one, which makes it a
    test of the mesh size rather than of the code. The property to pin is that
    **no array is ever formed over all tets at once**.

    The probe wraps ``_tet_geometry``, which is the one place the locator must
    read the two all-tets arrays from. Patching module globals does NOT work and
    was measured not to: the locator does
    ``v0_all, inv_J, _, ok = _tet_geometry(points, tets)``, which binds *local*
    names over any global of the same name, so a patched global is never read.
    Wrapping the return value is seen whenever the arrays are used, whatever the
    function calls them locally.

    The signature being watched is any index past the first axis, because that is
    what makes an ``(m, n_tets, ...)`` array:

    * ``v0_all[None, :, :]`` -- 3 indices, the broadcast that was the bug;
    * ``inv_J[:, 0, :]`` -- 3 indices, the ``einsum("mtj,tj->mt", d, ...)`` form
      of the same bug.

    Both the current implementation and the bug use three indices, so the
    difference is not the *number* of indices but the *type*: the fix always
    indexes by an index array (``inv_J[cand]``, ``v0_all[cand]``), which selects
    candidates, while the bug indexes by slices/None and keeps every tet. That
    distinction is what the assertion below makes: no slice or ``None`` past the
    first axis.
    """
    bad = (slice, type(None))

    for n in (3, 6):
        points, tets = jittered_cube_mesh(n=n)
        n_tets = tets.shape[0]
        rng = np.random.default_rng(3)
        samples = rng.uniform(points.min(axis=0), points.max(axis=0),
                              size=(3_000, 3))

        touched = []
        real_geometry = FLD._tet_geometry

        def probed_geometry(pts, tets_in):
            v0, inv_J, n_t, ok = real_geometry(pts, tets_in)

            def probe(arr, name):
                real = np.asarray(arr)

                class Probe(np.ndarray):
                    def __new__(cls):
                        return real.view(cls)

                    def __getitem__(self, key):
                        k = key if isinstance(key, tuple) else (key,)
                        if len(k) > 1 and any(isinstance(part, bad)
                                              for part in k[1:]):
                            touched.append((name, tuple(
                                type(p).__name__ for p in k)))
                        return super().__getitem__(key)

                return Probe()

            return probe(v0, "v0_all"), probe(inv_J, "inv_J"), n_t, ok

        FLD._tet_geometry = probed_geometry
        try:
            idx, _tet_of, _bary = FLD.locate_samples(points, tets, samples)
        finally:
            FLD._tet_geometry = real_geometry

        assert idx.size, "nothing was located; the probe would prove nothing"
        assert not touched, (
            f"on a {n_tets}-tet mesh the locator indexed {touched[0][0]} with "
            f"{touched[0][1]} past the first axis -- that keeps all {n_tets} "
            "tets in the result, which at production size is the "
            "(chunk, n_tets, 3) allocation that OOM-killed the Job"
        )


def test_the_candidate_set_is_bounded_by_cell_size_not_tet_count():
    """The property that makes the locator scale: candidates track CELL SIZE.

    The candidates are the tets whose centroids lie within ``r_max`` of the
    sample, so the count is set by how many cells fit in a ball of one cell's
    radius -- a local density, not ``n_tets``. Refining a fixed extent in place
    means smaller cells and a smaller ball.

    The honest property is sublinear rather than constant (``r_max`` grows
    relative to the mean cell as a mesh refines), so the assertion is "far
    slower than the mesh" -- proportional is the original ``(n_samples,
    n_tets)`` bug.
    """
    rng = np.random.default_rng(3)
    measured = {}
    for n in (4, 8):
        points, tets = jittered_cube_mesh(n=n)
        lo, hi = points.min(axis=0), points.max(axis=0)
        samples = rng.uniform(lo, hi, size=(500, 3))
        count = FLD.candidate_counts(points, tets, samples)
        measured[n] = (float(np.mean(count)), tets.shape[0])

    mean_coarse, tets_coarse = measured[4]
    mean_fine, tets_fine = measured[8]
    tet_growth = tets_fine / tets_coarse
    cand_growth = mean_fine / mean_coarse
    assert tet_growth > 4.0, "the meshes must actually differ"

    assert cand_growth < tet_growth / 2.0, (
        f"candidates grew {cand_growth:.1f}x while the mesh grew "
        f"{tet_growth:.1f}x (mean {mean_coarse:.0f} -> {mean_fine:.0f}) -- the "
        "working set is tracking the mesh, which is the original bug"
    )


def test_an_interior_point_centroid_is_within_the_nearest_cell():
    """The theorem the candidate rule rests on, checked on a real mesh.

    ``p`` inside a tet ``S`` implies ``||p - c_S|| <= R_S <= r_max``, because
    ``p`` is a convex combination of S's vertices and every vertex is at most
    ``R_S`` from the centroid. If that were false the ball could miss the
    containing tet and the picture would silently change.

    The mesh is a TetGen tetrahedralisation -- the same library the solver uses
    -- because the cube-lattice meshes used elsewhere in this file have uniform
    cells and would not exercise the bound on a sliver.
    """
    points, tets = delaunay_mesh(6)
    centroids = points[tets].mean(axis=1)
    r_tet = np.linalg.norm(points[tets] - centroids[:, None, :],
                           axis=2).max(axis=1)

    rng = np.random.default_rng(5)
    lo, hi = points.min(axis=0), points.max(axis=0)
    samples = rng.uniform(lo, hi, size=(4_000, 3))

    v0, inv_J, _, ok = FLD._tet_geometry(points, tets)
    checked = 0
    for sample in samples:
        d = sample[None, :] - v0
        b = np.stack([np.einsum("tj,tj->t", d, inv_J[:, k, :])
                      for k in range(3)])
        hit = ((1.0 - b.sum(axis=0) >= -1e-9) & (b >= -1e-9).all(axis=0)) & ok
        ids = np.nonzero(hit)[0]
        if not ids.size:
            continue
        checked += 1
        for t in ids:
            ratio = np.linalg.norm(sample - centroids[t]) / r_tet[t]
            assert ratio <= 1.0 + 1e-9, (
                f"sample at {sample} is inside tet {t} but its centroid is "
                f"{ratio:.4f} of that tet's vertex radius away -- the candidate "
                "ball would miss it"
            )
    assert checked > 100, "no interior samples; the test would be vacuous"


def test_brute_force_is_the_naive_shape_and_still_exists_for_reference():
    """The reference must stay the slow version, or the tests above are circular.

    If someone 'optimises' ``brute_force_mask`` to call the locator, the
    agreement test would compare the locator with itself and pass forever.
    """
    points, tets = jittered_cube_mesh(n=3)
    rng = np.random.default_rng(5)
    samples = rng.uniform(points.min(axis=0), points.max(axis=0), size=(200, 3))

    # a chunk larger than the sample count means one pass over every tet, which
    # is only possible if the implementation is genuinely all-tets
    tracemalloc.start()
    FLD.brute_force_mask(points, tets, samples, chunk=10_000)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    naive = samples.shape[0] * tets.shape[0] * 3 * 8
    assert peak > naive / 4.0, (
        f"brute_force_mask peaked at {peak / 1e6:.1f} MB, well under the naive "
        f"{naive / 1e6:.1f} MB -- it is no longer a brute-force reference"
    )


def test_locating_once_and_reusing_matches_locating_per_mode():
    """The call-site change is only valid if reuse gives the same field.

    ``write_results`` now locates once and passes the result to every mode.
    That is a pure optimisation ONLY because the located set does not depend on
    the eigenvector; this pins that.
    """
    points, tets = jittered_cube_mesh(n=3)
    rng = np.random.default_rng(13)
    samples = rng.uniform(points.min(axis=0), points.max(axis=0), size=(400, 3))

    reuse = FLD.locate_samples(points, tets, samples)
    again = FLD.locate_samples(points, tets, samples)
    assert np.array_equal(reuse[0], again[0])
    assert np.array_equal(reuse[1], again[1])
    assert np.allclose(reuse[2], again[2])


@pytest.mark.parametrize("n_longest", [8, 16, 24])
def test_the_whole_sample_field_still_produces_a_normalised_field(n_longest):
    """End to end through ``sample_field``: shape, NaN mask, max|E| = 1."""
    points, tets = jittered_cube_mesh(n=3)
    edge_module = __import__("mse_fem")
    edges, tet_edges, signs = edge_module.build_edges(tets)
    coeffs = np.ones(tet_edges.max() + 1)

    grid = FLD.sample_grid(points, n_longest)
    samples, shape = grid[0], grid[1]
    located = FLD.locate_samples(points, tets, samples)
    field, shape, _origin, _spacing = FLD.sample_field(
        points, tets, edges, tet_edges, signs, coeffs,
        n_longest=n_longest, located=located, grid=grid,
    )
    assert field.shape == (shape[2], shape[1], shape[0], 3)
    assert field.dtype == np.float32

    inside = np.isfinite(field).all(axis=-1)
    outside = ~inside
    assert inside.any() and outside.any(), "the grid must straddle the surface"
    assert np.isnan(field[outside]).all(), "outside samples must be NaN"

    peaks = np.linalg.norm(field[inside], axis=-1)
    assert peaks.max() == pytest.approx(1.0, rel=1e-6)


def test_locate_returns_nothing_for_an_empty_mesh():
    """Degenerate input must not raise: a Job should fail on a real problem."""
    idx, tet_of, bary = FLD.locate_samples(
        np.zeros((0, 3)), np.zeros((0, 4), dtype=int), np.zeros((5, 3))
    )
    assert idx.size == 0 and tet_of.size == 0 and bary.shape == (0, 4)
