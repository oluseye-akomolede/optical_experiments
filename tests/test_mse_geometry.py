"""Dependency-free geometry: containment, overlap, touching, and the bounding
box pre-filter that keeps the cost off the web request.

WHY THESE EXIST. Everything here runs in the SOLVER IMAGE, where ``rtree``,
``manifold3d`` and ``embree`` are all ABSENT and ``trimesh.contains`` /
``intersection`` / ``ray.contains_points`` all RAISE. A test written against
those passes in this venv -- which has them -- and fails in production, which is
the crashloop shape this repo has a memory for. So these tests exercise the
numpy implementation, and the solver leg of ``tests/image_smoke.sh`` proves the
same calls work inside a real solver container -- that leg runs point_in_mesh
(inside/on/outside), the torus and wall/core seeds, the part split and the
touching test, and exits non-zero on a wrong answer.

The controls matter more than the assertions here, and two of them exist because
a reference was wrong in an earlier session: a point-in-solid test compared
against ``|p| < r`` on a POLYHEDRAL sphere is testing the wrong solid, exactly
as a sphere benchmark once compared against the cylindrical Bessel zeros. Both
tests below therefore state a control that can fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_geometry as GEO  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def _icosphere(subdivisions=3, radius=1.0):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def _inradius(mesh):
    """Distance from the centre to a face plane: the inscribed radius.

    For a convex polyhedron whose centre is inside, a point closer than this is
    PROVABLY inside, and a point further than the circumradius is PROVABLY
    outside, whatever the surface's triangulation. The shell between the two is
    genuinely ambiguous for a polyhedron, so the controls below only use points
    outside that shell.
    """
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    centres = tri.mean(axis=1)
    norms = np.linalg.norm(centres, axis=1)
    unit = centres / norms[:, None]
    return float(np.min(np.einsum("ij,ij->i", centres, unit)))


# ---------------------------------------------------------------------------
# The winding test: exact on a closed surface, and it must never be trusted on
# one. The positive and negative controls are the point of this section.
# ---------------------------------------------------------------------------
def test_winding_number_is_exactly_one_inside_and_zero_outside():
    """On a closed surface the winding number is an integer, and it is computed.

    Not "approximately": the solid-angle sum gives exactly 1.0 and exactly 0.0
    here, which is what lets a threshold at 0.5 be a classification rather than
    a tolerance.
    """
    mesh = _icosphere(subdivisions=3)
    tris = GEO.of_faces(mesh.vertices, mesh.faces)
    w = GEO._winding_numbers(tris, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert w[0] == pytest.approx(1.0, abs=1e-12)
    assert w[1] == pytest.approx(0.0, abs=1e-12)


def test_point_in_mesh_agrees_with_the_polyhedron_not_the_ideal_sphere():
    """The control that a wrong reference would fail.

    A subdivided icosphere is INSCRIBED in its circumsphere, so a point at
    radius 0.9999 is outside the polyhedron even though it is inside the
    mathematical sphere. This pins the test to the mesh's own geometry.

    Measured inradius 0.999101 at 5120 faces: 512 points strictly inside the
    inradius all classify inside, 3486 strictly outside the circumradius all
    classify outside. Only the inradius is asserted on, with a margin, because
    the circumradius is the fixture's own approximation.
    """
    mesh = _icosphere(subdivisions=4)
    r = _inradius(mesh)
    rng = np.random.default_rng(7)
    P = rng.uniform(-1.6, 1.6, size=(4000, 3))
    radius = np.linalg.norm(P, axis=1)
    # no explicit chunk: the derived one must be small enough to allocate here
    inside = GEO.point_in_mesh(mesh, P)[0]

    strictly_in = radius < r - 1e-9
    strictly_out = radius > 1.0 + 1e-9
    # control: the fixture must actually produce both classes, or the test
    # proves nothing about either
    assert strictly_in.sum() > 100, "fixture produced too few provably-inside points"
    assert strictly_out.sum() > 100, "fixture produced too few provably-outside points"
    assert inside[strictly_in].all(), "a point provably inside was called outside"
    assert not inside[strictly_out].any(), "a point provably outside was called inside"


def test_a_point_on_the_surface_is_reported_not_guessed():
    """A point exactly on the surface has no defined side, and says so.

    This is the case that produced a 251-straddler miscount in planning: a point
    ON an interface was counted as strictly inside, so a correctly-conforming
    mesh looked broken. The caller must be told which points are untrustworthy
    rather than being handed a classification.
    """
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    on_face = np.array([[1.0, 0.0, 0.0]])
    _, on_surface = GEO.point_in_mesh(mesh, on_face)
    assert on_surface[0], "a point on the +x face must be reported as on-surface"

    _, clean = GEO.point_in_mesh(mesh, np.array([[0.5, 0.0, 0.0]]))
    assert not clean[0]


def test_nested_boxes_classify_independently():
    """A point in the inner box is inside BOTH; between them, only the outer.

    The inner box is 0.5 across, so its half-extent is 0.25: the middle point
    at 0.4 is OUTSIDE the inner box and inside the outer. An earlier version of
    this test asserted both points were inside the inner box, which is a
    statement about a different box than the one it built.
    """
    outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    inner = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])

    inner_inside = GEO.point_in_mesh(inner, points)[0]
    outer_inside = GEO.point_in_mesh(outer, points)[0]
    assert inner_inside.tolist() == [True, False]
    assert outer_inside.tolist() == [True, True]


# ---------------------------------------------------------------------------
# The chunking the lead required: bounded memory must not change the answer.
# ---------------------------------------------------------------------------
def test_chunking_does_not_change_the_answer():
    """A moderate size where the chunked and unchunked paths are compared.

    Required explicitly, because the chunking exists to bound a temporary array
    and a bound that silently dropped or duplicated a chunk would still return a
    plausible-looking classification. The sizes cross several chunk boundaries
    and include a chunk of 1, where every point is its own call.
    """
    mesh = _icosphere(subdivisions=3)
    rng = np.random.default_rng(11)
    P = rng.uniform(-1.5, 1.5, size=(500, 3))

    reference = GEO.point_in_mesh(mesh, P, chunk=10 ** 9)[0]
    for chunk in (1, 7, 64, 4096):
        got = GEO.point_in_mesh(mesh, P, chunk=chunk)[0]
        assert np.array_equal(reference, got), f"chunk={chunk} changed the answer"
    # control: the reference is not trivially all-False, which would make the
    # comparison above pass for any implementation that also returned all-False
    assert reference.any() and not reference.all()


def test_a_large_point_set_is_chunked_within_a_memory_bound():
    """The cost story, as a test: unbounded is refused, bounded succeeds.

    ``chunk`` bounds the ``(n_chunk, n_faces, 3)`` temporary. At 4000 points
    against 5120 faces the unbounded form wants ~469 MiB in ONE allocation and
    dies under a 4 GiB address-space limit; the same call at chunk=256 is fine.
    A future "optimisation" that removes the chunking fails here rather than in
    the solver pod.
    """
    mesh = _icosphere(subdivisions=4)
    assert len(mesh.faces) >= 5000, "fixture must be big enough to matter"
    P = np.zeros((4000, 3))
    P[:, 0] = np.linspace(-0.5, 0.5, 4000)

    # The DERIVED chunk must be small enough to allocate: this is the default
    # path, and it is what a caller gets without passing anything.
    derived = GEO._chunk_for(len(mesh.faces))
    assert derived * len(mesh.faces) <= GEO.WINDING_PAIR_BUDGET or derived == 1

    inside = GEO.point_in_mesh(mesh, P)[0]
    assert inside.shape == (4000,)
    assert inside.any()

    # ...and the SAME call with the bound removed must fail on this machine, or
    # the test is not measuring the bound it claims to. Under the tier-1
    # address-space limit the unbounded allocation raises MemoryError; where a
    # bigger limit is available it may succeed, so a success is not a failure
    # of the code -- it is reported by the assertion below being skipped.
    try:
        GEO.point_in_mesh(mesh, P, chunk=10 ** 9)[0]
    except MemoryError:
        pass
    else:
        pytest.skip(
            "no address-space limit in force, so the unbounded form did not "
            "fail; the bound is still asserted above"
        )


# ---------------------------------------------------------------------------
# The bounding-box pre-filter: the cheap test that keeps e-mail-sized work off
# the web request.
# ---------------------------------------------------------------------------
def test_bounding_boxes_separate_the_pairs_that_cannot_interact():
    """Disjoint boxes are PROVEN apart, so the expensive test never sees them."""
    near = trimesh.creation.box(extents=(1.0, 1.0, 1.0))          # +/-0.5
    far = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    far.apply_translation([5.0, 0.0, 0.0])                        # 4.5 gap
    assert GEO.parts_may_interact([near, far]) == []

    touching = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    touching.apply_translation([1.0, 0.0, 0.0])                   # faces meet
    assert GEO.parts_may_interact([near, touching]) == [(0, 1)]


def test_the_pre_filter_is_conservative_not_clever():
    """A pair that is merely CLOSE must still be tested.

    The filter exists to skip work, and skipping a pair that really does touch
    would turn a refusal into a silent wrong material -- so the filter must err
    toward testing. With a tolerance it does: boxes 0.05 apart are candidates
    at tol=0.1 and not at tol=0.
    """
    a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b.apply_translation([1.05, 0.0, 0.0])
    assert GEO.parts_may_interact([a, b], tol=0.0) == []
    assert GEO.parts_may_interact([a, b], tol=0.1) == [(0, 1)]


# ---------------------------------------------------------------------------
# Overlap and touching are DIFFERENT instruments and must not be confused.
# ---------------------------------------------------------------------------
def test_a_nested_part_is_containment_not_overlap_or_touching():
    """Containment is legal: the row's whole feature is parts inside parts.

    Note what the vertex measurement actually says here, because it is easy to
    misread: the inner box's vertices ARE strictly inside the outer box, so
    ``overlap_vertices`` returns them. That is not a bug -- it is what "inside"
    means. The distinction between a nested part and a colliding one is the
    DIRECTION: containment is inner-inside-outer with nothing of the outer
    inside the inner, and ``part_contains`` is what decides it.

    So a caller must ask ``part_contains``; treating ``overlap_vertices``'
    non-empty result as "these parts collide" would refuse every multi-material
    upload, which is the failure this test exists to prevent.
    """
    outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    inner = trimesh.creation.box(extents=(0.5, 0.5, 0.5))

    assert GEO.part_contains(outer, inner), "the inner part is inside the outer"
    assert not GEO.part_contains(inner, outer), "not the other way round"

    # ...and the parts do not TOUCH: there is a gap between them, so TetGen has
    # somewhere to put the interface it needs.
    assert len(GEO.touching_vertices(outer, inner)[0]) == 0
    assert len(GEO.touching_vertices(outer, inner)[1]) == 0


def test_an_overlapping_pair_is_reported_with_vertex_indices():
    """A vertex-detected interpenetration is found, and named by vertex index.

    A small box buried in a big one: the small box's vertices are strictly
    inside the big one, which is the case a vertex test CAN decide.
    """
    big = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    small = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    small.apply_translation([0.5, 0.0, 0.0])        # fully inside, off-centre
    small_in_big, big_in_small = GEO.overlap_vertices(big, small)
    assert len(big_in_small) > 0, "the buried box's vertices are inside the big one"
    assert big_in_small.max() < len(small.vertices), "indices must be into small"


def test_a_vertex_test_cannot_see_a_same_size_overlap_and_says_so():
    """The documented blind spot, asserted rather than assumed.

    Two identical boxes offset by half a box overlap substantially, and every
    vertex of each lands exactly ON the other's surface -- so a vertex test
    finds NO overlap. Measured: zero vertices strictly inside on either side.

    This is pinned as a test because the empty result is indistinguishable from
    "no overlap", which is the could-not-ask shape: if anyone later treats this
    function's empty result as proof of separation, this test records why that
    is wrong and points at ``docheck``, which is the instrument that does decide
    it.
    """
    a = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    b.apply_translation([1.0, 0.0, 0.0])            # a real half-box overlap
    a_in_b, b_in_a = GEO.overlap_vertices(a, b)
    assert len(a_in_b) == 0 and len(b_in_a) == 0, (
        "if this ever finds vertices, the docstring's stated limit has changed "
        "and the overlap decision may no longer need docheck"
    )
    # ...but the touching test DOES see them, because they are exactly on the
    # other's surface. The two functions answer different questions.
    assert len(GEO.touching_vertices(a, b)[0]) > 0


def test_a_touching_pair_is_reported_by_the_touching_test_not_the_overlap_test():
    """The measured split: two boxes sharing a face plane.

    TetGen's own ``docheck`` accepted exactly this pair (12 tets, no error), so
    this is the case the mesher cannot refuse on its own. It must be found here,
    and it must NOT be reported as an overlap -- the two refusals name different
    problems, and a message that says "overlap" for a touch sends the owner
    looking for geometry that is not there.
    """
    a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b.apply_translation([1.0, 0.0, 0.0])            # the +x face coincides
    a_on_b, b_on_a = GEO.touching_vertices(a, b)
    assert len(a_on_b) > 0 and len(b_on_a) > 0, "the shared face must be found"
    a_in_b, b_in_a = GEO.overlap_vertices(a, b)
    assert len(a_in_b) == 0 and len(b_in_a) == 0, "a touch is not an overlap"


def test_a_clear_gap_reports_neither():
    """The negative control: two parts that are genuinely apart.

    Without this, a function that returned "touching" for every pair would pass
    both tests above.
    """
    a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b.apply_translation([1.5, 0.0, 0.0])
    assert len(GEO.touching_vertices(a, b)[0]) == 0
    assert len(GEO.overlap_vertices(a, b)[0]) == 0


# ---------------------------------------------------------------------------
# Refusal messages name the part.
# ---------------------------------------------------------------------------
def test_a_refusal_names_the_part_it_is_about():
    """A refusal the owner cannot act on is the could-not-ask shape."""
    err = GEO.formation_error("two parts overlap", "core")
    assert "core" in str(err)
    assert "overlap" in str(err)
