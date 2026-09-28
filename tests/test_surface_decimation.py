"""Surface decimation: an over-fine surface must be coarsened toward the target.

Why this exists: TetGen cannot make interior elements much larger than the
surface facets, so a surface finer than the target element edge floors the whole
mesh no matter what ``maxvolume`` says. The lead's Jetson run proved it -- a
radius-5 mm sphere went 212,000 -> 196,569 -> 202,931 elements across two
``maxvolume`` triplings and was then refused, because the *surface* (not the
volume bound) was setting the size. Owner uploads from CAD or a 3-D scan are
routinely far finer than the target edge, so before this, every such upload was
refused as "too large" even for a simple shape.

These tests use real trimesh surfaces and real pyvista decimation. They are
deliberately small: a few-thousand-face icosphere, decimated, is well inside the
"no multi-second or large-mesh work on the VM" rule and runs in about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_meshing as MESH  # noqa: E402

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("pyvista")

#: small enough to stay quick, fine enough that its facets are well under the
#: target edge -- the shape of a real CAD/scan upload at toy size
OVER_REFINED_SUBDIVISIONS = 4
SPHERE_RADIUS_MM = 5.0


def _sphere(subdivisions=OVER_REFINED_SUBDIVISIONS, radius=SPHERE_RADIUS_MM):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.merge_vertices()
    return mesh


def _target(mesh, n_modes=20):
    return MESH.target_edge_length(mesh, n_modes=n_modes, units="mm")


# ---------------------------------------------------------------------------
# The trigger: only an over-fine surface is touched.
# ---------------------------------------------------------------------------
def test_an_over_fine_surface_is_coarsened_toward_the_target():
    """The decimation must move the median facet edge toward the target edge."""
    mesh = _sphere()
    target = _target(mesh)
    before_faces = len(mesh.faces)
    before_edge = MESH.median_surface_edge(mesh)

    # this is the premise of the whole fix: the surface really is far finer
    assert before_edge < 0.7 * target, (
        f"test fixture is not over-refined: facet edge {before_edge:.4g} vs "
        f"target {target:.4g}"
    )

    decimated, note = MESH.decimate_to_edge_length(mesh, target)

    assert note, "a decimation that changed the surface must explain itself"
    assert len(decimated.faces) < before_faces
    after_edge = MESH.median_surface_edge(decimated)
    assert after_edge > before_edge, "the surface must have coarsened"
    # the whole point: it lands near the target, not still an order below it
    assert after_edge > 0.7 * target
    assert after_edge < 2.0 * target


def test_the_decimated_surface_stays_watertight():
    """Watertightness is what the PEC boundary condition rests on.

    A decimation that opens the surface would produce a leaking cavity, so this
    is a correctness requirement, not a nicety.
    """
    mesh = _sphere()
    target = _target(mesh)
    decimated, _note = MESH.decimate_to_edge_length(mesh, target)
    assert decimated.is_watertight
    assert len(decimated.faces) > 0


def test_decimation_preserves_the_enclosed_volume():
    """Coarsening must not change the object the owner uploaded.

    The resonance frequencies come from the enclosed volume, so a decimation
    that changed it materially would move every mode.
    """
    mesh = _sphere()
    target = _target(mesh)
    decimated, _note = MESH.decimate_to_edge_length(mesh, target)
    assert decimated.volume == pytest.approx(mesh.volume, rel=0.02)


def test_a_coarse_surface_is_left_alone():
    """Facets larger than the target need no work: TetGen adds Steiner points.

    The trigger must be one-sided, or a legitimately coarse upload would be
    "decimated" toward a target it already exceeds.
    """
    mesh = _sphere(subdivisions=2)          # facets comfortably larger than h
    target = _target(mesh)
    assert MESH.median_surface_edge(mesh) / target >= MESH.SURFACE_DECIMATE_TRIGGER

    out, note = MESH.decimate_to_edge_length(mesh, target)
    assert out is mesh, "an already-coarse surface must be returned untouched"
    assert note is None


def test_a_surface_already_at_the_target_is_left_alone():
    """Exactly at the trigger counts as coarse enough."""
    mesh = _sphere()
    target = _target(mesh)
    at_target = target * MESH.SURFACE_DECIMATE_TRIGGER
    out, note = MESH.decimate_to_edge_length(mesh, at_target)
    assert out is mesh
    assert note is None


# ---------------------------------------------------------------------------
# target_face_count: the request is the isotropically correct count.
# ---------------------------------------------------------------------------
def test_target_face_count_matches_the_triangle_area_formula():
    """``faces ~ area / (sqrt(3)/4 * h^2)`` -- an equilateral triangle of edge h."""
    mesh = _sphere()
    h = 0.5
    expect = mesh.area / (np.sqrt(3.0) / 4.0 * h ** 2)
    # the result is an integer face count, so allow the truncation
    assert MESH.target_face_count(mesh, h) == pytest.approx(expect, abs=1)


def test_target_face_count_scales_as_inverse_edge_squared():
    """Halving the target edge must quadruple the requested face count."""
    mesh = _sphere()
    a = MESH.target_face_count(mesh, 1.0)
    b = MESH.target_face_count(mesh, 0.5)
    # 4x up to the integer truncation at each end
    assert b == pytest.approx(4 * a, abs=4)


# ---------------------------------------------------------------------------
# decimate_surface: the pyvista call is easy to get silently wrong.
# ---------------------------------------------------------------------------
def test_decimate_surface_takes_a_count_not_a_fraction():
    """Regression: pyvista's ``decimate`` takes a reduction FRACTION.

    An integer above 1 is passed through to VTK's ``SetTargetReduction`` and
    returns an EMPTY mesh with no error at all -- a silent total loss of the
    surface. The wrapper must convert a face count into ``1 - want/n`` and must
    reject an empty result rather than hand it back.
    """
    mesh = _sphere()
    want = 400
    out = MESH.decimate_surface(mesh, want)
    assert out is not None
    assert out.is_watertight
    # a fraction would have been read as "remove 400x", so this is the guard
    assert 0.5 * want <= len(out.faces) <= 2.0 * want


def test_decimate_surface_never_returns_an_empty_mesh():
    """Out-of-range requests must clamp, not vanish.

    Asking for more faces than exist, or absurdly few, must never come back as
    a mesh with no faces: the caller treats ``None`` as "keep the original" but
    an empty Trimesh would sail through as a zero-triangle surface.
    """
    mesh = _sphere()
    assert MESH.decimate_surface(mesh, len(mesh.faces) * 10) is not None
    tiny = MESH.decimate_surface(mesh, 1)
    assert tiny is None or len(tiny.faces) > 0


def test_decimate_surface_does_nothing_when_already_small_enough():
    """Asking for at least as many faces as exist returns the surface itself."""
    mesh = _sphere()
    out = MESH.decimate_surface(mesh, len(mesh.faces))
    assert out is mesh


# ---------------------------------------------------------------------------
# The production path: an over-fine surface must now MESH, not be refused.
# ---------------------------------------------------------------------------
def test_an_over_fine_surface_meshes_under_the_cap():
    """End of the story: the upload that used to blow the cap now fits.

    The element count is the number that matters -- it is what the cap refuses
    on -- and the achieved edge must be within the resolution guard's tolerance,
    or the fix would just trade a refusal for silently wrong frequencies.
    """
    mesh = _sphere()
    cap = 80_000

    points_m, tets, _ = MESH.tetrahedralize(
        mesh, "mm", n_modes=20, max_tets=cap, verbose=False
    )

    assert len(tets) <= cap
    # a sphere of R=5mm at h~0.72mm lands around 13k; allow generous headroom
    assert len(tets) < 40_000, (
        f"{len(tets):,} elements: decimation did not take effect"
    )

    target = _target(mesh)
    achieved = MESH.achieved_edge_length(points_m, tets)
    assert achieved <= MESH.MAX_EDGE_OVERSHOOT * target * MESH.unit_scale("mm")
