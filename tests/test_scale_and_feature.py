"""The two owner-settable mesh inputs: the smallest feature, and the scale.

Both are properties of the *upload*, not of the machine, and both move the
element count by more than a routing threshold -- which is why they are tested
together and why the estimate is tested against them rather than only the mesh.

The specific defect this file exists to prevent: ``min_feature_mm`` was accepted
by the upload form, validated, and then dropped on the floor. The route was
therefore computed from the *physics* edge for a job that would be meshed at a
much finer *feature* edge, so the detailed uploads -- exactly the ones the GPU
queue exists for -- were the ones routed to the machine that cannot solve them.
That is the rig's ``index-and-fetcher-must-agree`` shape one level up: two
components computing one quantity from different inputs.

The scale tests are the same defect with a different lever: the estimate and the
solver must both see the scaled object, and an unscaled job must be byte-
identical to what it was before the parameter existed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))
sys.path.insert(0, str(ROOT / "web"))

import mse_meshing as MESH  # noqa: E402

import meshing as WEB  # noqa: E402


# ---------------------------------------------------------------------------
# The unit table
# ---------------------------------------------------------------------------

def test_the_optical_units_are_the_ones_they_look_like():
    """um/nn are off by 1000x and 1e6 if anyone ever transposes them.

    Pinned as exact ratios to the metre rather than to each other, because a
    table that is uniformly wrong by a factor still has correct ratios.
    """
    assert MESH.UNIT_SCALE["nm"] == 1e-9
    assert MESH.UNIT_SCALE["um"] == 1e-6
    assert MESH.UNIT_SCALE["mm"] == 1e-3
    assert MESH.UNIT_SCALE["cm"] == 1e-2
    assert MESH.UNIT_SCALE["m"] == 1.0
    assert MESH.UNIT_SCALE["in"] == 0.0254
    # and the ordering is a real ordering: nm < um < mm < cm < m
    ordered = [MESH.UNIT_SCALE[u] for u in ("nm", "um", "mm", "cm", "m")]
    assert ordered == sorted(ordered)


def test_the_web_unit_menu_offers_exactly_what_the_solver_supports():
    """A menu entry the solver would reject is a form that cannot be submitted.

    The web app validates against its own list and the solver against
    ``UNIT_SCALE``; if they drift, the failure is a confusing 500 after a long
    upload rather than a clear early refusal.
    """
    import config as CFG

    assert set(CFG.UNIT_CHOICES) == set(MESH.UNIT_SCALE)
    assert CFG.DEFAULT_UNITS == "mm"


# ---------------------------------------------------------------------------
# min_feature_mm: smaller than the physics edge, or no effect at all
# ---------------------------------------------------------------------------

def _box_surface(extent_mm: float = 10.0):
    return trimesh.creation.box(extents=[extent_mm * 0.8, extent_mm * 0.6,
                                         extent_mm * 0.4])


def test_a_feature_smaller_than_the_physics_edge_lowers_the_target_edge():
    """The whole point of the control: the mesh must actually get finer."""
    surface = _box_surface()
    physics = WEB.target_edge_length_mm(surface, 20, "mm")
    feature = physics / 4.0

    effective = WEB.effective_edge_mm(surface, 20, "mm", feature)
    assert effective == pytest.approx(feature)
    assert effective < physics


def test_a_feature_larger_than_the_physics_edge_changes_nothing():
    """The mesh is never made coarser than the physics requires.

    Naming a large feature is a statement that you do not need fine resolution,
    not a licence to under-resolve the modes -- so the answer is the physics
    edge, exactly.
    """
    surface = _box_surface()
    physics = WEB.target_edge_length_mm(surface, 20, "mm")

    assert WEB.effective_edge_mm(surface, 20, "mm", physics * 10.0) == physics
    assert WEB.effective_edge_mm(surface, 20, "mm", None) == physics


def test_auto_is_byte_identical_to_the_previous_behaviour():
    """'auto' must not move a single bit, not merely land close.

    The lead asked for byte-identical, and 'close' is the weaker claim that lets
    a refactor quietly change every existing job's mesh. Exact equality also
    catches a pass-through that re-derives the edge through a different formula.
    """
    surface = _box_surface()
    without = WEB.estimate_tets(surface, 20, "mm", c=14.0)
    with_auto = WEB.estimate_tets(surface, 20, "mm", c=14.0,
                                  min_feature_mm=None, scale=1.0)

    assert without == with_auto
    assert WEB.effective_edge_mm(surface, 20, "mm") == \
        WEB.target_edge_length_mm(surface, 20, "mm")


def test_a_smaller_feature_raises_the_estimate_not_just_the_edge():
    """The routing consequence, which is the reason this matters.

    Edge /4 is 64x the elements by the h^-3 law, so a feature that is below the
    physics edge must move the estimate by roughly that factor -- and must move
    it in the safe direction, because the whole point is that these jobs go to
    the GPU.
    """
    surface = _box_surface()
    physics = WEB.target_edge_length_mm(surface, 20, "mm")
    auto = WEB.estimate_tets(surface, 20, "mm", c=14.0)
    fine = WEB.estimate_tets(surface, 20, "mm", c=14.0,
                             min_feature_mm=physics / 4.0)

    assert fine > auto
    # 4^3 = 64 within the rounding of an int estimate
    assert fine / auto == pytest.approx(64.0, rel=0.05)


def test_a_nonsense_feature_is_ignored_rather_than_dividing_by_it():
    """0, negative and NaN all mean 'no opinion', not 'infinitely fine'.

    The upload form refuses these, so this is the second line of defence -- and
    it has to be: an estimate with an edge of 0 is a division by zero, and a
    routing decision computed from an exception is the thing this file is about.
    """
    surface = _box_surface()
    physics = WEB.target_edge_length_mm(surface, 20, "mm")

    for bad in (0.0, -1.0, float("nan")):
        assert WEB.effective_edge_mm(surface, 20, "mm", bad) == physics


# ---------------------------------------------------------------------------
# scale: applied to the object before anything measures it
# ---------------------------------------------------------------------------

def test_apply_scale_multiplies_every_vertex():
    surface = _box_surface(10.0)
    scaled = MESH.apply_scale(surface, 3.0)

    assert np.allclose(np.asarray(scaled.vertices) * 1.0,
                       np.asarray(surface.vertices) * 3.0)
    # and the original is untouched: the caller still holds it
    assert not np.allclose(np.asarray(surface.vertices),
                           np.asarray(scaled.vertices))


def test_scale_of_one_returns_the_same_object_not_a_copy():
    """Cheap AND meaningful: an unscaled upload takes no copy path at all."""
    surface = _box_surface()
    assert MESH.apply_scale(surface, 1.0) is surface


def test_a_bad_scale_raises_rather_than_producing_an_empty_mesh():
    surface = _box_surface()
    for bad in (0.0, -2.0, float("nan")):
        with pytest.raises(MESH.MeshingError):
            MESH.apply_scale(surface, bad)


def test_a_scaled_object_meshes_to_the_same_element_count():
    """V scales as s^3 and h scales as s, so V/h^3 scales as s^2.

    Getting this wrong in either direction sends a job across the routing
    boundary the wrong way, which is why it is asserted as a ratio and not as
    'it went up'.
    """
    surface = _box_surface()
    base = WEB.estimate_tets(surface, 20, "mm", c=14.0)
    bigger = WEB.estimate_tets(surface, 20, "mm", c=14.0, scale=10.0)

    # A ten-times-larger object needs a ten-times-longer edge, so the count goes
    # as 10^3 / 10^3 ... but the volume is the ENCLOSED volume and the edge is
    # the physics edge, which is set by the volume: both scale, and the count of
    # an exactly-similar object is SCALE INVARIANT.
    assert bigger == pytest.approx(base, rel=1e-9), (
        "a similar object meshes to the same element count -- if this moves, "
        "the estimate is not scale-invariant and routing depends on the units"
    )


def test_the_enclosed_volume_the_estimate_uses_actually_scales():
    """The companion to the test above: it must be similarity, not a no-op.

    The estimate being scale-invariant is only meaningful if the volume it
    divides by really did grow -- otherwise both could be ignoring the scale
    factor entirely and the invariant would prove nothing.
    """
    surface = _box_surface()
    small = MESH.enclosure_volume_m3(MESH.apply_scale(surface, 1.0), "mm")
    large = MESH.enclosure_volume_m3(MESH.apply_scale(surface, 10.0), "mm")

    assert large / small == pytest.approx(1000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# The solver really does use them (not just the web app's copy)
# ---------------------------------------------------------------------------
#
# These three go through the real ``tetrahedralize`` with TetGen faked, and
# assert on the ``maxvolume`` the loop asked for -- the only outward trace of
# the target edge, and therefore the only way to see from outside whether the
# feature size and the scale actually reached the mesher.
#
# ``enforce_resolution=False`` is not a convenience here, it is what makes the
# measurement possible: the shared ``FakeTetGen`` returns a fixed unit cube
# whose every edge is 1.0, which is far coarser than any target these tests use,
# so with the guard armed the mesh is refused before ``maxvolume`` can be read.
# Disarming it takes the bench's documented ``--allow-coarse`` path, which breaks
# out on the first attempt -- and that matters, because the retry path DIVIDES
# maxvolume by the overshoot ratio, so an armed retry would report a number that
# is not the one the target edge implies. The guard itself is tested properly,
# with a mesh that genuinely is too coarse, in test_meshing_guard.py.

def _fake_mesher(monkeypatch, target_edges=(0.5,)):
    """Install FakeTetGen and a scripted ``target_edge_length``.

    ``target_edges`` is consumed in order, so a test can give successive calls
    different physics edges -- which is how the scale tests separate "the edge
    moved because the object got bigger" from "the edge was overridden".
    """
    import types

    from test_meshing_guard import FakeTetGen  # reuse the shared tiny mesh

    FakeTetGen.calls = []
    edges = list(target_edges)

    def scripted(*a, **k):
        return edges[0] if len(edges) == 1 else edges.pop(0)

    monkeypatch.setattr(MESH, "target_edge_length", scripted)
    monkeypatch.setattr(MESH, "decimate_to_edge_length",
                        lambda surface, edge_length, verbose=True: (surface, None))

    module = types.ModuleType("tetgen")
    module.TetGen = FakeTetGen
    monkeypatch.setitem(sys.modules, "tetgen", module)
    return FakeTetGen


def _mesh_volume(fake, surface, **kwargs):
    """Run the real mesher and return the first ``maxvolume`` it asked for."""
    del fake.calls[:]
    MESH.tetrahedralize(surface, "mm", n_modes=20, max_tets=10_000,
                        verbose=False, enforce_resolution=False, **kwargs)
    assert fake.calls, "the mesher never reached TetGen"
    return fake.calls[0]


def test_the_solver_uses_the_feature_edge_it_is_given(monkeypatch):
    """A feature below the physics edge really does refine the mesher.

    If the feature size never reached the mesher, ``maxvolume`` would be the
    physics one and this test would read that back instead.
    """
    fake = _fake_mesher(monkeypatch, target_edges=(0.5,))

    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    with_feature = _mesh_volume(fake, surface, min_feature_mm=0.125)
    without = _mesh_volume(fake, surface)

    assert with_feature < without
    # A 4x finer target edge is a 64x smaller maxvolume: target_maxvolume is
    # proportional to edge^3, and this is the exact factor the routing estimate
    # depends on.
    assert without / with_feature == pytest.approx(64.0, rel=0.01)


def test_the_feature_never_coarsens_the_mesher_target(monkeypatch):
    """A feature larger than the physics edge leaves maxvolume alone."""
    fake = _fake_mesher(monkeypatch, target_edges=(0.5,))

    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    coarse_feature = _mesh_volume(fake, surface, min_feature_mm=5.0)
    plain = _mesh_volume(fake, surface)

    assert coarse_feature == pytest.approx(plain), (
        "naming a LARGE feature must not make the mesh coarser than the "
        "physics alone requires"
    )


def test_the_mesher_is_handed_the_scaled_surface(monkeypatch):
    """The scale must reach the SURFACE, proved by measuring the surface.

    An earlier version of this test asserted on ``maxvolume`` alone, and that
    was not evidence: neutralizing ``apply_scale`` left it green, because the
    physics edge comes from ``enclosure_volume_m3(surface)`` and both the
    volume and the edge are read from the same (unscaled) surface, so their
    ratio is scale-invariant under that mutation -- the number it checked
    barely moved either way. The quantity that actually distinguishes fixed
    from unfixed is the object the mesher was handed, so that is what is
    measured here.

    Neutering ``apply_scale``'s multiply reddens this test; it does not redden
    the maxvolume one.
    """
    seen = {}

    real_apply = MESH.apply_scale

    def recording_apply(surface, factor):
        out = real_apply(surface, factor)
        seen["extent"] = float(np.max(np.asarray(out.vertices, dtype=float))
                               - np.min(np.asarray(out.vertices, dtype=float)))
        return out

    monkeypatch.setattr(MESH, "apply_scale", recording_apply)
    fake = _fake_mesher(monkeypatch, target_edges=(0.5,))

    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    _mesh_volume(fake, surface, scale=10.0)
    scaled_extent = seen["extent"]

    _mesh_volume(fake, surface)
    unscaled_extent = seen["extent"]

    assert scaled_extent / unscaled_extent == pytest.approx(10.0, rel=1e-9), (
        "the surface the mesher was handed did not grow by the scale factor"
    )


def test_the_estimate_is_linear_in_the_mode_count():
    """The control that DOES move the estimate, so the invariant means something.

    ``test_a_scaled_object_meshes_to_the_same_element_count`` asserts a null
    result, and a null result is only evidence if the instrument can detect
    motion at all. This is that motion: the estimate is linear in the mode count
    (Weyl's law cancels the exponents -- k ~ N^(1/3), h ~ N^(-1/3), and
    tets ~ h^-3, so tets ~ N exactly). An estimate that had frozen into a
    constant would pass the invariant test and fail this one.

    Linear -- not N^1.5. I asserted N^1.5 first and the measurement refuted it.
    """
    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    counts = {n: WEB.estimate_tets(surface, n, "mm", c=14.0) for n in (5, 20, 80)}

    assert counts[20] == pytest.approx(4 * counts[5], rel=0.01)
    assert counts[80] == pytest.approx(16 * counts[5], rel=0.01)


def test_the_scale_cannot_move_the_estimate_and_that_is_correct():
    """A scale is a similarity transform, so the element count cannot change.

    This is worth stating as a test because it looks like a bug and is not one.
    ``est_tets`` is ``C * V / h^3`` with V the enclosed volume and h the physics
    edge, and h is derived from V -- so both scale with the object and their
    ratio does not move. A similar object meshes to a similar mesh; that is the
    correct physics, and it means the scale factor is deliberately NOT an input
    to the routing decision.

    The consequence, stated plainly because it is the kind of thing that gets
    mistaken for a defect later: a scale factor can never move a job between the
    cluster and the GPU queue. What DOES move it is ``min_feature_mm``, which
    breaks the similarity by naming an absolute length -- see
    ``test_a_smaller_feature_raises_the_estimate_not_just_the_edge``.
    """
    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    base = WEB.estimate_tets(surface, 20, "mm", c=14.0)

    for factor in (2.0, 10.0, 1000.0, 1e6):
        assert WEB.estimate_tets(surface, 20, "mm", c=14.0,
                                 scale=factor) == base, (
            f"scale {factor} moved the estimate; if the scale is meant to reach "
            f"the mesher as a similarity transform it must not"
        )


def test_the_feature_size_is_what_moves_the_estimate():
    """The contrast: an ABSOLUTE length does move it, and hard.

    Naming a feature breaks the similarity that makes the estimate scale-free,
    which is why `min_feature_mm` -- not `scale` -- is the input the routing
    decision is sensitive to.
    """
    surface = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
    physics = WEB.target_edge_length_mm(surface, 20, "mm")
    auto = WEB.estimate_tets(surface, 20, "mm", c=14.0)
    fine = WEB.estimate_tets(surface, 20, "mm", c=14.0,
                             min_feature_mm=physics / 4.0)

    assert fine == pytest.approx(64 * auto, rel=0.05), (
        "an absolute feature edge 4x below the physics one is 4^3 = 64x the "
        "elements -- the factor that crosses the GPU threshold"
    )
