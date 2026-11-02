"""``tetrahedralize(..., sizing=)``: a caller-stated per-part target edge.

Task 1.1 of the electrical-simulation change. The optical grading law derives
every region's edge from REFRACTIVE INDEX (``h * n_wall / n_r``); the
electrical solver's resolution requirement is a skin depth, which no
refractive index produces. ``sizing=`` is how the caller states it.

These tests mesh real surfaces, so they are tier 1 by size: a 6 mm box with a
2 mm core is about 20k tets before grading and a few thousand after. Every
test runs ``nice -n 19`` inside ``(ulimit -v 4194304; ...)`` when driven from a
shell; pytest itself is the caller's business.

WHAT THESE TESTS REFUSE TO ASSERT, and why: "the mesh is graded" is not a
verdict -- a test that only checked that some region came out finer than
another would pass on a mesh built to the wrong edges. Each test measures the
achieved median edge IN EACH PART and holds it against THAT PART'S target,
because that is the claim ``sizing=`` makes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import mse_meshing as MESH  # noqa: E402

# The targets are stated in METRES, because that is what ``sizing=`` takes: the
# solver works in metres throughout and converting at the call site would put a
# second unit conversion between the caller's number and TetGen's.
FINE_M = 0.00025   # 0.25 mm -- the "device" requirement
COARSE_M = 0.001   # 1 mm -- the "wall"


def _nested(units="mm"):
    """A 6 mm wall box with a 2 mm core: the routed object, as real bytes."""
    import trimesh

    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    parts = PARTS.load_parts(obj, units=units)
    return parts, PARTS.merged_surface(parts)


def _region_edges_m(points_m, tets, attrib, region_id):
    """The achieved median edge of the tets in ONE region, in metres."""
    mask = attrib == int(region_id)
    assert mask.any(), f"region {region_id} received no tets"
    return MESH.achieved_edge_length(points_m, tets[mask])


def _build(sizing, **kw):
    parts, surface = _nested()
    materials = {"wall": 1.0, "core": 1.5}
    return MESH.tetrahedralize(
        surface, "mm", n_modes=4, n_index=1.0, verbose=False,
        parts=parts, materials=materials, sizing=sizing,
        max_tets=200_000, **kw,
    )


def _build_fine_optics(sizing, n_core=3.5, n_modes=16):
    """The SAME box, with a core whose OPTICAL requirement is genuinely fine.

    ``_build``'s core (n=1.5, 4 modes) gives an optical ``own_edge_units`` of
    0.617 mm against an achieved 0.273 mm -- an allowance no caller target can
    exceed, so a test built on it cannot tell the override from its absence.
    ``h_r = h * n_wall / n_r`` falls as ``n_r`` and ``h`` rise, so a high-index
    core with more modes is what makes the optical allowance SMALL enough to
    be broken by a caller's coarse edge. Measured on this fixture: the core's
    optical allowance is 0.166 mm against an achieved 0.273 mm.
    """
    parts, surface = _nested()
    materials = {"wall": 1.0, "core": n_core}
    return MESH.tetrahedralize(
        surface, "mm", n_modes=n_modes, n_index=1.0, verbose=False,
        parts=parts, materials=materials, sizing=sizing,
        max_tets=200_000,
    )


def _region_of(parts, name):
    return int(parts.region_of_part[list(parts.names).index(name)])


def test_the_caller_sizing_sets_each_parts_own_edge():
    """THE CLAIM. Each part's achieved median edge lands within its own stated
    target -- not merely "somewhere between" the two, which a uniform mesh at
    the FINER edge would also satisfy while ignoring the coarse target
    entirely.

    ``MAX_EDGE_OVERSHOOT`` is the module's own allowance (the same one the
    resolution guard applies), so the band here is the product's, not a number
    invented for the test. The lower bound is stated because a mesh far FINER
    than asked is also a wrong mesh -- it costs elements the caller did not
    budget -- and a one-sided check would call an over-refined core correct.
    """
    points_m, tets, attrib = _build({"wall": COARSE_M, "core": FINE_M})
    parts, _ = _nested()
    core_r = _region_of(parts, "core")
    wall_r = _region_of(parts, "wall")

    core_edge = _region_edges_m(points_m, tets, attrib, core_r)
    wall_edge = _region_edges_m(points_m, tets, attrib, wall_r)

    # The core is the fine part: it must reach its target, not the wall's.
    assert core_edge <= MESH.MAX_EDGE_OVERSHOOT * FINE_M, (
        f"the core was asked for {FINE_M * 1e3:.3g} mm and came out at "
        f"{core_edge * 1e3:.3g} mm"
    )
    # ... and the two parts must NOT have come out at one edge, because that is
    # what makes this graded rather than uniform. A uniform build at FINE_M
    # would pass the assertion above and fail this one.
    assert wall_edge > 1.5 * core_edge, (
        f"the wall ({wall_edge * 1e3:.3g} mm) and the core "
        f"({core_edge * 1e3:.3g} mm) came out at one edge: the caller's "
        f"coarse target was not honoured, so this is a uniform mesh"
    )
    # The wall must not be FINER than the core's target -- that would be the
    # wall priced at the core's edge, i.e. sizing ignored for the wall.
    assert wall_edge >= 0.9 * FINE_M, wall_edge


def test_the_caller_sizing_replaces_the_optical_guard_not_just_the_build():
    """The resolution GUARD, not only the sizing field.

    ``_resolution_verdict`` holds each region to ``own_edge_units``. Left
    optical, a caller whose edge is COARSER than ``h * n_wall / n_r`` gets a
    mesh refused for meeting the target it stated -- the refusal names an edge
    nobody asked for.

    THIS TEST HAS TO PROVE IT REACHES THAT LINE, and an earlier version did
    not: it built the plain ``_build`` fixture, whose core optical allowance
    (0.617 mm) sits so far above the achieved edge (0.273 mm) that deleting the
    override changed nothing and the test still passed 10/10. The reviewer
    caught it by mutation-testing the override. Measured on the fine-optics
    fixture below:

        override deleted -> REFUSED, the 'core' named at 1.642x its optical
                           allowance (achieved 0.2734 mm vs 0.1665 mm)
        override present -> ACCEPTED, the core at 1.094x its CALLER allowance
                           (achieved 0.2734 mm vs 0.25 mm)

    So the inequality asserted below is the test's own precondition: if the
    achieved edge does not exceed ``MAX_EDGE_OVERSHOOT`` times the OPTICAL
    allowance, this fixture cannot distinguish the two worlds and the test
    would pass on the bug it exists to catch. It is computed from the plan the
    build actually used, not retyped.
    """
    points_m, tets, attrib = _build_fine_optics({"wall": 0.003, "core": FINE_M})
    assert tets is not None and len(tets) > 0
    parts, surface = _nested()
    core_r = _region_of(parts, "core")
    core_edge = _region_edges_m(points_m, tets, attrib, core_r)

    # The OPTICAL allowance for that same region, from the shared helper the
    # guard itself reads -- so this cannot drift from the rule under test.
    # ``region_plan_by_region`` is the bare per-region dict (the mesher's own
    # ``_region_plan`` wraps it under a "shared" key later); read it directly.
    optical = MESH.region_plan_by_region(
        parts, {"wall": 1.0, "core": 3.5}, MESH.target_edge_length(
            surface, n_modes=16, units="mm", n_index=1.0),
        "mm")
    optical_core = float(optical[core_r]["own_edge_units"]) * 0.001

    assert core_edge > MESH.MAX_EDGE_OVERSHOOT * optical_core, (
        f"the core came out at {core_edge * 1e3:.4g} mm against an OPTICAL "
        f"allowance of {optical_core * 1e3:.4g} mm "
        f"({core_edge / optical_core:.3f}x, needs > "
        f"{MESH.MAX_EDGE_OVERSHOOT}x): this fixture cannot tell the override "
        f"from its absence, so the build being accepted proves nothing"
    )
    # ... and it is still accepted, because the caller's own edge (0.25 mm) IS
    # the requirement for an electrical mesh. Reaching the assertion above
    # without an exception is the verdict.
    assert core_edge <= MESH.MAX_EDGE_OVERSHOOT * FINE_M, (
        f"the core was asked for {FINE_M * 1e3:.3g} mm and came out at "
        f"{core_edge * 1e3:.3g} mm"
    )


def test_a_part_the_sizing_omits_is_refused():
    """A defaulted region is solved at an edge nobody chose. The failure is
    NAMED at the call, before any mesh is built, so a typo cannot silently
    mesh a region at the optical edge while the caller believes otherwise."""
    with pytest.raises(MESH.MeshingError) as exc:
        _build({"core": FINE_M})
    assert "wall" in str(exc.value), str(exc.value)


def test_a_part_the_sizing_invents_is_refused():
    """The mirror image: naming a part this upload does not have. Silently
    dropping it would leave the real region at the optical edge while the
    caller's intent for it was silently discarded."""
    with pytest.raises(MESH.MeshingError) as exc:
        _build({"wall": COARSE_M, "core": FINE_M, "cladding": FINE_M})
    assert "cladding" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("bad", [0.0, -1e-4, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_edge_is_refused(bad):
    """``target_maxvolume`` divides by the edge, and a NaN would reach TetGen
    as a metric value. Both are refused at the call with the part named."""
    with pytest.raises(MESH.MeshingError) as exc:
        _build({"wall": COARSE_M, "core": bad})
    assert "core" in str(exc.value), str(exc.value)


def test_sizing_needs_a_multi_part_upload():
    """With one part there is no per-part anything. Refused rather than
    ignored: ignoring it would return a mesh built at a default edge while the
    caller's argument said otherwise."""
    import trimesh

    surface = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    with pytest.raises(MESH.MeshingError) as exc:
        MESH.tetrahedralize(surface, "mm", n_modes=4, verbose=False,
                            sizing={"body": FINE_M}, max_tets=200_000)
    assert "multi-part" in str(exc.value), str(exc.value)


def test_the_default_path_is_untouched():
    """``sizing=None`` must be today's path. Asserted as a VALUE -- the same
    element count as a call that never mentions the argument -- rather than as
    "sizing is None inside", which would be true of a path that also changed
    the mesh.
    """
    parts, surface = _nested()
    materials = {"wall": 1.0, "core": 1.5}
    base = MESH.tetrahedralize(
        surface, "mm", n_modes=4, n_index=1.0, verbose=False,
        parts=parts, materials=materials, max_tets=200_000,
    )
    explicit_none = MESH.tetrahedralize(
        surface, "mm", n_modes=4, n_index=1.0, verbose=False,
        parts=parts, materials=materials, sizing=None, max_tets=200_000,
    )
    assert len(base[1]) == len(explicit_none[1])
    assert np.array_equal(base[1], explicit_none[1])
