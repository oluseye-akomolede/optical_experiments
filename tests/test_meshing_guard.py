"""The resolution guard, pinned without running TetGen.

These tests fake TetGen (a few-dozen-tet mesh is the most this VM may build),
because the bug they exist to catch is in the *loop logic*, not in the
tetrahedralization: the lead's tier-2 run hit a mesh whose achieved median edge
was 2.4x the target and the guard did not fire. The cause was
``too_coarse = enforce_resolution and achieved > allowed_edge`` -- with
``enforce_resolution`` set to its own default of ``True`` the guard looked
armed, but every *caller* passes ``enforce_resolution=False`` in the paths that
matter, so the refusal never happened.

They also pin the cap retry: tets scale as 1/maxvolume, so the loop must land
just under the cap in one jump rather than crawling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_meshing as MESH  # noqa: E402


class FakeGrid:
    """The interface ``mse_meshing`` reads: tetgen's own ``.node`` and ``.elem``.

    This fake used to expose ``points`` and ``cells_dict[VTK_TETRA]``, which is
    the ``tet.grid`` (pyvista) shape. The solver no longer builds a pyvista grid
    at all -- VTK is what the image is shedding -- so a fake still modelling
    ``cells_dict`` would be a fake of an interface nothing calls, and every test
    below would pass while proving nothing about the code that runs. The rig has
    a memory for exactly this ("a double that models the fix, not the bug").

    ``test_the_fake_models_what_the_solver_reads`` asserts the fake and the real
    ``mse_meshing`` agree on the shape, so this cannot drift back.
    """

    def __init__(self, points, tets):
        self.node = points
        self.elem = tets


#: A real Kuhn-split cube: 6 tets that tile the unit cube exactly. Every edge
#: has positive length, so ``achieved_edge_length`` returns a sensible median.
#: (An earlier version of this fixture gave all six tets the SAME vertex tuple,
#: which made every edge zero-length and the median 0 -- the guard then compared
#: 0 against the limit and accepted. The fixture has to be a mesh.)
_CUBE_POINTS = np.array(
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
     [1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]],
    dtype=float,
)
_CUBE_TETS = np.array(
    [[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 4, 5], [2, 3, 4, 6],
     [1, 3, 4, 5], [2, 4, 5, 6]],
    dtype=np.int64,
)


def _cube_median_edge_over_side() -> float:
    """``_CUBE_TETS``' median edge per unit of box side.

    A Kuhn split's 36 edges are 16 sides, 15 face diagonals and 5 body
    diagonals, so the MEDIAN edge is the face diagonal, ``sqrt(2)``, not the
    side. Derived from the arrays rather than written as 1.4142, so a change to
    the fixture's topology cannot leave a stale constant behind.
    """
    local = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    xyz = _CUBE_POINTS[_CUBE_TETS]
    lengths = np.linalg.norm(
        xyz[:, [a for a, _ in local], :] - xyz[:, [b for _, b in local], :],
        axis=2,
    )
    return float(np.median(lengths))


#: Multiply a target median edge by this to get the box side ``_points_for``
#: must use, so the fake's *measured* (median) edge is the edge it was asked
#: for. Without it every guard ratio is off by sqrt(2).
CUBE_EDGE_TO_BOX_SIDE = 1.0 / _cube_median_edge_over_side()


class FakeTetGen:
    """Stand-in for ``tetgen.TetGen`` returning a fixed, tiny real mesh.

    It records every ``maxvolume`` it was asked for, which is how the retry
    tests see the loop's decisions. The mesh is one Kuhn-divided unit cube; a
    subclass adds a ``_repeat`` count when a test needs more elements, which is
    also how the 1/maxvolume law is faked (duplicating elements scales the count
    exactly while leaving the edge statistics unchanged).
    """

    calls: list[float] = []

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int32)

    def tetrahedralize(self, order=1, mindihedral=20, minratio=2.0,
                       maxvolume=None, fixedvolume=False):
        type(self).calls.append(float(maxvolume))
        copies = self._copies_for(maxvolume)
        points = self._points_for()
        tets = np.tile(_CUBE_TETS, (copies, 1))
        # ``.node``/``.elem`` -- the interface mse_meshing reads now. Provided
        # alongside a ``grid`` so a future caller cannot silently depend on the
        # pyvista shape coming back.
        self.node = points
        self.elem = tets

    def _copies_for(self, maxvolume) -> int:
        """How many times to replicate the unit cube **as an element count**.

        Nothing here is a real tetrahedralization: duplicating elements is how a
        fake reproduces a *count* while leaving the edge statistics (and so
        ``achieved_edge_length``) alone.
        """
        return 1

    def _points_for(self) -> np.ndarray:
        """``_CUBE_POINTS`` placed in the frame THIS CALL was handed.

        A real TetGen fills the surface it is given and returns points in that
        surface's frame, so the mesh it produces is the size of its input. The
        fixed-unit-cube version this replaces could not see the difference: it
        returned a 1.0-cube whatever it was handed, so ``achieved_edge_length``
        was measured on a mesh that had nothing to do with the vertices, and the
        two sides of the guard's comparison could not be told apart.

        The cube spans the input's bounding box, so its MEDIAN edge is the
        caller's longest side times ``_cube_median_edge_over_side()`` -- the
        face diagonal, which is what a ``_CUBE_TETS`` Kuhn split actually
        measures. The box is scaled by ``CUBE_EDGE_TO_BOX_SIDE`` to put that
        median back on the caller's side, so the achieved edge a test reads is
        the edge the mesh was asked for rather than 41% more.
        """
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        side = float(np.max(hi - lo)) * CUBE_EDGE_TO_BOX_SIDE
        return _CUBE_POINTS * side + lo


class FakeSurface:
    """The minimum trimesh surface ``tetrahedralize`` touches."""

    def __init__(self, extent=1.0):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 dtype=float) * extent
        self.faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                              dtype=np.int32)
    @property
    def bounds(self) -> np.ndarray:
        """DERIVED from the vertices, never stored beside them.

        Storing it made a state no real surface can be in: ``apply_scale``
        multiplies ``vertices``, so a copy sat at one size while ``bounds``
        still described the other, and a subclass rebuilding itself from
        ``bounds`` produced a surface contradicting its own vertices.
        """
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])


@pytest.fixture(autouse=True)
def _reset_calls():
    FakeTetGen.calls = []
    yield
    FakeTetGen.calls = []


@pytest.fixture(autouse=True)
def _no_surface_decimation(monkeypatch):
    """Keep the loop tests focused on the loop.

    ``decimate_to_edge_length`` runs before tetrahedralization and would see
    ``FakeSurface`` (4 triangles, huge median edge) -- harmless, but it would
    reach for pyvista and make these tests about decimation. The decimation is
    covered properly in test_surface_decimation.py against real surfaces.
    """
    monkeypatch.setattr(MESH, "decimate_to_edge_length",
                        lambda surface, edge_length, verbose=True: (surface, None))


def _install(monkeypatch, fake_cls):
    """Make ``mse_meshing``'s lazy ``import tetgen`` resolve to our fake."""
    import types

    module = types.ModuleType("tetgen")
    module.TetGen = fake_cls
    monkeypatch.setitem(sys.modules, "tetgen", module)


# ---------------------------------------------------------------------------
# The guard fires when the mesh is too coarse.
# ---------------------------------------------------------------------------
def test_too_coarse_mesh_is_refused_not_accepted(monkeypatch):
    """A mesh 2.4x coarser than the target must refuse, not return.

    This is the lead's tier-2 shape: achieved median edge 0.40 mm against a
    0.165 mm target is 2.4x, well past MAX_EDGE_OVERSHOOT = 1.6, and it was
    accepted. It must raise.
    """

    class CoarseTetGen(FakeTetGen):
        # unit cube: every edge is 1.0, far above the 0.05 * 1.6 limit
        pass

    _install(monkeypatch, CoarseTetGen)

    surface = FakeSurface(extent=1.0)
    # target_edge_length returns the requested edge in the user's units; force a
    # target far below what the fake mesh achieves so "too coarse" is certain.
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 0.05)

    with pytest.raises(MESH.MeshingError) as excinfo:
        MESH.tetrahedralize(surface, "mm", n_modes=20, max_tets=10_000,
                            verbose=False)
    message = str(excinfo.value)
    assert "too large or detailed" in message
    assert "fewer modes" in message
    # and it must not have silently succeeded after burning every attempt
    assert len(FakeTetGen.calls) >= 1


def test_the_guard_is_on_by_default(monkeypatch):
    """The old bug: the guard was gated on ``enforce_resolution``.

    The default is ``True``, but the callers that matter passed ``False``, so
    the refusal silently did not happen and a 2.4x-coarse mesh came back as a
    result. ``enforce_resolution`` now means *only* "the bench asked to chart a
    coarse mesh deliberately": leaving it at its default must refuse, and asking
    for ``False`` must be the only way through.
    """

    class CoarseTetGen(FakeTetGen):
        pass

    _install(monkeypatch, CoarseTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 0.05)

    # default: refused
    with pytest.raises(MESH.MeshingError):
        MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20, max_tets=10_000,
                            verbose=False)

    # explicit True: refused
    with pytest.raises(MESH.MeshingError):
        MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20, max_tets=10_000,
                            verbose=False, enforce_resolution=True)

    # explicit False: the deliberate escape hatch, and it says so on stdout
    points, tets, _ = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=10_000, verbose=False,
                                       enforce_resolution=False)
    assert tets.shape[0] == len(_CUBE_TETS)


def test_a_resolvable_mesh_is_accepted(monkeypatch):
    """The guard must not fire on a mesh that meets the resolution rule."""

    class FineTetGen(FakeTetGen):
        # ten copies of the cube: a real mesh, and a deterministic count
        def _copies_for(self, maxvolume):
            return 10

    _install(monkeypatch, FineTetGen)
    # target edge 1.0 x MAX_EDGE_OVERSHOOT (1.6) stays above the achieved 1.0
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 1.0)

    points, tets, _ = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=10_000, verbose=False)
    assert tets.shape[0] == 10 * len(_CUBE_TETS)
    assert points.shape[1] == 3


# ---------------------------------------------------------------------------
# The guard is PER REGION: a region is judged against its OWN required edge.
#
# The defect these pin: the guard compared the WHOLE mesh's median edge against
# MAX_EDGE_OVERSHOOT times the WALL's edge for every upload, while ``maxvolume``
# comes from ``h_min``. A dense core can therefore be built several times coarser
# than the rule its own refractive index sets, with the wall's fine mesh hiding it
# in the median. The units that make this worst: 1.6x the wall is 5.6x the rule
# for an n=3.5 core.
# ---------------------------------------------------------------------------
class FakeTwoRegionTetGen(FakeTetGen):
    """A two-region mesh with a FINE wall and a COARSE core, labelled per region.

    Everything the mesher READS from a real ``tetgen.TetGen`` in the multi-part
    branch, and nothing else: ``.node``/``.elem`` (the base fake's contract),
    ``.attributes`` -- one region id per tet as ``(n, 1)`` float64, the shape the
    real wrapper returns -- and ``add_region``.

    Both regions reuse the ``_CUBE_TETS`` Kuhn split, so a region's MEDIAN edge
    is that cube's face diagonal times the side it is given. The mesher hands
    its vertices in the CANONICAL frame and maps the points back by
    ``1/canonical_scale``; on a 1 mm fixture ``canonical_scale`` is 1.0, so these
    sides ARE the achieved edges in millimetres and the test can DERIVE its
    expectation instead of reading the verdict out of the code under test.

    The mesh deliberately does not conform to the surface it was handed: a fake
    cannot be a tetrahedralization of an arbitrary input, and what is under test
    here is the guard's reading of ``(points, tets, attrib)``, not TetGen. The
    real, surface-conforming two-region mesh is covered by
    ``test_the_region_attribute_array_is_tet_indexed_and_names_the_regions``.
    """

    wall_side = 0.1
    core_side = 1.0
    wall_region_id = 0
    core_region_id = 1

    def add_region(self, region_id, seed):
        self.regions = getattr(self, "regions", [])
        self.regions.append((int(region_id), tuple(seed)))

    def tetrahedralize(self, order=1, mindihedral=20, minratio=2.0,
                       maxvolume=None, fixedvolume=False, regionattrib=False,
                       docheck=False, metric=False, bgmesh=None):
        # ``metric``/``bgmesh`` (the sizing-field branch's keywords) are
        # accepted and not read: grading is decided by WHICH branch calls this,
        # and the switch's effect is read as the region verdict that follows.
        type(self).calls.append(float(maxvolume))
        n = len(_CUBE_POINTS)
        self.node = np.vstack([
            _CUBE_POINTS * self.wall_side,
            _CUBE_POINTS * self.core_side + np.array([2.0, 0.0, 0.0]),
        ])
        self.elem = np.vstack([_CUBE_TETS, _CUBE_TETS + n])
        self.attributes = np.concatenate([
            np.full(len(_CUBE_TETS), float(self.wall_region_id)),
            np.full(len(_CUBE_TETS), float(self.core_region_id)),
        ]).reshape(-1, 1)


def _two_region_parts():
    """A nested wall (1 mm cube) and core (0.4 mm cube) as a real ``PartSet``.

    Real rather than faked on purpose: the plan the guard reads is derived from
    it, so a hand-built stand-in would let the fixture and the rule drift apart.
    The 1 mm wall also fixes ``canonical_scale`` at 1.0, which is what lets the
    fake's canonical sides be read as millimetres.
    """
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    core = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    parts = PARTS.load_parts(
        trimesh.Scene({"wall": wall, "core": core}).export(
            file_type="obj").encode("utf-8"),
        units="mm",
    )
    assert len(parts) == 2, "this fixture needs a two-region upload"
    return parts


def _wire_two_region(monkeypatch, core_side):
    import mse_parts as PARTS

    parts = _two_region_parts()
    names = list(parts.names)
    monkeypatch.setattr(FakeTwoRegionTetGen, "core_side", core_side)
    monkeypatch.setattr(FakeTwoRegionTetGen, "wall_region_id",
                        int(parts.region_of_part[names.index("wall")]))
    monkeypatch.setattr(FakeTwoRegionTetGen, "core_region_id",
                        int(parts.region_of_part[names.index("core")]))
    _install(monkeypatch, FakeTwoRegionTetGen)
    # One target edge for every region, so the ONLY difference between the wall's
    # allowance and the core's is the core's refractive index.
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: TARGET_MM)
    return parts, PARTS.merged_surface(parts)


#: The wall's target edge, in the upload's units. Chosen so the three numbers the
#: fixture is ABOUT are comfortably separated -- see the test docstring.
TARGET_MM = 0.6

#: The index ratio the fixture uses. n=3 makes the core's own required edge a
#: third of the wall's, so a core at the wall's allowance is 3x past its own.
CORE_INDEX = 3.0


def test_a_coarse_core_is_refused_by_its_own_rule(monkeypatch):
    """The hole: a fine wall hides a coarse core from a whole-mesh median.

    Fixture arithmetic, all derived from the fixture rather than the guard and
    reproduced by running it:

      * every edge is in mm (a 1 mm fixture, so ``canonical_scale`` is 1.0);
      * the wall's median edge is ``wall_side`` face diagonals = 0.141 mm
        (0.147x the wall's allowance, so the wall PASSES);
      * the core's is ``core_side`` face diagonals      = 1.414 mm;
      * the core's OWN edge is 0.6 / 3 = 0.2 mm, so its allowance is 0.32 mm
        -> the core is 4.42x past its own rule.

    The whole-mesh median over all 36 tet edges is 0.587 mm -- NOT the mean of
    the two per-region medians, because each region's median is itself an
    average of its own two middle edges -- and that sits INSIDE the wall's
    0.96 mm allowance at 0.61x. So the old whole-mesh test accepted this mesh.
    That is the defect, and it is why this test reds (with this docstring's own
    message) if the verdict is reverted to the global comparison.
    """
    parts, surface = _wire_two_region(monkeypatch, core_side=1.0)

    with pytest.raises(MESH.MeshingError) as excinfo:
        MESH.tetrahedralize(surface, "mm", n_modes=20, n_index=1.0,
                            max_tets=10_000, verbose=False,
                            materials={"wall": 1.0, "core": CORE_INDEX},
                            parts=parts)
    message = str(excinfo.value)
    # The refusal must NAME the region and give the user something to change.
    assert "core" in message, (
        f"the refusal does not name the failing region: {message!r}"
    )
    assert "refractive index" in message
    assert "fewer modes" in message
    # The allowance in the sentence is the constant the guard used, not a retyped
    # number, so it must agree with it.
    assert f"{MESH.MAX_EDGE_OVERSHOOT:g}x" in message
    # ...and it must NOT be the one-part refusal, which says nothing about
    # regions and would send the owner to "a smaller object", which cannot help.
    assert "too large or detailed" not in message

    # The refine arm was tried, not skipped: within the cap, a too-coarse region
    # is refined before it is refused. One attempt would mean the guard refused
    # without using the lever it has.
    assert len(FakeTetGen.calls) >= 2, (
        f"the guard refused after {len(FakeTetGen.calls)} attempt(s) without "
        f"refining -- the refine arm did not run"
    )


def test_a_fine_core_is_accepted(monkeypatch):
    """The control: the SAME two-region path must pass when the core is fine.

    Without this, "the guard names the core" would also be satisfied by a guard
    that refuses every multi-region upload. The core here is finer than its own
    allowance, so the mesh must come back with both regions tagged and no
    refusal.
    """
    parts, surface = _wire_two_region(monkeypatch, core_side=0.1)

    points, tets, region_of_tet = MESH.tetrahedralize(
        surface, "mm", n_modes=20, n_index=1.0, max_tets=10_000, verbose=False,
        materials={"wall": 1.0, "core": CORE_INDEX}, parts=parts,
    )
    assert tets.shape[0] == 2 * len(_CUBE_TETS)
    assert len(np.unique(region_of_tet)) == 2, (
        "the accepted mesh must still carry both region ids"
    )


def test_the_per_region_rule_reduces_to_the_global_one_for_one_part():
    """One region: the per-region verdict IS the whole-mesh verdict.

    The product requirement is that a one-part upload's verdict does not move.
    A one-part upload never enters the region branch (``region_plan is None``),
    so this pins the branch that DOES: with one region in the plan whose index
    equals the wall's, its own edge is the wall's edge, and the per-region verdict
    must equal the global one computed from the same points and tets. Two
    different code paths, one answer.
    """
    import mse_common as COMMON

    # A one-region plan, built by the real helper from a real PartSet of one box.
    import mse_parts as PARTS

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    one = PARTS.load_parts(
        trimesh.Scene({"wall": box}).export(file_type="obj").encode("utf-8"),
        units="mm",
    )
    # ``n_wall`` is passed explicitly: a one-part PartSet has no wall part to
    # infer it from, and the point here is the RULE, not the inference.
    plan_shared = COMMON.region_plan(one, {"wall": 1.0}, 0.6, "mm", n_wall=1.0)
    region_plan = {"shared": plan_shared, "region_of_part": one.region_of_part}
    (region_id,) = list(plan_shared)
    assert plan_shared[region_id]["own_edge_units"] == pytest.approx(0.6), (
        "a wall-only region's own edge must be the wall's edge"
    )

    # The same tets at two spacings: one too coarse for the rule, one inside it.
    # BOTH verdicts must agree, or the branch a caller takes would change the
    # answer -- an accept on one path and a refusal on the other.
    for unit_scale, label in ((1e-3, "too coarse"), (1e-4, "fine")):
        points = _CUBE_POINTS.copy()
        tets = _CUBE_TETS.copy()
        attrib = np.zeros(len(tets), dtype=np.int64)
        achieved_m = MESH.achieved_edge_length(points * unit_scale, tets)
        allowed_edge = MESH.MAX_EDGE_OVERSHOOT * 0.6 * unit_scale

        global_verdict = MESH._resolution_verdict(
            None, points, tets, None, achieved_m, allowed_edge, unit_scale)
        region_verdict = MESH._resolution_verdict(
            region_plan, points, tets, attrib, achieved_m, allowed_edge,
            unit_scale)

        assert region_verdict[0] == global_verdict[0], (
            f"the one-region and one-part verdicts disagree on a {label} mesh: "
            f"{region_verdict[0]} against {global_verdict[0]}"
        )
        assert region_verdict[1][0] == pytest.approx(global_verdict[1][0]), (
            "the one-region verdict and the one-part verdict disagree about the "
            "ratio, so a caller switching branches would change an answer"
        )
        assert region_verdict[1][3] == pytest.approx(global_verdict[1][3]), (
            "the one-region own edge is not the wall's edge"
        )


def test_the_guard_measures_in_one_space_at_scale_1000(monkeypatch):
    """The leg that was broken, and the one no existing test covered.

    Every guard test above runs at the default scale, where the old bug's two
    length spaces differed by exactly the scale factor -- 1.0 -- so the error
    cancelled and the suite stayed green over it. Here the scale is 1000, which
    is where ``allowed_edge`` used to come out in metres of the UNSCALED object
    while ``achieved`` was metres of the scaled one.

    The fixture is a surface ``scale`` units on a side, scaled once more by
    ``scale``, so the object TetGen is handed is ``scale**2`` units across.
    ``target_edge_length`` is scripted to that same side -- a length the object
    is 1:1 with, in the SCALED object's units -- so the allowance is
    ``scale**2 * unit_scale('mm') * 1.6`` metres. The double returns a mesh
    spanning the object, whose median edge is ``sqrt(2)`` times the box side,
    which lands just inside that allowance and is accepted. Under the broken
    arithmetic the allowance was computed against the object BEFORE the scale
    was applied and came out ``scale**2`` times too small, so this same mesh was
    refused.

    ``target_edge_length`` was 1.0 here before, independent of size. That passed
    only because the fake returned a fixed unit cube whatever it was handed; a
    double that returns a mesh of its input (as TetGen does) makes "1.0" mean
    "resolve a 1e6-unit object to 1 unit", which the guard correctly refuses.
    The size is scripted rather than the assertion loosened, so the test still
    pins the ratio it exists for.
    """

    class OneTetGen(FakeTetGen):
        """One copy of the cube: a real mesh, and a deterministic count."""

        def _copies_for(self, maxvolume):
            return 1

    _install(monkeypatch, OneTetGen)
    scale = 1000.0
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: scale ** 2)

    class ScaledSurface(FakeSurface):
        """``apply_scale`` copies at a non-unit scale; the stub must too."""

        def copy(self):
            return ScaledSurface(extent=self.bounds[1][0])

    points, tets, _ = MESH.tetrahedralize(ScaledSurface(extent=scale), "mm",
                                       n_modes=20, max_tets=10_000,
                                       verbose=False, scale=scale)
    assert points.shape[1] == 3

    achieved_m = MESH.LAST_ACHIEVED_EDGE_M
    allowed_m = (scale ** 2) * MESH.unit_scale("mm") * MESH.MAX_EDGE_OVERSHOOT
    # Both sides are metres of the SCALED object, which is what makes the ratio
    # ~1/1.6. Under the old arithmetic the denominator was scale**2 too small
    # and the ratio was 1e9, which is the whole defect in one number.
    ratio = achieved_m / allowed_m
    assert ratio == pytest.approx(1.0 / MESH.MAX_EDGE_OVERSHOOT, rel=1e-3), (
        f"achieved {achieved_m:g} m vs allowed {allowed_m:g} m (ratio {ratio:g})"
    )
    assert achieved_m < allowed_m, "the accept leg must actually be accepted"

    # And the guard is genuinely armed at this scale: a feature edge 4x finer
    # than the achieved mesh must be refused, so the pass above is not a guard
    # that stopped working.
    with pytest.raises(MESH.MeshingError):
        MESH.tetrahedralize(ScaledSurface(extent=scale), "mm", n_modes=20,
                            max_tets=10_000, verbose=False, scale=scale,
                            min_feature_m=0.25 * scale ** 2 * MESH.unit_scale("mm"))


def test_allow_coarse_path_still_solves(monkeypatch):
    """The bench's deliberate ``enforce_resolution=False`` escape hatch stays."""

    class CoarseTetGen(FakeTetGen):
        pass

    _install(monkeypatch, CoarseTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 0.05)

    points, tets, _ = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=10_000, verbose=False,
                                       enforce_resolution=False)
    assert tets.shape[0] == len(_CUBE_TETS)


# ---------------------------------------------------------------------------
# The cap retry lands in one jump.
# ---------------------------------------------------------------------------
def test_cap_retry_lands_within_two_passes(monkeypatch):
    """tets ~ 1/maxvolume, so one scaling step must fit the cap.

    The lead measured three passes (573k -> 139k -> 46.7k, 24.7s). Since the
    count is inversely proportional to ``maxvolume``, one jump of
    ``(n_tets / cap) * 1.15`` should land just under the cap.
    """
    cap = 1_000

    class ScalingTetGen(FakeTetGen):
        # n_tets = 1000 / maxvolume, i.e. exactly the 1/maxvolume law
        def _copies_for(self, maxvolume):
            return max(1, int(round(1000.0 / max(float(maxvolume), 1e-9))))

    _install(monkeypatch, ScalingTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 1.0)
    # start far over the cap: maxvolume 1.0 -> 1000 tets would be at the cap, so
    # ask for a tiny maxvolume to get ~100_000 tets
    monkeypatch.setattr(MESH, "target_maxvolume", lambda *a, **k: 0.01)

    points, tets, _ = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=cap, verbose=False)
    assert tets.shape[0] <= cap
    # the first call is the requested volume, the second is the jump: two passes
    assert len(FakeTetGen.calls) == 2, FakeTetGen.calls


def test_cap_retry_gives_up_after_max_attempts(monkeypatch):
    """A mesh that never fits the cap must be refused, with the count in words."""

    class StubbornTetGen(FakeTetGen):
        def _copies_for(self, maxvolume):
            return 10_000          # ignores maxvolume entirely

    _install(monkeypatch, StubbornTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 1.0)

    with pytest.raises(MESH.MeshingError) as excinfo:
        MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20, max_tets=100,
                            verbose=False)
    assert "limit" in str(excinfo.value)


def test_an_unresponsive_count_stops_the_retry_immediately(monkeypatch):
    """Raising maxvolume without moving the count means the SURFACE is the limit.

    This is what the lead saw on the Jetson sphere: 212,000 -> 196,569 ->
    202,931 across two maxvolume triplings, ~200k every time because TetGen
    cannot make interior elements much larger than a facet. Spending three more
    tetrahedralizations to learn that is wasted time, so the loop must stop at
    the first pass that does not respond and say why.
    """

    class StuckTetGen(FakeTetGen):
        # A count that ignores maxvolume entirely: the surface is the floor.
        def _copies_for(self, maxvolume):
            return 20_000

    _install(monkeypatch, StuckTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 1.0)

    with pytest.raises(MESH.MeshingError) as excinfo:
        MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20, max_tets=100,
                            verbose=False)

    message = str(excinfo.value)
    # the message must name the real cause: a too-fine surface, not just "too big"
    assert "surface is too finely triangulated" in message
    # and it must have stopped rather than burning every attempt
    assert len(FakeTetGen.calls) == 2, (
        f"expected the pass plus one unresponsive retry, got "
        f"{len(FakeTetGen.calls)} passes: {FakeTetGen.calls}"
    )


def test_a_responsive_retry_still_gets_its_passes(monkeypatch):
    """A count that DOES respond must be allowed to keep retrying.

    The early stop must not fire on a normal cap retry, where each pass really
    does reduce the count.
    """
    cap = 1_000

    class RespondingTetGen(FakeTetGen):
        # n_tets = 1000/maxvolume * 100: over the cap, but responsive
        def _copies_for(self, maxvolume):
            return max(1, int(round(100.0 / max(float(maxvolume), 1e-9))))

    _install(monkeypatch, RespondingTetGen)
    monkeypatch.setattr(MESH, "target_edge_length", lambda *a, **k: 1.0)
    monkeypatch.setattr(MESH, "target_maxvolume", lambda *a, **k: 0.001)

    points, tets, _ = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=cap, verbose=False)
    assert tets.shape[0] <= cap
    # It iterated (rather than stopping after an unresponsive first pass) and
    # landed well inside MAX_MESH_ATTEMPTS -- one jump plus a correction.
    assert 2 <= len(FakeTetGen.calls) <= MESH.MAX_MESH_ATTEMPTS


# ---------------------------------------------------------------------------
# Resolution sizing: Weyl's law, not a linear mode count.
# ---------------------------------------------------------------------------
def test_box_target_edge_is_in_the_expected_band():
    """On the 10x7x5 box at 20 modes the target edge must land in 0.5-0.8 mm.

    The band is what the tier-2 bench proved is enough: 0.40 mm gave 0.06% on 20
    modes. The old rule (k_high = n_modes * pi / diagonal) asked for 0.165 mm,
    about 4x too fine, which is what made the mesher blow the element cap on
    ordinary uploads.
    """
    import trimesh

    surface, _ = MESH.load_surface(
        trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode(),
        units="mm",
    )
    edge = MESH.target_edge_length(surface, n_modes=20, units="mm")
    assert 0.5 <= edge <= 0.8, edge


def test_target_edge_follows_the_cube_root_of_the_mode_count():
    """Weyl's law: N ~ V k^3, so edge length goes as n_modes^(-1/3).

    Doubling the requested mode count must shrink the target edge to about
    1/2^(1/3) = 0.794 of its value -- not to half of it, which is what a linear
    rule gives.
    """
    import trimesh

    surface, _ = MESH.load_surface(
        trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode(),
        units="mm",
    )
    e10 = MESH.target_edge_length(surface, n_modes=10, units="mm")
    e20 = MESH.target_edge_length(surface, n_modes=20, units="mm")
    assert e10 / e20 == pytest.approx(2 ** (1 / 3), rel=0.02)


def test_target_edge_uses_the_enclosed_volume_not_the_bounding_box():
    """A hollow object has far less enclosed volume than its bbox, so it needs a
    coarser mesh -- using the bbox would over-refine it."""
    import trimesh

    outer = trimesh.creation.box(extents=[10, 7, 5])
    # a shell: a small box subtracted from the big one, still closed
    inner = trimesh.creation.box(extents=[9, 6, 4])
    try:
        shell = trimesh.boolean.difference([outer, inner])
    except Exception:  # noqa: BLE001 - no boolean backend available
        pytest.skip("trimesh has no boolean backend in this environment")
    surface, _ = MESH.load_surface(shell.export(file_type="obj").encode(),
                                   units="mm")
    volume = MESH.enclosure_volume_m3(surface, "mm")
    bbox_volume = 10 * 7 * 5 * 1e-9
    assert volume < bbox_volume / 2, (volume, bbox_volume)
    # less enclosed volume -> higher k_N -> finer edge, and the bbox would have
    # given the coarser one
    assert MESH.target_edge_length(surface, n_modes=20, units="mm") < \
        MESH.target_edge_length(
            MESH.load_surface(outer.export(file_type="obj").encode(),
                              units="mm")[0],
            n_modes=20, units="mm")


def test_target_edge_does_not_depend_on_the_refractive_index():
    """The geometric k_N is a property of the cavity, not of what fills it."""
    import trimesh

    surface, _ = MESH.load_surface(
        trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode(),
        units="mm",
    )
    assert MESH.target_edge_length(surface, 20, "mm", n_index=1.0) == \
        MESH.target_edge_length(surface, 20, "mm", n_index=1.5)


def test_target_edge_unit_change_cancels_out_of_the_returned_number():
    """The returned edge length is invariantly the same number of user units.

    Dimensional analysis: the scale factor S enters the volume as S^3, so k_N
    goes as 1/S; the wavelength goes as S, and dividing by S to express it back
    in the caller's unit leaves S^0. So "the same numbers, read as cm instead of
    mm" is a ten-times-larger physical object whose target edge is ten times
    larger in metres -- and it comes back as the same count of cm.
    """
    import trimesh

    obj = trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode()
    in_mm, _ = MESH.load_surface(obj, units="mm")
    in_cm, _ = MESH.load_surface(obj, units="cm")

    edge_mm = MESH.target_edge_length(in_mm, 20, "mm")           # in mm
    edge_cm = MESH.target_edge_length(in_cm, 20, "cm")           # in cm

    assert edge_cm == pytest.approx(edge_mm, rel=1e-9)
    # the physical sizes really are different, so the equality above is the
    # cancellation rather than the unit being ignored entirely
    assert edge_cm * 1e-2 == pytest.approx(10 * edge_mm * 1e-3, rel=1e-9)


# ---------------------------------------------------------------------------
# The canonical frame: TetGen meshes a UNIT-INDEPENDENT object
#
# TetGen's Steiner insertion is sensitive to the absolute coordinate magnitude,
# so the same physical object spelled in mm, m and um used to mesh to different
# element counts (measured 35237..47323 tets, 34%, for one icosphere). The
# surface now goes to TetGen in a frame whose scale is fixed by the OBJECT --
# centred on its bounding box, divided by its longest side -- and the points are
# mapped back before returning. These three legs are the row's own tests.
# ---------------------------------------------------------------------------

def _frame_of(monkeypatch, surface, units, **kw):
    """Run one ``tetrahedralize`` and return what TetGen was actually handed.

    The fake records the vertices it was constructed with, so this reads the
    frame the product chose rather than recomputing it: a test that re-derived
    ``(v - centre) / longest`` would only be checking its own arithmetic.
    """
    seen = {}

    class Recording(FakeTetGen):
        def __init__(self, vertices, faces):
            super().__init__(vertices, faces)
            seen["vertices"] = np.asarray(vertices, dtype=float).copy()

        def _copies_for(self, maxvolume):
            seen["maxvolume"] = float(maxvolume)
            return 1

    _install(monkeypatch, Recording)
    # These legs are about the FRAME, not about resolution quality, and the
    # fake's mesh is object-sized rather than physics-edge-sized. The module's
    # own ``enforce_resolution=False`` is the documented lever for running the
    # tetrahedralization without the accuracy gate -- the same one the bench's
    # ``--allow-coarse`` path uses.
    kw.setdefault("enforce_resolution", False)
    points, tets, _ = MESH.tetrahedralize(surface, units, n_modes=20,
                                       max_tets=10_000, verbose=False, **kw)
    return seen, points, tets


def test_the_canonical_frame_is_bounded_and_centred(monkeypatch):
    """Whatever the spelling, TetGen is handed an O(1) object at the origin.

    This is the mechanism, asserted directly: the frame's longest side is 1 and
    its centre is the origin. Without it the magnitude sensitivity is back and
    the counts drift with the unit label.
    """
    for units, scale in (("mm", 1000.0), ("m", 1.0), ("um", 1e6)):
        surface = trimesh.creation.box(extents=[10, 7, 5])
        seen, _, _ = _frame_of(monkeypatch, surface, units, scale=scale)
        v = seen["vertices"]
        longest = float(np.max(v.max(axis=0) - v.min(axis=0)))
        centre = (v.max(axis=0) + v.min(axis=0)) / 2.0
        assert longest == pytest.approx(1.0, rel=1e-12), (
            f"{units}: longest side of the frame TetGen got was {longest}, "
            "so this spelling is meshed at its own magnitude"
        )
        assert np.allclose(centre, 0.0, atol=1e-12), (
            f"{units}: the frame was not centred on the object's bbox"
        )


def test_the_three_spellings_give_tetgen_the_identical_object(monkeypatch):
    """INVARIANCE -- the row's first requirement, at the seam that matters.

    One physical object written three ways must reach TetGen as the SAME
    vertices, because TetGen's count is a function of those bytes. Byte identity
    is the strongest possible form of "the same element count": if the vertices
    agree, no float sensitivity can make the counts differ.
    """
    frames = []
    for units, scale in (("mm", 1000.0), ("m", 1.0), ("um", 1e6)):
        surface = trimesh.creation.box(extents=[10, 7, 5])
        seen, _, _ = _frame_of(monkeypatch, surface, units, scale=scale)
        frames.append(seen["vertices"])

    base = frames[0]
    for units, v in zip(("m", "um"), frames[1:]):
        assert np.allclose(v, base, rtol=1e-12, atol=1e-15), (
            f"{units} reached TetGen in a different frame than mm; max "
            f"difference {np.abs(v - base).max():g}"
        )


def test_a_five_nanometre_object_meshes(monkeypatch):
    """The plasmonics leg: a 5e-8 m object must produce a mesh, not a failure.

    Coordinates this small used to sit near TetGen's tolerance; the canonical
    frame puts every object at O(1) whatever its size, so the same call is
    numerically ordinary. The assertion is that a mesh comes back with the
    object's shape -- a count alone would pass for a frame that produced
    garbage.
    """
    surface = trimesh.creation.box(extents=[50.0, 30.0, 20.0])   # in metres
    seen, points, tets = _frame_of(monkeypatch, surface, "m", scale=1.0)

    assert tets.shape[0] > 0, "a 5e-8 m object produced no elements"
    assert points.shape[1] == 3
    # The returned points are in metres of the SCALED object, so a 50 m box
    # spelled in metres comes back 50 m across -- the mapping is the exact
    # inverse, which is what stops the frame leaking into reported lengths.
    span = points.max(axis=0) - points.min(axis=0)
    assert np.all(span > 0), f"degenerate mesh span {span}"
    # ...and the frame TetGen saw was still the O(1) one.
    v = seen["vertices"]
    assert float(np.max(v.max(axis=0) - v.min(axis=0))) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The two facts the tet.grid -> tet.node/tet.elem swap rests on
#
# Both are silent when wrong, which is why they are pinned against the REAL
# tetgen rather than argued in a comment. The rig's recurring defect is a check
# that cannot fail -- a fake agreeing with itself -- so these run real meshes and
# compare the new read against the old one that is being removed.
# ---------------------------------------------------------------------------

def _real_tetgen_available() -> bool:
    try:
        import tetgen  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001 - absence is a fact to report, not an error
        return False
    return True


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
def test_tetrahedralize_returns_zero_based_in_range_indices():
    """Drive the REAL ``tetrahedralize`` and check the indices it hands back.

    This replaces a test that built its own ``tetgen.TetGen`` and compared
    ``tet.elem`` against ``cells_dict[VTK_TETRA]``. That test could not fail on
    the code it was written to protect: it never called ``tetrahedralize``, so
    it pinned *tetgen's* behaviour, and mutating the solver's read to ``elem - 1``
    left it green. It was a check that could not see the defect. (Measured: the
    mutation passed it.)

    The property that actually matters is the one the solver depends on -- every
    index it returns is a valid row of the points it returns alongside. A
    ``- 1`` shifts the minimum below zero; a ``+ 1`` runs past the end; either
    is caught here, and neither is caught by comparing two raw tetgen arrays.
    """
    surface = trimesh.creation.box(extents=[4.0, 6.0, 3.0])   # metres
    points, tets, _ = MESH.tetrahedralize(surface, "m", n_modes=4, max_tets=5000,
                                       verbose=False, enforce_resolution=False)

    assert tets.ndim == 2 and tets.shape[1] == 4, tets.shape
    assert tets.shape[0] > 1, "need more than one element to mean anything"
    n_points = np.asarray(points).shape[0]
    assert int(tets.min()) == 0, (
        f"lowest index is {int(tets.min())}, not 0 -- the read is off by a "
        f"constant (a 1-based read gives -1 here)"
    )
    assert int(tets.max()) == n_points - 1, (
        f"highest index {int(tets.max())} != n_points-1 ({n_points - 1}); the "
        f"indices do not address the points returned with them"
    )


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
def test_tetrahedralize_preserves_the_axis_order_of_the_surface():
    """Drive the REAL ``tetrahedralize`` and check the axes are not swapped.

    Same lesson as the index test above: a test that builds its own ``TetGen``
    pins tetgen, not this module. This one goes through ``tetrahedralize`` and
    asserts the returned points carry the SURFACE's own extents in the same
    order -- so reading ``node[:, [0, 2, 1]]`` (a y/z swap, which every edge
    length survives) fails here.

    A box with three distinct extents is the fixture: a transposed read cannot
    reproduce them, and an (x,y,z)-preserving read must.
    """
    extents = [4.0, 6.0, 3.0]     # metres, all different
    surface = trimesh.creation.box(extents=extents)
    points, _, _ = MESH.tetrahedralize(surface, "m", n_modes=4, max_tets=5000,
                                    verbose=False, enforce_resolution=False)

    span = np.asarray(points).max(axis=0) - np.asarray(points).min(axis=0)
    # The box is centred on the origin and lies on the axes, so the mesh span
    # must reproduce the extents in the SAME order. Rounded: TetGen's hull is
    # the box to within its own tolerance.
    assert np.allclose(span, extents, rtol=1e-6, atol=1e-9), (
        f"mesh spans {span} do not match the box extents {extents} in order -- "
        f"the coordinate columns are not x, y, z (a swap looks plausible on "
        f"every edge length and every element count)"
    )


def test_the_fake_models_what_the_solver_reads():
    """The FakeGrid must expose ``.node``/``.elem``, the interface actually read.

    A fake of an interface nothing calls makes every test above vacuous: it can
    only fail on a change to the fake. This assertion is what keeps the fake and
    ``mse_meshing`` from drifting apart again -- it is deliberately a test about
    the DOUBLE, because the double is where this class of bug hides.
    """
    fake = FakeGrid(np.zeros((4, 3)), np.zeros((1, 4), dtype=np.int64))
    assert hasattr(fake, "node"), "the solver reads tet.node; the fake must too"
    assert hasattr(fake, "elem"), "the solver reads tet.elem; the fake must too"
    assert np.asarray(fake.elem).shape[1] == 4, "tetrahedra are 4-node cells"
    # And the source really is read through those names -- not through a
    # ``grid`` that happens to still exist.
    #
    # COMMENTS ARE STRIPPED FIRST, and that is not tidiness. The first version
    # of this check searched the raw source and failed on its own explanatory
    # comment, which mentions the old attribute by name to say why it is gone.
    # A probe that cannot tell prose from code reports a defect in a comment --
    # the same "the instrument, not the world" shape this whole file guards
    # against. Strip first, then ask.
    import inspect

    src = inspect.getsource(MESH.tetrahedralize)
    code = _strip_comments_and_docstrings(src)
    assert ".elem" in code and ".node" in code, (
        "mse_meshing no longer reads .node/.elem; this fake and the whole block "
        "of tests built on it would be modelling nothing"
    )
    assert "cells_dict" not in code, (
        f"mse_meshing still reads cells_dict in CODE (comment mentions are "
        f"fine and expected); the pyvista grid was supposed to be gone.\n{code}"
    )


def _strip_comments_and_docstrings(src: str) -> str:
    """Source with comments and docstrings removed, leaving executable code.

    ``tokenize`` rather than a regex: a ``#`` inside a string literal is not a
    comment, and a naive strip would miscount those. ``AST`` is the other
    candidate but it discards nothing -- the tokens are what we want to read.
    """
    import io
    import tokenize

    kept: list[tokenize.TokenInfo] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            kept.append(tok)
    except tokenize.TokenError:
        # An unparsable fragment means the probe cannot speak -- say so by
        # returning it unchanged rather than silently reporting "no defect".
        return src
    # ``untokenize``, NOT a " ".join of token strings: joining bare tokens puts
    # spaces inside ``tet.elem`` (it lexes as NAME . NAME) so the substring is
    # gone and every assert below reads as a missing attribute. An earlier
    # version of this helper did exactly that and reported a defect that was
    # not there -- the join was the bug, not the source.
    return tokenize.untokenize(kept)


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
def test_the_region_attribute_array_is_tet_indexed_and_names_the_regions():
    """Drive the REAL ``tetrahedralize`` and check the region ids it hands back.

    The third return value is what decides every tet's PERMITTIVITY, so an
    array that is the wrong length, in the wrong order, or off by one produces a
    plausible-looking solve for a cavity the owner did not upload. That is the
    defect class this whole feature is about, and it is invisible downstream:
    the eigenproblem stays well-posed and the eigenvalues barely move.

    Two independent things are pinned, and both must be measured rather than
    taken from tetgen's docs -- the docs' own example shows a differently
    shaped return, which is exactly why ``.attrib`` (a name that does not
    exist) survived in the source until a solve reached it:

      1. the array has one row PER TET, in tet order, and
      2. its values are the REGION IDS: a tet strictly inside the core region
         carries the core's id, and no tet strictly inside the core carries the
         wall's. The geometric control is independent of the array.

    The interface is handled with an explicit tolerance, not ``<=``: a tet with
    three vertices ON the interface plane is CORRECTLY conforming, and an
    inclusive test reads it as a straddler. An earlier version of this check
    made exactly that mistake and reported 251 straddlers on a clean mesh.
    """
    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    import mse_parts as PARTS

    parts = PARTS.load_parts(trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8"), units="mm")
    assert len(parts) == 2, "this test needs a two-region fixture"

    points, tets, region_of_tet = MESH.tetrahedralize(
        PARTS.merged_surface(parts), "mm", n_modes=3, n_index=1.0,
        max_tets=60_000, verbose=False, materials={"wall": 1.0, "core": 2.0},
        parts=parts,
    )

    # 1. One region id per tet, addressing the same array the tets do.
    assert region_of_tet.shape == (tets.shape[0],), (
        f"{region_of_tet.shape[0]} region ids for {tets.shape[0]} tets -- the "
        f"array does not address the tetrahedra returned with it"
    )
    # Both regions must appear. A single-valued array is the "quietly reverted
    # to one material" failure, and it is the one that looks most like success.
    ids = np.unique(region_of_tet)
    assert len(ids) == 2, (
        f"the mesh reported region ids {ids}; a two-part upload must produce "
        f"two regions, not one uniformly-filled cavity"
    )

    # 2. The ids are MEANINGFUL -- checked against the geometry, not the array.
    #
    # ``points`` comes back in METRES while the parts are in MILLIMETRES, so the
    # core's half-extent is 1e-3 here, not 1. Comparing the metre centroids
    # against 1.0 selected EVERY tet in the 6 mm wall and made this test report
    # that one region carried two ids -- an instrument error that read exactly
    # like a mesher bug. The conversion is done once, on the centroid, from the
    # SAME unit_scale the mesher used rather than a remembered 1e-3.
    import mse_common as COMMON

    core_half_extent = 1.0 * COMMON.unit_scale("mm")       # 2 mm cube -> 1e-3 m
    centroids = np.asarray(points)[np.asarray(tets)].mean(axis=1)
    # Strictly inside the core, with a tolerance so a centroid exactly on the
    # interface plane is neither in nor out.
    in_core = (np.abs(centroids) < core_half_extent - 1e-12).all(axis=1)
    assert in_core.sum() > 0, (
        "no tet centroid is strictly inside the core; the fixture does not "
        "exercise the nested region, so this test could not see a mis-tagging"
    )
    core_tets = region_of_tet[in_core]
    # The core's region id is whichever one the deep-inside tets carry; take it
    # from the geometry rather than assuming a numbering tetgen does not promise.
    assert len(np.unique(core_tets)) == 1, (
        f"tets strictly inside the core carry ids {np.unique(core_tets)}; one "
        f"region cannot have two ids"
    )
    core_id = int(np.unique(core_tets)[0])
    wall_tets = region_of_tet[~in_core]
    assert core_id not in np.unique(wall_tets), (
        f"region id {core_id} is carried by tets both inside and outside the "
        f"core -- the ids are not regions"
    )

    # ...and the PARTITION must be the geometric one, not merely a partition.
    # Everything above is invariant under relabelling, and that is not a
    # hypothetical: shifting every id by one -- exactly what the ``.ele`` file's
    # 1-based convention suggests, and what an ``elem - 1``-style read does --
    # still leaves the core carrying a single id and the wall another, so none
    # of it can see a shift. Measured on this fixture: with the read mutated to
    # ``... - 1`` the ids become [-1, 0] and every check above stays green.
    #
    # The fix is to stop reading the expected value OUT of the array under
    # test. An earlier draft took the core's id from ``rot[in_core]`` and then
    # asserted ``rot == that`` -- which is ``A == A`` with extra steps, and it
    # passed the mutation for exactly that reason. The id comes from the MESH
    # instead: the region the mesher was asked to put the core in, read from the
    # same plan that placed it there.
    from mse_parts import PartSet  # noqa: F401,PLC0415  (documents the type)

    core_index = list(parts.names).index("core")
    expected_core_id = int(parts.region_of_part[core_index])
    assert expected_core_id != int(parts.region_of_part[
        list(parts.names).index("wall")]), (
        "the fixture's two parts were given the same region id; the upload "
        "would be one material and this test could not tell them apart"
    )
    assert np.array_equal(region_of_tet == expected_core_id, in_core), (
        f"{int((region_of_tet == expected_core_id).sum())} tet(s) carry the "
        f"core's region id {expected_core_id} against {int(in_core.sum())} "
        f"whose centroid is strictly inside the core -- the region map does not "
        f"agree with the geometry it was built from"
    )


# ---------------------------------------------------------------------------
# The sizing field: the switches it may share a call with, and the grid it is
# handed. Both are silent when wrong -- a global cap un-grades every region,
# and a mis-shaped background grid either raises inside tetgen or, worse,
# loads something that grades nothing.
# ---------------------------------------------------------------------------
# The graded path is the PRODUCT RULE since ``graded-meshing-default``
# (``MSE_GRADED_MESHING`` unset means ON). The tests below pin it through the
# module flag rather than the environment, so they read the same setting however
# the suite is invoked -- and, being explicit, they still pass a deploy that
# rolls grading back with ``MSE_GRADED_MESHING=0``.
@pytest.fixture
def graded_meshing_on(monkeypatch):
    monkeypatch.setattr(MESH, "GRADED_MESHING", True)


def _graded_plan(parts, materials, edge_m, units, verbose=False):
    """``_region_plan`` with the graded overlay, from a base edge in METRES.

    Main's ``_region_plan`` takes the edge in the SCALED SURFACE's units (the
    live contract); the WIP's took metres. This converts once, here, with the
    product's own unit table, so the tests keep the WIP's metre fixtures.
    """
    import mse_common as COMMON
    edge_units = float(edge_m) / COMMON.UNIT_SCALE[units]
    return MESH._region_plan(parts, materials, edge_units, units, verbose,
                             graded=True)


def test_the_graded_switches_refuse_a_global_cap_and_need_the_metric_flag():
    """The guard runs on the switch dict the mesh is BUILT with, not a copy.

    ``_sizing_field_switches`` is the one place a graded call's switches live,
    and ``tetrahedralize`` calls ``_check_sizing_field_switches`` on that dictionary
    immediately before the call. Two measured negatives are encoded:

      * ``fixedvolume=True`` or a POSITIVE ``maxvolume`` caps the whole mesh at
        the coarse target and silently loses the fine region -- grading returns
        a uniform mesh with no error;
      * ``metric=False`` makes tetgen read the background mesh's third array as
        the wrong quantity, so the field grades nothing.

    Each is asserted as a RAISE, and the real switch set is asserted to pass,
    so a later edit that re-adds one of them fails here.
    """
    good = MESH._sizing_field_switches()
    # The good set must pass its own guard; otherwise the guard is a rule
    # nothing satisfies and the graded arm is dead code.
    assert MESH._check_sizing_field_switches(good) is None

    with pytest.raises(MESH.MeshingError, match="fixedvolume"):
        MESH._check_sizing_field_switches({**good, "fixedvolume": True})

    with pytest.raises(MESH.MeshingError, match="maxvolume"):
        MESH._check_sizing_field_switches({**good, "maxvolume": 1e-9})

    with pytest.raises(MESH.MeshingError, match="metric"):
        MESH._check_sizing_field_switches({**good, "metric": False})

    # The REAL dict is the one that must be clean: a guard proven only on
    # synthetic dicts says nothing about the call the mesher makes.
    assert float(good.get("maxvolume", 0.0)) == 0.0, (
        "a graded mesh must not carry a positive global maxvolume"
    )
    assert not good.get("fixedvolume"), (
        "a graded mesh must not make a global maxvolume binding"
    )
    assert good.get("metric") and good.get("regionattrib"), (
        "a graded mesh needs metric=True (the field carries LENGTHS) and "
        "regionattrib=True (the per-region achieved check reads the tags)"
    )


def test_the_background_grid_is_the_shape_tetgens_public_route_reads():
    """``_sizing_background_grid`` is fed to ``tetrahedralize(bgmesh=...)``.

    tetgen's public route takes a ``pyvista.UnstructuredGrid`` of linear
    tetrahedra whose point data has ``target_size``, then reads exactly three
    arrays out of it. Four things about that hand-off are silent when wrong, so
    each is pinned:

      1. the point-data KEY is tetgen's own (``MTR_POINTDATA_KEY``);
      2. every cell is a TETRA (a mixed grid is refused by tetgen's own check);
      3. the cell array is VTK's prefix form, four ids per tet -- an off-by-one
         here loads a background mesh of the wrong shape rather than failing;
      4. a real ``TetGen`` accepts the grid the PRODUCT builds, which is the
         claim that matters.

    (4) is the one that cannot be faked. The background arrays are the ones
    ``_sizing_background`` returns for a real two-region upload, and the same
    grid topology is driven with two DIFFERENT per-node size arrays -- coarse
    then fine -- so the element count must move. A grid tetgen could not read
    would raise or ignore the field and the two counts would match.

    A hand-built background mesh (a bare box tetrahedralization with no region
    attributes) SEGFAULTS tetgen's field lookup rather than raising, so this
    test deliberately uses the product's own background: it is the only
    background shape the mesher ever hands over, and reproducing it by hand is
    both more code and a worse fixture.
    """
    import tetgen
    import tetgen.pytetgen as TTG

    assert MESH.TETGEN_MTR_POINTDATA_KEY == TTG.MTR_POINTDATA_KEY, (
        "the point-data key the mesher writes is not the one tetgen looks for; "
        "tetgen would raise, but only on the graded path in production"
    )

    # --- the shape of the grid, on a tiny hand-made one --------------------
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    sizes = np.array([0.5, 0.5, 0.5, 0.5])
    grid = MESH._sizing_background_grid(pts, tets, sizes)

    assert TTG.MTR_POINTDATA_KEY in grid.point_data, (
        f"the background grid carries {list(grid.point_data.keys())}; tetgen "
        f"reads {TTG.MTR_POINTDATA_KEY!r} and nothing else"
    )
    assert np.array_equal(grid.point_data[TTG.MTR_POINTDATA_KEY], sizes)
    assert (grid.celltypes == grid.celltypes.max()).all() and int(
        grid.celltypes[0]) == 10, (
        "every background cell must be a linear tetrahedron (VTK type 10)"
    )
    assert grid.n_cells == 1 and grid.n_points == 4
    assert np.allclose(grid.points, pts)
    # The cells are VTK's PREFIX form: 4 ids per tet, not a flat 3-per-triangle
    # style array. Read back through the same route tetgen's reader uses.
    assert np.asarray(grid.cells[:5]).tolist() == [4, 0, 1, 2, 3], (
        f"the cell array is not VTK's prefix form: {grid.cells[:5].tolist()}"
    )

    # --- (4) a REAL call, on the product's own background -------------------
    # A coarse-ish plan keeps the background small (8.6k cells); the fixture is
    # the nested 6 mm / 2 mm pair, the object the whole feature was measured on.
    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    import mse_parts as PARTS
    parts = PARTS.load_parts(obj, units="mm")
    merged = PARTS.merged_surface(PARTS.load_parts(obj, units="mm"))
    materials = {"wall": 1.0, "core": 2.0}
    plan = _graded_plan(parts, materials, 1.8e-3, "mm", verbose=False)

    verts = np.asarray(merged.vertices, dtype=np.float64)
    faces = np.asarray(merged.faces, dtype=np.int32)
    # The mesher tetrahedralizes in a canonical (O(1)) frame; the background
    # must be built in the frame it will be looked up in, so do the same.
    offset = (verts.min(0) + verts.max(0)) / 2.0
    scale = 1.0 / float(np.max(verts.max(0) - verts.min(0)))
    canonical_verts = (verts - offset) * scale
    bg_pts, bg_tets, bg_sizes = MESH._sizing_background(
        plan, canonical_verts, faces, offset, scale, "mm")
    assert bg_tets.shape[0] > 0

    def count_with_sizes(size_array):
        g = MESH._sizing_background_grid(bg_pts, bg_tets, size_array)
        t2 = tetgen.TetGen(canonical_verts, faces)
        for region_id, seed in MESH._region_requests(plan, offset, scale):
            t2.add_region(region_id, tuple(seed))
        t2.tetrahedralize(**MESH._sizing_field_switches(), bgmesh=g)
        return int(np.asarray(t2.elem).shape[0])

    coarse_edge = max(plan["edge_m"].values()) / 1e-3 * scale
    fine_edge = min(plan["edge_m"].values()) / 1e-3 * scale
    coarse = count_with_sizes(np.full(bg_pts.shape[0], coarse_edge))
    fine = count_with_sizes(np.full(bg_pts.shape[0], fine_edge))
    assert fine > coarse * 2, (
        f"the same background grid with a {coarse_edge * 1e3 / scale:.3f} mm "
        f"node size gave {coarse:,} tets and with "
        f"{fine_edge * 1e3 / scale:.3f} mm {fine:,}; the field is not being "
        f"honoured, so the grid is the wrong shape even though tetgen accepted it"
    )
    # And the product's OWN field (per-node sizes from the regions) lands
    # between the two, which is what "graded" means numerically.
    graded = count_with_sizes(bg_sizes)
    assert coarse <= graded <= fine, (
        f"the graded field gave {graded:,} tets, outside the uniform bracket "
        f"[{coarse:,}, {fine:,}] -- it is not a graded mesh at all"
    )


def _thin_shell_parts(units="mm"):
    """Three nested solids whose MIDDLE region is a thin slab.

    The clamp needs a region that is not the finest and whose thickness is
    under its own ramp edge -- the only arrangement where it can bite. The
    finest region is floored at ``h_min`` by construction, so a two-region
    fixture can never show it.
    """
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    slab = trimesh.creation.box(extents=(0.8, 0.8, 0.3))
    core = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    obj = trimesh.Scene(
        {"wall": wall, "slab": slab, "core": core}
    ).export(file_type="obj").encode("utf-8")
    return PARTS.load_parts(obj, units=units)


def test_the_uniform_call_is_exactly_the_call_the_mesher_always_made(monkeypatch):
    """Byte-identity, proved on the CALL rather than on a count.

    A single-material upload must mesh exactly as it did before per-region
    grading existed. Two meshers could agree on a count by luck, so the thing
    that actually makes the bytes identical is that ``tetrahedralize`` is handed
    the same switch set: for a one-part job, precisely the five keywords it
    always got, with no ``bgmesh``, no ``metric`` and no region attribute. The
    real TetGen is wrapped to record what it was called with, so this reads the
    product's own call rather than a retyped copy of it -- and it is the failure
    mode that matters: a stray ``metric=True`` or an empty ``bgmesh`` would not
    raise, it would quietly rebuild every single-material mesh a different way.

    The second half does the same for a MULTI-part job whose parts share one
    index: it has regions, so it does not take the one-part call, but its edges
    are equal, so it must not take the sizing field either -- the region
    attribute and ``docheck`` ride along, and nothing else does.
    """
    import tetgen as TET

    recorded: list[dict] = []
    real = TET.TetGen

    class Recorder(real):
        def tetrahedralize(self, **kwargs):
            recorded.append(kwargs)
            return super().tetrahedralize(**kwargs)

    monkeypatch.setattr(TET, "TetGen", Recorder)

    # (a) A single material, no parts: the one-part call.
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    _pts, one_tets, _reg = MESH.tetrahedralize(
        box, "mm", n_modes=3, n_index=1.0, max_tets=200_000,
        min_feature_m=0.35e-3, verbose=False,
        materials={"body": 1.0}, parts=None)
    assert len(recorded) == 1, f"the one-part mesh tetrahedralized {len(recorded)}x"
    call = recorded[0]
    assert call == {"order": 1, "mindihedral": 20, "minratio": 2.0,
                    "maxvolume": call["maxvolume"], "fixedvolume": True}, (
        f"the one-part call set changed: {sorted(call)} -- a single-material "
        "mesh is no longer byte-identical to the pre-feature one"
    )
    assert call["maxvolume"] > 0, (
        "the one-part call must still pass a POSITIVE maxvolume; without it "
        "and without fixedvolume TetGen meshes unconstrained"
    )
    # And the same call must still produce the SAME mesh: the switch set alone
    # proves the graded path is not reached, not that the output survived it.
    # Measured on this fixture (a 1 mm box, min_feature_m = 0.35 mm, 3 modes)
    # with TetGen 0.8.4 through the real wrapper, deterministic for one input
    # and one switch set. A stray metric=True or a bgmesh of any size moves it.
    assert int(one_tets.shape[0]) == 1257, (
        f"the one-part mesh is {int(one_tets.shape[0])} tets, not the 1257 "
        "measured before this feature -- a single-material mesh changed"
    )
    # And no region report: a one-part job has no regions to grade, and the
    # manifest's `mesh` block is written only from this report, so a
    # single-material modes.json is the file it was before the feature.
    assert MESH.LAST_MESH_REPORT is None, (
        "a one-part mesh must leave no mesh report; otherwise every "
        "single-material job's modes.json gains a `mesh` block"
    )

    # (b) Two parts, ONE shared index: regions, but one edge, so still uniform.
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    parts = PARTS.load_parts(obj, units="mm")
    merged = PARTS.merged_surface(parts)
    recorded.clear()
    MESH.tetrahedralize(merged, "mm", n_modes=3, n_index=1.0, max_tets=200_000,
                        min_feature_m=0.9e-3, verbose=False,
                        materials={"wall": 2.0, "core": 2.0}, parts=parts)
    assert MESH.LAST_MESH_REPORT is not None, "a multi-part mesh must be reported"
    assert MESH.LAST_MESH_REPORT["graded"] is False, (
        "equal indices must NOT grade -- there is one edge to grade"
    )
    assert len(recorded) == 1, f"the equal-index mesh tetrahedralized {len(recorded)}x"
    call = recorded[0]
    assert call == {"order": 1, "mindihedral": 20, "minratio": 2.0,
                    "maxvolume": call["maxvolume"], "fixedvolume": True,
                    "regionattrib": True, "docheck": True}, (
        f"the equal-index call set changed: {sorted(call)} -- the graded "
        "sizing field must not be reachable when the edges are equal"
    )


@pytest.mark.usefixtures("graded_meshing_on")
def test_a_coarse_target_is_clamped_under_its_own_thickness():
    """The clamp, on the fixture where it can fire.

    Measured on this fixture: the slab's thickness proxy is 0.171 mm and its
    ramp edge at n=1 is 0.5 mm, so ``min(ramp, thickness)`` is the thickness --
    the region is held FINER than its material index asks for. That is the
    direction that matters, and it is not the obvious one: the clamp exists
    because a target that APPROACHES the thickness makes TetGen's sizing field
    collapse and drop the fine region, so the target is pulled UNDER it.

    Three things are asserted, and the second is the one the acceptance rule
    depends on:

      * the clamp bites on the slab (edge < unclamped) and is reported;
      * the FINEST region is never moved: its edge is its own ramp, unclamped,
        so the job's accuracy does not move with this feature;
      * the thickness proxy on a slab equals the analytic 2*V/A.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                              / "dockerfiles" / "maxwell_solver"))
    import mse_common as COMMON

    parts = _thin_shell_parts()
    h_base_m = 0.5e-3            # 0.5 mm in metres, the caller's coarse edge
    materials = {"wall": 1.0, "slab": 1.0, "core": 4.0}
    # The ONE helper, called the way the mesher calls it...
    raw = COMMON.graded_region_edges(h_base_m, 1.0, materials, parts, "mm")
    # ...and the mesher's own view of it (names live here, not in the helper).
    plan = _graded_plan(parts, materials, h_base_m, "mm", verbose=False)
    by_part = {info["part"]: rid for rid, info in plan["shared"].items()}
    slab = by_part["slab"]
    core = by_part["core"]
    assert [round(raw[r]["edge"], 15) for r in sorted(raw)] == [
        round(plan["edge_m"][r], 15) for r in sorted(raw)
    ], "the mesher's plan must be the helper's own output, not a second opinion"

    # The analytic proxy for a slab: 2*V/A is the thickness itself when the
    # boundary is two big faces and four edges.
    slab_index = [i for i, info in plan["shared"].items() if info["part"] == "slab"][0]
    slab_area = COMMON.region_boundary_area(parts, slab_index)
    slab_fill_m3 = parts.filled_volume_m3(slab_index, "mm")
    expected_thickness_m = 2.0 * slab_fill_m3 / (slab_area * 1e-6)
    assert plan["thickness_m"][slab] == pytest.approx(expected_thickness_m, rel=1e-9)

    assert raw[slab]["unclamped"] > raw[slab]["thickness_m"], (
        "the fixture must have a ramp edge COARSER than the slab thickness, or "
        "the clamp cannot bite and this test proves nothing"
    )
    assert raw[slab]["edge"] == pytest.approx(raw[slab]["thickness_m"], rel=1e-12), (
        f"the slab's edge should be clamped to its thickness "
        f"{raw[slab]['thickness_m'] * 1e3:.4f} mm; got "
        f"{raw[slab]['edge'] * 1e3:.4f} mm"
    )
    assert raw[slab]["edge"] < raw[slab]["unclamped"], (
        "the clamp must make the region FINER than its index asks, never coarser"
    )
    assert plan["clamped"][slab] is True, (
        "the clamp bit on the slab but the mesher's plan did not record it, so "
        "no job caveat can be written"
    )

    # The finest region: its edge IS its ramp, and its ramp IS h_min. Nothing
    # about this feature may move it.
    h_min = min(info["edge"] for info in raw.values())
    assert raw[core]["edge"] == pytest.approx(h_min, rel=1e-12)
    assert raw[core]["edge"] == pytest.approx(raw[core]["unclamped"], rel=1e-12), (
        "the finest region must be untouched by the clamp, or accuracy moves"
    )
    assert plan["clamped"].get(core, False) is False, (
        "the finest region must not be clamped, or the acceptance rule is "
        "holding the job to an edge nobody asked for"
    )


@pytest.mark.usefixtures("graded_meshing_on")
def test_the_clamp_and_the_grading_are_stated_in_the_job_caveats():
    """A clamped or graded job says so in its ``mesh`` block.

    A clamp that is silent is a resolution the owner never asked for and
    cannot see; the row asked for it to be stated when it bites. The control
    is the second half: a plan with no clamp and no grading must carry NO
    caveat, or the caveat is decoration.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                              / "dockerfiles" / "maxwell_solver"))
    import mse_common as COMMON

    parts = _thin_shell_parts()
    materials = {"wall": 1.0, "slab": 1.0, "core": 4.0}
    edge_m = 0.5e-3
    plan = _graded_plan(parts, materials, edge_m, "mm", verbose=False)

    n_tets = int(_CUBE_TETS.shape[0])
    tets = _CUBE_TETS.copy()
    points_m = _CUBE_POINTS * 1e-3
    # Every region gets at least one tet, so no region reports ``None``.
    region_of_tet = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)[:n_tets]

    report = MESH._mesh_report(
        plan, region_of_tet, points_m, tets, "mm", achieved=0.4e-3)

    assert report["graded"] is True, "this fixture must take the graded path"
    text = " || ".join(report["caveats"])
    assert "GRADED" in text, f"the grading is not stated: {report['caveats']}"
    assert "FINER than" in text and "thickness" in text, (
        f"the clamp bit on this fixture but no caveat states it: "
        f"{report['caveats']}"
    )
    # The clamp caveat names the REGION that was clamped, not just the fact.
    clamped_regions = [rid for rid, bit in plan["clamped"].items() if bit]
    assert clamped_regions, "this fixture must have clamped a region"
    for rid in clamped_regions:
        assert f"region {rid}" in text, (
            f"no caveat names region {rid}, the one the clamp moved"
        )

    # CONTROL: the same report for a plan with no clamping and no grading must
    # carry no caveat at all.
    plain_plan = {
        **plan,
        "graded": False,
        "clamped": {rid: False for rid in plan["clamped"]},
        "unclamped_m": {rid: plan["edge_m"][rid] for rid in plan["edge_m"]},
    }
    plain = MESH._mesh_report(
        plain_plan, region_of_tet, points_m, tets, "mm", achieved=0.4e-3)
    assert plain["caveats"] == [], (
        f"a uniform, unclamped mesh must carry no caveat; got "
        f"{plain['caveats']}"
    )


def test_a_single_material_job_never_takes_the_graded_path(monkeypatch):
    """The byte-identity control: single-material meshes cannot grade.

    The acceptance criterion for per-region grading is that a single-material
    upload meshes EXACTLY as it did before the feature existed. That is not
    provable by comparing counts in general (two meshers could agree by luck);
    it is provable by showing the graded path is never ENTERED, because the
    graded arm passes a different switch set to TetGen. So both grading
    entry points are made to explode, and the single-material job must still
    come back with a mesh.

    This is the control the row asked for: it fails if the graded path is taken.
    """
    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "a single-material job entered the graded (sizing-field) path: the "
            "mesh is no longer byte-identical to the pre-feature one"
        )

    monkeypatch.setattr(MESH, "_sizing_background_grid", _boom)
    monkeypatch.setattr(MESH, "_sizing_background", _boom)
    monkeypatch.setattr(MESH, "_region_requests", _boom, raising=False)

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # A single material, and NO parts: exactly the shape a one-material upload
    # arrives in.
    points, tets, region_of_tet = MESH.tetrahedralize(
        box, "mm", n_modes=3, n_index=1.0, max_tets=200_000,
        min_feature_m=0.35e-3, verbose=False,
        materials={"body": 1.0}, parts=None,
    )
    assert int(tets.shape[0]) > 0, "the single-material job built no mesh"

    # And the report the job page reads says plainly that nothing was graded.
    report = MESH.LAST_MESH_REPORT
    assert report is None or report["graded"] is False, (
        "a single-material job must not report itself as graded"
    )
    assert not report or report["caveats"] == [], (
        f"a single-material job routes through the ungraded call and has no "
        f"graded caveat to state; got {report['caveats']}"
    )


def _fine_nested_parts(units="mm"):
    """Two nested boxes, each faceted to its OWN region edge.

    The graded path can only be exercised with facets fine enough for the
    fine region's interior: TetGen cannot refine a surface below its facets,
    so a 12-triangle box would floor the core and the mesh would be refused as
    too coarse (a different failure, and the one that would hide what these
    tests are about). The wall is faceted at the wall's edge and the core at
    the core's, the shape a real upload of a graded object arrives in.
    """
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    wall = wall.subdivide_to_size(max_edge=1.2)
    core = core.subdivide_to_size(max_edge=0.4)
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    return PARTS.load_parts(obj, units=units)


def _mildly_graded_parts(units="mm"):
    """``_fine_nested_parts`` graded by less than 2x: the one regime the
    background floor binds in.

    ``bg_edge = min(0.5 * coarse, fine)``, so the floored coarsening
    ``min(widened, fine)`` can only bite once the pre-floor widening overshoots
    the FINE edge -- which needs ``0.5 * coarse < fine``, a grading under 2x. At
    3x (``_fine_nested_parts``) the fine edge is never the binding term, which is
    why the floor's mutant survived the suite. Same shape: each box faceted to
    its own region's edge.
    """
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    wall = wall.subdivide_to_size(max_edge=1.2)
    core = core.subdivide_to_size(max_edge=0.8)
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    return PARTS.load_parts(obj, units=units)


def _record_tetgen(monkeypatch) -> list[dict]:
    """Every call the mesher makes to TetGen, in order.

    Wrapping the class's method rather than patching a private helper means the
    count is of TETRAHEDRALIZATIONS, whatever the code between them did. That is
    the quantity the row's question is about: a retry is a second full
    tetrahedralization with a global ``maxvolume``, and a global maxvolume
    OVERRIDES the sizing field, so one that happened would silently drop the
    grading for that job.
    """
    import tetgen as TET

    recorded: list[dict] = []
    real = TET.TetGen

    class Recorder(real):
        def tetrahedralize(self, **kwargs):
            recorded.append(kwargs)
            return super().tetrahedralize(**kwargs)

    monkeypatch.setattr(TET, "TetGen", Recorder)
    return recorded


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
@pytest.mark.usefixtures("graded_meshing_on")
def test_a_graded_mesh_over_the_cap_refuses_instead_of_retrying(monkeypatch):
    """C: the graded path cannot reach the global-``maxvolume`` retry.

    ``maxvolume`` is what the RETRY raises, and a raised ``maxvolume`` overrides
    the sizing field -- so a graded mesh that retried would come back with every
    region at ONE edge and nothing in the mesh would say so. Measured here on
    the calls: a graded over-cap request tetrahedralizes exactly TWICE (one
    sizing background, one mesh, the second carrying ``bgmesh`` and no positive
    ``maxvolume``) and raises the graded refusal. It never reaches the retry.

    The control is the SECOND half, and it is what makes the first half mean
    anything: the same object with one shared index takes the uniform arm, which
    IS allowed to retry -- so the same instrument must see two or more meshes
    there. A count of one on the graded side proves nothing unless the
    instrument can see a retry when one exists.
    """
    import mse_parts as PARTS

    parts = _fine_nested_parts()
    merged = PARTS.merged_surface(parts)

    recorded = _record_tetgen(monkeypatch)
    with pytest.raises(MESH.MeshingError) as caught:
        MESH.tetrahedralize(
            merged, "mm", n_modes=3, n_index=1.0, max_tets=1,
            min_feature_m=1.2e-3, verbose=False,
            materials={"wall": 1.0, "core": 3.0}, parts=parts,
        )
    graded_calls = list(recorded)
    message = str(caught.value)
    assert "graded object would need about" in message, (
        f"an over-cap graded mesh must raise the GRADED refusal (the fine "
        f"region's own acceptance is the only thing a graded mesh may fail on); "
        f"got: {message}"
    )
    assert len(graded_calls) == 2, (
        f"a graded mesh tetrahedralized {len(graded_calls)}x; it must be two "
        f"(the sizing background and the mesh) and never a third -- a retry "
        f"would raise a global maxvolume and drop the grading"
    )
    assert sum(1 for c in graded_calls if "bgmesh" in c) == 1, (
        "exactly one of the two calls must be the graded one (it carries the "
        "sizing field); the other is the background build"
    )
    graded_call = next(c for c in graded_calls if "bgmesh" in c)
    assert graded_call.get("maxvolume", 0.0) in (0.0, None), (
        f"the graded call must carry NO global maxvolume (got "
        f"{graded_call.get('maxvolume')}): a global maxvolume overrides the "
        f"sizing field and silently ungrades the mesh"
    )

    # THE CONTROL: one shared index -> the uniform arm, which retries.
    recorded.clear()
    with pytest.raises(MESH.MeshingError):
        MESH.tetrahedralize(
            merged, "mm", n_modes=3, n_index=1.0, max_tets=1,
            min_feature_m=1.2e-3, verbose=False,
            materials={"wall": 1.0, "core": 1.0}, parts=parts,
        )
    assert len(recorded) >= 2, (
        f"the uniform over-cap arm retried {len(recorded) - 1} time(s); this "
        f"control exists to prove the counter can SEE a retry, so it must be "
        f"at least two meshes -- got {len(recorded)}"
    )
    assert len({c.get("maxvolume") for c in recorded}) > 1, (
        "the uniform retry must actually change the global maxvolume"
    )


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
@pytest.mark.usefixtures("graded_meshing_on")
def test_a_graded_mesh_under_the_cap_comes_back_graded(monkeypatch):
    """C, the other half: the mesh that DOES come back is still graded.

    Refusing over the cap is only half the answer -- the row also asks whether
    the graded path can hand back an ungraded mesh. This reads the per-region
    achieved edges EACH REGION'S OWN TETS, which is the only place a lost
    grading is visible: a whole-mesh median is dominated by the wall and would
    look the same either way.
    """
    import mse_parts as PARTS

    parts = _fine_nested_parts()
    merged = PARTS.merged_surface(parts)
    points_m, tets, region_of_tet = MESH.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=600_000,
        min_feature_m=1.2e-3, verbose=False,
        materials={"wall": 1.0, "core": 3.0}, parts=parts,
    )
    report = MESH.LAST_MESH_REPORT
    assert report is not None and report["graded"] is True
    regions = report["regions"]
    assert len(regions) == 2, f"expected two regions, got {sorted(regions)}"

    (fine_id, fine), = [(rid, r) for rid, r in regions.items() if r["n"] == 3.0]
    (coarse_id, coarse), = [(rid, r) for rid, r in regions.items() if r["n"] == 1.0]
    # The two regions are asked for measurably different edges (3x by index,
    # pulled in by the thickness clamp for the wall, so this is an inequality
    # rather than the ratio of the indices).
    assert coarse["target_edge_m"] > 2.0 * fine["target_edge_m"], (
        f"the two regions were asked for "
        f"{coarse['target_edge_m'] * 1e3:.4g} and "
        f"{fine['target_edge_m'] * 1e3:.4g} mm: this fixture is not graded"
    )
    assert fine["tets"] and coarse["tets"], "a region came back with no tets"

    # Each region's OWN tets: the fine one is at its own (fine) edge, the coarse
    # one is measurably coarser -- which is what "still graded" means.
    assert fine["achieved_edge_m"] <= MESH.MAX_EDGE_OVERSHOOT * fine["target_edge_m"], (
        f"the fine region's own tets are at {fine['achieved_edge_m'] * 1e3:.4f} "
        f"mm against its {fine['target_edge_m'] * 1e3:.4f} mm target "
        f"({fine['achieved_edge_m'] / fine['target_edge_m']:.2f}x, allowance "
        f"{MESH.MAX_EDGE_OVERSHOOT:.2f}x)"
    )
    assert coarse["achieved_edge_m"] > fine["achieved_edge_m"] * 1.2, (
        f"the coarse region ({coarse['achieved_edge_m'] * 1e3:.4f} mm) is not "
        f"measurably coarser than the fine one "
        f"({fine['achieved_edge_m'] * 1e3:.4f} mm): the mesh came back UNGRADED "
        f"even though the report says otherwise"
    )
    # And the report's top-level fine measurement is the fine region's own.
    assert report["fine_achieved_edge_m"] == pytest.approx(
        fine["achieved_edge_m"], rel=1e-12)


def _fine_nested_parts_thin_wall(units="mm"):
    """``_fine_nested_parts`` with the core grown so the WALL is thin.

    The wall is the 1 mm shell between the 6 mm and the 4 mm box: thin enough
    that the thickness clamp bites, which is the fixture the estimate's
    correction exists for -- the sizing field's transition refines a thin wall
    hardest, and the plain sum prices the wall at its own target. Facets follow
    the same rule as its sibling (each part at its OWN region edge), because
    TetGen cannot refine a surface below its facets and a coarse one would
    refuse the mesh for a reason that is not what the test is about.
    """
    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    wall = wall.subdivide_to_size(max_edge=0.8)
    core = core.subdivide_to_size(max_edge=0.2667)
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    return PARTS.load_parts(obj, units=units)


def test_a_refusal_names_the_part_it_is_about():
    """The never-silent refusals must NAME the region, and must not crash.

    ``plan["region_of_part"]`` is a LIST in part order (``mse_parts.py``), and an
    earlier version of both refusal sites read ``.items()`` off it, so the
    refusal raised ``AttributeError`` instead of the ``MeshingError`` it exists
    for -- the caller's ``except MeshingError`` never sees it, and the user gets
    a traceback instead of "try a coarser target for the core". The plan handed
    in here is the MESHER's own (``_region_plan``), not a hand-built dict, so
    what this reads is the shape the refusal actually meets.
    """
    import mse_parts as PARTS

    parts = _fine_nested_parts()
    plan = _graded_plan(parts, {"wall": 1.0, "core": 3.0}, 1.2e-3, "mm",
                             verbose=False)
    assert isinstance(plan["region_of_part"], list), (
        "the plan's region_of_part is not the list mse_parts builds; the "
        "refusals' lookup and this test both assume part-order"
    )
    fine = plan["fine_region"]
    coarse = next(rid for rid in plan["edge_m"] if rid != fine)
    assert MESH._part_of_region(plan, fine) == "core"
    assert MESH._part_of_region(plan, coarse) == "wall"
    # A region the plan cannot name falls back to the id -- never to a raise.
    assert MESH._part_of_region({"shared": {}}, 7) == "region 7"


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
@pytest.mark.usefixtures("graded_meshing_on")
def test_a_background_over_the_cap_refuses_before_building_it(monkeypatch):
    """A: an object too big for one job is refused BEFORE the build.

    The sizing background is the one mesh built before the object exists, so a
    job that dies building it has already burnt the owner's time and, on the
    Orin, RAM the solver needed. The cap is priced from the PLC's own volume
    before the build; when even the coarsest grid that still carries the fine
    region would not fit, the call refuses with the region named in plain words.
    The recorder is the assertion: "refused" and "refused after paying for a
    build" are the same exception, and only the calls tell them apart.

    The control is the same call with the estimate left alone: it builds and
    comes back graded, so the guard fired on the price and not on the fixture.
    """
    import mse_parts as PARTS

    parts = _fine_nested_parts()
    merged = PARTS.merged_surface(parts)
    call = dict(n_modes=3, n_index=1.0, max_tets=600_000,
                min_feature_m=1.2e-3, verbose=False,
                materials={"wall": 1.0, "core": 3.0}, parts=parts)

    recorded = _record_tetgen(monkeypatch)
    monkeypatch.setattr(MESH, "_background_tet_estimate", lambda *a, **k: 1.0e9)
    with pytest.raises(MESH.MeshingError) as caught:
        MESH.tetrahedralize(merged, "mm", **call)
    message = str(caught.value)
    assert "core" in message and "cap" in message, (
        f"the over-cap refusal must name the fine region and the limit it "
        f"broke; got: {message}"
    )
    assert recorded == [], (
        f"the over-cap refusal tetrahedralized {len(recorded)}x; it must refuse "
        f"on the PRICE, before the build -- refusing after the build has already "
        f"paid for the background it refuses"
    )

    # THE CONTROL: the real estimate, the same call -- it builds and it grades.
    monkeypatch.undo()   # also undoes graded_meshing_on, so opt in by name
    _pts, tets, _reg = MESH.tetrahedralize(merged, "mm", graded=True, **call)
    assert MESH.LAST_MESH_REPORT["graded"] is True
    assert len(tets) > 0, "the control build came back empty"


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
@pytest.mark.usefixtures("graded_meshing_on")
def test_the_graded_estimate_lands_on_the_mesh_the_mesher_builds():
    """B: the estimate that ROUTES a graded job, against the mesh that follows.

    The plain per-region sum prices a coarse region at its own target, but the
    sizing field ramps from the fine edge across every boundary, so the mesher
    builds more elements than that sum says -- and the sum is what picks the
    machine. Measured here on a thick wall and a thin one: the corrected
    estimate must land INSIDE the band on both, and the control below shows the
    uncorrected form leaving it on the thick wall, so the correction is
    load-bearing rather than decoration.

    The built count is TetGen's OWN element array from the mesher's graded
    call and the estimate is the overlay's graded estimator, so this compares
    the two things that must agree -- not two copies of one formula. (The WIP
    tested ``web.meshing.estimate_tets`` here; on main that entry prices the
    UNIFORM product mesh and grading is opt-in, so routing a graded job by this
    estimate is part of the NO-GO step 3.) An under-count is
    the unsafe direction: it routes the job onto a machine it does not fit.

    Measured on the port (est/built, uncorrected in brackets): thick wall 0.85x
    (0.47x), thin wall 0.69x (0.61x). The WIP measured 1.04x (0.57x) and 0.85x
    (0.74x) with the coefficient it then had, ``web.meshing.EST_TETS_C = 14``;
    main's estimator (1cca12b) prices with the volume law, ``EST_TETS_MARGIN *
    TETS_PER_UNIT_VOLUME`` = 1.35 * 6 sqrt(2) = 11.46, which is 0.818x of 14 --
    exactly the shift on all four numbers -- so the overlay prices on main's
    coefficient and the band is the WIP's (0.70, 1.35) moved by that factor.
    BAND is wide enough for TetGen-version
    variation in the built count and narrow enough to fail a 3x over-correction
    on either fixture; the thick wall's uncorrected 0.47x is what makes the
    floor load-bearing. The thin wall is the model's tightest point (0.69x), and
    it is also where the limit shows: the same index ratio leaves the coarse
    region a different share of the count in each fixture, and one exponent
    cannot hold both -- ``TRANSITION_EDGE_EXPONENT`` carries the wider measured
    table and states that as the model's accuracy limit.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                              / "dockerfiles" / "maxwell_solver"))
    import meshing as WEB
    import mse_common as COMMON

    BAND = (0.55, 1.10)     # the WIP's (0.70, 1.35) x 0.818; see the docstring
    fixtures = [
        ("thick wall", _fine_nested_parts(), {"wall": 1.0, "core": 3.0}, 1.2e-3),
        ("thin wall", _fine_nested_parts_thin_wall(),
         {"wall": 1.0, "core": 3.0}, 0.8e-3),
    ]
    lines, plain_ratios = [], []
    for label, parts, materials, min_feature_m in fixtures:
        import mse_parts as PARTS

        merged = PARTS.merged_surface(parts)
        # The routing entry (``web.meshing.estimate_tets``) now prices the
        # GRADED mesh for a multi-part upload, so for these graded fixtures it
        # is this same ``estimate_tets_graded`` call -- see
        # ``test_grading_moves_the_estimate_with_the_mesh`` in
        # test_estimate_families. This test calls the estimator directly because
        # it needs ``min_feature_m`` pinned per fixture.
        h_base = COMMON.effective_edge_m(
            merged, n_modes=2, units="mm", min_feature_m=min_feature_m,
            parts=parts, materials=materials)
        margin = WEB.EST_TETS_MARGIN
        est = COMMON.estimate_tets_graded(parts, materials, h_base, merged,
                                          "mm", margin)
        _pts, tets, _reg = MESH.tetrahedralize(
            merged, "mm", n_modes=2, n_index=1.0, max_tets=600_000,
            min_feature_m=min_feature_m, verbose=False,
            materials=materials, parts=parts,
        )
        built = int(len(tets))
        assert built > 0, f"{label}: the mesh came back empty"
        ratio = est / built

        # THE CONTROL: the SAME plan, priced with no transition correction.
        plan = COMMON.graded_region_plan(parts, materials, h_base, "mm")
        plain = int(margin * COMMON.TETS_PER_UNIT_VOLUME * sum(
            info["filled_volume_m3"] / (info["edge_m"] ** 3)
            for info in plan.values()))
        plain_ratios.append(plain / built)
        lines.append(
            f"{label}: est {est:,} vs built {built:,} = {ratio:.2f}x "
            f"(uncorrected {plain:,} = {plain / built:.2f}x)")

        assert BAND[0] <= ratio <= BAND[1], (
            f"{label}: the routing estimate is {ratio:.2f}x the mesh TetGen "
            f"builds, outside the stated band {BAND}. " + "; ".join(lines)
        )

    # The control IS the revert: ``est`` above is the routing entry with the
    # exponent in place, and ``plain`` is the same plan with ``h_eff = h_r`` --
    # what the code computes when the exponent is 1, i.e. what it would compute
    # if the correction were removed. An under-count here is a mutation the
    # band assertion above catches, so this names the mechanism, not a copy of
    # the formula.
    assert min(plain_ratios) < BAND[0], (
        "the uncorrected sum stays inside the band on every fixture, so this "
        "test cannot tell the correction from its absence: " + "; ".join(lines)
    )


@pytest.mark.skipif(not _real_tetgen_available(),
                    reason="tetgen not installed in this environment")
@pytest.mark.usefixtures("graded_meshing_on")
def test_the_background_coarsening_is_floored_at_the_fine_regions_edge(monkeypatch):
    """B: the coarsening floor is load-bearing, on the one plan where it bites.

    Over the cap the sizing background's cells are widened to bring the volume
    price down -- but never past the FINEST region's own edge: a background cell
    wider than the region it must resolve leaves that region with no node of its
    own, and the mesh comes back quietly ungraded. The line is a ``min`` against
    the fine edge, and the mutant that dropped it left this suite green: no
    fixture drove the price over the cap while the pre-floor widening overshot
    the fine edge AND the price at the floored edge stayed legal.

    What is measured is the maxvolume TetGen was HANDED for the background build
    (``_record_tetgen``), converted back with the same regular-tet volume
    relation the code uses -- the edge the code CHOSE, not a re-derivation of the
    formula. The price is supplied here (to the code under test it is a black
    box, the way TetGen is); the arms make the instrument visible: a price under
    the cap builds at the WANTED edge, the overshooting price builds at the FINE
    edge, and the edge the floor refused -- the unfloored widening, 3.3x past --
    is a value the recorded build must never show.
    """
    import mse_common as COMMON
    import mse_parts as PARTS

    parts = _mildly_graded_parts()
    merged = PARTS.merged_surface(parts)
    call = dict(n_modes=3, n_index=1.0, max_tets=600_000,
                min_feature_m=1.2e-3, verbose=False,
                materials={"wall": 1.0, "core": 1.5}, parts=parts)
    CAP = float(MESH.MAX_BACKGROUND_TETS)

    # The plan's own canonical edges, captured from the call the code makes --
    # not recomputed here, so this arithmetic cannot drift from the code's.
    seen: dict = {}
    real_bg = MESH._sizing_background

    def spy(region_plan, vertices, faces, canonical_offset, canonical_scale,
            units, verbose):
        edge_c = {int(r): float(h) / COMMON.unit_scale(units) * canonical_scale
                  for r, h in region_plan["edge_m"].items()}
        seen["coarse"] = max(edge_c.values())
        seen["fine"] = min(edge_c.values())
        return real_bg(region_plan, vertices, faces, canonical_offset,
                       canonical_scale, units, verbose)

    monkeypatch.setattr(MESH, "_sizing_background", spy)

    def volume(edge):
        # the code's own regular-tet cell volume
        return edge ** 3 / (6.0 * np.sqrt(2.0))

    def bg_built(recorded):
        """The background build -- the only one with no sizing field -- as the
        cell edge its maxvolume stands for."""
        return [c for c in recorded if "bgmesh" not in c]

    # ---- arm 1, the control: a price UNDER the cap -> no coarsening -------
    recorded = _record_tetgen(monkeypatch)
    monkeypatch.setattr(MESH, "_background_tet_estimate",
                        lambda v, f, vol: CAP / 10.0)
    MESH.tetrahedralize(merged, "mm", **call)
    wanted = min(0.5 * seen["coarse"], seen["fine"])   # the pre-price edge
    fine = seen["fine"]
    ctrl = bg_built(recorded)
    assert len(ctrl) == 1, f"expected one background build, got {len(ctrl)}"
    assert (float(ctrl[0]["maxvolume"]) * 6.0 * np.sqrt(2.0)) ** (1 / 3) == \
        pytest.approx(wanted, rel=1e-9), (
            "with the price under the cap nothing is widened: the control build "
            "must land on the wanted edge, or the priced arm proves nothing")
    assert wanted < fine * (1.0 - 1e-9), (
        "this fixture must be graded under 2x, or the floor never binds: "
        f"wanted {wanted:.6g} vs fine {fine:.6g} canonical")

    # ---- arm 2, priced: over the cap, with the floored edge legal ---------
    # E(V) = (cap/2) * (V / V_fine) ** -6 -- an edge dependence steeper than
    # 1/volume, so the price at the FLOORED edge is legal (the call builds)
    # while the unfloored widening still overshoots it.
    recorded.clear()
    monkeypatch.setattr(
        MESH, "_background_tet_estimate",
        lambda v, f, vol: (CAP / 2.0) * (vol / volume(seen["fine"])) ** -6)
    MESH.tetrahedralize(merged, "mm", **call)

    e_wanted = (CAP / 2.0) * (volume(wanted) / volume(fine)) ** -6
    unfloored = wanted * (e_wanted / CAP) ** (1.0 / 3.0)
    priced = bg_built(recorded)
    assert priced, (
        "the priced arm must BUILD -- the price at the floored edge is legal; a "
        "refusal here means the fixture is not the one the floor protects")
    assert e_wanted > CAP, "the branch must run, or the floor is not exercised"
    assert unfloored > fine * 1.5, (
        "the unfloored widening must overshoot the fine edge by a margin the "
        f"recording can see: unfloored {unfloored:.6g} vs fine {fine:.6g}")
    built = (float(priced[0]["maxvolume"]) * 6.0 * np.sqrt(2.0)) ** (1 / 3)
    assert built == pytest.approx(fine, rel=1e-9), (
        "over the cap the cells were widened PAST the fine region's own edge -- "
        f"the floor did not hold: built {built:.6g} vs fine {fine:.6g} (the "
        f"unfloored edge the floor refused would be {unfloored:.6g})")
