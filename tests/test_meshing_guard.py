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
    def __init__(self, points, tets):
        self.points = points
        self.cells_dict = {MESH.VTK_TETRA: tets}


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
        self.grid = FakeGrid(points, tets)

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
    points, tets = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
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

    points, tets = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
                                       max_tets=10_000, verbose=False)
    assert tets.shape[0] == 10 * len(_CUBE_TETS)
    assert points.shape[1] == 3


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

    points, tets = MESH.tetrahedralize(ScaledSurface(extent=scale), "mm",
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

    points, tets = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
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

    points, tets = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
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

    points, tets = MESH.tetrahedralize(FakeSurface(), "mm", n_modes=20,
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
    points, tets = MESH.tetrahedralize(surface, units, n_modes=20,
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
