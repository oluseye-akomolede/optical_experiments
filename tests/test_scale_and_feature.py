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

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))
sys.path.insert(0, str(ROOT / "web"))

import mse_maxwell_solver as MESH_SOLVER  # noqa: E402
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


# ---------------------------------------------------------------------------
# F2: the element cap is ONE number, and the solver really uses it
# ---------------------------------------------------------------------------
#
# The defect: ``web/config.py`` capped uploads at 600_000 while the solver's own
# ``DEFAULT_MAX_TETS`` was 80_000, and nothing ever passed the app's value to the
# solver. So the app accepted an upload, told the owner it was fine, and the
# solver refused it at 80k -- failing precisely the large jobs the GPU exists
# for. The lead's benchmarks never hit it because they passed ``--max-tets``
# explicitly, which is why it survived this long.

def test_the_solver_cap_default_matches_the_web_apps_ceiling():
    """The app IMPORTS this, so the two cannot drift apart again.

    A test asserting ``600_000 == 600_000`` would pass against two copies that
    agree today; that is how the bug appeared. The real claim is that there is
    one definition, and it is checked from both sides: the constant's value, and
    the fact that ``web/config.py`` gets it by import.
    """
    import config as CFG

    assert MESH_SOLVER.DEFAULT_MAX_TETS == 600_000
    assert MESH_SOLVER.DEFAULT_CPU_MAX_TETS == 150_000
    assert CFG.DEFAULT_MAX_TETS is MESH_SOLVER.DEFAULT_MAX_TETS, (
        "the app must import the solver's constant, not restate its value"
    )


def test_the_mesh_cap_is_a_limit_not_a_target(monkeypatch):
    """The two ways a mesh can come out over the cap, and what the code says.

    This pins the contract the worker's ``solve()`` call rests on. There were
    THREE different gates in this path and a first version of this test drove
    the wrong one:

    * ``mse_meshing.tetrahedralize`` COARSENS to honour ``max_tets`` and
      returns happily -- measured: a 10 mm box that builds 6 elements at
      ``max_tets=1``. So "it returns" is NOT evidence the limit was respected,
      which is why the assertion below is on the returned COUNT.
    * When coarsening cannot help because the SURFACE is setting the element
      size, it refuses with a message naming the cap (``above the N limit``).
      That is the message production owners actually saw, and it is the one the
      app's 600_000 / solver's 80_000 disagreement turned into a lie.
    * ``solve()`` re-checks the exact count at ``mse_maxwell_solver.py:255-260``
      and refuses above ``max_tets``.

    Asserting only "something raised" would have passed on the broken code: the
    old ``solve()`` guard DID raise, it just used a number the app never sent.
    What is asserted here is the number the caller passed.
    """
    from test_meshing_guard import FakeSurface, FakeTetGen, _install

    class CapFlat(FakeTetGen):
        """A count that does NOT respond to ``maxvolume``, by fiat.

        The physical case the refusal describes is a surface whose facets floor
        the interior size, so no amount of coarsening shrinks the count. Stating
        it as a flag on the class rather than deriving it from the surface is
        deliberate: ``tetrahedralize`` passes ``np.asarray(surface.vertices)``
        to TetGen, so the surface object never reaches the fake and a
        surface-carried count silently did nothing.
        """

        #: how many elements to report, or ``None`` for the real 1/maxvolume law
        flat_copies: int | None = None

        def _copies_for(self, maxvolume) -> int:
            if type(self).flat_copies is not None:
                return type(self).flat_copies
            return super()._copies_for(maxvolume)

    _install(monkeypatch, CapFlat)
    monkeypatch.setattr(MESH, "decimate_to_edge_length",
                        lambda surface, edge_length, verbose=True: (surface, None))

    def mesh(copies: int | None, max_tets: int):
        CapFlat.calls = []
        CapFlat.flat_copies = copies
        return MESH.tetrahedralize(FakeSurface(extent=10.0), "mm", n_modes=4,
                                   n_index=1.5, max_tets=max_tets, verbose=False,
                                   min_feature_mm=None, scale=1.0)

    # 1. The cap is a LIMIT: a mesh that fits it is returned as-is, and one
    #    attempt is all it takes. The fake's default law reports 6 elements
    #    whatever ``maxvolume`` is, so 6 is exactly at the boundary.
    points, tets = mesh(None, max_tets=6)
    assert tets.shape[0] == 6, (
        f"the mesher returned {tets.shape[0]} elements for a cap of 6")
    assert len(CapFlat.calls) == 1, (
        "a mesh already inside the cap must not be coarsened")

    # ...and one element less must NOT be quietly accepted. This is the
    # assertion that fails if the cap is ever treated as advisory.
    with pytest.raises(MESH.MeshingError, match=r"above the 5 limit"):
        mesh(None, max_tets=5)

    # 2. A count the surface floors cannot be coarsened under the cap, so the
    #    refusal must NAME the limit -- the number the caller passed, not a
    #    constant of the module's own.
    #    Thousands separator included: the message formats with one, and a
    #    pattern that ignores it is a pattern that pins nothing but the digits.
    with pytest.raises(MESH.MeshingError, match=r"above the 4,242 limit"):
        mesh(4243, max_tets=4242)


def test_the_cli_flag_does_not_freeze_the_cap_at_parser_build_time():
    """``--max-tets`` defaults to None so ``main`` resolves it once, later.

    A parser default would bake the value in when ``build_parser()`` ran, so a
    ``set_max_tets`` (in a test) or a changed environment (in a deploy) would
    silently not apply -- and the guard would use a stale number. This pins the
    property that makes the resolution point single.
    """
    args = MESH_SOLVER.build_parser().parse_args(
        ["--job-id", "j", "--units", "mm", "--n-index", "1.0", "--n-modes", "2"])
    assert args.max_tets is None, (
        "the flag must default to None; resolution belongs in main()"
    )
    # The override is module state that OUTLIVES the call which set it, so this
    # test owns putting it back -- and the test that proves the leak is repaired
    # is test_a_solver_main_run_does_not_leak_its_cap_into_the_next below.
    MESH_SOLVER.set_max_tets(None)
    try:
        # and the env var is honoured by the accessor main() uses
        os.environ["MSE_MAX_TETS"] = "4321"
        assert MESH_SOLVER.max_tets() == 4321
    finally:
        os.environ.pop("MSE_MAX_TETS", None)
        MESH_SOLVER.set_max_tets(None)


def test_a_solver_main_run_leaves_the_module_cap_where_it_found_it(tmp_path):
    """The pollution that actually bit, pinned as a property rather than a hope.

    ``main()`` resolves its cap into a module global (``set_max_tets``). Two
    ``main()`` runs in one process are fine -- the second overwrites the global
    at its start, so it solves at its own cap (measured: 1,100 -> 1,058
    elements, then 3,000 -> 1,294). The damage lands on whatever comes *after*:
    a later call that does NOT go through ``main()`` -- a test asserting on the
    default guard, the bench, any reuse -- inherits the last run's cap.

    That is not hypothetical. ``test_maxwell_solver`` runs ``main(["--max-tets",
    "4000"])``, and the leaked 4,000 then made ``max_tets()`` return 4,000 in
    this file for the rest of the session, which is how this was found.

    The property asserted is the honest one: after a ``main()`` run returns,
    the module cap is what it was before. A first version of this test ran two
    ``main()`` calls and compared their meshes -- and PASSED with the restore
    deleted, because the second run overwrites the global anyway. Comparing two
    runs cannot see a leak that only affects a third party.
    """
    import mse_maxwell_solver as SOL
    from test_maxwell_solver import JOB, _write_box

    MESH_SOLVER.set_max_tets(None)
    try:
        before = SOL.max_tets()
        _write_box(tmp_path, JOB)
        rc = SOL.main([
            "--job-id", JOB, "--units", "mm", "--n-index", "1.0", "--n-modes",
            "3", "--local-in", str(tmp_path), "--local-out", str(tmp_path),
            "--max-tets", "1100", "--grid", "16", "--quiet",
        ])
        assert rc == 0
        assert SOL.max_tets() == before, (
            "main() left the module cap at its own run's value; the next caller "
            "that does not pass a cap explicitly will inherit it")
    finally:
        MESH_SOLVER.set_max_tets(None)


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


# ---------------------------------------------------------------------------
# F2: the element cap is ONE number, and the solver really uses it
# ---------------------------------------------------------------------------
#
# The defect: ``web/config.py`` capped uploads at 600_000 while the solver's own
# ``DEFAULT_MAX_TETS`` was 80_000, and nothing ever passed the app's value to the
# solver. So the app accepted an upload, told the owner it was fine, and the
# solver refused it at 80k -- failing precisely the large jobs the GPU exists
# for. The lead's benchmarks never hit it because they passed ``--max-tets``
# explicitly, which is why it survived this long.

def test_the_solver_cap_default_matches_the_web_apps_ceiling():
    """The app IMPORTS this, so the two cannot drift apart again.

    A test asserting ``600_000 == 600_000`` would pass against two copies that
    agree today; that is how the bug appeared. The real claim is that there is
    one definition, and it is checked from both sides: the constant's value, and
    the fact that ``web/config.py`` gets it by import.
    """
    import config as CFG

    assert MESH_SOLVER.DEFAULT_MAX_TETS == 600_000
    assert MESH_SOLVER.DEFAULT_CPU_MAX_TETS == 150_000
    assert CFG.DEFAULT_MAX_TETS is MESH_SOLVER.DEFAULT_MAX_TETS, (
        "the app must import the solver's constant, not restate its value"
    )


def test_the_mesh_cap_is_a_limit_not_a_target(monkeypatch):
    """The two ways a mesh can come out over the cap, and what the code says.

    This pins the contract the worker's ``solve()`` call rests on. There were
    THREE different gates in this path and a first version of this test drove
    the wrong one:

    * ``mse_meshing.tetrahedralize`` COARSENS to honour ``max_tets`` and
      returns happily -- measured: a 10 mm box that builds 6 elements at
      ``max_tets=1``. So "it returns" is NOT evidence the limit was respected,
      which is why the assertion below is on the returned COUNT.
    * When coarsening cannot help because the SURFACE is setting the element
      size, it refuses with a message naming the cap (``above the N limit``).
      That is the message production owners actually saw, and it is the one the
      app's 600_000 / solver's 80_000 disagreement turned into a lie.
    * ``solve()`` re-checks the exact count at ``mse_maxwell_solver.py:255-260``
      and refuses above ``max_tets``.

    Asserting only "something raised" would have passed on the broken code: the
    old ``solve()`` guard DID raise, it just used a number the app never sent.
    What is asserted here is the number the caller passed.
    """
    from test_meshing_guard import FakeSurface, FakeTetGen, _install

    class CapFlat(FakeTetGen):
        """A count that does NOT respond to ``maxvolume``, by fiat.

        The physical case the refusal describes is a surface whose facets floor
        the interior size, so no amount of coarsening shrinks the count. Stating
        it as a flag on the class rather than deriving it from the surface is
        deliberate: ``tetrahedralize`` passes ``np.asarray(surface.vertices)``
        to TetGen, so the surface object never reaches the fake and a
        surface-carried count silently did nothing.
        """

        #: how many elements to report, or ``None`` for the real 1/maxvolume law
        flat_copies: int | None = None

        def _copies_for(self, maxvolume) -> int:
            if type(self).flat_copies is not None:
                return type(self).flat_copies
            return super()._copies_for(maxvolume)

    _install(monkeypatch, CapFlat)
    monkeypatch.setattr(MESH, "decimate_to_edge_length",
                        lambda surface, edge_length, verbose=True: (surface, None))

    def mesh(copies: int | None, max_tets: int):
        CapFlat.calls = []
        CapFlat.flat_copies = copies
        return MESH.tetrahedralize(FakeSurface(extent=10.0), "mm", n_modes=4,
                                   n_index=1.5, max_tets=max_tets, verbose=False,
                                   min_feature_mm=None, scale=1.0)

    # 1. The cap is a LIMIT: a mesh that fits it is returned as-is, and one
    #    attempt is all it takes. The fake's default law reports 6 elements
    #    whatever ``maxvolume`` is, so 6 is exactly at the boundary.
    points, tets = mesh(None, max_tets=6)
    assert tets.shape[0] == 6, (
        f"the mesher returned {tets.shape[0]} elements for a cap of 6")
    assert len(CapFlat.calls) == 1, (
        "a mesh already inside the cap must not be coarsened")

    # ...and one element less must NOT be quietly accepted. This is the
    # assertion that fails if the cap is ever treated as advisory.
    with pytest.raises(MESH.MeshingError, match=r"above the 5 limit"):
        mesh(None, max_tets=5)

    # 2. A count the surface floors cannot be coarsened under the cap, so the
    #    refusal must NAME the limit -- the number the caller passed, not a
    #    constant of the module's own.
    #    Thousands separator included: the message formats with one, and a
    #    pattern that ignores it is a pattern that pins nothing but the digits.
    with pytest.raises(MESH.MeshingError, match=r"above the 4,242 limit"):
        mesh(4243, max_tets=4242)


def test_the_cli_flag_does_not_freeze_the_cap_at_parser_build_time():
    """``--max-tets`` defaults to None so ``main`` resolves it once, later.

    A parser default would bake the value in when ``build_parser()`` ran, so a
    ``set_max_tets`` (in a test) or a changed environment (in a deploy) would
    silently not apply -- and the guard would use a stale number. This pins the
    property that makes the resolution point single.
    """
    args = MESH_SOLVER.build_parser().parse_args(
        ["--job-id", "j", "--units", "mm", "--n-index", "1.0", "--n-modes", "2"])
    assert args.max_tets is None, (
        "the flag must default to None; resolution belongs in main()"
    )
    # The override is module state that OUTLIVES the call which set it, so this
    # test owns putting it back -- and the test that proves the leak is repaired
    # is test_a_solver_main_run_does_not_leak_its_cap_into_the_next below.
    MESH_SOLVER.set_max_tets(None)
    try:
        # and the env var is honoured by the accessor main() uses
        os.environ["MSE_MAX_TETS"] = "4321"
        assert MESH_SOLVER.max_tets() == 4321
    finally:
        os.environ.pop("MSE_MAX_TETS", None)
        MESH_SOLVER.set_max_tets(None)


def test_a_solver_main_run_does_not_leak_its_cap_into_the_next(tmp_path):
    """``main()`` sets a process-wide cap; a finished run must not leave it set.

    This is the real defect that surfaced as a cross-test failure, and it was
    mine, not the test runner's. ``main()`` resolves the cap once and stores it
    in a module global, so a Job that ran with ``--max-tets 4000`` left the
    *module* capped at 4,000 for everything afterwards in that process. Two
    ``main()`` calls in one process -- a test session, or any future reuse --
    then solved a second job at the first job's limit.

    In production each Job is its own process, so the leak cannot cross jobs
    there; it bites exactly where two runs share a process. The fix (documented
    on ``set_max_tets``) is that a caller owns restoring the override, and this
    test is what holds that: it runs two real ``main()`` calls with different
    caps and asserts the second one's cap is not the first one's.
    """
    import json

    import mse_maxwell_solver as SOL
    from test_maxwell_solver import JOB, _write_box

    def run(job_id, cap):
        _write_box(tmp_path, job_id)
        rc = SOL.main([
            "--job-id", job_id, "--units", "mm", "--n-index", "1.0",
            "--n-modes", "3", "--local-in", str(tmp_path),
            "--local-out", str(tmp_path), "--max-tets", str(cap),
            "--grid", "16", "--quiet",
        ])
        assert rc == 0
        return json.loads(
            (tmp_path / "jobs" / job_id / "modes.json").read_text())

    # Caps chosen to BIND, measured on this exact box: at 1,100 the mesher
    # coarsens the interior to 665 elements, at 3,000 it leaves the natural
    # 1,257. (4,000 vs 6,000 does NOT work and is worth knowing: both sit above
    # the natural 1,257, so the two runs are identical by physics and the test
    # would fail while proving nothing about the leak.)
    first = run(JOB, 1100)
    second = run("99999999-2222-3333-4444-555555555555", 3000)
    assert second["n_tets"] > first["n_tets"], (
        f"both runs meshed to {first['n_tets']} elements, so the second run "
        "solved at the first run's cap: the module override leaked")
