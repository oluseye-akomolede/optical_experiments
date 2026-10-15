"""Tier-1 tests for ``benchmarks/run_grading_matrix.py``.

Step 2 of row qitem-20261013130141-600283f7. These are tier-1: no meshing, no
solving, no Orin. The product's estimator is monkeypatched wherever a test only
needs the REPORTING path, so the suite stays fast; the last test in the file is
the one that runs the estimator for real, over the whole case list.

The criterion is the owner's bar -- graded error no worse than the uniform
mesh's -- so the arms that matter are the ones where a bar CANNOT be measured.
An unmeasured bar that reads as a pass is the failure this file exists to make
impossible, and the first test below to redden if the ``UNMEASURED`` arm is
folded back into ``PASS``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "benchmarks", _ROOT / "dockerfiles" / "maxwell_solver"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_grading_matrix as GM  # noqa: E402


# --------------------------------------------------------------------------
# the case list covers the axes an arbitrary upload varies
# --------------------------------------------------------------------------

def test_matrix_covers_every_axis_value():
    cov = GM._coverage(GM.MATRIX)
    assert set(cov["contrast"]) == {1.5, 2.0, 3.0, 4.0}, cov["contrast"]
    assert set(cov["shell"]) == {"thick", "thin"}, cov["shell"]
    assert set(cov["regions"]) == {2, 3}, cov["regions"]
    assert set(cov["reference"]) == {"analytic", "uniform-refined"}


def test_matrix_has_a_non_spherical_and_a_non_concentric_case():
    """The owner's question is whether grading holds for any mesh, so the shape
    axis must carry cases a centred sphere cannot stand in for."""
    shapes = set(GM._coverage(GM.MATRIX)["shape"])
    assert "box" in shapes, f"no non-spherical case: {shapes}"
    assert "offset" in shapes, f"no non-concentric case: {shapes}"


def test_the_coverage_check_can_fail():
    """A coverage check that cannot fail leaves the gap in. Drop the only
    contrast-4 case and the missing value has to be visible."""
    without = [c for c in GM.MATRIX if c["axes"]["contrast"] != 4.0]
    assert len(without) < len(GM.MATRIX), "no case carried contrast 4 to drop"
    assert 4.0 not in set(GM._coverage(without)["contrast"])


def test_case_labels_are_unique():
    labels = [c["label"] for c in GM.MATRIX]
    assert len(labels) == len(set(labels)), "duplicate case label"


def test_the_thin_cases_trip_the_clamp_and_the_thick_ones_do_not():
    """The shell axis is 'thin where the clamp bites'. That has to be true of
    the plan the rule produces, not of the label -- a thin case whose clamp
    never fires would leave the axis uncovered while reading as covered."""
    def clamp_bites(label):
        case = next(c for c in GM.MATRIX if c["label"] == label)
        plan = GM._build(case, case["n_modes"])["plan"]
        return [rid for rid, v in plan.items()
                if v["edge_m"] < v["unclamped_m"] - 1e-12]

    assert clamp_bites("c2-thin"), "the thin case's clamp never bit"
    assert not clamp_bites("c2-thick"), "the thick case's clamp bit"


def test_the_anchor_case_is_the_other_bench_s_fixture():
    """'c3-thick' is the case the row cites (the 199,062-element uniform mesh,
    0.75% vs 0.66%): b = 10 mm, core radius b/3, n = 1 and 3, ten modes. If its
    sizing drifts, the row's own numbers stop describing this case."""
    case = next(c for c in GM.MATRIX if c["label"] == "c3-thick")
    assert case["n_modes"] == 10, "the cited case must keep the sibling's modes"
    assert case["materials"] == {"wall": 1.0, "core": 3.0}


# --------------------------------------------------------------------------
# the criterion: PASS / FAIL / UNMEASURED
# --------------------------------------------------------------------------

def test_criterion_passes_when_graded_is_no_worse():
    verdict, detail = GM.criterion(0.0034, 0.0056, None)
    assert verdict == "PASS"
    assert "0.340%" in detail and "0.560%" in detail


def test_criterion_fails_when_graded_is_worse():
    verdict, _ = GM.criterion(0.00749, 0.00662, None)
    assert verdict == "FAIL"


def test_the_bar_is_no_worse_and_the_boundary_is_inclusive():
    """'No worse than' -- equal errors pass. Pinning the limit keeps a later
    strict-inequality change from moving the bar silently."""
    assert GM.criterion(0.005, 0.005, None)[0] == "PASS"
    assert GM.criterion(0.005001, 0.005, None)[0] == "FAIL"


def test_an_unmeasured_bar_never_passes():
    """The mutant this test is for: drop the ``if unmeasured_reason`` arm and
    the first assertion reddens -- a better-than-uniform graded error over an
    unsolved reference would read PASS."""
    verdict, detail = GM.criterion(0.001, 0.010, "uniform reference did not solve")
    assert verdict == "UNMEASURED", "an unmeasured bar must not read as a pass"
    assert "uniform reference" in detail

    assert GM.criterion(None, 0.010, None)[0] == "UNMEASURED"
    assert GM.criterion(0.001, None, None)[0] == "UNMEASURED"


# --------------------------------------------------------------------------
# the listing: predictions, sizing and honest failures
# --------------------------------------------------------------------------

def test_routing_note_reads_production_ceilings():
    """Read from web.config, the same source the upload route reads. If the
    import breaks, the note says so instead of omitting the ceiling."""
    note = GM._routing_note()
    assert not note.startswith("PRODUCTION CEILINGS UNAVAILABLE"), note
    assert "refuses an upload above" in note


def test_over_ceiling_cases_are_reported(monkeypatch, capsys):
    """A case too big for tier 2 must be visible in the list rather than met as
    an OOM mid-run."""
    def fake(case):
        # the real ``predicted_counts`` shape (see its ``return``): the two
        # floors, the probe's own count, and the ratio between the floors
        return {"h_min_mm": 0.5961, "probe_h_min_mm": 0.5961,
                "probe_uniform_tets": 250_000, "baseline_ratio": 1.0,
                "counts": {"graded": 1_000, "uniform": 250_000},
                "over_ceiling": ["uniform"]}

    monkeypatch.setattr(GM, "predicted_counts", fake)
    rc = GM.list_cases(GM.MATRIX[:2])
    out = capsys.readouterr().out
    assert rc == GM.MATRIX_OK
    assert "over the tier-2 mark" in out, out


def test_a_failed_estimate_is_incomplete_not_a_zero(monkeypatch, capsys):
    """The first ``--list`` run failed every count on a bad config import and
    said so. A count that could not be taken must not print as 0."""
    def boom(case):
        raise RuntimeError("no estimator")

    monkeypatch.setattr(GM, "predicted_counts", boom)
    rc = GM.list_cases(GM.MATRIX[:1])
    out = capsys.readouterr().out
    assert rc == GM.MATRIX_INCOMPLETE
    assert "COUNT FAILED" in out, out
    assert "RuntimeError" in out


#: Which meshes the row expects the Orin may not solve: label -> the predicted
#: meshes over the tier-2 mark, measured with the production estimator. The row
#: asks for this list "plainly", so it is a checked artifact rather than a line
#: in a report: a sizing change or an estimator change that moves a case across
#: the mark has to update this map.
#:
#: Corrected on 2026-10-14, and the change is the point of the correction: the
#: uniform side is now sized at the floor the PRODUCT's own plan asks for, not
#: at a fixed icosphere probe's. `c3-thin`'s uniform mesh drops under the mark
#: (274,415 -> 154,673 predicted) while its graded side stays on it, and
#: `box-sphere`'s two reference levels -- 250,439 and 432,759 -- come over it,
#: which is a case whose reference cannot be built on tier 2 at all.
EXPECTED_OVER_MARK = {
    "c3-thick": ["uniform"],             # the row's own 199,062-element mesh
    "c3-thin": ["graded"],               # the clamp, not the mode count, sizes it
    "box-sphere": ["ref1.25", "ref1.5"],  # both reference levels, on the box's
                                          # own (much finer) floor
}


def test_every_case_is_sized_for_tier_2_or_listed_as_unmeasurable():
    """The row's 'each case sized to fit tier 2', locked. Also the estimator
    seam: every predicted count here comes from the product's own
    ``estimate_tets`` with the constant ``web.config`` routes on, so a broken
    import shows up as a failed estimate for every case, not as a green suite
    over a dead estimator. This is the slow test in this file.

    The map below is the LIST's flag, not a prediction that the Orin will OOM:
    the estimator reads 1.2-3.3x high on the meshes we have built, so a case
    flagged here can still solve, and the Orin's own run decides."""
    over = {}
    for case in GM.MATRIX:
        pred = GM.predicted_counts(case)
        assert pred["counts"]["graded"] > 0, (case["label"], pred)
        assert pred["counts"]["uniform"] > 0, (case["label"], pred)
        if pred["over_ceiling"]:
            over[case["label"]] = pred["over_ceiling"]
    assert over == EXPECTED_OVER_MARK, (
        "the set of cases the Orin cannot solve moved; either the sizing "
        f"changed or the estimator did: {over} != {EXPECTED_OVER_MARK}")


# --------------------------------------------------------------------------
# a missing side says which side it is (the Orin's box-sphere row)
# --------------------------------------------------------------------------

def test_a_missing_side_names_the_side_and_what_it_produced():
    """The Orin's box-sphere row came back ``graded_worst=None`` over a solve
    that SUCCEEDED, and the message said only "one of the two worst errors is
    missing". Which side, and what did it produce, is the whole diagnosis."""
    score = {"worst_rel_error": None, "n_clusters": 4, "n_matched": 0,
             "n_returned": 2,
             "spurious": [{"nearest_rel": 0.0512}, {"nearest_rel": 0.19}],
             "unmatched_reference": ["a", "b", "c"]}
    reason = GM.missing_side_reason("the GRADED mesh", score, ["r"] * 5)
    assert reason is not None
    assert "the GRADED mesh" in reason
    assert "0 of its 4 clusters" in reason
    assert "3 of 5 reference multiplets unmatched" in reason
    assert "5.12%" in reason, reason        # the NEAREST cluster, not the worst
    # the tolerance in the message is the SCORER's, read from its signature --
    # pin both, so a change to either shows up here rather than in a report
    assert GM.SCORE_TOL == 0.02, GM.SCORE_TOL
    assert f"{GM.SCORE_TOL * 100:.2f}%" in reason, reason
    assert "19.00%" not in reason, reason


def test_a_measured_side_carries_no_reason():
    """The other arm: a side that produced a number must not be described as
    missing, or the detail would name the wrong side on a good run."""
    assert GM.missing_side_reason("x", {"worst_rel_error": 0.01}, ["r"]) is None


def test_a_missing_side_with_no_cluster_at_all_still_says_so():
    """A side that produced nothing at all must say that, rather than print a
    distance it does not have."""
    reason = GM.missing_side_reason(
        "the UNIFORM mesh",
        {"worst_rel_error": None, "n_clusters": 0, "n_matched": 0,
         "spurious": [], "unmatched_reference": ["a"]}, ["r"])
    assert "the UNIFORM mesh" in reason
    assert "no cluster came within" in reason, reason
    assert "distance" not in reason, reason


# --------------------------------------------------------------------------
# item 5: how much the reference's own quality can flatter the uniform side
# --------------------------------------------------------------------------

def test_the_uniform_sides_favourable_bound_is_arithmetic_not_opinion():
    """Item 5, with the Orin's own two uniform-refined FAILs. Both sides are
    scored against a finer mesh of the uniform side's own family, so the
    comparison leans that way; the reference's convergence is the only bound on
    by how much. Adding ALL of it to the uniform side is the best case for the
    uniform mesh, and the verdict that survives that reading is the one worth
    acting on."""
    # three-region: uniform 0.106%, graded 0.231%, reference converges 0.046%
    b = GM.favourable_bound(0.00106, 0.00231, 0.00046)
    assert abs(b["uniform"] - 0.00152) < 1e-12, b
    assert abs(b["ratio"] - 1.5197) < 1e-3, b
    assert b["still_fail"] is True, b
    # offset-sphere: uniform 0.028%, graded 0.091%, reference converges 0.005%
    b2 = GM.favourable_bound(0.00028, 0.00091, 0.00005)
    assert abs(b2["uniform"] - 0.00033) < 1e-12, b2
    assert abs(b2["ratio"] - 2.7576) < 1e-3, b2
    assert b2["still_fail"] is True, b2


def test_the_raw_bound_is_a_floor_on_the_correction_not_a_ceiling():
    """The coarse-vs-finest difference is NOT the reference's own error: it is
    ``1 - r**-p`` of it, so the correction that a same-family reference gives
    away is a multiple of that difference, and the raw bound above is the
    smallest defensible version of it. Three-region survives the scaled one --
    by 1.10x, which is the margin that has to travel with the verdict."""
    r = GM.RICHARDSON_SCALE
    # derived from the bench's own refinement pair, not typed in: 1/(1.44-1)
    assert abs(r - 1.0 / ((GM.REFINE_FACTORS[-1] / GM.REFINE_FACTORS[0]) ** 2 - 1.0)) < 1e-12
    assert abs(r - 2.2727) < 1e-3, r
    b = GM.favourable_bound(0.00106, 0.00231, 0.00046, richardson=r)
    assert abs(b["richardson_uniform"] - (0.00106 + r * 0.00046)) < 1e-12, b
    assert abs(b["richardson_ratio"] - 1.0975) < 1e-3, b
    assert b["richardson_still_fail"] is True, b
    # offset-sphere's 3.27x stands up to the same scaling: 0.091% against
    # 0.028% + 2.27 * 0.005% = 0.039%, still 2.3x.
    b2 = GM.favourable_bound(0.00028, 0.00091, 0.00005, richardson=r)
    assert abs(b2["richardson_ratio"] - 2.3116) < 1e-3, b2
    assert b2["richardson_still_fail"] is True, b2
    # And the honest limit, as arithmetic: the correction would have to top
    # (0.231 - 0.106) / 0.046 = 2.72x the difference it measures -- 1.20x the
    # O(h^2) estimate -- for three-region's FAIL to become a tie.
    assert GM.favourable_bound(0.00106, 0.00231, 0.00046,
                               richardson=2.7)["richardson_still_fail"] is True
    flip = GM.favourable_bound(0.00106, 0.00231, 0.00046, richardson=2.8)
    assert flip["richardson_still_fail"] is False, flip
    assert flip["still_fail"] is True, flip  # the raw reading is unaffected
    # offset-sphere's threshold is (0.091 - 0.028) / 0.005 = 12.6x, far
    # outside the same assumption.
    assert GM.favourable_bound(0.00028, 0.00091, 0.00005,
                               richardson=12.0)["richardson_still_fail"] is True
    assert GM.favourable_bound(0.00028, 0.00091, 0.00005,
                               richardson=13.0)["richardson_still_fail"] is False


def test_a_bound_that_settles_nothing_says_so():
    """The honest half: when the convergence is large enough to swallow the
    difference, the bound decides nothing and must not print the reassuring
    line. box-sphere is this shape in the real run -- a 1.007% convergence
    against a uniform side whose whole score is 1.442% (70% of it), so the
    convergence is 2.3x the difference it is supposed to explain, and the pair
    is checked here on those proportions rather than on a verdict nobody
    measured. (Its graded side has no number at all, so the real run gets no
    bound there: see the test below.)"""
    # box-sphere's proportions: uniform 1.44%, graded 1.88%, convergence 1.01%
    assert GM.favourable_bound(0.0144, 0.0188, 0.0101)["still_fail"] is False
    assert GM.favourable_bound(0.0144, 0.0188, 0.0101)["ratio"] < 1.0
    # the same pair with a convergence small enough to be a bound at all
    assert GM.favourable_bound(0.0144, 0.0188, 0.0020)["still_fail"] is True


def test_a_bound_needs_all_three_operands():
    """A bound missing a term would print as a measurement. An analytic
    reference has no convergence to give away, and a side with no number has no
    bound either."""
    assert GM.favourable_bound(None, 0.01, 0.001) is None
    assert GM.favourable_bound(0.01, None, 0.001) is None
    assert GM.favourable_bound(0.01, 0.02, None) is None


# --------------------------------------------------------------------------
# the corrected baseline: the uniform side is sized by THIS case's plan
# --------------------------------------------------------------------------

#: The graded side's one frequency, in Hz. It only has to differ from the
#: uniform side's, because the fake's two answers are selected by the value the
#: runner's own ``_kb`` makes of it.
GRADED_FREQ_HZ = 1.0


def _fake_run_case(monkeypatch, baseline: str = "product", ladder: bool = False,
                   fine_reference=None, rotations: int = 0):
    """``run_case`` with every heavy call faked out. Returns ``(result, seen)``.

    ``seen`` records the floor the uniform side was built at, which is the
    thing the baseline choice moves. ``_build`` reports the two floors the real
    one measures for ``box-sphere`` (1.0105 mm of base edge on the box's own
    surface against 2.2533 mm on the fixed icosphere probe), so the arithmetic
    a test sees is the arithmetic the Orin saw, without meshing anything.

    The two sides are told apart by their SPECTRUM, run through the runner's own
    ``_kb``: a score fake that branches on a hand-picked number is how this test
    first returned PASS while asserting UNMEASURED -- the graded frequency
    landed in the uniform branch, and a faked side that is never asked for a
    number cannot be the side under test.
    """
    import numpy as np

    seen: dict = {}
    graded_kb = float(GM._kb(np.array([GRADED_FREQ_HZ]), 5.0)[0])

    def fake_build(case, n_modes):
        return {"obj": (b"o wall\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                        b"f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"),
                "parts": None,
                "plan": {"0": {"edge_m": 1e-3, "unclamped_m": 1e-3}},
                "h_min_mm": 0.3368, "probe_h_min_mm": 0.7511,
                "layers": [], "scale_mm": 5.0}

    def fake_graded(case, obj, work, *, grid, backend, verbose):
        return ({"n_tets": 1000},
                {"mesh": {"regions": {"0": {"target_edge_units": 1.0}}}},
                np.array([GRADED_FREQ_HZ]), 0.5)

    def fake_uniform(materials, obj, case, h_min_mm, *, k, layers, backend,
                     np_verbose=False, rot=None, about=None):
        seen.setdefault("uniform_floors", []).append((k, h_min_mm))
        if rot is not None:
            # A rotated build carries no rung: it rebuilds the k=1.0 uniform
            # mesh and must be handed its pivot. Each orientation gets its OWN
            # spectrum, so a band that ignored the rotations would read 0.
            seen.setdefault("rot_pivots", []).append(tuple(np.asarray(about, float)))
            seen["rot_n"] = seen.get("rot_n", 0) + 1
            return (np.zeros((7, 3), float), np.zeros((7, 4), int),
                    np.array([3.0 + seen["rot_n"]]))
        # A NODE CLOUD, not None. The runner now checks the points a level's
        # build returned before a reference is allowed to come from it, so a
        # fake that returned None would model a producer the runner refuses --
        # and every level would come back as a failed build.
        return np.zeros((7, 3), float), np.zeros((7, 4), int), np.array([2.0])

    def fake_uniform_regions(case, obj, h_min_mm, verbose=False):
        seen["region_aware_floor"] = h_min_mm
        return (np.zeros((5, 4), int), np.zeros(5, int),
                {"regions": {"0": {"target_edge_units": h_min_mm,
                                   "achieved_edge_units": 0.9}}})

    def fake_score(kb, reference, n_requested):
        seen.setdefault("scored", []).append(float(kb[0]))
        if float(kb[0]) == graded_kb:         # the graded spectrum
            return {"worst_rel_error": None, "n_clusters": 3, "n_matched": 0,
                    "n_returned": 1, "spurious": [{"nearest_rel": 0.4}],
                    "unmatched_reference": ["a", "b", "c"]}
        if float(kb[0]) >= 3.0:               # a rotation member of the band
            return {"worst_rel_error": 0.02 + 0.002 * (float(kb[0]) - 3.0),
                    "n_clusters": 3, "n_matched": 3, "n_returned": 3,
                    "spurious": [], "unmatched_reference": []}
        return {"worst_rel_error": 0.02, "n_clusters": 3, "n_matched": 3,
                "n_returned": 3, "spurious": [], "unmatched_reference": []}

    monkeypatch.setattr(GM, "_build", fake_build)
    monkeypatch.setattr(GM, "_graded", fake_graded)
    monkeypatch.setattr(GM, "_uniform", fake_uniform)
    monkeypatch.setattr(GM, "_uniform_regions", fake_uniform_regions)
    # The reference covers the five modes the case asks for: a reference that
    # did NOT would be refused by the runner's reference gate, which is a
    # different test's subject.
    monkeypatch.setattr(GM, "_clusters_pairs", lambda kb: [(1.0, 0, "f", 5)])
    monkeypatch.setattr(GM.REF, "score_multiplets", fake_score)
    case = next(c for c in GM.MATRIX if c["label"] == "box-sphere")
    result = GM.run_case(case, grid=10, backend="cpu", work=Path("/tmp"),
                         verbose=False, baseline=baseline, ladder=ladder,
                         fine_reference=fine_reference, rotations=rotations)
    return result, seen


def _fake_run_case_with(monkeypatch, **overrides):
    """``_fake_run_case`` with named runner functions replaced, then re-run.

    ``_fake_run_case`` patches the whole call graph, so an override has to be
    applied after it and the case rebuilt; this keeps that ordering in one
    place instead of in each test that needs a different producer.
    """
    _fake_run_case(monkeypatch)
    for name, fn in overrides.items():
        monkeypatch.setattr(GM, name, fn)
    case = next(c for c in GM.MATRIX if c["label"] == "box-sphere")
    return GM.run_case(case, grid=10, backend="cpu", work=Path("/tmp"),
                       verbose=False)


def test_the_uniform_side_is_sized_by_the_products_own_floor(monkeypatch):
    """Item 3 of the Orin diagnosis: the bench's uniform mesh must be the mesh
    the product builds today. Before 2026-10-14 the bench sized it from a fixed
    icosphere probe; for the box-shaped case the two floors are 2.23x apart, so
    every verdict on that case was against a mesh the product never builds."""
    result, seen = _fake_run_case(monkeypatch)
    out = result["case"]
    assert out["baseline"] == "product"
    assert out["h_min_mm"] == 0.3368, out
    assert out["h_min_mm_product"] == 0.3368 and out["h_min_mm_probe"] == 0.7511
    floors = {h for _k, h in seen["uniform_floors"]}
    assert floors == {0.3368}, f"the uniform side was not built at h_min: {floors}"
    assert seen["region_aware_floor"] == 0.3368
    # a missing side now reaches the criterion as a REASON, not as a gap
    assert result["verdict"] == "UNMEASURED"
    assert "the GRADED mesh" in result["detail"], result["detail"]
    assert out["graded_score"]["n_returned"] == 1
    assert out["uniform_regions"]["regions"]["0"]["achieved_edge_units"] == 0.9


def test_the_probe_baseline_is_still_selectable_and_both_floors_are_reported(
        monkeypatch):
    """The pre-2026-10-14 numbers were measured against the probe floor. They
    stay reproducible -- and the run says which floor it used either way, so a
    number cannot be read against the wrong baseline again."""
    result, seen = _fake_run_case(monkeypatch, baseline="probe")
    out = result["case"]
    assert out["baseline"] == "probe"
    assert out["h_min_mm"] == 0.7511
    assert {h for _k, h in seen["uniform_floors"]} == {0.7511}
    # ... and the product's floor is still reported, so the two can be compared
    assert out["h_min_mm_product"] == 0.3368


# --------------------------------------------------------------------------
# the uniform side's per-region build (item 2)
# --------------------------------------------------------------------------

def test_the_region_aware_uniform_build_pins_every_region_and_lifts_the_pin(
        monkeypatch):
    """Two claims in one: every region was pinned to the SAME floor (or the
    per-region table would not describe a uniform mesh), and the product's own
    rule is back afterwards (or the graded side of the same case would have
    been meshed by the bench)."""
    import numpy as np
    import mse_common as C
    import mse_meshing as MESH

    real_edges = C.region_edges
    captured: dict = {}

    def fake_tetrahedralize(merged, units, **kw):
        edges = C.region_edges(0.5e-3, 1.0, kw["materials"], kw["parts"], "mm")
        captured["edges"] = edges
        captured["still_the_real_one"] = C.region_edges is real_edges
        MESH.LAST_MESH_REPORT = {"regions": {"0": {"target_edge_units": 0.3368}}}
        return None, np.zeros((3, 4), int), np.zeros(3, int)

    case = next(c for c in GM.MATRIX if c["label"] == "c1.5-thick")
    built = GM._build(case, case["n_modes"])
    monkeypatch.setattr(MESH, "tetrahedralize", fake_tetrahedralize)
    tets, regions, report = GM._uniform_regions(case, built["obj"], 0.3368)

    assert C.region_edges is real_edges, "the product's rule was left patched"
    assert captured["still_the_real_one"] is False, "the pin never went in"
    assert captured["edges"], "the real region_edges returned no region"
    for rid, info in captured["edges"].items():
        assert info["edge"] == 0.3368 * 1e-3, (rid, info)
        assert info["unclamped"] == 0.3368 * 1e-3, (rid, info)
    assert len(captured["edges"]) == len(case["materials"]), (
        "fewer regions were pinned than the case has: "
        f"{sorted(captured['edges'])}")
    assert report == {"regions": {"0": {"target_edge_units": 0.3368}}}
    assert tets.shape[0] == 3 and regions.shape[0] == 3


# --------------------------------------------------------------------------
# item 4's two new case types
# --------------------------------------------------------------------------

def test_a_core_lighter_than_its_shell_plans_uniformly():
    """A dense shell around a lighter core: no region is denser than the wall,
    so the rule's raw edges are h*2/2 = h and h*2/1 = h. The case exists to say
    whether the rule does anything at all there; the plan is the answer, and it
    is measurable without meshing."""
    case = next(c for c in GM.MATRIX if c["label"] == "shell-dense")
    plan = GM._build(case, case["n_modes"])["plan"]
    edges = [v["edge_m"] for v in plan.values()]
    assert len(edges) >= 2, plan
    assert max(edges) - min(edges) <= 1e-12 + 1e-6 * min(edges), plan


def test_a_wall_with_index_above_one_is_divided_by_its_own_index():
    """The other new type: n_wall = 2.0, which every earlier case left at 1.0.
    The rule is ``raw_r = min(h, h * n_wall / n_r)``, so the wall keeps h and
    the core -- the only region denser than the wall -- gets 2h/3. The middle
    region asks h*2/1.5, ABOVE h, so it lands on the outer ``min(h, .)``: the
    rule never coarsens a region past the wall, and this case is the only one
    whose ramp would go there. Measured on the corrected baseline: 2.4313 mm
    for the wall and the middle region, 1.6209 mm for the core.
    """
    case = next(c for c in GM.MATRIX if c["label"] == "ramp-outward")
    plan = GM._build(case, case["n_modes"])["plan"]
    edges = sorted(v["edge_m"] for v in plan.values())
    assert len(edges) == 3, plan
    assert abs(edges[0] - (2.0 / 3.0) * edges[1]) < 1e-9 * edges[1], edges
    assert edges[1] == edges[2], (
        f"the middle region should have been capped at the wall's edge: {edges}")
    # the cap, not the material ratio, is what keeps the wall from coarsening:
    # the largest raw ask in the plan IS the wall's own edge, nothing is past it
    h = max(v["unclamped_m"] for v in plan.values())
    assert h == edges[2], (edges, h)
    assert any(v["edge_m"] == h and v["unclamped_m"] == h
               for v in plan.values()), plan


def test_the_shell_dense_case_predicts_an_equal_pair():
    """The estimator's two paths (the graded one and the min_feature_m one)
    must agree when the plan is uniform, or a 'uniform' case would be scored
    against a differently-sized uniform mesh. This is the case built to have an
    exactly uniform plan."""
    case = next(c for c in GM.MATRIX if c["label"] == "shell-dense")
    pred = GM.predicted_counts(case)
    assert pred["counts"]["graded"] == pred["counts"]["uniform"], pred["counts"]


# --------------------------------------------------------------------------
# the list says when a number was measured against the other baseline
# --------------------------------------------------------------------------

def test_the_list_gives_the_base_edge_ratio_and_not_the_count_ratio(
        monkeypatch, capsys):
    """The floors are lengths and the counts are volumes, so the count ratio is
    the length ratio CUBED. An earlier version of this line printed the count
    ratio as if it were the length: box-sphere read 'the old probe floor
    0.7511mm is 0.09x the product's 0.3368mm' when the floor is 2.23x the
    product's and the MESH is 0.09x the elements."""
    def fake(case):
        return {"h_min_mm": 0.3368, "probe_h_min_mm": 0.7511,
                "probe_uniform_tets": 11_564, "baseline_ratio": 0.0901,
                "counts": {"graded": 18_466, "uniform": 128_225},
                "over_ceiling": []}

    monkeypatch.setattr(GM, "predicted_counts", fake)
    GM.list_cases(GM.MATRIX[:1])
    out = capsys.readouterr().out
    assert "2.23x the product's 0.3368mm" in out, out
    assert "11,564 tets on the old baseline and 128,225" in out, out
    assert "0.09x" not in out, out
    # the count runs once per row and once in the summary, so one case with a
    # deviating floor says "1 of 1" rather than leaving the reader to count
    # warning lines.
    assert "1 of 1 cases" in out, out


def test_the_list_says_nothing_when_the_two_floors_agree(monkeypatch, capsys):
    """The warning is a claim about a DIFFERENCE; with the floors equal it must
    be silent, or it would colour every row."""
    def fake(case):
        return {"h_min_mm": 0.5, "probe_h_min_mm": 0.501,
                "probe_uniform_tets": 1000, "baseline_ratio": 1.0,
                "counts": {"graded": 900, "uniform": 1000},
                "over_ceiling": []}

    monkeypatch.setattr(GM, "predicted_counts", fake)
    GM.list_cases(GM.MATRIX[:1])
    out = capsys.readouterr().out
    assert "base edge" not in out, out
    assert "cases:" not in out, out


def test_the_case_print_block_survives_a_result_with_gaps(monkeypatch, capsys):
    """``main --case`` is the only place the per-region lines are printed, and
    for a while nothing reached it: a real run needs a solve, which is tier 2
    and not this suite's to start.

    So it is fed ``run_case``'s own output shape with the heavy calls faked --
    and deliberately a HOLEY one, because every gap is a branch: the graded side
    has no ``achieved_edge_units``, so the block must print the missing number
    rather than a zero, and the fake's graded side is the missing one, so the
    UNMEASURED detail has to survive the trip to stdout too.
    """
    _fake_run_case(monkeypatch)
    rc = GM.main(["--case", "box-sphere", "--grid", "10", "--backend", "cpu",
                  "--quiet"])
    out = capsys.readouterr().out
    assert "RAISED" not in out, out
    assert rc == GM.MATRIX_INCOMPLETE, out
    assert "graded 1,000 tets; uniform" in out, out
    assert "baseline product: h_min 0.3368 mm (product 0.3368, probe 0.7511)" in out, out
    assert "graded region 0: asked 1.0 units, built None" in out, out
    assert "uniform region 0: asked 0.3368 units, built 0.9" in out, out


# ------------------------------------------------- step 2b: the five items
#
# 1  SAME MESH when the plan is uniform and the counts match
# 2  signed per-mode errors for both sides, and UNRELIABLE when they cannot be
#    read (non-monotone ladder, one multiplet, a crossing)
# 3  the tier-2 mark read off BUILT counts, not estimates
# 5  the estimator's two paths against what the mesher built


def test_a_uniform_plan_is_recognised_by_its_targets():
    """Item 1's plan half: all regions asking the same edge is a uniform plan."""
    assert GM.plan_is_uniform({"0": {"edge_m": 1e-3}, "1": {"edge_m": 1e-3}})
    assert not GM.plan_is_uniform({"0": {"edge_m": 1e-3}, "1": {"edge_m": 2e-3}})
    # One region has no contrast to flatten, so it is not evidence about
    # grading either way -- and calling it SAME MESH would retire a case that
    # never tested anything as if it had passed.
    assert not GM.plan_is_uniform({"0": {"edge_m": 1e-3}})
    assert not GM.plan_is_uniform({})
    assert not GM.plan_is_uniform(None)


def test_same_mesh_outranks_pass_and_fail():
    """Grading that did nothing is neither a win nor a loss, whatever the numbers."""
    verdict, detail = GM.criterion(0.001, 0.001, None, same_plan=True)
    assert verdict == "SAME MESH", (verdict, detail)
    assert "same mesh" in detail
    # Even when the graded mesh is STRICTLY worse -- the comparison is vacuous
    # either way, so the worse reading must not be promoted to a FAIL.
    assert GM.criterion(0.010, 0.001, None, same_plan=True)[0] == "SAME MESH"


def test_the_real_c2_thin_pair_is_same_mesh_and_the_strict_mutant_is_not():
    """The lead's item 1, on the Orin's own c2-thin numbers.

    c2-thin's graded plan asks both regions for 1.0825 mm, so the graded build
    IS the uniform build: 22,947 elements both sides, and the solver's two
    readings differ in the 13th digit. Under the old strict ``<=`` that pair
    reads FAIL -- the graded number is larger -- which is how a case that
    exercised nothing came back as a loss.

    The mutant is the second half: the SAME numbers with the guard removed must
    come back FAIL, or the guard is not what produced SAME MESH and this test
    would pass with the fix reverted.
    """
    graded, uniform = 0.0019135674962923, 0.0019135674962911
    guard = GM.criterion(graded, uniform, None, same_plan=True)
    assert guard[0] == "SAME MESH", guard

    mutant = GM.criterion(graded, uniform, None)      # guard removed
    assert mutant[0] == "FAIL", mutant
    assert graded > uniform                           # why it flips


def test_the_tolerance_is_the_measured_residual_or_nothing():
    """Item 1's other half: no picked number, and no silent zero."""
    graded, uniform, residual = 0.051, 0.050, 0.002
    assert GM.criterion(graded, uniform, None, tol=0.0)[0] == "FAIL"
    verdict, detail = GM.criterion(graded, uniform, None, tol=residual)
    assert verdict == "PASS", detail
    assert "measured residual" in detail and "2.00e-03" in detail
    # A gap wider than the residual is still a real difference.
    assert GM.criterion(graded + 0.003, uniform, None, tol=residual)[0] == "FAIL"


def test_the_runner_reports_a_missing_residual_as_unavailable(monkeypatch):
    """No residual anywhere means a strict comparison, said out loud."""
    import sys
    import types

    stub = types.ModuleType("mse_eigen")           # no LAST_WORST_RESIDUAL
    monkeypatch.setitem(sys.modules, "mse_eigen", stub)
    assert GM._last_residual() is None
    # The success path is the control: a module that HAS the attribute is read.
    stub.LAST_WORST_RESIDUAL = 1.5e-5
    assert GM._last_residual() == 1.5e-5


def test_the_ladder_flag_adds_exactly_one_coarser_rung(monkeypatch):
    """Item 2's instrument: a rung below 1.0, and it is opt-in."""
    _result, seen = _fake_run_case(monkeypatch)
    assert sorted(k for k, _h in seen["uniform_floors"]) == [1.0, 1.25, 1.5]
    _result, seen = _fake_run_case(monkeypatch, ladder=True)
    ks = sorted(k for k, _h in seen["uniform_floors"])
    assert ks == [GM.COARSE_RUNG, 1.0, 1.25, 1.5], ks
    assert GM.COARSE_RUNG < 1.0, "the rung must be COARSER, not finer"


def test_a_reference_that_does_not_cover_the_asked_modes_is_unmeasured(monkeypatch):
    """The gate, and its positive control.

    A reference covering one of five asked-for modes cannot answer the case's
    question, and the scorer would not complain: it matches what it can and
    reports the rest unmatched, so the verdict would read as a comparison
    against one target. The control is the same fake with a covering reference,
    which must NOT carry the gate's message.
    """
    partial = GM._clusters_pairs
    result = _fake_run_case_with(
        monkeypatch, _clusters_pairs=lambda kb: [(1.0, 0, "f", 1)])
    assert result["verdict"] == "UNMEASURED", result["verdict"]
    assert "covers 1 of the 5" in result["detail"], result["detail"]
    assert partial is not None

    control = _fake_run_case_with(monkeypatch)      # 5-fold reference
    assert "covers" not in (control.get("detail") or ""), control


def test_a_reference_build_that_is_not_a_mesh_is_named(monkeypatch):
    """The build-time half of the gate: a partial cloud, refused by name.

    ``np.zeros((0, 3))`` is the shape an incremental build leaves behind when
    its refinement is refused -- truthy but empty -- so the check has to look
    at the array, not at whether the call returned something.
    """
    import numpy as np

    def empty_uniform(materials, obj, case, h_min_mm, *, k, layers, backend,
                      np_verbose=False):
        return np.zeros((0, 3), float), np.zeros((7, 4), int), np.array([2.0])

    result = _fake_run_case_with(monkeypatch, _uniform=empty_uniform)
    assert result["verdict"] == "UNMEASURED", result["verdict"]
    assert "not a finite (n, 3) node cloud" in result["detail"], result["detail"]


def test_signed_errors_keep_the_side_of_the_reference():
    """Item 2's core: a magnitude rectifies, a sign does not."""
    # ``centre`` is the key the real scorer (run_multimaterial_bench) emits;
    # reading anything else would mean this test passes on a schema the
    # producer never writes.
    score = {"matched": [
        {"l": 1, "family": "TE", "reference_kb": 4.0, "centre": 4.04,
         "count": 3, "reference_multiplicity": 3},
        {"l": 2, "family": "TM", "reference_kb": 8.0, "centre": 7.92,
         "count": 5, "reference_multiplicity": 5},
    ]}
    rows = GM.signed_pairs(score)
    assert [round(r["signed_rel"], 6) for r in rows] == [0.01, -0.01]
    # A mode whose reference energy is zero has no relative error, and that is
    # reported as missing rather than as +inf or 0.
    assert GM.signed_pairs({"matched": [
        {"l": 1, "family": "TE", "reference_kb": 0.0, "centre": 1.0}]})[0][
        "signed_rel"] is None


def test_the_per_mode_table_joins_both_sides_and_finds_crossings():
    graded = {"matched": [{"l": 1, "family": "TE", "reference_kb": 4.0,
                           "centre": 4.0004}]}
    uniform = {"matched": [{"l": 1, "family": "TE", "reference_kb": 4.0,
                            "centre": 3.9996}]}
    rows = GM.per_mode_table(graded, uniform)
    assert len(rows) == 1 and rows[0]["reference_kb"] == 4.0
    assert rows[0]["graded_signed_rel"] > 0 > rows[0]["uniform_signed_rel"]
    assert GM.zero_crossings(rows) == rows
    # Same side: no crossing, and that must not be reported as one.
    same = GM.per_mode_table(graded, {"matched": [{"l": 1, "family": "TE",
                                                   "reference_kb": 4.0,
                                                   "centre": 4.0002}]})
    assert GM.zero_crossings(same) == []
    # A pair only one side matched cannot be a crossing: absent is not equal.
    assert GM.zero_crossings(GM.per_mode_table(graded, {"matched": []})) == []


def test_reliability_names_the_reason_it_refuses():
    """Each route to UNRELIABLE, and the control that must stay reliable."""
    rung = lambda w: {"k": 1.0, "worst_rel_error": w, "n_matched": 1}
    monotone = [rung(0.03), rung(0.02), rung(0.01)]
    assert GM.reliability(monotone, 2, []) is None
    up = [rung(0.01), rung(0.02)]
    assert "does not fall" in GM.reliability(up, 2, [])
    # The finest rung of a uniform-refined case IS the reference, so its error is
    # 0 by construction -- monotone, and invisible to this check. That is a
    # blind spot of the ladder, not a verdict; --ladder adds a coarser rung to
    # see the other side of a crossing (its own test).
    assert GM.reliability([rung(0.02), rung(0.0)], 2, []) is None
    assert "one signed" in GM.reliability(monotone, 1, [])
    cross = [{"l": 1, "family": "TE", "reference_kb": 4.0,
              "graded_signed_rel": 0.001, "uniform_signed_rel": -0.001}]
    assert "change sign" in GM.reliability(monotone, 2, cross)
    assert "did not match" in GM.reliability(
        [rung(0.03), {"k": 1.25, "worst_rel_error": None, "n_matched": 0}], 2, [])


def test_the_reachable_multiplets_are_the_ones_the_budget_can_see():
    """One target for three modes on a triplet; the spectrum's tail is not it."""
    triplet = [(4.0, 1, "TE", 3)] + [(9.0 + i, 2, "TM", 5) for i in range(6)]
    assert GM.reference_multiplets_in_reach(triplet, 3) == 1
    assert GM.reference_multiplets_in_reach(triplet, 5) == 2
    assert GM.reference_multiplets_in_reach(triplet, 40) == 7
    assert GM.reference_multiplets_in_reach([], 3) == 1     # never zero


def test_a_case_over_the_mark_on_built_counts_is_unmeasured():
    """Item 3: the mark is judged on what was BUILT, and it refuses to guess."""
    verdict, detail = GM.criterion(0.001, 0.002, None, sides_over_mark=["graded"])
    assert verdict == "UNMEASURED", (verdict, detail)
    assert "BUILT counts" in detail and "200,000" in detail
    assert GM.criterion(0.001, 0.002, None)[0] == "PASS"


def test_a_gap_under_the_references_convergence_gets_no_verdict():
    """box-sphere's shape of answer: the reference cannot resolve the gap."""
    verdict, detail = GM.criterion(0.01452, 0.01442, None,
                                   reference_convergence=0.01007)
    assert verdict == "NO VERDICT", (verdict, detail)
    assert "smaller than the reference's own convergence" in detail
    # A gap wider than the convergence is judged normally.
    assert GM.criterion(0.0300, 0.01442, None,
                        reference_convergence=0.01007)[0] == "FAIL"


def test_every_non_verdict_has_its_own_exit_code():
    """A script must not read "no answer" as "fine"."""
    codes = {GM.MATRIX_OK, GM.MATRIX_USAGE, GM.MATRIX_FAIL,
             GM.MATRIX_INCOMPLETE, GM.MATRIX_SAME_MESH, GM.MATRIX_UNRELIABLE,
             GM.MATRIX_NO_VERDICT}
    assert len(codes) == 7, codes
    assert GM.MATRIX_SAME_MESH not in (GM.MATRIX_OK, GM.MATRIX_FAIL)
    assert GM.MATRIX_NO_VERDICT not in (GM.MATRIX_OK, GM.MATRIX_INCOMPLETE)


def test_the_print_block_shows_both_sides_signed_errors(monkeypatch, capsys):
    """The new stdout path, driven through the real print block."""
    import numpy as np

    # The two sides are told apart by their SPECTRUM (the graded one comes back
    # through ``_kb`` and is tiny; the uniform rung's is 2.0), and each carries
    # the opposite SIGN about the same reference -- which is the whole point of
    # item 2 and must survive to stdout.
    def scored(kb, reference, n_requested):
        graded = float(np.asarray(kb)[0]) < 1.0
        worst = 0.0 if graded else 0.02
        centre = 4.0 * (1 + (0.0001 if graded else -0.0001))
        return {"worst_rel_error": worst, "n_clusters": 1, "n_matched": 1,
                "n_returned": 3, "spurious": [], "unmatched_reference": [],
                "matched": [{"l": 1, "family": "TE", "reference_kb": 4.0,
                             "centre": centre, "count": 3,
                             "reference_multiplicity": 3}]}

    _fake_run_case(monkeypatch)
    monkeypatch.setattr(GM.REF, "score_multiplets", scored)
    case = next(c for c in GM.MATRIX if c["label"] == "box-sphere")
    rc = GM.main(["--case", "box-sphere", "--grid", "10", "--backend", "cpu",
                  "--quiet"])
    out = capsys.readouterr().out
    assert "per-multiplet SIGNED error" in out, out
    assert "l=1 TE ref 4.0 GHz" in out, out
    assert "graded +0.0100%" in out and "uniform -0.0100%" in out, out
    # The graded side is 0.0 and the uniform 0.02, so the reading is a PASS; but
    # the reference has ONE reachable multiplet, so one signed number would
    # decide it and the lead asked for exactly that to be flagged. It must leave
    # as UNRELIABLE, not as a PASS with a number in it.
    assert rc == GM.MATRIX_UNRELIABLE, (rc, out)
    assert case["label"] == "box-sphere"


def test_the_estimator_table_reports_both_paths_against_built(monkeypatch, capsys,
                                                             tmp_path):
    """Item 5: the table, with a real result file behind the built columns."""
    import json

    (tmp_path / "c4-thin.json").write_text(json.dumps(
        {"label": "c4-thin", "n_tets_graded": 61244,
         "uniform": {"1.0": {"n_tets": 57872}}}))
    (tmp_path / "c3-thin.json").write_text(json.dumps(
        {"label": "c3-thin", "n_tets_graded": 220160,
         "uniform": {"1.0": {"n_tets": 154673}}}))
    _fake_run_case(monkeypatch)                      # patches _build
    monkeypatch.setattr(GM, "_outer_surface", lambda parts: None)
    monkeypatch.setattr("meshing.estimate_tets", lambda *a, **k: 1234)
    rc = GM.estimator_table(GM.MATRIX, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert rc == GM.MATRIX_OK, (rc, out)
    assert "est(materials=)" in out and "est(min_feature_m=)" in out
    assert "61,244" in out and "57,872" in out, out
    assert "no built counts here" in out           # the cases without a file
    assert "over the tier-2 mark 200,000 on built graded" in out, out


# --------------------------------------------------------------------------
# step 2c: the finer reference rung, the self-ladder, and one line per number
# --------------------------------------------------------------------------

def test_a_finer_reference_rung_is_finer_and_the_richardson_scale_follows_it():
    """Item 1's arithmetic: the rungs and the correction come from ONE tuple."""
    assert GM.refine_factors() == GM.REFINE_FACTORS
    assert GM.refine_factors(2.0) == (1.25, 1.5, 2.0)
    # Not finer is not a finer rung -- the option must not silently REPLACE the
    # finest rung with a coarser one, which would loosen the reference.
    with pytest.raises(ValueError, match="must be >"):
        GM.refine_factors(1.5)
    with pytest.raises(ValueError, match="must be >"):
        GM.refine_factors(1.0)
    # The default pair's documented constant still derives from the default.
    assert GM.RICHARDSON_SCALE == pytest.approx(2.27, abs=0.01)
    # A wider coarse:fine ratio means a SMALLER correction: the finer reference
    # is closer to converged, so its own error is a smaller share of the
    # measured convergence. Read from the tuple, never a constant.
    assert GM.richardson_scale((1.25, 1.5, 2.0)) == pytest.approx(
        1.0 / (1.6 ** 2 - 1.0), abs=1e-9)


def test_the_fine_reference_flag_adds_exactly_one_finer_rung(monkeypatch):
    """Item 1: the knob is opt-in and adds ONE rung through the real run_case."""
    _result, seen = _fake_run_case(monkeypatch)
    assert sorted(k for k, _h in seen["uniform_floors"]) == [1.0, 1.25, 1.5]
    result, seen = _fake_run_case(monkeypatch, fine_reference=1.75)
    ks = sorted(k for k, _h in seen["uniform_floors"])
    assert ks == [1.0, 1.25, 1.5, 1.75], ks
    # The run records the tuple it actually used, so the Richardson scale the
    # report quotes can never come from a different ladder.
    assert result["case"]["refine_factors"] == [1.25, 1.5, 1.75]


def test_the_self_ladder_scores_every_rung_against_the_finest(monkeypatch):
    """Item 2, step 2c: the ladder with the reference taken out.

    Every rung but the finest is scored against the FINEST rung's own spectrum,
    so a non-falling error can be split into "the mesh is moving" vs "the
    reference is moving". The finest rung would score against itself, so it is
    left out rather than listed as a meaningless 0.0.
    """
    import numpy as np

    seen = {}

    def pairs(kb):
        seen["reference_from"] = float(np.asarray(kb)[0])
        return [(1.0, 0, "f", 2)]

    def score(kb, reference, n_requested):
        return {"worst_rel_error": 0.005 * float(np.asarray(kb)[0]),
                "n_matched": 2}

    monkeypatch.setattr(GM, "_clusters_pairs", pairs)
    monkeypatch.setattr(GM.REF, "score_multiplets", score)
    levels = {1.0: np.array([1.0]), 1.25: np.array([2.0]), 1.5: np.array([4.0])}
    ladder = GM.self_ladder(levels, 2)
    assert [r["k"] for r in ladder] == [1.0, 1.25]
    assert all(r["against"] == 1.5 for r in ladder)
    assert seen["reference_from"] == 4.0          # built from the FINEST rung
    assert [r["self_rel_error"] for r in ladder] == [0.005, 0.01]
    # One rung is not a ladder, and says so by returning nothing rather than a
    # single row that would read as a converged mesh.
    assert GM.self_ladder({1.0: np.array([1.0])}, 2) == []


def test_the_self_ladder_note_attributes_the_non_falling_error():
    """The two readings of a non-falling ladder, and which way each points."""
    settled = [{"k": 1.25, "against": 1.5, "self_rel_error": 0.0004,
                "n_matched": 5}]
    # The two rungs agree with each other (0.040%) far better than with the
    # reference (0.430%): the mesh has converged, so the REFERENCE is what moved.
    note = GM._self_ladder_note(settled, 0.0043)
    assert "REFERENCE is what moved" in note, note
    # The rungs disagree by as much as the reference step: the mesh itself is
    # still moving and no reference at these levels can decide it.
    moving = [{"k": 1.25, "against": 1.5, "self_rel_error": 0.006,
               "n_matched": 5}]
    note = GM._self_ladder_note(moving, 0.0043)
    assert "mesh itself is still moving" in note, note
    # No self-ladder (a single level) or an unmatched step => no clause, rather
    # than a clause built on a missing number.
    assert GM._self_ladder_note([], 0.0043) is None
    assert GM._self_ladder_note(settled, None) is None
    assert GM._self_ladder_note(
        [{"k": 1.25, "against": 1.5, "self_rel_error": None}], 0.0043) is None


def test_reliability_marks_an_analytic_reference_unusable():
    """Item 2, step 2c: a closed-form reference has no finer level to converge
    into, so a non-falling ladder condemns the CASE, not merely the verdict."""
    up = [{"k": 1.0, "worst_rel_error": 0.01, "n_matched": 1},
          {"k": 1.25, "worst_rel_error": 0.02, "n_matched": 1}]
    analytic = GM.reliability(up, 2, [], reference_kind="analytic")
    assert "does not fall" in analytic and "ANALYTIC" in analytic, analytic
    # Same diagnosis for a meshed reference, WITHOUT the unusable clause: it can
    # be refined, so the case is unjudged, not condemned.
    meshed = GM.reliability(up, 2, [], reference_kind="uniform-refined")
    assert "does not fall" in meshed and "ANALYTIC" not in meshed, meshed
    # The clause attaches only to the non-falling branch: a ladder that does not
    # rise is judged, not marked unusable.
    flat = [{"k": 1.0, "worst_rel_error": 0.02, "n_matched": 1},
            {"k": 1.25, "worst_rel_error": 0.02, "n_matched": 1}]
    assert GM.reliability(flat, 2, [], reference_kind="analytic") is None


def test_the_reference_convergence_line_does_not_duplicate_the_favourable_one(
        monkeypatch, capsys):
    """Item 3 of step 2c: one number, one line.

    The favourable line states the reference's convergence inside its own
    arithmetic, so a standalone line beside it printed the same percentage
    twice -- which reads as two measurements. It is kept only for the case the
    favourable line cannot be built (a missing worst error), where it is the
    only carrier.
    """
    import numpy as np

    label = "reference's own convergence (coarse vs finest)"

    def scored(kb, reference, n_requested):
        graded = float(np.asarray(kb)[0]) < 1.0
        return {"worst_rel_error": 0.001 if graded else 0.02, "n_clusters": 3,
                "n_matched": 3, "n_returned": 3, "spurious": [],
                "unmatched_reference": []}

    _fake_run_case(monkeypatch)
    monkeypatch.setattr(GM.REF, "score_multiplets", scored)
    GM.main(["--case", "box-sphere", "--grid", "10", "--backend", "cpu",
             "--quiet"])
    out = capsys.readouterr().out
    # Step 2e replaced the favourable line with the bound against the verdict
    # actually reached; with a NO VERDICT that is the undecided line. Either way
    # it carries the convergence inside its arithmetic, so the standalone line
    # stays suppressed -- which is what this test is about.
    assert "no verdict to overturn" in out, out
    assert label not in out, out

    # The positive control: with no graded worst error the favourable line is
    # not built, and the standalone line must be its ONLY carrier -- once.
    _fake_run_case(monkeypatch)                    # the default fake: graded None
    GM.main(["--case", "box-sphere", "--grid", "10", "--backend", "cpu",
             "--quiet"])
    control = capsys.readouterr().out
    assert "most favourable reading" not in control, control
    assert control.count(label) == 1, control


def test_list_says_it_ignores_a_fine_reference(monkeypatch, capsys):
    """--list sizes the DEFAULT ladder, so the flag must not be absorbed in
    silence -- a reader could size the requested rung off the wrong row."""
    monkeypatch.setattr(GM, "list_cases", lambda cases: GM.MATRIX_OK)
    assert GM.main(["--list"]) == GM.MATRIX_OK
    assert "note:" not in capsys.readouterr().out      # control: silent unused
    assert GM.main(["--list", "--fine-reference", "1.75"]) == GM.MATRIX_OK
    out = capsys.readouterr().out
    assert "1.75 is a case-run rung" in out, out
    assert "NOT in this table" in out, out


# --------------------------------------------------------------------------
# step 2e, item 1: the bound that could OVERTURN the verdict actually reached
# --------------------------------------------------------------------------

#: The lead's own 2026-10-14 ladder numbers, as fractions. Uniform and graded
#: are the two sides' worst errors; ``C`` is the reference's own convergence
#: between its two levels.
RAMPOUT_UNIFORM, RAMPOUT_GRADED, RAMPOUT_C = 0.00779, 0.00279, 0.00096
THREEREG_UNIFORM, THREEREG_GRADED, THREEREG_C = 0.00274, 0.00221, 0.00089


def test_a_pass_that_holds_is_read_against_uniform_minus_the_convergence():
    """ramp-outward, the lead's worked example: "graded wins even if you
    SUBTRACT the reference's convergence from uniform (0.779 - 0.096 = 0.683;
    Richardson 0.779 - 0.218 = 0.561)". The uniform side is a PASS's loser, so
    the reading that could overturn it is the uniform number at its BEST."""
    b = GM.favourable_bound(RAMPOUT_UNIFORM, RAMPOUT_GRADED, RAMPOUT_C,
                            richardson=GM.RICHARDSON_SCALE)
    rows = {r["label"]: r for r in GM.adversarial_bounds(b, "PASS")}
    raw, rich = rows["raw"], rows["Richardson"]
    assert abs(raw["bound"] - 0.00683) < 1e-9, raw
    assert abs(rich["bound"] - 0.00561) < 2e-5, rich
    assert raw["flips"] is False and rich["flips"] is False, rows
    # The regression's own positive control: the OLD line read its ``still_fail``
    # flag, which is false by construction on a PASS -- so it printed "FLIPS:
    # not decidable from these runs" for a verdict that never moved.
    assert b["still_fail"] is False, b
    assert b["richardson_still_fail"] is False, b


def test_a_pass_that_flips_is_one_the_old_line_could_not_report():
    """three-region: uniform 0.274%, graded 0.221% reads PASS, but the uniform
    side credited with the reference's 0.089% convergence falls to 0.185%,
    BELOW graded -- so the PASS does not survive and the run must say so, in the
    opposite direction from the ramp-outward case above. Both printed "FLIPS:
    not decidable" before this change."""
    b = GM.favourable_bound(THREEREG_UNIFORM, THREEREG_GRADED, THREEREG_C,
                            richardson=GM.RICHARDSON_SCALE)
    rows = {r["label"]: r for r in GM.adversarial_bounds(b, "PASS")}
    assert abs(rows["raw"]["bound"] - 0.00185) < 1e-9, rows["raw"]
    assert rows["raw"]["flips"] is True, rows["raw"]
    assert rows["raw"]["other"] == "FAIL", rows["raw"]
    # the scaled reading is harsher still, and agrees in direction
    assert rows["Richardson"]["bound"] < rows["raw"]["bound"], rows
    assert rows["Richardson"]["flips"] is True, rows["Richardson"]


def test_a_fail_reads_against_uniform_plus_the_convergence():
    """A FAIL's loser is the graded side, and only the uniform number can move,
    so the overturning reading is uniform at its WORST -- the de-flattered
    number, which is the direction a same-family reference biases. A solid FAIL
    (graded far above even that) keeps its verdict; a marginal one does not."""
    holds = GM.favourable_bound(THREEREG_UNIFORM, 0.00500, THREEREG_C,
                                richardson=GM.RICHARDSON_SCALE)
    flip = GM.favourable_bound(THREEREG_UNIFORM, 0.00290, THREEREG_C,
                               richardson=GM.RICHARDSON_SCALE)
    for b, want_flip, verdict in ((holds, False, "FAIL"), (flip, True, "FAIL")):
        rows = {r["label"]: r for r in GM.adversarial_bounds(b, verdict)}
        assert abs(rows["raw"]["bound"] - (THREEREG_UNIFORM + THREEREG_C)) < 1e-9
        assert rows["raw"]["flips"] is want_flip, (b["graded_worst"], rows)
        assert rows["raw"]["other"] == "PASS"
        # the scaled reading moves the bound the same way it moves a PASS's:
        # further out, away from the measured uniform number
        assert rows["Richardson"]["bound"] > rows["raw"]["bound"]


def test_the_overturning_bound_is_verdict_specific_and_undefined_off_pass_fail():
    """One bound cannot serve both verdicts -- they sit on opposite sides of the
    measured uniform number, which is exactly why the old single line was wrong
    for one of them. A verdict with nothing to overturn gets no bound (and the
    printer says so instead of inventing one)."""
    b = GM.favourable_bound(RAMPOUT_UNIFORM, RAMPOUT_GRADED, RAMPOUT_C)
    pass_b = GM.adversarial_bounds(b, "PASS")[0]["bound"]
    fail_b = GM.adversarial_bounds(b, "FAIL")[0]["bound"]
    assert pass_b < RAMPOUT_UNIFORM < fail_b, (pass_b, fail_b)
    for verdict in ("UNRELIABLE", "NO VERDICT", "SAME MESH", "UNMEASURED"):
        assert GM.adversarial_bounds(b, verdict) == [], verdict
    assert GM.adversarial_bounds(None, "PASS") == []
    # A lower bound below zero is the reference's convergence swallowing the
    # whole uniform number: clamped at 0, and flagged so the flip is not read
    # as a measurement.
    tiny = GM.favourable_bound(0.00050, 0.00020, 0.00300)
    row = GM.adversarial_bounds(tiny, "PASS")[0]
    assert row["bound"] == 0.0 and row["raw"] < 0.0 and row["clamped"] is True, row


@pytest.mark.parametrize("verdict,uniform,graded,c,sign,word", [
    ("PASS", RAMPOUT_UNIFORM, RAMPOUT_GRADED, RAMPOUT_C, "-", "HOLDS"),
    ("PASS", THREEREG_UNIFORM, THREEREG_GRADED, THREEREG_C, "-", "FLIPS"),
    ("FAIL", THREEREG_UNIFORM, 0.00500, THREEREG_C, "+", "HOLDS"),
    ("FAIL", THREEREG_UNIFORM, 0.00290, THREEREG_C, "+", "FLIPS"),
])
def test_the_printer_names_the_verdict_the_sign_and_the_outcome(
        monkeypatch, capsys, verdict, uniform, graded, c, sign, word):
    """Each of the four sides through the REAL printer. The defect lived in the
    printed line, so a test of the arithmetic alone would not have caught it:
    the old print emitted one string -- "FLIPS: not decidable from these runs"
    -- for all four of these cases. The line must name the verdict it is trying
    to overturn, SUBTRACT for a PASS and ADD for a FAIL, and read HOLDS or FLIPS
    from the numbers."""
    rows = {r["label"]: r for r in GM.adversarial_bounds(
        GM.favourable_bound(uniform, graded, c,
                            richardson=GM.RICHARDSON_SCALE), verdict)}
    bound = rows["raw"]["bound"]
    base = {"verdict": verdict, "detail": "synthetic case for the printer",
            "meta": {},
            "case": {"favourable_uniform_bound": GM.favourable_bound(
                         uniform, graded, c, richardson=GM.RICHARDSON_SCALE),
                     "graded_score": {"worst_rel_error": graded},
                     "uniform_score": {"worst_rel_error": uniform}}}
    monkeypatch.setattr(GM, "run_case", lambda *a, **k: copy.deepcopy(base))
    code = GM.main(["--case", "box-sphere", "--grid", "10", "--backend", "cpu",
                    "--quiet"])
    out = capsys.readouterr().out
    assert f"could overturn the {verdict}" in out, out
    assert f"the {verdict} {word}" in out, out
    assert f"uniform {uniform * 100:.3f}% {sign} " in out, out
    assert f"= {bound * 100:.3f}%, against graded {graded * 100:.3f}%" in out, out
    assert code == {"PASS": GM.MATRIX_OK, "FAIL": GM.MATRIX_FAIL}[verdict]


# --------------------------------------------------------------------------
# step 2e, item 2: a mesh-to-mesh band that is noise, not convergence
# --------------------------------------------------------------------------

def test_a_rotated_build_reads_its_material_in_the_objects_own_frame():
    """A rigid rotation of the upload rotates the LAYERS with it, because the
    layers say where the material is ON the object. Rotating the mesh but
    testing it against the unrotated layer frame still runs and still returns
    numbers -- so the frame is pinned here on geometry that CAN tell the two
    apart: an outer box with an OFF-CENTRE core, and one tet whose centroid is
    the core's centre in the ROTATED frame."""
    import numpy as np
    layers = [{"n": 1.5, "kind": "box", "centre": (0.0, 0.0, 0.0),
               "half": (2.0, 2.0, 2.0)},
              {"n": 3.0, "kind": "box", "centre": (1.0, 0.0, 0.0),
               "half": (0.5, 0.5, 0.5)}]
    rot = GM._rotation_matrix(90.0, (0.0, 0.0, 1.0))
    about = np.zeros(3)
    p_rot = rot @ np.array([1.0, 0.0, 0.0]) + about       # the core's centre
    assert np.allclose(p_rot, [0.0, 1.0, 0.0]), p_rot
    points = np.asarray([p_rot * 1e-3])                   # metres, one node
    tets = np.asarray([[0, 0, 0, 0]])
    assert GM._eps_by_layers(points, tets, layers, rot=rot, about=about)[0] == 9.0
    # the positive control: the same point read with NO rotation lands in the
    # outer box, so the assertion above is not "everything is the core"
    assert GM._eps_by_layers(points, tets, layers)[0] == 2.25
    # the direction is the whole point: the rotated centroid mapped back through
    # ``@ rot`` recovers the object frame (1, 0, 0), the core -- while the
    # FORWARD rotation sends it to (-1, 0, 0), outside the core, so a transposed
    # frame is caught here instead of silently reading the outer box
    assert np.allclose(np.array([0.0, 1.0, 0.0]) @ rot, [1.0, 0.0, 0.0])
    assert np.allclose(rot @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0])


def test_the_rotation_band_is_the_spread_and_a_failure_unmeasures_it():
    """The band is max - min of the measured errors, and it is None -- not the
    spread of the survivors -- as soon as one orientation could not be measured:
    a spread over the rest would shrink toward "clean mesh" exactly when the run
    knew least. The identity build is one of the members, because it is the
    uniform number the verdict itself used."""
    ok = [{"degrees": 0.0, "worst_rel_error": 0.020},
          {"degrees": 23.0, "worst_rel_error": 0.025},
          {"degrees": 47.0, "worst_rel_error": 0.018}]
    band, note = GM.rotation_band(ok)
    assert abs(band - 0.007) < 1e-12, band
    # the note names the extremes by their OWN degree: 1.800% is the low (47
    # deg) and 2.500% the high (23 deg) -- not the first member
    assert "47 deg" in note and "23 deg" in note, note
    assert "1.800%" in note and "2.500%" in note, note
    # one rotation matched nothing: the band is unmeasured, and the note names
    # WHICH orientation failed
    bad = ok + [{"degrees": 71.0, "worst_rel_error": None, "n_matched": 0}]
    band, note = GM.rotation_band(bad)
    assert band is None and "band unmeasured" in note, note
    assert "71 deg" in note and "3 of 4" in note, note
    # a rotation that RAISED is the same honest answer, with the reason
    raised = ok + [{"degrees": 101.0, "error": "RuntimeError: boom"}]
    band, note = GM.rotation_band(raised)
    assert band is None and "RuntimeError: boom" in note, note
    # fewer than two members is not a spread
    assert GM.rotation_band([{"degrees": 0.0, "worst_rel_error": 0.02}])[0] is None


def test_the_rotated_upload_differs_only_in_the_vertex_coordinates():
    """``_rotate_obj`` must move the vertices and NOTHING else -- same object,
    same faces, same vertex order, extra per-vertex columns kept -- because the
    rotated bytes go back through the product's own ``load_parts``."""
    import numpy as np
    obj = (b"o wall\nv 1 0 0\nv 0 2 0.5 7\nvt 0 0\nf 1 2 1\no core\nv 3 3 3\nf 1 1 1\n")
    about = np.zeros(3)
    rot = GM._rotation_matrix(90.0, (0.0, 0.0, 1.0))
    out = GM._rotate_obj(obj, rot, about)
    lines = out.decode("utf-8").splitlines()
    assert lines[0] == "o wall" and "vt 0 0" in lines and "o core" in lines
    assert lines[-1] == "f 1 1 1" and "f 1 2 1" in lines
    v2 = [ln for ln in lines if ln.startswith("v ")][1]
    assert v2.split()[4:] == ["7"], v2          # the extra column survived
    assert np.allclose([float(x) for x in v2.split()[1:4]], [-2.0, 0.0, 0.5]), v2
    assert np.allclose(GM._obj_centroid(obj), [4.0 / 3.0, 5.0 / 3.0, 3.5 / 3.0])


def test_run_case_reports_the_band_and_the_print_shows_it(monkeypatch, capsys):
    """The wiring: ``rotations`` reaches ``run_case``, each orientation is
    scored against the SAME reference, the band is the spread of those scores,
    and it lands in the JSON. The rotation spectra differ by construction, so a
    band that ignored the orientations would read 0 here."""
    result, seen = _fake_run_case(monkeypatch, rotations=3)
    out = result["case"]
    assert len(out["uniform_rotations"]) == 4, out["uniform_rotations"]
    assert abs(out["uniform_rotation_band"] - 0.006) < 1e-12, out["uniform_rotation_band"]
    assert "4 rigid orientations" in out["uniform_rotation_note"], out["uniform_rotation_note"]
    assert out["uniform_rotations"][0]["identity"] is True
    assert [r["degrees"] for r in out["uniform_rotations"][1:]] == \
        [r[0] for r in GM.ROTATIONS[:3]]
    # every orientation used the SAME pivot: a rotation about different points
    # would MOVE the object as well as turn it, and the band would then be
    # measuring a translation
    assert len({p for p in seen["rot_pivots"]}) == 1, seen["rot_pivots"]
    assert seen["rot_n"] == 3, seen


def test_the_band_prints_as_a_percent_and_none_prints_as_unmeasured(
        monkeypatch, capsys):
    """The printer reads the JSON, so it must state the band in percent and, at
    a band of ``None``, say UNMEASURED -- a 0 here would read as "clean mesh",
    which is the one thing an unmeasured band must not look like."""
    base = {"verdict": "UNMEASURED", "detail": "graded matched nothing",
            "meta": {}, "case": {
                "uniform_rotations": [{"degrees": 0.0, "worst_rel_error": 0.020},
                                      {"degrees": 23.0, "error": "RuntimeError: boom"}],
                "uniform_rotation_band": None,
                "uniform_rotation_note": "band unmeasured -- 23 deg: RuntimeError: boom",
                "graded_score": {"worst_rel_error": 0.010},
                "uniform_score": {"worst_rel_error": 0.020}}}
    monkeypatch.setattr(GM, "run_case", lambda *a, **k: copy.deepcopy(base))
    assert GM.main(["--case", "box-sphere", "--rotations", "1",
                    "--quiet"]) == GM.MATRIX_INCOMPLETE
    printed = capsys.readouterr().out
    assert "mesh-to-mesh band: UNMEASURED" in printed, printed
    assert "2.00x the band" not in printed, printed     # no gap ratio off None
    base["case"].update({"uniform_rotation_band": 0.05,
                         "uniform_rotation_note": "max-min across 2 orientations"})
    assert GM.main(["--case", "box-sphere", "--rotations", "1",
                    "--quiet"]) == GM.MATRIX_INCOMPLETE
    printed = capsys.readouterr().out
    assert "mesh-to-mesh band: 5.000%" in printed, printed
    # gap = uniform 0.020 - graded 0.010 = 0.010, which is INSIDE the band
    assert "0.20x the larger measured band (5.000%) (within mesh noise)" in printed, \
        printed


def test_rotations_out_of_range_is_refused_not_silently_sliced(
        monkeypatch, capsys):
    """``ROTATIONS[:n]`` would quietly run fewer orientations than asked and
    report the spread as the requested one."""
    monkeypatch.setattr(GM, "run_case", lambda *a, **k: None)
    assert GM.main(["--case", "box-sphere", "--rotations", "99"]) == GM.MATRIX_USAGE
    assert f"1..{len(GM.ROTATIONS)}" in capsys.readouterr().err


# --------------------------------------------------------------------------
# step 2f item 1: the reference bound is SHARED by the two sides
# --------------------------------------------------------------------------


def _signed_rows(*pairs):
    """A ``per_mode_table``-shaped table from (graded, uniform) signed errors."""
    return [{"reference_kb": 10.0 + i, "l": 0, "family": "f",
             "graded_signed_rel": g, "uniform_signed_rel": u}
            for i, (g, u) in enumerate(pairs)]


def test_the_shared_shift_holds_where_the_one_sided_bound_flips():
    """Item 1, the lead's three-region numbers (2026-10-14).

    uniform 0.274%, graded 0.221%, the reference's own convergence 0.089%. The
    one-sided rule credits the whole convergence to the uniform side alone
    (0.274 - 0.089 = 0.185, below graded 0.221) and flips the PASS. The shared
    shift moves BOTH sides by the same d, so the gap is preserved: the PASS
    holds at both ends.
    """
    u, g, c = 0.00274, 0.00221, 0.00089
    fb = GM.favourable_bound(u, g, c)
    assert GM.adversarial_bounds(fb, "PASS")[0]["flips"] is True      # one-sided
    row = GM.shared_reference_bounds(_signed_rows((g, u)), c)[0]
    assert GM.shared_verdict_supported(row) == "PASS"
    # both signed errors are positive, so the shift cancels exactly and the two
    # ends agree to the digit -- which is the common case, not a lucky one
    assert row["ends"][0]["margin"] == pytest.approx(row["ends"][1]["margin"])
    assert row["ends"][0]["margin"] == pytest.approx(u - g)


def test_both_readings_flip_when_a_signed_error_sits_across_zero():
    """The one-sided bound clamps at 0 and flips; the shared ends disagree.

    A uniform error of 0.05% and a graded one of 0.02% with a reference
    convergence of 0.10%: the two ends of the shared shift straddle a zero
    crossing, so they disagree and NEITHER verdict is supported. This is the
    one case where the shift does NOT cancel, and it is exactly the
    zero-crossing worry item 2 measures.
    """
    u, g, c = 0.0005, 0.0002, 0.0010
    fb = GM.favourable_bound(u, g, c)
    one = GM.adversarial_bounds(fb, "PASS")[0]
    assert one["flips"] is True and one["clamped"] is True
    row = GM.shared_reference_bounds(_signed_rows((g, u)), c)[0]
    assert GM.shared_verdict_supported(row) == "NEITHER"
    assert (row["ends"][0]["margin"] > 0.0) != (row["ends"][1]["margin"] > 0.0)


def test_a_shared_bound_counts_the_rows_it_actually_paired():
    """A table where uniform matched only one of the two multiplets.

    The margin is a per-multiplet statement, so only paired rows may enter it --
    comparing one side's maximum over two multiplets against the other's over
    one is a different measurement. The count travels with the bound.
    """
    rows = [{"reference_kb": 10.0, "l": 0, "family": "f",
             "graded_signed_rel": 0.01, "uniform_signed_rel": None},
            {"reference_kb": 11.0, "l": 0, "family": "f",
             "graded_signed_rel": 0.02, "uniform_signed_rel": 0.03}]
    row = GM.shared_reference_bounds(rows, 0.001)[0]
    assert row["n_rows"] == 2 and row["n_paired"] == 1
    assert row["ends"][0]["margin"] is not None


def test_a_shared_bound_with_no_paired_row_is_not_a_bound():
    rows = [{"reference_kb": 10.0, "l": 0, "family": "f",
             "graded_signed_rel": 0.01, "uniform_signed_rel": None},
            {"reference_kb": 11.0, "l": 0, "family": "f",
             "graded_signed_rel": None, "uniform_signed_rel": 0.03}]
    row = GM.shared_reference_bounds(rows, 0.001)[0]
    assert [e["margin"] for e in row["ends"]] == [None, None]
    assert GM.shared_verdict_supported(row) is None
    assert row["weakest"] is None


# --------------------------------------------------------------------------
# step 1, 2026-10-15: the bound supports the verdict, it does not replace it
# --------------------------------------------------------------------------


def test_a_bound_that_disagrees_makes_the_case_too_close_to_call():
    """The wire from the case's own table to the verdict (lead's step 1).

    A graded mesh whose worst error is smaller than the uniform one's reads PASS
    from the two numbers alone. Here the SIGNED per-multiplet errors sit on
    opposite sides of zero, so the reference's own error -- which moves both
    sides by the same d -- puts the two meshes on opposite sides depending on
    its sign. The ends disagree, the bound supports NEITHER, and a case two
    readings place on opposite sides is not a PASS.
    """
    g, u, c = 0.0002, 0.0005, 0.0010
    rows = _signed_rows((-g, +u))
    support = GM.shared_support_for(rows, c)
    assert support == ["NEITHER", "NEITHER"], support
    verdict, detail = GM.criterion(g, u, None, shared_support=support)
    assert verdict == "TOO CLOSE TO CALL"
    assert "too close to call" in detail and "ends disagree" in detail
    # and the same two numbers with no bound in hand read exactly as before:
    # this is what the case said before the wire, not a weaker criterion
    assert GM.criterion(g, u, None)[0] == "PASS"


def test_a_bound_that_backs_the_verdict_leaves_it_alone():
    """The three-region numbers of step 2f item 1: both ends say PASS."""
    u, g, c = 0.00274, 0.00221, 0.00089
    support = GM.shared_support_for(_signed_rows((g, u)), c)
    assert support == ["PASS", "PASS"], support
    verdict, detail = GM.criterion(g, u, None, shared_support=support)
    assert verdict == "PASS"
    assert "too close" not in detail and "not decidable" not in detail


def test_the_other_side_is_a_disagreement_too():
    """A bound that supports PASS does not carry a FAIL, and says so."""
    verdict, detail = GM.with_shared_support(
        "FAIL", "graded 0.900% vs uniform 0.100%", ["PASS"])
    assert verdict == "TOO CLOSE TO CALL"
    assert "the bound supports a PASS" in detail
    assert "the one-sided rule reads FAIL" in detail


def test_an_undecidable_bound_leaves_the_verdict_and_names_what_it_rests_on():
    """No paired signed error is not a bound, so it overturns nothing."""
    verdict, detail = GM.with_shared_support("PASS", "graded 0.001% vs uniform 0.002%",
                                             [None, None])
    assert verdict == "PASS"
    assert "not decidable here" in detail and "one-sided rule alone" in detail


def test_no_bound_in_hand_is_not_a_disagreement():
    """A run with no signed table, or an analytic reference, has no bound.

    ``None`` means the caller had nothing to compare against: the verdict stands
    and nothing is added to it. Marking these too close to call would report a
    missing instrument as a finding.
    """
    assert GM.shared_support_for([], 0.001) is None
    assert GM.shared_support_for(_signed_rows((0.001, 0.002)), None) is None
    verdict, detail = GM.criterion(0.001, 0.002, None, shared_support=None)
    assert verdict == "PASS" and detail == GM.criterion(0.001, 0.002, None)[1]


# --------------------------------------------------------------------------
# step 2f item 2: the GRADED side gets its own rotation band
# --------------------------------------------------------------------------


def test_graded_rotations_build_a_band_from_rotated_uploads(monkeypatch):
    """``_graded_rotations`` rebuilds the graded mesh per orientation.

    The graded path carries its own region attributes (the mesher reads them
    off the object), so the rotation only has to reach the upload -- and the
    band is the spread of the graded score across those orientations.
    """
    import numpy as np
    seen = {"work": []}

    def fake_graded(case, obj, work, *, grid, backend, verbose):
        text = obj if isinstance(obj, bytes) else Path(obj).read_bytes()
        seen["work"].append(Path(work))
        # key the answer on the rotated BYTES, so a band that ignored them
        # would read 0; the frequency is not the graded side's own
        n = 1.0 + 0.01 * len(seen["work"])
        assert b"v " in text, "the rotated upload must still be an OBJ"
        return ({"n_tets": 100}, {"mesh": {"regions": {}}},
                np.array([n]), 0.0)

    monkeypatch.setattr(GM, "_graded", fake_graded)
    # ``_kb``'s constant is not this test's subject: pin it to identity so the
    # band arithmetic is the arithmetic of the numbers handed in, not of a
    # conversion factor.
    monkeypatch.setattr(GM, "_kb", lambda freqs, scale: np.asarray(freqs, float))
    monkeypatch.setattr(GM.REF, "score_multiplets",
                        lambda kb, ref, n_requested: {
                            "worst_rel_error": float(np.asarray(kb)[0])
                            / 100.0, "n_matched": 3})
    out = GM._graded_rotations(
        {"label": "x"}, b"o wall\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 1 1\n",
        Path("/tmp/mse-rot-test"), reference=None, scale_mm=5.0, n_modes=5,
        n=3, identity_error=0.0100, grid=10, backend="cpu", verbose=False)
    assert len(out["graded_rotations"]) == 4          # identity + 3 rotations
    assert out["graded_rotations"][0]["degrees"] == 0.0
    assert [r["degrees"] for r in out["graded_rotations"][1:]] == \
        [d for d, _a in GM.ROTATIONS[:3]]
    # the identity member reuses the run's own graded number (no second solve),
    # so only the three rotations were built
    assert len(seen["work"]) == 3
    assert all("graded-rot" in str(w) for w in seen["work"])
    # errors 0.0100 (identity), 0.0101, 0.0102, 0.0103 -> spread 0.0003
    assert out["graded_rotation_band"] == pytest.approx(0.0003)


def test_the_gap_is_read_against_the_wider_of_the_two_measured_bands(
        monkeypatch, capsys):
    """Both sides' bands, and the gap against the larger one.

    A verdict compares two numbers; a gap smaller than EITHER number's own
    spread has not been shown to exist, so the tolerance is the larger measured
    band -- the case's own number, never a constant chosen here. Here the graded
    band is the wider one, so it is the graded band that decides the reading.
    """
    # A run whose payload carries only the band fields exercises the print
    # without a build; ``modes_errors`` is absent, so the one-sided bound is the
    # fallback for the bound line.
    base = {"verdict": "NO VERDICT", "detail": "graded and uniform inside one another",
            "meta": {}, "case": {
                "uniform_score": {"worst_rel_error": 0.00814},
                "graded_score": {"worst_rel_error": 0.00559},
                "uniform_rotations": [{"degrees": 0.0, "identity": True,
                                       "worst_rel_error": 0.00814}],
                "uniform_rotation_band": 0.00078,
                "uniform_rotation_note": "spread of 2 members",
                "graded_rotations": [{"degrees": 0.0, "identity": True,
                                      "worst_rel_error": 0.00559}],
                "graded_rotation_band": 0.00200,
                "graded_rotation_note": "spread of 2 members"}}
    monkeypatch.setattr(GM, "run_case", lambda *a, **k: copy.deepcopy(base))
    assert GM.main(["--case", "box-sphere", "--rotations", "1", "--quiet"]) \
        == GM.MATRIX_NO_VERDICT
    printed = capsys.readouterr().out
    assert "graded side at rigid rotations" in printed
    assert "graded mesh-to-mesh band: 0.200%" in printed
    # 0.255% of gap / 0.200% (the wider, GRADED band) = 1.275x -> printed 1.27x
    assert "1.27x the larger measured band (0.200%)" in printed
    # 0.00255 of gap > 0.00200 of band: the gap is bigger than the noise, so
    # this line must not call it noise
    assert "outside mesh noise" in printed


# --------------------------------------------------------------------------
# step 2f item 3: the error ladder's tolerance is the case's OWN band
# --------------------------------------------------------------------------


def test_a_rise_within_the_measured_band_is_not_a_rise():
    """Item 3. A ladder rise no larger than the case's own measured band.

    Comparing two meshes of the SAME resolution moves the number by this much
    already, so a rise the same size has not been shown to be a rise -- and the
    tolerance is that case's own measured number, not a constant.
    """
    errors = [{"k": 1.0, "worst_rel_error": 0.00271},
              {"k": 1.25, "worst_rel_error": 0.00271},
              {"k": 1.5, "worst_rel_error": 0.00292}]
    assert GM.reliability(errors, 6, [], tolerance=0.00021) is None
    # the SAME ladder with no measured band is still UNRELIABLE: "small enough
    # that it might be noise" is not the same as "measured to be noise"
    assert "does not fall" in GM.reliability(errors, 6, [])


def test_a_rise_larger_than_the_band_still_falls_into_unreliable():
    errors = [{"k": 1.0, "worst_rel_error": 0.00271},
              {"k": 1.25, "worst_rel_error": 0.00271},
              {"k": 1.5, "worst_rel_error": 0.00292}]
    msg = GM.reliability(errors, 6, [], tolerance=0.00010)
    assert "does not fall" in msg
    assert "rising by more than the case's own measured mesh-to-mesh band" in msg
    assert "0.010%" in msg          # the tolerance it used is printed


def test_the_excused_rise_is_named_so_the_reading_is_checkable():
    errors = [{"k": 1.0, "worst_rel_error": 0.00100},
              {"k": 1.25, "worst_rel_error": 0.00115},     # +0.015%, within
              {"k": 1.5, "worst_rel_error": 0.00160}]      # +0.045%, above
    msg = GM.reliability(errors, 6, [], tolerance=0.00020)
    # the rise that IS counted is 0.115%->0.160%; the one excused is
    # 0.100%->0.115%, and it is named so the reader can check the comparison
    assert "does not fall" in msg
    assert "a smaller rise 0.100%->0.115% is within that band and is not counted" \
        in msg


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))