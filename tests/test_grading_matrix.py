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


def _fake_run_case(monkeypatch, baseline: str = "product", ladder: bool = False):
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
        return {"obj": b"o wall\no core\n", "parts": None,
                "plan": {"0": {"edge_m": 1e-3, "unclamped_m": 1e-3}},
                "h_min_mm": 0.3368, "probe_h_min_mm": 0.7511,
                "layers": [], "scale_mm": 5.0}

    def fake_graded(case, obj, work, *, grid, backend, verbose):
        return ({"n_tets": 1000},
                {"mesh": {"regions": {"0": {"target_edge_units": 1.0}}}},
                np.array([GRADED_FREQ_HZ]), 0.5)

    def fake_uniform(materials, obj, case, h_min_mm, *, k, layers, backend,
                     np_verbose=False):
        seen.setdefault("uniform_floors", []).append((k, h_min_mm))
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
                         verbose=False, baseline=baseline, ladder=ladder)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
