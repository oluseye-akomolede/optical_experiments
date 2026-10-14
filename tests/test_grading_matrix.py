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
        _o, _p, plan, _h, _l, _s = GM._build(case, case["n_modes"])
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
        # the real ``predicted_counts`` shape
        return {"h_min_mm": 0.5961,
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
EXPECTED_OVER_MARK = {
    "c3-thick": ["uniform"],             # the row's own 199,062-element mesh
    "c3-thin": ["graded", "uniform"],    # the clamp, not the mode count, sizes it
}


def test_every_case_is_sized_for_tier_2_or_listed_as_unmeasurable():
    """The row's 'each case sized to fit tier 2', locked. Also the estimator
    seam: every predicted count here comes from the product's own
    ``estimate_tets`` with the constant ``web.config`` routes on, so a broken
    import shows up as a failed estimate for all ten cases, not as a green
    suite over a dead estimator. This is the slow test in this file."""
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
