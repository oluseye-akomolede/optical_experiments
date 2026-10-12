"""The bench's ACCEPTANCE verdict, pinned without meshing or solving anything.

``benchmarks/run_multimaterial_bench.py`` prints the two claims per-region
grading makes -- the dense core's error stays within twice the empty sphere's,
and the mesh costs fewer tets than the same object meshed uniformly -- as
PASS/FAIL lines. Those lines are what the lead reads on the Orin, and a wrong
verdict there is worse than no verdict: it would either bless a regression or
fail a good mesh, and neither is visible from the JSON alone.

So both ends of the report are replaced with canned values: ``_run_case``
returns a case whose mesher and solver never ran, and ``_score`` returns the
error already computed. What is left under test is exactly the verdict logic --
the code that turns those numbers into PASS/FAIL lines and an exit code. No
TetGen, no matrix, no GPU: this is the tier-1 test of the tier-2 report.

The third case is the one that matters most. With no ``empty`` case in the run,
the error criterion cannot be decided at all -- and a verdict block that printed
only the tets half would read as if the error half had been fine. That is the
rig's signature defect ("could not ask must not look like nothing there"), so it
is asserted explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "benchmarks", _ROOT / "dockerfiles" / "maxwell_solver"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_multimaterial_bench as BENCH  # noqa: E402


def _case(label, n_core, worst, n_tets, uniform, uniform_error=None):
    """One canned case: enough of the shape ``main`` reads, none of the work."""
    return {
        "label": label, "n_core": n_core, "n_tets": n_tets,
        "uniform_n_tets": uniform, "uniform_error": uniform_error,
        "tets_vs_uniform": (n_tets / uniform) if uniform else None,
        "fine_edge_units": 0.19, "elapsed_s": 1.0,
        "solver_backend": "cpu", "kb_computed": [], "reference": [],
        "reference_ideal": [], "a_over_b": 0.33, "a_over_b_ideal": 0.33,
        "a_eff_mm": 3.3, "b_eff_mm": 10.0, "n_modes_requested": 10,
        "score": {"worst_rel_error": worst, "n_matched": 10, "n_clusters": 20,
                  "matched": []},
    }


def _run(monkeypatch, cases, capsys, tmp_path, argv=None):
    """Run ``main`` over canned cases; return (exit code, stdout)."""
    by_label = {c["label"]: c for c in cases}
    monkeypatch.setattr(BENCH, "_run_case",
                        lambda label, n_core, **kw: dict(by_label[label]))
    # ``_score`` too: the real one clusters a computed spectrum that these
    # canned cases do not have, and the verdict block is what is under test.
    monkeypatch.setattr(BENCH, "_score", lambda case, key="reference": case["score"])
    out = tmp_path / "report.json"
    rc = BENCH.main(argv or ["--cases", "empty,n2,n3", "--out", str(out)])
    return rc, capsys.readouterr().out


def test_a_good_graded_run_passes_both_criteria(monkeypatch, capsys, tmp_path):
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n2", 2.0, 0.015, 40_000, 76_000),
        _case("n3", 3.0, 0.019, 90_000, 258_000),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path)
    assert rc == 0
    assert "empty (baseline): worst 1.000%" in text, text
    # n3 at 1.90x the baseline: inside the 2x allowance, and 34.9% of uniform.
    assert "n3: worst 1.900% (1.90x the empty sphere) -> PASS (<= 2.00x)" in text
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text
    assert "FAIL" not in text, text


def test_a_bad_graded_run_fails_the_criterion_it_should(monkeypatch, capsys,
                                                       tmp_path):
    """The verdict must FAIL when a criterion is broken, and name which one.

    Two independent ways to fail, both present at once so neither can be
    satisfied by the other: n3's error is 2.4x the baseline (past the 2x
    allowance) while its mesh IS cheaper, and n2's error is fine while its mesh
    is BIGGER than uniform.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n2", 2.0, 0.012, 90_000, 76_000),
        _case("n3", 3.0, 0.024, 90_000, 258_000),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path)
    assert rc == 0, "a broken criterion is a FAIL line, not a crash"
    assert "n2: worst 1.200% (1.20x the empty sphere) -> PASS (<= 2.00x)" in text
    assert "n2: 118.4% of uniform tets -> FAIL (< 100%)" in text
    assert "n3: worst 2.400% (2.40x the empty sphere) -> FAIL (<= 2.00x)" in text
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text


def test_no_baseline_says_unmeasured_rather_than_passing(monkeypatch, capsys,
                                                         tmp_path):
    """Running without the control case must not look like a clean result."""
    cases = [_case("n3", 3.0, 0.019, 90_000, 258_000)]
    _rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                     argv=["--cases", "n3", "--out",
                           str(tmp_path / "r.json")])
    assert "NO empty-sphere baseline in this run" in text, text
    assert "UNMEASURED" in text, text
    assert "n3: worst 1.900% -- NO BASELINE to compare it against" in text
    # The tets half IS measurable without the baseline, and is still reported.
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text
    # And nothing in this run may read as an error pass.
    assert "<= 2.00x)" not in text, text


def test_a_reference_that_could_not_be_built_is_not_a_pass(monkeypatch, capsys,
                                                           tmp_path):
    """A uniform reference that raised leaves the tets criterion UNMEASURED."""
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n2", 2.0, 0.012, 40_000, None,
              uniform_error="MeshingError: would need 900,000 elements"),
    ]
    _rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                     argv=["--cases", "empty,n2", "--out",
                           str(tmp_path / "r.json")])
    assert "tets vs uniform UNMEASURED" in text, text
    assert "MeshingError" in text
    assert "% of uniform tets -> PASS" not in text, text


def test_a_case_that_raised_is_recorded_and_does_not_take_the_others_down(
        monkeypatch, capsys, tmp_path):
    """One case raising must leave the rest scored, and the exit code non-zero."""
    def canned(label, n_core, **kw):
        if label == "n2":
            raise RuntimeError("the GPU venv forgot cupy")
        worst = {"empty": 0.010, "n3": 0.019}.get(label, 0.015)
        return _case(label, n_core, worst, 26_000, None)

    monkeypatch.setattr(BENCH, "_run_case", canned)
    monkeypatch.setattr(BENCH, "_score", lambda case, key="reference": case["score"])
    rc = BENCH.main(["--cases", "empty,n2,n3", "--out",
                     str(tmp_path / "r.json")])
    text = capsys.readouterr().out
    assert rc == 1, "a raised case must make the run exit non-zero"
    assert "case n2 RAISED" in text
    assert "2/3 cases ran; 1 raised" in text
    # The surviving cases are still scored, including the one after the failure.
    assert "n3: worst 1.900%" in text, text


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Guard the test itself: nothing here may reach a real backend.

    ``main`` defaults to ``--backend gpu``. A canned ``_run_case`` never gets
    far enough to use it, but if a future edit makes the default matter, this
    test should say so rather than quietly try to initialise CUDA on the VM.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield
