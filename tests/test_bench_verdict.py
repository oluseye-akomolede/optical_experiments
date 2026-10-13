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


def _region(part, n, target_mm, achieved_mm, tets):
    return {"part": part, "n": n, "target_edge_units": target_mm,
            "target_edge_m": target_mm * 1e-3, "achieved_edge_units": achieved_mm,
            "achieved_edge_m": (None if achieved_mm is None else achieved_mm * 1e-3),
            "unclamped_edge_m": target_mm * 1e-3, "thickness_m": 0.5e-3,
            "clamped": False, "filled_volume_m3": 1e-8, "tets": tets}


def _case(label, n_core, worst, n_tets, uniform, uniform_error=None,
          mesh=None, bands=None, remesh_matches=True, eps_agreement=0.999,
          uniform_score=None, uniform_solve_error=None):
    """One canned case: enough of the shape ``main`` reads, none of the work.

    The ``mesh`` block is the product's own ``LAST_MESH_REPORT`` as the real
    ``_run_case`` copies it into the case, because the acceptance criteria read
    it: leaving it out would make the fine-region criterion UNMEASURED for every
    case and the fixture would then be testing a shape the bench never sees.

    ``bands`` is the per-band achieved edge the report prints (core / shell /
    wall, graded and uniform). The default SHELL is deliberately COARSER than
    the core -- that is the transition the ``--solve-uniform`` run exists to
    test -- so a check that the shell is printed cannot pass on a fixture where
    the two bands happen to be equal.
    """
    fine_tets = int(n_tets * 0.05)

    def _band(core_mm, shell_mm, wall_mm):
        return {
            "core": {"n_tets": fine_tets, "edge_mm": core_mm},
            "shell": {"n_tets": 300, "edge_mm": shell_mm},
            "wall": {"n_tets": n_tets - fine_tets, "edge_mm": wall_mm},
        }

    if bands is None and n_core is not None:
        bands = {"graded": _band(0.21, 0.40, 0.62),
                 "uniform": _band(0.19, 0.19, 0.19)}
    if mesh is None:
        mesh = {
            "graded": label != "empty",
            "fine_region": 1,
            "target_edge_m": 0.19e-3, "target_edge_units": 0.19,
            "coarse_edge_m": 0.57e-3,
            "achieved_edge_m": 0.30e-3, "fine_achieved_edge_m": 0.21e-3,
            "regions": {
                "1": _region("core", n_core or 1.0, 0.19, 0.21, fine_tets),
                "0": _region("wall", 1.0, 0.57, 0.62, n_tets - fine_tets),
            },
            "caveats": [],
        }
    graded = n_core is not None
    return {
        "label": label, "n_core": n_core, "n_tets": n_tets,
        "uniform_n_tets": uniform, "uniform_error": uniform_error,
        "uniform_solve_error": uniform_solve_error,
        "tets_vs_uniform": (n_tets / uniform) if uniform else None,
        "fine_edge_units": 0.19, "elapsed_s": 1.0,
        "solver_backend": "cpu", "kb_computed": [], "reference": [],
        "reference_ideal": [], "a_over_b": 0.33, "a_over_b_ideal": 0.33,
        "a_eff_mm": 3.3, "b_eff_mm": 10.0, "n_modes_requested": 10,
        "mesh": mesh, "bands": bands,
        "graded_remesh_note": (None if remesh_matches else
                               "the rebuilt graded mesh is not the one solved"),
        "remesh_matches": remesh_matches if graded else None,
        "eps_attribution_agreement": eps_agreement if graded else None,
        "uniform_score": uniform_score,
        "score": {"worst_rel_error": worst, "n_matched": 10, "n_clusters": 20,
                  "matched": []},
    }


def _match(reference_kb, l, family, multiplicity, centre, rel_error,
           count=None, count_matches=True, truncated_at_request=False):
    """One matched multiplet, with EVERY key the printer reads.

    ``main`` prints a matched entry with ``count``, ``count_matches`` and
    ``truncated_at_request`` as well as the scored fields; a fixture that omits
    them raises inside the printer and reads as a harness failure. Supply them.
    """
    return {
        "reference_kb": reference_kb, "l": l, "family": family,
        "reference_multiplicity": multiplicity, "centre": centre,
        "rel_error": rel_error, "count": count if count is not None else multiplicity,
        "count_matches": count_matches,
        "truncated_at_request": truncated_at_request,
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
    assert rc == BENCH.BENCH_OK
    assert "empty (baseline): worst 1.000%" in text, text
    # n3 at 1.90x the baseline: inside the 2x allowance, and 34.9% of uniform.
    assert "n3: worst 1.900% (1.90x the empty sphere) -> PASS (<= 2.00x)" in text
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text
    assert "FAIL" not in text, text
    assert "EXIT 0" in text, text


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
    # The exit code carries the verdict: a FAIL that printed and still exited 0
    # would let a remote run read as success.
    assert rc == BENCH.BENCH_DEFECT, "a FAILED criterion must set the exit code"
    assert "n2: worst 1.200% (1.20x the empty sphere) -> PASS (<= 2.00x)" in text
    assert "n2: 118.4% of uniform tets -> FAIL (< 100%)" in text
    assert "n3: worst 2.400% (2.40x the empty sphere) -> FAIL (<= 2.00x)" in text
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text
    assert "EXIT 3: a criterion FAILED" in text, text


def test_no_baseline_says_unmeasured_rather_than_passing(monkeypatch, capsys,
                                                         tmp_path):
    """Running without the control case must not look like a clean result."""
    cases = [_case("n3", 3.0, 0.019, 90_000, 258_000)]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "n3", "--out",
                          str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_INCOMPLETE, "an undecidable criterion is not 0"
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
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n2", "--out",
                          str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_INCOMPLETE, "an unbuilt reference is not a pass"
    assert "tets vs uniform UNMEASURED" in text, text
    assert "MeshingError" in text
    assert "% of uniform tets -> PASS" not in text, text
    assert "EXIT 4: a criterion is UNMEASURED" in text, text


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
    assert rc == BENCH.BENCH_RAISED, "a raised case must make the run exit non-zero"
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


def test_a_fine_region_that_lost_its_edge_fails_even_when_the_averages_pass(
        monkeypatch, capsys, tmp_path):
    """The fine region's OWN tets, not the whole-mesh median.

    This is the case the whole-mesh metric cannot see: the wall holds most of
    the tets, so a dense core that came back several times coarser than it was
    asked for moves the median by nothing while the solver's own acceptance --
    which is measured on the fine region's tets -- would have refused the mesh.
    Every OTHER criterion here passes, so before the fine-region criterion
    existed this run printed a clean PASS.
    """
    def coarse_core(target_mm, achieved_mm):
        fine_tets = 4_500
        return {
            "graded": True, "fine_region": 1,
            "target_edge_m": target_mm * 1e-3, "target_edge_units": target_mm,
            "coarse_edge_m": 0.57e-3,
            "achieved_edge_m": 0.30e-3,
            "fine_achieved_edge_m": achieved_mm * 1e-3,
            "regions": {
                "1": _region("core", 3.0, target_mm, achieved_mm, fine_tets),
                "0": _region("wall", 1.0, 0.57, 0.62, 85_500),
            },
            "caveats": [],
        }

    argv = ["--cases", "empty,n3", "--out", str(tmp_path / "r.json")]
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000,
              mesh=coarse_core(0.19, 0.21)),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path, argv=argv)
    assert rc == BENCH.BENCH_OK, text
    assert "n3: fine region (core) achieved 0.21 mm" in text, text

    # Same case, but the core's own tets came back at 0.40 mm against its
    # 0.19 mm target: 2.1x, past the 1.6x the product itself refuses on.
    cases[1] = _case("n3", 3.0, 0.019, 90_000, 258_000,
                     mesh=coarse_core(0.19, 0.40))
    rc, text = _run(monkeypatch, cases, capsys, tmp_path, argv=argv)
    assert rc == BENCH.BENCH_DEFECT, (
        "a fine region at 2.11x its own target is a FAILED criterion"
    )
    assert "n3: fine region (core) achieved 0.4 mm vs its 0.19 mm target " \
           "(2.11x, allowance 1.60x) -> FAIL" in text, text
    # And the criteria whose medians DID pass must still read as passes: the
    # verdict has to name the fine region, not smear it over the whole mesh.
    assert "n3: worst 1.900% (1.90x the empty sphere) -> PASS (<= 2.00x)" in text
    assert "n3: 34.9% of uniform tets -> PASS (< 100%)" in text


def test_an_unmeasurable_fine_region_is_unmeasured_not_a_pass(
        monkeypatch, capsys, tmp_path):
    """No tets of its own is the failure mode worth naming.

    A `regions` block with the fine region absent is exactly what a mesh whose
    fine region vanished looks like, and the one thing the criterion must not do
    is treat it as "no news is good news".
    """
    mesh = {"graded": True, "fine_region": 1,
            "target_edge_m": 0.19e-3, "target_edge_units": 0.19,
            "coarse_edge_m": 0.57e-3, "achieved_edge_m": 0.30e-3,
            "fine_achieved_edge_m": None,
            "regions": {"0": _region("wall", 1.0, 0.57, 0.62, 90_000)},
            "caveats": []}
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000, mesh=mesh),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n3", "--out",
                          str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_INCOMPLETE, "an unmet criterion is not a pass"
    assert "n3: fine-region achieved edge UNMEASURED (region 1 has no tets)" \
        in text, text
    assert "fine region (core) achieved" not in text, text


# --------------------------------------------------------------------------
# --solve-uniform, the per-multiplet labelling and the shell band.
#
# These are the three things the row asked to be able to SEE: WHICH mode the
# worst error belongs to, what the coarse shell just outside the interface came
# back at, and -- with the flag -- whether the uniform mesh misses the same
# yardstick the graded one is judged against. None of them is a new criterion;
# they are evidence, and the tests below hold them to that too (a uniform
# result must NOT move the exit code).
# --------------------------------------------------------------------------


def test_solve_uniform_is_off_by_default_and_solved_only_with_the_flag(
        monkeypatch, capsys, tmp_path):
    """The uniform SOLVE is opt-in -- it is minutes of GPU on the largest mesh.

    Asserted on the ARGUMENT ``_run_case`` receives, not on the printed text:
    "not solved" is a message the report can print whether or not a solve was
    attempted, and a default that quietly turned the solve on would still print
    a number. The call record is the only thing the wording cannot fake.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n2", 2.0, 0.015, 40_000, 76_000),
        _case("n3", 3.0, 0.019, 90_000, 258_000),
    ]
    by_label = {c["label"]: c for c in cases}
    seen: dict[str, bool] = {}

    def spy(label, n_core, **kw):
        seen[label] = kw.get("solve_uniform")
        c = dict(by_label[label])
        if kw.get("solve_uniform") and c["n_core"] is not None:
            c["uniform_score"] = {
                "worst_rel_error": 0.024, "n_matched": 1, "n_clusters": 20,
                "matched": [_match(2.0, 2, "TE", 5, 2.048, 0.024)]}
        return c

    monkeypatch.setattr(BENCH, "_run_case", spy)
    monkeypatch.setattr(BENCH, "_score",
                        lambda case, key="reference": case["score"])
    argv = ["--cases", "empty,n2,n3", "--out", str(tmp_path / "r.json")]

    BENCH.main(argv)
    text = capsys.readouterr().out
    assert seen == {"empty": False, "n2": False, "n3": False}, seen
    assert "not solved (--solve-uniform off)" in text, text
    assert "uniform worst" not in text, text

    BENCH.main(argv + ["--solve-uniform"])
    text = capsys.readouterr().out
    assert seen == {"empty": True, "n2": True, "n3": True}, seen
    assert "not solved (--solve-uniform off)" not in text, text
    # the reproduction hint has to name the flag, or the log cannot be replayed
    assert "--solve-uniform" in text, text


def test_the_uniform_error_is_evidence_and_does_not_move_the_verdict(
        monkeypatch, capsys, tmp_path):
    """A uniform mesh that ALSO misses 2x must not turn the graded PASS red.

    The row's question is "is the yardstick wrong, or did grading lose it", and
    the graded mesh is judged by the SAME yardstick either way. So the uniform
    number is printed and NAMED -- "the yardstick is suspect" -- while the exit
    code stays the graded run's. If the uniform result were folded into the
    verdicts, this run would exit 3 and the two questions could not be told
    apart.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000,
              uniform_score={"worst_rel_error": 0.024, "n_matched": 10,
                             "n_clusters": 20, "matched": []}),
    ]
    argv = ["--cases", "empty,n3", "--solve-uniform", "--out",
            str(tmp_path / "r.json")]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path, argv=argv)
    assert rc == BENCH.BENCH_OK, text
    assert ("n3: uniform worst 2.400% (2.40x the empty sphere) -> also over "
            "2.00x -- the yardstick is suspect; graded 1.90x") in text, text
    assert "n3: worst 1.900% (1.90x the empty sphere) -> PASS (<= 2.00x)" in text

    # The other arm: a uniform mesh within the yardstick reads as such.
    cases[1] = _case("n3", 3.0, 0.019, 90_000, 258_000,
                     uniform_score={"worst_rel_error": 0.015, "n_matched": 10,
                                    "n_clusters": 20, "matched": []})
    rc, text = _run(monkeypatch, cases, capsys, tmp_path, argv=argv)
    assert rc == BENCH.BENCH_OK, text
    assert "n3: uniform worst 1.500% (1.50x the empty sphere) -> within 2.00x" \
        in text, text


def test_a_uniform_solve_that_raised_is_unmeasured_not_a_number(
        monkeypatch, capsys, tmp_path):
    """A reference that could not be SOLVED is a missing measurement.

    The uniform mesh built fine, so the failure is in the solve -- and the one
    thing the report must never do is print a uniform error it does not have.
    The graded verdict is untouched either way.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000,
              uniform_solve_error="RuntimeError: the GPU venv forgot cupy"),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n3", "--solve-uniform", "--out",
                          str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_OK, text
    assert ("n3: uniform mesh UNMEASURED (RuntimeError: the GPU venv forgot "
            "cupy)") in text, text
    assert "uniform worst" not in text, text


def test_the_worst_multiplet_is_named_per_case(monkeypatch, capsys, tmp_path):
    """WHICH mode is worst, for graded and uniform, with its index and kb.

    A bare "worst 2.5%" does not say whether the bad mode is the one the mesh
    under-resolved or one the scorer misidentified -- and those want opposite
    responses. The index is 1-based in reference-frequency order, so the same
    mode has the same name on every run.
    """
    graded = {
        "worst_rel_error": 0.019, "n_matched": 3, "n_clusters": 4,
        "matched": [
            _match(1.0, 1, "TE", 3, 1.010, 0.010),
            _match(1.5, 1, "TM", 3, 1.500, 0.005),
            _match(2.0, 2, "TE", 5, 2.019, 0.019),
        ]}
    uniform = {
        "worst_rel_error": 0.018, "n_matched": 2, "n_clusters": 4,
        "matched": [
            _match(2.0, 2, "TE", 5, 2.006, 0.006),
            _match(1.0, 1, "TE", 3, 1.018, 0.018),
        ]}
    n3 = _case("n3", 3.0, 0.019, 90_000, 258_000, uniform_score=uniform)
    n3["score"] = graded
    cases = [_case("empty", None, 0.010, 26_000, None), n3]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n3", "--solve-uniform", "--out",
                          str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_OK, text
    # graded's worst is the 2.0 kb mode, third in reference-frequency order
    assert ("n3: worst multiplet graded: #3 ref kb 2.0000 (l=2 TE x5), "
            "computed 2.0190, err 1.900%") in text, text
    # uniform's worst is a DIFFERENT mode (the 1.0 kb one) -- index 1
    assert ("n3: worst multiplet uniform: #1 ref kb 1.0000 (l=1 TE x3), "
            "computed 1.0180, err 1.800%") in text, text


def test_worst_multiplet_index_is_by_reference_frequency_not_input_order():
    """The index must name the mode, not its slot in a list whose order is the
    scorer's business: reversing the input must not rename it."""
    a = {"reference_kb": 1.0, "l": 1, "family": "TE",
         "reference_multiplicity": 3, "centre": 1.01, "rel_error": 0.010}
    b = {"reference_kb": 2.0, "l": 2, "family": "TE",
         "reference_multiplicity": 5, "centre": 2.05, "rel_error": 0.025}
    c = {"reference_kb": 1.5, "l": 1, "family": "TM",
         "reference_multiplicity": 3, "centre": 1.5, "rel_error": 0.005}
    fwd = BENCH._worst_multiplet({"matched": [a, b, c]})
    rev = BENCH._worst_multiplet({"matched": [c, b, a]})
    assert fwd == rev, (fwd, rev)
    assert fwd["index"] == 3 and fwd["reference_kb"] == 2.0
    assert fwd["rel_error"] == 0.025
    assert BENCH._worst_multiplet({"matched": []}) is None
    assert BENCH._worst_multiplet(None) is None


def test_the_bands_print_the_shell_not_only_the_core(monkeypatch, capsys,
                                                    tmp_path):
    """The shell must appear as its OWN band, and be the SHELL's number.

    A per-region median cannot see the first millimetre outside the interface
    (the wall beside it is coarse), so the check is that the printed shell edge
    is the shell's -- the fixture's shell is 0.40 mm against a 0.21 mm core, so
    a report that printed the core twice would read 0.21 and fail here.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n3", "--out", str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_OK, text
    assert ("n3: bands graded (median edge): core 0.21 mm/4,500 tets, "
            "shell 0.4 mm/300 tets, wall 0.62 mm/85,500 tets") in text, text
    assert ("n3: bands uniform (median edge): core 0.19 mm/4,500 tets, "
            "shell 0.19 mm/300 tets, wall 0.19 mm/85,500 tets") in text, text
    assert ("n3: eps-by-radius agrees with the product's region_of_tet on "
            "99.90% of tets") in text, text
    assert "empty: bands: n/a (one region, no interface)" in text, text


def test_bands_are_withdrawn_when_the_rebuilt_mesh_is_not_the_one_solved(
        monkeypatch, capsys, tmp_path):
    """A re-mesh that disagrees with the product's own mesh block is named.

    The bands come from a SECOND mesh of the same bytes, so they are only about
    the solve if that mesh is the one the solver used. When the rebuilt mesh
    differs the report says so rather than printing bands as if they described
    the run.
    """
    cases = [
        _case("empty", None, 0.010, 26_000, None),
        _case("n3", 3.0, 0.019, 90_000, 258_000, remesh_matches=False),
    ]
    rc, text = _run(monkeypatch, cases, capsys, tmp_path,
                    argv=["--cases", "empty,n3", "--out", str(tmp_path / "r.json")])
    assert rc == BENCH.BENCH_OK, text
    assert ("n3: WARNING graded bands are about a different mesh -- the rebuilt "
            "graded mesh is not the one solved") in text, text


def test_a_rebuilt_mesh_that_is_not_the_solved_one_is_rejected_by_the_trigger():
    """The withdrawal DECISION, not just the warning it prints.

    ``test_bands_are_withdrawn_...`` hands ``_run_case`` a case whose
    ``remesh_matches`` is already ``False``, so it exercises the warning a canned
    value asserted. This drives the trigger the report actually withdraws on:
    a re-mesh is the solved mesh only when it has the solved mesh's tet count AND
    a fine edge within the tolerance of the report's own achieved edge. Probed on
    both sides of that tolerance, so neither a widened nor a tightened band could
    pass, and on the two ways a mesh can differ (count, edge).
    """
    # The same mesh: accepted, no note.
    assert BENCH._remesh_verdict(90_000, 90_000, 0.210, 0.210) == (True, None)

    # A different element count is a different object even with the edge right.
    count_ok, count_note = BENCH._remesh_verdict(91_000, 90_000, 0.210, 0.210)
    assert count_ok is False, count_ok
    assert "91,000 tets vs 90,000" in count_note, count_note

    # 2% of 0.210 mm is 0.0042: 0.2139 (1.86%) is the same mesh, 0.2143 (2.05%)
    # is not. Both directions of the band, so its value cannot be off.
    inside, _ = BENCH._remesh_verdict(90_000, 90_000, 0.2139, 0.210)
    outside, edge_note = BENCH._remesh_verdict(90_000, 90_000, 0.2143, 0.210)
    assert inside is True, inside
    assert outside is False, outside
    assert "fine edge 0.2143 vs 0.21" in edge_note, edge_note

    # An edge that could not be measured is not a match either: "could not ask"
    # must not read as "unchanged".
    unmeasured, unmeasured_note = BENCH._remesh_verdict(90_000, 90_000, None,
                                                        0.210)
    assert unmeasured is False, unmeasured
    assert "fine edge None vs 0.21" in unmeasured_note, unmeasured_note


def test_an_empty_band_is_reported_as_empty_not_as_zero():
    """A band with no tet is a measurement (nothing was meshed there), and a
    zero edge would read as a perfectly resolved region -- the opposite."""
    bands = {
        "core": {"n_tets": 10, "edge_mm": 0.21},
        "shell": {"n_tets": 0, "edge_mm": None},
        "wall": {"n_tets": 5, "edge_mm": 0.62},
    }
    s = BENCH._fmt_bands(bands)
    assert "shell --/0 tets" in s, s
    assert "0.0 mm" not in s, s
    assert BENCH._fmt_bands(None) == "not measured"
    assert BENCH._fmt_worst(None) == "none matched"


def _regular_tet_at(edge_m: float, radius_m: float) -> np.ndarray:
    """Four vertices of a REGULAR tet (all six edges equal), centroid on +x.

    A regular tet's median edge IS its edge, so a mesh built from these has an
    arithmetic answer and a wrong band boundary shows up as a wrong number.
    """
    import numpy as np

    l = edge_m
    v = np.array([
        [0.0, 0.0, 0.0],
        [l, 0.0, 0.0],
        [l / 2, l * 3 ** 0.5 / 2, 0.0],
        [l / 2, l * 3 ** 0.5 / 6, l * (2.0 / 3.0) ** 0.5],
    ])
    return v + (np.array([radius_m, 0.0, 0.0]) - v.mean(axis=0))


def test_band_edges_measure_the_shell_by_distance_not_by_region():
    """The band split is measured from the interface, on real tets.

    The formatter tests above do not run this code at all, so this is the check
    that the bands are the tets the words say. Three regular tets sit at 0.5,
    1.5 and 3.0 mm from the origin against an interface at 1.0 mm, so the bands
    are arithmetic and the shell is the one in between -- a version that took
    the shell to be the whole wall would read n=2 and swallow it.
    """
    import numpy as np
    import mse_meshing  # noqa: F401 - the real achieved_edge_length is used

    pts = np.vstack([
        _regular_tet_at(0.2e-3, 0.5e-3),   # inside the interface -> core
        _regular_tet_at(0.4e-3, 1.5e-3),   # first mm outside -> shell
        _regular_tet_at(0.8e-3, 3.0e-3),   # far outside -> wall, beyond the shell
    ])
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    b = BENCH._band_edges_mm(pts, tets, 1.0, shell_mm=1.0)
    assert b["core"]["n_tets"] == 1, b
    assert b["shell"]["n_tets"] == 1, b
    # The wall is everything OUTSIDE the core, so it CONTAINS the shell: the
    # two tets outside the interface, edged 0.4 and 0.8, median 0.6. A wall read
    # as "beyond the shell" would be one tet at 0.8 -- a finer number for the
    # coarser half of the object, which is the mistake this pins down.
    assert b["wall"]["n_tets"] == 2, b
    assert b["core"]["edge_mm"] == pytest.approx(0.2, rel=1e-6), b
    assert b["shell"]["edge_mm"] == pytest.approx(0.4, rel=1e-6), b
    assert b["wall"]["edge_mm"] == pytest.approx(0.6, rel=1e-6), b

    # A region label decides the CORE when the mesh carries one (the graded
    # mesh does); the shell stays geometric either way, because no label can
    # say "within 1 mm of the interface".
    labelled = BENCH._band_edges_mm(pts, tets, 1.0, region_of_tet=[1, 1, 0],
                                    fine_region=1, shell_mm=1.0)
    assert labelled["core"]["n_tets"] == 2, labelled
    assert labelled["shell"]["n_tets"] == 1, labelled

    # A shell with no tet in it is an empty measurement, not an error.
    empty = BENCH._band_edges_mm(pts, tets, 10.0, shell_mm=1.0)
    assert empty["shell"]["n_tets"] == 0, empty
    assert empty["shell"]["edge_mm"] is None, empty


def test_eps_by_radius_splits_the_core_at_the_interface():
    """The uniform mesh has no region labels, so its eps comes from the radius.

    One tet inside the interface and one outside: the core must take n^2 and the
    rest vacuum, or the uniform spectrum would be solved on the wrong object and
    the whole comparison would be meaningless.
    """
    import numpy as np

    pts = np.vstack([
        _regular_tet_at(0.2e-3, 0.5e-3),   # inside
        _regular_tet_at(0.4e-3, 1.5e-3),   # outside
    ])
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    eps = BENCH._eps_by_radius(pts, tets, 1.0, 3.0)
    assert eps.tolist() == [9.0, 1.0], eps


def test_a_tet_straddling_the_interface_is_banded_by_its_centroid():
    """The band rule measures a tet's CENTROID -- and the fixture makes it bite.

    A tet big enough to sit across the interface has vertices on BOTH sides, so
    "which side" is a choice; the code takes the centroid (``_tet_radius_mm``),
    the point the eps rule is validated against. Two tets are built so every
    single-vertex rule answers differently: tet A's centroid is inside the 1.0 mm
    interface while its FURTHEST vertex is outside and its vertices' MEAN
    distance is also outside (a norm's mean is at least the centroid's, so this is
    the only way the two can disagree); tet B's centroid is outside while its
    NEAREST vertex is inside. The earlier band test's tets (0.2-0.8 mm edges at
    0.5/1.5/3.0 mm) lie wholly on one side, so they cannot see the choice at all.
    """
    import numpy as np
    import mse_meshing  # noqa: F401 - the real achieved_edge_length is used

    edge = 1.0e-3
    pts = np.vstack([
        _regular_tet_at(edge, 0.9e-3),    # centroid 0.9 mm -> inside
        _regular_tet_at(edge, 1.4e-3),    # centroid 1.4 mm -> first mm outside
    ])
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

    # The fixture must actually straddle, or it proves nothing: A's vertices span
    # the interface and their mean distance is on the far side from its centroid.
    va = np.linalg.norm(pts[:4], axis=1) / 1e-3
    vb = np.linalg.norm(pts[4:], axis=1) / 1e-3
    assert va.min() < 1.0 < va.max(), va
    assert va.mean() > 1.0, va
    assert vb.min() < 1.0 < vb.max(), vb

    # The radius IS the centroid's -- 0.9 and 1.4, not any of the vertices'.
    r = BENCH._tet_radius_mm(pts, tets)
    assert r == pytest.approx([0.9, 1.4], abs=1e-9), r

    # So the bands fall by the centroid: A core, B the shell just outside it.
    bands = BENCH._band_edges_mm(pts, tets, 1.0, shell_mm=1.0)
    assert bands["core"]["n_tets"] == 1, bands
    assert bands["shell"]["n_tets"] == 1, bands
    assert bands["wall"]["n_tets"] == 1, bands

    # And the uniform mesh's per-tet eps follows the same rule: A core, B vacuum.
    eps = BENCH._eps_by_radius(pts, tets, 1.0, 3.0)
    assert eps.tolist() == [9.0, 1.0], eps
