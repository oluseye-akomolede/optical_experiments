"""The V5 bench's judgement, without running the solver.

The mode list is checked against hand-worked numbers, the matching and the
verdict against hand-built peak lists. The run itself is the lead's Jetson
job.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_bench_v5", ROOT / "scripts" / "fdtd_bench_v5.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

C0 = 299792458.0


def test_the_modes_are_the_lateral_family_and_skip_zero_indices():
    # a = 0.03, b = 0.02, n = 1.5: c/2n = 99930819.3; (1,1) is
    # 99930819.3 * sqrt(1/0.03^2 + 1/0.02^2) = 6.005095e9 Hz (worked by hand:
    # sqrt(1111.11 + 2500) = 60.09252)
    lines = bench.lateral_modes(0.03, 0.02, 1.5, 1.0e10)
    assert abs(lines[0][0] - 6.005095e9) / 6.005095e9 < 1e-5
    assert lines[0][1] == [(1, 1)]
    pairs = [pair for _f, group in lines for pair in group]
    assert all(m >= 1 and p >= 1 for m, p in pairs)
    # in band: (1,1) 6.00e9, (2,1) 8.32e9, (1,2) 1.0533e10 is out
    assert pairs == [(1, 1), (2, 1)]


def test_a_square_plate_merges_its_degenerate_pair_into_one_line():
    lines = bench.lateral_modes(0.02, 0.02, 1.0, 2.0e10)
    groups = [sorted(group) for _f, group in lines]
    assert [(1, 2), (2, 1)] in groups
    assert all(len(g) == 1 or g == [(1, 2), (2, 1)] for g in groups)


def test_a_mode_with_a_node_at_the_probe_has_no_overlap():
    # (2, 1) has a node at x = a/2
    assert bench.overlap((2, 1), 1.0, 1.0, (0.3, 0.4), (0.5, 0.6)) < 1e-12
    assert bench.overlap((1, 1), 1.0, 1.0, (0.3, 0.4), (0.5, 0.6)) > 0.5


LINES = [(1.00e9, [(1, 1)]), (1.30e9, [(2, 1)]), (1.60e9, [(1, 2)])]


def test_peaks_within_tolerance_pass():
    rows, failures = bench.judge([1.004e9, 1.297e9, 1.605e9], LINES, [0, 1, 2])
    assert failures == []
    assert sorted(j for j, _p, _e in rows) == [0, 1, 2]


def test_two_peaks_cannot_both_claim_one_mode():
    # both sit near (1,1), and (2,1) is not gated, so nothing else can fail:
    # nearest matching lets them share (1,1) and passes an unexplained line
    rows, failures = bench.judge([1.001e9, 1.009e9], LINES[:2], [0])
    assert len(failures) == 1 and "matched to [(2, 1)]" in failures[0]


def test_a_gated_mode_without_a_peak_fails():
    _rows, failures = bench.judge([1.0e9, 1.6e9], LINES, [0, 1, 2])
    assert failures == ["no peak for mode [(2, 1)] at 1.3e+09 Hz"]


def test_an_ungated_mode_without_a_peak_passes():
    _rows, failures = bench.judge([1.0e9, 1.6e9], LINES, [0, 2])
    assert failures == []


def test_a_peak_no_mode_explains_fails():
    _rows, failures = bench.judge([1.0e9, 1.3e9, 1.45e9, 1.6e9], LINES,
                                  [0, 1, 2])
    assert len(failures) == 1 and "from every mode" in failures[0]


def test_a_matched_peak_outside_tolerance_fails():
    _rows, failures = bench.judge([1.0e9, 1.32e9, 1.6e9], LINES, [0, 1, 2])
    assert len(failures) == 1 and "1.54%" in failures[0]


def test_a_failure_outranks_a_setup_problem():
    assert bench.v5_verdict(["x"], ["y"]) == ("FAIL", 1)
    assert bench.v5_verdict([], ["y"]) == ("NOT A SPEC SETUP", 2)
    assert bench.v5_verdict([], []) == ("PASS", 0)


# The spec's "how the V5 bench decides" thresholds, each pinned on both sides.

def test_a_mode_at_the_overlap_gate_is_gated_and_one_just_under_is_not():
    gate = bench.EXCITABLE
    assert gate == 0.05
    assert bench.gated_modes([gate, 0.049, 0.8, 0.0]) == [0, 2]


def test_a_quarter_cell_film_is_far_thinner_and_one_percent_more_is_not():
    dx = 8e-9
    assert bench.film_problem(dx / 4, dx) is None
    problem = bench.film_problem(dx / 4 * 1.01, dx)
    assert problem is not None and "1/4 of the coarse cell" in problem


def test_a_floor_at_half_the_tolerance_passes_and_just_over_is_flagged():
    assert bench.TOLERANCE == 0.01
    assert bench.floor_problem(0.005) is None
    problem = bench.floor_problem(0.0051)
    assert problem is not None and "over half the 1% tolerance" in problem


def test_a_flagged_setup_without_failures_exits_2():
    verdict, code = bench.v5_verdict([], [bench.floor_problem(0.0051)])
    assert code == 2


def test_a_missing_solver_module_on_the_bench_host_exits_two_not_one(tmp_path):
    # the Jetson crash: the archive list left out mse_eigen.py, which V1's
    # bench_box imports. Copy what the docstring archives, minus that file,
    # and run the real script as the lead does.
    for rel in ("scripts", "tests/bench_box.py", "dockerfiles/maxwell_solver"):
        src, dst = ROOT / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "mse_eigen.py"))
        else:
            shutil.copy(src, dst)
    assert not (tmp_path / "dockerfiles/maxwell_solver/mse_eigen.py").exists()
    done = subprocess.run(
        [sys.executable, "scripts/fdtd_bench_v5.py"], cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 2, done.stdout + done.stderr
    assert "V5 DID NOT RUN" in done.stdout and "mse_eigen" in done.stdout


def test_the_docstring_archives_the_whole_solver_directory():
    archive = bench.__doc__.split("archive <sha>")[1].split("|")[0]
    assert archive.split() == ["scripts", "tests/bench_box.py", "\\",
                               "dockerfiles/maxwell_solver"]


def test_a_crash_inside_main_exits_two_not_one(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("setup blew up")
    monkeypatch.setattr(bench, "lateral_modes", boom)
    with pytest.raises(RuntimeError):         # main itself does not hide it
        bench.main([])
    assert bench.run([]) == 2
    out = capsys.readouterr()
    assert "V5 DID NOT RUN" in out.out and "setup blew up" in out.err


def test_a_solver_exception_mid_run_exits_one_not_two(monkeypatch, capsys):
    calls = []

    def solver_bug(*args, **kwargs):
        calls.append(args)
        raise FloatingPointError("solver blew up mid-run")
    monkeypatch.setattr(bench.fdtd, "simulate", solver_bug)
    assert bench.run(["--single"]) == 1
    out = capsys.readouterr()
    assert calls, "the fake solver was never reached; this tested the setup"
    assert "V5 FAIL: the solver raised" in out.out
    assert "V5 DID NOT RUN" not in out.out
    assert "solver blew up mid-run" in out.err


def test_the_script_run_as_main_turns_a_crash_into_exit_two():
    # --cells-per-wavelength 0 divides by zero inside main() before any solve
    done = subprocess.run(
        [sys.executable, "scripts/fdtd_bench_v5.py",
         "--cells-per-wavelength", "0"], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert "ZeroDivisionError" in done.stderr, done.stderr
    assert done.returncode == 2, done.stdout + done.stderr
    assert "V5 DID NOT RUN" in done.stdout


def test_the_upload_layout_path_builds_v5s_film_case_exactly():
    # The lead's 5.4d ask: V5's film case (a quarter-cell n = 2 film filling a
    # metal plate pair) laid out as an uploaded part must solve on the same
    # grid and give the same trace as V5's hand-built construction. A full
    # 16 x 11 run at 2 cells/wavelength, 28k steps, matched every V5 line to
    # the same frequency on both paths (-0.30% .. -1.24% from analytic); that
    # table compares the two paths at a coarse 2 cells per wavelength and is
    # not an accuracy result. Here 400 steps, where a pec mask eating one wall
    # cell already moves the trace.
    import numpy as np
    import trimesh
    sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))
    import fdtd
    import fdtd_layout

    A, B, n, dx = 16, 11, 2.0, 1e-6
    a, b, film = A * dx, B * dx, 0.25 * dx
    nodes, spacings, eps = bench.film_grid(A, B, dx, film, n * n)
    z = nodes[2]
    k = (z.size - 1) // 2
    src, prb = (6, 3, k), (11, 7, max(k - 1, 0), "z")
    centre, bw = 1.2e13, 4e12

    part = trimesh.creation.box(extents=(a, b, film))
    part.apply_translation((a / 2, b / 2, film / 2))
    request = {"boundary": "metal", "source_kind": "point", "profile": "pulse",
               "frequency_hz": centre, "bandwidth": bw,
               "position": [src[0] * dx, src[1] * dx, float(z[src[2]])],
               "direction": None, "polarisation": "z", "samples": None,
               "sample_duration_s": None, "duration_s": 1e-13,
               "probes": [[prb[0] * dx, prb[1] * dx, float(z[prb[2]]), "z"]],
               "dx_m": dx}
    lay = fdtd_layout.layout(["film"], [part], request, length_scale=1.0,
                             n_index=n)
    assert lay["shape"] == eps.shape
    for got, want in zip(lay["nodes"], nodes):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(lay["eps"]), eps)
    assert tuple(lay["cell"]) == src and tuple(lay["probes"][0]) == prb

    def trace(eps, spacings, pec):
        source = fdtd.Source(src, component="z", frequency=centre, bandwidth=bw)
        r = fdtd.simulate(np.asarray(eps, np.float32), source, 400, [prb],
                          spacings=spacings, boundary="metal", pec=pec)
        return np.asarray(r["traces"][0], float)

    hand = trace(eps, spacings, None)
    assert np.abs(hand).max() > 0
    np.testing.assert_array_equal(trace(lay["eps"], lay["spacings"], lay["pec"]), hand)
