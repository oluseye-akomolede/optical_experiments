"""The V6 bench: its light lines and verdict by hand, and the tier-1 run.

The light lines are checked against hand-worked |k + G| / n, the verdict
against hand-built peak lists. ``--tiny`` runs the homogeneous part for real
(about 30 s under the usual caps): it is the test that the Bloch phase maps k
to the right frequency, f a / c = |k + G| / n, below and at the zone edge.
The full run, crystal against the pc band solver, is the lead's Jetson job.
"""

import importlib.util
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_bench_v6", ROOT / "scripts" / "fdtd_bench_v6.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _freqs(lines):
    return [round(f, 5) for f, _gs in lines]


def test_a_light_line_off_the_zone_edge_is_k_plus_g_over_n():
    # k = (0.1, 0), n = 1.5, band 0.2-0.7: |k| / n = 0.0667 is below the
    # band; G = (-1, 0) gives 0.9 / 1.5 = 0.6; G = (0, +-1) give
    # hypot(0.1, 1) / 1.5 = 0.66999 as one degenerate line; (1, 0) gives
    # 1.1 / 1.5 = 0.733, above it
    lines = bench.homogeneous_lines((0.1, 0.0), 1.5, (0.2, 0.7))
    assert _freqs(lines) == [0.6, 0.66999]
    assert sorted(lines[1][1]) == [(0, -1), (0, 1)]


def test_at_the_zone_edge_k_and_k_minus_g_are_one_line():
    # X = (0.5, 0): |k| = |k - (1, 0)| = 0.5, so 0.5 / 1.5 = 1/3 twice
    lines = bench.homogeneous_lines((0.5, 0.0), 1.5, (0.2, 0.7))
    assert _freqs(lines) == [0.33333]
    assert sorted(lines[0][1]) == [(-1, 0), (0, 0)]
    # M = (0.5, 0.5): four G at hypot(0.5, 0.5) / 1.5 = 0.47140
    lines = bench.homogeneous_lines((0.5, 0.5), 1.5, (0.2, 0.7))
    assert _freqs(lines) == [0.4714]
    assert len(lines[0][1]) == 4


def test_a_general_k_has_no_degeneracy():
    # (0.3, 0.2): |k| / 1.5 = 0.24037, |k - (1, 0)| / 1.5 = 0.48534,
    # |k - (0, 1)| / 1.5 = 0.56960
    lines = bench.homogeneous_lines((0.3, 0.2), 1.5, (0.2, 0.7))
    assert _freqs(lines) == [0.24037, 0.48534, 0.5696]
    assert all(len(gs) == 1 for _f, gs in lines)


def test_the_overlap_of_one_g_is_its_cosine_at_the_best_probe():
    # one probe a quarter wavelength along x from the source: cos(pi / 2) = 0
    assert bench.overlap((0.25, 0.0), [(0, 0)], source=(0.0, 0.0),
                         probes=((1.0, 0.0),)) == pytest.approx(0.0, abs=1e-12)
    # the best of several probes counts
    assert bench.overlap((0.25, 0.0), [(0, 0)], source=(0.0, 0.0),
                         probes=((1.0, 0.0), (2.0, 0.0))) == pytest.approx(1.0)


def test_the_dispersion_floor_flags_a_coarse_grid_and_passes_a_fine_one():
    assert bench.floor_problem(6, 1.5, 0.7) is not None
    assert bench.floor_problem(48, 1.5, 0.8) is None
    # the tiny setup's own floor is under half the tolerance
    tiny = bench.TINY
    assert bench.floor_problem(tiny["cells"], tiny["index"], tiny["band"][1]) is None


def test_the_crystal_lines_are_the_pc_bands_in_the_band():
    lines = bench.crystal_lines((0.5, 0.0), "TM", (0.15, 0.8), 16)
    assert lines and all(0.15 <= f <= 0.8 for f, _ in lines)
    assert [f for f, _ in lines] == sorted(f for f, _ in lines)


def _fake_cell(shift):
    """A run_cell that returns the pc bands (crystal) or the light lines
    (homogeneous), each multiplied by ``1 + shift``."""
    def run_cell(eps, cells, a, k, polarisation, band, periods, xp=None):
        if eps.std() == 0.0:
            lines = bench.homogeneous_lines(k, float(math.sqrt(eps.flat[0])), band)
        else:
            lines = bench.crystal_lines(k, polarisation, band, 16)
        return [f * (1 + shift) for f, _ in lines], 1, 0.0
    return run_cell


@pytest.mark.parametrize("shift, code", [(0.004, 0), (0.02, 1)])
def test_the_verdict_passes_inside_one_percent_and_fails_outside(
        monkeypatch, capsys, shift, code):
    monkeypatch.setattr(bench, "run_cell", _fake_cell(shift))
    assert bench.run(["--cells", "48", "--divisions", "16",
                      "--k", "0.5", "0", "--k", "0.3", "0.2"]) == code
    out = capsys.readouterr().out
    assert ("V6 PASS" in out) == (code == 0), out
    assert "crystal" in out and "homogeneous" in out


def test_a_missing_band_fails(monkeypatch, capsys):
    real = _fake_cell(0.0)

    def drop_first(*args, **kwargs):
        peaks, steps, seconds = real(*args, **kwargs)
        return peaks[1:], steps, seconds
    monkeypatch.setattr(bench, "run_cell", drop_first)
    assert bench.run(["--cells", "48", "--divisions", "16", "--skip-crystal",
                      "--k", "0.3", "0.2"]) == 1
    assert "NOT FOUND" in capsys.readouterr().out


def test_a_missing_solver_module_on_the_bench_host_exits_two_not_one(tmp_path):
    for rel in ("scripts", "tests/bench_box.py", "dockerfiles/maxwell_solver"):
        src, dst = ROOT / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "fdtd.py"))
        else:
            shutil.copy(src, dst)
    done = subprocess.run(
        [sys.executable, "scripts/fdtd_bench_v6.py", "--tiny"], cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 2, done.stdout + done.stderr
    assert "V6 DID NOT RUN" in done.stdout


def test_the_docstring_archives_the_whole_solver_directory():
    archive = bench.__doc__.split("archive <sha>")[1].split("|")[0]
    assert archive.split() == ["scripts", "tests/bench_box.py", "\\",
                               "dockerfiles/maxwell_solver"]


def test_a_crash_inside_main_exits_two_not_one(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("setup blew up")
    monkeypatch.setattr(bench, "homogeneous_lines", boom)
    assert bench.run(["--tiny"]) == 2
    out = capsys.readouterr()
    assert "V6 DID NOT RUN" in out.out and "setup blew up" in out.err


def test_a_solver_exception_mid_run_exits_one_not_two(monkeypatch, capsys):
    calls = []

    def solver_bug(*args, **kwargs):
        calls.append(args)
        raise FloatingPointError("solver blew up mid-run")
    monkeypatch.setattr(bench.fdtd, "simulate", solver_bug)
    assert bench.run(["--tiny"]) == 1
    out = capsys.readouterr()
    assert calls, "the fake solver was never reached; this tested the setup"
    assert "V6 FAIL: the solver raised" in out.out
    assert "V6 DID NOT RUN" not in out.out


def test_tiny_bloch_runs_hit_the_exact_light_lines(capsys):
    """Tier 1, for real: homogeneous n = 1.5 at k = (0.1, 0), (0.3, 0.2),
    X and M against f a / c = |k + G| / n, every gated line found within 1%.
    A phase that ignored k, or scaled it wrongly, puts the peaks on other
    lines."""
    assert bench.run(["--tiny"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "V6 PASS: 4 run(s)" in out, out
    assert "NOT FOUND" not in out and "matches no line" not in out
