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


def _fake_cell(shift, order=1):
    """A run_cell that returns the pc bands (crystal) or the light lines
    (homogeneous), each multiplied by ``1 + shift``. A crystal's shift is the
    one at 48 cells and scales as (48 / cells) ** ``order``, so ``order=1``
    converges at first order and ``order=0`` sits still."""
    def run_cell(eps, cells, a, k, polarisation, band, periods, xp=None):
        if eps.std() == 0.0:
            lines = bench.homogeneous_lines(k, float(math.sqrt(eps.flat[0])), band)
            error = shift
        else:
            lines = bench.crystal_lines(k, polarisation, band, 16)
            error = shift * (48 / cells) ** order
        return [f * (1 + error) for f, _ in lines], 1, 0.0
    return run_cell


@pytest.mark.parametrize("shift, code", [(0.004, 0), (0.02, 1)])
def test_the_verdict_passes_inside_one_percent_and_fails_outside(
        monkeypatch, capsys, shift, code):
    monkeypatch.setattr(bench, "run_cell", _fake_cell(shift))
    assert bench.run(["--cells", "48", "--coarse-cells", "32", "--divisions",
                      "16", "--k", "0.5", "0", "--k", "0.3", "0.2"]) == code
    out = capsys.readouterr().out
    assert ("V6 PASS" in out) == (code == 0), out
    assert "crystal" in out and "homogeneous" in out


def test_a_crystal_under_one_percent_that_does_not_converge_fails(
        monkeypatch, capsys):
    """The reviewer's case: every line 0.6% off at both resolutions is under
    the 1% tolerance, but nothing shows it approaching the pc value."""
    monkeypatch.setattr(bench, "run_cell", _fake_cell(0.006, order=0))
    assert bench.run(["--cells", "48", "--coarse-cells", "32", "--divisions",
                      "16", "--skip-homogeneous", "--k", "0.5", "0"]) == 1
    out = capsys.readouterr().out
    assert "does not approach the pc value" in out, out
    assert "is +0.60% from it" not in out   # it is under 1%: the gate is new


def test_the_same_offset_converging_at_first_order_passes(monkeypatch, capsys):
    monkeypatch.setattr(bench, "run_cell", _fake_cell(0.006, order=1))
    assert bench.run(["--cells", "48", "--coarse-cells", "32", "--divisions",
                      "16", "--skip-homogeneous", "--k", "0.5", "0"]) == 0
    assert "V6 PASS" in capsys.readouterr().out


def test_polarisation_runs_the_crystal_in_that_polarisation_only(
        monkeypatch, capsys):
    seen = []
    fake = _fake_cell(0.006)

    def run_cell(eps, cells, a, k, polarisation, *rest, **kw):
        seen.append(polarisation)
        return fake(eps, cells, a, k, polarisation, *rest, **kw)
    monkeypatch.setattr(bench, "run_cell", run_cell)
    assert bench.run(["--cells", "48", "--coarse-cells", "32", "--divisions",
                      "16", "--skip-homogeneous", "--polarisation", "TE",
                      "--k", "0.5", "0"]) == 0
    assert seen == ["TE", "TE"], seen     # both resolutions, TE only


def test_the_crystal_eps_is_sampled_finely_enough_at_the_rod_edge(monkeypatch):
    """3 samples per cell edge aliased TE band 1 at X to a false plateau."""
    calls = []
    real = bench.fdtd_grid.average_eps

    def spy(*args, **kw):
        calls.append(kw.get("samples"))
        return real(*args, **kw)
    monkeypatch.setattr(bench.fdtd_grid, "average_eps", spy)
    bench.cell_eps(8, 1e-3)
    assert calls == [bench.EPS_SAMPLES] and bench.EPS_SAMPLES >= 9


LINES = [(0.30, ["a"]), (0.45, ["b"]), (0.60, ["c"])]


def _peaks(errors):
    return [f * (1 + e) for (f, _), e in zip(LINES, errors) if e is not None]


def test_convergence_approaching_lines_pass_with_their_first_order_limit():
    rows, failures = bench.convergence(
        LINES, _peaks([-0.015, 0.004, 0.002]), _peaks([-0.009, 0.001, 0.001]),
        32, 64)
    assert failures == []
    limits = {j: limit for j, _c, _f, limit in rows}
    assert limits[0] == pytest.approx(-0.003)   # 2 x -0.9% - (-1.5%)
    assert limits[1] == pytest.approx(-0.002)


def test_convergence_a_line_moving_away_fails_and_names_both_errors():
    _rows, failures = bench.convergence(
        LINES, _peaks([-0.005, 0.001, 0.001]), _peaks([-0.007, 0.001, 0.001]),
        32, 64)
    assert len(failures) == 1
    assert "0.30000 does not approach" in failures[0]
    assert "-0.500% at 32" in failures[0] and "-0.700% at 64" in failures[0]


def test_convergence_lines_settled_at_both_resolutions_pass():
    _rows, failures = bench.convergence(
        LINES, _peaks([0.001, -0.002, 0.0]), _peaks([0.0015, -0.0024, 0.0]),
        32, 64)
    assert failures == []


def test_convergence_one_settled_end_is_not_enough():
    """0.2% then 0.3%: the fine run left the settled band, so it fails."""
    _rows, failures = bench.convergence(
        LINES, _peaks([0.002, 0.0, 0.0]), _peaks([0.003, 0.0, 0.0]), 32, 64)
    assert len(failures) == 1 and "0.30000" in failures[0]


@pytest.mark.parametrize("missing", ["coarse", "fine"])
def test_convergence_a_line_missing_at_either_resolution_fails(missing):
    present, absent = [0.004, 0.002, 0.001], [None, 0.002, 0.001]
    coarse, fine = (absent, present) if missing == "coarse" else (present, absent)
    _rows, failures = bench.convergence(LINES, _peaks(coarse), _peaks(fine),
                                        32, 64)
    assert len(failures) == 1
    cells = 32 if missing == "coarse" else 64
    assert f"no peak at {cells} cells" in failures[0]


def test_two_close_lines_shifted_alike_keep_their_own_peaks():
    """0.8% apart, both +0.6%: closest-first pairs them crosswise (+1.4%);
    the frequency-ordered pairing keeps each peak with its line."""
    lines = [(0.78885, "a"), (0.79525, "b")]
    peaks = [f * 1.006 for f, _ in lines]
    rows, failures = bench.judge(peaks, lines, [0, 1])
    assert failures == []
    assert sorted(round(e, 6) for j, _p, e in rows if j is not None) == [0.006, 0.006]


def test_the_judge_fails_a_stray_peak_and_a_far_pair():
    lines = [(0.30, "a"), (0.60, "b")]
    _rows, failures = bench.judge([0.30, 0.45, 0.6 * 1.03], lines, [0, 1])
    assert any("0.45000 is" in f and "from every line" in f for f in failures)
    assert any("line 0.60000 b is +3.00% from it" in f for f in failures)


def test_a_coarse_peak_beyond_the_match_cap_is_no_match():
    errors = bench.line_errors([0.30 * 1.06], LINES[:1])
    assert errors == {}


@pytest.mark.parametrize("coarse", ["48", "64", "4"])
def test_coarse_cells_must_be_coarser_than_cells_and_at_least_8(capsys, coarse):
    assert bench.run(["--cells", "48", "--coarse-cells", coarse,
                      "--skip-homogeneous", "--k", "0.5", "0"]) == 2
    assert "--coarse-cells must be" in capsys.readouterr().out


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


def _hann_lobes(f0, amp, record_bin):
    """A Hann main peak and its first three sidelobes each side, as zero
    padding resolves them: 2.37, 3.47, 4.48 bins out; 31.5, 41, 48 dB down."""
    lobes = [(f0, amp)]
    for bins, db in ((2.37, 31.5), (3.47, 41.0), (4.48, 48.0)):
        for side in (-1, 1):
            lobes.append((f0 + side * bins * record_bin, amp * 10 ** (-db / 20)))
    return lobes


def test_every_hann_sidelobe_over_the_floor_is_dropped():
    peaks = _hann_lobes(0.4136, 1.0, 1e-3)
    over = [p for p in peaks if p[1] >= bench.PEAK_FLOOR]
    # the second and third sidelobes (41 dB, 0.9%; 48 dB, 0.4%) clear the
    # 0.2% floor, so the envelope, not just the first sidelobe, removes them
    assert len(over) == 7
    assert bench.drop_sidelobes(over, 1e-3) == [(0.4136, 1.0)]


def test_a_lone_far_sidelobe_is_dropped_by_the_envelope_reach():
    # with all lobes present each is dropped by its stronger neighbour, so the
    # test above cannot see the reach; here only the third sidelobe is present
    main = (0.4136, 1.0)
    for lobe in _hann_lobes(*main, 1e-3)[5:]:
        assert bench.drop_sidelobes([main, lobe], 1e-3) == [main], lobe


def test_a_weak_line_away_from_the_main_peak_is_kept():
    # TE band 2 at X, measured: 2.1% of band 1's peak, 48 bins away
    peaks = _hann_lobes(0.4136, 1.0, 1e-3) + [(0.46184, 0.0214)]
    kept = bench.drop_sidelobes(peaks, 1e-3)
    assert (0.46184, 0.0214) in kept
    assert sorted(f for f, _a in kept) == [0.4136, 0.46184]


def test_a_strong_line_three_bins_out_is_not_a_sidelobe():
    # 10 dB down three bins out is far above Hann's 41 dB there
    peaks = [(0.5, 1.0), (0.503, 0.3)]
    assert bench.drop_sidelobes(peaks, 1e-3) == peaks
