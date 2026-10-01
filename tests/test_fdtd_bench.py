"""The V1 bench's gate against the cavity solver, without running either solver.

compare_with_cavity is the only judgement in --compare-cavity; the solvers it
compares are the lead's Jetson runs. These fixtures are analytic box modes, so
the indices line up the way the real lists do (ascending, degenerate repeated).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_bench_v1", ROOT / "scripts" / "fdtd_bench_v1.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

# k for a 2 x 3 x 5 box (arbitrary units), with one degenerate pair repeated,
# the shape analytic_box_k returns.
MODES = np.array([10.0, 12.0, 12.0, 15.0])


def freq(k):
    return bench.fdtd.C0 * k / (2.0 * math.pi)


def keyed(peaks):
    """Peaks matched one-to-one to the distinct keys, as main() does."""
    keys = sorted(set(round(float(k), 9) for k in MODES))
    return bench.match_peaks(peaks, keys)


def test_matching_peaks_agree_with_the_cavity_solver():
    peaks = [freq(10.0) * 1.002, freq(12.0) * 0.998, freq(15.0) * 1.001]
    cavity = [freq(10.0), freq(12.0) * 0.999, freq(12.0) * 1.001, freq(15.0)]
    rows, problems = bench.compare_with_cavity(
        peaks, keyed(peaks), MODES, cavity, 1.0, 1.0)
    assert problems == []
    assert len(rows) == 3
    # the degenerate pair is compared with its mean, not with either member
    assert rows[1][1] == (cavity[1] + cavity[2]) / 2


def test_a_peak_two_percent_from_the_cavity_solver_fails():
    peaks = [freq(10.0), freq(12.0), freq(15.0)]
    cavity = [freq(10.0), freq(12.0), freq(12.0), freq(15.0) * 1.02]
    rows, problems = bench.compare_with_cavity(
        peaks, keyed(peaks), MODES, cavity, 1.0, 1.0)
    assert len(rows) == 3
    assert len(problems) == 1 and "from the cavity solver" in problems[0]


def test_a_short_cavity_list_fails_rather_than_comparing_fewer_rows():
    peaks = [freq(10.0), freq(12.0), freq(15.0)]
    cavity = [freq(10.0), freq(12.0), freq(12.0)]          # 15 is missing
    rows, problems = bench.compare_with_cavity(
        peaks, keyed(peaks), MODES, cavity, 1.0, 1.0)
    assert len(rows) == 2
    assert any("returned 3 mode(s) for 4" in p for p in problems)
    assert any("no cavity frequency" in p for p in problems)
