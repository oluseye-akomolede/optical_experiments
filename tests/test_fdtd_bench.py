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


_spec2 = importlib.util.spec_from_file_location(
    "fdtd_bench_v2", ROOT / "scripts" / "fdtd_bench_v2.py")
bench2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(bench2)


def test_the_v2_fit_reads_the_amplitude_not_the_peak_sample():
    """A tone sampled off its crests, with an offset, still fits its amplitude.

    Four samples per period at a phase of pi/4 put every sample at |cos| =
    0.707, so max(|trace|) reads 0.49 of a 0.7 tone; the fit must not.
    """
    omega = 2.0 * math.pi
    t = np.arange(16) / 4.0
    trace = 0.7 * np.cos(omega * t + math.pi / 4) + 0.05
    assert np.abs(trace - 0.05).max() < 0.5
    assert abs(bench2.amplitude(trace, t, omega) - 0.7) < 1e-9


def test_v2_falloff_is_flat_for_1_over_r_and_off_for_a_steeper_power():
    radii = list(np.linspace(1.0, 3.0, 9))
    flat = bench2.falloff(radii, [2.0 / r for r in radii])
    assert max(abs(v - 1.0) for v in flat) < 1e-12
    # 1 / r^1.1 over a factor of 3 is 3^-0.1 = 0.896 at the far end: a 10%
    # drift the 5% gate must see, read from the nearest probe in full.
    steep = bench2.falloff(radii, [2.0 / r ** 1.1 for r in radii])
    assert abs(steep[-1] - 3.0 ** -0.1) < 1e-12


def test_v2_by_default_gates_only_probes_the_spec_puts_in_the_far_field():
    """The spec's V2 probes sit at r >= 2 wavelengths; the default must too.

    Read against the spec's own floor, so a default of 1 wavelength (where the
    near field still bends r * |E|) cannot pass as the spec's benchmark.
    """
    spec = (ROOT / "openspec" / "changes" / "time-domain-fdtd" / "specs"
            / "time-domain" / "spec.md").read_text()
    assert "r ≥ 2λ" in spec
    args = bench2.build_parser().parse_args([])
    assert args.r_min >= 2.0, f"V2's default --r-min is {args.r_min}, under 2"
    assert args.r_max > args.r_min


VERDICT = dict(worst=0.4, n_peaks=5, found=5, excitable=5, ungated=0,
               tolerance=1.0, dispersion=0.002)


def test_a_clean_v1_run_passes_and_exits_zero():
    verdict, line, code = bench.v1_verdict([], False, **VERDICT)
    assert (verdict, code) == ("PASS", 0)
    assert line.startswith("V1 PASS")


def test_a_widened_v1_run_is_not_reported_as_a_spec_pass():
    """Spec: "a widened run does not report a spec pass" -- not in the word,
    not in the exit code, even when every peak is inside the widened gate."""
    verdict, line, code = bench.v1_verdict(
        [], True, **dict(VERDICT, tolerance=1.6))
    assert "PASS" not in line.replace("NOT A SPEC PASS", "")
    assert "NOT A SPEC PASS" in line
    assert verdict != "PASS"
    assert code != 0
    failed = bench.v1_verdict(["worst error 2% is over 1.6%"], True,
                              **dict(VERDICT, tolerance=1.6))
    assert failed[0] == "FAIL" and failed[2] == 1


def test_the_v1_verdict_counts_the_modes_left_ungated():
    for ungated in (0, 2):
        _verdict, line, _code = bench.v1_verdict(
            [], False, **dict(VERDICT, ungated=ungated))
        assert (f"{ungated} mode(s) under the {bench.EXCITABLE} coupling "
                f"threshold, not gated") in line
