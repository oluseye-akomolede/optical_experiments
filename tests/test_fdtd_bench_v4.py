"""The V4 bench's judgement, without running the solver.

The analytic side is checked against limits that do not come from the same
formula (a half-space, a half-wave layer, a lossless film's energy balance);
the verdict against hand-built spectra. The run itself is the lead's Jetson
job.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_bench_v4", ROOT / "scripts" / "fdtd_bench_v4.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
fdtd = bench.fdtd

F = np.array([2.0e14, 3.0e14, 4.5e14])


def test_a_thick_lossy_slab_reflects_like_a_half_space():
    eps = np.array([-20 + 2j, -10 + 1j, -5 + 1.5j])
    n = np.sqrt(eps)
    half_space = np.abs((1 - n) / (1 + n)) ** 2
    assert np.allclose(bench.slab_reflectance(eps, 2e-6, F), half_space,
                       rtol=0, atol=1e-9)


def test_a_half_wave_dielectric_layer_is_invisible_and_a_quarter_wave_is_not():
    n, f = 2.0, 3.0e14
    wavelength = fdtd.C0 / f
    assert bench.slab_reflectance(n * n, wavelength / (2 * n), f) < 1e-20
    quarter = bench.slab_reflectance(n * n, wavelength / (4 * n), f)
    assert math.isclose(quarter, ((1 - n * n) / (1 + n * n)) ** 2,
                        rel_tol=1e-9)


def test_the_index_branch_decays_into_the_slab():
    # exp(-i w t): a decaying wave needs Im n > 0, for loss and for a metal.
    # complex(-3, -0.0): a lossless metal written with a signed zero; the
    # principal root of it is -1.73j, the growing branch. (-3 - 0.0j in
    # Python has a +0.0 imaginary part and never reaches the flip.)
    n = bench.refractive_index(np.array([-20 + 2j, 4 + 0.5j,
                                         complex(-3, -0.0)]))
    assert np.all(n.imag >= 0) and n.imag[2] > 1.7
    eps = np.array([-20 + 2j])
    depth = bench.skin_depth(eps, np.array([3e14]))
    assert math.isclose(float(depth[0]),
                        fdtd.C0 / (2 * math.pi * 3e14 * np.sqrt(eps).imag[0]),
                        rel_tol=1e-12)


def test_the_band_and_the_tail_read_what_they_say():
    amplitude = np.array([0.0, 0.009, 0.01, 0.5, 1.0, 0.02, 0.001])
    assert bench.band(amplitude).tolist() == [False, False, True, True, True,
                                              True, False]
    assert not bench.band(np.zeros(5)).any()
    trace = np.zeros(100)
    trace[10] = -2.0
    trace[98] = 0.004
    assert math.isclose(bench.tail(trace), 0.002)
    trace[94] = 1.0                     # just outside the last 5%: not a tail
    assert math.isclose(bench.tail(trace), 0.002)
    assert bench.tail(np.zeros(10)) == math.inf


def _verdict(measured, tails=None):
    frequencies = np.array([1e14, 2e14, 3e14, 4e14])
    analytic = np.array([0.9, 0.8, 0.7, 0.6])
    in_band = np.array([False, True, True, False])
    tails = tails or {"incident": 1e-5, "reflected": 1e-5}
    return bench.v4_verdict(frequencies, np.asarray(measured), analytic,
                            in_band, tails)


def test_the_verdict_gates_the_worst_in_band_error_only():
    passed, failures, worst, where = _verdict([0.0, 0.81, 0.715, 0.0])
    assert passed and not failures
    assert math.isclose(worst, 0.015) and where == 3e14
    # Out-of-band bins are ignored however wrong they are; in band, 0.021 fails.
    passed, failures, worst, where = _verdict([0.9, 0.821, 0.7, 0.6])
    assert not passed and math.isclose(worst, 0.021) and where == 2e14
    assert "misses Fresnel by 0.0210" in failures[0]


def test_the_verdict_fails_a_cut_off_trace_and_a_non_finite_ratio():
    passed, failures, _, _ = _verdict([0.9, 0.8, 0.7, 0.6],
                                      {"incident": 1e-5, "reflected": 0.01})
    assert not passed and "reflected trace still holds" in failures[0]
    passed, failures, _, _ = _verdict([0.9, math.nan, 0.7, 0.6])
    assert not passed and "not finite" in failures[0]


def test_a_band_outside_the_metal_fit_exits_two_before_any_run(capsys,
                                                              monkeypatch):
    def no_run(*args, **kwargs):
        raise AssertionError("simulate ran for a refused band")
    monkeypatch.setattr(fdtd, "simulate", no_run)
    assert bench.main(["--centre-ev", "0.05"]) == 2
    assert "refused" in capsys.readouterr().out
