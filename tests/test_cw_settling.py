"""Tier 1: the CW arm's settling curve and its classification (change cw-settling).

The paper-1 CW arm reads F = P/P0 over the last ``--window-periods`` of its
record, and used to say whether that reading had stopped moving with one
two-point line: the same reading at two thirds of the record. On the full-size
scene that line read +32.8% at 100 periods and +7.2% at 300, which one mode
with the locator's 113 fs could not do. The arm now prints a curve (the same
reading at cuts through the record), what the window's phase alone does to the
reading, a fit of the curve, and which of four classes the record is in.

What is checked here, all on tiny grids or closed-form series:

- the four classes on exact inputs (a settled tone, one exponential, two
  exponentials, a slow exponential growth, a straight rise, a short record);
- the fit's error bars against the scatter of many noisy records, so a
  reported +- means what it says;
- the window-phase swing against its own prediction (2 x the Hann's leakage of
  the -f image x the field's quadrature ratio), on closed-form series and on
  an open vacuum box run through the solver, and the phase-locked cuts that
  keep it off the curve;
- a lossy cavity whose settling time is measured independently by its own
  ring-down: the curve's tau_s and F_inf against it;
- the new lines only ADD to the arm's output, and ``--save-series`` round-trips
  to the identical F.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "paper1_nanowire.py"


def _load():
    spec = importlib.util.spec_from_file_location("paper1_nanowire", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load()
fdtd = m.fdtd

# The closed-form records: a 330 THz drive (the product's own cw_profile, ramp
# included) sampled 64.37 steps a period -- not a whole number, so the 5% marks
# land at every phase of the drive, which is the real case.
F0 = 3.3e14
PER = 64.37
DT = 1.0 / (F0 * PER)
READING = {"volume": 1e-24, "eps_r": 1.0, "index": 1.0, "frequency": F0, "periods": 8}
NEW_LINES = ("phase ", "curve ", "fit ", "class ", "locator ", "series ")


def _drive(n):
    return np.vectorize(fdtd.cw_profile)(np.arange(n) * DT, F0)


def _series(envelope, periods=200.0, quadrature=2.0):
    """A drive and a field whose HALF-STEP average (what the delivered power
    reads) is ``envelope(t in periods) * (sin - quadrature * cos)``: the
    in-phase share is 1/quadrature exactly, so the reading follows the
    envelope and the quadrature is a knob."""
    n = int(periods * PER)
    t = np.arange(n) * DT
    w = 2.0 * math.pi * F0
    s = t + DT / 2.0
    field = (envelope(t * F0) * (np.sin(w * s) - quadrature * np.cos(w * s))
             / math.cos(w * DT / 2.0))
    return _drive(n), field


def _verdict(drive, field, reading=READING, dt=DT):
    """The classification exactly as ``settling_report`` makes it."""
    phase = m.window_phase(drive, field, dt, **reading)
    curve = m.settling_curve(drive, field, dt, **reading)
    miss = max(r["mismatch"] for r in curve["rows"] if r["fitted"])
    return m.classify_settling(curve, floor=phase["swing"] * 4 * math.pi * miss), curve


def _report(drive, field, *, locator=None, reading=READING, dt=DT):
    """``settling_report``'s printed lines and its verdict."""
    F = m.cw_reading(drive, field, dt, **reading)
    cut = 2 * np.size(field) // 3
    early = m.cw_reading(drive[:cut], field[:cut], dt, **reading)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verdict = m.settling_report(argparse.Namespace(locator_tau=locator), drive,
                                    field, dt, F=F, early=early, **reading)
    return buffer.getvalue().splitlines(), verdict


SETTLED_F = m.cw_reading(*_series(lambda p: np.ones_like(p)), DT, **READING)


# --- the four classes, on exact inputs ---------------------------------------------

def test_a_settled_tone_is_class_a():
    """A tone that never changes: the final third's line has no slope beyond
    its own error (measured 2.1e-7 of F against an error of 8.5e-7: what is
    left of the 2f swing at the cuts' phase mismatch)."""
    lines, verdict = _report(*_series(lambda p: np.ones_like(p)))
    assert verdict["kind"] == "settled", lines
    assert abs(verdict["drift"]) <= m.SIGNIFICANCE * verdict["drift_err"]
    assert any(line.startswith("class     (a) settled") for line in lines), lines


def test_one_exponential_approach_is_class_b_with_its_own_tau_and_limit():
    """F rising as 1 - 0.8 exp(-t/30 periods) over 200 periods: the Hann mean of
    an exponential is the same exponential, so the fit must return tau_s = 30
    periods and the settled tone's own F as F_inf. Measured: tau_s off by
    7.5e-7 (error 3.0e-6) and F_inf by 3.2e-8 (error 2.8e-7) of themselves --
    the 2f swing at the cuts' phase mismatch; 4e-14 on a whole-step period --
    so both are asserted within 1e-5 AND within 3 of their own errors."""
    lines, verdict = _report(*_series(lambda p: 1 - 0.8 * np.exp(-p / 30.0)))
    fit = verdict["fit"]
    assert verdict["kind"] == "approaching", lines
    assert fit["tau"] * F0 == pytest.approx(30.0, rel=1e-5)
    assert fit["F_inf"] == pytest.approx(SETTLED_F, rel=1e-5)
    assert abs(fit["tau"] - 30.0 / F0) <= m.SIGNIFICANCE * fit["tau_err"]
    assert abs(fit["F_inf"] - SETTLED_F) <= m.SIGNIFICANCE * fit["F_inf_err"]
    assert fit["A"] > 0                                # it rises towards F_inf
    text = "\n".join(lines)
    assert "EXTRAPOLATION" in text and "(b) approaching" in text, text
    # the fitted line names tau_s in fs and in periods
    assert re.search(r"tau_s = 90\.9\d* \+- \S+ fs = 30 periods", text), text


def test_two_time_constants_is_an_approach_that_says_it_holds_more_than_one():
    """1 - 0.5 exp(-t/10) - 0.3 exp(-t/60 periods): still an approach (b). The
    whole-curve fit has a tau_s between the two (measured 37.4 +- 2.2 periods)
    and a residual far above the floor (1.1e4 x), which the arm has to say;
    its F_inf sat 1.72% under the limit, below the end's own 1.14%. Since
    change settling-tail-fit the forecast is refit over the last half: tau_s
    the slow 60 periods (measured 59.91) and F_inf 3.2e-5 under the limit,
    but 3.7 of its errors, and its residual is 3.84 x the floor, so the class
    line calls it UNRELIABLE."""
    lines, verdict = _report(*_series(
        lambda p: 1 - 0.5 * np.exp(-p / 10.0) - 0.3 * np.exp(-p / 60.0)))
    whole, fit = verdict["whole"], verdict["fit"]
    assert verdict["kind"] == "approaching", lines
    assert 10.0 < whole["tau"] * F0 < 60.0
    assert whole["rms"] > m.SIGNIFICANCE * verdict["floor"]
    assert any("more than this one time constant" in line for line in lines), lines
    assert fit is not whole and fit["tau"] * F0 == pytest.approx(60.0, rel=5e-3)
    assert fit["F_inf"] == pytest.approx(SETTLED_F, rel=1e-4)
    assert verdict["unreliable"] and not verdict["inconsistent"]
    assert any("forecast is UNRELIABLE" in line for line in lines), lines


def test_a_slow_exponential_growth_is_class_c_and_never_a_purcell_factor():
    """1 + 0.02 (exp(t/100 periods) - 1): growth the approach model cannot fit.
    The fitted rate is negative by 4.3e4 of its errors, and the growth time it
    reads is the 100 periods put in (measured 6e-6 off; asserted 1e-4)."""
    lines, verdict = _report(*_series(lambda p: 1 + 0.02 * np.expm1(p / 100.0)))
    fit = verdict["fit"]
    assert verdict["kind"] == "growing", lines
    assert -fit["span"] / fit["u"] * F0 == pytest.approx(100.0, rel=1e-4)
    assert "F_inf" not in fit
    assert any("NOT a Purcell factor" in line for line in lines), lines


def test_a_straight_rise_and_a_short_record_are_class_d():
    """A straight line cannot say whether it bends later: the rate is within its
    errors of zero (measured -1.0e-5 +- 4.4e-5), so (d). A 14-period record has
    two cuts past the ramp and one window, under the five a fit needs."""
    lines, verdict = _report(*_series(lambda p: 0.9 + 0.1 * p / 200.0))
    assert verdict["kind"] == "unresolved", lines
    assert "straight" in verdict["why"]
    short, _ = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0), periods=14.0), None
    verdict = m.classify_settling(m.settling_curve(*short, DT, **READING))
    assert verdict["kind"] == "unresolved" and "record longer" in verdict["why"]


# --- the forecast follows the slow tail (change settling-tail-fit) ---------------------

# A 600-period CW run's printed curve: (cut in periods, F). Its whole-curve fit
# printed F_inf = 13.1935 +- 0.02, tau_s = 82.99 periods, residual rms 0.0552 =
# 2.83e4 x the floor -- a limit below the 13.26 the curve had already reached
# while still rising.
CURVE_600 = np.array([
    (31.0, 3.31597), (68.0, 6.9017), (84.0, 8.08599), (121.0, 9.90457),
    (158.0, 11.111), (174.0, 11.4304), (203.0, 11.9286), (248.0, 12.4086),
    (264.0, 12.5121), (293.0, 12.6826), (338.0, 12.883), (367.0, 12.974),
    (383.0, 13.0154), (428.0, 13.1109), (457.0, 13.1565), (473.0, 13.1772),
    (510.0, 13.2134), (547.0, 13.2382), (563.0, 13.2466), (600.0, 13.26)])
FLOOR_600 = 0.0552 / 2.83e4


def _classify(t, F, floor, steps):
    """``classify_settling`` on a bare curve: every cut fitted, cut = t (so the
    record's last half is ``t >= steps / 2``), and uncorrelated cuts -- the
    600-period run's cuts are 16 or more periods apart under 8-period windows,
    where ``hann_correlation`` is the identity."""
    rows = [{"t": a, "F": b, "cut": a, "fitted": True} for a, b in zip(t, F)]
    return m.classify_settling({"rows": rows, "steps": steps,
                                "corr": np.eye(len(rows))}, floor=floor)


def test_the_600_period_curve_s_forecast_is_not_below_its_end():
    """The run's own curve: the whole-curve fit is the one it printed (13.1935
    +- 0.020, 82.99 periods, 2.83e4 x the floor), and the forecast is now the
    last half's fit (10 cuts from 338 periods): F_inf = 13.3068 +- 0.0039 and
    tau_s = 115.1 periods (measured), at or above the 13.26 the curve reached.
    Its residual is still 1029 x the floor, so it is UNRELIABLE, not
    inconsistent."""
    t, F = CURVE_600[:, 0], CURVE_600[:, 1]
    verdict = _classify(t, F, FLOOR_600, 600.0)
    whole, fit = verdict["whole"], verdict["fit"]
    assert whole["F_inf"] == pytest.approx(13.1935, abs=1e-4)
    assert whole["F_inf_err"] == pytest.approx(0.020, abs=1e-3)
    assert whole["tau"] == pytest.approx(82.99, abs=1e-2)
    assert whole["rms"] / FLOOR_600 == pytest.approx(2.83e4, rel=1e-3)
    assert verdict["kind"] == "approaching"
    assert fit is not whole and fit["points"] == 10 and fit["t0"] == 338.0
    assert fit["F_inf"] >= 13.26, fit
    assert not verdict["inconsistent"] and verdict["unreliable"]


def test_the_check_says_the_old_forecast_is_inconsistent_with_the_record():
    """The whole-curve forecast checked against the record it came from: still
    rising, and F_inf 13.1935 is 3.33 of hypot(its error, the floor) below the
    end's 13.26 (measured), over INCONSISTENT_SIGMA = 2.5. The same curve
    mirrored (falling, F -> 26.52 - F) puts F_inf above its end by the same
    3.33 and is caught the same way."""
    t, F = CURVE_600[:, 0], CURVE_600[:, 1]
    for values in (F, 26.52 - F):
        verdict = _classify(t, values, FLOOR_600, 600.0)
        check = m.forecast_check(verdict["whole"], end=values[-1],
                                 drift=verdict["drift"],
                                 drift_err=verdict["drift_err"], floor=FLOOR_600)
        assert check["inconsistent"], check
        assert abs(check["gap"]) / check["gap_err"] == pytest.approx(3.33, abs=0.01)
        assert not verdict["inconsistent"]          # the tail's forecast is not


def _two_exp(seed, fast, slow):
    """1 - 0.7 exp(-t/fast) - 0.3 exp(-t/slow) at 20 cuts 30 apart (a 600-long
    record), with white noise of 1e-5 and the floor at that noise."""
    t = np.arange(1, 21) * 30.0
    F = 1.0 - 0.7 * np.exp(-t / fast) - 0.3 * np.exp(-t / slow)
    return t, F + 1e-5 * np.random.default_rng(seed).standard_normal(t.size)


@pytest.mark.parametrize("fast, slow", [(10.0, 80.0), (10.0, 300.0), (25.0, 150.0),
                                        (25.0, 300.0)])
def test_two_exponential_curves_recover_their_limit(fast, slow):
    """Sums of two exponentials with limit 1. Measured over 20 seeds x 6 shapes
    (fast 10, 25; slow 80, 150, 300): the forecast is within 2.53 of its own
    errors and 1.73e-4 of the limit, never flagged (residual at most 1.41 x the
    floor); the whole-curve fit misses by 1.6e-3 or more. Asserted: within 3
    errors and 5e-4, and the whole-curve fit at least 1e-3 off."""
    verdict = _classify(*_two_exp(0, fast, slow), 1e-5, 600.0)
    fit = verdict["fit"]
    assert verdict["kind"] == "approaching"
    assert abs(fit["F_inf"] - 1.0) <= m.SIGNIFICANCE * fit["F_inf_err"], fit
    assert abs(fit["F_inf"] - 1.0) <= 5e-4, fit
    assert abs(verdict["whole"]["F_inf"] - 1.0) >= 1e-3
    assert not verdict["inconsistent"] and not verdict["unreliable"]


def test_a_clean_single_exponential_keeps_the_whole_curve_fit():
    """No regression where one time constant holds: 1 - 0.8 exp(-t/tau) with
    1e-5 noise, tau 60, 100 and 200 over a 600-long record. Measured over 900
    such records: the whole-curve residual reached 1.44 x the floor, under the
    3 that sends a forecast to the tail, so none was refit. The forecast is
    the whole-curve fit, value for value."""
    t = np.arange(1, 21) * 30.0
    rng = np.random.default_rng(2)
    for tau in (60.0, 100.0, 200.0):
        F = 1.0 - 0.8 * np.exp(-t / tau) + 1e-5 * rng.standard_normal(t.size)
        verdict = _classify(t, F, 1e-5, 600.0)
        before = m.fit_approach(t, F, floor=1e-5, corr=np.eye(t.size))
        assert verdict["fit"] is verdict["whole"]
        assert verdict["fit"]["F_inf"] == before["F_inf"]
        assert verdict["fit"]["tau"] == before["tau"]
        assert verdict["tail_from"] == 0 and not verdict["unreliable"]


def test_correct_forecasts_are_never_called_inconsistent():
    """The false-alarm side of INCONSISTENT_SIGMA: single exponentials with
    noise from 1e-5 to 1e-3 of the limit, records 2.4 to 10 time constants
    long (the end up to within noise of the limit). Measured over 1623 such
    records: the forecast's wrong-side gap reached 1.92 of its error, under
    2.5; and over 206 noisy series through the reading (0.05 of the in-phase
    amplitude, 60 periods), 0.998. The 1800 records are run here (about
    5 s): the worst is the measured 1.92, so a threshold of 1 fails."""
    t = np.arange(1, 21) * 30.0
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(150):
        for tau in (60.0, 100.0, 150.0, 250.0):
            for noise in (1e-5, 1e-4, 1e-3):
                F = 13.5 * (1 - 0.75 * np.exp(-t / tau)) + noise * 13.5 * rng.standard_normal(t.size)
                verdict = _classify(t, F, noise * 13.5, 600.0)
                if verdict["kind"] != "approaching":
                    continue
                assert not verdict["inconsistent"], verdict
                worst = max(worst, -np.sign(verdict["drift"]) * verdict["gap"]
                            / verdict["gap_err"])
    assert worst < m.INCONSISTENT_SIGMA


def _replay_600():
    """The 600-period curve as a closed-form record, ten times faster (60
    periods, 3,862 samples): its in-phase envelope is the curve itself."""
    t, F = CURVE_600[:, 0], CURVE_600[:, 1]
    return _series(lambda p: np.interp(10.0 * p, np.r_[0.0, t], np.r_[0.0, F]) / 13.26,
                   periods=60.0)


def test_the_replayed_curve_prints_the_tail_forecast(monkeypatch):
    """Through the reading and the printed lines: the whole-curve F_inf sits
    under the end (measured 0.16% under), the refit line names the last 50%
    (11 cuts from 30 periods) and its F_inf is above the end (measured 0.56%),
    and the class line gives it as the extrapolation, UNRELIABLE (residual
    49.7 x the floor). With the refit switched off and the threshold dropped
    to 0.5 (the old forecast is 0.82 of its error under the end, measured) the
    class line says INCONSISTENT and gives no extrapolation."""
    lines, verdict = _report(*_replay_600())
    text = "\n".join(lines)
    assert verdict["whole"]["F_inf"] < verdict["end"] < verdict["fit"]["F_inf"]
    assert re.search(r"refit over the last 50% of the record \(11 cuts from 30\.0"
                     r" periods\)", text), text
    assert f"so F_inf = {verdict['fit']['F_inf']:.6g} +-" in text, text
    assert "forecast is UNRELIABLE" in text and "INCONSISTENT" not in text, text
    monkeypatch.setattr(m, "TAIL_SHARE", 1.0)
    monkeypatch.setattr(m, "INCONSISTENT_SIGMA", 0.5)
    lines, verdict = _report(*_replay_600())
    text = "\n".join(lines)
    assert verdict["inconsistent"] and "refit over" not in text, text
    assert "forecast is INCONSISTENT with the record" in text, text
    assert "EXTRAPOLATION" not in text, text


# --- the error bars -----------------------------------------------------------------

def test_the_error_bars_cover_the_truth_on_noisy_records():
    """200 records of the 30-period approach over 60 periods, each with white
    noise on the field (0.05 of its in-phase amplitude). The cuts are 3 periods
    apart under 8-period windows, so neighbouring readings share samples
    (measured correlation 0.35-0.43) and the fit's sandwich covariance has to
    allow for it. Measured over 300 records: z = (fit - truth) / error has a
    spread of 1.17 for F_inf and 1.16 for tau_s, 65-66% inside 1 and 98-99%
    inside 3 -- against 0.32/0.15 inside 1 when the cuts were treated as
    independent with a flat overlap factor.

    This noise is far above the records' phase floor, so since change
    settling-tail-fit their residual sends the forecast to the last half of
    the record (all 200 here). Both are held to the same bounds: the
    whole-curve fit (measured over 6 seeds of 200: spread 0.93-1.21, 97.5-99.5%
    inside 3) and the forecast (1.12-1.34, 96-99%; this seed is the widest)."""
    envelope = lambda p: 1 - 0.8 * np.exp(-p / 30.0)       # noqa: E731
    drive, clean = _series(envelope, periods=60.0)
    truth = _verdict(drive, clean)[0]["fit"]
    rng = np.random.default_rng(1)
    z = {"whole": ([], []), "fit": ([], [])}
    for _ in range(200):
        verdict, _curve = _verdict(drive, clean + 0.05 * rng.standard_normal(clean.size))
        for key, (z_f, z_tau) in z.items():
            fit = verdict[key]
            z_f.append((fit["F_inf"] - truth["F_inf"]) / fit["F_inf_err"])
            z_tau.append((fit["tau"] - truth["tau"]) / fit["tau_err"])
    for pair in z.values():
        for zs in map(np.array, pair):
            assert 0.75 < np.std(zs) < 1.4, np.std(zs)
            assert np.mean(np.abs(zs) <= 3.0) >= 0.95, np.mean(np.abs(zs) <= 3.0)


def test_the_fit_is_as_well_conditioned_at_any_scale():
    """The same points at 1 and at 1e17: the errors scale with them exactly.
    Fitted on raw values, the covariance's pseudo-inverse dropped a direction
    at 1e17 and reported a tau error 4.7 x too small (measured)."""
    t = np.linspace(0.0, 1.0, 16)
    rng = np.random.default_rng(3)
    y = 1 - 0.8 * np.exp(-t / 0.5) + 0.01 * rng.standard_normal(t.size)
    one = m.fit_approach(t, y)
    big = m.fit_approach(t, 1e17 * y)
    assert big["tau"] == pytest.approx(one["tau"], rel=1e-8)    # golden's own
    assert big["tau_err"] == pytest.approx(one["tau_err"], rel=1e-6)
    assert big["F_inf_err"] == pytest.approx(1e17 * one["F_inf_err"], rel=1e-6)


# --- the window's phase -------------------------------------------------------------

@pytest.mark.parametrize("quadrature", [100.0, 2000.0])
def test_the_window_phase_swing_is_the_leakage_times_the_quadrature(quadrature):
    """A settled tone whose half-step in-phase share is 1/quadrature: the
    reading swings at 2f as its window's end moves, by 2 x leakage x
    quadrature of F. Measured: swing over prediction within 2e-4 for 4, 8 and
    16-period windows at both quadratures (0.37 of F at 4 periods and 2000);
    the quadrature it reads back is the one put in to 4e-4, and the phase
    mean is the same F at every window to 1e-5."""
    drive, field = _series(lambda p: np.ones_like(p), periods=61.3,
                           quadrature=quadrature)
    means = []
    for periods in (4, 8, 16):
        phase = m.window_phase(drive, field, DT, **{**READING, "periods": periods})
        assert phase["swing"] == pytest.approx(phase["predicted"], rel=1e-3)
        assert phase["quadrature"] == pytest.approx(quadrature, rel=1e-3)
        means.append(phase["mean"])
    assert max(means) / min(means) - 1.0 < 1e-5, means


def test_the_curve_is_cut_at_the_end_s_own_phase():
    """The same settled, quadrature-2000 tone at 61.3 periods: cuts at free
    5% marks see the 2f swing (spread 0.059 of F, measured), the phase-locked
    cuts see 0.0017 of it -- what is left at a phase mismatch of 0.0026
    periods -- and the arm calls the record settled. The floor the arm prints
    (the swing x 4 pi x that mismatch) bounds what is left of the swing on the
    locked cuts, so the curve's residual sits under it (measured 0.50 of it
    here; 0.48 to 0.96 on the tiny scenes' records)."""
    drive, field = _series(lambda p: np.ones_like(p), periods=61.3, quadrature=2000.0)
    n = np.size(field)
    free = []
    for k in range(5, 21):
        power, reference = m.cw_powers(drive[:int(k * 0.05 * n)],
                                       field[:int(k * 0.05 * n)], DT, **READING)
        free.append(power / reference)
    verdict, curve = _verdict(drive, field)
    locked = [r["F"] for r in curve["rows"] if r["fitted"]]
    spread = lambda v: (max(v) - min(v)) / np.mean(v)      # noqa: E731
    assert spread(free) > 0.04, spread(free)
    assert spread(locked) < 0.005, spread(locked)
    assert max(r["mismatch"] for r in curve["rows"]) < 0.004
    assert verdict["kind"] == "settled"
    lines, reported = _report(drive, field)
    assert reported["kind"] == "settled", lines
    assert 0.0 < reported["fit"]["rms"] <= reported["floor"], reported


def test_an_open_vacuum_box_swings_by_its_own_prediction():
    """Tiny-grid finding, through the solver: an x dipole in a 25^3-node open
    vacuum box (dx 20 nm, CPML 6, 900 nm, 40 periods, ~2 s). The field at the
    source cell is 2344 times its own in-phase part, and the reading swings by
    21.9%, 2.45% and 0.245% of F with 4, 8 and 16-period windows -- each its
    prediction to 1e-4 -- while the phase mean is one F for all three (0.97453,
    0.97455, 0.97457). The record's own two-point "settling" line (+0.0198) is
    0.42 of what the phase alone can put between two readings, and the curve
    is class (a). Nothing here is the full-size scene."""
    f0 = fdtd.C0 / 900e-9
    eps = np.ones((25, 25, 25))
    source = fdtd.Source((12, 12, 12), frequency=f0, profile="cw", component="x")
    n = int(math.ceil(40.0 / (f0 * fdtd.courant_dt(20e-9, 20e-9, 20e-9))))
    sim = fdtd.simulate(eps, source, n, (), dx=20e-9, boundary="open", pml_cells=6,
                        record_drive=True, record_source_field=True)
    reading = {"volume": float(sim["source_volume"]), "eps_r": 1.0, "index": 1.0,
               "frequency": f0, "periods": 8}
    means = []
    for periods in (4, 8, 16):
        phase = m.window_phase(sim["drive"], sim["source_field"], sim["dt"],
                               **{**reading, "periods": periods})
        assert phase["swing"] == pytest.approx(phase["predicted"], rel=1e-3)
        means.append(phase["mean"])
    assert max(means) / min(means) - 1.0 < 2e-4, means
    lines, verdict = _report(sim["drive"], sim["source_field"], reading=reading,
                             dt=sim["dt"])
    assert verdict["kind"] == "settled", lines
    assert verdict["fit"]["rms"] <= verdict["floor"], verdict
    phase = verdict["phase"]
    F = m.cw_reading(sim["drive"], sim["source_field"], sim["dt"], **reading)
    cut = 2 * n // 3
    early = m.cw_reading(sim["drive"][:cut], sim["source_field"][:cut], sim["dt"],
                         **reading)
    assert 0.1 * phase["swing"] < abs(F - early) <= 2.0 * phase["swing"]


# --- a cavity whose settling time is known from its own ring-down -------------------

BOX_DX = 40e-9


def _box_run(profile, frequency, steps, bandwidth=None):
    """A 9 x 17 x 19-node PEC box, two layers of Rakic gold on all six walls, an
    x dipole at the centre: one mode at 272 THz with a 100 fs decay."""
    shape = (9, 17, 19)
    gold = np.zeros(shape, dtype=bool)
    gold[:, :, -2:] = gold[:, :, :2] = gold[:, -2:, :] = gold[:, :2, :] = True
    gold[:2] = gold[-2:] = True
    source = fdtd.Source((4, 8, 9), frequency=frequency, profile=profile,
                         bandwidth=bandwidth, component="x")
    return fdtd.simulate(np.ones(shape), source, steps, (), dx=BOX_DX,
                         boundary="metal",
                         ade_terms=[fdtd.LorentzDrude("Au", gold)],
                         record_drive=True, record_source_field=True)


@pytest.fixture(scope="module")
def lossy_box():
    """The ring-down (6,000 steps) and two CW records at its mode (2 and 10
    decay times): about 10 s together."""
    guess = 272e12
    band = 0.05 * guess
    ring = _box_run("pulse", guess, 6000, band)
    dt = ring["dt"]
    start = m.drive_done(ring["drive"])
    fit = m.fit_decay(ring["source_field"], dt, start, frequency=guess,
                      band=(guess - band, guess + band))
    tail = ring["source_field"][start:].astype(np.float64)
    seg = int(4.0 / (fit["f"] * dt))
    rms = np.array([np.std(tail[a:a + seg]) for a in range(0, tail.size - seg, seg)])
    mid = (np.arange(rms.size) * seg + seg / 2.0) * dt
    envelope_tau = -1.0 / np.polyfit(mid, np.log(rms), 1)[0]
    reading = {"volume": float(ring["source_volume"]), "eps_r": float(ring["source_eps"]),
               "index": 1.0, "frequency": fit["f"], "periods": 8}
    out = {"ring": fit, "envelope_tau": envelope_tau, "reading": reading, "dt": dt}
    for decays in (2, 10):
        cw = _box_run("cw", fit["f"], int(math.ceil(decays * fit["tau"] / dt)))
        out[decays] = _report(cw["drive"], cw["source_field"], reading=reading,
                              dt=dt, locator=(fit["tau"] / 1e-15,
                                              fit["tau_err"] / 1e-15))
    return out


def test_the_lossy_box_settles_on_its_own_ring_down_clock(lossy_box):
    """The ring-down's tau (fit_decay over a band that resolves the line, and a
    straight line through the log of the tail's envelope) against the CW
    curve's tau_s over 2 decay times. Measured: 99.75 fs, 99.93 fs and 99.748
    fs. One mode, so the two must agree, and the arm says so."""
    ring = lossy_box["ring"]
    lines, verdict = lossy_box[2]
    assert ring["resolved"] and ring["rms"] / ring["peak"] < 1e-3, ring
    assert lossy_box["envelope_tau"] == pytest.approx(ring["tau"], rel=5e-3)
    assert verdict["kind"] == "approaching", lines
    assert verdict["fit"]["tau"] == pytest.approx(ring["tau"], rel=1e-3)
    assert any("agrees with the locator's tau" in line for line in lines), lines


def test_the_lossy_box_s_limit_is_the_one_a_longer_record_reaches(lossy_box):
    """F_inf extrapolated from 2 decay times against the same arm on a record of
    10 (whose own end is within 6e-5 of its limit): 561.1335 against 561.1373,
    7e-6 apart (measured); asserted within 2e-5. The 10-decay record is the
    same approach with the same tau, not a growth: the gold's ADE in this
    closed box shows no slow drift over 272 periods at that level."""
    short = lossy_box[2][1]["fit"]
    lines, long = lossy_box[10]
    assert long["kind"] != "growing", lines
    assert long["fit"]["tau"] == pytest.approx(short["tau"], rel=1e-3)
    assert short["F_inf"] == pytest.approx(long["fit"]["F_inf"], rel=2e-5)
    assert long["end"] == pytest.approx(long["fit"]["F_inf"], rel=1e-4)


def test_the_locator_line_says_when_one_mode_cannot_make_the_curve():
    """The 30-period approach (90.9 fs) set against a locator of 113 fs: they
    agree within neither's error only when the locator says 90.9; against
    9 fs the curve is too slow for that mode, and the line says so in words
    without naming a mechanism; on a settled tone there is no tau_s at all."""
    series = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0))
    tau_fs = 30.0 / F0 / 1e-15
    agree, _ = _report(*series, locator=(tau_fs, 1.0))
    assert any("agrees with the locator's tau" in line for line in agree), agree
    slow, _ = _report(*series, locator=(tau_fs / 10.0, 0.3))
    text = "\n".join(slow)
    assert "cannot make this curve" in text and "or the locator's tau is not" in text
    assert re.search(r"against the locator's 9\.09\d* \+- 0\.3 fs: 10 x", text), text
    settled, _ = _report(*_series(lambda p: np.ones_like(p)), locator=(113.0, 3.0))
    assert any("no tau_s to set against it" in line for line in settled), settled


# --- off by default, and the series on disk -----------------------------------------

def _measure(monkeypatch, sim, argv, *, request=None):
    """``measure`` on a canned record (``_run`` stubbed): its printed lines."""
    args = m.parse_args(argv)
    monkeypatch.setattr(m, "_run", lambda *a, **k: (sim, 1.0e-24, 1.0, 1.0))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        F = m.measure(args, None, request or {"frequency_hz": F0}, {"est": {"dt": DT}})
    return F, buffer.getvalue().splitlines()


def test_without_the_new_flags_the_old_lines_are_unchanged(monkeypatch, tmp_path):
    """The curve is printed by default -- it is the whole point of a GPU record,
    it costs about 50 readings of one window, and no existing line moves for
    it. So the claim is: every line the arm printed before is printed with the
    same text in the same order, the curve's lines added after; the two numbers
    are the ones fdtd's own functions give; and nothing is written to disk."""
    drive, field = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0), periods=60.0)
    sim = {"drive": drive, "source_field": field}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "save_series",
                        lambda *a, **k: pytest.fail("written without --save-series"))
    F, lines = _measure(monkeypatch, sim, ["--tiny"])
    old = [line for line in lines if not line.startswith(NEW_LINES)]
    # the two lines the arm always printed, from fdtd's functions directly
    reading = {"volume": 1.0e-24, "eps_r": 1.0, "frequency": F0, "periods": 8.0}

    def ratio(k):
        power = fdtd.delivered_power(drive[:k], field[:k], DT, **reading)
        moment = fdtd.dipole_moment(drive[:k], DT, **reading)
        return power, fdtd.larmor_power(moment, F0, index=1.0)
    power, reference = ratio(drive.size)
    end = fdtd.purcell_factor(power, reference)
    early = fdtd.purcell_factor(*ratio(2 * drive.size // 3))
    assert F == end
    assert old == [
        f"measured  F = {end:.4f} at 900 nm  (P = {power:.6e} W, P0 = {reference:.6e} W)",
        f"settling  F = {early:.4f} at two thirds of the record against {end:.4f} at"
        f" its end, {100.0 * (end / early - 1.0):+.1f}% between them"]
    assert lines[:2] == old                       # first, and in their order
    assert not list(tmp_path.iterdir())
    assert any(line.startswith("class ") for line in lines)


def test_the_new_flags_only_add_lines(monkeypatch, tmp_path):
    """``--save-series`` and ``--locator-tau`` on the same canned record: the
    plain run's lines, unchanged and in order, plus the series and locator
    lines -- and F is the same number."""
    drive, field = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0), periods=60.0)
    sim = {"drive": drive, "source_field": field}
    F_off, off = _measure(monkeypatch, sim, ["--tiny"])
    F_on, on = _measure(monkeypatch, sim, ["--tiny", "--locator-tau", "90,2",
                                           "--save-series", str(tmp_path / "rec")])
    assert F_on == F_off
    assert [line for line in on if not line.startswith(("series ", "locator "))] == off
    assert any(line.startswith("series ") for line in on)
    assert any(line.startswith("locator ") for line in on)


def test_the_saved_series_gives_the_identical_reading(monkeypatch, tmp_path):
    """``--save-series`` writes the record at the dtype it was recorded in (float32
    from the solver) with every constant the reading needs; read back by the
    argument it was given, the same reading is the same F bit for bit, and
    ``--from-series`` on the command line prints the run's own reading lines."""
    drive, field = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0), periods=60.0)
    sim = {"drive": drive, "source_field": field.astype(np.float32)}
    path = tmp_path / "record"                    # no suffix: np.savez adds .npz
    F, lines = _measure(monkeypatch, sim, ["--tiny", "--save-series", str(path)])
    saved = tmp_path / "record.npz"
    assert saved.exists() and any(str(saved) in line for line in lines), lines
    back = m.load_series(path)
    assert back["source_field"].dtype == np.float32
    assert set(back) == set(m.SERIES_KEYS)
    again = m.cw_reading(back["drive"], back["source_field"], back["dt"],
                         volume=back["volume"], eps_r=back["eps_r"], index=back["index"],
                         frequency=back["frequency_hz"], periods=back["window_periods"])
    assert again == F
    done = subprocess.run([sys.executable, str(SCRIPT), "--from-series", str(saved)],
                          capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    keep = ("measured ", "settling ", "phase ", "curve ", "fit ", "class ")
    assert ([line for line in done.stdout.splitlines() if line.startswith(keep)]
            == [line for line in lines if line.startswith(keep)]), done.stdout


def test_a_series_that_cannot_be_written_still_leaves_the_reading(monkeypatch, tmp_path):
    """A path under a file: a warning naming the error, the reading printed,
    the curve still printed, and no line claiming a record was written."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x\n")
    drive, field = _series(lambda p: 1 - 0.8 * np.exp(-p / 30.0), periods=60.0)
    _F, lines = _measure(monkeypatch, {"drive": drive, "source_field": field},
                         ["--tiny", "--save-series", str(blocker / "rec")])
    text = "\n".join(lines)
    assert re.search(r"^warning   the series was not written \(NotADirectoryError:",
                     text, re.M), text
    assert text.startswith("measured  F = ") and "\nclass " in text
    assert not any(line.startswith("series ") for line in lines)


@pytest.mark.parametrize("argv, word", [
    (["--spectrum", "--save-series", "x"], "--save-series"),
    (["--ringdown", "--save-series", "x"], "--save-series"),
    (["--tiny", "--save-series", "x"], "--tiny"),
    (["--estimate", "--save-series", "x"], "--estimate"),
    (["--spectrum", "--locator-tau", "113"], "--locator-tau"),
    (["--from-series", "x.npz", "--tiny"], "--tiny"),
    (["--from-series", "x.npz", "--save-series", "y"], "--save-series"),
    (["--from-series", "x.npz", "--collection-na", "0.8"], "--collection-na"),
])
def test_the_new_flags_are_refused_where_they_would_be_inert(argv, word):
    why = m.arm_conflicts(m.parse_args(argv))
    assert why.startswith("refused:") and word in why, why


def test_the_locator_and_series_flags_parse_what_they_say(tmp_path, capsys):
    assert m.parse_args(["--locator-tau", "113,3"]).locator_tau == (113.0, 3.0)
    assert m.parse_args(["--locator-tau", "113"]).locator_tau == (113.0, 0.0)
    for bad in ("-5", "abc", "1,2,3", "113,-1"):
        with pytest.raises(SystemExit):
            m.parse_args(["--locator-tau", bad])
    # a file that is not a series is refused, not read as zeros
    np.savez(tmp_path / "other.npz", values=np.ones(3))
    assert m.main(["--from-series", str(tmp_path / "other.npz")]) == 2
    assert "is not a --save-series record" in capsys.readouterr().out
