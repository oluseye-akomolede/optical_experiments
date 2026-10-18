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
    """1 - 0.5 exp(-t/10) - 0.3 exp(-t/60 periods): still an approach (b), with
    a tau_s between the two (measured 37.4 +- 2.2 periods) and a residual far
    above the floor (1.1e4 x), which the class line has to say."""
    lines, verdict = _report(*_series(
        lambda p: 1 - 0.5 * np.exp(-p / 10.0) - 0.3 * np.exp(-p / 60.0)))
    fit = verdict["fit"]
    assert verdict["kind"] == "approaching", lines
    assert 10.0 < fit["tau"] * F0 < 60.0
    assert fit["rms"] > m.SIGNIFICANCE * verdict["floor"]
    assert any("more than this one time constant" in line for line in lines), lines


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


# --- the error bars -----------------------------------------------------------------

def test_the_error_bars_cover_the_truth_on_noisy_records():
    """200 records of the 30-period approach over 60 periods, each with white
    noise on the field (0.05 of its in-phase amplitude). The cuts are 3 periods
    apart under 8-period windows, so neighbouring readings share samples
    (measured correlation 0.35-0.43) and the fit's sandwich covariance has to
    allow for it. Measured over 300 records: z = (fit - truth) / error has a
    spread of 1.17 for F_inf and 1.16 for tau_s, 65-66% inside 1 and 98-99%
    inside 3 -- against 0.32/0.15 inside 1 when the cuts were treated as
    independent with a flat overlap factor."""
    envelope = lambda p: 1 - 0.8 * np.exp(-p / 30.0)       # noqa: E731
    drive, clean = _series(envelope, periods=60.0)
    truth = _verdict(drive, clean)[0]["fit"]
    rng = np.random.default_rng(1)
    z_f, z_tau = [], []
    for _ in range(200):
        verdict, _curve = _verdict(drive, clean + 0.05 * rng.standard_normal(clean.size))
        fit = verdict["fit"]
        z_f.append((fit["F_inf"] - truth["F_inf"]) / fit["F_inf_err"])
        z_tau.append((fit["tau"] - truth["tau"]) / fit["tau_err"])
    for z in (np.array(z_f), np.array(z_tau)):
        assert 0.75 < np.std(z) < 1.4, np.std(z)
        assert np.mean(np.abs(z) <= 3.0) >= 0.95, np.mean(np.abs(z) <= 3.0)


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
