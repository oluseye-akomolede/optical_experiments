"""Tier 1: fit_decay's search cannot stop in a wrong basin, and says when it is a tie.

Change ``decay-fit-basin``. ``fit_decay`` used to scan 21 omegas across the
band it was given and descend from the best cell. Where that grid step was
wider than the mode's own basin, the descent started in a wrong one: change
``cw-settling`` measured 116.3 fs for a gold box whose envelope decays in
1060 fs. The scan's omega bins are now 1/(8 windows) apart at every tau, and a
second, separated minimum is refined beside the fit and reported.

Every record here is a tiny synthetic tail -- 3,000 samples, numpy only, no
solver -- so the truth is known exactly. Each fit costs about half a second.
The one solver-free arm test stubs ``_run`` with a canned record.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "paper1_nanowire.py"


def _load():
    spec = importlib.util.spec_from_file_location("paper1_nanowire", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load()

# The frame the measurements were taken in: a 331.3 THz drive, a +-15% band
# (the box case's own: the old 21-column grid step is 4.97 THz), 3,000 samples
# of 0.15 fs (a 450 fs window, 20 samples a period).
F_C = 331.3e12
DT = 1.5e-16
N = 3000
T = np.arange(N) * DT
BAND = (0.85 * F_C, 1.15 * F_C)
GRID_STEP = 0.3 * F_C / 20.0          # the old scan's omega column spacing, Hz


def _between_columns(column, offset):
    """A frequency ``offset`` of the way from old grid column ``column`` to the next."""
    return BAND[0] + GRID_STEP * (column + offset)


def _mode(tau, f, phase=0.3, amplitude=1.0):
    return amplitude * np.exp(-T / tau) * np.cos(2.0 * math.pi * f * T + phase)


def _fit(y):
    return m.fit_decay(y, DT, 0, frequency=F_C, band=BAND)


def test_a_1060_fs_mode_between_two_grid_columns_recovers_its_tau():
    """The cw-settling case on a synthetic tail: tau 1060 fs, midway between two
    of the old scan's columns, 1e-3 of noise. The old search returned 108.05 fs
    here with rms 0.38 of the peak (measured; the box gave 116.3 fs and 0.37).

    Tolerance 0.2% (2.1 fs): 40 noise seeds of this record scatter by 0.271 fs
    with a mean bias of 0.004 fs (measured), so 0.2% is 7.8 of that scatter,
    and the wrong basin was 90% off."""
    tau = 1060e-15
    f0 = _between_columns(7, 0.5)
    y = _mode(tau, f0) + 1e-3 * np.random.default_rng(3).standard_normal(N)
    fit = _fit(y)
    assert abs(fit["tau"] / tau - 1.0) < 2e-3, fit
    assert abs(fit["f"] - f0) < 0.25 / (math.pi * tau), fit   # inside its own line
    assert fit["rms"] / fit["peak"] < 2e-3, fit               # noise: 1e-3
    assert fit["ambiguous"] is None, fit
    assert fit["resolved"], fit


def test_no_wrong_basin_across_the_linewidth_to_grid_ratio():
    """tau from 60 fs to 4000 fs -- a line 0.43 of the old grid step down to
    0.016 of it -- at four places between two old columns. The old search got
    33 of 60 such tails wrong (errors from 14.6% to 97%); multi-starting its
    descent from the best 3 or 5 cells still 33 and 31; this search none, at
    every bin spacing from 1/window to 1/(8 windows) (measured).

    The second half is long lines at 1e-4 of noise, where bins 1/window apart
    are not enough even with this descent: 5 of 60 such tails (700 fs to 8000
    fs, 12 places between columns) came back 12.7% to 18.3% short, f right
    and tau wrong, where bins 1/(2, 4 or 8 windows) apart left none (worst
    0.04%, measured). These four are among those five.

    Tolerance 1%: the worst of the 60 measured with 1/(8 windows) bins was
    0.19% at 1e-3 noise, and the smallest wrong-basin error was 12.7%."""
    rng = np.random.default_rng(11)
    wrong = []
    cases = [(tau_fs, 7, offset, 1e-3, 1.1) for tau_fs in (60, 150, 400, 1060, 4000)
             for offset in (0.0, 0.33, 0.5, 0.83)]
    cases += [(1060, 3, 0.75, 1e-4, 2.3), (4000, 3, 0.75, 1e-4, 2.3),
              (8000, 3, 0.083, 1e-4, 2.3), (8000, 3, 0.417, 1e-4, 2.3)]
    for tau_fs, column, offset, noise, phase in cases:
        tau = tau_fs * 1e-15
        f0 = _between_columns(column, offset)
        fit = _fit(_mode(tau, f0, phase=phase) + noise * rng.standard_normal(N))
        ratio = 1.0 / (math.pi * tau) / GRID_STEP
        if abs(fit["tau"] / tau - 1.0) > 0.01:
            wrong.append((tau_fs, round(ratio, 3), offset, fit["tau"] / 1e-15))
    assert not wrong, wrong


def test_a_line_wider_than_the_band_is_searched_across_its_width():
    """The --tiny vacuum tail's shape: a 1.9 fs decay (under one period) in a
    60 fs window of 2,571 samples, with a +-8% band at 333.1 THz. The scan's
    approximate residual is only good to O(1/(omega tau)) there -- 25% -- so
    omega is searched across the line's own width, 1/tau either side. With two
    bins either side the 1.9 fs fit was 6 to 31 THz off and its rms 34 to 160
    x the noise (three places in the band); the old search and this one land
    within 0.02 THz and 0.05% of tau at 1, 1.9 and 3 fs (measured).
    Tolerances: 0.2 THz and 0.5%."""
    dt, n, band = 2.322e-17, 2571, (307.2e12, 359.0e12)
    t = np.arange(n) * dt
    for f0 in (315e12, 332.6e12):
        y = (np.exp(-t / 1.9e-15) * np.cos(2.0 * math.pi * f0 * t + 0.7)
             + 1e-4 * np.random.default_rng(4).standard_normal(n))
        fit = m.fit_decay(y, dt, 0, frequency=333.1e12, band=band)
        assert abs(fit["f"] - f0) < 0.2e12, fit
        assert abs(fit["tau"] / 1.9e-15 - 1.0) < 5e-3, fit
        assert fit["rms"] < 1.2e-4, fit


def test_two_equal_modes_are_flagged_ambiguous():
    """Two modes of equal amplitude in the band are two answers the record does
    not choose between: the rival's residual is 1.0 x the fit's for two 1060 fs
    modes 10 THz apart and about 1.5 x for a 400 fs and a 250 fs mode
    (measured), both under the factor of 12. The rival named is the other mode."""
    rng = np.random.default_rng(5)
    f1 = 0.95 * F_C
    for t1, t2 in ((1060e-15, 1060e-15), (400e-15, 250e-15)):
        y = (_mode(t1, f1, 0.4) + _mode(t2, f1 + 10e12, 2.0)
             + 1e-3 * rng.standard_normal(N))
        fit = _fit(y)
        assert fit["ambiguous"] and "second minimum" in fit["ambiguous"], fit
        assert fit["rival"]["rss_ratio"] < 2.0, fit
        found = sorted((fit["f"], fit["rival"]["f"]))
        assert abs(found[0] - f1) < 0.5e12 and abs(found[1] - f1 - 10e12) < 0.5e12, fit


def test_a_mode_under_the_noise_and_noise_alone_are_flagged():
    """Noise alone: every separated minimum is the noise's, and the rival's
    residual is within 0.3% of the fit's (measured, 16 seeds: 1.000 to
    1.003). A 1060 fs mode under noise of its own amplitude: 1.31 to 1.35
    (measured, 3 seeds) -- its fitted tau was off by up to 96%."""
    rng = np.random.default_rng(900)
    alone = _fit(rng.standard_normal(N))
    assert alone["ambiguous"] and alone["rival"]["rss_ratio"] < 1.05, alone
    buried = _fit(_mode(1060e-15, _between_columns(7, 0.5)) + rng.standard_normal(N))
    assert buried["ambiguous"] and buried["rival"]["rss_ratio"] < 2.0, buried


def test_a_second_mode_that_biases_tau_is_flagged_below_the_factor():
    """A 400 fs mode with a 250 fs one at 0.6 of its amplitude 10 THz above:
    the one-mode fit lands on the stronger and reads it 24% long, and the
    rival's residual is 4.1 x the fit's (measured). That is above a factor of 3
    and below 12, so it is the case that pins the factor from below. In 105
    measured records (60 two-mode, 45 one-mode under noise) 16 fits were more
    than 10% off their mode's tau; 14 had a rival at 1.05 to 11.1 x, and 12 is
    the round factor over all of them. The other two are not caught: a second
    mode at 0.12 amplitude whose rival was 122 x, and one at 0.25 that formed
    no separated minimum."""
    f1 = 0.95 * F_C
    y = (_mode(400e-15, f1, 0.4) + _mode(250e-15, f1 + 10e12, 2.0, 0.6)
         + 1e-3 * np.random.default_rng(16).standard_normal(N))
    fit = _fit(y)
    assert abs(fit["f"] - f1) < 0.5e12, fit
    assert 3.0 < fit["rival"]["rss_ratio"] < m.AMBIGUITY_RATIO, fit
    assert fit["ambiguous"] and "second minimum" in fit["ambiguous"], fit


def test_a_clean_mode_and_a_weak_second_mode_are_not_flagged():
    """The other side of the factor. A clean mode has no separated minimum at
    all (none in 26 of the 45 measured one-mode records, every one under 0.12
    noise; the first flagged one-mode record was at 0.16, ratio 11.5, its tau
    1.7% off -- the price of the factor). A 250 fs mode at 0.2 of a 400 fs one's amplitude, 25 THz away,
    leaves the rival 37 x the fit's residual (measured) and the fit within
    3.3% of its tau: above 12, so not a tie -- the factor is not set so high
    that a second mode a record can see past reads as one."""
    clean = _fit(_mode(400e-15, _between_columns(7, 0.5))
                 + 1e-2 * np.random.default_rng(2).standard_normal(N))
    assert clean["ambiguous"] is None, clean
    f1 = 0.95 * F_C
    y = (_mode(400e-15, f1, 0.4) + _mode(250e-15, f1 + 25e12, 2.0, 0.2)
         + 1e-3 * np.random.default_rng(27).standard_normal(N))
    weak = _fit(y)
    assert weak["rival"]["rss_ratio"] > 20.0, weak
    assert weak["ambiguous"] is None, weak
    assert abs(weak["tau"] / 400e-15 - 1.0) < 0.05, weak


def test_the_answer_does_not_depend_on_which_cell_the_scan_ranked_first(monkeypatch):
    """The scan only chooses basins, from an approximate residual. Here it is
    made to rank a wrong cell first -- the real scan, with one cell at the
    mode's own tau row and an omega 12% below the mode pushed under its
    minimum -- and the fit must still
    come back with the mode, because the wrong cell's exact residual is
    refined against the mode's and loses. The wrong cell is then the rival,
    with a residual far over the mode's, so the fit is not flagged."""
    real = m._decay_scan

    def misranked(y, dt, taus, lo_w, hi_w):
        taus, omegas, S = real(y, dt, taus, lo_w, hi_w)
        i = int(np.argmin(np.abs(np.log(taus / 400e-15))))
        j = int(np.argmin(np.abs(omegas - 2.0 * math.pi * 0.86 * F_C)))
        S = S.copy()
        S[i, j] = S.min() - 1.0
        return taus, omegas, S

    monkeypatch.setattr(m, "_decay_scan", misranked)
    tau, f0 = 400e-15, _between_columns(7, 0.5)
    fit = _fit(_mode(tau, f0) + 1e-3 * np.random.default_rng(8).standard_normal(N))
    assert abs(fit["tau"] / tau - 1.0) < 0.01, fit
    assert abs(fit["f"] - f0) < 0.25 / (math.pi * tau), fit
    assert abs(fit["rival"]["f"] - 0.86 * F_C) < 0.01 * F_C, fit
    assert fit["ambiguous"] is None, fit


def test_a_degenerate_tau_direction_is_flagged():
    """A constant tail: the covariance's tau direction vanishes (tau_err None),
    and that is a reason too. f_err is not read: it is None on well-fitted
    tails as well (the 1060 fs case above), where the pinv drops omega's
    direction for scale alone."""
    flat = m.fit_decay(np.full(2000, 1e-9), DT, 0, frequency=F_C, band=BAND)
    assert flat["tau_err"] is None, flat
    assert "degenerate" in flat["ambiguous"], flat
    good = _fit(_mode(1060e-15, _between_columns(7, 0.5))
                + 1e-3 * np.random.default_rng(3).standard_normal(N))
    assert good["f_err"] is None and good["ambiguous"] is None, good


def _ringdown(monkeypatch, tail):
    """The ring-down arm on a canned record: a drive that ends at step 50, then
    ``tail``. ``_run`` is stubbed; fit_decay and everything after it are real."""
    drive = np.zeros(50 + tail.size)
    drive[:50] = np.sin(np.arange(50))
    sim = {"drive": drive, "source_field": np.concatenate((np.zeros(50), tail))}
    monkeypatch.setattr(m, "_run", lambda *a, **k: (sim, 1.0e-24, 1.0, 1.0))
    args = m.parse_args(["--tiny", "--ringdown"])
    request = {"frequency_hz": F_C, "bandwidth": 0.15 * F_C}
    measured = {"est": {"dt": DT}, "steps": drive.size, "dlambda_nm": 1.0}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fit, code = m.ringdown(args, None, request, measured)
    return fit, code, out.getvalue()


def test_the_arm_prints_the_reason_where_tau_would_be(monkeypatch):
    """Two equal modes: the fit line says AMBIGUOUS with the reason in place of
    "tau = ... +- ...", no Q is printed and no bound is formed, and the code is
    the unresolved one. The control is one clean mode, which prints its tau and
    its Q exactly as before."""
    rng = np.random.default_rng(5)
    f1 = 0.95 * F_C
    two = _mode(1060e-15, f1, 0.4) + _mode(1060e-15, f1 + 10e12, 2.0)
    fit, code, out = _ringdown(monkeypatch, two + 1e-3 * rng.standard_normal(N))
    assert code == m.UNRESOLVED, out
    assert "tau AMBIGUOUS (" in out and "second minimum" in out, out
    assert "unresolved: the fit is ambiguous" in out, out
    assert "\nQ         " not in out and "Q >" not in out, out
    assert " tau = " not in out, out

    one = _mode(400e-15, _between_columns(7, 0.5))
    fit, code, out = _ringdown(monkeypatch, one + 1e-3 * rng.standard_normal(N))
    assert code == 0, out
    assert " tau = " in out and "AMBIGUOUS" not in out, out
    assert "\nQ         " in out, out
