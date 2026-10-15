"""Harmonic inversion of short records: ``dockerfiles/maxwell_solver/resonances.py``.

Every record here is synthetic and closed-form (``damped_sinusoids``), so
each number has an exact truth. Everything is tier 1: a few thousand
samples, under a few seconds per test at ``OMP_NUM_THREADS=2``.

Each test's docstring names the mutation of ``resonances.py`` it goes red on;
``docs/resonances.md`` lists the commands that were run to confirm it.

Units: the synthetic cases use f = 1 with dt = 0.25 (4 samples per period),
so that T = N dt is in periods; one test repeats the headline case at
f = c / 1536 nm in SI units, with ``C0`` imported from ``fdtd``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402
import resonances as R  # noqa: E402

F0 = 1.0
DT = 0.25


def record(modes, fraction_of_decay, *, real=False, noise=0.0, rng=None, q_ref=None):
    """A record ``fraction_of_decay`` of the slowest mode's 1/gamma long."""
    q_ref = q_ref if q_ref is not None else max(m[1] for m in modes)
    gamma = math.pi * F0 / q_ref
    n = int(fraction_of_decay / gamma / DT)
    return R.damped_sinusoids(modes, DT, n, real=real, noise=noise, rng=rng), n


def only(resonances, f, tolerance=0.01):
    """The one resonance within ``tolerance`` of ``f`` (asserting there is one)."""
    near = [r for r in resonances if abs(r.f - f) <= tolerance]
    assert len(near) == 1, f"expected one mode near {f}, got {[r.f for r in resonances]}"
    return near[0]


# --------------------------------------------------------------------------
# 1. Q = 1e5 from a twentieth of a decay time
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["fdm", "pencil"])
@pytest.mark.parametrize("real", [False, True])
def test_q_1e5_from_a_twentieth_of_a_decay_time(method, real):
    """f to 1e-7 relative and Q to 1% from a record 1/20 of 1/gamma long.

    Mutation (``resonances-gamma-halved`` in ``check_discriminating.py``):
    in ``_package`` replace ``gamma = -np.log(np.abs(u)) / dt`` by
    ``/ (2 * dt)``: Q comes out 2e5 and every parametrisation fails on the
    1% bound, while f stays exact.
    """
    Q = 1e5
    amplitude = 0.7 - 0.2j
    sig, n = record([(F0, Q, amplitude)], 0.05, real=real)
    assert n * DT * math.pi * F0 / Q == pytest.approx(0.05, rel=1e-3)
    res = R.find_resonances(sig, DT, 0.99, 1.01, method=method)
    assert len(res) == 1
    r = res[0]
    assert abs(r.f - F0) / F0 < 1e-7
    assert abs(r.Q - Q) / Q < 0.01
    assert abs(r.amplitude - amplitude) < 1e-6
    # the four-tuple contract
    f, q, a, err = r
    assert (f, q, a) == (r.f, r.Q, r.amplitude) and err is r.error
    assert r.decay == pytest.approx(math.pi * F0 / Q, rel=1e-6)
    # and the fit reproduces the record
    fit = R.reconstruct(res, DT, n, real=real)
    assert np.max(np.abs(fit - sig)) < 1e-6


def test_q_1e5_at_1536_nm_in_si_units():
    """The same case at the erbium line, f = c / 1536 nm, dt in seconds.

    Mutation (``resonances-f-per-sample``): in ``_package`` replace
    ``f = -np.angle(u) / (2 * np.pi * dt)`` by ``/ (2 * np.pi)``: f is in
    cycles per sample, nothing lands in the window, and this test fails on
    the count.
    """
    f = fdtd.C0 / 1536e-9
    Q = 1e5
    dt = 1 / (4 * f)
    gamma = math.pi * f / Q
    n = int(0.05 / gamma / dt)
    sig = R.damped_sinusoids([(f, Q, 1.0)], dt, n)
    res = R.find_resonances(sig, dt, 0.99 * f, 1.01 * f)
    assert len(res) == 1
    assert abs(res[0].f - f) / f < 1e-7
    assert abs(res[0].Q - Q) / Q < 0.01


def test_fft_cannot_resolve_the_line_the_inversion_can():
    """The control for the whole module: ``fdtd.spectrum`` on the same record
    has a bin far wider than the linewidth, so no FFT peak width gives Q.

    This is a statement about the FFT, not a mutation target; it is here so
    the claim "plain FFT needs far longer records" is a measured one.
    """
    Q = 1e5
    sig, n = record([(F0, Q, 1.0)], 0.05)
    freqs, amp = fdtd.spectrum(sig.real, DT)
    bin_width = freqs[1] - freqs[0]
    linewidth = F0 / Q
    # the record is 1/(20 gamma) long, so the bin 1/T is 20 pi = 63 linewidths
    assert bin_width > 60 * linewidth
    assert abs(freqs[np.argmax(amp)] - F0) < bin_width


# --------------------------------------------------------------------------
# 2. two close modes and a distant strong one
# --------------------------------------------------------------------------

Q_PAIR = 2e4
LINEWIDTH = F0 / Q_PAIR
TRIO = [(F0, Q_PAIR, 1.0),
        (F0 + 2 * LINEWIDTH, 1.1 * Q_PAIR, 0.5 * np.exp(1j)),
        (F0 + 0.15, 5e3, 20.0)]


@pytest.mark.parametrize("method", ["fdm", "pencil"])
@pytest.mark.parametrize("window", [(0.98, 1.02), (0.98, 1.2)])
def test_two_close_modes_beside_a_strong_distant_one(method, window):
    """Two modes 2 linewidths apart (0.32 of a Fourier bin) and a mode 40 times
    stronger 480 bins away are all found, in a window holding the pair alone
    and in one holding all three, with and without 60 dB noise.

    Mutations: ``resonances-fdm-threshold-coarse`` (``SINGULAR_THRESHOLD``
    1e-7 -> 1e-1) projects the weaker partner out of the FDM's eigenbasis
    and the FDM cases fail on the count; ``resonances-pencil-rank-one``
    (``K = min(K, 1)``) does the same to the pencil cases; and
    ``resonances-no-background`` (``background=False`` as the default)
    fails the noisy narrow-window FDM case alone: the distant mode is then
    not in the model and the pair's Q is off by far more than its bar.
    """
    expected = [m for m in TRIO if window[0] <= m[0] <= window[1]]
    for noise in (0.0, 1e-3):
        sig, n = record(TRIO, 0.5, q_ref=Q_PAIR, noise=noise, rng=1)
        res = R.find_resonances(sig, DT, *window, method=method)
        assert [round(r.f, 3) for r in res] == [round(m[0], 3) for m in expected]
        for r, (f, q, a) in zip(res, expected):
            assert abs(r.f - f) <= max(r.error.f, 1e-12 * f)
            assert abs(r.Q - q) <= max(r.error.Q, 1e-6 * q)
            assert abs(r.amplitude - a) <= max(r.error.amplitude, 1e-9 * abs(a))
            if noise == 0.0:
                assert abs(r.f - f) < 1e-9 and abs(r.Q - q) / q < 1e-6


# --------------------------------------------------------------------------
# 3. the noise sweep
# --------------------------------------------------------------------------

SWEEP_DB = (0, 10, 20, 40, 60)
SWEEP_SEEDS = (0, 1, 2)


@pytest.mark.parametrize("real", [False, True])
def test_noise_sweep_bars_bound_the_error(real):
    """Q error against SNR for one mode with gamma T = 0.5: the reported
    ``error.Q`` bounds the actual error at every SNR and seed, is not loose
    (the median of actual/bar is above 0.05), and the median relative Q error
    follows the Cramér-Rao prediction sqrt(6 / (SNR N)) / (gamma T) (twice
    that for a real record) within a factor 2. The curve is printed.

    Mutation (``resonances-bars-too-narrow``): ``sigma2 / 100`` in
    ``_package``'s ``cramer_rao`` call makes the bars 10 times too narrow
    and the bound fails at every SNR.
    """
    Q = 1e4
    gamma = math.pi * F0 / Q
    sig0, n = record([(F0, Q, 1.0)], 0.5, real=real)
    T = n * DT
    rows = []
    ratios = []
    for snr_db in SWEEP_DB:
        snr = 10 ** (snr_db / 10)
        errors = []
        for seed in SWEEP_SEEDS:
            sig = R.damped_sinusoids([(F0, Q, 1.0)], DT, n, real=real,
                                     noise=1 / math.sqrt(snr), rng=seed)
            r = only(R.find_resonances(sig, DT, 0.95, 1.05), F0)
            assert abs(r.Q - Q) <= r.error.Q, (snr_db, seed, r)
            assert abs(r.f - F0) <= r.error.f, (snr_db, seed, r)
            ratios.append(abs(r.Q - Q) / r.error.Q)
            errors.append(abs(r.Q - Q) / Q)
        predicted = math.sqrt(6 / (snr * n)) / (gamma * T) * (2 if real else 1)
        rows.append((snr_db, float(np.median(errors)), predicted))
        # median |x| of a normal is 0.674 sigma; allow a factor 2 either way
        assert 0.674 / 2 < np.median(errors) / predicted < 0.674 * 2, rows[-1]
    print(f"\nQ error vs SNR ({'real' if real else 'complex'} record, N={n}, gammaT=0.5):")
    print("  SNR dB   median |dQ|/Q   Cramér-Rao 1-sigma")
    for snr_db, err, predicted in rows:
        print(f"  {snr_db:6d}   {err:13.2e}   {predicted:.2e}")
    assert np.median(ratios) > 0.05, ratios


# --------------------------------------------------------------------------
# 4. pure noise
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["fdm", "pencil"])
@pytest.mark.parametrize("real", [False, True])
def test_pure_noise_reports_nothing(method, real):
    """White noise gives an empty list; the control with the significance
    test switched off (``min_significance=0``) reports modes from the same
    record, so the empty list is the test's doing.

    Mutation (``resonances-accept-everything``): in ``_greedy_select``
    replace ``if best_gain < min_significance ** 2 * quad: break`` by ``if
    best_gain < 0: break``: every FDM candidate is accepted and the FDM
    cases fail on every seed. The pencil cases stay green under it because
    the pencil's rank selection (``NOISE_GAP`` over the singular-value
    floor) already gives noise zero poles; ``resonances-f-per-sample``
    is the mutation that fails them (its control window empties).
    """
    n = 4000
    for seed in range(3):
        rng = np.random.default_rng(seed)
        if real:
            sig = rng.standard_normal(n)
        else:
            sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
        assert R.find_resonances(sig, DT, 0.95, 1.05, method=method) == []
        # the control: with the test switched off, raw poles come through
        # (the pencil's 32 poles land anywhere in the band, so its control
        # window is the band; the FDM's are in its own window)
        if method == "fdm":
            control = R.find_resonances(sig, DT, 0.95, 1.05, method=method,
                                        min_significance=0.0, polish=False)
        else:
            control = R.find_resonances(sig, DT, 0.05, 1.95, method=method, rank=32,
                                        residual_threshold=np.inf,
                                        min_significance=0.0, polish=False)
        assert len(control) >= 1


# --------------------------------------------------------------------------
# 5. the two methods agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("real", [False, True])
def test_raw_fdm_and_raw_pencil_agree(real):
    """Without the polish (which would make them meet at the same optimum),
    the two methods' own poles agree: on noise-free records to 1e-8 in f and
    1e-3 in Q relative; on the 40 dB record within ten 1-sigma Cramér-Rao
    bars of each other and of the truth (each raw method scatters by a few
    bars, see docs).

    Mutation (``resonances-pencil-shift-swapped``): in ``matrix_pencil``
    swap the shift, ``V1, V2 = Vk[:, 1:], Vk[:, :-1]``: the pencil's poles
    become 1/u, Q changes sign, and the agreement fails on every record.
    """
    cases = [([(F0, 1e5, 0.7 - 0.2j)], 0.05, 0.0, (0.99, 1.01)),
             (TRIO, 0.5, 0.0, (0.98, 1.2)),
             ([(F0, 1e4, 1.0)], 0.5, 1e-2, (0.95, 1.05))]
    for modes, fraction, noise, window in cases:
        sig, n = record(modes, fraction, real=real, noise=noise, rng=0,
                        q_ref=max(m[1] for m in modes if window[0] <= m[0] <= window[1]))
        a = R.find_resonances(sig, DT, *window, method="fdm", polish=False, error_sigma=1.0)
        b = R.find_resonances(sig, DT, *window, method="pencil", polish=False, error_sigma=1.0)
        assert len(a) == len(b) == sum(window[0] <= m[0] <= window[1] for m in modes)
        for ra, rb, (f, q, _) in zip(a, b, [m for m in modes if window[0] <= m[0] <= window[1]]):
            if noise == 0.0:
                assert abs(ra.f - rb.f) / f < 1e-8
                assert abs(ra.Q - rb.Q) / q < 1e-3
            else:
                bar = 10 * max(ra.error.Q, rb.error.Q)
                assert abs(ra.Q - rb.Q) < bar and abs(ra.Q - q) < bar and abs(rb.Q - q) < bar
                assert abs(ra.f - rb.f) < 10 * max(ra.error.f, rb.error.f)


# --------------------------------------------------------------------------
# 6. the real path equals the complex path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["fdm", "pencil"])
def test_real_path_equals_complex_path(method):
    """A cosine record (the real part of the complex one) gives the same f,
    Q and amplitude as the complex record: f to 1e-9, Q to 1e-5, amplitude
    to 1e-6 relative.

    Mutation (``resonances-real-half-amplitude``): in ``polish`` replace
    ``d = 2 * d[:K]`` by ``d = d[:K]``: the real path's amplitude is half
    the complex one's and the amplitude comparison fails for both methods.
    """
    modes = [(F0, 1e5, 0.7 - 0.2j), (F0 + 0.005, 3e4, 0.3j)]
    sig, n = record(modes, 0.05)
    res_c = R.find_resonances(sig, DT, 0.99, 1.01, method=method)
    res_r = R.find_resonances(sig.real, DT, 0.99, 1.01, method=method)
    assert len(res_c) == len(res_r) == 2
    for rc, rr, (f, q, a) in zip(res_c, res_r, modes):
        assert abs(rc.f - rr.f) / f < 1e-9
        assert abs(rc.Q - rr.Q) / q < 1e-5
        assert abs(rc.amplitude - rr.amplitude) / abs(a) < 1e-6
        assert abs(rr.amplitude - a) / abs(a) < 1e-6
    fit = R.reconstruct(res_r, DT, n, real=True)
    assert np.max(np.abs(fit - sig.real)) < 1e-6


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def test_bad_inputs_are_refused_with_a_message():
    """A window outside Nyquist, a reversed window, a 2-D record and an
    unknown method each raise ``ResonanceModel`` naming the problem.

    Mutation (``resonances-nyquist-unchecked``): delete the ``f_max > nyq``
    arm of ``_check_window``: the window above Nyquist is accepted and the
    first ``raises`` fails.
    """
    sig = np.cos(np.arange(100) * 0.3)
    with pytest.raises(R.ResonanceModel, match="Nyquist"):
        R.find_resonances(sig, DT, 0.5, 3.0)
    with pytest.raises(R.ResonanceModel, match="f_min < f_max"):
        R.find_resonances(sig, DT, 1.0, 0.5)
    with pytest.raises(R.ResonanceModel, match="1-D"):
        R.find_resonances(np.zeros((4, 4)), DT, 0.1, 0.2)
    with pytest.raises(R.ResonanceModel, match="unknown method"):
        R.find_resonances(sig, DT, 0.1, 0.2, method="fft")
    with pytest.raises(R.ResonanceModel, match="too short"):
        R.find_resonances(sig[:5], DT, 0.1, 0.2)
