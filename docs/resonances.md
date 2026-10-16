# Resonances from short records: `resonances.py`

*Module notes for paper 2 (arXiv 2505.00224, erbium in silicon
nanophotonic cavities near 1536 nm, Q from 1e4 to above 1e5). Written
2026-10-15 with the module; every number below has the command that
reproduces it.*

## What it does

`dockerfiles/maxwell_solver/resonances.py` fits a uniformly sampled time
record to a sum of damped sinusoids and returns the resonances in a
frequency window:

```python
import resonances as R
modes = R.find_resonances(signal, dt, f_min, f_max)      # list of Resonance
f, Q, amplitude, error = modes[0]                         # the four-tuple
error.f, error.Q, error.amplitude                         # 3-sigma bars
error.residual, error.significance                        # diagnostics
fit = R.reconstruct(modes, dt, len(signal), real=np.isrealobj(signal))
```

The record can be real (an FDTD probe trace) or complex. Conventions,
options and the numerical limits are in the module docstring; this page
holds the measurements behind the limits and the test record.

Why not the FFT: a mode of quality factor Q rings for Q/(pi f) before it
has decayed by 1/e. `fdtd.spectrum` resolves its linewidth only once the
record is a few decay times long. At Q = 1e5 and 195 THz that is hundreds
of picoseconds, millions of FDTD steps, beyond `fdtd.MAX_STEPS`. A damped
sinusoid is a four-parameter model, and a record of a few thousand samples
determines it from one twentieth of a decay time (first table below).

## The two methods and the pipeline

* **`fdm`**: filter diagonalization (Mandelshtam & Taylor, J. Chem. Phys.
  107, 6756 (1997); the algorithm of Harminv, S. G. Johnson,
  github.com/NanoComp/harminv). A Fourier basis restricted to the window
  turns the record into a small generalised eigenproblem whose eigenvalues
  are the poles. O(J N) for J basis functions, J proportional to the
  window width in Fourier bins.
* **`pencil`**: matrix pencil (Hua & Sarkar, IEEE Trans. ASSP 38, 814
  (1990)). SVD of a Hankel matrix of the record; the poles are the
  eigenvalues of the shifted right singular vectors. Global, algebraically
  unrelated to the FDM: the cross-check.

Both go through one pipeline, in this order:

1. **Candidates**: raw poles that are finite, within 50 e-foldings over the
   record and consistent (`error.residual <= residual_threshold`). For the
   FDM path the pencil's poles *outside* the window are added
   (`background=True`), because a strong mode outside the window is in the
   record but not in the basis; with noise it is smeared over spurious
   poles at the window edge and biases the modes inside (measured below).
2. **Selection**: greedy forward selection by the power a pole removes
   from the residual, accepted if that power is at least
   `min_significance`^2 (default 36) noise variances per quadrature. The
   noise level starts from the record's own window periodogram (an upper
   bound) and is re-read from the residual after each acceptance, so a
   weak mode under a strong one's leakage is found on the next round and a
   pole fitted to noise never lowers the level it is judged against.
   Harminv's eigen-residual is *not* this filter: on noise at 0 dB, 86 of
   87 noise-fitted FDM poles had residual below 1e-4.
3. **Polish** (`polish=True`): Levenberg-Marquardt on (f T, gamma T) with
   the amplitudes eliminated (variable projection, Kaufman Jacobian), a
   step cap of 0.5 Fourier bins, never worse than its start. Then each
   polished pole is judged again by its amplitude against its own standard
   deviation from the full Fisher matrix, the least significant is dropped
   if below `min_significance`, and the polish repeats (a raw pole a few
   bars off leaves a misfit that a second raw pole "explains" given the
   first one's amplitude; with the first one's f and gamma free, the second
   explains only noise). A real record keeps its conjugate symmetry: only
   the positive-frequency pole is free.
4. **Bars**: Cramér-Rao bound of the full multi-mode model (Kay 1993,
   ch. 3), noise variance from the residual's window periodogram (the mean
   of its lower 80%, unbiased to 1% and 9% scatter at 159 bins), times
   `error_sigma` (default 3). Only the in-window modes are returned; the
   nuisance poles stay in the model and in the Fisher matrix. The matrix is
   scaled to unit diagonal before it is inverted, and a mode lying along an
   eigen-direction below `1 / FISHER_CONDITION_LIMIT` (1e-12) of the
   largest eigenvalue gets NaN bars (`error.f`, `error.Q`,
   `error.amplitude`) instead of a number: a duplicated pole, which the
   pseudo-inverse used to give exactly half the single-mode bars, or two
   equal modes closer than about 0.02 Fourier bins at 40 dB. The other
   modes of the call keep their bars.

## Measurements

All with `OMP_NUM_THREADS=2`, f = 1, dt = 0.25 (4 samples per period), on
the cloud session's 2-thread container. Scripts: the tests in
`tests/test_resonances.py` (the numbers marked *test*) and the sweep script
reproduced at the end of this page.

### Q = 1e5 from one twentieth of a decay time (no noise) — *test*

Record: N = 6366 samples, gamma T = 0.05, amplitude 0.7 - 0.2i, window
[0.99, 1.01] (32 Fourier bins, 18 FDM basis functions).

| record | method | polish | relative f error | relative Q error |
|---|---|---|---|---|
| complex | fdm | off | 0 | 3.3e-11 |
| complex | fdm | on | 0 | 4.9e-12 |
| complex | pencil | off | 0 | 3.3e-11 |
| complex | pencil | on | 0 | 4.9e-12 |
| real | fdm | off | 1.9e-15 | 1.2e-10 |
| real | fdm | on | 0 | 4.9e-12 |
| real | pencil | off | 0 | 7.6e-11 |
| real | pencil | on | 0 | 4.9e-12 |

The FFT of the same record (`fdtd.spectrum`) has a bin of 1/T = 6.3e-4,
63 linewidths wide. The 1536 nm case in SI units (`fdtd.C0 / 1536e-9`,
dt = 1/(4f)) gives the same figures (`test_q_1e5_at_1536_nm_in_si_units`).

### Noise sweep — *test* (3 seeds) and 40-seed sweep

One mode, Q = 1e4, gamma T = 0.5, N = 6366, window [0.95, 1.05]. Per-sample
SNR = |a|^2 / sigma^2. Polished FDM (the pencil gives the identical
polished numbers; both reach the same maximum-likelihood point).

| record | SNR | median \|dQ\|/Q (40 seeds) | short-record formula sqrt(6/(SNR N))/(gamma T) | reported 1-sigma (median) | largest \|dQ\| in reported sigmas | largest \|df\| in sigmas |
|---|---|---|---|---|---|---|
| complex | 0 dB | 5.8e-2 | 6.1e-2 | 7.5e-2 | 3.5 | 2.7 |
| complex | 10 dB | 1.9e-2 | 1.9e-2 | 2.5e-2 | 3.1 | 2.8 |
| complex | 20 dB | 6.0e-3 | 6.1e-3 | 8.0e-3 | 3.0 | 2.8 |
| complex | 30 dB | 1.9e-3 | 1.9e-3 | 2.5e-3 | 2.9 | 2.8 |
| complex | 40 dB | 6.0e-4 | 6.1e-4 | 7.9e-4 | 2.9 | 2.8 |
| complex | 60 dB | 6.0e-5 | 6.1e-5 | 7.9e-5 | 2.9 | 2.8 |
| real | 0 dB | 9.9e-2 | 1.2e-1 | 1.4e-1 | 5.4 | 2.7 |
| real | 10 dB | 3.2e-2 | 3.9e-2 | 4.8e-2 | 4.3 | 2.7 |
| real | 20 dB | 1.0e-2 | 1.2e-2 | 1.5e-2 | 3.9 | 2.7 |
| real | 30 dB | 3.2e-3 | 3.9e-3 | 4.9e-3 | 3.8 | 2.7 |
| real | 40 dB | 1.0e-3 | 1.2e-3 | 1.6e-3 | 3.7 | 2.7 |
| real | 60 dB | 1.0e-4 | 1.2e-4 | 1.6e-4 | 3.7 | 2.7 |

Reading the table: the median actual error is 0.67 to 0.8 of the reported
1-sigma (0.674 for an efficient Gaussian estimator), so the bars are
neither loose nor optimistic; the short-record formula is a 20% optimistic
version of the full Fisher bound at gamma T = 0.5 (it assumes gamma T << 1).
The default `error_sigma = 3` bar held at every seed from 10 dB up on
complex records (largest excursion 3.1 sigma in 240 draws). On real records
one seed (25) sits at 3.7 to 4.3 sigma at every SNR: it is the same noise
realisation scaled, and against the bound computed from the *true* noise
variance it is 3.4 to 3.5 sigma, so it is a genuine tail draw, not a bar
error (over 40 seeds at 40 dB the reported bar is 0.97 +- 0.04 of the
true-noise bound and the scatter of the estimate is 1.03 bounds). At 0 dB
on a real record one seed in 40 was 5.4 sigma off in Q: at a per-sample
SNR of 1 the likelihood in (f, gamma) is no longer single-peaked and
Q = pi f / gamma has a heavy tail when gamma is only determined to 12%.
The test runs seeds 0, 1 and 2 at 0, 10, 20, 40 and 60 dB and asserts the
3-sigma bound at each; it is deterministic, and a reseeded run can
legitimately see a 3-sigma excursion once in a few hundred draws.

Raw methods at 20 dB (`polish=False`), median |dQ| in reported 1-sigma
units (0.674 if efficient):

| record | method | median \|dQ\| / reported 1-sigma | largest |
|---|---|---|---|
| complex | fdm | 1.76 | 5.6 |
| complex | pencil | 1.34 | 5.9 |
| real | fdm | 1.34 | 5.9 |
| real | pencil | 1.25 | 5.3 |

This is why the polish is on by default: the raw FDM scatters at 2.5 to
3.5 times the bound and the capped pencil at about 2 times, and bars
drawn at the bound would not cover them.

### Pure noise — *test*

White Gaussian noise, N = 4000, window [0.95, 1.05] (= 159 Fourier bins,
87 FDM basis functions): 0 modes reported in 10 seeds for each of real /
complex and fdm / pencil. The raw FDM gives 87 poles per record, with
significances up to 1.7; the threshold is 6.

### Resolution: two close modes — *test* at 2 linewidths, sweep below

Two modes Q = 2e4 and 2.2e4, amplitudes 1 and 0.5 e^i, separated by two
linewidths (2 f/Q = 1e-4, which is 0.32 of a Fourier bin at gamma T =
0.5), plus a mode 40 times stronger at +0.15 (480 bins). N = 12732.

| noise | window | method | f errors | Q errors (bars) |
|---|---|---|---|---|
| none | pair only | fdm / pencil | < 1e-9 | 1e-6 relative or better |
| none | all three | fdm / pencil | < 1e-9 | 1e-6 relative or better |
| 60 dB | pair only | fdm | 5e-8, 1e-7 (bars 1.0e-7, 2.0e-7) | 52, 122 (bars 81, 192) |
| 60 dB | all three | fdm | same, third mode 1e-9 | same, third mode 3e-3 (bar 2e-2) |

Two equal modes (Q = 2e4, 40 dB, five seeds), separation in Fourier bins.
At 0.1 bins the second pole is no longer significant once the first one's
f and gamma are free, and the backward test reports the pair as one mode
rather than two badly determined ones:

| separation | both found (5 seeds) | worst relative Q error |
|---|---|---|
| 2 bins | 5/5 | 1.2e-3 |
| 1 bin | 5/5 | 2.6e-3 |
| 0.5 bin | 5/5 | 1.3e-2 |
| 0.32 bin | 5/5 | 3.0e-2 |
| 0.2 bin | 5/5 | 6.7e-2 |
| 0.1 bin | 0/5 (reported as one mode) | - |

### A strong mode outside the window

Without noise the raw FDM is exact whatever lies outside the window: a
Q = 1e5 mode at gamma T = 0.05 with a Q = 5e3 mode 1 or 100 times stronger
at 20 or 159 Fourier bins outside a [0.99, 1.01] window comes back with
f exact and Q to 5e-12, `background` on or off (the far pole is in the
record's Krylov space and the basis spans it). With noise the far pole is
not separable from the noise directions of the basis. The three-mode
record above (pair at f = 1, mode 40 times stronger at 480 bins, 60 dB) in
the narrow window [0.98, 1.02]:

| `background` | modes reported | pair's f errors | pair's Q errors (bars) |
|---|---|---|---|
| True (default) | 2 | 2.7e-8, 4.4e-8 (bars 1.0e-7, 2.0e-7) | 30, 78 (bars 80, 189) |
| False | 53 | 1.8e-2, 1.8e-2 | 6347, 28453 (one of them negative) |

Without the pencil's far pole the leakage of the strong mode is modelled
by dozens of edge poles, each "significant" against the leakage-inflated
noise level, and the pair is lost. The mutation case
`resonances-no-background` is exactly this.

### Record length (no noise, Q = 1e5)

| gamma T | N | relative f error | relative Q error |
|---|---|---|---|
| 0.05 | 6366 | 0 | 4.9e-12 |
| 0.01 | 1273 | 0 | 4.9e-12 |
| 0.002 | 254 | 0 | 4.9e-12 |
| 0.0005 | 63 | 0 | 4.9e-12 |

Without noise there is no record-length floor down to 63 samples; the
residual 5e-12 is the polish's stopping tolerance. With noise the floor is
the bound above: the relative Q error is sqrt(6/(SNR N)) / (gamma T), so a
record of gamma T = 0.05 needs SNR N above 2.4e7 (40 dB over 2400
samples, say) for Q to 1%.

### Cost

Two modes (Q = 1e5 and 5e4, 0.01 apart) at 60 dB, window [0.99, 1.02],
`find_resonances` end to end, 2 threads:

| N | fdm (complex) | fdm (real) | pencil |
|---|---|---|---|
| 6366 | 1.0 s | 0.3 s | 0.3 s |
| 50000 | 4.3 s | 4.4 s | 0.6 s |
| 200000 | 37 s | 71 s | 2.5 s |

The FDM path grows as J N for the matrices and as (candidates x N) per
selection round, with J = 1.1 x window width x N dt / 2 basis functions
(825 at N = 200000 here, twice that for the mirrored real window), plus
the eigen-solve of the J x J pencil; the pencil path is capped at a
8192 x 257 Hankel matrix and grows only through the amplitude fit and the
polish. For a long FDTD record, decimate to a few samples per period of
the band of interest before calling, or use `method="pencil"` for a
first look.

`polish` is O(N K^2) per iteration for K poles (a few to 50 iterations);
the pencil's SVD is capped at 8192 x 257; the FDM is O(J N) with its basis
streamed in 64 MB blocks.

## Mutation record

Each test names the mutation it goes red on; the mutations are cases in
`tests/check_discriminating.py` (prefix `resonances-`), which applies the
edit, runs `tests/test_resonances.py`, restores the file by hash and
reports the exact failing set:

```
python tests/check_discriminating.py resonances-gamma-halved
python tests/check_discriminating.py resonances-f-per-sample
python tests/check_discriminating.py resonances-significance-1e6
python tests/check_discriminating.py resonances-no-background
python tests/check_discriminating.py resonances-bars-too-narrow
python tests/check_discriminating.py resonances-accept-everything
python tests/check_discriminating.py resonances-pencil-shift-swapped
python tests/check_discriminating.py resonances-real-half-amplitude
python tests/check_discriminating.py resonances-fisher-unguarded
python tests/check_discriminating.py resonances-nyquist-unchecked
```

| case | edit | tests that go red |
|---|---|---|
| gamma-halved | `_package`: gamma / (2 dt) | the Q = 1e5 tests (5), the three-mode tests (4), the sweep (2), raw agreement (2), real path (2) |
| f-per-sample | `_package`: f without / dt | all of the above plus the FDM pure-noise tests (17) |
| fdm-threshold-coarse | `SINGULAR_THRESHOLD` 1e-1 | the three-mode FDM tests (2), raw agreement (2) |
| pencil-rank-one | `K = min(K, 1)` | the three-mode pencil tests (2), raw agreement (2), Q = 1e5 real pencil, real path pencil |
| no-background | `background=False` default | the three-mode FDM narrow-window test alone |
| bars-too-narrow | `cramer_rao(..., sigma2 / 100)` | the sweep (2), the three-mode tests (4), raw agreement (2) |
| accept-everything | no significance test | the FDM pure-noise tests (2), the sweep (2), raw agreement (2), three-mode FDM (2) |
| pencil-shift-swapped | `V1, V2` swapped | raw agreement (2), Q = 1e5 complex pencil, real path pencil, three-mode (3) |
| real-half-amplitude | `polish`: `d = d[:K]` | real path (2), Q = 1e5 real (2) |
| fisher-unguarded | `cramer_rao`: `np.linalg.pinv(F)`, every parameter determined | duplicate poles (2), near-duplicate pair (2), weak nuisance (2); the separated-modes test stays green by design |
| nyquist-unchecked | `_check_window` without the `f_max > nyq` arm | the bad-inputs test |

Every case was run on 2026-10-15 and reported `OK - exactly the expected
tests fail, and only they` once the sets above were recorded; the pencil
pure-noise tests stay green under `accept-everything` because the pencil's
rank floor already gives noise zero poles (the f-per-sample case is what
fails them).

## Reproducing the sweep

```
OMP_NUM_THREADS=2 python -m pytest tests/test_resonances.py -q -s   # prints the 3-seed curve
```

The 40-seed table and the other sweeps above come from the script below
(run it from the repository root; it takes about six minutes):

```python
"""Final numbers for docs/resonances.md (40 seeds per point)."""
import sys, time, math
import numpy as np
sys.path.insert(0, "dockerfiles/maxwell_solver")
import resonances as R

f0 = 1.0
dt = 0.25
Q1 = 1e4
g1 = math.pi * f0 / Q1
T1 = 0.5 / g1
N1 = int(T1 / dt)
SEEDS = 40
print(f"N={N1} gammaT={g1*T1:.2f} seeds={SEEDS}")
print("# polished (default) FDM: median |dQ|/Q, CRB(short-record formula), CRB(full Fisher, reported 1-sigma median), max |dQ|/bar(1s), max |df|/bar(1s)")
for real in (False, True):
    for snr_db in (0, 10, 20, 30, 40, 60):
        snr = 10 ** (snr_db / 10)
        eq, rq, rf, bars = [], [], [], []
        for seed in range(SEEDS):
            sig = R.damped_sinusoids([(f0, Q1, 1.0)], dt, N1, real=real, noise=1 / math.sqrt(snr), rng=seed)
            res = R.find_resonances(sig, dt, 0.95, 1.05, error_sigma=1.0)
            r = max(res, key=lambda r: abs(r.amplitude))
            eq.append(abs(r.Q - Q1) / Q1); rq.append(abs(r.Q - Q1) / r.error.Q); rf.append(abs(r.f - f0) / r.error.f); bars.append(r.error.Q / Q1)
        crb = math.sqrt(6 / (snr * N1)) / (g1 * T1) * (2 if real else 1)
        print(f"real={real!s:5} {snr_db:2d} dB | median|dQ|/Q {np.median(eq):.2e} | formula {crb:.2e} | reported 1s {np.median(bars):.2e} | max|dQ|/1s {max(rq):.2f} | max|df|/1s {max(rf):.2f}")
print("# raw methods (polish=False) at 20 dB: median |dQ|/Q relative to the reported 1-sigma")
for real in (False, True):
    for method in ("fdm", "pencil"):
        rq = []
        for seed in range(SEEDS):
            sig = R.damped_sinusoids([(f0, Q1, 1.0)], dt, N1, real=real, noise=0.1, rng=seed)
            res = R.find_resonances(sig, dt, 0.95, 1.05, method=method, polish=False, error_sigma=1.0)
            r = max(res, key=lambda r: abs(r.amplitude))
            rq.append(abs(r.Q - Q1) / r.error.Q)
        print(f"real={real!s:5} {method:6s} raw: median |dQ|/1s {np.median(rq):.2f} (0.674 for an efficient estimator), max {max(rq):.2f}")
print("# round-off floor: Q=1e5, gammaT=0.05, no noise")
Q0 = 1e5; g0 = math.pi * f0 / Q0; N0 = int(0.05 / g0 / dt)
for real in (False, True):
    for method in ("fdm", "pencil"):
        sig = R.damped_sinusoids([(f0, Q0, 0.7 - 0.2j)], dt, N0, real=real)
        for pol in (False, True):
            r = R.find_resonances(sig, dt, 0.99, 1.01, method=method, polish=pol)[0]
            print(f"real={real!s:5} {method:6s} polish={pol!s:5} |df|/f={abs(r.f-f0):.1e} |dQ|/Q={abs(r.Q-Q0)/Q0:.1e} residual={r.error.residual:.1e}")
print("# gammaT floor: Q=1e5 no noise, shorter records")
for frac in (0.05, 0.01, 0.002, 0.0005):
    N = int(frac / g0 / dt)
    sig = R.damped_sinusoids([(f0, Q0, 1.0)], dt, N)
    r = R.find_resonances(sig, dt, 0.99, 1.01)[0]
    print(f"gammaT={frac} N={N}: |df|/f={abs(r.f-f0):.1e} |dQ|/Q={abs(r.Q-Q0)/Q0:.1e}")
print("# a mode outside the window (FDM, Q=1e5 mode at f=1 with gammaT=0.05; strong mode of Q=5e3 outside the window [0.99,1.01])")
for amp, dist in ((1.0, 0.0125), (100.0, 0.0125), (100.0, 0.1)):
    sig = R.damped_sinusoids([(f0, Q0, 1.0), (f0 + dist, 5e3, amp)], dt, N0)
    for bg in (True, False):
        res = R.find_resonances(sig, dt, 0.99, 1.01, background=bg)
        r = [x for x in res if abs(x.f - f0) < 0.005]
        if r:
            r = r[0]
            print(f"amp {amp} at {dist*N0*dt:.0f}/T background={bg}: |df|/f={abs(r.f-f0):.1e} |dQ|/Q={abs(r.Q-Q0)/Q0:.1e} bar/Q={r.error.Q/Q0:.1e} modes={len(res)}")
        else:
            print(f"amp {amp} at {dist*N0*dt:.0f}/T background={bg}: mode missing ({len(res)} reported)")
    sig = R.damped_sinusoids([(f0, Q0, 1.0), (f0 + dist, 5e3, amp)], dt, N0, noise=1e-3, rng=0)
    for bg in (True, False):
        res = R.find_resonances(sig, dt, 0.99, 1.01, background=bg)
        r = [x for x in res if abs(x.f - f0) < 0.005]
        if r:
            r = r[0]
            print(f"  with 60 dB noise, background={bg}: |df|/f={abs(r.f-f0):.1e} |dQ|/Q={abs(r.Q-Q0)/Q0:.1e} bar/Q={r.error.Q/Q0:.1e} modes={len(res)}")
        else:
            print(f"  with 60 dB noise, background={bg}: mode missing ({len(res)} reported)")
print("# resolution: two equal modes at separation s/T, Q=2e4, gammaT=0.5, 40 dB; found both?")
Q2 = 2e4; g2 = math.pi * f0 / Q2; N2 = int(0.5 / g2 / dt); T2 = N2 * dt
for sep in (2.0, 1.0, 0.5, 0.32, 0.2, 0.1):
    found = 0; errs = []
    for seed in range(5):
        sig = R.damped_sinusoids([(f0, Q2, 1.0), (f0 + sep / T2, Q2, 1.0j)], dt, N2, noise=1e-2, rng=seed)
        res = R.find_resonances(sig, dt, 0.98, 1.02)
        found += len(res) == 2
        if len(res) == 2:
            errs.append(max(abs(res[0].Q - Q2), abs(res[1].Q - Q2)) / Q2)
    print(f"separation {sep}/T: both found in {found}/5 seeds; worst |dQ|/Q {max(errs) if errs else float('nan'):.2e}")
print("# timing")
for N in (6366, 50000, 200000):
    sig = R.damped_sinusoids([(f0, 1e5, 1.0), (f0 + 0.01, 5e4, 0.5)], dt, N, noise=1e-3, rng=0)
    for method in ("fdm", "pencil"):
        t0 = time.time(); res = R.find_resonances(sig, dt, 0.99, 1.02, method=method); el = time.time() - t0
        print(f"N={N} {method}: {len(res)} modes in {el:.2f} s")
```
