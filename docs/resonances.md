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
   step cap of 0.5 Fourier bins, never worse than its start. The selection
   is repeated on the polished poles (a raw pole a few bars off leaves a
   misfit that a second raw pole "explains"; polished together, the second
   explains only noise). A real record keeps its conjugate symmetry: only
   the positive-frequency pole is free.
4. **Bars**: Cramér-Rao bound of the full multi-mode model (Kay 1993,
   ch. 3), noise variance from the residual's window periodogram (the mean
   of its lower 80%, unbiased to 1% and 9% scatter at 159 bins), times
   `error_sigma` (default 3). Only the in-window modes are returned; the
   nuisance poles stay in the model and in the Fisher matrix.

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
REAL_ROWS

Reading the table: the median actual error is 0.67 to 0.8 of the reported
1-sigma (0.674 for an efficient Gaussian estimator), so the bars are
neither loose nor optimistic; the short-record formula is a 20% optimistic
version of the full Fisher bound at gamma T = 0.5 (it assumes gamma T << 1).
The default `error_sigma = 3` bar held at every seed from 10 dB up (largest
excursion 3.1 sigma in 240 draws). At 0 dB on a real record one seed in 40
was 5.4 sigma off in Q: at a per-sample SNR of 1 the likelihood in (f,
gamma) is no longer single-peaked and Q = pi f / gamma has a heavy upper
tail when gamma is only determined to 12%. The test runs seeds 0, 1 and 2
at 0, 10, 20, 40 and 60 dB and asserts the 3-sigma bound at each; it is
deterministic, and a reseeded run at 0 dB can legitimately see a 3-sigma
excursion once in a few hundred draws.

Raw methods at 20 dB (`polish=False`), median |dQ| in reported 1-sigma
units (0.674 if efficient):

RAW_ROWS

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

Two equal modes (Q = 2e4, 40 dB, five seeds), separation in Fourier bins:

RES_ROWS

### A strong mode outside the window

Q = 1e5 mode at f = 1 with gamma T = 0.05 (N = 6366), window [0.99,
1.01]; a second mode of Q = 5e3 outside the window.

FAR_ROWS

Without noise the raw FDM is exact whatever is outside the window (the
far pole is in the record's Krylov space and the basis spans it). With
noise the far pole is not separable from the noise directions of the
basis, and the `background` pencil pass is what keeps the in-window mode
right. A far mode 100 times stronger is still fine with it.

### Record length (no noise, Q = 1e5)

RECORD_ROWS

The round-off floor on Q scales as 1/(gamma T): the pole must be placed to
gamma dt in modulus, and gamma dt = gamma T / N.

### Cost

TIMING_ROWS

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
python tests/check_discriminating.py resonances-nyquist-unchecked
```

MUTATION_TABLE

## Reproducing the sweep

```
OMP_NUM_THREADS=2 python -m pytest tests/test_resonances.py -q -s   # prints the 3-seed curve
```

The 40-seed table and the other sweeps above come from the script below
(run it from the repository root; it takes about six minutes):

```python
SWEEP_SCRIPT
```
