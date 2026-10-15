"""High-Q resonances from a short time record: harmonic inversion.

WHY NOT THE FFT. A cavity mode with quality factor Q rings for Q/(pi f)
before it has decayed by 1/e. A plain spectrum (``fdtd.spectrum``) resolves
its linewidth only once the record is a few decay times long, which at
Q = 1e5 and 195 THz (the erbium line near 1536 nm) is several hundred
picoseconds of FDTD: millions of steps. Harmonic inversion instead *fits* the
record to a sum of damped sinusoids

    c(t) = sum_k  a_k exp(-2 pi i f_k t - gamma_k t),      Q_k = pi f_k / gamma_k

so the record only has to be long enough to *determine* the decay, not to
display it. With noise-free data a record of 1/20 of a decay time gives Q to
far better than 1% (``docs/resonances.md`` has the measured numbers); with
noise the record length enters through the Cramér-Rao bound below.

TWO METHODS, ONE PIPELINE.

* ``fdm``: the filter-diagonalization method of Mandelshtam & Taylor,
  J. Chem. Phys. 107, 6756 (1997), as implemented in Harminv (S. G. Johnson,
  https://github.com/NanoComp/harminv). The signal is a Krylov sequence of an
  unknown evolution operator; a Fourier basis restricted to the frequency
  window [f_min, f_max] turns the global problem into a small generalised
  eigenproblem U1 b = u U0 b whose eigenvalues u_k = exp(-2 pi i f_k dt -
  gamma_k dt) are the poles and whose eigenvectors give the amplitudes. Cost
  O(J N) for J basis functions, with J proportional to the window width in
  units of the Fourier resolution 1/T. As in Harminv the problem is solved a
  second time with the accepted poles themselves as the basis, which removes
  the noise directions the first basis carried.
* ``pencil``: the matrix pencil of Hua & Sarkar, IEEE Trans. ASSP 38, 814
  (1990). A Hankel matrix of the record has rank equal to the number of
  poles; the poles are the generalised eigenvalues of its two column-shifted
  halves. It is global (no window enters the algebra) and algebraically
  unrelated to the FDM, which is what makes it a cross-check.

Both return the same ``Resonance`` records and go through the same
post-processing: keep the poles inside the window, drop the ones whose
eigen-solution is inconsistent (``error.residual`` above
``residual_threshold``), keep the ones that are statistically significant
against the noise level, then (``polish=True``) move the kept poles to the
maximum-likelihood fit by Gauss-Newton on the nonlinear least squares with
the amplitudes eliminated (variable projection), and attach Cramér-Rao error
bars at that point. The polish is what makes the bars honest: the raw FDM
and pencil are linear-algebra estimators whose scatter is a few times the
bound (measured in ``docs/resonances.md``), the maximum-likelihood fit
reaches it.

CONVENTIONS. A complex record is fitted as written above, so a positive f is
a phase rotating as exp(-2 pi i f t) (Harminv's convention; conjugate a
record that rotates the other way, or ask for a negative window). A real
record (``numpy.isrealobj``) is fitted as Re of the same sum, with the
amplitude a_k being the full cosine amplitude: ``c(t) = Re(sum_k a_k exp(...))``.
In both cases ``reconstruct`` rebuilds the fit from the returned list.

ERROR BARS. ``error.f``, ``error.Q`` and ``error.amplitude`` are
``error_sigma`` standard deviations from the Cramér-Rao bound (Kay,
*Fundamentals of Statistical Signal Processing: Estimation Theory*, 1993,
ch. 3) of the full multi-mode model, with the noise variance read from the
residual's periodogram inside the window. In the short-record limit
(gamma T << 1) the bound for a single mode of amplitude A in complex white
noise of variance sigma^2 per sample is

    std(f) = std(gamma) / (2 pi) = sqrt(6 / (SNR N)) / (2 pi T),   SNR = |A|^2 / sigma^2

so the relative Q error is sqrt(6 / (SNR N)) / (gamma T): shorter records and
higher Q both cost precision in proportion. A real record at the same |A| and
noise variance has twice these standard deviations (half the signal power
in the same noise). ``error.residual`` is a different thing: the dimensionless
consistency of the eigen-solution (Harminv's "error"); it is near round-off
for a well-represented pole and is a check on the algebra, not a confidence
interval, and it does not tell a noise-fitted mode from a real one.

SPURIOUS MODES. A mode is reported only if it passes two tests at
``min_significance`` noise standard deviations: forward, the power its pole
removes from the residual given the poles already accepted (a chi-square
on the amplitude fit, with the noise level first bounded from above by the
record's own window periodogram and then read from what the accepted modes
leave behind, so a weak mode under a strong one's leakage is found on the
next round and a pole fitted to noise never lowers the level it is judged
against); and backward, after the polish, its amplitude against its own
standard deviation from the full Fisher matrix, with the other poles'
frequencies and decays free (a second pole that only explained the misfit
of a raw first pole is dropped here). ``error.significance`` is the second.

NUMERICAL LIMITS (measured in ``tests/test_resonances.py`` and
``docs/resonances.md``; f = 1, 4 samples per period unless stated):

* Round-off: with no noise, Q = 1e5 from a record of gamma T = 0.05 comes
  back with f exact to the last bit and Q to 5e-12 relative (polished; the
  raw FDM and pencil give 3e-11 to 1e-10), and the same to 63 samples
  (gamma T = 5e-4). There is no record-length floor without noise.
* Noise: the relative Q error is sqrt(6 / (SNR N)) / (gamma T) for a
  complex record and twice that for a real one, within the 20% by which
  the short-record formula under-states the full Fisher bound at
  gamma T = 0.5. The polished estimate reaches the bound (its scatter is
  1.03 times it over 40 seeds); the raw FDM scatters at 2.5 to 3.5 times
  it and the capped pencil at about 2 times, which is why the polish is
  on by default and why ``polish=False`` bars should not be trusted.
* Bars: ``error_sigma = 3`` held at every seed from 10 dB up in a 240-draw
  sweep of complex records (largest excursion 3.1 sigma) and in all but
  one draw of real records (one seed at 3.7 to 4.3 sigma, the same noise
  realisation at every SNR, with the same excursion against the bound
  computed from the true noise). At 0 dB per-sample SNR on a real record
  the likelihood is no longer single-peaked and one seed in 40 was 5.4
  sigma off: do not read the bars as bounds below about 10 dB.
* Resolution: two equal modes at 40 dB with gamma T = 0.5 are both found
  down to 0.2 Fourier bins (1/T) apart, with the worst Q error growing
  from 1e-3 at 2 bins to 7e-2 at 0.2 bins and the bars with it; at 0.1
  bins the pair is reported as one mode (the second is not significant
  once the first one's f and gamma are free).
* Window: without noise the raw FDM is exact whatever lies outside the
  window. With noise a strong mode outside the window is smeared over
  spurious poles at the window edge and biases the modes inside unless
  it is in the model; ``background=True`` (the pencil's out-of-window
  poles) is what keeps the three-mode test right in a window that holds
  only its close pair.
* Spurious modes: white noise gives no mode at the default
  ``min_significance`` (the largest FDM candidate significance on noise
  was 1.7 against the threshold of 6). The eigen ``residual`` does not
  tell a noise-fitted pole from a real one (86 of 87 FDM poles fitted to
  0 dB noise had residual below 1e-4) and is kept as an algebra check.
* Cost: O(J N) for the FDM matrices plus O(K_cand N) per selection round,
  the pencil's SVD capped at ``MAX_ROWS`` x (``MAX_PENCIL`` + 1), and the
  polish at O(N K^2) per iteration; the FDM basis is streamed in
  ``_BLOCK_BYTES`` blocks. Timings are in ``docs/resonances.md``.

Only numpy is required; scipy is not used.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = [
    "Resonance", "ResonanceError", "ResonanceModel",
    "find_resonances", "fdm", "matrix_pencil", "damped_sinusoids",
    "reconstruct", "cramer_rao", "polish",
]


class ResonanceError(NamedTuple):
    """Error bars of one resonance.

    ``f`` (Hz), ``Q`` and ``amplitude`` (same units as the signal) are
    ``error_sigma`` standard deviations from the Cramér-Rao bound; ``residual``
    is the dimensionless eigen-solution consistency of the raw solve (0 is
    perfect); ``significance`` is |amplitude| in units of its own noise
    standard deviation, the quantity the spurious-mode filter thresholds.
    """

    f: float
    Q: float
    amplitude: float
    residual: float
    significance: float


class Resonance(NamedTuple):
    """One damped sinusoid ``amplitude * exp(-2 pi i f t - gamma t)``.

    Unpacks as the four-tuple ``(f, Q, amplitude, error)``; ``decay`` is the
    amplitude decay rate gamma = pi f / Q in 1/s.
    """

    f: float
    Q: float
    amplitude: complex
    error: ResonanceError

    @property
    def decay(self) -> float:
        return math.pi * self.f / self.Q

    @property
    def omega(self) -> complex:
        """Complex angular frequency 2 pi f - i gamma."""
        return 2 * math.pi * self.f - 1j * self.decay


class ResonanceModel(Exception):
    """The record cannot be fitted as asked (bad window, too short, ...)."""


# --------------------------------------------------------------------------
# synthetic signals and reconstruction
# --------------------------------------------------------------------------

def _poles(modes, dt: float):
    """``(u, a)`` arrays from ``(f, Q, amplitude)`` triples."""
    modes = list(modes)
    if not modes:
        return np.zeros(0, complex), np.zeros(0, complex)
    f = np.array([m[0] for m in modes], float)
    Q = np.array([m[1] for m in modes], float)
    a = np.array([m[2] for m in modes], complex)
    gamma = np.pi * f / Q
    u = np.exp((-2j * np.pi * f - gamma) * dt)
    return u, a


def damped_sinusoids(modes, dt: float, n: int, *, real: bool = False,
                     noise: float = 0.0, rng=None) -> np.ndarray:
    """``n`` samples of ``sum_k a_k exp(-2 pi i f_k t - gamma_k t)``.

    ``modes`` is an iterable of ``(f, Q, amplitude)``; ``gamma = pi f / Q``.
    With ``real=True`` the real part is returned (a cosine of full amplitude
    ``|a|``). ``noise`` is the standard deviation of white Gaussian noise per
    sample: for a complex record it is split equally between the two
    quadratures so that ``E|w|^2 = noise^2``; for a real record it is the
    standard deviation of the real noise. This is the convention the
    Cramér-Rao bound uses, so ``SNR = |a|^2 / noise^2`` in both cases.
    """
    u, a = _poles(modes, dt)
    signal = _vandermonde(u, n) @ a if u.size else np.zeros(n, complex)
    if real:
        signal = signal.real
    if noise:
        rng = np.random.default_rng(rng)
        if real:
            signal = signal + noise * rng.standard_normal(n)
        else:
            w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            signal = signal + noise * w / math.sqrt(2.0)
    return signal


def reconstruct(resonances, dt: float, n: int, *, real: bool):
    """The fitted signal of ``find_resonances`` over ``n`` samples.

    ``real`` must say whether the record was real (the amplitudes then are
    cosine amplitudes); the list does not carry that.
    """
    modes = [(r.f, r.Q, r.amplitude) for r in resonances]
    return damped_sinusoids(modes, dt, n, real=real)


def _vandermonde(u, n: int) -> np.ndarray:
    """Columns ``u_k^t`` for t = 0..n-1, by exp of the log (no repeated products)."""
    t = np.arange(n)
    return np.exp(np.outer(t, np.log(u.astype(complex))))


# --------------------------------------------------------------------------
# the filter-diagonalization method
# --------------------------------------------------------------------------

#: Bytes of basis matrix held at once; blocks of the J x (M+1) Fourier
#: matrix are streamed so a long record does not allocate J*N complex numbers.
_BLOCK_BYTES = 64 << 20

#: Eigenvalues of U0 below this fraction of the largest are the null space
#: of the (rank-deficient) overlap and are projected out before the
#: generalised eigenproblem is solved, as Harminv does.
SINGULAR_THRESHOLD = 1e-7

#: Fewest basis functions a window is given. A narrower window is still
#: given this many (more closely spaced), so that a mode near its edge and a
#: few spurious partners can all be represented.
MIN_BASIS = 8

#: Poles growing or decaying by more than this many e-foldings over the
#: record are not resonances of anything and would overflow the model.
MAX_EFOLDINGS = 50.0

#: Largest move of a pole in one polish step, in units of the Fourier
#: resolution (f T) and of e-foldings over the record (gamma T).
MAX_STEP = 0.5


def _window_basis(f_min: float, f_max: float, dt: float, n_basis: int,
                  real: bool) -> np.ndarray:
    """Basis poles ``z_j = exp(-2 pi i f_j dt)`` on a cell-centred grid of the window.

    A real record's negative-frequency poles are put inside a mirrored
    window so that they are represented exactly; otherwise they are "far"
    poles and perturb the result.
    """
    f = f_min + (np.arange(n_basis) + 0.5) * (f_max - f_min) / n_basis
    if real:
        spacing = (f_max - f_min) / n_basis
        if f_min <= spacing:
            k = int(math.ceil(f_max / spacing))
            f = (np.arange(-k, k) + 0.5) * spacing
        else:
            f = np.concatenate([-f[::-1], f])
    return np.exp(-2j * np.pi * f * dt)


def _fdm_matrices(c: np.ndarray, z: np.ndarray, M: int):
    """``(U0, U1, U2, G0)`` of Mandelshtam & Taylor for basis poles ``z``.

    U^p_{jj'} = sum_{n,n'=0..M} z_j^-n z_j'^-n' c_{n+n'+p}, through the
    closed-form geometric sums (their eqs. 11-13), which cost O(J M) rather
    than O(J^2 M). ``G0_j = sum_n z_j^-n c_n`` is kept for the amplitudes.
    ``z`` may be complex (inside or outside the unit circle), so a caller can
    pass uses the found poles as the basis.
    """
    J = z.size
    logz = np.log(z)                      # z^-m = exp(-m log z)
    zinvM = np.exp(-M * logz)
    zinvM1 = np.exp(-(M + 1) * logz)
    G = np.zeros((3, J), complex)
    H = np.zeros((3, J), complex)
    D = np.zeros((3, J), complex)
    m = np.arange(M + 1)
    w_lo = (m + 1).astype(float)            # weight of c_{m+p}, m <= M
    w_hi = (M - m[:M]).astype(float)        # weight of c_{M+1+m+p}, m < M
    block = max(1, _BLOCK_BYTES // (16 * (M + 1)))
    for j0 in range(0, J, block):
        j1 = min(J, j0 + block)
        E = np.exp(-np.outer(logz[j0:j1], m))     # z_j^-m
        for p in range(3):
            lo = c[p:p + M + 1]
            hi = c[M + 1 + p:2 * M + 1 + p]
            G[p, j0:j1] = E @ lo
            H[p, j0:j1] = E[:, :M] @ hi
            D[p, j0:j1] = E @ (w_lo * lo) + zinvM1[j0:j1] * (E[:, :M] @ (w_hi * hi))
    U = np.empty((3, J, J), complex)
    denom = z[:, None] - z[None, :]
    np.fill_diagonal(denom, 1.0)
    for p in range(3):
        num = (z[:, None] * G[p][None, :] - z[None, :] * G[p][:, None]
               + zinvM[None, :] * H[p][:, None] - zinvM[:, None] * H[p][None, :])
        Up = num / denom
        np.fill_diagonal(Up, D[p])
        U[p] = Up
    return U[0], U[1], U[2], G[0]


def _solve_fdm(U0, U1, U2, G0, singular_threshold: float):
    """Poles ``u``, amplitudes ``d`` and residuals of U1 b = u U0 b.

    U0 is symmetric and rank-deficient (its rank is the number of poles the
    record actually contains), so the pencil is regularised the way Harminv
    does it: diagonalise U0, keep the eigen-directions whose eigenvalue is
    above ``singular_threshold`` of the largest, and solve the ordinary
    eigenproblem of U1 projected onto them. Amplitudes follow from
    d_k = (b_k^T G0)^2 with b_k^T U0 b_k = 1 (Mandelshtam & Taylor eq. 17).
    The residual is |b^T U2 b - u^2| / |u|^2: U2 is the same pencil shifted
    by two samples, which a pole of the Krylov sequence satisfies with u^2.
    """
    lam, S = np.linalg.eig(U0)
    keep = np.abs(lam) > singular_threshold * np.abs(lam).max()
    if not keep.any():
        return np.zeros(0, complex), np.zeros(0, complex), np.zeros(0)
    P = S[:, keep]
    Sinv = np.linalg.solve(S, np.eye(S.shape[0]))
    Qr = Sinv[keep]
    reduced = (Qr @ U1 @ P) / lam[keep][:, None]
    u, W = np.linalg.eig(reduced)
    B = P @ W
    norm = np.einsum("jk,jl,lk->k", B, U0, B)
    B = B / np.sqrt(norm)[None, :]
    d = (B.T @ G0) ** 2
    u2 = np.einsum("jk,jl,lk->k", B, U2, B)
    residual = np.abs(u2 - u * u) / np.abs(u * u)
    return u, d, residual


def fdm(signal, dt: float, f_min: float, f_max: float, *, density: float = 1.1,
        n_basis: int | None = None, singular_threshold: float = SINGULAR_THRESHOLD,
        basis=None):
    """Raw filter-diagonalization poles: ``(u, amplitude, residual)`` arrays.

    ``u_k = exp(-2 pi i f_k dt - gamma_k dt)``; every eigen-solution is
    returned, inside the window or not, significant or not; the amplitude of
    a real record's pole is the complex half-amplitude d_k (the cosine
    amplitude is 2 d_k). ``find_resonances`` is the filtered form.

    ``n_basis`` defaults to ``density`` basis functions per Fourier-resolution
    cell (M+1) dt (f_max - f_min) of the window, at least ``MIN_BASIS`` and
    at most M+1 (the rank of the overlap), M = (N-3)//2. ``basis`` replaces
    the Fourier grid by explicit poles z_j.
    """
    c, real = _as_record(signal)
    N = c.size
    M = (N - 3) // 2
    if M < 2:
        raise ResonanceModel(f"a record of {N} samples is too short; need at least 7")
    _check_window(f_min, f_max, dt, real)
    if basis is not None:
        z = np.asarray(basis, complex)
    else:
        if n_basis is None:
            n_basis = int(math.ceil(density * (f_max - f_min) * (M + 1) * dt))
            n_basis = min(max(n_basis, MIN_BASIS), M + 1)
        if n_basis < 1:
            raise ResonanceModel("n_basis must be positive")
        z = _window_basis(f_min, f_max, dt, n_basis, real)
    U0, U1, U2, G0 = _fdm_matrices(c, z, M)
    return _solve_fdm(U0, U1, U2, G0, singular_threshold)


# --------------------------------------------------------------------------
# the matrix pencil
# --------------------------------------------------------------------------

#: Default pencil parameter cap; the Hankel matrix has L+1 columns. Hua &
#: Sarkar put the noise-optimal L between N/3 and N/2, but the SVD costs
#: O(rows L^2), so a long record is capped here and in ``MAX_ROWS``.
MAX_PENCIL = 256
MAX_ROWS = 8192

#: Singular values below this multiple of the noise floor (the median of the
#: lower half) do not count as signal: the Marchenko-Pastur edge of the noise
#: block is within a factor 1.5 of its median for the shapes used here.
NOISE_GAP = 2.5


def matrix_pencil(signal, dt: float, *, pencil_size: int | None = None,
                  max_rows: int = MAX_ROWS, rank: int | None = None,
                  rank_threshold: float = 1e-10):
    """Raw matrix-pencil poles: ``(u, amplitude, residual)`` arrays.

    The Hankel matrix Y[r, l] = c[r + l] (a random subset of rows when there
    are more than ``max_rows``) is factored by SVD; the rows of V^H spanning
    the signal subspace are the right singular vectors, and the poles are
    the eigenvalues of Vh[:, 1:] pinv(Vh[:, :-1]) (Hua & Sarkar eq. 25).
    ``rank`` fixes the number of poles; otherwise it is the number of
    singular values above both ``rank_threshold`` of the largest and
    ``NOISE_GAP`` times the noise floor read from the smallest ones, at most
    half the pencil size. Amplitudes are the least-squares fit of the
    Vandermonde system over the whole record (half-amplitudes for a real
    record, as in ``fdm``). The residual is the normalised defect of each
    pole's left eigenvector in the shifted pencil, |l (Vh2 - u Vh1)| / |l Vh2|.
    """
    c, real = _as_record(signal)
    N = c.size
    if pencil_size is None:
        L = min(N // 3, MAX_PENCIL)
    else:
        L = int(pencil_size)
    if L < 2 or L + 1 >= N:
        raise ResonanceModel(f"pencil size {L} does not fit a record of {N} samples")
    rows = N - L
    if rows > max_rows:
        # A random subset of rows, not a stride: rows r*s turn each pole u
        # into u^s, and two poles whose frequencies differ by a multiple of
        # 1/(s dt) then share a row vector and the rank collapses (a cosine
        # at 4 samples per period with s = 2 is one such pair). Any row set
        # keeps the column shift that the pencil uses; the first and last
        # rows are kept so the whole record is spanned.
        pick = np.random.default_rng(0).choice(rows - 2, max_rows - 2, replace=False)
        r = np.concatenate([[0], np.sort(pick) + 1, [rows - 1]])
    else:
        r = np.arange(rows)
    Y = c[r[:, None] + np.arange(L + 1)[None, :]]
    _, s, Vh = np.linalg.svd(Y, full_matrices=False)
    if rank is None:
        floor = np.median(s[s.size // 2:])
        K = int(np.sum((s > rank_threshold * s[0]) & (s > NOISE_GAP * floor)))
        K = min(K, L // 2)
    else:
        K = int(rank)
    if K == 0:
        return np.zeros(0, complex), np.zeros(0, complex), np.zeros(0)
    Vk = Vh[:K]
    V1, V2 = Vk[:, :-1], Vk[:, 1:]
    A = V2 @ np.linalg.pinv(V1)
    u, T = np.linalg.eig(A)
    Tinv = np.linalg.inv(T)
    defect = Tinv @ V2 - u[:, None] * (Tinv @ V1)
    residual = (np.linalg.norm(defect, axis=1)
                / np.maximum(np.linalg.norm(Tinv @ V2, axis=1), 1e-300))
    d = _amplitudes(c, u)
    return u, d, residual


# --------------------------------------------------------------------------
# shared post-processing
# --------------------------------------------------------------------------

def _as_record(signal):
    c = np.asarray(signal)
    if c.ndim != 1:
        raise ResonanceModel(f"the record must be 1-D, got shape {c.shape}")
    real = bool(np.isrealobj(c))
    c = c.astype(float if real else complex)
    if not np.all(np.isfinite(c)):
        raise ResonanceModel("the record has non-finite samples")
    return c, real


def _check_window(f_min, f_max, dt, real):
    if not dt > 0:
        raise ResonanceModel(f"dt must be positive, got {dt}")
    if not f_min < f_max:
        raise ResonanceModel(f"need f_min < f_max, got {f_min} and {f_max}")
    nyq = 0.5 / dt
    if real:
        if f_min < 0 or f_max > nyq:
            raise ResonanceModel(
                f"a real record's window must lie in [0, {nyq:g}] (Nyquist); "
                f"got [{f_min:g}, {f_max:g}]")
    elif f_min < -nyq or f_max > nyq:
        raise ResonanceModel(
            f"a complex record's window must lie in [{-nyq:g}, {nyq:g}] (Nyquist); "
            f"got [{f_min:g}, {f_max:g}]")


def _amplitudes(c, u):
    """Least-squares complex amplitudes of poles ``u`` over the whole record.

    The Vandermonde columns are scaled to unit norm so a decaying pole does
    not swamp the conditioning. For a real record the fit is complex; the
    conjugate partner of each pole is expected to be in ``u`` as well.
    """
    if u.size == 0:
        return np.zeros(0, complex)
    V = _vandermonde(u, c.size)
    scale = np.linalg.norm(V, axis=0)
    d, *_ = np.linalg.lstsq(V / scale, c.astype(complex), rcond=None)
    return d / scale


def _noise_variance(residual, dt, f_min, f_max, real, n_modes):
    """White-noise variance per sample equivalent to the residual inside the window.

    The periodogram of white noise is exponential with mean N sigma^2 per bin;
    the mean of its lower 80% is 0.5981 of that, which is what is read here
    (robust to a few leaked bins at the window edges). The fitted modes have
    absorbed 2 complex degrees of freedom each, so the count of bins is
    reduced accordingly. Falls back to the whole band when the window has
    fewer bins than that.
    """
    N = residual.size
    R = np.fft.fft(residual)
    freqs = np.fft.fftfreq(N, d=dt)
    inside = (freqs >= f_min) & (freqs <= f_max)
    if real:
        inside |= (freqs >= -f_max) & (freqs <= -f_min)
    power = np.abs(R[inside]) ** 2
    n_bins = power.size // (2 if real else 1)
    dof = n_bins - 2 * n_modes
    if dof < 2:
        power = np.abs(R) ** 2
        n_bins = N // (2 if real else 1)
        dof = n_bins - 2 * n_modes
        if dof < 2:
            return float("nan")
    power = np.sort(power)
    kept = power[:max(1, int(round(0.8 * power.size)))]
    return float(kept.mean() / 0.5981 / N * n_bins / dof)


def _jacobian(modes, dt, n, real):
    """Columns d(model)/d(ln|a|, arg a, f T, gamma T) per mode; real for a real record."""
    t = np.arange(n) * dt
    T = t[-1] if n > 1 else dt
    cols = []
    for f, gamma, a in modes:
        m = a * np.exp((-2j * np.pi * f - gamma) * t)
        cols += [m, 1j * m, -2j * np.pi * (t / T) * m, -(t / T) * m]
    Jm = np.stack(cols, axis=1)
    return (Jm.real if real else Jm), T


def cramer_rao(modes, dt: float, n: int, sigma2: float, *, real: bool):
    """Cramér-Rao standard deviations of ``(f, gamma, ln|a|, arg a)`` per mode.

    ``modes`` is a list of ``(f, gamma, amplitude)``; ``sigma2`` the noise
    variance per sample (``E|w|^2`` for complex noise). Returns an array of
    shape ``(len(modes), 4)`` with the standard deviations, from the inverse
    of the Fisher matrix of the full multi-mode model (Kay 1993, eq. 3.31),
    computed with the dimensionless time t/T so that the inversion is well
    scaled.
    """
    K = len(modes)
    if K == 0:
        return np.zeros((0, 4))
    Jm, T = _jacobian(modes, dt, n, real)
    if real:
        F = (Jm.T @ Jm) / sigma2
    else:
        F = 2 * (Jm.conj().T @ Jm).real / sigma2
    cov = np.linalg.pinv(F, hermitian=True)
    var = np.clip(np.diag(cov), 0, None).reshape(K, 4)
    std = np.sqrt(var)
    std[:, 2] /= T         # f = (f T) / T
    std[:, 3] /= T         # gamma
    return std[:, [2, 3, 0, 1]]


def polish(c, dt: float, u, *, real: bool, iterations: int = 50,
           tolerance: float = 1e-12):
    """Levenberg-Marquardt refinement of poles ``u`` to the least-squares fit of ``c``.

    For a real record ``u`` holds only the non-negative-frequency poles; the
    model is Re(sum a v) and the returned amplitudes are the full cosine
    amplitudes. Variable projection: for each trial pole set the amplitudes are the
    linear least-squares solution, and the step is taken on
    (f T, gamma T) with the Kaufman Jacobian (the amplitude columns of the
    full Jacobian are what the projection removes). The damping grows
    tenfold until a step lowers the sum of squares and shrinks after each
    accepted one; no step moves a pole by more than ``MAX_STEP`` in
    f T or gamma T, so the fit never leaves the basin it started in and
    never gets worse than its start. Iteration stops when the sum of
    squares changes by less than ``tolerance`` relatively, or when no
    damping gives an improvement. Returns ``(u, amplitudes)``.
    """
    c = np.asarray(c)
    n = c.size
    u = np.asarray(u, complex)
    K = u.size
    if K == 0:
        return u, np.zeros(0, complex)
    T = (n - 1) * dt if n > 1 else dt

    def fit(u_try):
        # a real record's model is Re(sum a v): the conjugate partner of
        # each pole is implied, never a free pole of its own
        pair = _with_conjugates(u_try, real)
        d = _amplitudes(c, pair)
        r = c - (_vandermonde(pair, n) @ d)
        if real:
            r = r.real
            d = 2 * d[:K]
        return d, r, float(np.vdot(r, r).real)

    a, r, rss = fit(u)
    damping = 1e-3
    for _ in range(iterations):
        f = -np.angle(u) / (2 * np.pi * dt)
        gamma = -np.log(np.abs(u)) / dt
        Jm, _ = _jacobian(list(zip(f, gamma, a)), dt, n, real)
        Jm = Jm.reshape(n, K, 4)                       # (.., ln|a|, arg a, fT, gammaT)
        # Kaufman's Jacobian: the pole derivatives with their component along
        # the amplitude columns removed, since the amplitudes re-fit at
        # every trial (Golub & Pereyra's second term vanishes against r)
        D = Jm[:, :, :2].reshape(n, 2 * K)
        Jp = Jm[:, :, 2:].reshape(n, 2 * K)
        Jp = Jp - D @ np.linalg.lstsq(D, Jp, rcond=None)[0]
        if real:
            A, b = Jp, r
        else:
            A = np.concatenate([Jp.real, Jp.imag])
            b = np.concatenate([r.real, r.imag])
        scale = np.maximum(np.linalg.norm(A, axis=0), 1e-300)
        A = A / scale
        accepted = False
        first = True
        for _attempt in range(20):
            aug = np.concatenate([A, math.sqrt(damping) * np.eye(2 * K)])
            rhs = np.concatenate([b, np.zeros(2 * K)])
            step, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
            step = step / scale
            largest = np.abs(step).max()
            if largest > MAX_STEP:
                step = step * (MAX_STEP / largest)
            step = step.reshape(K, 2) / T
            f_try = f + step[:, 0]
            g_try = gamma + step[:, 1]
            if np.abs(g_try).max() * n * dt > MAX_EFOLDINGS:
                damping *= 10
                first = False
                continue
            u_try = np.exp((-2j * np.pi * f_try - g_try) * dt)
            a_try, r_try, rss_try = fit(u_try)
            if rss_try < rss:
                accepted = True
                # a small gain after a damped (shortened) step says nothing
                # about the optimum; only an undamped step that gains
                # nothing, or a vanishing step, does
                converged = (first and rss - rss_try < tolerance * rss) or largest < 1e-10
                u, a, r, rss = u_try, a_try, r_try, rss_try
                damping = max(damping / 10, 1e-12)
                break
            damping *= 10
            first = False
        if not accepted or converged:
            break
    return u, a


def _candidates(u, d, residual, dt, n, real, residual_threshold):
    """Raw poles worth testing: finite, not absurdly growing or decaying, consistent.

    A real record's poles come in conjugate pairs and only the non-negative
    frequency of each pair is a candidate (its partner is implied).
    """
    f, gamma, _ = _pole_to_mode(u, d, dt, real)
    ok = (np.isfinite(gamma) & (np.abs(gamma) * n * dt <= MAX_EFOLDINGS)
          & (residual <= residual_threshold))
    if real:
        ok &= f >= 0
    return np.flatnonzero(ok)


#: Bytes of candidate columns built at once in the selection.
_SELECT_BYTES = 64 << 20


def _greedy_select(c, dt, f_min, f_max, real, u, idx, min_significance):
    """Forward selection of the candidate poles ``u[idx]`` by significance.

    Each round adds the candidate whose column(s), orthogonalised against the
    poles already chosen, remove the most residual power, if that power is
    at least ``min_significance``^2 noise variances per quadrature (the
    chi-square of the amplitude fit). The noise variance starts from the
    record's own periodogram inside the window, an upper bound, and is
    re-read from the residual after every acceptance, so a weak mode hidden
    under a strong one's leakage is found once the strong one is in the
    model, and a mode fitted to noise never lowers the level it is judged
    against. Returns the chosen candidate indices (in order of acceptance)
    and the noise variance.

    A complex record's pole is one complex column; a real record's is the
    two real columns Re v and -Im v (Re(a v) = a_r Re v - a_i Im v). The
    gains are computed for blocks of candidates at once.
    """
    N = c.size
    r = c.astype(float if real else complex)
    sigma2 = _noise_variance(c, dt, f_min, f_max, real, 0)
    chosen = []
    Qsel = np.zeros((N, 0), float if real else complex)
    remaining = np.array(idx, int)
    block = max(1, _SELECT_BYTES // (16 * N))
    while remaining.size and sigma2 > 0:
        gains = np.zeros(remaining.size)
        for j0 in range(0, remaining.size, block):
            cand = remaining[j0:j0 + block]
            V = _vandermonde(u[cand], N)                    # N x B
            if real:
                W1 = V.real - Qsel @ (Qsel.T @ V.real)
                W2 = -V.imag - Qsel @ (Qsel.T @ (-V.imag))
                g11 = np.einsum("nk,nk->k", W1, W1)
                g22 = np.einsum("nk,nk->k", W2, W2)
                g12 = np.einsum("nk,nk->k", W1, W2)
                p1 = W1.T @ r
                p2 = W2.T @ r
                det = g11 * g22 - g12 ** 2
                # the two-column fit, or the better single column where
                # the pair is (near) dependent (a pole at 0 or Nyquist)
                single = np.maximum(p1 ** 2 / np.maximum(g11, 1e-300),
                                    p2 ** 2 / np.maximum(g22, 1e-300))
                with np.errstate(divide="ignore", invalid="ignore"):
                    pair = (g22 * p1 ** 2 - 2 * g12 * p1 * p2 + g11 * p2 ** 2) / det
                ok = det > 1e-16 * np.maximum(g11 * g22, 1e-300)
                gains[j0:j0 + cand.size] = np.where(ok, pair, single)
            else:
                W = V - Qsel @ (Qsel.conj().T @ V)
                norms = np.einsum("nk,nk->k", W.conj(), W).real
                proj = W.conj().T @ r
                gains[j0:j0 + cand.size] = np.abs(proj) ** 2 / np.maximum(norms, 1e-300)
        best = int(np.argmax(gains))
        quad = sigma2 if real else 0.5 * sigma2
        if gains[best] < min_significance ** 2 * quad:
            break
        k = remaining[best]
        cols = _columns(u[k], N, real)
        cols = cols - Qsel @ (Qsel.conj().T @ cols)
        q, rr = np.linalg.qr(cols)
        keep = np.abs(np.diag(rr)) > 1e-8 * max(np.linalg.norm(cols), 1e-300)
        q = q[:, keep]
        chosen.append(k)
        remaining = np.delete(remaining, best)
        r = r - q @ (q.conj().T @ r)
        Qsel = np.concatenate([Qsel, q], axis=1)
        sigma2 = _noise_variance(r, dt, f_min, f_max, real, len(chosen))
    return np.array(chosen, int), sigma2


def _columns(u, n, real):
    """Model columns of one pole: one complex column, or two real ones for a real record."""
    v = _vandermonde(np.atleast_1d(u), n)[:, 0]
    if real:
        cols = np.stack([v.real, -v.imag], axis=1)      # Re(a v) = a_r Re v - a_i Im v
    else:
        cols = v[:, None]
    return cols


def _model(modes, dt, n, real):
    """``sum a exp(-2 pi i f t - gamma t)`` (its real part for a real record)."""
    if not modes:
        return np.zeros(n, float if real else complex)
    u = np.array([np.exp((-2j * np.pi * f - g) * dt) for f, g, _ in modes])
    a = np.array([a for _, _, a in modes], complex)
    fit = _vandermonde(u, n) @ a
    return fit.real if real else fit


def _with_conjugates(u, real):
    """A real record's model needs each pole's conjugate partner as well."""
    return np.concatenate([u, np.conj(u)]) if real else u


def _pole_to_mode(u, d, dt, real):
    """``(f, gamma, amplitude)`` from a pole; real records double the amplitude."""
    f = -np.angle(u) / (2 * np.pi * dt)
    gamma = -np.log(np.abs(u)) / dt
    a = 2 * d if real else d
    return f, gamma, a


def _bars(c, dt, f_min, f_max, real, u, a):
    """``(modes, sigma2, std)``: the fitted modes, the noise they leave, their bounds."""
    N = c.size
    f = -np.angle(u) / (2 * np.pi * dt)
    gamma = -np.log(np.abs(u)) / dt
    modes = list(zip(f, gamma, a))
    if modes:
        sigma2 = _noise_variance(c - _model(modes, dt, N, real), dt, f_min, f_max,
                                 real, len(modes))
    else:
        sigma2 = _noise_variance(c, dt, f_min, f_max, real, 0)
    return modes, sigma2, cramer_rao(modes, dt, N, sigma2, real=real)


def _package(c, dt, f_min, f_max, real, u, d, residual, *, residual_threshold,
             min_significance, min_Q, error_sigma, do_polish):
    """Significance selection, polish, noise level, error bars, window; sorted by f.

    Every significant pole, inside the window or not, stays in the model
    through the polish and the noise estimate (an out-of-window mode left
    out would bias both); only the in-window ones are returned.
    """
    N = c.size
    idx = _candidates(u, d, residual, dt, N, real, residual_threshold)
    keep, sigma2 = _greedy_select(c, dt, f_min, f_max, real, u, idx, min_significance)
    u, residual = u[keep], residual[keep]
    if do_polish and u.size:
        # Polish, then judge each polished pole by its joint significance
        # |a| / std(|a|) from the full Fisher matrix: the forward selection
        # tests a pole against the others' amplitudes only, so a raw pole a
        # few bars off the truth lets a second pole explain its misfit
        # "significantly"; once they are polished together, with the
        # others' f and gamma free, that second pole explains only noise.
        # Drop the least significant and polish again while that happens.
        for _ in range(8):
            u, a = polish(c, dt, u, real=real)
            modes, sigma2, std = _bars(c, dt, f_min, f_max, real, u, a)
            signif = np.where(std[:, 2] > 0, 1.0 / np.maximum(std[:, 2], 1e-300), np.inf)
            if u.size == 0 or signif.min() >= min_significance:
                break
            keep = np.arange(u.size) != int(np.argmin(signif))
            u, residual = u[keep], residual[keep]
        if u.size == 0:
            a = np.zeros(0, complex)
    else:
        a = _amplitudes(c, _with_conjugates(u, real))[:u.size]
        a = 2 * a if real else a
    modes, sigma2, std = _bars(c, dt, f_min, f_max, real, u, a)
    f = np.array([m[0] for m in modes])
    out = []
    for k in np.argsort(f):
        fk, gk, ak = modes[k]
        if not f_min <= fk <= f_max:
            continue
        Q = math.pi * fk / gk if gk != 0 else math.inf
        if min_Q is not None and not Q > min_Q:
            continue
        df, dg, dlna, _ = error_sigma * std[k]
        # Q = pi f / gamma, with f and gamma uncorrelated to first order in
        # the short-record limit (their Fisher cross term vanishes)
        dQ = abs(Q) * math.hypot(df / fk if fk else 0.0, dg / gk if gk else math.inf)
        signif = 1.0 / std[k, 2] if std[k, 2] > 0 else math.inf   # |a| / std(|a|)
        err = ResonanceError(float(df), float(dQ), float(abs(ak) * dlna),
                             float(residual[k]), float(signif))
        out.append(Resonance(float(fk), float(Q), complex(ak), err))
    return out


def find_resonances(signal, dt: float, f_min: float, f_max: float, *,
                    method: str = "fdm", min_significance: float = 6.0,
                    residual_threshold: float = 0.1, error_sigma: float = 3.0,
                    min_Q: float | None = None, polish: bool = True,
                    background: bool = True, **options) -> list[Resonance]:
    """Resonances of a uniformly sampled record inside ``[f_min, f_max]``.

    ``signal`` is a 1-D real or complex array sampled every ``dt`` seconds;
    the window is in Hz. Returns ``Resonance`` four-tuples
    ``(f, Q, amplitude, error)`` sorted by frequency, where ``error`` carries
    ``error_sigma``-standard-deviation bars on f, Q and |amplitude|, the raw
    eigen-solution ``residual`` and the ``significance`` of the amplitude.

    A mode is reported when it lies in the window, its raw residual is at
    most ``residual_threshold``, its amplitude is at least
    ``min_significance`` noise standard deviations (a chi-square test of the
    power it removes, see ``_greedy_select``), and (when ``min_Q`` is given)
    its Q is above ``min_Q``. With ``polish`` the kept poles are moved to the
    maximum-likelihood fit before the bars are computed; without it the raw
    method's poles are reported with least-squares amplitudes. With
    ``background`` (FDM only) the matrix pencil's poles outside the window
    are carried as nuisance modes, so a strong mode just outside the window
    does not bias the ones inside.

    ``method`` is ``"fdm"`` (options ``density``, ``n_basis``,
    ``singular_threshold``) or ``"pencil"`` (options ``pencil_size``,
    ``max_rows``, ``rank``, ``rank_threshold``). The module docstring states
    the conventions and the numerical limits.
    """
    c, real = _as_record(signal)
    _check_window(f_min, f_max, dt, real)
    if method == "fdm":
        u, d, residual = fdm(c, dt, f_min, f_max, **options)
        if background:
            # A strong mode outside the window is in the record but not in
            # the basis; with noise it is smeared over spurious poles at the
            # window edge instead of being one clean far pole, and the
            # modes inside are biased. The global pencil supplies it as a
            # nuisance pole; its in-window poles are not used (the FDM's
            # are the accurate ones there).
            ub, db, rb = matrix_pencil(c, dt)
            fb = -np.angle(ub) / (2 * np.pi * dt)
            outside = (fb < f_min) | (fb > f_max)
            u = np.concatenate([u, ub[outside]])
            d = np.concatenate([d, db[outside]])
            residual = np.concatenate([residual, rb[outside]])
    elif method == "pencil":
        u, d, residual = matrix_pencil(c, dt, **options)
    else:
        raise ResonanceModel(f"unknown method {method!r}; use 'fdm' or 'pencil'")
    return _package(c, dt, f_min, f_max, real, u, d, residual,
                    residual_threshold=residual_threshold,
                    min_significance=min_significance, min_Q=min_Q,
                    error_sigma=error_sigma, do_polish=polish)
