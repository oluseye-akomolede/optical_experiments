"""Time-domain Maxwell on a graded Yee grid.

WHERE EACH COMPONENT LIVES. A tensor product of three per-axis node lists. Each
axis carries a primal list (the E nodes) and a dual list (the half-nodes between
them). A component is DUAL along its own axis and PRIMAL along the other two:

    Ex(i+1/2, j, k)     Ey(i, j+1/2, k)     Ez(i, j, k+1/2)
    Hx(i, j+1/2, k+1/2) Hy(i+1/2, j, k+1/2) Hz(i+1/2, j+1/2, k)

so each H component is the complement of the E one it curls. Two consequences
are worth stating because both are easy to get backwards:

* Two E components are tangential to every face, and in this layout they are the
  two that lie ON that face. A PEC wall is therefore "zero the tangential E on
  this layer", not an accident of where a slice happens to stop.
* Every difference in an update reads a neighbour that exists. The regions used
  below are exactly those index sets, and the untouched outer layers are exactly
  the layers a metal wall pins.

GRADED SPACINGS. Each axis carries ``primal[i]`` (E node i to i+1) and
``dual[i]``, the distance between the half-nodes either side of E node i, i.e.
``(primal[i-1] + primal[i]) / 2``. A curl term divides by whichever family spans
the two samples it differences: two E nodes differ across a primal spacing, two
half-nodes across a dual one. A uniform grid is the special case with every
primal spacing equal. ``_node_thickness`` gives the cell width AROUND each
primal node (half a cell at the two ends), which is what a CPML ramp and the
energy weight are measured in.

THE ENERGY THE SCHEME CONSERVES. On a graded grid the plain sum of squares is
NOT conserved; the quadratic that is conserved is weighted by the cell width
around each sample, and the weights are the cell VOLUMES

    W(Ex) = cell_x * node_y * node_z      W(Hx) = node_x * cell_y * cell_z

and their cyclic permutations, with ``cell`` the width around a dual node. Each
weight pairing cancels the spacing in the update it belongs to, which is why the
balance holds term by term with no assumption that the grid is uniform. On a
uniform grid every weight is a constant and this reduces to the usual sum.

And it pairs E with H across the half step they are staggered by -- it is
``|E^n|^2`` and ``H^(n+1/2) . H^(n-1/2)``, not ``|H|^2`` at one instant. That
distinction is not pedantry: the one-instant form ripples at ``omega*dt/2``,
around 15% in a run at 12 cells per wavelength, which looks exactly like a
solver losing energy.

THE TIME STEP. Yee's scheme is stable only while the round trip over the cell
diagonal fits in one step, so ``dt <= 1 / (c * sqrt(sum 1/d_min^2))`` when the
minimum on each axis is used. Going over does not fail loudly -- it grows the
field until it overflows and reads as a physics result -- so the limit is
computed from the actual spacings, never assumed.

UNITS. ``eps_r`` is relative permittivity; geometry is in metres. ``dx = 1.0`` by
default, so a test can treat the grid as unit-less as long as it stays
consistent -- and the same code has to hold at 10 nm (plasmonics) as at 1 mm.
The energy reduction is therefore done in float64: at 10 nm the cell volumes are
~1e-26 m^3 against field samples of ~1e-16, so a squared sample times its volume
is ~1e-58, below float32's smallest normal (1.2e-38) -- a float32 reduction
reports exactly zero energy from a run that is behaving perfectly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

C0 = 299792458.0                     # m/s
EPS0 = 8.8541878128e-12              # F/m
MU0 = 1.0 / (EPS0 * C0 * C0)         # H/m
ETA0 = math.sqrt(MU0 / EPS0)         # ohm

BOUNDARIES = ("metal", "open")
# Per axis, the solver also takes "periodic" (k = 0 wrap-around faces). The
# form offers only BOUNDARIES until phase 6.
AXIS_BOUNDARIES = ("metal", "open", "periodic")
PROFILES = ("cw", "pulse", "drawn")
KINDS = ("point", "plane")
DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")
COMPONENTS = ("x", "y", "z")

SAFETY = 0.99                        # fraction of the Courant limit
MAX_DRAWN_SAMPLES = 10_000           # the spec's cap on a drawn waveform
CELLS_PER_WAVELENGTH = 10            # sets the resolvable frequency f_res
DRAWN_ENERGY_LIMIT = 0.01            # spectral energy allowed above f_res
# CPML grading, chosen by measuring the 10-cell layer's reflection on the
# 1-D reduction of this update (scripts/fdtd_cpml_reflection.py), over 0-75
# degrees and 10-40 cells per wavelength. Worst reflected amplitude: order 2 at
# R = 1e-12 (this setting) 3.6e-3, at 75 degrees and 40 cells/wavelength;
# order 3 at 1e-12 7.5e-3; order 2 at 1e-8 4.2e-3; order 3 at 1e-6 (the old
# setting) 1.5e-2. The price is normal incidence: 3.0e-4 here against the old
# setting's 2.6e-5 (20 cells/wavelength). Grazing incidence on a fine grid is
# what kappa = 1, alpha = 0 cannot fix by grading alone. R here is a knob on
# sigma_max, not the reflection you get.
PML_ORDER = 2
PML_REFLECTION = 1e-12
DEFAULT_PML_CELLS = 10


class FDTDError(Exception):
    """Anything this module refuses to run, named in the message."""


# --------------------------------------------------------------------------
# the time step
# --------------------------------------------------------------------------

def courant_limit(*spacings) -> float:
    """The largest stable ``dt`` for these spacings (metres, scalar or array).

    A single argument is taken as all three axes, so ``courant_limit(dx)`` is
    ``dx / (c * sqrt(3))`` -- the cell diagonal, not one edge, because the wave
    that runs corner to corner is the one that sets the step. An array uses its
    smallest entry, which is the tightest that axis gets.
    """
    if len(spacings) == 1:
        spacings = spacings * 3
    total = 0.0
    for axis, spacing in enumerate(spacings):
        values = np.asarray(spacing, dtype=np.float64)
        if values.size == 0:
            raise FDTDError(f"axis {axis} has no spacings")
        smallest = float(values.min())
        if not smallest > 0:
            raise FDTDError(
                f"axis {axis} has a spacing of {smallest!r}; spacings must be "
                f"positive")
        total += 1.0 / smallest ** 2
    if total <= 0:
        raise FDTDError("at least one spacing is needed")
    return 1.0 / (C0 * math.sqrt(total))


def courant_dt(*spacings, safety: float = SAFETY) -> float:
    """The step actually used: the limit times ``safety``, which must be < 1."""
    if not 0.0 < safety < 1.0:
        raise FDTDError(
            f"the safety factor must be inside (0, 1); {safety!r} is not. A "
            f"factor at or above one steps past the stability limit, which "
            f"grows the field until it overflows instead of failing here")
    return safety * courant_limit(*spacings)


# --------------------------------------------------------------------------
# source profiles
# --------------------------------------------------------------------------

def cw_profile(t: float, frequency: float, ramp_periods: float = 5.0) -> float:
    """A raised-cosine turn-on over ``ramp_periods``, then a pure tone.

    The ramp is what keeps a continuous drive from putting a step's worth of
    every frequency into the grid at t = 0.

    The profile is the exact time derivative of ``-envelope * cos(wt) / w``,
    so its running integral is that expression, which averages to zero once
    the ramp is over. ``envelope * sin(wt)`` alone does not: its integral
    settles about an offset of 1% of its swing, and a soft current with a net
    integral leaves a static field that no absorbing boundary removes. After
    the ramp the two are the same pure sine.
    """
    if t <= 0.0:
        return 0.0
    ramp = ramp_periods / frequency
    omega = 2.0 * math.pi * frequency
    if t >= ramp:
        envelope, slope = 1.0, 0.0
    else:
        envelope = 0.5 * (1.0 - math.cos(math.pi * t / ramp))
        slope = 0.5 * math.pi / ramp * math.sin(math.pi * t / ramp)
    return envelope * math.sin(omega * t) - slope * math.cos(omega * t) / omega


def pulse_profile(t: float, frequency: float, bandwidth: float,
                  start: float = 5.0) -> float:
    """A Gaussian-modulated carrier; ``bandwidth`` is the envelope's 1/e width.

    The envelope's spectrum is also Gaussian, with a 1/e half-width of
    ``bandwidth`` in Hz, so the drive covers ``frequency +/- bandwidth``.
    """
    sigma = 1.0 / (math.pi * bandwidth)
    offset = t - start * sigma
    return (math.exp(-(offset / sigma) ** 2)
            * math.sin(2.0 * math.pi * frequency * offset))


def _drawn_waveform(samples, duration: float, step: float) -> np.ndarray:
    """The drawing at times ``0, step, 2 step, ...`` before ``duration``, mean
    removed. The samples span ``[0, duration]`` evenly, both ends included,
    joined by straight lines; after ``duration`` the waveform is zero."""
    times = np.arange(int(math.ceil(duration / step))) * step
    times = times[times < duration]
    grid = np.linspace(0.0, duration, len(samples))
    values = np.interp(times, grid, samples)
    return values - values.mean()


def resolvable_frequency(spacings, n_max: float = 1.0) -> float:
    """f_res = c / (10 |n|_max Delta_max): the highest frequency the grid
    carries at ten cells per wavelength in its densest material, with
    Delta_max its largest cell spacing (per axis a scalar or primal lengths)."""
    largest = max(float(np.max(np.asarray(h, dtype=float))) for h in spacings)
    return C0 / (CELLS_PER_WAVELENGTH * float(n_max) * largest)


def drawn_energy_above(samples, duration: float, dt: float,
                       f_res: float) -> float:
    """The fraction of a drawn waveform's spectral energy above ``f_res``.

    Measured on the drawing as the grid would get it (mean removed, zero
    outside ``duration``, so its end steps count), sampled at the finer of dt
    and the drawing's own spacing: a drawing finer than dt is not decimated
    before it is judged, so detail dt would alias cannot hide.
    """
    samples = np.asarray(samples, dtype=float)
    step = min(float(dt), float(duration) / (len(samples) - 1))
    values = _drawn_waveform(samples, duration, step)
    padded = np.zeros(4 * len(values) + 2)
    padded[1:1 + len(values)] = values     # the zero before t = 0 is real too
    power = np.abs(np.fft.rfft(padded)) ** 2
    freqs = np.fft.rfftfreq(len(padded), step)
    total = float(power.sum())
    if not total > 0.0:
        raise FDTDError(
            "the drawn waveform is flat: with its mean removed nothing is left "
            "to drive the grid")
    return float(power[freqs > f_res].sum()) / total


class Source:
    """A soft current on one E component at one cell, or a plane wave.

    ``kind="point"``: soft means it adds to the field there instead of pinning
    it, so the cell still carries the wave it launched.

    ``kind="plane"``: a plane wave travelling along ``direction`` (``"+x"``,
    ``"-z"``, ...) with E along ``component``, which must be transverse to it.
    It is injected through the faces of a total-field/scattered-field ``box``
    ``((x0, y0, z0), (x1, y1, z1))`` of node indices, both corners inclusive:
    inside the box the grid carries the incident wave plus whatever the
    geometry scatters, outside it only the scattered part. ``amplitude`` is the
    incident E amplitude in V/m. A plane source has no cell.

    ``profile="drawn"``: ``samples`` (at most 10 000 dimensionless amplitudes,
    spanning ``[0, duration]`` evenly) times ``amplitude``. ``simulate`` binds
    it to its dt (``bind``): resampled to dt, mean removed, zero after
    ``duration``; between steps the value is read off straight lines.
    ``frequency`` is still the run's reference frequency.
    """

    def __init__(self, cell=None, *, frequency: float, profile: str = "pulse",
                 bandwidth: float | None = None, amplitude: float = 1.0,
                 component: str = "z", kind: str = "point",
                 direction: str | None = None, box=None,
                 samples=None, duration: float | None = None):
        if kind not in KINDS:
            raise FDTDError(f"source kind must be one of {KINDS}, got {kind!r}")
        if profile not in PROFILES:
            raise FDTDError(f"profile must be one of {PROFILES}, got {profile!r}")
        if component not in COMPONENTS:
            raise FDTDError(
                f"component must be one of {COMPONENTS}, got {component!r}")
        if not frequency > 0:
            raise FDTDError(f"frequency must be positive, got {frequency!r}")
        if profile == "pulse":
            if bandwidth is None:
                raise FDTDError("a pulse needs a bandwidth")
            if not bandwidth > 0:
                raise FDTDError(
                    f"bandwidth must be positive, got {bandwidth!r}")
        elif bandwidth is not None:
            raise FDTDError(
                f"a {profile} source has no bandwidth, got {bandwidth!r}; only "
                f"a pulse has one")
        self.samples = None
        self.duration = None
        self._table = None
        self._table_dt = None
        if profile == "drawn":
            if samples is None or duration is None:
                raise FDTDError("a drawn source needs samples and a duration")
            values = np.asarray(samples, dtype=float)
            if values.ndim != 1 or not 2 <= len(values) <= MAX_DRAWN_SAMPLES:
                raise FDTDError(
                    f"a drawn waveform is a list of 2 to {MAX_DRAWN_SAMPLES} "
                    f"samples, got shape {values.shape}")
            if not np.all(np.isfinite(values)):
                raise FDTDError("a drawn waveform's samples must be finite")
            if not (math.isfinite(duration) and duration > 0):
                raise FDTDError(
                    f"a drawn waveform's duration must be positive seconds, "
                    f"got {duration!r}")
            self.samples = values
            self.duration = float(duration)
        elif samples is not None or duration is not None:
            raise FDTDError(
                f"samples and a duration belong to profile='drawn', not "
                f"{profile!r}")
        self.direction = None
        self.axis = None
        self.box = None
        if kind == "point":
            if direction is not None or box is not None:
                raise FDTDError(
                    "a point source has a cell, not a direction or a box; "
                    "those belong to kind='plane'")
            if cell is None or len(tuple(cell)) != 3:
                raise FDTDError(f"a cell is three indices, got {cell!r}")
            self.cell = tuple(int(c) for c in cell)
        else:
            if cell is not None:
                raise FDTDError(
                    f"a plane wave has no cell (got {cell!r}); it enters "
                    f"through the faces of its box")
            if direction not in DIRECTIONS:
                raise FDTDError(
                    f"a plane wave's direction must be one of {DIRECTIONS}, "
                    f"got {direction!r}")
            if component == direction[1]:
                raise FDTDError(
                    f"a plane wave travelling {direction} cannot be polarised "
                    f"along {component}: E must be transverse to the direction "
                    f"of travel")
            if box is None or len(tuple(box)) != 2 or any(
                    len(tuple(corner)) != 3 for corner in box):
                raise FDTDError(
                    f"a plane wave needs a box ((x0, y0, z0), (x1, y1, z1)), "
                    f"got {box!r}")
            low, high = (tuple(int(i) for i in corner) for corner in box)
            if any(h <= l for l, h in zip(low, high)):
                raise FDTDError(
                    f"the plane wave's box {box!r} is empty on some axis; each "
                    f"upper corner index must exceed the lower one")
            self.cell = None
            self.direction = direction
            self.axis = COMPONENTS.index(direction[1])
            self.box = (low, high)
        self.frequency = float(frequency)
        self.profile = profile
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.amplitude = float(amplitude)
        self.component = component
        self.kind = kind

    def bind(self, dt: float):
        """Resample a drawn waveform to ``dt`` (see the class); a no-op for
        the other profiles, which are formulas in t."""
        if self.profile != "drawn" or self._table_dt == dt:
            return
        if self.duration < 2.0 * dt:
            raise FDTDError(
                f"the drawn waveform lasts {self.duration:.4g} s, under two "
                f"time steps ({dt:.4g} s each): nothing of it would be left "
                f"once its mean is removed")
        self._table = _drawn_waveform(self.samples, self.duration, dt)
        self._table_dt = dt

    def value(self, t: float) -> float:
        if self.profile == "drawn":
            if self._table is None:
                raise FDTDError(
                    "a drawn source is resampled to the run's dt; bind(dt) it "
                    "first (simulate does)")
            position = t / self._table_dt
            if position < 0.0 or position >= len(self._table):
                return 0.0          # after its last step it ramps to zero
            k = int(position)
            after = self._table[k + 1] if k + 1 < len(self._table) else 0.0
            return self.amplitude * float(
                self._table[k] + (position - k) * (after - self._table[k]))
        if self.profile == "cw":
            return self.amplitude * cw_profile(t, self.frequency)
        return self.amplitude * pulse_profile(t, self.frequency, self.bandwidth)

    def end_time(self) -> float:
        """When a pulse is done, or ``inf`` for a source that never stops."""
        if self.profile == "cw":
            return math.inf
        if self.profile == "drawn":
            return self.duration
        sigma = 1.0 / (math.pi * self.bandwidth)
        return 10.0 * sigma          # the envelope's centre plus 5 sigma


class PolarisationTerm:
    """The seam for a dispersive material (phase 2: Drude-Lorentz metals).

    Phase 1 runs with no terms at all. A term is anything with a callable
    ``advance(fields, dt, step)`` returning ``{component: increment_array}``,
    which the update adds to the named E components after the curl update and
    before the conductor mask. Nothing else in the core needs to know which
    materials exist.
    """

    name = "term"

    def advance(self, fields, dt, step):     # pragma: no cover - the base seam
        raise NotImplementedError(f"{self.name} must implement advance()")


# --------------------------------------------------------------------------
# dispersive metals: Rakić et al. 1998 Lorentz-Drude
# --------------------------------------------------------------------------

#: Angular frequency of one electronvolt, e / hbar (CODATA 2018, exact).
EV = 1.602176634e-19 / 1.054571817e-34


@dataclass(frozen=True)
class Metal:
    """One Lorentz-Drude fit: ``eps(w) = 1 + sum_j f_j wp^2 / (w_j^2 - w^2 - i w G_j)``.

    The first pole is the Drude term (``w_0 = 0``). Energies in eV, as the
    paper gives them; ``fit_ev`` is the photon-energy range the fit covers.
    The time convention is ``exp(-i w t)``, so a lossy metal has Im eps > 0
    (the paper writes the complex conjugate).
    """

    name: str
    plasma_ev: float
    poles: tuple                    # (f_j, w_j eV, Gamma_j eV), Drude first
    fit_ev: tuple                   # (lowest, highest) photon energy, eV


#: A. D. Rakić, A. B. Djurišić, J. M. Elazar and M. L. Majewski, "Optical
#: properties of metallic films for vertical-cavity optoelectronic devices",
#: Applied Optics 37, 5271-5283 (1998), Table 1 (Lorentz-Drude model), with
#: eps_inf = 1. Values as reproduced in Meep's materials library, which cites the
#: paper; they reproduce refractiveindex.info's Rakić-LD tables to 4 digits
#: (tests/test_fdtd_metals.py). ``fit_ev`` is the NARROWER of the two sources'
#: stated ranges (Meep and refractiveindex.info disagree for Al and Cr).
RAKIC_1998 = {
    "Ag": Metal("Ag", 9.01, ((0.845, 0.0, 0.048), (0.065, 0.816, 3.886),
                             (0.124, 4.481, 0.452), (0.011, 8.185, 0.065),
                             (0.840, 9.083, 0.916), (5.646, 20.29, 2.419)),
                (0.1, 5.0)),
    "Au": Metal("Au", 9.03, ((0.760, 0.0, 0.053), (0.024, 0.415, 0.241),
                             (0.010, 0.830, 0.345), (0.071, 2.969, 0.870),
                             (0.601, 4.304, 2.494), (4.384, 13.32, 2.214)),
                (0.2, 5.0)),
    "Cu": Metal("Cu", 10.83, ((0.575, 0.0, 0.030), (0.061, 0.291, 0.378),
                              (0.104, 2.957, 1.056), (0.723, 5.300, 3.213),
                              (0.638, 11.18, 4.305)),
                (0.1, 6.0)),
    "Al": Metal("Al", 14.98, ((0.523, 0.0, 0.047), (0.227, 0.162, 0.333),
                              (0.050, 1.544, 0.312), (0.166, 1.808, 1.351),
                              (0.030, 3.473, 3.382)),
                (0.1, 6.0)),
    "Cr": Metal("Cr", 10.75, ((0.168, 0.0, 0.047), (0.151, 0.121, 3.175),
                              (0.150, 0.543, 1.305), (1.149, 1.970, 2.676),
                              (0.825, 8.775, 1.335)),
                (0.1, 5.0)),
}
METALS = tuple(sorted(RAKIC_1998))


def _metal(metal) -> Metal:
    if isinstance(metal, Metal):
        return metal
    if metal not in RAKIC_1998:
        raise FDTDError(f"unknown metal {metal!r}; the library has {METALS}")
    return RAKIC_1998[metal]


def permittivity(metal, frequency):
    """The model's complex relative permittivity at ``frequency`` (Hz)."""
    metal = _metal(metal)
    w = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64)
    wp = metal.plasma_ev * EV
    eps = np.ones_like(w, dtype=complex)
    for f, w0, gamma in metal.poles:
        eps = eps + f * wp ** 2 / ((w0 * EV) ** 2 - w ** 2 - 1j * w * gamma * EV)
    return eps


def check_metal_band(metal, f_low: float, f_high: float):
    """Refuse a band that leaves the metal's fitted range, naming both."""
    metal = _metal(metal)
    to_ev = 2.0 * np.pi / EV
    low_ev, high_ev = f_low * to_ev, f_high * to_ev
    fit_low, fit_high = metal.fit_ev
    if low_ev < fit_low or high_ev > fit_high:
        raise FDTDError(
            f"the pulse band {low_ev:.3g}-{high_ev:.3g} eV "
            f"({f_low / 1e12:.4g}-{f_high / 1e12:.4g} THz) leaves the range "
            f"Rakić et al. fitted for {metal.name}, {fit_low:g}-{fit_high:g} eV; "
            f"move the centre frequency or narrow the bandwidth")


def lorentz_drude_coefficients(metal, dt: float) -> list:
    """Per pole, (alpha, xi, gamma) of ``P^{n+1} = alpha P^n + xi P^{n-1} +
    gamma E^n``: ``P'' + G P' + w0^2 P = eps0 f wp^2 E`` central-differenced."""
    metal = _metal(metal)
    wp = metal.plasma_ev * EV
    out = []
    for f, w0, gamma in metal.poles:
        w0, gamma = w0 * EV, gamma * EV
        denominator = 1.0 + 0.5 * gamma * dt
        out.append(((2.0 - (w0 * dt) ** 2) / denominator,
                    -(1.0 - 0.5 * gamma * dt) / denominator,
                    EPS0 * f * wp ** 2 * dt ** 2 / denominator))
    return out


#: A growth factor this close to 1 is rounding, not growth. The Drude pole's
#: static mode is a near-defective eigenvalue at exactly 1, which eigvals
#: returns up to 6e-8 high (measured over the five metals, 2-20 nm, 0.03-1 of
#: Courant); every real instability measured is above 0.16.
GROWTH_TOLERANCE = 1e-6


def lorentz_drude_growth(coefficients, dt: float, spacings,
                         samples: int = 64) -> float:
    """The largest per-step growth factor of the update inside the metal.

    Von Neumann on a homogeneous metal (eps_inf = 1): a lattice mode of the
    curl-curl with eigenvalue S obeys ``E^{n+1} - 2E^n + E^{n-1} = -S E^n -
    (dP^{n+1} - dP^n)/eps0`` with each pole's recursion. S runs from 0 to the
    lattice maximum ``4 (c dt)^2 sum 1/dx_a^2`` (the finest spacing per axis
    bounds a graded grid). The result is the spectral radius of the companion
    matrix, maximised over S; above 1 a mode grows every step.

    Exact for the scheme on a uniform metal, not a guess like ``w dt < 1``:
    on 20 nm gold that rule refused a dt this check passes.
    """
    poles = len(coefficients)
    s_max = 4.0 * (C0 * dt) ** 2 * sum(1.0 / float(h) ** 2 for h in spacings)
    worst = 0.0
    for s in np.linspace(0.0, s_max, samples):
        # state: E^n, E^{n-1}, then (Q_j^n, Q_j^{n-1}) with Q = P/eps0
        size = 2 + 2 * poles
        m = np.zeros((size, size))
        for j, (alpha, xi, gamma) in enumerate(coefficients):
            row = 2 + 2 * j
            m[row, 0] = gamma / EPS0
            m[row, row] = alpha
            m[row, row + 1] = xi
            m[row + 1, row] = 1.0
        # E^{n+1} = E^n - sum (Q^{n+1} - Q^n) + curl, where the curl term's
        # step-to-step change is -S E^n: written first-order in E as
        # E^{n+1} = (2 - S) E^n - E^{n-1} - sum (Q^{n+1} - 2Q^n + Q^{n-1})
        m[0, 0] = 2.0 - s
        m[0, 1] = -1.0
        for j in range(poles):
            row = 2 + 2 * j
            m[0] -= m[row]                        # -Q_j^{n+1}
            m[0, row] += 2.0
            m[0, row + 1] -= 1.0
        m[1, 0] = 1.0
        worst = max(worst, float(np.abs(np.linalg.eigvals(m)).max()))
    return worst


class LorentzDrude(PolarisationTerm):
    """A Rakić metal on a node mask, stepped with auxiliary differential equations.

    Each pole carries a polarisation density obeying
    ``P'' + G P' + w0^2 P = eps0 f wp^2 E``, central-differenced in time:
    ``P^{n+1}`` from ``P^n``, ``P^{n-1}`` and ``E^n`` (``prepare``, before the
    E update), then ``E^{n+1} = E^n + dt/eps0 curl H - (P^{n+1} - P^n)/eps0``
    (``advance``, after it).

    An E sample lies on the segment between two nodes; it takes the metal when
    either node is metal, so a sample on a metal face takes the metal. Such a
    sample must be updated with eps_inf = 1, but the core used ``eps_r`` at its
    array index, which is shared with the other two components and may be the
    neighbouring dielectric's; ``advance`` rescales the curl increment it sees
    (``E* - E^n``) to eps_inf = 1 for those samples.

    Storage is per E component on the bounding box of that component's metal
    samples only: two P arrays per pole, plus E^n and the summed dP.
    """

    def __init__(self, metal, region):
        self.metal = _metal(metal)
        self.name = f"Lorentz-Drude {self.metal.name}"
        self.region = np.asarray(region, dtype=bool)
        if self.region.ndim != 3:
            raise FDTDError(f"{self.name}: the region must be a 3-D node mask")
        self.parts: dict = {}

    def stable_dt(self, dt: float, spacings) -> float:
        """``dt`` if the metal is stable at it, else 0.98 of the largest that is.

        At the Courant dt the lattice's stiffest mode already uses (w dt)^2 =
        3.92 of the leapfrog's 4, and the metal adds about sum f wp^2 dt^2 on
        top, so the explicit ADE needs a smaller step: 5 nm gold at 0.99 of
        Courant grows by 1.16 per step.
        """
        def growth(trial):
            return lorentz_drude_growth(
                lorentz_drude_coefficients(self.metal, trial), trial, spacings)

        if growth(dt) <= 1.0 + GROWTH_TOLERANCE:
            return dt
        low, high = 0.0, 1.0                 # stable and unstable fractions
        while high - low > 1e-4:
            middle = 0.5 * (low + high)
            if growth(middle * dt) <= 1.0 + GROWTH_TOLERANCE:
                low = middle
            else:
                high = middle
        return 0.98 * low * dt

    def bind(self, *, shape, dt: float, inv_eps, xp, spacings=None,
             periodic=(False, False, False)):
        if tuple(self.region.shape) != tuple(shape):
            raise FDTDError(
                f"{self.name}: the region is {self.region.shape} but the grid "
                f"is {tuple(shape)}; they must be the same shape")
        f32 = xp.float32
        self.coefficients = lorentz_drude_coefficients(self.metal, dt)
        if spacings is not None:
            radius = lorentz_drude_growth(self.coefficients, dt, spacings)
            if radius > 1.0 + GROWTH_TOLERANCE:
                raise FDTDError(
                    f"{self.name}: dt = {dt:.3g} s is unstable in the metal (a "
                    f"step multiplies its stiffest mode by {radius:.4g}); "
                    f"refine the grid so dt shrinks")
        eps_host = 1.0 / (EPS0 * np.asarray(_host_result(inv_eps, xp),
                                            dtype=np.float64))
        # The samples the E update advances: half positions 0..n-2 along the
        # component's own axis; integer positions 1..n-2 between walls, 1..n-1
        # on a periodic axis (node n-1 is node 0, copied back after the terms).
        # The wall ranges on a periodic axis left row n-1 out of the metal: a
        # vacuum slot through every periodic slab.
        live = {component: tuple(
                    slice(0, -1) if a == axis else
                    slice(1, None) if periodic[a] else slice(1, -1)
                    for a in range(3))
                for axis, component in enumerate(COMPONENTS)}
        self.parts = {}
        for axis, component in enumerate(COMPONENTS):
            samples = self.region.copy()
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(0, -1), slice(1, None)
            samples[tuple(low)] |= self.region[tuple(high)]
            updated = np.zeros(shape, dtype=bool)
            updated[live[component]] = True
            samples &= updated
            if not samples.any():
                continue
            where = np.argwhere(samples)
            box = tuple(slice(int(lo), int(hi) + 1)
                        for lo, hi in zip(where.min(0), where.max(0)))
            keep = samples[box].astype(np.float32)
            size = keep.shape
            self.parts[component] = {
                "box": box,
                "keep": xp.asarray(keep, dtype=f32),
                # eps_r - 1 at metal samples: rescales the curl increment
                "rescale": xp.asarray((eps_host[box] - 1.0) * keep, dtype=f32),
                "now": [xp.zeros(size, dtype=f32) for _ in self.metal.poles],
                "before": [xp.zeros(size, dtype=f32) for _ in self.metal.poles],
                "e_then": xp.zeros(size, dtype=f32),
                "d_p": xp.zeros(size, dtype=f32),
            }

    def prepare(self, fields, dt, step):
        """Before the E update: P to n+1 from E^n, and E^n kept."""
        for component, part in self.parts.items():
            e_then = part["e_then"]
            e_then[...] = fields[component][part["box"]]
            drive = e_then * part["keep"]
            d_p = part["d_p"]
            d_p *= 0.0
            for index, (alpha, xi, gamma) in enumerate(self.coefficients):
                now, before = part["now"][index], part["before"][index]
                before *= xi                    # becomes P^{n+1} in place
                before += alpha * now
                before += gamma * drive
                d_p += before
                d_p -= now
                part["now"][index], part["before"][index] = before, now

    def advance(self, fields, dt, step):
        """After the E update: the rescaled curl and the polarisation current."""
        out = {}
        for component, part in self.parts.items():
            curl = fields[component][part["box"]] - part["e_then"]
            out[component] = (part["box"],
                              curl * part["rescale"] - part["d_p"] / EPS0)
        return out


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------

def _spacing_array(spacing, n: int, axis: int) -> np.ndarray:
    """One axis's primal spacings: a scalar broadcast, or ``n-1`` lengths."""
    values = np.asarray(spacing, dtype=np.float64)
    if values.ndim == 0:
        if not float(values) > 0:
            raise FDTDError(f"the spacing on axis {axis} must be positive")
        return np.full(n - 1, float(values))
    if values.shape != (n - 1,):
        raise FDTDError(
            f"axis {axis} has {n} nodes, so its spacings must be {n - 1} "
            f"lengths between them; got shape {values.shape}")
    if np.any(values <= 0):
        raise FDTDError(f"every spacing on axis {axis} must be positive")
    return values


def _dual_spacing(primal: np.ndarray) -> np.ndarray:
    """The span between the half-nodes either side of each primal E node.

    Element 0 is a half-cell and is never used: every primal node the E update
    touches has index >= 1.
    """
    dual = np.empty(primal.size)
    dual[0] = 0.5 * primal[0]
    dual[1:] = 0.5 * (primal[:-1] + primal[1:])
    return dual


def _node_thickness(primal: np.ndarray) -> np.ndarray:
    """The cell width around each primal node: a full cell inside, half at the end."""
    n = primal.size + 1
    thickness = np.empty(n)
    thickness[0] = 0.5 * primal[0]
    thickness[-1] = 0.5 * primal[-1]
    thickness[1:-1] = 0.5 * (primal[:-1] + primal[1:])
    return thickness


def _dual_thickness(primal: np.ndarray) -> np.ndarray:
    """The cell width around each dual node: the primal cell it sits in.

    Length ``n`` like the field arrays; the last slot is a ghost (no dual node
    there, and the field at that slot is never updated), kept so the weights can
    be built by broadcasting against a full-shape array.
    """
    return np.concatenate((primal, primal[-1:]))


def _along(values, axis: int):
    """A 1-D profile reshaped to broadcast along one axis of a 3-D field."""
    shape = [1, 1, 1]
    shape[axis] = -1
    return values.reshape(shape)


# --------------------------------------------------------------------------
# CPML
# --------------------------------------------------------------------------

def _pml_sigma(primal: np.ndarray, pml: int, family: str) -> np.ndarray:
    """The CPML conductivity at each sample of one family along one axis.

    ONE profile sigma(x) per slab, as a function of physical depth into it,
    sampled where each family actually sits: ``family="node"`` at the primal
    nodes (the E update's derivative along this axis), ``"cell"`` at the cell
    centres (the H update's). The slab is the outer ``pml`` primal cells, so a
    graded grid gets its ramp measured in the cells it actually has.

    Sampling one profile matters: when E took a ramp measured over its node
    thicknesses ((pml - 1/2) cells) and H one over pml cells, the two halves of
    the update saw two different layers, and a 10-cell CPML reflected 2.7% of a
    normally incident pulse at 20 cells per wavelength (order 3, R = 1e-6);
    one profile with the same grading reflects 2.6e-5 (measured on the 1-D
    reduction, scripts/fdtd_cpml_reflection.py).
    """
    nodes = np.concatenate(([0.0], np.cumsum(primal)))
    if family == "node":
        x = nodes
    elif family == "cell":
        x = nodes[:-1] + 0.5 * primal
    else:
        raise FDTDError(f"internal: CPML family {family!r}")
    length = float(nodes[-1])
    sigma = np.zeros(x.size)
    for thickness, depth in (
            (float(primal[:pml].sum()), nodes[pml] - x),
            (float(primal[-pml:].sum()), x - (length - float(primal[-pml:].sum())))):
        sigma_max = ((PML_ORDER + 1) * math.log(1.0 / PML_REFLECTION)
                     / (2.0 * ETA0 * thickness))
        inside = depth > 0
        sigma[inside] = np.maximum(
            sigma[inside], sigma_max * (depth[inside] / thickness) ** PML_ORDER)
    return sigma


def _pml_profiles(primal: np.ndarray, pml: int, dt: float, xp, family: str):
    """``(b, c)`` per sample of ``family`` (see ``_pml_sigma``) along one axis.

    The ramp is zero at the inner interface and reaches sigma_max at the outer
    face, so the layer absorbs more the deeper the wave gets, and the return
    trip is suppressed twice over. ``"node"`` profiles have one entry per
    primal node (the end nodes are walls the caller drops), ``"cell"`` ones
    one per primal cell.
    """
    n = primal.size + (1 if family == "node" else 0)
    if pml <= 0:
        b = np.ones(n)
        return xp.asarray(b, dtype=xp.float32), xp.asarray(b - 1.0, dtype=xp.float32)
    if primal.size + 1 <= 2 * pml:
        raise FDTDError(
            f"a CPML of {pml} cells does not fit in {primal.size + 1} nodes on "
            f"an axis: the interior would be empty (it needs more than "
            f"{2 * pml} nodes). Give more cells, or a smaller pml_cells")
    sigma = _pml_sigma(primal, pml, family)
    b = np.exp(-sigma * dt / EPS0)
    return (xp.asarray(b, dtype=xp.float32),
            xp.asarray(b - 1.0, dtype=xp.float32))     # kappa = 1, alpha = 0


# --------------------------------------------------------------------------
# probes and spectra
# --------------------------------------------------------------------------

def _normalise_probes(probes):
    """``(i, j, k)`` or ``(i, j, k, component)``, component defaulting to |E|."""
    out = []
    for probe in probes:
        if len(probe) == 3:
            out.append((tuple(int(c) for c in probe), "mag"))
        elif len(probe) == 4:
            cell, component = probe[:3], probe[3]
            if component not in COMPONENTS + ("mag",):
                raise FDTDError(
                    f"a probe component must be one of {COMPONENTS + ('mag',)}, "
                    f"got {component!r}")
            out.append((tuple(int(c) for c in cell), component))
        else:
            raise FDTDError(
                f"a probe is (i, j, k) or (i, j, k, component), got {probe!r}")
    return out


def spectrum(trace, dt: float, *, window: str | None = "hann", xp=None):
    """``(frequencies_hz, amplitude)`` from the rfft of a windowed trace.

    The window is applied because a recorded trace is a finite slice of a signal
    that is still ringing when it ends; without it, the discontinuity at the two
    ends lays a sinc skirt over every peak. It is built in numpy and moved to
    ``xp``, so both backends window identically.
    """
    if xp is None:
        xp = np
    values = xp.asarray(trace)
    n = int(values.shape[0])
    if n < 2:
        raise FDTDError(f"a spectrum needs at least 2 samples, got {n}")
    if window == "hann":
        scale = np.hanning(n)
    elif window in (None, "none"):
        scale = np.ones(n)
    else:
        raise FDTDError(f"unknown window {window!r}; use 'hann' or None")
    values = values.astype(xp.float64) * xp.asarray(scale)
    amplitude = xp.abs(xp.fft.rfft(values))
    frequencies = np.fft.rfftfreq(n, d=dt)
    return frequencies, amplitude


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# plane wave: total-field / scattered-field
# --------------------------------------------------------------------------

#: Absorbing cells at each end of the 1-D incident grid. It is one line of
#: cells, so a thick layer costs nothing and keeps its echo far below the
#: leakage the box is checked for.
INCIDENT_PML_CELLS = 40
#: Cells between the TF/SF box and both the CPML and every object (the spec's 3).
BOX_CLEARANCE = 3


class _PlaneWave:
    """The incident wave of a plane source and the TF/SF corrections it needs.

    The incident field lives on a 1-D Yee line along the direction of travel,
    with the SAME primal spacings along that axis and the same dt as the 3-D
    grid, padded at both ends and closed by a CPML. For a wave with no
    transverse variation the 3-D update collapses to exactly this 1-D update,
    so the incident field it carries has the 3-D grid's own numerical
    dispersion, and the corrections cancel the wave outside the box to
    rounding rather than to a dispersion mismatch.

    Only the polarisation component ``E_p`` and the one H component it drives,
    ``H_q``, are nonzero in the incident wave. A curl term reaches across a box
    face wherever one side of the difference is in the total-field region and
    the other in the scattered-field region; the fix adds the incident value
    of the far sample when the updated sample is inside (it wants a total) and
    subtracts it when the updated sample is outside (it wants a scattered
    field). That happens in eight places: H_q and E_p on the two faces normal
    to the direction of travel, H_a on the two faces normal to q, and E_a on
    the two faces normal to p (a being the direction's axis).

    Box convention: the faces are the node planes ``low`` and ``high``. A
    sample at an integer position along an axis is inside for ``low..high``,
    one at a half position for ``low+1/2 .. high-1/2``.
    """

    def __init__(self, source: Source, shape, primal, dual, eps_host,
                 pec_host, inv_eps, dt: float, pml_cells: int, xp,
                 metal_host=None, periodic=(False, False, False)):
        self.source = source
        self.xp = xp
        f32 = xp.float32
        a = source.axis
        p = COMPONENTS.index(source.component)
        q = 3 - a - p
        low, high = source.box
        self.a = a

        # The corrections are the plain curl update's, so the faces must sit
        # in the untouched interior, and the spec keeps them 3 cells clear of
        # the CPML: the first CPML-free E node is ``pml_cells``.
        # A periodic axis has no faces: the box must span it, node 0 to node
        # n-1, and the wave enters only through the two faces normal to its
        # direction of travel.
        for axis in range(3):
            if periodic[axis] and (low[axis], high[axis]) != (0, shape[axis] - 1):
                raise FDTDError(
                    f"the plane wave's box {source.box} must span the periodic "
                    f"axis {axis} from node 0 to node {shape[axis] - 1}: a "
                    f"periodic axis has no faces to inject through")
        first = int(pml_cells) + BOX_CLEARANCE
        for axis in range(3):
            if periodic[axis]:
                continue
            last = shape[axis] - 1 - int(pml_cells) - BOX_CLEARANCE
            if not (first <= low[axis] and high[axis] <= last):
                raise FDTDError(
                    f"the plane wave's box {source.box} is closer than "
                    f"{BOX_CLEARANCE} cells to the absorbing layer on axis "
                    f"{axis}: with {pml_cells} CPML cells on a "
                    f"{shape[axis]}-node axis its faces must lie in "
                    f"{first}..{last}")

        # The 1-D line carries the wave through ONE background medium, so
        # everything within BOX_CLEARANCE cells of a face, either side, must be
        # that medium: an object there would sit across the injection layer
        # (and the spec keeps objects 3 cells clear). Objects wholly inside
        # the inner region, or wholly outside, are what the box is for.
        outer = tuple(slice(None) if periodic[a] else
                      slice(low[a] - BOX_CLEARANCE, high[a] + BOX_CLEARANCE + 1)
                      for a in range(3))
        shell = np.ones(shape, dtype=bool)
        near = np.zeros(shape, dtype=bool)
        near[outer] = True
        if all(periodic[a] or high[a] - low[a] > 2 * BOX_CLEARANCE
               for a in range(3)):
            inner = tuple(slice(None) if periodic[a] else
                          slice(low[a] + BOX_CLEARANCE, high[a] - BOX_CLEARANCE + 1)
                          for a in range(3))
            shell[inner] = False
        shell &= near
        background = float(eps_host[low])
        foreign = shell & (eps_host != eps_host[low])
        if pec_host is not None:
            foreign |= shell & pec_host
        if metal_host is not None:
            foreign |= shell & metal_host
        if foreign.any():
            where = tuple(int(i) for i in np.argwhere(foreign)[0])
            if pec_host is not None and pec_host[where]:
                what = "a conductor"
            elif metal_host is not None and metal_host[where]:
                what = "a dispersive metal"
            else:
                what = (f"relative permittivity {float(eps_host[where]):.4g} "
                        f"against the background {background:.4g}")
            raise FDTDError(
                f"the plane wave's box {source.box} has an object within "
                f"{BOX_CLEARANCE} cells of its faces ({what} at node {where}); "
                f"the incident wave is defined in one background medium, so "
                f"keep every object at least {BOX_CLEARANCE} cells from the box")

        # ---- the 1-D incident line ---------------------------------------
        pad = INCIDENT_PML_CELLS + 4
        self.pad = pad
        line = np.concatenate((np.full(pad, primal[a][0]), primal[a],
                               np.full(pad, primal[a][-1])))
        length = line.size + 1
        self.P1 = xp.asarray(line, dtype=f32)
        self.D1 = xp.asarray(_dual_spacing(line)[1:], dtype=f32)  # nodes 1..L-2
        self.bh, self.ch = _pml_profiles(line, INCIDENT_PML_CELLS, dt, xp, "cell")
        be, ce = _pml_profiles(line, INCIDENT_PML_CELLS, dt, xp, "node")
        self.be, self.ce = be[1:-1], ce[1:-1]
        self.E1 = xp.zeros(length, dtype=f32)
        self.H1 = xp.zeros(length - 1, dtype=f32)
        self.psi_h = xp.zeros(length - 1, dtype=f32)
        self.psi_e = xp.zeros(length - 2, dtype=f32)
        # The curl's sign for this (direction axis, polarisation) pair:
        # dH_q/dt = -(sign/mu0) dE_p/da and dE_p/dt = -(sign/eps) dH_q/da.
        sign = 1.0 if p == (a + 1) % 3 else -1.0
        self.h_coef = -sign * dt / MU0
        self.e_coef = -sign * dt / (EPS0 * background)
        # The feed is a one-way TF/SF point on the line itself, two nodes
        # outside the 3-D grid on the side the wave comes from: nodes from the
        # feed onward carry the wave, the side behind it carries only what the
        # line scatters, into its CPML. NOT a hard source: a node clamped to
        # the profile is clamped to zero once the pulse ends, a wall that
        # traps the small reflections off a graded axis between it and the
        # box (measured: 2e-4 of the peak energy left on a +/-10% grading,
        # ringing on after 1200 steps).
        self.travel = 1.0 if source.direction[0] == "+" else -1.0
        if self.travel > 0:
            self.feed = INCIDENT_PML_CELLS + 2
            self.feed_h = self.feed - 1           # H sample behind the feed
        else:
            self.feed = length - 1 - (INCIDENT_PML_CELLS + 2)
            self.feed_h = self.feed
        self.dt = dt
        self.speed = C0 / math.sqrt(background)
        self.feed_width_h = float(line[self.feed_h])
        self.feed_width_e = float(_dual_spacing(line)[self.feed])
        # The incident H that goes with E = g(t -/+ x/v): from the line's own
        # update dH/dt = (h_coef/dt) dE/dx.
        self.h_of_e = -self.travel * self.h_coef / (dt * self.speed)

        # ---- the eight face corrections ----------------------------------
        # Each entry: (family, target component, target slice, coefficient
        # array, incident array name, incident slice, varies along a).
        self.corrections: list = []

        def position_is_half(family: str, component: int, axis: int) -> bool:
            # E_c is half along c and whole elsewhere; H_c the reverse.
            return (axis == component) == (family == "E")

        def inside(family, component, axis):
            if periodic[axis]:
                # every real sample: n-1 at half positions (slot n-1 is a
                # ghost -- refilled for H, unused for E), n at nodes
                if position_is_half(family, component, axis):
                    return 0, shape[axis] - 1
                return 0, shape[axis]
            if position_is_half(family, component, axis):
                return low[axis], high[axis]            # half indices lo..hi-1
            return low[axis], high[axis] + 1            # whole indices lo..hi

        def add(family: str, c: int, d: int):
            # the curl term of target (family, c) that differentiates along d
            term_sign = 1.0 if d == (c + 1) % 3 else -1.0
            for face, delta in (("low", -1.0), ("high", 1.0)):
                if family == "H":
                    # target at half index along d, incident E at node
                    index = low[d] - 1 if face == "low" else high[d]
                    incident_at = low[d] if face == "low" else high[d]
                    width = primal[d][index]
                    factor = -dt / MU0 * term_sign * delta / width
                    name = "E1"
                else:
                    # target at node along d, incident H at half index
                    index = low[d] if face == "low" else high[d]
                    incident_at = low[d] - 1 if face == "low" else high[d]
                    width = dual[d][index]
                    factor = dt * term_sign * delta / width
                    name = "H1"
                region = []
                for axis in range(3):
                    if axis == d:
                        region.append(slice(index, index + 1))
                    else:
                        region.append(slice(*inside(family, c, axis)))
                region = tuple(region)
                if family == "E":
                    coefficient = factor * inv_eps[region]
                else:
                    coefficient = xp.full(
                        tuple(r.stop - r.start for r in region), factor,
                        dtype=f32)
                if d == a:
                    span = slice(incident_at + pad, incident_at + pad + 1)
                    along = False
                else:
                    start, stop = inside(family, c, a)
                    span = slice(start + pad, stop + pad)
                    along = True
                self.corrections.append(
                    (family, COMPONENTS[c], region,
                     xp.asarray(coefficient, dtype=f32), name, span, along))

        add("H", q, a)      # H_q across the faces normal to a
        add("E", p, a)      # E_p across the faces normal to a
        if not periodic[q]:
            add("H", a, q)  # H_a across the faces normal to q
        if not periodic[p]:
            add("E", a, p)  # E_a across the faces normal to p

    def _apply(self, family: str, fields: dict):
        for entry_family, component, region, coefficient, name, span, along \
                in self.corrections:
            if entry_family != family:
                continue
            incident = getattr(self, name)[span]
            if along:
                incident = _along(incident, self.a)
            fields[component][region] += coefficient * incident

    def correct_h(self, h_fields: dict):
        """After the 3-D H update: uses the incident E at the same level, n."""
        self._apply("H", h_fields)

    def advance_h(self, step: int):
        """The incident H from n-1/2 to n+1/2 (n = ``step``)."""
        raw = (self.E1[1:] - self.E1[:-1]) / self.P1
        self.psi_h *= self.bh
        self.psi_h += self.ch * raw
        self.H1 += self.h_coef * (raw + self.psi_h)
        # The H sample behind the feed differenced the feed's total E; it
        # wants a scattered field, so take the injected E back out.
        injected = self.source.value(step * self.dt)
        self.H1[self.feed_h] -= (self.travel * self.h_coef * injected
                                 / self.feed_width_h)

    def correct_e(self, e_fields: dict):
        """After the 3-D E update: uses the incident H at n+1/2."""
        self._apply("E", e_fields)

    def advance_e(self, step: int):
        """The incident E from n to n+1 (n = ``step``)."""
        raw = (self.H1[1:] - self.H1[:-1]) / self.D1
        self.psi_e *= self.be
        self.psi_e += self.ce * raw
        self.E1[1:-1] += self.e_coef * (raw + self.psi_e)
        # The feed's E differenced the scattered H behind it; it wants a
        # total, so add the injected H there, at n+1/2 and half a cell back.
        lag = 0.5 * self.feed_width_h / self.speed
        injected = self.h_of_e * self.source.value((step + 0.5) * self.dt + lag)
        self.E1[self.feed] -= (self.travel * self.e_coef * injected
                               / self.feed_width_e)


def _axis_boundaries(boundary) -> tuple:
    """``boundary`` as one kind per axis: a single kind applies to all three."""
    sides = (boundary,) * 3 if isinstance(boundary, str) else tuple(boundary)
    if len(sides) != 3 or any(side not in AXIS_BOUNDARIES for side in sides):
        raise FDTDError(
            f"boundary must be one of {BOUNDARIES}, or one kind per axis "
            f"(x, y, z) from {AXIS_BOUNDARIES}; got {boundary!r}")
    return sides


def _set_slab(values, axis: int, index: int, slab) -> None:
    """Write ``slab`` into ``values`` at ``index`` along ``axis``."""
    where = [slice(None)] * 3
    where[axis] = index
    values[tuple(where)] = slab


def _take(values, axis: int, index: int):
    """The slab of ``values`` at ``index`` along ``axis``."""
    where = [slice(None)] * 3
    where[axis] = index
    return values[tuple(where)]


# Short-lived full-grid arrays alive at the step loop's busiest moment, the Ex
# update: the stale term_x left by the Hz update, term_y and term_z, and the
# three expression temporaries (dt*inv_eps[...], the difference, the product).
# Reviewer's trace, qitem-20261002014955-c62b2de2.
TRANSIENT_ARRAYS = 6
# cupy's pool keeps freed blocks it cannot reuse for differently shaped slice
# temporaries, and the CPML psi arrays are allocated lazily among them, so the
# device high-water exceeds the live set. Set from the Jetson rate table
# (6ffa8bd, cupy, 300 steps, 1 M and 4 M cells; floats per cell = peak bytes /
# 4 / cells): metal 14.10 / 14.16, open 27.65 / 27.87, open + 1/8 gold
# 33.65 / 33.83. These constants put every row at 0.97-0.99 of the estimate,
# never above it, so the 3.0 GB refusal errs on the safe side.
POOL_BASE = 0.4
POOL_PER_OPEN_AXIS = 0.6


def memory_estimate(shape, *, boundary="metal", metals=(), pec=False,
                    record_peak=False) -> int:
    """Bytes ``simulate`` holds at its peak on the device (float32 arrays).

    Counts, per cell: six fields, eps and 1/eps, the transients, the device
    pool's cached blocks, four CPML psi arrays per open axis (two E and two H
    derivatives along it), the conductor mask and its multiplier, and the peak
    |E_c| record (3 arrays). Each metal
    ``(name, region)`` adds, per component, a box over its samples (the region
    widened one node along the component) holding 2 per pole + 4 arrays, plus
    the solver's boolean union of the regions.
    """
    shape = tuple(int(s) for s in shape)
    cells = int(np.prod(shape))
    sides = _axis_boundaries(boundary)
    floats = (6 + 2 + TRANSIENT_ARRAYS + POOL_BASE
              + (4 + POOL_PER_OPEN_AXIS) * sides.count("open"))
    if pec:
        floats += 1 + 0.25                  # multiplier + boolean mask
    if record_peak:
        floats += 3                         # the E components only
    total = 4.0 * cells * floats
    if metals:
        total += cells                      # the boolean union of the regions
    for name, region in metals:
        where = np.argwhere(np.asarray(region, dtype=bool))
        if where.size == 0:
            continue
        extent = where.max(0) - where.min(0) + 1
        arrays = 2 * len(_metal(name).poles) + 4
        for axis in range(3):
            box = extent.copy()
            box[axis] = min(box[axis] + 1, shape[axis])
            total += 4.0 * int(np.prod(box)) * arrays
    return int(total)


# The spec's per-run limits (requirement "A run SHALL be refused above its
# limits before it is queued").
MAX_CELLS = 8_000_000
MAX_STEPS = 20_000
MAX_BYTES = 3.0e9
MAX_SECONDS = 30 * 60

# Measured Jetson rates in Mcell-steps/s (scripts/fdtd_rate.py, cupy, 300
# steps; queue row 4a5da177). Each run is (1 M cells, 4 M cells); "dispersive"
# is open faces with 1/8 of the box gold. Keep every run here: the rate a run
# is timed by is derived from them, never typed.
JETSON_RATE_RUNS = {
    "6ffa8bd": {"metal": (106.69, 114.10), "open": (57.53, 56.82),
                "dispersive": (47.53, 46.30)},
    "836c4b8": {"metal": (106.52, 111.00), "open": (55.73, 55.11),
                "dispersive": (45.67, 45.30)},
}
# Each setup takes the slowest row over every run. Unmeasured setups take the
# slower measured rate: periodic or mixed faces the open rate, metal walls with
# a metal the gold rate.
JETSON_RATES = {
    setup: 1e6 * min(row for run in JETSON_RATE_RUNS.values()
                     for row in run[setup])
    for setup in ("metal", "open", "dispersive")}
# Run-to-run spread is ~3% under background load. The time estimate divides by
# this so it errs long: over-estimating only refuses borderline runs, while
# under-estimating would admit runs over MAX_SECONDS.
RATE_SAFETY = 0.9


class EstimateError(FDTDError):
    """A run over one or more limits. ``exceeded`` holds one
    ``(name, value, limit, lever)`` per limit broken, in the spec's order."""

    def __init__(self, exceeded):
        self.exceeded = list(exceeded)
        super().__init__(" ".join(lever for *_, lever in self.exceeded))


def rate_setup(boundary="metal", metals=()) -> str:
    """The rate-table row a run is timed by (see ``JETSON_RATES``)."""
    if len(metals):
        return "dispersive"
    if all(side == "metal" for side in _axis_boundaries(boundary)):
        return "metal"
    return "open"


def estimate_run(nodes, *, duration: float, boundary="metal", metals=None,
                 pec=None, record_peak=False) -> dict:
    """Cells, dt, steps, device bytes and Jetson seconds for a planned run.

    ``nodes`` are the three node-coordinate arrays the run will use and
    ``metals``/``pec`` the voxeliser's masks on them. dt is found the way
    ``simulate`` finds it: 0.99 of Courant on the finest spacing per axis,
    then lowered by each metal's ``stable_dt``, so the step count is the
    one the run will take.
    """
    nodes = [np.asarray(n, dtype=float) for n in nodes]
    shape = tuple(len(n) for n in nodes)
    finest = tuple(float(np.diff(n).min()) for n in nodes)
    metals = dict(metals or {})
    dt = courant_dt(*finest)
    for name, region in metals.items():
        dt = min(dt, LorentzDrude(name, region).stable_dt(dt, finest))
    cells = int(np.prod(shape))
    # rounding noise must not add a step to a duration of whole steps
    steps = int(np.ceil(float(duration) / dt * (1 - 1e-9)))
    setup = rate_setup(boundary, metals)
    return {
        "shape": shape,
        "cells": cells,
        "dt": dt,
        "finest": min(finest),
        "steps": steps,
        "bytes": memory_estimate(shape, boundary=boundary,
                                 metals=list(metals.items()),
                                 pec=pec is not None, record_peak=record_peak),
        "setup": setup,
        "seconds": cells * steps / (RATE_SAFETY * JETSON_RATES[setup]),
    }


def check_limits(estimate: dict) -> None:
    """Raise ``EstimateError`` naming every limit ``estimate`` exceeds, the
    value that exceeded it, and the form setting that reduces it."""
    e = estimate
    finest = f"{e['finest'] * 1e9:.3g} nm"
    exceeded = []
    if e["cells"] > MAX_CELLS:
        exceeded.append(("cells", e["cells"], MAX_CELLS,
            f"The grid has {e['cells']:,} cells, over the limit of "
            f"{MAX_CELLS:,}. Increase dx to use fewer, larger cells."))
    if e["steps"] > MAX_STEPS:
        exceeded.append(("steps", e["steps"], MAX_STEPS,
            f"The run needs {e['steps']:,} time steps, over the limit of "
            f"{MAX_STEPS:,} (dt is {e['dt']:.3g} s, set by the finest cell, "
            f"{finest}). Shorten the duration, or increase dx."))
    if e["bytes"] > MAX_BYTES:
        exceeded.append(("memory", e["bytes"], MAX_BYTES,
            f"The run needs about {e['bytes'] / 1e9:.2f} GB of GPU memory, "
            f"over the limit of {MAX_BYTES / 1e9:.1f} GB. Increase dx, or use "
            f"metal walls instead of open ones (each open axis adds an "
            f"absorbing layer's storage across the whole grid)."))
    if e["seconds"] > MAX_SECONDS:
        exceeded.append(("time", e["seconds"], MAX_SECONDS,
            f"The run would take about {e['seconds'] / 60:.0f} minutes, over "
            f"the limit of {MAX_SECONDS // 60} minutes. Shorten the duration, "
            f"or increase dx."))
    if exceeded:
        raise EstimateError(exceeded)


def simulate(eps_r, source: Source, n_steps: int, probes=(),
             *, spacings=None, dx: float = 1.0,
             boundary: str = "metal", pec=None, pml_cells: int = DEFAULT_PML_CELLS,
             xp=None, record_energy: bool = False, energy_region=None,
             record_peak: bool = False, ade_terms=()) -> dict:
    """Run the grid and return the probe traces.

    ``eps_r`` ``(nx, ny, nz)`` is the relative permittivity of each node and it
    DEFINES the grid; ``pec`` is an optional boolean conductor mask of the same
    shape. ``spacings`` is a ``(sx, sy, sz)`` triple, each a scalar (uniform
    along that axis) or ``n-1`` primal lengths; ``dx`` is the shorthand for a
    uniform grid.

    ``boundary="metal"`` closes the outer faces with a PEC wall;
    ``boundary="open"`` puts a CPML of ``pml_cells`` on each face instead (the
    open box is still PEC-backed at the outermost layer, which is how a PML is
    terminated in practice). ``boundary`` may also be one kind per axis,
    ``(bx, by, bz)``, where an axis can additionally be ``"periodic"``: wrap-
    around faces at k = 0, the cell repeated forever along that axis. On a
    periodic axis node ``n-1`` IS node 0 (the period is the sum of the axis's
    ``n-1`` spacings), so ``eps_r``, ``pec`` and every metal region must agree
    on those two slabs; the solver updates node ``n-1`` and copies it to node
    0. A plane wave may run with periodic transverse axes (the slab-under-a-
    plane-wave case, e.g. V4), and then enters through the two box faces
    normal to its direction only. ``ade_terms`` are polarisation-current terms (see
    ``PolarisationTerm``), e.g. ``LorentzDrude("Au", mask)``. A term with a
    ``stable_dt`` can shrink dt below the Courant step (a metal's explicit ADE
    needs it); ``"dt"`` in the result is the step actually used.

    Returns ``{"traces": (n_probes, n_steps) float32, "dt", "steps",
    "spacings", "energy", "peak"}``, and every one of them is HOST numpy whatever ``xp``
    was: the arrays live on the device for the whole run and cross back in a
    single transfer at the end, so a consumer never has to know which backend
    ran (and never pays a per-element transfer of its own). ``energy`` is the
    cell-volume-weighted total per step, or ``None`` unless ``record_energy``
    asked for it -- it builds six float64 weight arrays and reduces six arrays
    every step, and each ``float(xp.sum(...))`` is a device sync, so it is a
    diagnostic: free on the numpy path, six syncs a step on cupy. Cheap on a
    24^3 test grid, not on a production one. ``energy_region``, three slices of
    node indices, restricts that sum to a sub-box (V3 measures the physical
    region with the CPML left out).

    ``record_peak`` adds ``"peak"``: per E component, the largest ``|E_c|``
    each sample reached over the whole run, as a full-shape array. It is a
    running maximum kept on the device (no sync a step), so a check that a
    region stayed quiet "throughout the run" reads every cell of it rather
    than a handful of probes.
    """
    xp = np if xp is None else xp
    f32 = xp.float32

    eps = xp.asarray(eps_r, dtype=f32)
    if eps.ndim != 3:
        raise FDTDError(f"eps_r must be 3-D (nx, ny, nz), got shape {eps.shape}")
    if not bool(xp.all(eps > 0)):
        raise FDTDError("eps_r must be positive everywhere")
    shape = tuple(int(s) for s in eps.shape)
    nx, ny, nz = shape

    sides = _axis_boundaries(boundary)
    periodic = tuple(side == "periodic" for side in sides)
    if int(n_steps) < 1:
        raise FDTDError(f"n_steps must be at least 1, got {n_steps!r}")
    n_steps = int(n_steps)

    if spacings is None:
        spacings = (dx, dx, dx)
    if len(tuple(spacings)) != 3:
        raise FDTDError(
            f"spacings must be one (sx, sy, sz) triple, got {spacings!r}")
    primal = [_spacing_array(spacings[axis], shape[axis], axis)
              for axis in range(3)]
    dual = [_dual_spacing(p) for p in primal]
    dt = courant_dt(*(p.min() for p in primal))

    mask = None
    keep = None
    if pec is not None:
        mask = xp.asarray(pec, dtype=bool)
        if tuple(mask.shape) != shape:
            raise FDTDError(
                f"the conductor mask is {tuple(mask.shape)} but the grid is "
                f"{shape}; they must be the same shape")
        # 0 inside the conductor, 1 elsewhere. The step loop MULTIPLIES by it
        # rather than writing E[mask] = 0: a boolean-mask assignment is an
        # indexing operation whose cost depends on the mask, and on a device
        # backend it can need the mask's true-count on the host, a sync per
        # component per step. A multiply is one elementwise kernel and never
        # leaves the device.
        keep = 1.0 - mask.astype(f32)

    terms = list(ade_terms)
    for term in terms:
        if not callable(getattr(term, "advance", None)):
            raise FDTDError(
                f"{term!r} is not a polarisation term: it has no advance()")
    # A dispersive term can need a dt below the Courant one (see
    # lorentz_drude_growth); every later use of dt (CPML, plane wave, terms)
    # must see the reduced value, so this comes before all of them.
    finest = tuple(float(p.min()) for p in primal)
    for term in terms:
        stable_dt = getattr(term, "stable_dt", None)
        if stable_dt is not None:
            dt = min(dt, stable_dt(dt, finest))

    # A drawn waveform is judged and resampled at the dt the run really uses.
    # The guard here sees only the dielectric index; the queue-time check,
    # which knows the metals' |n| at the run's frequencies, is the one that
    # stops a job before it is queued.
    if source.profile == "drawn":
        f_res = resolvable_frequency(primal, math.sqrt(float(eps.max())))
        above = drawn_energy_above(source.samples, source.duration, dt, f_res)
        if above > DRAWN_ENERGY_LIMIT:
            raise FDTDError(
                f"{above:.1%} of the drawn waveform's spectral energy is above "
                f"{f_res:.4g} Hz, the highest frequency this grid resolves at "
                f"{CELLS_PER_WAVELENGTH} cells per wavelength (the limit is "
                f"{DRAWN_ENERGY_LIMIT:.0%}); smooth its edges, lengthen it, "
                f"or refine the grid")
        source.bind(dt)

    # A cell the update never touches is not a place you can drive: the curl for
    # it is never evaluated, so a source there would be silently ignored and
    # read as "the run did nothing". The live region is the update region: the
    # component's own axis runs [0, n-2], the other two [1, n-2].
    def live(component: str, cell):
        own = COMPONENTS.index(component)
        for axis, index in enumerate(cell):
            if not 0 <= index < shape[axis]:
                raise FDTDError(
                    f"the cell {cell} is outside the grid {shape} "
                    f"(axis {axis} has {shape[axis]} cells)")
            if axis == own:
                if index == shape[axis] - 1:
                    raise FDTDError(
                        f"the cell {cell} is on the last {component}-node of "
                        f"axis {axis}, where the update never touches it; the "
                        f"live range on its own axis is 0..{shape[axis] - 2}")
            elif periodic[axis]:
                if index == 0:
                    raise FDTDError(
                        f"the cell {cell} is on node 0 of the periodic axis "
                        f"{axis}, which is a copy of node {shape[axis] - 1} "
                        f"(the same point, one period on); put the source on "
                        f"node {shape[axis] - 1}")
            elif index in (0, shape[axis] - 1):
                raise FDTDError(
                    f"the cell {cell} is in the outer layer of axis {axis}, "
                    f"where a {component}-polarised source is on the wall and "
                    f"the update never touches it; move it inward")

    if source.kind == "plane":
        travel = source.axis
        transverse = [sides[axis] for axis in range(3) if axis != travel]
        if "metal" in sides:
            raise FDTDError(
                "a plane wave cannot run with boundary 'metal': a plane wave "
                "is an infinite wave arriving from outside, and a closed metal "
                "box has no outside -- the walls would reflect it back into "
                "the box for the whole run. Use boundary 'open' (or periodic "
                "faces across the direction of travel)")
        if sides[travel] != "open":
            raise FDTDError(
                f"a plane wave travelling {source.direction} needs an open "
                f"boundary along {COMPONENTS[travel]}, got "
                f"{sides[travel]!r}: on a periodic axis the wave would leave "
                f"through one face and come back in through the other forever")
        if len(set(transverse)) != 1:
            raise FDTDError(
                f"a plane wave needs both axes across its direction of travel "
                f"open or both periodic, got {tuple(sides)}; mixing them is "
                f"not supported")
    else:
        live(source.component, source.cell)
    if source.kind == "point" and mask is not None and bool(mask[source.cell]):
        raise FDTDError(
            f"the source cell {source.cell} is inside the conductor; a soft "
            f"current there is zeroed by the PEC update on the same step and "
            f"would excite nothing")

    probe_list = _normalise_probes(probes)
    for probe_cell, _component in probe_list:
        for axis, index in enumerate(probe_cell):
            if not 0 <= index < shape[axis]:
                raise FDTDError(
                    f"the probe cell {probe_cell} is outside the grid {shape} "
                    f"(axis {axis} has {shape[axis]} cells)")

    inv_eps = 1.0 / (EPS0 * eps)
    Ex = xp.zeros(shape, dtype=f32)
    Ey = xp.zeros(shape, dtype=f32)
    Ez = xp.zeros(shape, dtype=f32)
    Hx = xp.zeros(shape, dtype=f32)
    Hy = xp.zeros(shape, dtype=f32)
    Hz = xp.zeros(shape, dtype=f32)
    field = {"x": Ex, "y": Ey, "z": Ez}
    h_field = {"x": Hx, "y": Hy, "z": Hz}

    for term in terms:
        bind = getattr(term, "bind", None)
        if bind is not None:
            bind(shape=shape, dt=dt, inv_eps=inv_eps, xp=xp,
                 spacings=tuple(float(p.min()) for p in primal),
                 periodic=periodic)
    metal_host = None
    for term in terms:
        region = getattr(term, "region", None)
        if region is not None:
            metal_host = (np.asarray(region, dtype=bool) if metal_host is None
                          else metal_host | np.asarray(region, dtype=bool))

    # On a periodic axis the first and last node slabs are one point; a
    # material that differs there would be two different answers for it.
    if any(periodic):
        eps_host = np.asarray(_host_result(eps, xp))
        mask_host = (None if mask is None
                     else np.asarray(_host_result(mask, xp)))
        for axis in range(3):
            if not periodic[axis]:
                continue
            for what, values in (("eps_r", eps_host), ("the conductor mask",
                                 mask_host), ("a metal region", metal_host)):
                if values is None:
                    continue
                if not np.array_equal(_take(values, axis, 0),
                                      _take(values, axis, shape[axis] - 1)):
                    raise FDTDError(
                        f"{what} differs between node 0 and node "
                        f"{shape[axis] - 1} on the periodic axis {axis}; on a "
                        f"periodic axis those two nodes are the same point, "
                        f"so they must hold the same material")

    plane = None
    if source.kind == "plane":
        plane = _PlaneWave(
            source, shape, primal, dual, np.asarray(_host_result(eps, xp)),
            None if mask is None else np.asarray(_host_result(mask, xp)),
            inv_eps, dt, pml_cells, xp, metal_host=metal_host,
            periodic=periodic)

    P = [xp.asarray(p, dtype=f32) for p in primal]      # n-1 primal lengths

    # CPML: one sigma(x) per axis, sampled per family. The E update's derivative
    # along an axis sits at the primal nodes, the H update's at the cell
    # centres; both read the SAME profile at their own positions.
    bc: dict = {}
    for axis in range(3):
        if sides[axis] == "open":
            bc[(axis, "primal")] = _pml_profiles(
                primal[axis], int(pml_cells), dt, xp, "node")
            bc[(axis, "dual")] = _pml_profiles(
                primal[axis], int(pml_cells), dt, xp, "cell")

    # The E update's range per axis at the INTEGER positions: 1..n-2 between
    # two walls, 1..n-1 on a periodic axis (node n-1 is node 0, whose lower
    # neighbour is the half sample n-2 and whose upper one is the ghost half
    # sample n-1, refilled from half sample 0 before every E update). ``lower``
    # is the same range shifted down one, ``span`` the dual widths there.
    nodes, lower, span = [], [], []
    for axis in range(3):
        n = shape[axis]
        if periodic[axis]:
            nodes.append(slice(1, n))
            lower.append(slice(0, n - 1))
            wrap = 0.5 * (primal[axis][-1] + primal[axis][0])
            span.append(xp.asarray(np.append(dual[axis][1:], wrap), dtype=f32))
        else:
            nodes.append(slice(1, n - 1))
            lower.append(slice(0, n - 2))
            span.append(xp.asarray(dual[axis][1:], dtype=f32))
    sx, sy, sz = nodes
    lx, ly, lz = lower

    psi: dict = {}
    coef: dict = {}

    def corrected(raw, key: tuple, axis: int, family: str):
        """The CPML-corrected derivative, or ``raw`` off an open axis."""
        if sides[axis] != "open":
            return raw
        b1d, c1d = bc[(axis, family)]
        if family == "primal":
            # the outer primal nodes are pinned walls, never corrected
            b1d, c1d = b1d[1:-1], c1d[1:-1]
        if b1d.size != raw.shape[axis]:
            raise FDTDError(
                f"internal: the CPML profile for axis {axis} ({family}) has "
                f"{b1d.size} nodes but the term has {raw.shape[axis]}")
        entry = coef.get(key)
        if entry is None:
            entry = coef[key] = (_along(b1d, axis), _along(c1d, axis))
            psi[key] = xp.zeros(raw.shape, dtype=f32)
        store = psi[key]
        store *= entry[0]
        store += entry[1] * raw
        return raw + store                    # kappa = 1

    traces = xp.zeros((len(probe_list), n_steps), dtype=f32)
    energy: list[float] | None = [] if record_energy else None
    peak = ({c: xp.zeros(shape, dtype=f32) for c in COMPONENTS}
            if record_peak else None)
    if energy_region is None:
        region = (slice(None),) * 3
    else:
        region = tuple(energy_region)
        if len(region) != 3 or not all(isinstance(s, slice) for s in region):
            raise FDTDError(
                f"energy_region is three slices of node indices, got "
                f"{energy_region!r}")
        if any(len(range(*s.indices(n))) == 0 for s, n in zip(region, shape)):
            raise FDTDError(f"energy_region {energy_region!r} is empty on this "
                            f"{shape} grid")
    dt_over_mu0 = dt / MU0

    if energy is not None:
        # Cell volumes around each sample. These are what make the quadratic
        # conserved on a graded grid; on a uniform grid they are all equal and
        # the sum reduces to the usual one.
        node = [_node_thickness(p) for p in primal]
        wide = [_dual_thickness(p) for p in primal]
        for axis in range(3):
            if periodic[axis]:
                # Node 0 is a copy of node n-1, counted once with the full
                # width around it; the ghost half sample n-1 is a copy of
                # half sample 0 and not counted at all.
                node[axis][-1] = 0.5 * (primal[axis][-1] + primal[axis][0])
                node[axis][0] = 0.0
                wide[axis][-1] = 0.0

        def volumes(dual_axes):
            axes = [wide[a] if a in dual_axes else node[a] for a in range(3)]
            return xp.asarray(_outer(axes), dtype=xp.float64)[region]

        # Each component is dual along its own axis if it is E, and along the
        # other two if it is H.
        e_weights = {c: volumes({a}) for a, c in enumerate(COMPONENTS)}
        h_weights = {c: volumes({(a + 1) % 3, (a + 2) % 3})
                     for a, c in enumerate(COMPONENTS)}
        eps64 = xp.asarray(eps, dtype=xp.float64)[region]

    for step in range(n_steps):
        t = step * dt
        if energy is not None:
            h_before = (Hx.copy(), Hy.copy(), Hz.copy())      # H at n-1/2

        # ---- H from the curl of E (E at n, H advances to n+1/2) ----------
        # Hx at (i, j+1/2, k+1/2): live on the full x range, y <= ny-2, z <= nz-2
        term_z = (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / _along(P[1], 1)   # dEz/dy
        term_y = (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / _along(P[2], 2)   # dEy/dz
        term_z = corrected(term_z, ("Hx", "z"), 1, "dual")
        term_y = corrected(term_y, ("Hx", "y"), 2, "dual")
        Hx[:, :-1, :-1] -= dt_over_mu0 * (term_z - term_y)

        # Hy at (i+1/2, j, k+1/2): live on the full y range, x <= nx-2, z <= nz-2
        term_x = (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / _along(P[2], 2)   # dEx/dz
        term_z = (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / _along(P[0], 0)   # dEz/dx
        term_x = corrected(term_x, ("Hy", "x"), 2, "dual")
        term_z = corrected(term_z, ("Hy", "z"), 0, "dual")
        Hy[:-1, :, :-1] -= dt_over_mu0 * (term_x - term_z)

        # Hz at (i+1/2, j+1/2, k): live on the full z range, x <= nx-2, y <= ny-2
        term_y = (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / _along(P[0], 0)   # dEy/dx
        term_x = (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / _along(P[1], 1)   # dEx/dy
        term_y = corrected(term_y, ("Hz", "y"), 0, "dual")
        term_x = corrected(term_x, ("Hz", "x"), 1, "dual")
        Hz[:-1, :-1, :] -= dt_over_mu0 * (term_y - term_x)

        if plane is not None:
            plane.correct_h(h_field)          # uses the incident E at n
            plane.advance_h(step)             # incident H to n+1/2

        # Periodic axes: the H samples at HALF positions along the axis have
        # one more beyond the last, the ghost slot n-1, which is half sample 0
        # one period on. The E update at node n-1 reads it.
        for axis in range(3):
            if periodic[axis]:
                for component, values in h_field.items():
                    if COMPONENTS.index(component) != axis:
                        _set_slab(values, axis, shape[axis] - 1,
                                  _take(values, axis, 0))

        if energy is not None:
            # The quadratic the scheme conserves exactly pairs E and H ACROSS
            # the half step they are staggered by:
            #     1/2 eps |E^n|^2  +  1/2 mu H^(n+1/2) . H^(n-1/2)
            # Both E^n and the H pair are in hand RIGHT HERE, between the half
            # step that moved H and the full step that moves E. Measured one
            # update later, with E^(n+1), the same expression is the conserved
            # one plus dt<E^n, curl H^(n+1/2)> -- an O(dt) term that swings with
            # the field, and the run reads as a solver losing three quarters of
            # its energy and getting it back. Multiplying H by itself at a
            # single instant instead ripples at omega*dt/2, the same mistake at
            # O(dt^2).
            #
            # The sum is weighted by cell volume and done in float64, because at
            # 10 nm neither the product nor the square survives float32: a
            # squared sample (~1e-36) times a volume (~7.5e-25) is ~1e-61, under
            # float32's smallest normal (1.2e-38). Both factors are cast, not
            # just the weights -- the field sample itself is only ~20 orders
            # above that floor, so squaring it in float32 overflows to zero for
            # any source much weaker than the one the tests drive with, and the
            # run then reports a conserved zero for an energetic box.
            e_total = 0.0
            for component, values in (("x", Ex), ("y", Ey), ("z", Ez)):
                wide_values = values[region].astype(xp.float64)
                e_total += float(xp.sum(
                    eps64 * e_weights[component] * wide_values * wide_values))
            h_total = 0.0
            for index, (component, values) in enumerate(
                    (("x", Hx), ("y", Hy), ("z", Hz))):
                now = values[region].astype(xp.float64)
                before = h_before[index][region].astype(xp.float64)
                h_total += float(xp.sum(h_weights[component] * now * before))
            energy.append(0.5 * (EPS0 * e_total + MU0 * h_total))

        for term in terms:                    # dispersive terms see E^n
            prepare = getattr(term, "prepare", None)
            if prepare is not None:
                prepare(field, dt, step)

        # ---- E from the curl of H (H at n+1/2, E advances to n+1) --------
        # Each integer-position axis runs over ``nodes`` (1..n-2, or 1..n-1
        # when periodic); see where they are built.
        # Ex at (i+1/2, j, k)
        term_y = (Hz[:-1, sy, sz] - Hz[:-1, ly, sz]) / _along(span[1], 1)
        term_z = (Hy[:-1, sy, sz] - Hy[:-1, sy, lz]) / _along(span[2], 2)
        term_y = corrected(term_y, ("Ex", "y"), 1, "primal")
        term_z = corrected(term_z, ("Ex", "z"), 2, "primal")
        Ex[:-1, sy, sz] += dt * inv_eps[:-1, sy, sz] * (term_y - term_z)

        # Ey at (i, j+1/2, k)
        term_z = (Hx[sx, :-1, sz] - Hx[sx, :-1, lz]) / _along(span[2], 2)
        term_x = (Hz[sx, :-1, sz] - Hz[lx, :-1, sz]) / _along(span[0], 0)
        term_z = corrected(term_z, ("Ey", "z"), 2, "primal")
        term_x = corrected(term_x, ("Ey", "x"), 0, "primal")
        Ey[sx, :-1, sz] += dt * inv_eps[sx, :-1, sz] * (term_z - term_x)

        # Ez at (i, j, k+1/2)
        term_x = (Hy[sx, sy, :-1] - Hy[lx, sy, :-1]) / _along(span[0], 0)
        term_y = (Hx[sx, sy, :-1] - Hx[sx, ly, :-1]) / _along(span[1], 1)
        term_x = corrected(term_x, ("Ez", "x"), 0, "primal")
        term_y = corrected(term_y, ("Ez", "y"), 1, "primal")
        Ez[sx, sy, :-1] += dt * inv_eps[sx, sy, :-1] * (term_x - term_y)

        if plane is not None:
            plane.correct_e(field)            # uses the incident H at n+1/2
            plane.advance_e(step)             # incident E to n+1

        # ---- dispersive materials (phase 2 adds terms; phase 1 has none) --
        for term in terms:
            for name, increment in term.advance(field, dt, step).items():
                if name not in COMPONENTS:
                    raise FDTDError(
                        f"the polarisation term {getattr(term, 'name', term)!r} "
                        f"returned an increment for {name!r}; E components are "
                        f"{COMPONENTS}")
                if isinstance(increment, tuple):  # (region, values)
                    region, values = increment
                    field[name][region] += values
                else:
                    field[name] += increment

        # ---- source ------------------------------------------------------
        if plane is None:
            field[source.component][source.cell] += dt * source.value(t)

        # ---- PEC ---------------------------------------------------------
        if keep is not None:
            Ex *= keep
            Ey *= keep
            Ez *= keep
        # The metal faces, TANGENTIAL E only: two components per face, and in
        # this layout those are exactly the layers the update above never
        # touches. The normal component of E is not constrained by a PEC
        # wall, and pinning it would be a different boundary condition.
        if sides[0] == "metal":
            Ey[0, :, :] = 0.0       # x faces: Ey, Ez
            Ez[0, :, :] = 0.0
            Ey[-1, :, :] = 0.0
            Ez[-1, :, :] = 0.0
        if sides[1] == "metal":
            Ex[:, 0, :] = 0.0       # y faces: Ex, Ez
            Ez[:, 0, :] = 0.0
            Ex[:, -1, :] = 0.0
            Ez[:, -1, :] = 0.0
        if sides[2] == "metal":
            Ex[:, :, 0] = 0.0       # z faces: Ex, Ey
            Ey[:, :, 0] = 0.0
            Ex[:, :, -1] = 0.0
            Ey[:, :, -1] = 0.0

        # Periodic axes: node 0 is node n-1, which the update just advanced
        # (with every correction, term, source and conductor applied to it).
        for axis in range(3):
            if periodic[axis]:
                for component, values in field.items():
                    if COMPONENTS.index(component) != axis:
                        _set_slab(values, axis, 0,
                                  _take(values, axis, shape[axis] - 1))

        # ---- probes and energy -------------------------------------------
        for index, (cell, component) in enumerate(probe_list):
            if component == "mag":
                value = xp.sqrt(Ex[cell] ** 2 + Ey[cell] ** 2 + Ez[cell] ** 2)
            else:
                value = field[component][cell]
            traces[index, step] = value
        if peak is not None:
            for component in COMPONENTS:
                xp.maximum(peak[component], xp.abs(field[component]),
                           out=peak[component])

    if peak is not None:
        peak = {c: _host_result(values, xp) for c, values in peak.items()}
    return {"traces": _host_result(traces, xp), "dt": dt, "steps": n_steps,
            "spacings": [np.array(p) for p in primal], "energy": energy,
            "peak": peak}


def _host_result(value, xp):
    """Move a finished result array back to host numpy, once, at the end.

    The caller chose ``xp``; a cupy array is device-resident for the whole run
    and only crosses to the host here. cupy's arrays refuse ``np.asarray`` on
    purpose (that refusal is what stops a stray per-element transfer from
    looking free), so the copy comes from ``xp.asnumpy`` -- and numpy, which has
    no such function, needs no copy at all.
    """
    to_host = getattr(xp, "asnumpy", None)
    return np.asarray(value if to_host is None else to_host(value))


def _outer(axes):
    """The 3-D cell volume from three 1-D profiles, one per axis."""
    return (axes[0][:, None, None] * axes[1][None, :, None]
            * axes[2][None, None, :])
