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
import time
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


#: How often ``simulate`` calls its ``check`` hook, in steps.
CHECK_EVERY = 64


#: How often the step loop reports itself -- step, elapsed and the rate the
#: machine is actually giving -- in steps. A tier-2 run has a wall-clock cap,
#: and the run that hits it is exactly the run whose output matters: without
#: these lines a killed run leaves its plan and nothing else, so neither where
#: it stopped nor the rate that stopped it can be recovered afterwards. The
#: line is flushed as it is printed, so it survives the kill (a redirected
#: stdout is block-buffered, and a buffer dies with the process).
PROGRESS_EVERY = 1000

#: The field movie's caps (proposal: at most 64 samples per axis, at most 60
#: frames by default and 120 at most). A frame is then at most 1 MiB of float32.
MAX_FRAMES = 60
MAX_MOVIE_FRAMES = 120
MAX_FRAME_SAMPLES = 64


def frame_plan(shape, n_steps: int, *, max_frames: int = MAX_FRAMES,
               max_samples: int = MAX_FRAME_SAMPLES,
               first_step: int = 0, last_step: int = None) -> dict:
    """The steps the field movie records and the stride along each axis.

    The stride is the smallest whole number that brings each axis to at most
    ``max_samples`` samples, and it does not depend on the steps recorded. The
    frames are spread evenly over the steps ``first_step``..``last_step`` and
    the last one is ``last_step``, so a movie over the whole run (the default)
    ends where the run ends. ``count`` is the requested frame count unless the
    range holds fewer steps.

    A window is only a restriction of the recorded steps: the run still takes
    every step, and the frames are the same nodes.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise FDTDError(f"a field movie needs at least 1 step, got {n_steps}")
    if last_step is None:
        last_step = n_steps - 1
    first, last = int(first_step), int(last_step)
    if not 0 <= first <= last < n_steps:
        raise FDTDError(
            f"a field movie's first and last step must lie in the run's "
            f"{n_steps} steps, with the first at or before the last, got "
            f"{first_step!r} and {last_step!r}")
    stride = tuple(max(1, -(-int(n) // int(max_samples))) for n in shape)
    span = last - first + 1
    count = min(int(max_frames), span)
    if count < 1:
        raise FDTDError(f"a field movie needs at least 1 frame, got "
                        f"{max_frames!r}")
    steps = [first + ((k + 1) * span) // count - 1 for k in range(count)]
    return {"steps": steps, "stride": stride}


def frame_shape(shape, stride) -> tuple:
    """Samples per axis that ``frame_plan``'s stride keeps."""
    return tuple(len(range(0, int(n), int(k))) for n, k in zip(shape, stride))


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

    #: The permittivity a metal sample's curl increment is rescaled to.
    EPS_INF = 1.0

    def samples(self, component: str, shape, periodic=(False, False, False)):
        """The ``component`` samples this metal owns, as a full-shape mask.

        A sample takes the metal when either node of its segment is metal, and
        only samples the E update advances count: half positions 0..n-2 along
        the component's own axis; integer positions 1..n-2 between walls,
        1..n-1 on a periodic axis (node n-1 is node 0, copied back after the
        terms). The wall ranges on a periodic axis left row n-1 out of the
        metal: a vacuum slot through every periodic slab. ``bind`` builds its
        storage from this mask, and :func:`source_permittivity` reads it.
        """
        axis = COMPONENTS.index(component)
        samples = self.region.copy()
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        samples[tuple(low)] |= self.region[tuple(high)]
        live = tuple(slice(0, -1) if a == axis else
                     slice(1, None) if periodic[a] else slice(1, -1)
                     for a in range(3))
        updated = np.zeros(tuple(shape), dtype=bool)
        updated[live] = True
        samples &= updated
        return samples

    def applied_permittivity(self, component: str, cell, shape,
                             periodic=(False, False, False)):
        """``EPS_INF`` if the ``component`` sample at ``cell`` is this metal's
        (its update is rescaled to it), else ``None``: the term leaves that
        sample's permittivity to the grid."""
        if tuple(self.region.shape) != tuple(shape):
            raise FDTDError(
                f"{self.name}: the region is {self.region.shape} but the grid "
                f"is {tuple(shape)}; they must be the same shape")
        if self.samples(component, shape, periodic)[tuple(cell)]:
            return self.EPS_INF
        return None

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
        self.parts = {}
        for component in COMPONENTS:
            samples = self.samples(component, shape, periodic)
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
                # eps_r / eps_inf - 1 at metal samples: rescales the curl
                # increment so the sample is updated with eps_inf
                "rescale": xp.asarray((eps_host[box] / self.EPS_INF - 1.0)
                                      * keep, dtype=f32),
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


def _window(n: int, window: str | None):
    """The window's own samples for a record of ``n`` steps.

    The one place a window is chosen. :func:`spectrum` takes the magnitude of
    the result; the dipole measurements multiply two of these, and their ratio
    is only a ratio of two measurements of the same signal if both went through
    the same window.
    """
    if window == "hann":
        return np.hanning(n)
    if window in (None, "none"):
        return np.ones(n)
    raise FDTDError(f"unknown window {window!r}; use 'hann' or None")


def _windowed_rfft(values, dt: float, window: str | None, xp=None):
    """``(frequencies_hz, spectrum, window)`` from one windowed rfft."""
    xp = np if xp is None else xp
    values = xp.asarray(values)
    n = int(values.shape[0])
    if n < 2:
        raise FDTDError(f"a spectrum needs at least 2 samples, got {n}")
    scale = _window(n, window)
    values = values.astype(xp.float64) * xp.asarray(scale)
    frequencies = np.fft.rfftfreq(n, d=dt)
    return frequencies, xp.fft.rfft(values), scale


def spectrum(trace, dt: float, *, window: str | None = "hann", xp=None):
    """``(frequencies_hz, amplitude)`` from the rfft of a windowed trace.

    The window is applied because a recorded trace is a finite slice of a signal
    that is still ringing when it ends; without it, the discontinuity at the two
    ends lays a sinc skirt over every peak. It is built in numpy and moved to
    ``xp``, so both backends window identically.
    """
    if xp is None:
        xp = np
    frequencies, values, _ = _windowed_rfft(trace, dt, window, xp)
    return frequencies, xp.abs(values)


# --------------------------------------------------------------------------
# a dipole's radiated power, and its Purcell factor
# --------------------------------------------------------------------------

def source_cell_volume(spacings, cell, component) -> float:
    """The volume a point source's current fills: its own E node's cell.

    The source adds ``dt * S`` to one E component at one node, so the current it
    stands for fills the cell that node owns -- dual along the component's own
    axis, node-width along the other two, which are the weights ``simulate``'s
    energy sum builds with ``volumes({a})``. On a graded grid that is not the
    primal cell, and the power below is proportional to it.

    ``spacings`` is a run's own ``"spacings"``: three arrays of primal lengths.
    """
    if component not in COMPONENTS:
        raise FDTDError(
            f"a source cell volume is asked for a component among "
            f"{COMPONENTS}, got {component!r}")
    if len(spacings) != 3:
        raise FDTDError(f"spacings is one entry per axis, got {len(spacings)}")
    axis = COMPONENTS.index(component)
    volume = 1.0
    for a, spacing in enumerate(spacings):
        primal = np.asarray(spacing, dtype=np.float64)
        if primal.ndim != 1 or primal.size < 1:
            raise FDTDError(
                f"axis {a} takes its primal lengths, got shape {primal.shape}")
        thickness = (_dual_thickness(primal) if a == axis
                     else _node_thickness(primal))
        index = int(cell[a])
        if not 0 <= index < thickness.size:
            raise FDTDError(
                f"cell {tuple(cell)!r} is off the {primal.size + 1}-node grid "
                f"on axis {a}")
        volume *= float(thickness[index])
    return volume


def source_permittivity(eps_r, cell, component, *, ade_terms=(),
                        boundary="metal") -> float:
    """The relative permittivity the E update applies at a point source's sample.

    This is the ``eps_r`` :func:`delivered_power`, :func:`delivered_power_spectrum`,
    :func:`purcell_spectrum` and :func:`dipole_moment` take: the source adds
    ``dt * S`` to its sample, which is the current ``J = -eps0 eps S`` only
    with the ``eps`` that sample's own update divides its curl by.

    ``simulate`` does no averaging of its own: the ``component`` sample at
    array index ``cell`` (half a cell above that node along the component's
    axis) is advanced with ``1 / (eps0 eps_r[cell])``, the node value of the
    array it is given (any sub-cell averaging happened when that array was
    built, e.g. ``fdtd_grid.average_eps``). So in a dielectric, at an interface
    or not, this is ``float(eps_r[cell])`` exactly. A polarisation term can
    override it: a Lorentz-Drude metal rescales the curl increment at every
    sample it owns (either node of the segment is metal) to its ``eps_inf`` of
    1, so a source next to a metal node, along its own axis, is driven in
    ``eps = 1`` whatever the node holds. Each term with an
    ``applied_permittivity(component, cell, shape, periodic)`` method that
    claims the sample adds its rescale ``eps / eps_inf - 1``, as ``advance``
    does, and the result is ``eps / (1 + sum)``.

    ``ade_terms`` and ``boundary`` are the ones the run is given; ``simulate``
    calls this with them and returns the value as ``"source_eps"``. The
    float32 storage of the grid (a relative rounding under 6e-8) is not part
    of the value.
    """
    if component not in COMPONENTS:
        raise FDTDError(
            f"a source permittivity is asked for a component among "
            f"{COMPONENTS}, got {component!r}")
    if not hasattr(eps_r, "ndim"):
        eps_r = np.asarray(eps_r, dtype=np.float64)
    if eps_r.ndim != 3:
        raise FDTDError(f"eps_r must be 3-D (nx, ny, nz), got shape {eps_r.shape}")
    shape = tuple(int(s) for s in eps_r.shape)
    cell = tuple(int(c) for c in cell)
    if len(cell) != 3 or not all(0 <= c < n for c, n in zip(cell, shape)):
        raise FDTDError(f"the cell {cell} is outside the grid {shape}")
    eps = float(eps_r[cell])
    if not eps > 0:
        raise FDTDError(f"the relative permittivity is positive, got {eps!r}")
    periodic = tuple(side == "periodic" for side in _axis_boundaries(boundary))
    rescale = 0.0
    for term in ade_terms:
        applied = getattr(term, "applied_permittivity", None)
        if applied is None:
            continue
        eps_inf = applied(component, cell, shape, periodic)
        if eps_inf is not None:
            rescale += eps / float(eps_inf) - 1.0
    return eps / (1.0 + rescale)


def _span_weights(primal: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Trapezoid weights over the nodes ``lo..hi``: half a cell at each end.

    They sum to ``x[hi] - x[lo]`` exactly, whatever the grid's grading -- the
    node-axis counterpart of a face's other partition, which is
    :func:`_dual_thickness` on the component's own axis. Each sample here sits
    ON a node, so the quadrature that spans the box's own rectangle is the
    trapezoid rule.

    The cell widths around the nodes (:func:`_node_thickness`) are the weights
    the energy sum wants and they are the wrong ones here: each would reach half
    a cell beyond the end node, two faces would each poke out past the other's
    layer, and the six faces would no longer meet along their shared edges. The
    surface would not be closed, and a flux through an open surface is not the
    power the source radiated.
    """
    primal = np.asarray(primal, dtype=np.float64)
    weights = np.empty(hi - lo + 1)
    weights[0] = 0.5 * primal[lo]
    weights[-1] = 0.5 * primal[hi - 1]
    # interior nodes: half of the cell either side
    weights[1:-1] = 0.5 * (primal[lo:hi - 1] + primal[lo + 1:hi])
    return weights


def flux_faces(box, shape, primal, pml=0):
    """The six faces of a closed box, as ``(component, slab, areas)`` terms.

    ``box`` is ``(centre, half_extent)`` in cells: ``centre`` and ``half_extent``
    are three integers each, the box's faces come out as the NODE layers
    ``centre +/- half_extent`` on the three axes, and their corners meet exactly
    -- the surface is closed.

    On a face normal to axis ``a`` at node layer ``i``, the box's own samples are
    the two E components tangential to that face, and in this layout those are
    exactly the two that lie ON it (``Ey[i, j, k]`` is dual in y, node in z;
    ``Ez[i, j, k]`` is node in y, dual in z). So a sample is a field value times
    the area of the face cell it owns, with nothing interpolated, and each of the
    two components is a quadrature of the face's own rectangle:

    * along the component's own axis the samples sit on the dual nodes, and the
      weight is the primal cell each one sits in (:func:`_dual_thickness`);
    * along the other tangential axis they sit on the nodes, and the weight is
      the trapezoid's, half a cell at each end (:func:`_span_weights`).

    Both partitions therefore span exactly the rectangle between the box's two
    node layers, one with ``(jh - jl)`` cells and the other with
    ``(kh - kl + 1)``, so every face is the box's own and no more. That is what
    closes the surface, and closure is what makes the flux the source's radiated
    power rather than the answer for whatever surface the weights happened to
    describe.

    The layers must be strictly inside the live region, 1..n-2: the outermost
    node layer is never written by the update (a wall copies the boundary value
    onto it), so a face there would measure a field that is not the field. That
    also keeps every sample's own axis in its own live range.

    ``pml`` is the damping layer's depth, the count of node layers the absorbing
    boundary occupies at each end of an axis: a scalar, or one count per axis
    (zero where that axis has no absorber, as on a periodic one). A face inside
    it is refused, because the field there is the damped one -- on a fixture whose
    whole floor sat two nodes above a six-node absorber the surface measured a
    dipole 0.52 times its own radiated power, and a mirror plane buried in the
    layer moved it by 0.2 per cent. Nothing raises on its own: the sum is finite
    and plausible. The first CPML-free E node is layer ``pml`` (the outermost
    ``pml`` primal cells carry the profile, and its ramp reaches zero at the
    interface), so these faces are the deepest a closed surface can be drawn.

    ``areas`` is the outer product of the two face-cell widths, pre-broadcast to
    one leading axis so a step's sum is ``sum(values**2 * areas)``.
    """
    try:
        centre, half = box
        centre = tuple(int(c) for c in centre)
        half = tuple(int(h) for h in half)
    except (TypeError, ValueError):
        raise FDTDError(
            f"flux_box is (centre_cell, half_extent_cells), three integers each,"
            f" got {box!r}") from None
    if len(centre) != 3 or len(half) != 3:
        raise FDTDError(
            f"flux_box needs three cell indices and three half extents, got "
            f"{centre!r} and {half!r}")
    if any(h < 1 for h in half):
        raise FDTDError(
            f"a flux box is at least one cell each way, got {half!r}")
    try:
        depth = [int(p) for p in pml]
    except TypeError:
        depth = [int(pml)] * 3
    if len(depth) != 3 or min(depth) < 0:
        raise FDTDError(
            f"pml is the absorbing layer's depth, a non-negative scalar "
            f"or one count per axis, got {pml!r}")
    lo = [centre[a] - half[a] for a in range(3)]
    hi = [centre[a] + half[a] for a in range(3)]
    for a in range(3):
        n = int(shape[a])
        if lo[a] < 1 or hi[a] > n - 2:
            raise FDTDError(
                f"the flux box's faces are node layers {lo[a]}..{hi[a]} on axis "
                f"{a}, and a face must be inside the live layers 1..{n - 2} "
                f"there (the outermost layer is never updated); the grid has "
                f"{n} nodes on that axis")
        if depth[a] and (lo[a] < depth[a] or hi[a] > n - 1 - depth[a]):
            raise FDTDError(
                f"the flux box's faces are node layers {lo[a]}..{hi[a]} on axis "
                f"{a} of {n}, and {depth[a]} of those nodes are the absorbing "
                f"layer: a surface inside it measures the damped field, not the "
                f"wave that left, so its faces must lie in "
                f"{depth[a]}..{n - 1 - depth[a]}. Widen the grid, or shrink the "
                f"box, by {max(depth[a] - lo[a], hi[a] - (n - 1 - depth[a]))} "
                f"cells on that axis")
    dual = [_dual_thickness(np.asarray(p, dtype=np.float64)) for p in primal]
    span = [_span_weights(p, lo[a], hi[a]) for a, p in enumerate(primal)]
    faces = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for i in (hi[a], lo[a]):
            for component in (COMPONENTS[b], COMPONENTS[c]):
                along = COMPONENTS.index(component)
                other = c if along == b else b
                slab = [slice(lo[axis], hi[axis] + 1) for axis in range(3)]
                slab[a] = slice(i, i + 1)
                slab[along] = slice(lo[along], hi[along])
                w_along = dual[along][lo[along]:hi[along]]
                w_other = span[other]
                # the outer product, laid out the way the slab is: a singleton
                # on the face's own axis and the two widths on the two axes
                # they belong to (which are not always their own order)
                areas = np.empty((1, w_along.size, w_other.size))
                areas[0] = w_along[:, None] * w_other[None, :]
                axes = [0, 0, 0]
                axes[a], axes[along], axes[other] = 0, 1, 2
                faces.append((component, tuple(slab), areas.transpose(axes)))
    return faces


def _flux_step(field, faces, xp):
    """``sum |E|^2 * area`` over every face: the box's instantaneous flux.

    Stays on the device, like the probe traces: the box is one scalar a step, so
    a caller gets a whole record from a single transfer at the end and the loop
    takes no sync.

    Cast to float64 before squaring, exactly as the energy sum is: a sample of
    ~1e-16 squared times a face cell of ~1e-14 m^2 is ~1e-46, under float32's
    smallest normal (1.2e-38), and the box then reports a flux of zero for a
    field that is radiating.
    """
    total = None
    for component, slab, areas in faces:
        values = field[component][slab].astype(xp.float64)
        term = xp.sum(values * values * areas)
        total = term if total is None else total + term
    return total


# --------------------------------------------------------------------------
# a phasor box: the closed surface the far-field transform reads
# --------------------------------------------------------------------------

def _is_half(family: str, component: str, axis: int) -> bool:
    """Whether a Yee component sits half a cell off the nodes along ``axis``.

    ``E_c`` is half a cell off along its own axis ``c`` and on the nodes along
    the other two; ``H_c`` the other way round (see ``update_h``/``update_e``).
    """
    own = COMPONENTS.index(component) == axis
    return own if family == "E" else not own


def phasor_box_layers(box, shape, pml=0):
    """The node layers ``(lo, hi)`` of a phasor box, three each, checked.

    ``box`` is ``(centre, half_extent)`` in cells, exactly as for
    :func:`flux_faces`: the six faces are the NODE layers ``centre +/-
    half_extent``. The rules are the flux box's, plus the half cell the H
    interpolation reaches:

    * every face must lie strictly inside the live layers ``1..n-2`` (the
      outermost node layer is never updated);
    * on an axis with an absorber ``pml`` node layers deep, a face reads the
      tangential H half a cell either side of itself, and the H half a cell
      outside layer ``pml`` is still in the layer's last, damped cell -- so the
      faces must lie in ``pml + 1 .. n - 2 - pml``, one cell further in than a
      flux box may go.

    ``pml`` is a scalar or one count per axis (zero where an axis has none).
    """
    try:
        centre, half = box
        centre = tuple(int(c) for c in centre)
        half = tuple(int(h) for h in half)
    except (TypeError, ValueError):
        raise FDTDError(
            f"phasor_box is (centre_cell, half_extent_cells), three integers "
            f"each, got {box!r}") from None
    if len(centre) != 3 or len(half) != 3:
        raise FDTDError(
            f"phasor_box needs three cell indices and three half extents, got "
            f"{centre!r} and {half!r}")
    if any(h < 1 for h in half):
        raise FDTDError(
            f"a phasor box is at least one cell each way, got {half!r}")
    try:
        depth = [int(p) for p in pml]
    except TypeError:
        depth = [int(pml)] * 3
    if len(depth) != 3 or min(depth) < 0:
        raise FDTDError(
            f"pml is the absorbing layer's depth, a non-negative scalar "
            f"or one count per axis, got {pml!r}")
    lo = [centre[a] - half[a] for a in range(3)]
    hi = [centre[a] + half[a] for a in range(3)]
    for a in range(3):
        n = int(shape[a])
        if lo[a] < 1 or hi[a] > n - 2:
            raise FDTDError(
                f"the phasor box's faces are node layers {lo[a]}..{hi[a]} on "
                f"axis {a}, and a face must be inside the live layers "
                f"1..{n - 2} there (the outermost layer is never updated); the "
                f"grid has {n} nodes on that axis")
        if depth[a] and (lo[a] < depth[a] + 1 or hi[a] > n - 2 - depth[a]):
            raise FDTDError(
                f"the phasor box's faces are node layers {lo[a]}..{hi[a]} on "
                f"axis {a} of {n}, and {depth[a]} of those nodes are the "
                f"absorbing layer: each face also reads H half a cell outside "
                f"itself, and a surface touching the absorber measures the "
                f"damped field, so its faces must lie in "
                f"{depth[a] + 1}..{n - 2 - depth[a]}. Widen the grid, or "
                f"shrink the box, by "
                f"{max(depth[a] + 1 - lo[a], hi[a] - (n - 2 - depth[a]))} "
                f"cells on that axis")
    return lo, hi


class _PhasorBox:
    """A running DFT of E and H on the six faces of a box, on the device.

    Each Yee component is read from the smallest slab of its own samples that
    the interpolation to the face-cell centres needs (two layers across the
    face where it sits half a cell off it, one where it sits on it), and folded
    into one ``complex128`` accumulator per (face, family, component) holding
    every frequency at once. The per-step phase factors are a precomputed
    device table, so a step reads no host value and takes no sync; the only
    transfer is :meth:`faces`, once, after the run.
    """

    def __init__(self, box, frequencies, *, shape, primal, pml, dt: float,
                 n_steps: int, start, window, xp):
        self.lo, self.hi = phasor_box_layers(box, shape, pml)
        try:
            freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        except (TypeError, ValueError):
            raise FDTDError(
                f"phasor_frequencies is one or more frequencies in Hz, got "
                f"{frequencies!r}") from None
        if freqs.ndim != 1 or freqs.size < 1:
            raise FDTDError(
                f"phasor_frequencies is one or more frequencies in Hz, got "
                f"{frequencies!r}")
        nyquist = 0.5 / dt
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0) \
                or np.any(freqs >= nyquist):
            raise FDTDError(
                f"every phasor frequency must be positive and below this run's "
                f"Nyquist frequency 1/(2 dt) = {nyquist:.6g} Hz, got "
                f"{freqs.tolist()!r}")
        try:
            start = int(start)
        except (TypeError, ValueError):
            raise FDTDError(f"phasor_start is a step index, got {start!r}") from None
        if not 0 <= start < n_steps:
            raise FDTDError(
                f"phasor_start is the first step the phasors read, in "
                f"0..{n_steps - 1} for this {n_steps}-step run, got {start}")
        count = n_steps - start
        weights = _window(count, window)
        total = float(np.sum(weights))
        if not total > 0:
            raise FDTDError(
                f"the {window!r} window sums to {total!r} over the {count} "
                f"step(s) from phasor_start {start}; read more steps")
        self.frequencies = freqs
        self.start = start
        self.primal = [np.asarray(p, dtype=np.float64) for p in primal]
        self.xp = xp
        # f(t) = Re[F exp(-i w t)] (the convention farfield.py uses) gives
        # F = 2 sum(w_n f_n exp(+i w t_n)) / sum(w_n) for a steady tone, the
        # amplitude _windowed_at reads. The sample times differ by family:
        # at the end of step s, E has just advanced to (s + 1) dt and H was
        # advanced half a step earlier, to (s + 1/2) dt.
        steps = np.arange(start, n_steps, dtype=np.float64)[:, None]
        omega = 2.0 * np.pi * freqs[None, :]
        scale = (2.0 / total) * weights[:, None]
        phase_sign = 1j
        self._phase = {
            "E": xp.asarray(scale * np.exp(phase_sign * omega * (steps + 1.0) * dt),
                            dtype=np.complex128),
            "H": xp.asarray(scale * np.exp(phase_sign * omega * (steps + 0.5) * dt),
                            dtype=np.complex128),
        }
        self._terms = []
        for a in range(3):
            for sign, layer in (("+", self.hi[a]), ("-", self.lo[a])):
                for family in ("E", "H"):
                    for component in COMPONENTS:
                        slab = []
                        for axis in range(3):
                            half = _is_half(family, component, axis)
                            if axis == a:
                                slab.append(slice(layer - 1, layer + 1) if half
                                            else slice(layer, layer + 1))
                            else:
                                slab.append(slice(self.lo[axis], self.hi[axis]
                                                  + (0 if half else 1)))
                        size = tuple(s.stop - s.start for s in slab)
                        acc = xp.zeros((freqs.size,) + size, dtype=np.complex128)
                        self._terms.append((sign + "xyz"[a], a, layer, family,
                                            component, tuple(slab), acc))

    def accumulate(self, step: int, e_field: dict, h_field: dict) -> None:
        """Fold one step's E (at (s+1) dt) and H (at (s+1/2) dt) in."""
        k = step - self.start
        if k < 0:
            return
        phase = {family: table[k][:, None, None, None]
                 for family, table in self._phase.items()}
        for _normal, _a, _layer, family, component, slab, acc in self._terms:
            source = e_field if family == "E" else h_field
            values = source[component][slab].astype(self.xp.float64)
            acc += values[None] * phase[family]

    def _at_centres(self, values, family, component, a, layer):
        """One component's slab, interpolated to the face-cell centres.

        ``values`` is the host ``(n_freq, s0, s1, s2)`` accumulator. Along the
        face's normal axis a sample half a cell off the face is interpolated
        linearly between its two layers, each weighted by the OTHER primal
        width (exact for a linear field on a graded grid); a sample on the face
        is taken as it is. Along each in-plane axis a node sample is the mean
        of the cell's two end nodes and a half-cell sample already sits at the
        centre. Returns ``(n_freq, n_u, n_v)``.
        """
        out = values
        for axis in range(3):
            half = _is_half(family, component, axis)
            ax = axis + 1
            if axis == a:
                if half:
                    below = float(self.primal[a][layer - 1])
                    above = float(self.primal[a][layer])
                    out = ((above * np.take(out, [0], axis=ax)
                            + below * np.take(out, [1], axis=ax))
                           / (below + above))
            elif not half:
                n = out.shape[ax]
                out = 0.5 * (np.take(out, range(n - 1), axis=ax)
                             + np.take(out, range(1, n), axis=ax))
        return np.squeeze(out, axis=a + 1)

    def faces(self) -> list:
        """One list of six ``farfield.Face`` per frequency, in DIRECTIONS order."""
        import farfield   # farfield imports fdtd; only a run that asks needs it
        nodes = [np.concatenate(([0.0], np.cumsum(p))) for p in self.primal]
        built: dict = {}
        for normal, a, layer, family, component, _slab, acc in self._terms:
            host = np.asarray(_host_result(acc, self.xp))
            entry = built.setdefault(normal, {"a": a, "layer": layer,
                                              "E": [None] * 3, "H": [None] * 3})
            entry[family][COMPONENTS.index(component)] = self._at_centres(
                host, family, component, a, layer)
        out = []
        for f in range(self.frequencies.size):
            faces = []
            for normal in DIRECTIONS:
                entry = built[normal]
                a, layer = entry["a"], entry["layer"]
                u, v = (axis for axis in range(3) if axis != a)
                faces.append(farfield.Face(
                    normal, float(nodes[a][layer]),
                    nodes[u][self.lo[u]:self.hi[u] + 1],
                    nodes[v][self.lo[v]:self.hi[v] + 1],
                    np.stack([c[f] for c in entry["E"]]),
                    np.stack([c[f] for c in entry["H"]])))
            out.append(faces)
        return out


def _windowed_at(values, dt: float, frequency: float, scale):
    """The AMPLITUDE at one frequency: ``2 sum(w v e^{-i w t}) / sum(w)``.

    The same sum :func:`_windowed_rfft` bins (its ``rfft[k]`` at
    ``f = k / (n dt)``), normalised into the number a measurement wants: for a
    tone ``a(t) = A cos(w t + phi)`` that sum is ``(A/2) e^{i phi} sum(w)``, so
    twice it over the window's own sum is the amplitude ``A`` -- a drive whose
    peak is 1 pays ``|A| = 1``. (A rectangular window has ``sum(w) = n``, and
    this is the familiar ``2 |X_k| / n``.)

    The frequency is free rather than binned, so a measurement does not have to
    tie its record length to its drive (``f = k / (n dt)``) and waste the steps
    between bins; an unsampled skirt cannot leak into the number either.
    """
    values = np.asarray(values, dtype=np.float64)
    weight = float(np.sum(scale))
    if not weight > 0:
        raise FDTDError(f"the window sums to {weight!r} over these steps")
    phase = np.exp(-2j * np.pi * frequency * dt * np.arange(values.size))
    return 2.0 * np.sum(scale * values * phase) / weight


def _steady_tail(values, dt: float, frequency: float, periods: int, *,
                 what: str):
    """The LAST ``periods`` drive periods of a record, as a slice.

    Every CW measurement here means the driven steady state, and a CW record
    carries its own turn-on: :func:`cw_profile` ramps over 5 periods, so a window
    over a whole record is a window over the ramp and the settling as well. The
    record's own length must not be part of the answer -- and it is, loudly: one
    dipole in vacuum (16 cells per wavelength, 25 nodes, absorber at 6) read as
    F = 1.33361 over the last 8 periods of a 296-step record against 2.10286 over
    the whole of it (296 steps is 10.7 periods there, so a window that long still
    reaches back into the 5-period ramp), and on a 1343-step record the same two
    windows read 1.00937 and 1.01854. A ratio of two numbers taken over the same
    window is what a Purcell factor is, so the window has to be one the caller
    chose and both readings share.
    """
    values = np.asarray(values)
    if int(periods) < 1:
        raise FDTDError(f"a mean is over at least one period, got {periods!r}")
    steps = int(round(int(periods) / (float(frequency) * dt)))
    if steps < 1:
        raise FDTDError(
            f"one drive period is {1.0 / float(frequency)!r} s against a step of "
            f"{dt!r} s: dt is too coarse to sample the drive at all")
    if steps > values.size:
        raise FDTDError(
            f"{int(periods)} periods is {steps} steps and the {what} is "
            f"{values.size} steps: record longer, or ask for fewer periods")
    return values[values.size - steps:]


def dipole_moment(drive, dt: float, *, volume: float, eps_r: float,
                  frequency: float, periods: int = 8,
                  window: str | None = "hann") -> float:
    """``|p~(w)|`` in C.m: the moment the source's own current stands for.

    The source adds ``dt * S(t)`` to one E component at one node, so the current
    it stands for is ``J = -eps0 eps_r S`` filling that node's cell ``volume``,
    and ``p~ = J~ / (-i w)``. With the amplitude :func:`_windowed_at` returns
    (``a(t) = Re[A e^{-i w t}]`` gives ``A = 2 S~ / W``, ``W = sum(window)``
    and ``S~`` the windowed sum),

        |p~(w)| = eps0 eps_r V * 2 |S~(w)| / (w W).

    ``drive`` is ``simulate(..., record_drive=True)``'s record. Derived, never
    fitted: the same run's far field carries this moment's Larmor power, which is
    what makes F a statement about the structure and not about the mesh.

    The amplitude is read over the last ``periods`` drive periods, as in
    :func:`delivered_power` and :func:`radiated_power` -- this is the numerator's
    own denominator, and a window over the drive's 5-period ramp would bias it
    against a power measured after the ramp. Pass the same ``periods`` you pass
    there and the ratio is two readings of one steady state.

    ``eps_r`` is the permittivity the update applies at the source's own
    sample: the run's ``"source_eps"`` (:func:`source_permittivity`). A scalar
    from the layout's node is the same number in a dielectric, at an interface
    or not, and wrong on a Lorentz-Drude metal's sample, which is updated with
    ``eps_inf`` = 1.
    """
    if not volume > 0:
        raise FDTDError(f"a cell volume is positive, got {volume!r}")
    if not eps_r > 0:
        raise FDTDError(f"the relative permittivity is positive, got {eps_r!r}")
    omega = 2.0 * np.pi * float(frequency)
    if not omega > 0:
        raise FDTDError(f"a dipole moment is asked for at a positive frequency, "
                        f"got {frequency!r} Hz")
    drive = np.asarray(drive)
    if drive.ndim != 1 or drive.size < 2:
        raise FDTDError(
            f"a drive record is one trace of at least 2 steps, got shape "
            f"{drive.shape}")
    tail = _steady_tail(drive, dt, frequency, periods, what="drive record")
    return (EPS0 * float(eps_r) * float(volume)
            * abs(_windowed_at(tail, dt, frequency, _window(tail.size, window)))
            / omega)


def larmor_power(moment: float, frequency: float, *, index: float) -> float:
    """(watts) ``P0 = mu0 index w^4 |p~|^2 / (12 pi c)`` in a medium.

    The power the same moment radiates in a homogeneous medium of index
    ``index`` -- the denominator of the Purcell factor, at the index the dipole
    itself sits in, which is what makes F a statement about the structure and
    not about the mesh.
    """
    if not moment > 0:
        raise FDTDError(f"a dipole moment is positive, got {moment!r}")
    if not index > 0:
        raise FDTDError(f"a medium's index is positive, got {index!r}")
    omega = 2.0 * np.pi * float(frequency)
    if not omega > 0:
        raise FDTDError(f"a Larmor power is asked for at a positive frequency, "
                        f"got {frequency!r} Hz")
    return (MU0 * float(index) * omega ** 4 * float(moment) ** 2
            / (12.0 * np.pi * C0))


def delivered_power(drive, trace, dt: float, *, volume: float, eps_r: float,
                    frequency: float, periods: int = 8,
                    window: str | None = "hann") -> float:
    """(watts) The time-averaged power a soft point source hands the grid.

    ``trace`` is a probe at the source's own cell and component and ``drive``
    the same run's record of the profile value ``S(t)`` (``record_drive=True``).
    This is the power the dipole DELIVERS -- radiated plus absorbed -- which is
    the numerator a Purcell factor wants in a structure that absorbs, where a
    surface only ever sees what leaves the box (see :func:`radiated_power`).

    ``eps_r`` is the permittivity the update applies at the source's own
    sample: the run's ``"source_eps"`` (:func:`source_permittivity`). A scalar
    from the layout's node is the same number in a dielectric, at an interface
    or not, and wrong on a Lorentz-Drude metal's sample, which is updated with
    ``eps_inf`` = 1.

    The source adds ``dt S(t)`` to one E component each step, so the energy it
    gives that element during step ``n`` is that element's own change of energy,

        W_n = eps0 eps_r V dt S_n (post_n + post_{n-1}) / 2,

    the two recorded field values being the ones on either side of the
    injection. This is the scheme's identity, not a model. On the closed-box
    fixture's own run -- 17 nodes, 16 cells per wavelength, a 30-period CW
    drive, no CPML and no dispersive term, so the source is the only thing that
    can move the energy -- the sum of the work above is 0.9999997 of the energy
    the grid gained over the same steps, while the one-sided spelling
    ``post - dt S / 2`` (the same midpoint, one step out) is -40.2180 of it: not
    merely wrong in size, but on the other side of zero. The leftover is a
    source-side constant at omega^0, so against a radiated power that goes as
    omega^2 it grows as (wavelength/dx)^2 without bound -- 4.7, 11.7 and 26.3 at
    10, 16 and 24 cells per wavelength, the measurement that made this module's
    earlier version call the delivered power the wrong instrument.

    Both series are tones once the drive has settled, so the mean of their
    product over whole periods is ``eps0 eps_r V Re[E~ S~*] / 2`` with the
    amplitudes :func:`_windowed_at` returns. The mean is over the LAST
    ``periods`` drive periods, as in :func:`radiated_power`, and that is what
    makes it the steady state: a CW record carries its own turn-on, and a Purcell
    factor is a statement about the driven steady state. Measured on a
    homogeneous open box (a dipole in vacuum, 16 cells per wavelength, 45
    nodes), the last 8 periods read 1.006 of the Larmor power at pml 6 and
    1.007 at pml 10, and the same trunk of the run read as a whole record
    instead -- a Hann over every step since the drive was switched on -- reads
    1.48; stretching that run until the transient is one part in thirty of it
    pulls the whole-record number down to 1.03 without moving the tail's own. A
    high-Q cavity builds up for tens of periods (a mode of Q = 225 takes ~36),
    so the record has to be longer than the build-up; the mean does not create
    the steady state, it declines to average anything else. Hann by default, so
    the seam at the window's own ends does not leak into a number read off a
    single frequency.

    The delivered power is the one measurement that needs no surface: it is the
    same number wherever the field is later lost (PEC walls, absorber, a metal's
    absorption), so it has none of a flux box's exposure to where the surface
    can safely be put.
    """
    drive = np.asarray(drive, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    if drive.ndim != 1 or trace.ndim != 1 or drive.size != trace.size:
        raise FDTDError(
            f"a drive and the trace of its own cell are one record each, got "
            f"shapes {drive.shape} and {trace.shape}")
    if drive.size < 2:
        raise FDTDError(f"a record is at least 2 steps, got {drive.size}")
    if not volume > 0:
        raise FDTDError(f"a cell volume is positive, got {volume!r}")
    if not eps_r > 0:
        raise FDTDError(f"the relative permittivity is positive, got {eps_r!r}")
    if not frequency > 0:
        raise FDTDError(f"a delivered power is asked for at a positive frequency, "
                        f"got {frequency!r} Hz")
    if int(periods) < 1:
        raise FDTDError(f"a mean is over at least one period, got {periods!r}")
    steps = int(round(int(periods) / (float(frequency) * dt)))
    if steps < 1:
        raise FDTDError(
            f"one drive period is {1.0 / float(frequency)!r} s against a step of "
            f"{dt!r} s: dt is too coarse to sample the drive at all")
    if steps > drive.size:
        raise FDTDError(
            f"{int(periods)} periods is {steps} steps and the record is "
            f"{drive.size} steps: record longer, or ask for fewer periods")
    # the two field samples either side of each injection, on the full record so
    # that the tail's own first sample still gets its real predecessor; the
    # record's first step has none, and stands in for it rather than an invented
    # zero.
    mid = 0.5 * (trace + np.concatenate((trace[:1], trace[:-1])))
    drive_tail = drive[drive.size - steps:]
    mid_tail = mid[mid.size - steps:]
    scale = _window(steps, window)
    source = _windowed_at(drive_tail, dt, frequency, scale)
    field = _windowed_at(mid_tail, dt, frequency, scale)
    return float(0.5 * EPS0 * float(eps_r) * float(volume)
                 * np.real(field * np.conj(source)))


def delivered_power_spectrum(drive, trace, dt: float, *, volume: float,
                             eps_r: float, window: str | None = None,
                             band=None):
    """``(frequencies_hz, watts)`` for one broadband run's whole band.

    The same ``eps0 eps_r V Re[E~ S~*] / 2`` as :func:`delivered_power`, bin by
    bin, from ONE pulse instead of one CW run per frequency: a three-point
    frequency ladder costs one run and one record, not three of each, and the
    curve between the points is measured rather than interpolated. Both
    transforms carry the same normalisation as the CW arm's ``_windowed_at`` --
    twice the windowed sum over the window's own sum -- so a bin the drive's own
    band sits on reads what the CW measurement reads at that frequency. A fixture
    pins that (measured: a vacuum pulse at 24 cells per wavelength reads F =
    1.00443 at its own bin against the CW arm's 1.00525 in the same box, one part
    in 800 apart), because a factor of two or a dropped window sum would survive
    every qualitative check. Over a sweep in frequency the same arm's delivered
    power follows the sampled bins' omega^2 -- measured exponent 1.966 across
    1.667x in omega, bins 0.015751, 0.012601 and 0.009450 GHz carrying 2.569484e-24,
    1.716588e-24 and 9.412558e-25 W -- since a fixed-current dipole's reaction on
    itself has one factor of omega from the moment and one from the field it
    drives.

    ``trace`` is a probe at the source's own cell and component and ``drive``
    the run's ``record_drive``. The field is read at the two samples either side
    of each injection, the CW arm's own half-step, so the two arms share a phase
    convention: at 24 cells per wavelength the half-step lag is 4.29 degrees
    (``½ ω dt``, and 6.43 at 16 cells and 3.43 at 30), which a band-wide glance
    would not notice and a single-frequency comparison would.

    The transform is over the WHOLE record and the default window is the boxcar,
    because a pulse is over at ``t = 0``: the record IS the signal, and the
    drive's own spectrum is the band. Windowing is then a choice with a cost, and
    the cost is measured. This estimator is a cross-spectral density, so it reads
    the drive's own response only where the window weights the drive and the
    field alike; a window that does not is not a small error, it is a sign error.
    Measured in a vacuum box at 24 cells per wavelength, F over 0.7-1.3 f0: a
    pulse centred in a 10-sigma record (which is what the drive's own
    ``start=5`` sigma delay does to it) reads 1.00143 to 1.00657 with the boxcar
    and 0.98530 to 1.16101 under a Hann; the SAME pulse in a 20-sigma record, its
    peak at 26 per cent of the window, reads -65.30 to -8.09 under a Hann, and at
    18 per cent of a 30-sigma record, -94.34 to -8.21. A longer record is therefore
    worse, not better, and the drive's delay is fixed: the record is what a
    caller moves to put the pulse in the window's middle. (:func:`cw_profile`'s
    steady state is the opposite case -- its record never decays, so a tail
    window is mandatory there and the record's own length is made moot by it.)

    It is a PULSE arm and not a substitute for the CW one on a resonator: a
    cavity's ring-down outlives any record the drive's fixed delay can be
    centred in, and the ratio above then divides a windowed drive by a field
    that is mostly ring-down. A high-Q structure wants :func:`delivered_power`
    once per frequency.

    ``band=(f_low, f_high)`` in Hz keeps the bins between them, both ends
    included: outside the drive's own band the ratio of two small transforms is
    noise, and at ``f = 0`` there is no dipole to radiate at all.
    """
    drive = np.asarray(drive, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    if drive.ndim != 1 or trace.ndim != 1 or drive.size != trace.size:
        raise FDTDError(
            f"a drive and the trace of its own cell are one record each, got "
            f"shapes {drive.shape} and {trace.shape}")
    if drive.size < 2:
        raise FDTDError(f"a record is at least 2 steps, got {drive.size}")
    if not volume > 0:
        raise FDTDError(f"a cell volume is positive, got {volume!r}")
    if not eps_r > 0:
        raise FDTDError(f"the relative permittivity is positive, got {eps_r!r}")
    mid = 0.5 * (trace + np.concatenate((trace[:1], trace[:-1])))
    frequencies, drive_hat, scale = _windowed_rfft(drive, dt, window)
    _, mid_hat, _ = _windowed_rfft(mid, dt, window)
    amplitude = 2.0 / float(np.sum(scale))
    power = (0.5 * EPS0 * float(eps_r) * float(volume) * amplitude ** 2
             * np.real(mid_hat * np.conj(drive_hat)))
    return _band(frequencies, (power,), band, what="delivered-power")


def _band(frequencies, values, band, *, what: str):
    """The bands' own bins of ``frequencies`` and of every array in ``values``.

    Returns everything unchanged when ``band`` is None, and raises rather than
    returning empty arrays when the band holds no bin: an empty curve looks like
    a measurement that found nothing, which is not what happened.
    """
    if band is None:
        return (frequencies, *values)
    low, high = (float(edge) for edge in band)
    keep = (frequencies >= low) & (frequencies <= high)
    if not keep.any():
        raise FDTDError(
            f"no bin of this {what} record lies between {low!r} and {high!r} "
            f"Hz; its own bins run from {frequencies[0]!r} to "
            f"{frequencies[-1]!r} Hz")
    return (frequencies[keep], *(at[keep] for at in values))


def purcell_spectrum(drive, trace, dt: float, *, volume: float, eps_r: float,
                     index: float, window: str | None = None, band=None):
    """``(frequencies_hz, power, reference, F)`` across one broadband run's band.

    :func:`delivered_power_spectrum` over that same run's own
    :func:`larmor_power` band -- the moment follows from the drive alone, so the
    denominator costs no second run and the ratio is two readings of one
    excitation at every frequency at once. ``index`` is the dipole's host index,
    as in :func:`purcell_factor`, which this is the band-wide spelling of; the
    reference is built by calling :func:`larmor_power` bin by bin, so the
    denominator has one spelling and this function cannot drift from it.
    ``eps_r`` is the run's ``"source_eps"``, as in :func:`delivered_power`.

    The DC bin has no moment (``p~ = J~ / (-i w)`` at ``w = 0`` is not a dipole),
    so the moment, the reference and F are all 0 there rather than divided by:
    a free dipole's own reaction vanishes at zero frequency and F would
    otherwise be 0/0. The delivered power is left as the record's own DC bin
    measured it, since that is a reading and not a definition; any band a caller
    names of interest is above DC, and the drive's own band never reaches it.
    ``band`` trims all four arrays together, as in
    :func:`delivered_power_spectrum`.
    """
    frequencies, power = delivered_power_spectrum(
        drive, trace, dt, volume=volume, eps_r=eps_r, window=window)
    drive = np.asarray(drive, dtype=np.float64)
    if drive.ndim != 1 or drive.size < 2:
        raise FDTDError(
            f"a drive record is one trace of at least 2 steps, got shape "
            f"{drive.shape}")
    _, drive_hat, scale = _windowed_rfft(drive, dt, window)
    amplitude = 2.0 * np.abs(drive_hat) / float(np.sum(scale))
    omega = 2.0 * np.pi * frequencies
    moment = np.zeros_like(omega)
    live = omega > 0
    moment[live] = (EPS0 * float(eps_r) * float(volume) * amplitude[live]
                    / omega[live])
    reference = np.array([
        0.0 if m <= 0 else larmor_power(float(m), float(f), index=index)
        for m, f in zip(moment, frequencies)])
    f_factor = np.zeros_like(reference)
    live = reference > 0.0
    f_factor[live] = power[live] / reference[live]
    return _band(frequencies, (power, reference, f_factor), band,
                 what="Purcell-spectrum")


def radiated_power(flux, dt: float, *, index: float, frequency: float,
                   periods: int = 8) -> float:
    """(watts) The power the flux box's surface carried out, over whole periods.

    ``flux`` is ``simulate(..., flux_box=...)``'s record: every step, ``sum
    |E|^2 * area`` over the box's six faces. In the wave zone the field on the
    surface is locally an outgoing plane wave, so the Poynting flux through the
    surface is ``|E|^2 / eta`` with ``eta = eta0 / index`` the medium's impedance
    there. (This is the statement the far field was already checked against: its
    amplitude carried the source moment's Larmor power to 1%.)

    The mean is over the LAST ``periods`` drive periods, so the record must end
    in the steady state. An instantaneous sample swings at 2w about the mean, so
    a fractional period leaves part of that swing in the answer; the tail is
    rounded to whole steps and the residual falls off as ``1/periods``.

    It is the RADIATED power: what crosses the surface. A lossy structure inside
    the box absorbs part of what the dipole delivers, and a Purcell factor for
    one wants that part added back in.
    """
    flux = np.asarray(flux)
    if flux.ndim != 1 or flux.size < 2:
        raise FDTDError(
            f"a flux record is one trace of at least 2 steps, got shape "
            f"{flux.shape}")
    if not index > 0:
        raise FDTDError(f"the surface's index is positive, got {index!r}")
    if not frequency > 0:
        raise FDTDError(f"the drive frequency is positive, got {frequency!r}")
    if int(periods) < 1:
        raise FDTDError(f"a mean is over at least one period, got {periods!r}")
    steps = int(round(int(periods) / (float(frequency) * dt)))
    if steps < 1:
        raise FDTDError(
            f"one drive period is {1.0 / float(frequency)!r} s against a step of "
            f"{dt!r} s: dt is too coarse to sample the drive at all")
    if steps > flux.size:
        raise FDTDError(
            f"{int(periods)} periods is {steps} steps and the record is "
            f"{flux.size} steps: record longer, or ask for fewer periods")
    return float(np.mean(flux[flux.size - steps:].astype(np.float64))) \
        * float(index) / ETA0


def purcell_factor(power: float, larmor: float) -> float:
    """```F = P / P0``: the Purcell factor of a dipole in a structure.

    ``power`` is what the box carried away (:func:`radiated_power`) and
    ``larmor`` is the same dipole's power in a homogeneous medium
    (:func:`larmor_power`), both watts. F is 1 for a dipole in a homogeneous
    medium; it is only as accurate as the grid's own dipole, so a value is
    quoted with its mesh.
    """
    if not power > 0:
        raise FDTDError(f"a radiated power is positive, got {power!r}")
    if not larmor > 0:
        raise FDTDError(f"a homogeneous power is positive, got {larmor!r}")
    return float(power) / float(larmor)


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


# A Bloch phase within this of 0 or pi (radians, per period) is taken as
# exactly that: the faces are then real (+1 or -1) and no sin set is stepped.
BLOCH_SNAP = 1e-12


def _bloch_faces(bloch, sides, primal) -> dict:
    """``{axis: (cos, sin)}`` of the Bloch phase k * period, periodic axes only.

    The field across a periodic axis obeys F(x + L) = exp(i k L) F(x), with L
    the period (the sum of the axis's ``n-1`` spacings). ``bloch`` is the wave
    vector ``(kx, ky, kz)`` in rad/m, or ``None`` for k = 0; a non-zero k on
    an axis that is not periodic is refused. A phase within ``BLOCH_SNAP`` of
    0 or pi is snapped to it (sin exactly 0), so k = 0 and the zone edge run
    the real core alone.
    """
    if bloch is None:
        bloch = (0.0, 0.0, 0.0)
    try:
        k = tuple(float(v) for v in bloch)
    except (TypeError, ValueError):
        raise FDTDError(f"bloch is a wave vector (kx, ky, kz) in rad/m, got "
                        f"{bloch!r}") from None
    if len(k) != 3 or not all(math.isfinite(v) for v in k):
        raise FDTDError(f"bloch is a wave vector (kx, ky, kz) in rad/m, got "
                        f"{bloch!r}")
    faces = {}
    for axis in range(3):
        if sides[axis] != "periodic":
            if k[axis] != 0.0:
                raise FDTDError(
                    f"a Bloch wave vector along {COMPONENTS[axis]} needs that "
                    f"axis periodic, got {sides[axis]!r}")
            continue
        phase = math.remainder(k[axis] * float(np.sum(primal[axis])),
                               2.0 * math.pi)            # in [-pi, pi]
        if abs(phase) <= BLOCH_SNAP:
            faces[axis] = (1.0, 0.0)
        elif math.pi - abs(phase) <= BLOCH_SNAP:
            faces[axis] = (-1.0, 0.0)
        else:
            faces[axis] = (math.cos(phase), math.sin(phase))
    return faces


def _wrap(sets, slab: int, faces: dict, shape) -> None:
    """The face hook: refill the copy slab of every periodic axis.

    ``slab`` 1 is the H ghost at half position ``n-1``, half sample 0 one period
    ON (times exp(i k L)); ``slab`` 0 is E node 0, node ``n-1`` one period BACK
    (times exp(-i k L)). Only the two components across the axis are copied.
    ``sets`` is the cos set, plus the sin set on the paired path, where the
    copy mixes them: (c + i s)(cos + i sin) = (c cos - s sin) + i (s cos + c sin).
    With one set, sin is 0 and the copy is a plain one (k = 0) or a sign flip
    (the zone edge).
    """
    family = 1 if slab == 1 else 0
    for axis, (c, s) in faces.items():
        n = shape[axis]
        to, frm = (n - 1, 0) if slab == 1 else (0, n - 1)
        if slab == 0:
            s = -s                                  # exp(-i k L)
        for component in COMPONENTS:
            if COMPONENTS.index(component) == axis:
                continue
            cos_values = sets[0][family][component]
            if len(sets) == 1:
                source = _take(cos_values, axis, frm)
                _set_slab(cos_values, axis, to,
                          source if c == 1.0 else -source)
                continue
            sin_values = sets[1][family][component]
            re = _take(cos_values, axis, frm)
            im = _take(sin_values, axis, frm)
            _set_slab(cos_values, axis, to, c * re - s * im)
            _set_slab(sin_values, axis, to, s * re + c * im)


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


def phasor_box_bytes(box, shape, *, frequencies=1, steps: int = 0) -> int:
    """Device bytes a ``simulate`` phasor box holds: what ``_PhasorBox`` allocates.

    ``box`` is the ``(centre, half_extent)`` the run is given, ``frequencies``
    the ``phasor_frequencies`` (a list in Hz, or just their count) and
    ``steps`` how many steps the box reads (``n_steps - phasor_start``).
    Counts, in ``complex128``: the 36 accumulators (six faces, E and H, three
    components each, every frequency), each on the slab its component needs
    -- two layers across the face where the sample sits half a cell off it,
    one where it sits on it, and the face's nodes or cells in plane -- and the
    two phase tables (one row per step read, one column per frequency); then
    the larger of two moments that never overlap: a step's transients (the
    largest slab's float64 copy and its product with the phase) and the
    faces' assembly after the last step, while the run's arrays are still
    alive (the interpolated components and the stacked faces, twice the
    returned ``E`` and ``H``, plus two of the largest slab's interpolation
    temporaries). On a discrete GPU the assembly is host memory; on the
    Jetson's unified memory it is the same RAM, so it is counted. The slab
    sizes are ``_PhasorBox``'s own, re-derived here so the estimate needs no
    device; ``tests/test_fdtd_memory.py`` holds the accumulators and tables
    equal to a real box's and the whole within a measured run's peak.
    """
    lo, hi = phasor_box_layers(box, shape)
    if isinstance(frequencies, (int, np.integer)):
        count = int(frequencies)                 # a count
    else:
        count = int(np.size(frequencies))        # the list (or one value) in Hz
    if count < 1 or int(steps) < 0:
        raise FDTDError(
            f"a phasor box reads at least one frequency over zero or more "
            f"steps, got {frequencies!r} over {steps!r}")
    total = 0
    largest = 0
    for a in range(3):
        for _layer in (hi[a], lo[a]):
            for family in ("E", "H"):
                for component in COMPONENTS:
                    size = 1
                    for axis in range(3):
                        half = _is_half(family, component, axis)
                        if axis == a:
                            size *= 2 if half else 1
                        else:
                            size *= hi[axis] - lo[axis] + (0 if half else 1)
                    total += 16 * count * size
                    largest = max(largest, size)
    total += 2 * 16 * count * int(steps)         # the E and H phase tables
    cells = sum(2 * (hi[u] - lo[u]) * (hi[v] - lo[v])
                for u, v in ((1, 2), (0, 2), (0, 1)))
    returned = 16 * count * 6 * cells            # E and H on every face cell
    step = (8 + 16 * count) * largest            # one step's transients
    assembly = 2 * returned + 2 * 16 * count * largest
    return int(total + max(step, assembly))


def memory_estimate(shape, *, boundary="metal", metals=(), pec=False,
                    record_peak=False, record_frames=False,
                    max_frames: int = MAX_FRAMES, phasor_box=None,
                    phasor_frequencies=1, phasor_steps: int = 0) -> int:
    """Bytes ``simulate`` holds at its peak on the device (float32 arrays).

    Counts, per cell: six fields, eps and 1/eps, the transients, the device
    pool's cached blocks, four CPML psi arrays per open axis (two E and two H
    derivatives along it), the conductor mask and its multiplier, and the peak
    |E_c| record (3 arrays), and the field movie (``max_frames`` frames at
    ``frame_plan``'s stride, plus ten of the same size: the three float64
    component copies and the float64 sum its |E| is built from). The frame
    count is the one the run will record, so a 120-frame movie is counted as
    one: the estimate errs high only if the window holds fewer steps than
    frames. Each metal
    ``(name, region)`` adds, per component, a box over its samples (the region
    widened one node along the component) holding 2 per pole + 4 arrays, plus
    the solver's boolean union of the regions. A ``phasor_box`` (the run's
    ``(centre, half_extent)``, with its ``phasor_frequencies`` and the
    ``phasor_steps`` it reads) adds :func:`phasor_box_bytes`; without one the
    estimate is what it was before the option existed.
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
    if record_frames:
        sampled = int(np.prod(frame_shape(shape, frame_plan(shape, 1)["stride"])))
        total += 4.0 * sampled * (int(max_frames) + 10)
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
    if phasor_box is not None:
        total += phasor_box_bytes(phasor_box, shape, frequencies=phasor_frequencies,
                                  steps=phasor_steps)
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
                 pec=None, record_peak=False, record_frames=False,
                 max_frames: int = MAX_FRAMES, phasor_box=None,
                 phasor_frequencies=1, phasor_steps: int = 0) -> dict:
    """Cells, dt, steps, device bytes and Jetson seconds for a planned run.

    ``nodes`` are the three node-coordinate arrays the run will use and
    ``metals``/``pec`` the voxeliser's masks on them. dt is found the way
    ``simulate`` finds it: 0.99 of Courant on the finest spacing per axis,
    then lowered by each metal's ``stable_dt``, so the step count is the
    one the run will take. ``phasor_box``, ``phasor_frequencies`` and
    ``phasor_steps`` are counted in the bytes as :func:`memory_estimate`
    counts them; the seconds do not count the box's per-step folding, which
    runs only over the steps it reads.
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
                                 pec=pec is not None, record_peak=record_peak,
                                 record_frames=record_frames,
                                 max_frames=max_frames, phasor_box=phasor_box,
                                 phasor_frequencies=phasor_frequencies,
                                 phasor_steps=phasor_steps),
        "setup": setup,
        "seconds": cells * steps / (RATE_SAFETY * JETSON_RATES[setup]),
        # the requested movie length, so a memory refusal can say that fewer
        # frames would fit; None when this run records no movie
        "frames": int(max_frames) if record_frames else None,
    }


def check_limits(estimate: dict, *, max_steps=None, max_seconds=None,
                 max_cells=None, max_bytes=None) -> None:
    """Raise ``EstimateError`` naming every limit ``estimate`` exceeds, the
    value that exceeded it, and the form setting that reduces it.

    ``max_cells``, ``max_steps``, ``max_seconds`` and ``max_bytes`` are the
    four limits this call applies, and each defaults to the product's own
    (``MAX_CELLS``, ``MAX_STEPS``, ``MAX_SECONDS``, ``MAX_BYTES``) when it is
    None. One caller ever passes them: the paper-1 bench script, whose
    ``--max-cells`` / ``--max-steps`` / ``--max-seconds`` / ``--max-bytes``
    raise a bench record's ceiling past what the product allows (a settled
    nanowire record needs more than 20,000 steps, and more than 30 minutes of
    Jetson, to stop rising; a dx-10 nm record of the same scene needs 9,166,668
    cells, over the 8 M cell cap by 14.6%, because the grid's fine layers do not
    scale with ``--dx``; and that scene at dx 5 nm estimates 11.55 GB of GPU
    memory on the bench's 16 GB A4000, over the 3.0 GB the product may assume).
    Nothing else can: there is no environment variable and no module global to
    set, and no product entry point passes them -- ``worker/td_solve.py`` and
    the web app call this with the defaults -- so a raised ceiling stays on the
    bench path alone.
    """
    # each read here, not bound at import: a test that moves a constant still
    # moves the limit that constant names
    if max_cells is None:
        max_cells = MAX_CELLS
    if max_steps is None:
        max_steps = MAX_STEPS
    if max_seconds is None:
        max_seconds = MAX_SECONDS
    if max_bytes is None:
        max_bytes = MAX_BYTES
    e = estimate
    finest = f"{e['finest'] * 1e9:.3g} nm"
    exceeded = []
    if e["cells"] > max_cells:
        exceeded.append(("cells", e["cells"], max_cells,
            f"The grid has {e['cells']:,} cells, over the limit of "
            f"{max_cells:,}. Increase dx to use fewer, larger cells."))
    if e["steps"] > max_steps:
        exceeded.append(("steps", e["steps"], max_steps,
            f"The run needs {e['steps']:,} time steps, over the limit of "
            f"{max_steps:,} (dt is {e['dt']:.3g} s, set by the finest cell, "
            f"{finest}). Shorten the duration, or increase dx."))
    if e["bytes"] > max_bytes:
        frames = e.get("frames") or 0
        exceeded.append(("memory", e["bytes"], max_bytes,
            f"The run needs about {e['bytes'] / 1e9:.2f} GB of GPU memory, "
            f"over the limit of {max_bytes / 1e9:.1f} GB. Increase dx, or use "
            f"metal walls instead of open ones (each open axis adds an "
            f"absorbing layer's storage across the whole grid)."
            + (f" Or record fewer than the {frames:,} movie frames asked for."
               if frames > MAX_FRAMES else "")))
    if e["seconds"] > max_seconds:
        exceeded.append(("time", e["seconds"], max_seconds,
            f"The run would take about {e['seconds'] / 60:.0f} minutes, over "
            f"the limit of {int(max_seconds) // 60} minutes. Shorten the "
            f"duration, or increase dx."))
    if exceeded:
        raise EstimateError(exceeded)


def simulate(eps_r, source: Source, n_steps: int, probes=(),
             *, spacings=None, dx: float = 1.0,
             boundary: str = "metal", pec=None, pml_cells: int = DEFAULT_PML_CELLS,
             xp=None, record_energy: bool = False, energy_region=None,
             record_peak: bool = False, record_drive: bool = False,
             record_source_field: bool = False,
             flux_box=None,
             phasor_box=None, phasor_frequencies=None, phasor_start: int = 0,
             phasor_window: str | None = "hann",
             ade_terms=(), check=None,
             frames=None, bloch=None) -> dict:
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
    "spacings", "energy", "peak", "drive", "source_field", "flux",
    "source_volume", "source_eps", "phasor_faces"}``, and every
    one of them is HOST numpy whatever ``xp``
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

    ``frames`` (a :func:`frame_plan` dict) adds ``"frames"``: ``|E|`` after
    the E update of each listed step, on the nodes ``stride`` keeps, as one
    ``(n_frames, sx, sy, sz)`` float32 array in the grid's (x, y, z) order.
    The three components are combined where they are stored, half a cell
    apart, without interpolation: it is a picture of the field, not a
    measurement. It is written into a device array (no sync a step).

    ``bloch`` is a Bloch wave vector ``(kx, ky, kz)`` in rad/m on the periodic
    axes (``None`` is k = 0): the cell then models an infinite crystal whose
    field obeys F(x + L) = exp(i k L) F(x). The core stays real float32. At
    k = 0 and the zone edge (k L = pi) the faces are real and only the cos set
    runs, so k = 0 is the plain periodic run. Anywhere else a second, sin,
    set of the six arrays is stepped beside the cos set through the same
    update, and the face hook (``_wrap``) cross-couples the two at the copy
    slabs; only the cos set is driven. The result's ``"traces_sin"`` holds the
    probes on the sin set (``None`` on the real path), energy sums both sets,
    a frame is the complex field's ``|E|``, and ``peak`` is the cos set's. A
    paired run refuses a plane wave and polarisation terms (metals).

    ``record_drive`` adds ``"drive"``: the value a point source added at each
    step, as recorded where the update read it, not recomputed from the
    ``Source`` afterwards. A measurement that needs the source's own moment
    (:func:`dipole_moment`) takes it from here, so it can never disagree with
    the waveform the run used. ``None`` when not asked for, and also when there
    is no point source to drive with (a plane wave, or a paired set, whose cos
    set is the only one driven).

    ``record_source_field`` adds ``"source_field"``: that same source cell's own
    E component each step, read just after the injection, under the same
    conditions. It is the other half of :func:`delivered_power` and it is
    recorded by the simulator rather than asked for as a probe, because a probe
    is a pair of indices the caller supplies and a delivered power computed
    against a probe anywhere else would be a plausible number for the wrong cell.

    ``source_volume`` is the cell the point source's current fills
    (:func:`source_cell_volume`), the volume a moment is derived over. ``None``
    with no point source.

    ``source_eps`` is the relative permittivity the E update applies at the
    point source's own sample (:func:`source_permittivity`, from this run's
    ``eps_r``, ``ade_terms`` and ``boundary``): the ``eps_r`` the power and
    moment readings take. It is ``eps_r[cell]`` in a dielectric and
    ``eps_inf`` (1) on a Lorentz-Drude metal's sample. ``None`` with no point
    source.

    ``flux_box`` adds ``"flux"``: ``(centre, half_extent)`` in cells (three
    integers each) makes a closed box of NODE layers, and the result is one
    per-step record of ``sum |E|^2 * area`` over its six faces
    (:func:`flux_faces`). That is the radiated power through the surface up to
    the wave-zone factor ``1 / eta`` (:func:`radiated_power`), and it is one
    scalar a step rather than a trace per surface cell, so it fits a production
    grid. The box must be strictly inside the live layers 1..n-2 on every axis,
    outside the absorbing layer (``pml_cells`` deep on an open axis: in the
    absorber the surface measures the damped field and reads low without
    complaining), and in the wave zone: at least a wavelength from every
    scatterer, so that the field on it is an outgoing wave. A paired (Bloch) run
    has no physical field to put a surface around and refuses a box.

    ``phasor_box`` with ``phasor_frequencies`` adds ``"phasor_faces"``: the
    closed surface ``farfield.py`` transforms, recorded as complex phasors. The
    box is ``(centre, half_extent)`` in cells, the flux box's spelling, and its
    faces are the same NODE layers ``centre +/- half_extent``; the rules are the
    flux box's too (strictly inside the live layers 1..n-2, never in the
    absorber, and a paired Bloch run refuses it), except that each face also
    reads H half a cell outside itself, so on an open axis the faces must lie in
    ``pml_cells + 1 .. n - 2 - pml_cells`` (:func:`phasor_box_layers`).
    ``phasor_frequencies`` is one frequency or several, in Hz, each positive
    and below ``1 / (2 dt)``. Every step from ``phasor_start`` on, the six
    components on each face are folded into a running DFT,

        F(f) = 2 sum_n w_n F_n exp(+i 2 pi f t_n) / sum_n w_n,

    with ``w`` the ``phasor_window`` (``"hann"`` or ``None``, the boxcar) over
    those steps and ``t_n`` each family's own sample time: ``(n + 1) dt`` for
    E, ``(n + 1/2) dt`` for H, the half step the leapfrog staggers them by. For
    a steady tone ``Re[A exp(-i w t)]`` that is the amplitude ``A``, in the
    ``exp(-i w t)`` convention ``farfield.py`` uses, and it is the
    normalisation :func:`delivered_power` reads, so a CW run read over its
    steady tail (``phasor_start`` at the tail's first step) gives powers in W.
    After the run the staggered samples are interpolated to the face-cell
    centres -- linearly across the face (H, and the normal E, sit half a cell
    either side of it), as the mean of the two end nodes along an in-plane axis
    where a sample sits on the nodes -- so E and H are colocated, as the
    far-field contract requires. ``"phasor_faces"`` is a list with one entry
    per frequency, in the order given, and each entry is six ``farfield.Face``
    in ``DIRECTIONS`` order: position and cell edges in metres with node 0 at
    the origin of every axis, the cells being the box's own primal cells. The
    medium outside the box must be vacuum for the transform to apply. The
    accumulators stay on the device and cross to the host once, at the end;
    they cost ``36 * n_freq`` complex128 slabs of about one face each. ``None``
    when not asked for.

    ``check``, if given, is called as ``check(step)`` before step 0 and every
    ``CHECK_EVERY`` steps after it, on the host; it stops the run by raising
    (the worker's 45-minute limit). It reads no device value, so it adds no
    sync: on cupy the host may be a launch queue ahead of the device, which
    is seconds, not minutes.
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

    faces = _bloch_faces(bloch, sides, primal)
    paired = any(s != 0.0 for _c, s in faces.values())
    if paired and source.kind == "plane":
        raise FDTDError(
            "a plane wave cannot run with a Bloch phase away from k = 0 and "
            "the zone edge; a Bloch cell is driven by a point source")
    if paired and terms:
        raise FDTDError(
            "a metal (a polarisation term) cannot run with a Bloch phase away "
            "from k = 0 and the zone edge yet: its polarisation state would "
            "need a sin copy too")

    inv_eps = 1.0 / (EPS0 * eps)

    def new_set():
        """One set of the six field arrays, keyed E x/y/z then H x/y/z."""
        return ({c: xp.zeros(shape, dtype=f32) for c in COMPONENTS},
                {c: xp.zeros(shape, dtype=f32) for c in COMPONENTS})

    # The cos set is the physical field and the only one driven; the sin set
    # exists only on the paired path, where the Bloch faces feed it from the
    # cos set (see _bloch_faces).
    sets = [new_set()] + ([new_set()] if paired else [])
    field, h_field = sets[0]
    Ex, Ey, Ez = (field[c] for c in COMPONENTS)
    Hx, Hy, Hz = (h_field[c] for c in COMPONENTS)

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
    traces_sin = (xp.zeros((len(probe_list), n_steps), dtype=f32)
                  if paired else None)
    energy: list[float] | None = [] if record_energy else None
    # The drive a point source added, where the update read it. Only the cos set
    # is driven, so a paired run has nothing to record beyond the plane-wave
    # case below; both end as None rather than an array of zeros.
    drive: list[float] | None = (
        [] if record_drive and not paired and source.kind == "point" else None)
    # The field at the source's own cell and component, each step after the
    # injection -- the other half of :func:`delivered_power`. Recorded here,
    # where the source's cell is already in hand, so a caller cannot pair a
    # drive with a probe that happens to sit somewhere else.
    source_field = (xp.zeros(n_steps, dtype=f32)
                    if record_source_field and plane is None and not paired
                    and source.kind == "point" else None)
    if flux_box is not None and paired:
        raise FDTDError(
            "a flux box needs the physical field, and away from k = 0 and the "
            "zone edge a Bloch run's cos set is not it (the field is the cos "
            "set plus i times the sin set); drop flux_box or the Bloch phase")
    flux_terms = (None if flux_box is None else
                  [(component, slab, xp.asarray(areas))
                   for component, slab, areas in flux_faces(
                       flux_box, shape, primal,
                       [int(pml_cells) if sides[axis] == "open" else 0
                        for axis in range(3)])])
    flux_array = (None if flux_terms is None
                  else xp.zeros(n_steps, dtype=xp.float64))
    if (phasor_box is None) != (phasor_frequencies is None):
        raise FDTDError(
            "a phasor box needs both phasor_box (where) and phasor_frequencies "
            "(at which frequencies); got only "
            + ("phasor_box" if phasor_frequencies is None
               else "phasor_frequencies"))
    if phasor_box is not None and paired:
        raise FDTDError(
            "a phasor box needs the physical field, and away from k = 0 and "
            "the zone edge a Bloch run's cos set is not it (the field is the "
            "cos set plus i times the sin set); drop phasor_box or the Bloch "
            "phase")
    phasor = (None if phasor_box is None else _PhasorBox(
        phasor_box, phasor_frequencies, shape=shape, primal=primal,
        pml=[int(pml_cells) if sides[axis] == "open" else 0
             for axis in range(3)],
        dt=dt, n_steps=n_steps, start=phasor_start, window=phasor_window,
        xp=xp))
    # The volume the source's own moment is taken over -- computed here, from
    # the grid the run actually used, so a consumer cannot pair a moment with
    # another run's cell.
    source_volume = (source_cell_volume(primal, source.cell, source.component)
                     if source.kind == "point" else None)
    # The permittivity the update applies at the source's own sample, from the
    # array and terms this run was given: the eps the power and moment
    # readings take, so a consumer need not pass the layout's node value.
    source_eps = (source_permittivity(eps_r, source.cell, source.component,
                                      ade_terms=terms, boundary=boundary)
                  if source.kind == "point" else None)
    peak = ({c: xp.zeros(shape, dtype=f32) for c in COMPONENTS}
            if record_peak else None)
    movie = frame_at = sampled = None
    if frames is not None:
        stride = tuple(int(k) for k in frames["stride"])
        if len(stride) != 3 or min(stride) < 1:
            raise FDTDError(f"a frame stride is three whole numbers >= 1, "
                            f"got {frames['stride']!r}")
        frame_at = {}
        for index, s in enumerate(frames["steps"]):
            if not 0 <= int(s) < n_steps or int(s) in frame_at:
                raise FDTDError(f"frame step {s!r} is repeated or outside "
                                f"the run's {n_steps} steps")
            frame_at[int(s)] = index
        sampled = tuple(slice(None, None, k) for k in stride)
        movie = xp.zeros((len(frame_at),) + frame_shape(shape, stride),
                         dtype=f32)
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

    def update_h(f, tag):
        """Advance one set's H by half a step (E at n to H at n+1/2)."""
        Ex, Ey, Ez = (f[0][c] for c in COMPONENTS)
        Hx, Hy, Hz = (f[1][c] for c in COMPONENTS)
        # ---- H from the curl of E (E at n, H advances to n+1/2) ----------
        # Hx at (i, j+1/2, k+1/2): live on the full x range, y <= ny-2, z <= nz-2
        term_z = (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / _along(P[1], 1)   # dEz/dy
        term_y = (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / _along(P[2], 2)   # dEy/dz
        term_z = corrected(term_z, (tag, "Hx", "z"), 1, "dual")
        term_y = corrected(term_y, (tag, "Hx", "y"), 2, "dual")
        Hx[:, :-1, :-1] -= dt_over_mu0 * (term_z - term_y)

        # Hy at (i+1/2, j, k+1/2): live on the full y range, x <= nx-2, z <= nz-2
        term_x = (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / _along(P[2], 2)   # dEx/dz
        term_z = (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / _along(P[0], 0)   # dEz/dx
        term_x = corrected(term_x, (tag, "Hy", "x"), 2, "dual")
        term_z = corrected(term_z, (tag, "Hy", "z"), 0, "dual")
        Hy[:-1, :, :-1] -= dt_over_mu0 * (term_x - term_z)

        # Hz at (i+1/2, j+1/2, k): live on the full z range, x <= nx-2, y <= ny-2
        term_y = (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / _along(P[0], 0)   # dEy/dx
        term_x = (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / _along(P[1], 1)   # dEx/dy
        term_y = corrected(term_y, (tag, "Hz", "y"), 0, "dual")
        term_x = corrected(term_x, (tag, "Hz", "x"), 1, "dual")
        Hz[:-1, :-1, :] -= dt_over_mu0 * (term_y - term_x)

    def update_e(f, tag):
        """Advance one set's E by a step (H at n+1/2 to E at n+1)."""
        Ex, Ey, Ez = (f[0][c] for c in COMPONENTS)
        Hx, Hy, Hz = (f[1][c] for c in COMPONENTS)
        # ---- E from the curl of H (H at n+1/2, E advances to n+1) --------
        # Each integer-position axis runs over ``nodes`` (1..n-2, or 1..n-1
        # when periodic); see where they are built.
        # Ex at (i+1/2, j, k)
        term_y = (Hz[:-1, sy, sz] - Hz[:-1, ly, sz]) / _along(span[1], 1)
        term_z = (Hy[:-1, sy, sz] - Hy[:-1, sy, lz]) / _along(span[2], 2)
        term_y = corrected(term_y, (tag, "Ex", "y"), 1, "primal")
        term_z = corrected(term_z, (tag, "Ex", "z"), 2, "primal")
        Ex[:-1, sy, sz] += dt * inv_eps[:-1, sy, sz] * (term_y - term_z)

        # Ey at (i, j+1/2, k)
        term_z = (Hx[sx, :-1, sz] - Hx[sx, :-1, lz]) / _along(span[2], 2)
        term_x = (Hz[sx, :-1, sz] - Hz[lx, :-1, sz]) / _along(span[0], 0)
        term_z = corrected(term_z, (tag, "Ey", "z"), 2, "primal")
        term_x = corrected(term_x, (tag, "Ey", "x"), 0, "primal")
        Ey[sx, :-1, sz] += dt * inv_eps[sx, :-1, sz] * (term_z - term_x)

        # Ez at (i, j, k+1/2)
        term_x = (Hy[sx, sy, :-1] - Hy[lx, sy, :-1]) / _along(span[0], 0)
        term_y = (Hx[sx, sy, :-1] - Hx[sx, ly, :-1]) / _along(span[1], 1)
        term_x = corrected(term_x, (tag, "Ez", "x"), 0, "primal")
        term_y = corrected(term_y, (tag, "Ez", "y"), 1, "primal")
        Ez[sx, sy, :-1] += dt * inv_eps[sx, sy, :-1] * (term_x - term_y)

    def walls(f):
        """Zero one set's E inside conductors and on the metal faces."""
        Ex, Ey, Ez = (f[0][c] for c in COMPONENTS)
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

    progress_start = time.perf_counter()
    n_cells = int(np.prod(shape))
    for step in range(n_steps):
        if step and step % PROGRESS_EVERY == 0:
            progress_seconds = time.perf_counter() - progress_start
            print(f"progress  step {step:,} of {n_steps:,},"
                  f" {progress_seconds:,.1f} s,"
                  f" {n_cells * step / progress_seconds / 1e6:,.1f}"
                  f" Mcell-steps/s", flush=True)
        if check is not None and step % CHECK_EVERY == 0:
            check(step)
        t = step * dt
        if energy is not None:
            h_before = [tuple(f[1][c].copy() for c in COMPONENTS)
                        for f in sets]                        # H at n-1/2

        for tag, f in enumerate(sets):
            update_h(f, tag)

        if plane is not None:
            plane.correct_h(h_field)          # uses the incident E at n
            plane.advance_h(step)             # incident H to n+1/2

        # Periodic axes: the H samples at HALF positions along the axis have
        # one more beyond the last, the ghost slot n-1, which is half sample 0
        # one period on. The E update at node n-1 reads it.
        _wrap(sets, 1, faces, shape)

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
            #
            # On the paired path the field is cos + i sin, and the quadratic is
            # the sum over both sets (|E|^2 = cos^2 + sin^2).
            e_total = 0.0
            h_total = 0.0
            for (e_set, h_set), before_set in zip(sets, h_before):
                for component in COMPONENTS:
                    wide_values = e_set[component][region].astype(xp.float64)
                    e_total += float(xp.sum(eps64 * e_weights[component]
                                            * wide_values * wide_values))
                for index, component in enumerate(COMPONENTS):
                    now = h_set[component][region].astype(xp.float64)
                    before = before_set[index][region].astype(xp.float64)
                    h_total += float(xp.sum(h_weights[component] * now
                                            * before))
            energy.append(0.5 * (EPS0 * e_total + MU0 * h_total))

        for term in terms:                    # dispersive terms see E^n
            prepare = getattr(term, "prepare", None)
            if prepare is not None:
                prepare(field, dt, step)

        for tag, f in enumerate(sets):
            update_e(f, tag)

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
                if isinstance(increment, tuple):  # (where, values)
                    # not ``region``: that name is the energy sum's box
                    where, values = increment
                    field[name][where] += values
                else:
                    field[name] += increment

        # ---- source ------------------------------------------------------
        if plane is None:
            value = source.value(t)
            field[source.component][source.cell] += dt * value
            if drive is not None:
                drive.append(value)
            if source_field is not None:
                source_field[step] = field[source.component][source.cell]

        # ---- PEC ---------------------------------------------------------
        for f in sets:
            walls(f)

        # Periodic axes: node 0 is node n-1, which the update just advanced
        # (with every correction, term, source and conductor applied to it),
        # one period back.
        _wrap(sets, 0, faces, shape)

        # ---- probes and energy -------------------------------------------
        for index, (cell, component) in enumerate(probe_list):
            if component == "mag":
                value = xp.sqrt(Ex[cell] ** 2 + Ey[cell] ** 2 + Ez[cell] ** 2)
            else:
                value = field[component][cell]
            traces[index, step] = value
            if paired:
                sin_field = sets[1][0]
                if component == "mag":
                    value = xp.sqrt(sum(sin_field[c][cell] ** 2
                                        for c in COMPONENTS))
                else:
                    value = sin_field[component][cell]
                traces_sin[index, step] = value
        if frame_at is not None and step in frame_at:
            # float64: a field of 1e-21 squares to below float32's normal
            # range and would lose digits before the square root. On the
            # paired path |E| is the complex field's, both sets together.
            total = 0.0
            for e_set, _h_set in sets:
                for c in COMPONENTS:
                    wide_values = e_set[c][sampled].astype(xp.float64)
                    total = total + wide_values * wide_values
            movie[frame_at[step]] = xp.sqrt(total)
        if peak is not None:
            for component in COMPONENTS:
                xp.maximum(peak[component], xp.abs(field[component]),
                           out=peak[component])
        if flux_array is not None:
            flux_array[step] = _flux_step(field, flux_terms, xp)
        if phasor is not None:
            phasor.accumulate(step, field, h_field)

    progress_seconds = time.perf_counter() - progress_start
    print(f"progress  step {n_steps:,} of {n_steps:,}, {progress_seconds:,.1f} s,"
          f" {n_cells * n_steps / progress_seconds / 1e6:,.1f} Mcell-steps/s"
          f"  (done)", flush=True)

    if peak is not None:
        peak = {c: _host_result(values, xp) for c, values in peak.items()}
    return {"traces": _host_result(traces, xp), "dt": dt, "steps": n_steps,
            "spacings": [np.array(p) for p in primal], "energy": energy,
            "peak": peak,
            "drive": (None if drive is None
                      else np.array(drive, dtype=np.float64)),
            "source_field": (_host_result(source_field, xp)
                             if source_field is not None else None),
            "flux": (None if flux_array is None
                     else _host_result(flux_array, xp)),
            "source_volume": source_volume,
            "source_eps": source_eps,
            "frames": None if movie is None else _host_result(movie, xp),
            "traces_sin": (_host_result(traces_sin, xp) if paired
                           else None),
            "phasor_faces": None if phasor is None else phasor.faces()}


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
