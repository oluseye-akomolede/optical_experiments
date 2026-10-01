"""3-D Yee FDTD on a uniform grid: the time-domain core.

WHY THIS FILE EXISTS. The owner asked for time-series simulations with a
user-defined source (Slack, 2026-10-01), and chose a full time-stepping
simulation on a grid. Nothing else in this directory can do it: the cavity and
photonic-crystal solvers are EIGEN-solvers, so they show a steady state and
never a field building up, a pulse travelling, or light scattering off an
object. This is the engine that can.

SCOPE (phase 1). The solver core only: the Yee update, PEC, a CPML absorbing
boundary, a point source with a CW and a pulse profile, and probes with their
spectra. The plane-wave source is phase 2, and the voxeliser, the estimate, the
job routing and the page are phases 3-5 -- there is deliberately no I/O, no
database and no web code in this file.

THE ARRAY MODULE IS THE CALLER'S CHOICE. ``xp`` is numpy for the tiny local
tests and cupy on the Jetson, and **cupy is never imported here** -- not at
module level, not in a helper. The same update code runs on both, which is what
makes the local tests evidence about the GPU run rather than about a second
implementation. numpy is imported for the default and for the Fourier window,
nothing else.

UNITS are SI and the grid is uniform: ``dx`` in metres, ``dt`` in seconds,
frequencies in Hz, E in V/m, H in A/m. The source is a current density in A/m2,
so the field scale it produces is not arbitrary -- but every test here is about
a SHAPE (a spectrum peak, a conserved energy), never about an absolute
amplitude.

THE STAGGERING IS IN THE INDICES. Each field component is stored on its own
lattice and the differences below are the half-cell offsets: a backward
difference for the E update (valid at ``i >= 1``), a forward one for the H
update (valid at ``i <= n-2``). The update regions are therefore not the whole
array and NOTHING reads outside its slice -- there is no edge padding to get
wrong, and a PEC face is a zero rather than a special case.

THE TIME STEP is ``0.99 * dx / (c * sqrt(3))``, which is the spec's value and
is below the 3-D Courant limit ``dx / (c * sqrt(3))`` on every axis at once.
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299792458.0                 # m/s
EPS0 = 8.8541878128e-12          # F/m
MU0 = 1.25663706212e-6           # H/m
ETA0 = math.sqrt(MU0 / EPS0)     # ohm

BOUNDARIES = ("metal", "open")
PROFILES = ("cw", "pulse")
COMPONENTS = ("x", "y", "z")

#: The Courant safety factor. The spec pins it, so it is a module constant
#: rather than a default someone can quietly raise.
SAFETY = 0.99

#: CPML constants (Roden & Gedney): a cubic conductivity ramp reaching
#: ``sigma_max`` at the outer face, sized for a round-trip reflection of -120 dB.
PML_ORDER = 3
PML_REFLECTION = 1e-6
DEFAULT_PML_CELLS = 10


class FDTDError(Exception):
    """A request this solver will not run, with the reason in the message."""


# --------------------------------------------------------------------------
# time step
# --------------------------------------------------------------------------

def courant_limit(dx: float) -> float:
    """The largest stable step for a 3-D Yee cell of side ``dx``."""
    if not dx > 0:
        raise FDTDError(f"dx must be positive, got {dx!r}")
    return dx / (C0 * math.sqrt(3.0))


def courant_dt(dx: float, safety: float = SAFETY) -> float:
    """The step the solver uses: ``safety * dx / (c * sqrt(3))``.

    The safety factor is checked here rather than trusted, because a factor at
    or above 1 does not fail loudly -- it grows the field until it overflows,
    which reads as a physics result.
    """
    if not 0 < safety < 1:
        raise FDTDError(
            f"the Courant safety factor must be in (0, 1), got {safety!r}: at "
            f"or above 1 the update is unstable on the diagonal of the cell"
        )
    return safety * courant_limit(dx)


# --------------------------------------------------------------------------
# time profiles
# --------------------------------------------------------------------------

def cw_profile(t: float, frequency: float, ramp_periods: float = 5.0) -> float:
    """Steady state at ``frequency``, turned on smoothly over a few periods.

    The turn-on is a raised cosine over ``ramp_periods`` periods. A hard start
    would put a step's worth of every frequency into the run, which is the one
    thing a CW source is meant not to do.
    """
    if t <= 0.0:
        return 0.0
    ramp = ramp_periods / frequency
    if t < ramp:
        envelope = 0.5 * (1.0 - math.cos(math.pi * t / ramp))
    else:
        envelope = 1.0
    return envelope * math.sin(2.0 * math.pi * frequency * t)


def pulse_profile(t: float, frequency: float, bandwidth: float,
                  start: float = 5.0) -> float:
    """A Gaussian envelope on a carrier at ``frequency``.

    ``bandwidth`` is the 1/e half-width of the envelope's SPECTRUM, in Hz, so
    the pulse has a definite meaning in the frequency domain: it is the width
    of the feature the run is probing with. The carrier puts the spectral peak
    at ``frequency``, and the envelope is started ``start`` standard
    deviations before its centre so the run begins at rest.
    """
    if bandwidth <= 0.0:
        raise FDTDError(
            f"a pulse needs a positive bandwidth, got {bandwidth!r}: it is the "
            f"width of the spectrum the run is probing with"
        )
    sigma = 1.0 / (math.pi * bandwidth)
    t0 = start * sigma
    return math.exp(-(((t - t0) / sigma) ** 2)) * math.sin(
        2.0 * math.pi * frequency * (t - t0))


class Source:
    """A soft current on one E component at one cell.

    ``kind="point"`` is the only kind in phase 1; ``plane`` is refused BY NAME
    here rather than silently ignored, because a plane wave that quietly ran as
    a dipole would look like a physics result.
    """

    def __init__(self, cell, frequency: float, *, component: str = "z",
                 profile: str = "pulse", bandwidth: float | None = None,
                 amplitude: float = 1.0, kind: str = "point",
                 ramp_periods: float = 5.0):
        if kind != "point":
            raise FDTDError(
                f"source kind {kind!r} is not in this phase: only 'point' "
                f"exists today, and the plane wave needs a TF/SF box (phase 2)"
            )
        if component not in COMPONENTS:
            raise FDTDError(
                f"component must be one of {COMPONENTS}, got {component!r}")
        if profile not in PROFILES:
            raise FDTDError(f"profile must be one of {PROFILES}, got {profile!r}")
        if not frequency > 0:
            raise FDTDError(f"frequency must be positive, got {frequency!r}")
        if profile == "pulse" and bandwidth is None:
            raise FDTDError("a 'pulse' source needs a bandwidth")
        if profile == "cw" and bandwidth is not None:
            raise FDTDError("a 'cw' source has no bandwidth; it is one frequency")
        self.cell = tuple(int(c) for c in cell)
        self.frequency = float(frequency)
        self.component = component
        self.profile = profile
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.amplitude = float(amplitude)
        self.ramp_periods = float(ramp_periods)

    def value(self, t: float) -> float:
        """The current density at time ``t``. Exactly zero once a pulse is over."""
        if self.profile == "cw":
            return self.amplitude * cw_profile(t, self.frequency,
                                               self.ramp_periods)
        return self.amplitude * pulse_profile(t, self.frequency, self.bandwidth)

    def end_time(self) -> float:
        """When a pulse is done, or ``inf`` for a source that never stops."""
        if self.profile == "cw":
            return math.inf
        sigma = 1.0 / (math.pi * self.bandwidth)
        return 5.0 * sigma + 5.0 * sigma


# --------------------------------------------------------------------------
# CPML
# --------------------------------------------------------------------------

def _pml_sigma(n: int, pml: int, dx: float) -> np.ndarray:
    """The conductivity ramp along one axis: 0 in the interior, max at the face."""
    sigma = np.zeros(n, dtype=np.float64)
    if pml <= 0:
        return sigma
    if n <= 2 * pml:
        raise FDTDError(
            f"a CPML of {pml} cells does not fit in {n} cells on an axis: the "
            f"interior would be empty (need more than {2 * pml}). Give more "
            f"cells, or a smaller pml_cells"
        )
    sigma_max = ((PML_ORDER + 1) * math.log(1.0 / PML_REFLECTION)
                 / (2.0 * ETA0 * pml * dx))
    for i in range(pml):
        # Half a cell in from the sample, so the deepest sample is not the face
        # itself and the ramp has cube growth from a half-cell, not from zero.
        depth = (pml - i - 0.5) / pml
        value = sigma_max * depth ** PML_ORDER
        sigma[i] = value
        sigma[n - 1 - i] = value
    return sigma


def _pml_coefficients(n: int, pml: int, dx: float, dt: float, xp):
    """The ``(b, c)`` pair CPML updates each auxiliary field with."""
    sigma = _pml_sigma(n, pml, dx)
    b = np.exp(-sigma * dt / EPS0)
    c = b - 1.0                       # kappa = 1, alpha = 0: plain conductivity
    return (xp.asarray(b, dtype=xp.float32), xp.asarray(c, dtype=xp.float32))


def _pml_term(raw, psi, b, c, axis: int):
    """The CPML-corrected derivative ``d/dx`` from its stored auxiliary field.

    ``psi`` is carried between steps; ``b`` and ``c`` are the 1-D profiles over
    the derivative's own axis, already sliced to the update's extent.
    """
    shape = [1, 1, 1]
    shape[axis] = -1
    bb = b.reshape(shape)
    cc = c.reshape(shape)
    psi *= bb
    psi += cc * raw
    return raw + psi                  # kappa = 1


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

    The window is applied because the trace is a finite slice of a signal that
    is still ringing when it ends; without it the discontinuity at the ends
    lays a sinc skirt over every peak. It is built in numpy and moved to
    ``xp``, so the same window is used on both backends.
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

def simulate(eps_r, source: Source, n_steps: int, probes=(),
             *, dx: float = 1.0,
             boundary: str = "metal", pec=None, pml_cells: int = DEFAULT_PML_CELLS,
             xp=None, record_energy: bool = False) -> dict:
    """Run the grid and return the probe traces.

    ``eps_r`` ``(nx, ny, nz)`` is the relative permittivity per cell and it
    DEFINES the grid; ``pec`` is an optional boolean conductor mask of the same
    shape. ``boundary="metal"`` closes the outer faces with a PEC wall;
    ``boundary="open"`` puts a CPML of ``pml_cells`` on each face instead.

    Returns ``{"traces": (n_probes, n_steps) float32, "dt", "steps",
    "energy"}``. ``energy`` is the total electromagnetic energy per step, or
    ``None`` unless ``record_energy`` asked for it -- it is a full reduction of
    six arrays, which is cheap on a 24^3 test grid and not on a production one.
    """
    xp = np if xp is None else xp
    f32 = xp.float32

    eps = xp.asarray(eps_r, dtype=f32)
    if eps.ndim != 3:
        raise FDTDError(f"eps_r must be 3-D (nx, ny, nz), got shape {eps.shape}")
    if not bool(xp.all(eps > 0)):
        raise FDTDError("eps_r must be positive everywhere")
    shape = tuple(int(s) for s in eps.shape)

    if boundary not in BOUNDARIES:
        raise FDTDError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    if int(n_steps) < 1:
        raise FDTDError(f"n_steps must be at least 1, got {n_steps!r}")
    n_steps = int(n_steps)

    mask = None
    if pec is not None:
        mask = xp.asarray(pec, dtype=bool)
        if tuple(mask.shape) != shape:
            raise FDTDError(
                f"the conductor mask is {tuple(mask.shape)} but the grid is "
                f"{shape}; they must be the same shape")

    if boundary == "open":
        for axis, n in enumerate(shape):
            if n <= 2 * int(pml_cells):
                raise FDTDError(
                    f"a CPML of {pml_cells} cells does not fit on axis {axis} "
                    f"({n} cells): the absorbing layer would leave no interior")

    cell = source.cell
    for axis, index in enumerate(cell):
        if not 0 <= index < shape[axis]:
            raise FDTDError(
                f"the source cell {cell} is outside the grid {shape} "
                f"(axis {axis} has {shape[axis]} cells)")
    if mask is not None and bool(mask[cell]):
        raise FDTDError(
            f"the source cell {cell} is inside the conductor; a soft current "
            f"there is zeroed by the PEC update on the same step and would "
            f"excite nothing")
    probe_list = _normalise_probes(probes)
    for probe_cell, _component in probe_list:
        for axis, index in enumerate(probe_cell):
            if not 0 <= index < shape[axis]:
                raise FDTDError(
                    f"the probe cell {probe_cell} is outside the grid {shape} "
                    f"(axis {axis} has {shape[axis]} cells)")

    dt = courant_dt(dx)
    inv_eps = 1.0 / (EPS0 * eps)

    Ex = xp.zeros(shape, dtype=f32)
    Ey = xp.zeros(shape, dtype=f32)
    Ez = xp.zeros(shape, dtype=f32)
    Hx = xp.zeros(shape, dtype=f32)
    Hy = xp.zeros(shape, dtype=f32)
    Hz = xp.zeros(shape, dtype=f32)
    field = {"x": Ex, "y": Ey, "z": Ez}

    nx, ny, nz = shape
    if boundary == "open":
        bx, cx = _pml_coefficients(nx, int(pml_cells), dx, dt, xp)
        by, cy = _pml_coefficients(ny, int(pml_cells), dx, dt, xp)
        bz, cz = _pml_coefficients(nz, int(pml_cells), dx, dt, xp)
        # One auxiliary field per derivative that crosses a PML face. Each is
        # shaped like the UPDATE REGION it corrects, so the index arithmetic is
        # the update's own and an out-of-range psi cannot happen.
        psi_Exy = xp.zeros((nx, ny - 1, nz - 1), f32)
        psi_Exz = xp.zeros((nx, ny - 1, nz - 1), f32)
        psi_Eyz = xp.zeros((nx - 1, ny, nz - 1), f32)
        psi_Eyx = xp.zeros((nx - 1, ny, nz - 1), f32)
        psi_Ezx = xp.zeros((nx - 1, ny - 1, nz), f32)
        psi_Ezy = xp.zeros((nx - 1, ny - 1, nz), f32)
        psi_Hxz = xp.zeros((nx, ny - 1, nz - 1), f32)
        psi_Hxy = xp.zeros((nx, ny - 1, nz - 1), f32)
        psi_Hyx = xp.zeros((nx - 1, ny, nz - 1), f32)
        psi_Hyz = xp.zeros((nx - 1, ny, nz - 1), f32)
        psi_Hzy = xp.zeros((nx - 1, ny - 1, nz), f32)
        psi_Hzx = xp.zeros((nx - 1, ny - 1, nz), f32)
        bx1, cx1 = bx[1:], cx[1:]
        by1, cy1 = by[1:], cy[1:]
        bz1, cz1 = bz[1:], cz[1:]
        bx0, cx0 = bx[:-1], cx[:-1]
        by0, cy0 = by[:-1], cy[:-1]
        bz0, cz0 = bz[:-1], cz[:-1]

    traces = xp.zeros((len(probe_list), n_steps), dtype=f32)
    energy: list[float] | None = [] if record_energy else None
    inv_dx = 1.0 / dx
    dt_over_mu0 = dt / MU0

    for step in range(n_steps):
        t = step * dt

        # ---- H from the curl of E (forward differences) ------------------
        raw_z = (Ey[:, :, 1:] - Ey[:, :, :-1]) * inv_dx
        raw_y = (Ez[:, 1:, :] - Ez[:, :-1, :]) * inv_dx
        term_z = raw_z[:, :-1, :]
        term_y = raw_y[:, :, :-1]
        if boundary == "open":
            term_z = _pml_term(term_z, psi_Hxz, bz0, cz0, 2)
            term_y = _pml_term(term_y, psi_Hxy, by0, cy0, 1)
        Hx[:, :-1, :-1] -= dt_over_mu0 * (term_z - term_y)

        raw_x = (Ez[1:, :, :] - Ez[:-1, :, :]) * inv_dx
        raw_z = (Ex[:, :, 1:] - Ex[:, :, :-1]) * inv_dx
        term_x = raw_x[:, :, :-1]
        term_z = raw_z[:-1, :, :]
        if boundary == "open":
            term_x = _pml_term(term_x, psi_Hyx, bx0, cx0, 0)
            term_z = _pml_term(term_z, psi_Hyz, bz0, cz0, 2)
        Hy[:-1, :, :-1] -= dt_over_mu0 * (term_x - term_z)

        raw_y = (Ex[:, 1:, :] - Ex[:, :-1, :]) * inv_dx
        raw_x = (Ey[1:, :, :] - Ey[:-1, :, :]) * inv_dx
        term_y = raw_y[:-1, :, :]
        term_x = raw_x[:, :-1, :]
        if boundary == "open":
            term_y = _pml_term(term_y, psi_Hzy, by0, cy0, 1)
            term_x = _pml_term(term_x, psi_Hzx, bx0, cx0, 0)
        Hz[:-1, :-1, :] -= dt_over_mu0 * (term_y - term_x)

        # ---- E from the curl of H (backward differences) -----------------
        raw_y = (Hz[:, 1:, :] - Hz[:, :-1, :]) * inv_dx
        raw_z = (Hy[:, :, 1:] - Hy[:, :, :-1]) * inv_dx
        term_y = raw_y[:, :, 1:]
        term_z = raw_z[:, 1:, :]
        if boundary == "open":
            term_y = _pml_term(term_y, psi_Exy, by1, cy1, 1)
            term_z = _pml_term(term_z, psi_Exz, bz1, cz1, 2)
        Ex[:, 1:, 1:] += dt * inv_eps[:, 1:, 1:] * (term_y - term_z)

        raw_z = (Hx[:, :, 1:] - Hx[:, :, :-1]) * inv_dx
        raw_x = (Hz[1:, :, :] - Hz[:-1, :, :]) * inv_dx
        term_z = raw_z[1:, :, :]
        term_x = raw_x[:, :, 1:]
        if boundary == "open":
            term_z = _pml_term(term_z, psi_Eyz, bz1, cz1, 2)
            term_x = _pml_term(term_x, psi_Eyx, bx1, cx1, 0)
        Ey[1:, :, 1:] += dt * inv_eps[1:, :, 1:] * (term_z - term_x)

        raw_x = (Hy[1:, :, :] - Hy[:-1, :, :]) * inv_dx
        raw_y = (Hx[:, 1:, :] - Hx[:, :-1, :]) * inv_dx
        term_x = raw_x[:, 1:, :]
        term_y = raw_y[1:, :, :]
        if boundary == "open":
            term_x = _pml_term(term_x, psi_Ezx, bx1, cx1, 0)
            term_y = _pml_term(term_y, psi_Ezy, by1, cy1, 1)
        Ez[1:, 1:, :] += dt * inv_eps[1:, 1:, :] * (term_x - term_y)

        # ---- PEC: the conductor body, then the outer faces ---------------
        if mask is not None:
            Ex[mask] = 0.0
            Ey[mask] = 0.0
            Ez[mask] = 0.0
        if boundary == "metal":
            # TANGENTIAL components only: the normal component of E is not
            # constrained by a PEC wall, and zeroing it would be a different
            # boundary condition than the one the box has.
            Ey[0, :, :] = 0.0
            Ez[0, :, :] = 0.0
            Ey[-1, :, :] = 0.0
            Ez[-1, :, :] = 0.0
            Ex[:, 0, :] = 0.0
            Ez[:, 0, :] = 0.0
            Ex[:, -1, :] = 0.0
            Ez[:, -1, :] = 0.0
            Ex[:, :, 0] = 0.0
            Ey[:, :, 0] = 0.0
            Ex[:, :, -1] = 0.0
            Ey[:, :, -1] = 0.0

        # ---- the source: a soft current density on one component ---------
        # After the PEC pass, so a source on a metal face still drives the run
        # rather than being erased by it.
        amplitude = source.value(t)
        if amplitude != 0.0:
            field[source.component][cell] += (
                dt * inv_eps[cell] * amplitude)

        # ---- probes and energy -------------------------------------------
        for index, (probe_cell, component) in enumerate(probe_list):
            if component == "mag":
                value = math.sqrt(
                    float(Ex[probe_cell]) ** 2 + float(Ey[probe_cell]) ** 2
                    + float(Ez[probe_cell]) ** 2)
            else:
                value = float(field[component][probe_cell])
            traces[index, step] = value

        if energy is not None:
            e2 = Ex * Ex + Ey * Ey + Ez * Ez
            h2 = Hx * Hx + Hy * Hy + Hz * Hz
            energy.append(0.5 * EPS0 * dx ** 3 * float(xp.sum(eps * e2, dtype=xp.float64))
                          + 0.5 * MU0 * dx ** 3 * float(xp.sum(h2, dtype=xp.float64)))

    return {"traces": traces, "dt": dt, "steps": n_steps, "energy": energy}
