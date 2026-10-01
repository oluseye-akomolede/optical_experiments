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

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

C0 = 299792458.0                     # m/s
EPS0 = 8.8541878128e-12              # F/m
MU0 = 1.0 / (EPS0 * C0 * C0)         # H/m
ETA0 = math.sqrt(MU0 / EPS0)         # ohm

BOUNDARIES = ("metal", "open")
PROFILES = ("cw", "pulse")
COMPONENTS = ("x", "y", "z")

SAFETY = 0.99                        # fraction of the Courant limit
PML_ORDER = 3
PML_REFLECTION = 1e-6                # target reflection at the CPML face
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
    """
    if t <= 0.0:
        return 0.0
    ramp = ramp_periods / frequency
    if t >= ramp:
        envelope = 1.0
    else:
        envelope = 0.5 * (1.0 - math.cos(math.pi * t / ramp))
    return envelope * math.sin(2.0 * math.pi * frequency * t)


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


class Source:
    """A soft current on one E component at one cell.

    Soft means it adds to the field there instead of pinning it, so the cell
    still carries the wave it launched. Phase 1 has point sources only; a plane
    wave needs a TF/SF box and is refused with the phase that adds it.
    """

    def __init__(self, cell, *, frequency: float, profile: str = "pulse",
                 bandwidth: float | None = None, amplitude: float = 1.0,
                 component: str = "z", kind: str = "point"):
        if kind != "point":
            raise FDTDError(
                f"source kind {kind!r} is not implemented; phase 1 has a point "
                f"soft current only, and a plane wave needs the TF/SF box of "
                f"phase 2")
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
                f"a cw source has no bandwidth, got {bandwidth!r}; the tone is "
                f"at the frequency alone")
        if len(tuple(cell)) != 3:
            raise FDTDError(f"a cell is three indices, got {cell!r}")
        self.cell = tuple(int(c) for c in cell)
        self.frequency = float(frequency)
        self.profile = profile
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.amplitude = float(amplitude)
        self.component = component
        self.kind = kind

    def value(self, t: float) -> float:
        if self.profile == "cw":
            return self.amplitude * cw_profile(t, self.frequency)
        return self.amplitude * pulse_profile(t, self.frequency, self.bandwidth)

    def end_time(self) -> float:
        """When a pulse is done, or ``inf`` for a source that never stops."""
        if self.profile == "cw":
            return math.inf
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

def _slab_sigma(widths: np.ndarray, from_low: bool) -> np.ndarray:
    """Cubic conductivity ramp through one slab, from its absorbing face.

    ``widths`` are the LOCAL cell widths of the nodes in the slab, so a graded
    grid gets its ramp measured in the cells it actually has. ``from_low`` says
    which end is the outer face.
    """
    ordered = widths if from_low else widths[::-1]
    # distance from the outer face to each node's centre
    distance = np.cumsum(ordered) - 0.5 * ordered
    total = float(widths.sum())
    sigma_max = ((PML_ORDER + 1) * math.log(1.0 / PML_REFLECTION)
                 / (2.0 * ETA0 * total))
    ramp = sigma_max * ((total - distance) / total) ** PML_ORDER
    return ramp if from_low else ramp[::-1]


def _pml_profiles(width: np.ndarray, pml: int, dt: float, xp):
    """``(b, c)`` per node along one axis, from a conductivity ramp.

    ``width`` is the cell width at each node of the family being corrected
    (``_node_thickness`` for E, the primal spacing for H), so the ramp and its
    total thickness follow the grid. ``sigma_max`` comes from the slab's own
    thickness, and the ramp reaches it at the outer face and falls to zero at the
    inner interface -- graded that way round, the layer absorbs more the deeper
    the wave gets, and the return trip is suppressed twice over.
    """
    n = width.size
    if pml <= 0:
        b = np.ones(n)
        return xp.asarray(b, dtype=xp.float32), xp.asarray(b - 1.0, dtype=xp.float32)
    if n <= 2 * pml:
        raise FDTDError(
            f"a CPML of {pml} cells does not fit in {n} nodes on an axis: the "
            f"interior would be empty (it needs more than {2 * pml} nodes). "
            f"Give more cells, or a smaller pml_cells")
    sigma = np.zeros(n, dtype=np.float64)
    sigma[:pml] = _slab_sigma(width[:pml], from_low=True)
    sigma[n - pml:] = _slab_sigma(width[n - pml:], from_low=False)
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

def simulate(eps_r, source: Source, n_steps: int, probes=(),
             *, spacings=None, dx: float = 1.0,
             boundary: str = "metal", pec=None, pml_cells: int = DEFAULT_PML_CELLS,
             xp=None, record_energy: bool = False, ade_terms=()) -> dict:
    """Run the grid and return the probe traces.

    ``eps_r`` ``(nx, ny, nz)`` is the relative permittivity of each node and it
    DEFINES the grid; ``pec`` is an optional boolean conductor mask of the same
    shape. ``spacings`` is a ``(sx, sy, sz)`` triple, each a scalar (uniform
    along that axis) or ``n-1`` primal lengths; ``dx`` is the shorthand for a
    uniform grid.

    ``boundary="metal"`` closes the outer faces with a PEC wall;
    ``boundary="open"`` puts a CPML of ``pml_cells`` on each face instead (the
    open box is still PEC-backed at the outermost layer, which is how a PML is
    terminated in practice). ``ade_terms`` are polarisation-current terms (see
    ``PolarisationTerm``); phase 1 runs with none.

    Returns ``{"traces": (n_probes, n_steps) float32, "dt", "steps",
    "spacings", "energy"}``, and every one of them is HOST numpy whatever ``xp``
    was: the arrays live on the device for the whole run and cross back in a
    single transfer at the end, so a consumer never has to know which backend
    ran (and never pays a per-element transfer of its own). ``energy`` is the
    cell-volume-weighted total per step, or ``None`` unless ``record_energy``
    asked for it -- it builds six float64 weight arrays and reduces six arrays
    every step, and each ``float(xp.sum(...))`` is a device sync, so it is a
    diagnostic: free on the numpy path, six syncs a step on cupy. Cheap on a
    24^3 test grid, not on a production one.
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

    if boundary not in BOUNDARIES:
        raise FDTDError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
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
    if pec is not None:
        mask = xp.asarray(pec, dtype=bool)
        if tuple(mask.shape) != shape:
            raise FDTDError(
                f"the conductor mask is {tuple(mask.shape)} but the grid is "
                f"{shape}; they must be the same shape")

    terms = list(ade_terms)
    for term in terms:
        if not callable(getattr(term, "advance", None)):
            raise FDTDError(
                f"{term!r} is not a polarisation term: it has no advance()")

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
            elif index in (0, shape[axis] - 1):
                raise FDTDError(
                    f"the cell {cell} is in the outer layer of axis {axis}, "
                    f"where a {component}-polarised source is on the wall and "
                    f"the update never touches it; move it inward")

    live(source.component, source.cell)
    if mask is not None and bool(mask[source.cell]):
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

    P = [xp.asarray(p, dtype=f32) for p in primal]      # n-1 primal lengths
    D = [xp.asarray(d, dtype=f32) for d in dual]        # n-1 dual lengths

    # CPML: one profile per (axis, family). The E update is evaluated at primal
    # nodes, so its ramp is measured in the cells around primal nodes; the H
    # update sits at dual nodes, whose cells are the primal lengths.
    bc: dict = {}
    if boundary == "open":
        for axis in range(3):
            bc[(axis, "primal")] = _pml_profiles(
                _node_thickness(primal[axis]), int(pml_cells), dt, xp)
            bc[(axis, "dual")] = _pml_profiles(
                primal[axis], int(pml_cells), dt, xp)

    psi: dict = {}
    coef: dict = {}

    def corrected(raw, key: tuple, axis: int, family: str):
        """The CPML-corrected derivative, or ``raw`` on a metal boundary."""
        if boundary != "open":
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
    dt_over_mu0 = dt / MU0

    if energy is not None:
        # Cell volumes around each sample. These are what make the quadratic
        # conserved on a graded grid; on a uniform grid they are all equal and
        # the sum reduces to the usual one.
        node = [_node_thickness(p) for p in primal]
        wide = [_dual_thickness(p) for p in primal]

        def volumes(dual_axes):
            axes = [wide[a] if a in dual_axes else node[a] for a in range(3)]
            return xp.asarray(_outer(axes), dtype=xp.float64)

        # Each component is dual along its own axis if it is E, and along the
        # other two if it is H.
        e_weights = {c: volumes({a}) for a, c in enumerate(COMPONENTS)}
        h_weights = {c: volumes({(a + 1) % 3, (a + 2) % 3})
                     for a, c in enumerate(COMPONENTS)}
        eps64 = xp.asarray(eps, dtype=xp.float64)

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
                wide_values = values.astype(xp.float64)
                e_total += float(xp.sum(
                    eps64 * e_weights[component] * wide_values * wide_values))
            h_total = 0.0
            for index, (component, values) in enumerate(
                    (("x", Hx), ("y", Hy), ("z", Hz))):
                now = values.astype(xp.float64)
                before = h_before[index].astype(xp.float64)
                h_total += float(xp.sum(h_weights[component] * now * before))
            energy.append(0.5 * (EPS0 * e_total + MU0 * h_total))

        # ---- E from the curl of H (H at n+1/2, E advances to n+1) --------
        # Ex at (i+1/2, j, k): live for j in [1, ny-2], k in [1, nz-2]
        term_y = (Hz[:-1, 1:-1, 1:-1] - Hz[:-1, :-2, 1:-1]) / _along(D[1][1:], 1)
        term_z = (Hy[:-1, 1:-1, 1:-1] - Hy[:-1, 1:-1, :-2]) / _along(D[2][1:], 2)
        term_y = corrected(term_y, ("Ex", "y"), 1, "primal")
        term_z = corrected(term_z, ("Ex", "z"), 2, "primal")
        Ex[:-1, 1:-1, 1:-1] += dt * inv_eps[:-1, 1:-1, 1:-1] * (term_y - term_z)

        # Ey at (i, j+1/2, k): live for i in [1, nx-2], k in [1, nz-2]
        term_z = (Hx[1:-1, :-1, 1:-1] - Hx[1:-1, :-1, :-2]) / _along(D[2][1:], 2)
        term_x = (Hz[1:-1, :-1, 1:-1] - Hz[:-2, :-1, 1:-1]) / _along(D[0][1:], 0)
        term_z = corrected(term_z, ("Ey", "z"), 2, "primal")
        term_x = corrected(term_x, ("Ey", "x"), 0, "primal")
        Ey[1:-1, :-1, 1:-1] += dt * inv_eps[1:-1, :-1, 1:-1] * (term_z - term_x)

        # Ez at (i, j, k+1/2): live for i in [1, nx-2], j in [1, ny-2]
        term_x = (Hy[1:-1, 1:-1, :-1] - Hy[:-2, 1:-1, :-1]) / _along(D[0][1:], 0)
        term_y = (Hx[1:-1, 1:-1, :-1] - Hx[1:-1, :-2, :-1]) / _along(D[1][1:], 1)
        term_x = corrected(term_x, ("Ez", "x"), 0, "primal")
        term_y = corrected(term_y, ("Ez", "y"), 1, "primal")
        Ez[1:-1, 1:-1, :-1] += dt * inv_eps[1:-1, 1:-1, :-1] * (term_x - term_y)

        # ---- dispersive materials (phase 2 adds terms; phase 1 has none) --
        for term in terms:
            for name, increment in term.advance(field, dt, step).items():
                if name not in COMPONENTS:
                    raise FDTDError(
                        f"the polarisation term {getattr(term, 'name', term)!r} "
                        f"returned an increment for {name!r}; E components are "
                        f"{COMPONENTS}")
                field[name] += increment

        # ---- source ------------------------------------------------------
        field[source.component][source.cell] += dt * source.value(t)

        # ---- PEC ---------------------------------------------------------
        if mask is not None:
            Ex[mask] = 0.0
            Ey[mask] = 0.0
            Ez[mask] = 0.0
        if boundary == "metal":
            # The six faces, TANGENTIAL E only: two components per face, and in
            # this layout those are exactly the layers the update above never
            # touches. The normal component of E is not constrained by a PEC
            # wall, and pinning it would be a different boundary condition.
            Ey[0, :, :] = 0.0       # x faces: Ey, Ez
            Ez[0, :, :] = 0.0
            Ey[-1, :, :] = 0.0
            Ez[-1, :, :] = 0.0
            Ex[:, 0, :] = 0.0       # y faces: Ex, Ez
            Ez[:, 0, :] = 0.0
            Ex[:, -1, :] = 0.0
            Ez[:, -1, :] = 0.0
            Ex[:, :, 0] = 0.0       # z faces: Ex, Ey
            Ey[:, :, 0] = 0.0
            Ex[:, :, -1] = 0.0
            Ey[:, :, -1] = 0.0

        # ---- probes and energy -------------------------------------------
        for index, (cell, component) in enumerate(probe_list):
            if component == "mag":
                value = xp.sqrt(Ex[cell] ** 2 + Ey[cell] ** 2 + Ez[cell] ** 2)
            else:
                value = field[component][cell]
            traces[index, step] = value

    return {"traces": _host_result(traces, xp), "dt": dt, "steps": n_steps,
            "spacings": [np.array(p) for p in primal], "energy": energy}


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
