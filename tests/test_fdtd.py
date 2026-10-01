"""The time-domain solver core: the claims phase 1 has to keep.

Every test here is a SHAPE, not an absolute amplitude (see fdtd.py on units):
a spectrum peaks where the source put it, a closed box keeps its energy, an
open one does not. They are tiny on purpose -- a 12^3 or 20^3 grid, a few
hundred steps -- because the point is the update arithmetic, and the analytic
bench in scripts/fdtd_bench_v1.py is what checks a real box against a real
mode spectrum.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402

DX = 1.0
C = fdtd.C0
DT = fdtd.courant_dt(DX)

#: One wavelength per 12 cells -- the usual dispersion-safe sampling, and small
#: enough that a 12^3 or 20^3 box is a couple of wavelengths across.
LAMBDA_CELLS = 12.0
F0 = C / LAMBDA_CELLS
#: A pulse whose spectrum is half as wide as its carrier: narrow enough that the
#: peak sits at F0, wide enough to be a pulse and not a tone.
BANDWIDTH = F0 / 2.0


def pulse(**kwargs):
    options = dict(cell=(6, 6, 6), frequency=F0, profile="pulse",
                   bandwidth=BANDWIDTH, amplitude=1.0)
    options.update(kwargs)
    return fdtd.Source(**options)


def steps_for(seconds):
    return int(math.ceil(seconds / DT))


def spectrum_peak(trace, dt=DT):
    frequencies, amplitude = fdtd.spectrum(trace, dt)
    index = int(np.argmax(amplitude))
    return float(frequencies[index]), frequencies, amplitude


# ---------------------------------------------------------------- time step

def test_the_time_step_is_below_the_courant_limit():
    limit = fdtd.courant_limit(DX)
    assert DT < limit
    assert DT == pytest.approx(fdtd.SAFETY * limit, rel=1e-12)
    # The limit is c*dt*sqrt(3) = dx: the diagonal of the cell is the worst
    # case, so the assertion above is only meaningful with this factor.
    assert C * limit * math.sqrt(3.0) == pytest.approx(DX, rel=1e-12)


def test_a_courant_safety_factor_at_or_above_one_is_refused():
    for safety in (1.0, 1.05):
        with pytest.raises(fdtd.FDTDError, match="safety factor"):
            fdtd.courant_dt(DX, safety=safety)


def test_the_time_step_uses_the_smallest_spacing_on_every_axis():
    """Three DIFFERENT minima, so an axis mix-up cannot pass.

    Each minimum is planted at a different place in an otherwise uniform array,
    and the three axes carry different values, so taking the limit from the
    first entry, from one axis's array reused three times, or from the wrong
    axis all give a different number from the right one.
    """
    sx = np.array([0.5, 1.0, 1.0, 1.0])
    sy = np.array([1.0, 0.4, 1.0, 1.0])
    sz = np.array([1.0, 1.0, 0.25, 1.0])
    expected = 1.0 / (C * math.sqrt(1 / 0.5 ** 2 + 1 / 0.4 ** 2 + 1 / 0.25 ** 2))
    limit = fdtd.courant_limit(sx, sy, sz)
    assert limit == pytest.approx(expected, rel=1e-12)
    # Each one-axis answer is a different number, so nothing below can be
    # matching by accident.
    alone = [fdtd.courant_limit(a) for a in (0.5, 0.4, 0.25)]
    for value in alone:
        assert limit != pytest.approx(value, rel=1e-6)
    assert fdtd.courant_limit(sx, sx, sx) != pytest.approx(limit, rel=1e-6)
    # A uniform array is the same number as the scalar form, only then equally
    # timed: the harshest axis has to be the one that sets the step.
    assert fdtd.courant_limit(np.full(3, 0.25)) == pytest.approx(
        fdtd.courant_limit(0.25), rel=1e-12)
    assert fdtd.courant_dt(sx, sy, sz) == pytest.approx(
        fdtd.SAFETY * limit, rel=1e-12)
    # Every claim above is about the FUNCTION. A correct function called with
    # one axis's number is still the wrong step, and nothing above would notice,
    # so the run has to be checked too -- a dt off the tightest axis does not
    # blow up on a short run, it just drifts.
    run = fdtd.simulate(np.ones((5, 5, 5), np.float32), pulse(cell=(2, 2, 2)), 2,
                        probes=[(2, 2, 2)], spacings=(sx, sy, sz),
                        boundary="metal")
    assert run["dt"] == pytest.approx(fdtd.SAFETY * expected, rel=1e-12)
    for value in alone:
        assert run["dt"] != pytest.approx(fdtd.SAFETY * value, rel=1e-6)


# ------------------------------------------------------- the return contract


def _device(value):
    return value._array if isinstance(value, _DeviceArray) else value


class _DeviceArray:
    """A host array that refuses to become a numpy array, like cupy's.

    cupy arrays raise on ``np.asarray``, and that refusal is deliberate: it is
    what stops an accidental per-element device transfer from silently costing a
    sync each time. A WRAPPER is needed rather than an ndarray subclass -- numpy
    does not consult ``__array__`` for subclasses, it just reinterprets the
    buffer and hands back a base-class view without raising (measured; the first
    version of this fake was a subclass and the control below caught it).

    ``__array_ufunc__ = None`` makes numpy return NotImplemented for a binary
    operation with one of these on either side, so the reflected methods below
    are used instead of an attempted conversion.
    """

    __array_ufunc__ = None

    def __init__(self, array):
        self._array = array

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Use .get() or xp.asnumpy() to fetch it to the host.")

    def __getattr__(self, name):        # .shape, .reshape, .astype, ...
        # Dunders are NOT delegated. __getattr__ would otherwise hand numpy the
        # underlying array's __array_interface__/__array_struct__, and numpy
        # takes the buffer protocol when it sees one -- it then never reaches
        # __array__ at all, so the refusal below would silently do nothing.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._array, name)

    def __getitem__(self, key):
        value = self._array[_device(key)]
        return _DeviceArray(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        self._array[_device(key)] = _device(value)

    def __sub__(self, other):  return _DeviceArray(self._array - _device(other))
    def __rsub__(self, other): return _DeviceArray(_device(other) - self._array)
    def __mul__(self, other):  return _DeviceArray(self._array * _device(other))
    def __rmul__(self, other): return _DeviceArray(_device(other) * self._array)
    def __truediv__(self, other):
        return _DeviceArray(self._array / _device(other))
    def __rtruediv__(self, other):
        return _DeviceArray(_device(other) / self._array)
    def __gt__(self, other):   return _DeviceArray(self._array > _device(other))

    def __iadd__(self, other):
        self._array += _device(other)
        return self

    def __isub__(self, other):
        self._array -= _device(other)
        return self


class _DeviceXp:
    """The numpy surface ``simulate`` uses, handing back arrays that refuse.

    Only the calls the entry point makes on its default path, so it cannot
    quietly become a second numpy with its own behaviour.
    """

    float32 = np.float32
    float64 = np.float64

    @staticmethod
    def asarray(value, dtype=None):
        return _DeviceArray(np.asarray(_device(value), dtype=dtype))

    @staticmethod
    def zeros(shape, dtype=None):
        return _DeviceArray(np.zeros(shape, dtype=dtype))

    @staticmethod
    def all(value):
        return np.all(_device(value))

    @staticmethod
    def sqrt(value):
        return np.sqrt(_device(value))

    @staticmethod
    def asnumpy(value):
        return _device(value)           # the transfer, the way cupy.asnumpy is


def test_the_traces_come_back_on_the_host_whatever_xp_built_them():
    """The result is host float32 even when the arrays lived on a device.

    Found on the Jetson: with xp=cupy the traces came back device-resident, so
    the bench's ``np.asarray(trace)`` raised "Implicit conversion to a NumPy
    array is not allowed". The control at the top is the load-bearing half --
    if this fake does not actually refuse, the assertions below would pass no
    matter what ``simulate`` returned.
    """
    xp = _DeviceXp()
    with pytest.raises(TypeError, match="Implicit conversion"):
        np.asarray(xp.zeros((2, 2), dtype=np.float32))

    problem = dict(eps_r=np.ones((6, 6, 6), np.float32),
                   source=pulse(cell=(3, 3, 3)), n_steps=40)
    device = fdtd.simulate(probes=[(3, 3, 3)], spacings=(0.9, 1.0, 1.1),
                           xp=xp, **problem)
    assert type(device["traces"]) is np.ndarray, (
        f"the traces came back as {type(device['traces']).__name__}; a device "
        f"array here is what made the bench raise on the Jetson")
    assert device["traces"].dtype == np.float32
    # ...and they are the SAME numbers the numpy path produces. A conversion
    # that dropped, reordered or partially copied samples satisfies the type
    # check above, so the values are what pins the transfer.
    host = fdtd.simulate(probes=[(3, 3, 3)], spacings=(0.9, 1.0, 1.1),
                         xp=np, **problem)
    assert np.any(device["traces"] != 0.0), "the source excited nothing"
    assert np.array_equal(device["traces"], host["traces"])


# ---------------------------------------------------------------- energy

def test_a_closed_pec_box_conserves_its_energy_after_the_source_stops():
    source = pulse()
    grid = np.ones((12, 12, 12), dtype=np.float32)
    run = fdtd.simulate(grid, source, steps_for(6e-7), probes=[(3, 4, 5)],
                        dx=DX, boundary="metal", record_energy=True)

    energy = np.asarray(run["energy"])
    assert energy.max() > 0, "the source excited nothing"
    # The source is silent from here on; a lossless box has nowhere for the
    # energy to go.
    silent_from = steps_for(source.end_time() * 1.1)
    after = energy[silent_from:]
    assert after.size > 20, "the run ended before the source did"
    spread = (after.max() - after.min()) / after.max()
    assert spread < 0.01, f"energy drifted {spread:.2%} after the source stopped"


def test_a_graded_grid_conserves_its_energy_too():
    """The same claim on a grid whose cells are not all the same size.

    x and y are graded, with NEIGHBOUR ratios of 1.04 and 1.06, well inside the
    1.3 the brief allows, so a failure here is the graded arithmetic and not a
    grid no scheme would put up with. Two axes, because a divisor that is wrong
    on one axis only is invisible wherever that axis is uniform: with y uniform,
    an off-by-one dual length on y passed this test (measured); the nanometre
    test below grades z for the same reason. The energy checked is the cell-volume-weighted
    quadratic, which on this grid is a different number from the plain sum of
    squares -- that is the point, because the plain sum is NOT conserved here.
    """
    ratios = 1.04 ** np.arange(15, dtype=np.float64)
    ratios_y = 1.06 ** np.arange(11, dtype=np.float64)
    spacings = (ratios, ratios_y, DX)
    assert float(ratios[-1] / ratios[0]) > 1.5, "the x axis is barely graded"
    assert float(ratios_y[-1] / ratios_y[0]) > 1.5, "the y axis is barely graded"
    grid = np.ones((16, 12, 12), dtype=np.float32)
    source = pulse(cell=(6, 6, 6))
    run = fdtd.simulate(grid, source, steps_for(6e-7), probes=[(3, 4, 5)],
                        spacings=spacings, boundary="metal", record_energy=True)

    energy = np.asarray(run["energy"])
    assert energy.max() > 0, "the source excited nothing"
    silent_from = steps_for(source.end_time() * 1.1)
    after = energy[silent_from:]
    assert after.size > 20, "the run ended before the source did"
    spread = (after.max() - after.min()) / after.max()
    assert spread < 0.01, f"energy drifted {spread:.2%} after the source stopped"


def test_a_graded_axis_gives_the_same_run_whichever_axis_it_is():
    """One graded problem, rotated x -> y -> z, must give the same traces.

    A cyclic relabelling of the axes maps the Yee lattice onto itself (Ex at
    (i+1/2, j, k) lands where an Ey sits) and keeps the curl's handedness, so
    the three runs are the same physics and differ only in float32 rounding.
    That makes every axis its own control: a divisor that is wrong on y or z
    alone moves that run's traces and nothing else. The energy tests above do
    not see all of those -- a wrong length still conserves a slightly
    different quadratic, and measured, an off-by-one dual length on y and a
    reversed primal length on z both stayed under their 1% drift.

    The fixture is deliberately lopsided: steep grading on one axis only, and
    source and probe off every diagonal, so no rotation can map the run onto
    itself by symmetry.
    """
    graded = 1.25 ** np.arange(11, dtype=np.float64) / 3.0
    shape = (12, 12, 12)
    cell, component = (5, 6, 4), "z"
    probes = [((3, 8, 7), "x"), ((7, 4, 6), "y"), ((8, 7, 3), "z")]

    def rotate(triple, times):
        values = tuple(triple)
        for _ in range(times):
            values = (values[2], values[0], values[1])   # axis a -> a+1
        return values

    def axis_name(name, times):
        return "xyz"[("xyz".index(name) + times) % 3]

    runs = []
    for times in range(3):
        spacings = rotate((graded, DX, DX), times)
        source = pulse(cell=rotate(cell, times),
                       component=axis_name(component, times))
        rotated = [rotate(c, times) + (axis_name(n, times),) for c, n in probes]
        run = fdtd.simulate(np.ones(shape, np.float32), source, 160, rotated,
                            spacings=spacings, boundary="metal")
        runs.append(np.asarray(run["traces"], dtype=np.float64))

    scale = float(np.abs(runs[0]).max())
    assert scale > 0.0, "the source excited nothing"
    assert all(float(np.abs(r).max()) > 0.1 * scale for r in runs[0]), (
        "a probe saw almost nothing, so it cannot tell the rotations apart")
    for times in (1, 2):
        gap = float(np.abs(runs[times] - runs[0]).max()) / scale
        assert gap < 1e-4, (
            f"grading {'xyz'[times]} instead of x moved the traces by "
            f"{gap:.2e} of their peak; the same problem on another axis must "
            f"give the same run")


def test_a_nanometre_grid_conserves_its_energy_too():
    """The 10 nm end of the range the product actually covers.

    float32 coefficients are not scale free and this is where they break: the
    cell volumes here are ~1e-26 m^3, so a squared sample times its volume is
    ~1e-58, below float32's smallest normal (1.2e-38) and the sum comes out 0.
    The run is real either way -- it is the REDUCTION that has to be float64 --
    and the control below proves this grid really is under that floor instead of
    assuming it, so a green result cannot come from a reduction that ignores the
    scale.
    """
    # x and z graded (z for the reason the test above grades y), y uniform.
    spacings = (2.5e-9 * 1.1 ** np.arange(12, dtype=np.float64), 1e-8,
                2e-8 * 1.07 ** np.arange(11, dtype=np.float64))
    # np.min, not s.min(): an axis may be given as a bare scalar.
    axis_min = [float(np.min(s)) for s in spacings]
    f_nm = C / (12.0 * min(axis_min))
    grid = np.ones((13, 12, 12), dtype=np.float32)
    source = fdtd.Source((6, 6, 6), frequency=f_nm, profile="pulse",
                         bandwidth=f_nm / 2.0)
    dt = fdtd.courant_dt(*axis_min)
    n_steps = int(math.ceil(1.1 * source.end_time() / dt)) + 80
    run = fdtd.simulate(grid, source, n_steps, probes=[(3, 4, 5)],
                        spacings=spacings, boundary="metal", record_energy=True)
    assert run["dt"] == dt

    sample = float(np.abs(run["traces"]).max())
    smallest_volume = float(np.prod(axis_min))
    assert np.float32(sample) ** 2 * np.float32(smallest_volume) == 0.0, (
        f"a sample of {sample:.3g} in a cell of {smallest_volume:.3g} does not "
        f"underflow float32, so this run is not testing the scale it claims to")

    energy = np.asarray(run["energy"])
    assert energy.max() > 0, "the source excited nothing at this scale"
    silent_from = int(math.ceil(1.1 * source.end_time() / dt))
    after = energy[silent_from:]
    assert after.size > 20, "the run ended before the source did"
    spread = (after.max() - after.min()) / after.max()
    assert spread < 0.01, f"energy drifted {spread:.2%} after the source stopped"


def test_an_open_boundary_absorbs_the_pulse_while_the_metal_wall_holds_it():
    """The open run's own control: same source, same grid, only the wall differs.

    Without the control, "the energy went away" is also what a solver that
    quietly decays everything would report.
    """
    grid = np.ones((20, 20, 20), dtype=np.float32)
    source = pulse(cell=(6, 6, 6))
    n_steps = steps_for(6e-7)

    metal = fdtd.simulate(grid, source, n_steps, probes=[(14, 13, 12)],
                          dx=DX, boundary="metal", record_energy=True)
    opened = fdtd.simulate(grid, source, n_steps, probes=[(14, 13, 12)],
                           dx=DX, boundary="open", pml_cells=5, record_energy=True)

    metal_energy = np.asarray(metal["energy"])
    open_energy = np.asarray(opened["energy"])

    # While the source is still driving, both boxes are filling up at the same
    # rate. Unless they are, the later comparison is between two different
    # experiments.
    during = slice(steps_for(2e-8), steps_for(1.2e-7))
    metal_peak = metal_energy[during].max()
    open_peak = open_energy[during].max()
    assert open_peak > 0.5 * metal_peak, (
        "the open run never built up the same pulse, so its residual says "
        "nothing about absorption")
    # The pulse runs to source.end_time(), so "after the source" has to be
    # derived from it: a window picked by eye sits inside the drive and the
    # metal box then looks like it lost half its energy between two moments of
    # the source, not of the box.
    after = steps_for(1.1 * source.end_time())
    assert after > during.stop, "the source is still driving this window"

    # One instant, two boxes. Everything the metal box has it still has, because
    # nothing left the room; at the SAME step the open box is down by five
    # orders of magnitude. Comparing each run against its own peak instead
    # would let both boxes decay and still agree.
    #
    # The MINIMUM over the window, not the maximum: the window starts at
    # metal_energy[after], so a max over it is never below 1 and a box that
    # leaked everything would still read as held.
    held = metal_energy[after:].min() / metal_energy[after]
    assert held > 0.9, f"the metal box kept only {held:.2%} of its energy"
    # Measured: 2.5e-5 here. With the CPML correction switched off on the
    # dual (H) family it is 1.3e-2, on the primal (E) family 2.0e-3, and with
    # no CPML at all 1.0, so 1e-4 fails each of those and passes the real one.
    left = open_energy[after:].max() / metal_energy[after:].max()
    assert left < 1e-4, (
        f"the CPML left {left:.3e} of what the metal wall held at the same "
        f"step, so the difference is the solver, not the boundary")


# ---------------------------------------------------------------- sources

def test_the_pulse_spectrum_peaks_at_the_requested_frequency():
    """Sampled off-grid, on a long window, so the bins are not the limit.

    The pulse is a modulated Gaussian: its two spectral lobes sit at +/-f, and
    what the run measures is the positive one. A bandwidth of f/4 keeps them
    far enough apart that the peak is the carrier and not the overlap.
    """
    source = fdtd.Source((0, 0, 0), frequency=F0, profile="pulse",
                         bandwidth=F0 / 4.0)
    sample_dt = (1.0 / F0) / 40.0
    n = 8000
    profile = np.array([source.value(k * sample_dt) for k in range(n)])
    frequencies, amplitude = fdtd.spectrum(profile, sample_dt)
    peak = float(frequencies[int(np.argmax(amplitude))])
    assert peak == pytest.approx(F0, rel=0.01), (
        f"the pulse peaked at {peak:.4g} Hz, not {F0:.4g}")


def test_a_cw_tone_is_quiet_before_it_starts_and_a_tone_after_the_ramp():
    assert fdtd.cw_profile(0.0, F0) == 0.0
    assert fdtd.cw_profile(-1.0, F0) == 0.0
    period = 1.0 / F0
    # A quarter period in: the carrier is at its peak, so the envelope is plain
    # to see. Half a period in would be a carrier zero crossing, where the
    # value is near zero whether or not any envelope is applied.
    ramp_value = fdtd.cw_profile(0.25 * period, F0)
    assert abs(ramp_value - math.sin(2 * math.pi * F0 * 0.25 * period)) > 0.9
    # After it: a pure tone, no envelope left.
    for k in (0.0, 0.25, 0.5, 0.75):
        t = 6.0 * period + k * period
        assert fdtd.cw_profile(t, F0) == pytest.approx(
            math.sin(2 * math.pi * F0 * t), rel=1e-12)


def test_a_cw_source_shows_up_at_its_own_frequency_in_a_probe():
    """Open boundary, so the only thing ringing is the source.

    A closed box would be the wrong instrument: the turn-on excites the box's
    own modes, and the tallest peak in the probe's spectrum could be a
    resonance of the box rather than the tone that was asked for.
    """
    grid = np.ones((20, 20, 20), dtype=np.float32)
    source = fdtd.Source((10, 10, 10), frequency=F0, profile="cw", component="z")
    # 40 periods of tone, sampled into a 30-period window, so the bins are
    # finer than the tolerance below.
    n_steps = steps_for(40.0 / F0)
    # A COMPONENT, not |E|. The magnitude of a tone is a rectified tone: its
    # spectrum peaks at twice the frequency, not at the frequency, so an |E|
    # probe would answer "2*f0" to a question about f0 and the run would look
    # like a solver ringing at the wrong rate.
    run = fdtd.simulate(grid, source, n_steps, probes=[(14, 13, 12, "z")], dx=DX,
                        boundary="open", pml_cells=5)
    trace = np.asarray(run["traces"][0])
    tail = trace[steps_for(10.0 / F0):]
    peak, _frequencies, _amplitude = spectrum_peak(tail)
    assert peak == pytest.approx(F0, rel=0.03), (
        f"the driven run rang at {peak:.4g} Hz, not {F0:.4g}")


def test_a_plane_source_is_refused_by_name():
    with pytest.raises(fdtd.FDTDError, match="phase 2"):
        fdtd.Source((1, 1, 1), frequency=F0, kind="plane", bandwidth=BANDWIDTH)


def test_a_cw_source_with_a_bandwidth_is_refused():
    with pytest.raises(fdtd.FDTDError, match="bandwidth"):
        fdtd.Source((1, 1, 1), frequency=F0, profile="cw", bandwidth=F0)


# ---------------------------------------------------------------- PEC

def test_a_conductor_mask_holds_the_field_at_zero_inside_it():
    """An inside probe AND an outside one: a mask that zeroed everything would
    pass the inside assertion on its own."""
    grid = np.ones((12, 12, 12), dtype=np.float32)
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    source = pulse(cell=(6, 6, 6), component="z")
    run = fdtd.simulate(grid, source, probes=[(3, 3, 3, "mag"), (8, 8, 8, "mag")],
                        n_steps=steps_for(2e-8), dx=DX, boundary="metal", pec=mask)
    inside = np.asarray(run["traces"][0])
    outside = np.asarray(run["traces"][1])
    assert np.any(np.abs(outside) > 0), "no field anywhere, so the mask says nothing"
    assert not np.any(inside != 0.0), "the conductor is not holding E at zero"


def test_a_source_inside_the_conductor_is_refused():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[6, 6, 6] = True
    with pytest.raises(fdtd.FDTDError, match="inside the conductor"):
        fdtd.simulate(np.ones((12, 12, 12), dtype=np.float32), pulse(),
                      n_steps=10, dx=DX, pec=mask)


# ---------------------------------------------------------------- refusals

def test_a_probe_outside_the_grid_is_refused():
    with pytest.raises(fdtd.FDTDError, match="outside the grid"):
        fdtd.simulate(np.ones((12, 12, 12), dtype=np.float32), pulse(),
                      probes=[(20, 1, 1)], n_steps=10, dx=DX)


def test_a_cpml_that_does_not_fit_the_grid_is_refused():
    with pytest.raises(fdtd.FDTDError, match="CPML"):
        fdtd.simulate(np.ones((12, 12, 12), dtype=np.float32), pulse(),
                      n_steps=10, dx=DX, boundary="open", pml_cells=6)
