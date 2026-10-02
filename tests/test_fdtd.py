"""The time-domain solver core: the claims phases 1 and 2 have to keep.

Every test here is a SHAPE, not an absolute amplitude (see fdtd.py on units):
a spectrum peaks where the source put it, a closed box keeps its energy, an
open one does not. They are tiny on purpose -- a 12^3 or 20^3 grid, a few
hundred steps -- because the point is the update arithmetic, and the analytic
bench in scripts/fdtd_bench_v1.py is what checks a real box against a real
mode spectrum.
"""

from __future__ import annotations

import math
import re
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
        key = _device(key)
        if isinstance(key, np.ndarray) and key.dtype == bool:
            # A boolean-mask assignment: what a device backend may have to
            # count on the host. The step loop must not need one.
            raise TypeError("boolean-mask assignment inside the step loop")
        self._array[key] = _device(value)

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

    def __imul__(self, other):
        self._array *= _device(other)
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


def test_an_energy_region_sums_only_its_own_cells():
    """A sub-box the wave has not reached holds exactly zero while the whole
    grid already holds energy; the whole grid spelled as slices is None."""
    eps = np.ones((12, 10, 8), dtype=np.float32)
    source = pulse(cell=(3, 5, 4))
    far = (slice(9, 12), slice(None), slice(None))
    whole = fdtd.simulate(eps, source, 30, record_energy=True)["energy"]
    spelled = fdtd.simulate(eps, source, 30, record_energy=True,
                            energy_region=(slice(None),) * 3)["energy"]
    region = fdtd.simulate(eps, source, 30, record_energy=True,
                           energy_region=far)["energy"]
    assert spelled == whole
    late = max(range(30), key=lambda n: whole[n])
    assert whole[late] > 0 and region[late] < whole[late]
    first = next(n for n in range(30) if whole[n] > 0)
    assert region[first] == 0.0, "energy appeared where no wave can be yet"
    assert max(region) > 0, "the wave never reached the far region"
    for bad in ((slice(None),) * 2, (slice(None), slice(4, 4), slice(None))):
        with pytest.raises(fdtd.FDTDError, match="energy_region"):
            fdtd.simulate(eps, source, 2, record_energy=True,
                          energy_region=bad)


def test_record_peak_is_the_running_maximum_of_every_sample():
    """At a probed sample the peak is the largest |trace| value, not the last
    value; every component comes back full-shape."""
    eps = np.ones((12, 10, 8), dtype=np.float32)
    probes = [(5, 5, 4, "z"), (7, 4, 3, "x"), (4, 6, 5, "y")]
    run = fdtd.simulate(eps, pulse(cell=(6, 5, 4)), steps_for(4.0 / F0),
                        probes=probes, record_peak=True)
    traces = np.abs(np.asarray(run["traces"]))
    for (i, j, k, c), trace in zip(probes, traces):
        assert run["peak"][c].shape == eps.shape
        assert run["peak"][c][i, j, k] == trace.max() > trace[-1], c
    assert fdtd.simulate(eps, pulse(cell=(6, 5, 4)), 2)["peak"] is None


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


def _running_integral_offset(profile, frequency):
    """The mean of a profile's running integral after it has settled,
    as a fraction of that integral's swing: what a soft current leaves behind
    as a static field."""
    dt = (1.0 / frequency) / 400.0
    t = np.arange(0.0, 40.0 / frequency, dt)
    charge = np.cumsum([profile(x) for x in t]) * dt
    tail = charge[-4000:]      # exactly the last 10 periods: whole cycles only
    swing = (tail.max() - tail.min()) / 2.0
    return abs(tail.mean()) / max(swing, abs(tail.mean()), 1e-300)


def test_every_profile_is_zero_mean():
    """A soft current with a net integral leaves a static field that no
    absorbing boundary removes (the spec's zero-mean rule). A ramped sine is
    not enough: its integral settles 1% off zero. The pulse has no swing left
    after it, so its offset is measured against its own peak running value."""
    assert _running_integral_offset(
        lambda t: fdtd.cw_profile(t, F0), F0) < 1e-4
    dt = (1.0 / F0) / 400.0
    t = np.arange(0.0, 40.0 / F0, dt)
    charge = np.cumsum([fdtd.pulse_profile(x, F0, BANDWIDTH) for x in t]) * dt
    assert abs(charge[-1]) < 1e-6 * np.abs(charge).max()


# ---------------------------------------------------------------- plane wave

#: 24^3 with a 4-cell CPML: the box faces may sit on nodes 7..16 (3 cells clear
#: of the CPML), which leaves a 4-node core for an object 3 cells clear of them.
PW_N = 24
PW_PML = 4
PW_BOX = ((7, 7, 7), (16, 16, 16))
PW_INSIDE = 11
#: Graded differently on each axis, so a correction that used the wrong
#: spacing, or a 1-D line spaced unlike the 3-D axis, does not cancel.
PW_SPACINGS = (np.linspace(0.9, 1.1, PW_N - 1), np.full(PW_N - 1, 1.0),
               np.linspace(1.1, 0.9, PW_N - 1))
PW_PAIRS = [(d, c) for d in fdtd.DIRECTIONS for c in fdtd.COMPONENTS
            if c != d[1]]


def plane(direction="+x", component="z", **kwargs):
    options = dict(kind="plane", direction=direction, component=component,
                   box=PW_BOX, frequency=F0, bandwidth=BANDWIDTH)
    options.update(kwargs)
    return fdtd.Source(**options)


def _plane_steps():
    # the pulse's own length plus a crossing of the whole grid
    sigma = 1.0 / (math.pi * BANDWIDTH)
    return steps_for(10.0 * sigma + 1.2 * PW_N * DX / C)


def _outside_probes(component):
    """One probe just beyond each of the box's six faces, in the scattered
    region and clear of the CPML, all at the box's transverse centre."""
    low, high = PW_BOX
    probes = []
    for axis in range(3):
        for index in (low[axis] - 2, high[axis] + 2):
            cell = [PW_INSIDE] * 3
            cell[axis] = index
            probes.append((*cell, component))
    return probes


def _leakage(eps, source):
    inside = (PW_INSIDE,) * 3 + (source.component,)
    run = fdtd.simulate(eps, source, _plane_steps(),
                        probes=[inside, *_outside_probes(source.component)],
                        spacings=PW_SPACINGS, boundary="open", pml_cells=PW_PML)
    traces = np.asarray(run["traces"])
    incident = float(np.abs(traces[0]).max())
    return incident, float(np.abs(traces[1:]).max()) / incident


@pytest.mark.parametrize("direction,component", PW_PAIRS)
def test_a_plane_wave_stays_inside_its_box(direction, component):
    """V3's leakage bound, on a graded grid, for every direction and
    polarisation: outside the box the incident wave cancels to < 1e-3."""
    eps = np.ones((PW_N,) * 3, dtype=np.float32)
    incident, leak = _leakage(eps, plane(direction, component))
    assert incident > 0.5, f"no wave reached the box centre ({incident:.3g})"
    assert leak < 1e-3, (
        f"{direction} {component}: {leak:.2e} of the incident wave leaked "
        f"out of the box")


def test_a_plane_wave_in_a_dielectric_background_stays_inside_its_box():
    """The 1-D line must carry the wave at the background's speed."""
    eps = np.full((PW_N,) * 3, 2.25, dtype=np.float32)
    incident, leak = _leakage(eps, plane("-y", "x"))
    assert incident > 0.3
    assert leak < 1e-3, f"{leak:.2e} leaked out of a dielectric background"


def test_an_object_in_the_box_scatters_out_of_it():
    """The control for the leakage tests: the same probes see a scatterer."""
    eps = np.ones((PW_N,) * 3, dtype=np.float32)
    eps[10:14, 10:14, 10:14] = 4.0
    _incident, leak = _leakage(eps, plane("+x", "z"))
    assert leak > 1e-2, f"a dielectric block scattered only {leak:.2e}"


@pytest.mark.parametrize("direction", ["+x", "-z"])
def test_the_incident_line_lets_its_wave_go_on_a_graded_axis(direction):
    """The 1-D line's feed must not be a wall. A hard source clamps its node
    to the profile, i.e. to zero once the pulse is over, and the small
    reflections off a graded axis then bounce between it and the grading for
    good: measured 2e-4 of the peak line energy still ringing after 1200
    steps at 6 cells per wavelength, against 3e-6 for the one-way feed (whose
    remainder is still draining). The 3-D probes cannot see this at the
    tests' 10+ cells per wavelength, so it is checked on the line itself.
    Both cases feed from the axis's fine (0.9) end; fed from the coarse end
    (-x, +z) a hard source measured clean too, so those cases could not
    fail and are left out."""
    n, cells_per_wavelength = 21, 6.0
    frequency = C / (cells_per_wavelength * DX)
    spacings = [DX * np.linspace(0.9, 1.1, n - 1), np.full(n - 1, DX),
                DX * np.linspace(1.1, 0.9, n - 1)]
    duals = [fdtd._dual_spacing(p) for p in spacings]
    dt = fdtd.courant_dt(*(s.min() for s in spacings))
    eps = np.ones((n,) * 3, dtype=np.float32)
    source = fdtd.Source(kind="plane", direction=direction, component="y",
                         box=((7,) * 3, (13,) * 3), frequency=frequency,
                         bandwidth=frequency / 2.0)
    line = fdtd._PlaneWave(source, eps.shape, spacings, duals, eps, None,
                           (1.0 / (fdtd.EPS0 * eps)).astype(np.float32), dt,
                           4, np)
    energy = []
    for step in range(1200):
        line.advance_h(step)
        line.advance_e(step)
        energy.append(float(np.sum(line.E1.astype(np.float64) ** 2)))
    left = energy[-1] / max(energy)
    assert left < 3e-5, (
        f"{left:.2e} of the line's peak energy is still on it after 1200 "
        f"steps: something on the line reflects")


@pytest.mark.parametrize("direction", ["+z", "-z"])
def test_a_plane_wave_travels_its_way_at_c_with_its_amplitude(direction):
    """The peak passes two points on the axis in the order the direction
    says, at c (a 1.5-cell tolerance on 7 cells: arrival is read from the
    sampled peak), and at the amplitude asked for."""
    eps = np.ones((PW_N,) * 3, dtype=np.float32)
    near, far = 8, 15
    source = plane(direction, "x", amplitude=2.0)
    run = fdtd.simulate(eps, source, _plane_steps(),
                        probes=[(11, 11, near, "x"), (11, 11, far, "x")],
                        spacings=PW_SPACINGS, boundary="open", pml_cells=PW_PML)
    traces = np.asarray(run["traces"])
    first, second = (0, 1) if direction == "+z" else (1, 0)
    delay = (int(np.argmax(np.abs(traces[second])))
             - int(np.argmax(np.abs(traces[first])))) * run["dt"]
    distance = float(np.sum(PW_SPACINGS[2][near:far]))
    assert delay == pytest.approx(distance / C, abs=1.5 * DX / C)
    t = np.arange(4000) * (5.0 / BANDWIDTH) / 4000
    peak = 2.0 * max(abs(fdtd.pulse_profile(x, F0, BANDWIDTH)) for x in t)
    assert float(np.abs(traces).max()) == pytest.approx(peak, rel=0.03)


def test_a_plane_wave_in_a_metal_box_is_refused_naming_the_conflict():
    eps = np.ones((PW_N,) * 3, dtype=np.float32)
    with pytest.raises(fdtd.FDTDError, match="plane wave.*metal.*open"):
        fdtd.simulate(eps, plane(), 10, spacings=PW_SPACINGS,
                      boundary="metal", pml_cells=PW_PML)


def test_a_longitudinal_plane_wave_is_refused():
    with pytest.raises(fdtd.FDTDError, match="transverse"):
        plane("+y", "y")


def test_a_plane_wave_box_must_clear_the_cpml_and_every_object_by_3_cells():
    eps = np.ones((PW_N,) * 3, dtype=np.float32)
    with pytest.raises(fdtd.FDTDError, match="3 cells to the absorbing layer"):
        fdtd.simulate(eps, plane(box=((6, 7, 7), (16, 16, 16))), 2,
                      spacings=PW_SPACINGS, boundary="open", pml_cells=PW_PML)
    with pytest.raises(fdtd.FDTDError, match="3 cells to the absorbing layer"):
        fdtd.simulate(eps, plane(box=((7, 7, 7), (16, 17, 16))), 2,
                      spacings=PW_SPACINGS, boundary="open", pml_cells=PW_PML)
    for near in (9, 18):            # 2 cells inside, then 2 cells outside a face
        blocked = eps.copy()
        blocked[near, 11, 11] = 4.0
        with pytest.raises(fdtd.FDTDError, match="within 3 cells"):
            fdtd.simulate(blocked, plane(), 2, spacings=PW_SPACINGS,
                          boundary="open", pml_cells=PW_PML)
    conductor = np.zeros((PW_N,) * 3, dtype=bool)
    conductor[11, 11, 5] = True
    with pytest.raises(fdtd.FDTDError, match="conductor"):
        fdtd.simulate(eps, plane(), 2, spacings=PW_SPACINGS, boundary="open",
                      pml_cells=PW_PML, pec=conductor)
    # 3 cells clear on both sides is allowed
    clear = eps.copy()
    clear[10, 11, 11] = clear[20, 11, 11] = 4.0
    fdtd.simulate(clear, plane(), 2, spacings=PW_SPACINGS, boundary="open",
                  pml_cells=PW_PML)


def test_a_plane_source_takes_a_direction_and_a_box_not_a_cell():
    with pytest.raises(fdtd.FDTDError, match="no cell"):
        fdtd.Source((1, 1, 1), kind="plane", direction="+x", box=PW_BOX,
                    frequency=F0, bandwidth=BANDWIDTH)
    with pytest.raises(fdtd.FDTDError, match="direction"):
        plane(direction="x")
    with pytest.raises(fdtd.FDTDError, match="not a direction or a box"):
        pulse(direction="+x")
    with pytest.raises(fdtd.FDTDError, match="empty"):
        plane(box=((7, 7, 7), (7, 16, 16)))


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


def test_a_conductor_mask_runs_without_a_boolean_index_and_holds_the_same_field():
    """The masked path, through the device fake, gives the numpy run's numbers.

    The fake refuses a boolean-mask assignment, so a step loop that writes
    E[mask] = 0 raises here; the control proves the fake really refuses one.
    The traces are then compared against the plain numpy run, so a mask that
    stopped being applied at all cannot pass by also not raising.
    """
    xp = _DeviceXp()
    with pytest.raises(TypeError, match="boolean-mask"):
        array = xp.zeros((2, 2), dtype=np.float32)
        array[np.array([[True, False], [False, True]])] = 0.0

    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:5, 3:5, 3:5] = True
    problem = dict(eps_r=np.ones((10, 10, 10), np.float32),
                   source=pulse(cell=(6, 6, 6)), n_steps=60,
                   probes=[(3, 3, 3, "x"), (4, 4, 3, "z"), (7, 6, 5, "z")],
                   dx=DX, boundary="metal", pec=mask)
    device = fdtd.simulate(xp=xp, **problem)
    host = fdtd.simulate(xp=np, **problem)
    unmasked = fdtd.simulate(xp=np, **dict(problem, pec=None))
    assert np.any(host["traces"][2] != 0.0), "the source excited nothing"
    assert not np.any(host["traces"][:2] != 0.0), "the conductor did not hold"
    assert np.any(unmasked["traces"][:2] != 0.0), (
        "the inside probes see nothing even without the conductor, so they "
        "cannot show the mask working")
    assert np.array_equal(device["traces"], host["traces"])


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


# ------------------------------------------------------------------- drawn

#: A slow carrier, so a smooth drawing of it sits far below the 1 m grid's
#: resolvable frequency c / 10 m (30 MHz, against 8 MHz here).
SLOW = F0 / 3.0


def drawn(samples, duration, **kwargs):
    options = dict(cell=(6, 6, 6), frequency=SLOW, profile="drawn",
                   samples=samples, duration=duration)
    options.update(kwargs)
    return fdtd.Source(**options)


def test_a_drawn_waveform_is_resampled_to_dt_with_its_mean_removed():
    """Samples span [0, duration] evenly; the grid reads them at whole steps,
    mean removed, and zero after the duration. A duration that is not a whole
    number of steps and a lopsided drawing, so an off-by-one step count or a
    mean taken before resampling shows."""
    samples = np.array([0.0, 3.0, 1.0, 2.0, 0.5])
    duration = 37.3 * DT
    source = drawn(samples, duration, amplitude=2.0)
    source.bind(DT)
    steps = np.arange(38) * DT                    # 0 .. 37 DT, all < duration
    expected = np.interp(steps, np.linspace(0.0, duration, 5), samples)
    expected -= expected.mean()
    got = np.array([source.value(t) for t in steps])
    np.testing.assert_allclose(got, 2.0 * expected, rtol=1e-12, atol=1e-12)
    assert abs(got.sum()) < 1e-9 * np.abs(got).sum()
    # the plane-wave feed reads between steps: straight lines, to zero at the end
    assert source.value(10.5 * DT) == pytest.approx(0.5 * (got[10] + got[11]))
    assert source.value(37.5 * DT) == pytest.approx(0.5 * got[37])
    assert source.value(38 * DT) == 0.0 and source.value(-DT) == 0.0
    assert source.end_time() == duration


def test_a_drawn_pulse_drives_the_grid_like_the_pulse_itself():
    """Drawing the pulse profile at 10 000 points must give the pulse's own run:
    simulate binds the drawing to the dt it uses, on the same time axis."""
    bandwidth = SLOW / 2.0
    reference = fdtd.Source((6, 6, 6), frequency=SLOW, profile="pulse",
                            bandwidth=bandwidth)
    duration = reference.end_time()
    times = np.linspace(0.0, duration, fdtd.MAX_DRAWN_SAMPLES)
    samples = [fdtd.pulse_profile(t, SLOW, bandwidth) for t in times]
    grid = np.ones((12, 12, 12), dtype=np.float32)
    n_steps = steps_for(1.3 * duration)
    want = fdtd.simulate(grid, reference, n_steps, probes=[(4, 7, 8, "z")],
                         dx=DX)["traces"][0]
    got = fdtd.simulate(grid, drawn(samples, duration), n_steps,
                        probes=[(4, 7, 8, "z")], dx=DX)["traces"][0]
    assert np.abs(want).max() > 0
    assert np.abs(got - want).max() < 1e-3 * np.abs(want).max()


def test_an_under_resolved_drawing_is_refused_naming_f_res():
    """f_res = c / (10 |n|_max Delta_max). The grid is graded to 1.1 m and has
    one eps = 4 node, so a check that used the finest spacing, or vacuum's
    index, names a different frequency. A half-on, half-off drawing three
    resolvable periods long puts well over 1% of its energy above f_res."""
    spacings = (np.linspace(0.9, 1.1, 11), 1.0, 1.0)
    grid = np.ones((12, 12, 12), dtype=np.float32)
    grid[3, 3, 3] = 4.0
    f_res = C / (10 * 2.0 * 1.1)
    square = [1.0] * 50 + [0.0] * 50
    with pytest.raises(fdtd.FDTDError, match=re.escape(f"{f_res:.4g} Hz")):
        fdtd.simulate(grid, drawn(square, 3.0 / f_res), 10,
                      spacings=spacings)
    # the control: a smooth drawing on the same grid runs
    duration = 6.0 / SLOW
    t = np.linspace(0.0, duration, 400)
    smooth = np.sin(2 * np.pi * SLOW * t) * np.sin(np.pi * t / duration) ** 2
    run = fdtd.simulate(grid, drawn(smooth, duration), 200,
                        probes=[(4, 7, 8, "z")], spacings=spacings)
    assert np.abs(run["traces"]).max() > 0


def test_a_drawing_finer_than_dt_is_judged_before_it_is_decimated():
    """+1/-1 every quarter step: read at whole steps it is constant, so judging
    the dt-resampled drawing would see nothing above f_res."""
    samples = np.tile([1.0, -1.0], 200)
    duration = (len(samples) - 1) * DT / 4.0
    f_res = fdtd.resolvable_frequency((DX, DX, DX))
    assert fdtd.drawn_energy_above(samples, duration, DT, f_res) > 0.5


def test_a_drawing_with_nothing_left_after_its_mean_is_refused():
    with pytest.raises(fdtd.FDTDError, match="flat"):
        fdtd.drawn_energy_above([2.0, 2.0, 2.0], 50 * DT, DT, F0)
    source = drawn([0.0, 1.0, 0.0], 1.5 * DT)
    with pytest.raises(fdtd.FDTDError, match="two time steps"):
        source.bind(DT)


def test_a_drawn_waveform_is_at_most_ten_thousand_samples():
    drawn(np.zeros(fdtd.MAX_DRAWN_SAMPLES), 1e-6)
    with pytest.raises(fdtd.FDTDError, match="10000"):
        drawn(np.zeros(fdtd.MAX_DRAWN_SAMPLES + 1), 1e-6)
    with pytest.raises(fdtd.FDTDError, match="duration"):
        drawn([0.0, 1.0], 0.0)
    with pytest.raises(fdtd.FDTDError, match="drawn"):
        fdtd.Source((6, 6, 6), frequency=F0, profile="cw", samples=[0, 1],
                    duration=1e-6)


def test_a_drawn_waveform_is_zero_mean_on_the_steps_the_run_takes():
    """What binding to the run's own dt buys: the soft current's sum over the
    steps the grid really takes is zero, so no static field is left. A
    lopsided (all-positive) drawing on a graded grid, whose dt is not the
    uniform-grid one."""
    duration = 60.0 / SLOW
    t = np.linspace(0.0, duration, 3000)
    bump = np.sin(np.pi * t / duration) ** 2 * (1.5 + np.sin(2 * np.pi * SLOW * t))
    source = drawn(bump, duration)
    run = fdtd.simulate(np.ones((10, 10, 10), dtype=np.float32), source, 5,
                        spacings=(np.linspace(0.8, 1.0, 9), 1.0, 1.0))
    values = np.array([source.value(k * run["dt"])
                       for k in range(int(duration / run["dt"]) + 2)])
    assert abs(values.sum()) < 1e-9 * np.abs(values).sum()


def test_check_hook_is_called_every_check_every_steps_and_can_stop_the_run():
    seen = []
    eps = np.ones((6, 5, 4), dtype=np.float32)
    n = 2 * fdtd.CHECK_EVERY + 2
    fdtd.simulate(eps, pulse(cell=(3, 2, 2)), n, check=seen.append)
    assert seen == [0, fdtd.CHECK_EVERY, 2 * fdtd.CHECK_EVERY]

    class Stop(Exception):
        pass

    def stop(step):
        if step:
            raise Stop(step)
    with pytest.raises(Stop) as info:
        fdtd.simulate(eps, pulse(cell=(3, 2, 2)), n, check=stop)
    assert info.value.args == (fdtd.CHECK_EVERY,)


# ---- the field movie (task 4.3) ---------------------------------------------

def test_the_frame_plan_spreads_its_frames_and_ends_on_the_last_step():
    plan = fdtd.frame_plan((130, 64, 65), 1000)
    # 130 needs a stride of 3 to fit 64 samples (2 would keep 65); 64 fits
    # as it is; 65 needs 2
    assert plan["stride"] == (3, 1, 2)
    assert fdtd.frame_shape((130, 64, 65), plan["stride"]) == (44, 64, 33)
    steps = plan["steps"]
    assert len(steps) == 60 and steps[-1] == 999
    assert all(a < b for a, b in zip(steps, steps[1:])) and steps[0] >= 0
    gaps = np.diff([-1] + steps)
    assert gaps.max() - gaps.min() <= 1          # evenly spread
    # a run shorter than the cap records every step
    assert fdtd.frame_plan((5, 5, 5), 10)["steps"] == list(range(10))
    with pytest.raises(fdtd.FDTDError):
        fdtd.frame_plan((5, 5, 5), 0)


def _movie_run(pec=None):
    shape = (9, 7, 8)                            # non-square on purpose
    eps = np.ones(shape, np.float32)
    eps[5:8, 1:4, 2:6] = 2.5
    probes = [(i, j, k, c) for (i, j, k) in ((4, 2, 3), (6, 5, 6), (2, 4, 0))
              for c in "xyz"]
    plan = {"steps": [5, 17, 39], "stride": (2, 1, 3)}
    run = fdtd.simulate(eps, pulse(cell=(4, 3, 4), component="y"), 40, probes,
                        pec=pec, frames=plan)
    return run, probes, plan


@pytest.mark.parametrize("with_pec", [False, True])
def test_a_frame_is_the_field_magnitude_the_probes_read(with_pec):
    pec = None
    if with_pec:
        pec = np.zeros((9, 7, 8), dtype=bool)
        pec[1:3, 1:3, 5:7] = True
    run, probes, plan = _movie_run(pec)
    frames = run["frames"]
    assert isinstance(frames, np.ndarray) and frames.dtype == np.float32
    assert frames.shape == (3, 5, 7, 3)          # 9/2 -> 5, 7, 8/3 -> 3
    traces = run["traces"]
    for node in range(0, len(probes), 3):
        i, j, k, _ = probes[node]
        for index, step in enumerate(plan["steps"]):
            components = traces[node:node + 3, step].astype(np.float64)
            expected = np.sqrt((components ** 2).sum())
            got = frames[index, i // 2, j, k // 3]
            assert got == pytest.approx(expected, rel=1e-6, abs=1e-30), (
                f"node {(i, j, k)} step {step}: frame {got}, probes {expected}")
    assert frames[-1].max() > 0                  # the pulse is in the picture
    if with_pec:
        # the stride keeps conductor nodes i=2, j=1..2, k=6: no field there,
        # so a frame taken before the conductor mask would show one
        assert not frames[:, 1, 1:3, 2].any()
        assert frames[:, 1, 4:, 2].any()           # but beside it, yes


def test_a_bad_frame_plan_is_refused():
    eps = np.ones((5, 5, 5), np.float32)
    for plan in ({"steps": [3, 3], "stride": (1, 1, 1)},
                 {"steps": [4], "stride": (1, 1, 1)},
                 {"steps": [-1], "stride": (1, 1, 1)},
                 {"steps": [1], "stride": (1, 0, 1)}):
        with pytest.raises(fdtd.FDTDError):
            fdtd.simulate(eps, pulse(cell=(2, 2, 2)), 4, frames=plan)


def test_no_frames_were_asked_for():
    run = fdtd.simulate(np.ones((5, 5, 5), np.float32), pulse(cell=(2, 2, 2)), 3)
    assert run["frames"] is None


def test_the_frames_come_back_on_the_host_whatever_xp_built_them():
    # the same refusing device fake as the traces test above: a frame left
    # device-resident would make the worker's np.asarray raise on the Jetson
    xp = _DeviceXp()
    plan = {"steps": [9, 39], "stride": (2, 1, 3)}
    problem = dict(eps_r=np.ones((7, 6, 8), np.float32),
                   source=pulse(cell=(3, 3, 3)), n_steps=40, frames=plan)
    device = fdtd.simulate(xp=xp, **problem)["frames"]
    host = fdtd.simulate(**problem)["frames"]
    assert type(device) is np.ndarray and device.dtype == np.float32
    assert host.any() and np.array_equal(device, host)
