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

    # Peak well after the start, so both runs are compared at the same point of
    # the pulse and neither is measured while the source is still driving.
    peak = slice(steps_for(2e-8), steps_for(1.2e-7))
    metal_peak = metal_energy[peak].max()
    open_peak = open_energy[peak].max()
    assert open_peak > 0.5 * metal_peak, (
        "the open run never built up the same pulse, so its residual says "
        "nothing about absorption")

    tail = slice(steps_for(3e-7), None)
    open_residual = open_energy[tail].max() / open_peak
    metal_residual = metal_energy[tail].max() / metal_peak
    assert open_residual < 0.02, f"the CPML left {open_residual:.2%} of the peak"
    assert metal_residual > 0.5, (
        f"the metal box lost its energy too ({metal_residual:.2%} left), so the "
        f"open result is the solver decaying, not the boundary absorbing")


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
    # Still inside the 5-period raised-cosine ramp.
    ramp_value = fdtd.cw_profile(2.5 * period, F0)
    assert abs(ramp_value - math.sin(2 * math.pi * F0 * 2.5 * period)) > 0.9
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
    run = fdtd.simulate(grid, source, n_steps, probes=[(14, 13, 12)], dx=DX,
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
