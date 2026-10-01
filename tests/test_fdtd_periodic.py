"""Periodic (k = 0) faces: the wrap, the seam, and a plane wave across them.

A uniform plane wave alone cannot test the wrap: every transverse slab holds
the same numbers, so a wrap from the wrong slab reads back the same values.
The wrap is tested by a field that varies across the periodic axis -- a point
source shifted across the seam must give exactly the rolled field -- and by
energy conservation in a closed, fully periodic, graded box, which a wrong
wrap width breaks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402

DX = 0.01


def _real(values, component, periodic):
    """Drop the ghost slot along the component's own (half-position) axis."""
    where = [slice(None)] * 3
    axis = fdtd.COMPONENTS.index(component)
    if periodic[axis]:
        where[axis] = slice(0, values.shape[axis] - 1)
    return values[tuple(where)]


def _plane_setup(direction, n=60, t=5):
    axis = fdtd.COMPONENTS.index(direction[1])
    periodic = tuple(a != axis for a in range(3))
    shape = tuple(t if periodic[a] else n for a in range(3))
    # graded along travel, and the two periodic axes unequal
    spacings = tuple(np.full(t - 1, 1.5 * DX * (1 + 0.1 * a)) if periodic[a]
                     else np.linspace(0.9, 1.1, n - 1) * 1.5 * DX
                     for a in range(3))
    box = (tuple(0 if periodic[a] else 14 for a in range(3)),
           tuple(t - 1 if periodic[a] else n - 15 for a in range(3)))
    boundary = tuple("periodic" if p else "open" for p in periodic)
    return shape, spacings, box, boundary, periodic


@pytest.mark.parametrize("direction,component",
                         [("+x", "y"), ("-y", "z"), ("+z", "x"), ("-z", "y")])
def test_a_plane_wave_across_periodic_faces_is_uniform_and_stays_in_its_box(
        direction, component):
    shape, spacings, box, boundary, periodic = _plane_setup(direction)
    source = fdtd.Source(kind="plane", direction=direction, component=component,
                         box=box, frequency=1e9, bandwidth=5e8)
    run = fdtd.simulate(np.ones(shape, np.float32), source, 1200,
                        spacings=spacings, boundary=boundary, record_peak=True)
    peak = run["peak"]
    incident = float(_real(peak[component], component, periodic).max())
    assert incident > 0.5, "the wave never arrived"
    axis = fdtd.COMPONENTS.index(direction[1])
    for c in fdtd.COMPONENTS:
        # the ghost slot along E_c's own periodic axis is not a sample, and a
        # reported peak there would be a fake field value
        own = fdtd.COMPONENTS.index(c)
        if periodic[own]:
            ghost = float(np.take(peak[c], -1, axis=own).max())
            assert ghost == 0.0, f"E{c}'s ghost slot holds {ghost:.3g}"
        along = np.moveaxis(_real(peak[c], c, periodic), axis, 0)
        spread = max(float(np.ptp(slab)) for slab in along)
        assert spread < 1e-6 * incident, (
            f"E{c} varies across the periodic faces by {spread / incident:.2e} "
            f"of the incident peak")
    # nothing outside the two injecting faces, in the physical region
    travel = np.moveaxis(peak[component], axis, 0)
    low, high = box[0][axis], box[1][axis]
    outside = max(float(travel[fdtd.DEFAULT_PML_CELLS:low - 1].max()),
                  float(travel[high + 2:shape[axis] - fdtd.DEFAULT_PML_CELLS].max()))
    assert outside < 1e-3 * incident


def _shifted_peaks(first, second):
    shape = (8, 7, 26)
    spacings = (np.full(7, DX), np.full(6, 1.3 * DX),
                np.linspace(0.9, 1.1, 25) * DX)

    def peaks(cell):
        source = fdtd.Source(cell, frequency=3e9, bandwidth=3e9, component="x")
        return fdtd.simulate(np.ones(shape, np.float32), source, 300,
                             spacings=spacings,
                             boundary=("periodic", "periodic", "open"),
                             record_peak=True)["peak"]

    return peaks(first), peaks(second)


def test_a_source_moved_across_the_seam_gives_the_rolled_field():
    # (6, 5) -> (2, 2) is a shift of (-4, -3) on periods (7, 6): the field has
    # to cross the seam on both axes to get there.
    a, b = _shifted_peaks((6, 5, 13), (2, 2, 13))
    scale = float(a["x"].max())
    assert scale > 0
    for c in fdtd.COMPONENTS:
        one_period_a = a[c][:7, :6]
        one_period_b = b[c][:7, :6]
        # positive control: the field really does vary across the period
        assert float(np.ptp(one_period_a[:, :, 13])) > 1e-2 * scale or c == "z"
        mismatch = float(np.abs(np.roll(one_period_a, (-4, -3), (0, 1))
                                - one_period_b).max())
        assert mismatch < 1e-6 * scale, (
            f"E{c} is not the rolled field: {mismatch / scale:.2e}")


def test_a_closed_periodic_graded_box_conserves_energy():
    shape = (9, 11, 13)
    spacings = tuple(np.linspace(0.8, 1.2, n - 1) * DX * (1 + 0.05 * a)
                     for a, n in enumerate(shape))
    source = fdtd.Source((3, 4, 5), frequency=3e9, bandwidth=3e9, component="y")
    dt = fdtd.courant_dt(*(s.min() for s in spacings))
    after = int(source.end_time() / dt) + 5
    run = fdtd.simulate(np.ones(shape, np.float32), source, after + 600,
                        spacings=spacings, boundary="periodic",
                        record_energy=True)
    tail = np.asarray(run["energy"][after:])
    assert tail.mean() > 0
    drift = (tail.max() - tail.min()) / tail.mean()
    assert drift < 1e-5, f"energy drifts {drift:.2e} after the source ends"


def _plane(box, direction="+z", component="x"):
    return fdtd.Source(kind="plane", direction=direction, component=component,
                       box=box, frequency=1e9, bandwidth=5e8)


SHAPE = (5, 5, 40)
SPAN = ((0, 0, 14), (4, 4, 25))


@pytest.mark.parametrize("boundary,words", [
    (("periodic", "periodic", "metal"), "metal"),
    (("open", "open", "periodic"), "open boundary along z"),
    (("periodic", "open", "open"), "both periodic"),
])
def test_a_plane_wave_refuses_boundaries_it_cannot_run_with(boundary, words):
    with pytest.raises(fdtd.FDTDError, match=words):
        fdtd.simulate(np.ones(SHAPE, np.float32), _plane(SPAN), 2, dx=DX,
                      boundary=boundary)


def test_a_box_that_does_not_span_a_periodic_axis_is_refused():
    with pytest.raises(fdtd.FDTDError, match="must span the periodic axis 0"):
        fdtd.simulate(np.ones(SHAPE, np.float32),
                      _plane(((1, 0, 14), (4, 4, 25))), 2, dx=DX,
                      boundary=("periodic", "periodic", "open"))


def test_material_that_differs_across_the_seam_is_refused():
    eps = np.ones(SHAPE, np.float32)
    eps[0, :, 20] = 2.0                 # node 0 only, not node n-1
    with pytest.raises(fdtd.FDTDError, match="periodic axis 0"):
        fdtd.simulate(eps, _plane(SPAN), 2, dx=DX,
                      boundary=("periodic", "periodic", "open"))
    eps[-1, :, 20] = 2.0                # now the same point on both sides
    fdtd.simulate(eps, _plane(SPAN), 2, dx=DX,
                  boundary=("periodic", "periodic", "open"))


def test_a_source_on_the_copied_node_is_refused():
    with pytest.raises(fdtd.FDTDError, match="put the source on node 4"):
        fdtd.simulate(np.ones(SHAPE, np.float32),
                      fdtd.Source((0, 2, 20), frequency=1e9, bandwidth=5e8,
                                  component="z"),
                      2, dx=DX, boundary=("periodic", "periodic", "open"))


def test_an_unknown_per_axis_boundary_is_refused():
    with pytest.raises(fdtd.FDTDError, match="one kind per axis"):
        fdtd.simulate(np.ones(SHAPE, np.float32),
                      fdtd.Source((2, 2, 20), frequency=1e9, bandwidth=5e8), 2,
                      dx=DX,
                      boundary=("periodic", "open"))
