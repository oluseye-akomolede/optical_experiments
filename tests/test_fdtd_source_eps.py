"""The permittivity the update applies at a point source: ``fdtd.source_permittivity``.

``delivered_power``, ``delivered_power_spectrum``, ``purcell_spectrum`` and
``dipole_moment`` take a scalar ``eps_r``: the ``eps`` in ``J = -eps0 eps S``,
the current the source's ``dt * S`` stands for. That is the permittivity the
source's own E sample is updated with, and ``simulate`` now returns it as
``"source_eps"``. ``simulate`` does no averaging of its own: the sample at
array index ``cell`` is advanced with ``1 / (eps0 eps_r[cell])``, so in a
dielectric -- homogeneous or at an interface -- the applied value IS the
layout's node value. A Lorentz-Drude metal is where they part: every sample it
owns (either node of the segment metal) is rescaled to ``eps_inf = 1``.

The reference is independent of the bookkeeping: in a lossless run the power
through a closed surface (``farfield.surface_power`` of the phasor box) is the
power the source delivers.

MEASURED ON: a 37^3 uniform grid, dx = 10 nm, CPML 6, a CW Ez dipole at the
centre node, 16 cells per wavelength (f = 1.874 PHz), the phasor box of
half-extent 8 cells read over the last 8 drive periods with a Hann window (the
geometry of ``tests/test_fdtd_phasor_box.py``), about 2 s a run with
``OMP_NUM_THREADS=2``:

* vacuum: delivered / surface - 1 = +0.0016;
* a dielectric face (eps 4 block, the source on its top node layer, the Ez
  sample reaching into vacuum): applied = cell = 4, error +0.0023 -- the
  same as vacuum, so the node value is the right one (an "averaged" 2.5 would
  read -0.37);
* a lossless Lorentz block (one pole, eps(f) = 2.99, no damping) on the node
  above the source, the source node in an eps 4 block: applied 1, cell 4;
  read with the applied value the error is +0.0008, with the cell value
  +3.0032 (four times the power).
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import farfield  # noqa: E402
import fdtd  # noqa: E402

DX = 1e-8
HALF = 8
PML = 6
GAP = 3
N = 2 * (HALF + PML + 1 + GAP) + 1          # 37
C = N // 2
FREQUENCY = fdtd.C0 / (16 * DX)
WINDOW_PERIODS = 8
#: One undamped Lorentz pole well above the drive: lossless, eps(f) = 2.99.
LOSSLESS = fdtd.Metal("lossless", 26.0, ((1.0, 20.0, 0.0),), (0.1, 30.0))


def _rel(expected, *, rel):
    return pytest.approx(expected, rel=rel, abs=0.0)


def _lossless_terms():
    """The lossless block on the two node layers above the source."""
    region = np.zeros((N,) * 3, dtype=bool)
    region[C - 1:C + 2, C - 1:C + 2, C + 1:C + 3] = True
    return (fdtd.LorentzDrude(LOSSLESS, region),)


def _scene(name):
    eps = np.ones((N,) * 3)
    terms = ()
    if name == "dielectric-face":
        eps[C - 3:C + 4, C - 3:C + 4, C - 3:C + 1] = 4.0  # top layer = node C
    elif name == "metal-face":
        eps[C - 3:C + 4, C - 3:C + 4, C - 2:C + 3] = 4.0
        terms = _lossless_terms()
    return eps, terms


@functools.lru_cache(maxsize=None)
def _run(name):
    """One open-box CW run with the phasor box; returns (run, surface power)."""
    eps, terms = _scene(name)
    dt = fdtd.courant_dt(DX, DX, DX)
    for term in terms:                       # what simulate does with dt
        dt = min(dt, term.stable_dt(dt, (DX, DX, DX)))
    travel = HALF * math.sqrt(3.0) * DX / fdtd.C0
    steps = (int(math.ceil((5.0 + WINDOW_PERIODS + 1.0) / (FREQUENCY * dt)))
             + int(math.ceil(1.5 * travel / dt)))
    tail = int(round(WINDOW_PERIODS / (FREQUENCY * dt)))
    source = fdtd.Source((C, C, C), frequency=FREQUENCY, profile="cw",
                         component="z")
    run = fdtd.simulate(eps, source, steps, dx=DX, boundary="open",
                        pml_cells=PML, ade_terms=terms, record_drive=True,
                        record_source_field=True,
                        phasor_box=((C,) * 3, (HALF,) * 3),
                        phasor_frequencies=[FREQUENCY],
                        phasor_start=steps - tail)
    assert run["dt"] == dt
    return run, farfield.surface_power(run["phasor_faces"][0])


def _delivered(run, eps_r):
    return fdtd.delivered_power(run["drive"], run["source_field"], run["dt"],
                                volume=run["source_volume"], eps_r=eps_r,
                                frequency=FREQUENCY, periods=WINDOW_PERIODS)


# --------------------------------------------------------------------------
# (a) homogeneous: the applied value is the cell value, the reading unchanged
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1.0, 2.25, 3.17 ** 2])
def test_a_homogeneous_dielectric_applies_its_cell_value_exactly(value):
    """In a homogeneous dielectric the applied value is ``eps_r[cell]`` to the
    bit (3.17^2 is not a float32, and the caller's float64 is what comes back),
    and ``delivered_power`` read with ``"source_eps"`` is the scalar reading,
    bit for bit. Catches ``simulate`` returning the float32 grid value or a
    rescale applied where no term claims the sample."""
    eps = np.full((12, 12, 12), value)
    cell = (6, 6, 6)
    assert fdtd.source_permittivity(eps, cell, "z") == float(eps[cell])
    source = fdtd.Source(cell, frequency=fdtd.C0 / (12 * DX), profile="cw",
                         component="z")
    run = fdtd.simulate(eps, source, 300, dx=DX, record_drive=True,
                        record_source_field=True)
    assert run["source_eps"] == float(eps[cell])
    kwargs = dict(volume=run["source_volume"], frequency=source.frequency,
                  periods=4)
    assert (fdtd.delivered_power(run["drive"], run["source_field"], run["dt"],
                                 eps_r=run["source_eps"], **kwargs)
            == fdtd.delivered_power(run["drive"], run["source_field"],
                                    run["dt"], eps_r=value, **kwargs))


def test_a_plane_wave_has_no_source_permittivity():
    """No point source, no sample to read: ``None``, like ``source_volume``."""
    source = fdtd.Source(kind="plane", direction="+x", component="y",
                         box=((7,) * 3, (13,) * 3),
                         frequency=fdtd.C0 / (12 * DX), profile="cw")
    run = fdtd.simulate(np.ones((21,) * 3), source, 2, dx=DX,
                        boundary="open", pml_cells=4)
    assert run["source_eps"] is None


# --------------------------------------------------------------------------
# (b) at an interface: the dielectric face and the metal face
# --------------------------------------------------------------------------

def test_a_dielectric_face_applies_the_node_value_and_reads_true():
    """The source on a dielectric block's top node layer, its Ez sample
    reaching into vacuum: the update still divides by ``eps_r[cell]`` = 4, and
    the reading with it matches the surface power as well as vacuum's does
    (measured +0.0023 against vacuum's +0.0016; tolerance 1%). A midpoint
    "averaged" 2.5 would read -0.37: the scalar node value is not the
    approximation here."""
    eps, terms = _scene("dielectric-face")
    run, surface = _run("dielectric-face")
    assert run["source_eps"] == fdtd.source_permittivity(
        eps, (C, C, C), "z", boundary="open") == 4.0
    vacuum, vacuum_surface = _run("vacuum")
    error = _delivered(run, run["source_eps"]) / surface - 1.0
    vacuum_error = _delivered(vacuum, vacuum["source_eps"]) / vacuum_surface - 1.0
    averaged = _delivered(run, 2.5) / surface - 1.0
    print(f"\ndielectric face: error {error:+.4f} (vacuum {vacuum_error:+.4f},"
          f" averaged 2.5 {averaged:+.4f})")
    assert abs(error) < 0.01
    assert abs(vacuum_error) < 0.01
    assert averaged < -0.3


def test_a_metal_face_applies_eps_inf_and_only_that_reads_true():
    """The source node in eps 4, the node above it (along z, the source's own
    axis) in a lossless Lorentz block: the Ez sample is the metal's, updated
    with eps_inf = 1. Read with the applied 1 the delivered power is the
    surface power (measured +0.0008; tolerance 1%); read with the cell's 4 it
    is four times it (measured +3.0032). Catches a ``source_permittivity``
    that returns the cell value, and ``simulate`` reporting the value without
    its terms."""
    eps, terms = _scene("metal-face")
    run, surface = _run("metal-face")
    cell = float(eps[C, C, C])
    applied = run["source_eps"]
    assert cell == 4.0
    assert applied == _rel(1.0, rel=1e-12)
    assert applied == fdtd.source_permittivity(
        eps, (C, C, C), "z", ade_terms=terms, boundary="open")
    good = _delivered(run, applied) / surface - 1.0
    bad = _delivered(run, cell) / surface - 1.0
    print(f"\nmetal face: applied {applied!r} error {good:+.4f},"
          f" cell {cell} error {bad:+.4f}")
    assert abs(good) < 0.01
    assert bad > 2.0


# --------------------------------------------------------------------------
# which sample a metal claims
# --------------------------------------------------------------------------

def _metal_at(node, shape=(9, 9, 9)):
    region = np.zeros(shape, dtype=bool)
    region[node] = True
    return fdtd.LorentzDrude("Au", region)


@pytest.mark.parametrize("node,component,expected", [
    ((4, 4, 5), "z", 1.0),    # the far end of the Ez segment: the metal's
    ((4, 4, 4), "z", 1.0),    # the source's own node: the metal's
    ((4, 4, 3), "z", 4.0),    # the segment below: another sample
    ((5, 4, 4), "z", 4.0),    # beside it along x: not on this segment
    ((5, 4, 4), "x", 1.0),    # ... but the Ex sample there is the metal's
    ((4, 5, 4), "x", 4.0),    # Ex beside a metal node along y: not claimed
])
def test_a_metal_claims_exactly_the_samples_its_update_rescales(
        node, component, expected):
    """The sample at ``cell`` spans nodes ``cell`` and ``cell + 1`` along the
    component's own axis, and a metal claims it when either is metal -- the
    mask ``bind`` builds its rescale from (``LorentzDrude.samples``). Catches
    reading another component's mask."""
    eps = np.full((9, 9, 9), 4.0)
    terms = (_metal_at(node),)
    assert fdtd.source_permittivity(eps, (4, 4, 4), component,
                                    ade_terms=terms) == _rel(expected, rel=1e-12)


def test_the_claim_is_the_mask_bind_rescales():
    """``bind``'s rescale is non-zero exactly where ``samples`` says, so the
    value the reading takes and the update cannot come from two rules."""
    shape = (9, 9, 9)
    term = _metal_at((4, 4, 5), shape)
    eps = np.full(shape, 4.0, dtype=np.float32)
    term.bind(shape=shape, dt=1e-18, inv_eps=1.0 / (fdtd.EPS0 * eps), xp=np)
    for component, part in term.parts.items():
        full = np.zeros(shape, dtype=bool)
        full[part["box"]] = part["rescale"] != 0
        assert np.array_equal(full, term.samples(component, shape)), component


def test_a_bad_request_is_refused():
    eps = np.ones((6, 6, 6))
    with pytest.raises(fdtd.FDTDError, match="component"):
        fdtd.source_permittivity(eps, (3, 3, 3), "w")
    with pytest.raises(fdtd.FDTDError, match="outside the grid"):
        fdtd.source_permittivity(eps, (3, 3, 6), "z")
    with pytest.raises(fdtd.FDTDError, match="3-D"):
        fdtd.source_permittivity(np.ones((6, 6)), (3, 3, 3), "z")
