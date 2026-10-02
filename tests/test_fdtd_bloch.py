"""Bloch-periodic faces (6.1b): a cell of an infinite crystal at wave vector k.

The check that cannot pass by accident is the supercell identity. A Bloch run
with a real source in one cell is, by linearity, the crystal driven by s e^{ik.R}
in the cell at lattice vector R, so its field in that cell is
Re(psi e^{ik.R}) = cos * psi_cos - sin * psi_sin. A plain k = 0 run of an
M-cell supercell (k = 2 pi p / (M L)) driven by s cos(k.R) in every cell must
then give the same numbers, cell by cell. A wrong sign, a wrong phase, a missing
sin set or a corner copied on one axis only all break it.

k = 0 must be the 6.1a run exactly; that is pinned by a digest of every output
taken from the 6.1a solver (30aed9c) on the same inputs.
"""

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402

DX = 0.01
CELL = (6, 5, 4)
SPACINGS = (np.linspace(0.8, 1.2, 5) * DX, np.full(4, 1.2 * DX),
            np.full(3, DX))
SOURCE_AT = (3, 2, 1)
COMPONENT = "y"
STEPS = 120
# (cell index, E component): seam nodes on every axis, and the interior
PROBES = [((4, 4, 2), "x"), ((0, 0, 0), "x"), ((5, 0, 3), "y"),
          ((0, 3, 0), "y"), ((5, 4, 2), "z"), ((0, 0, 1), "z"),
          ((3, 2, 2), "z")]


def _cell_eps():
    eps = np.ones(CELL, np.float32)
    eps[2:4, 1:3, :] = 4.0            # a rod; seam slabs stay vacuum
    return eps


def _source(cell):
    return fdtd.Source(cell, frequency=4e9, bandwidth=4e9, component=COMPONENT)


def _supercell(tiles):
    """The cell repeated ``tiles`` times per axis: eps, spacings, shape."""
    index = [np.arange(t * (n - 1) + 1) % (n - 1) for t, n in zip(tiles, CELL)]
    eps = _cell_eps()[np.ix_(*index)]
    spacings = tuple(np.tile(s, t) for s, t in zip(SPACINGS, tiles))
    return eps, spacings


def _offset(cell, m):
    return tuple(c + mi * (n - 1) for c, mi, n in zip(cell, m, CELL))


@pytest.mark.parametrize("tiles,turns", [
    ((3, 3, 1), (1, 2, 0)),           # paired on x and y (2pi/3, -2pi/3)
    ((2, 1, 1), (1, 0, 0)),           # zone edge on x: the real path, sign flip
    ((2, 3, 1), (1, 1, 0)),           # zone edge on x with paired y
])
def test_a_bloch_cell_is_one_cell_of_the_supercell(tiles, turns):
    period = [float(np.sum(s)) for s in SPACINGS]
    k = tuple(2 * math.pi * p / (t * L) for p, t, L in zip(turns, tiles, period))
    theta = [2 * math.pi * p / t for p, t in zip(turns, tiles)]
    paired = any(abs(math.sin(a)) > 1e-9 for a in theta)

    bloch = fdtd.simulate(_cell_eps(), _source(SOURCE_AT), STEPS,
                          probes=[c + (comp,) for c, comp in PROBES],
                          spacings=SPACINGS, boundary="periodic", bloch=k)
    if paired:
        assert bloch["traces_sin"] is not None
    else:
        assert bloch["traces_sin"] is None
    cos_tr = bloch["traces"].astype(np.float64)
    sin_tr = (np.zeros_like(cos_tr) if bloch["traces_sin"] is None
              else bloch["traces_sin"].astype(np.float64))
    scale = float(np.abs(cos_tr).max())
    assert scale > 0
    if paired:   # positive control: the sin set is really carrying field
        assert float(np.abs(sin_tr).max()) > 1e-2 * scale

    eps, spacings = _supercell(tiles)
    cells = [(a, b, c) for a in range(tiles[0]) for b in range(tiles[1])
             for c in range(tiles[2])]
    probes = [_offset(cell, m) + (comp,) for m in cells for cell, comp in PROBES]
    expected = np.zeros((len(probes), STEPS))
    for m in cells:             # one source per run; superpose with cos(k.R)
        weight = math.cos(sum(a * mi for a, mi in zip(theta, m)))
        if abs(weight) < 1e-12:
            continue
        run = fdtd.simulate(eps, _source(_offset(SOURCE_AT, m)), STEPS,
                            probes=probes, spacings=spacings,
                            boundary="periodic")
        expected += weight * run["traces"].astype(np.float64)

    differs = 0.0
    for index, m in enumerate(cells):
        phase = sum(a * mi for a, mi in zip(theta, m))
        got = math.cos(phase) * cos_tr - math.sin(phase) * sin_tr
        want = expected[index * len(PROBES):(index + 1) * len(PROBES)]
        mismatch = float(np.abs(got - want).max())
        assert mismatch < 5e-6 * scale, (
            f"cell {m}: the Bloch field is not the supercell's "
            f"({mismatch / scale:.2e} of the peak)")
        differs = max(differs, float(np.abs(want - expected[:len(PROBES)]).max()))
    # positive control: the cells really differ, so the phase was tested
    assert differs > 1e-2 * scale


def test_a_closed_bloch_box_conserves_energy_over_both_sets():
    shape = (9, 11, 13)
    spacings = tuple(np.linspace(0.8, 1.2, n - 1) * DX * (1 + 0.05 * a)
                     for a, n in enumerate(shape))
    period = [float(np.sum(s)) for s in spacings]
    k = (2.1 / period[0], -1.3 / period[1], 0.7 / period[2])   # all paired
    source = fdtd.Source((3, 4, 5), frequency=3e9, bandwidth=3e9, component="y")
    dt = fdtd.courant_dt(*(s.min() for s in spacings))
    after = int(source.end_time() / dt) + 5
    run = fdtd.simulate(np.ones(shape, np.float32), source, after + 600,
                        spacings=spacings, boundary="periodic", bloch=k,
                        record_energy=True)
    tail = np.asarray(run["energy"][after:])
    assert tail.mean() > 0
    drift = (tail.max() - tail.min()) / tail.mean()
    assert drift < 1e-5, f"energy drifts {drift:.2e} after the source ends"


# --- k = 0 is the 6.1a run, exactly -------------------------------------------

# sha256 over traces, dt, energy, peak (x, y, z) and frames, computed with the
# 6.1a solver at 30aed9c on exactly these inputs. Every operation in the step
# is a correctly rounded float32 elementwise one, so the digest does not depend
# on the CPU's vector width.
K0_DIGEST = "70531f91c493c52a468beb63c3bee90e19c0cc7677461e0c93d095dc534bd213"


def _k0_run(**extra):
    shape = (8, 7, 14)
    spacings = (np.linspace(0.9, 1.1, 7) * DX, np.full(6, 1.3 * DX),
                np.linspace(0.8, 1.2, 13) * DX)
    pec = np.zeros(shape, bool)
    pec[2:4, 2:4, 6:8] = True
    eps = np.ones(shape, np.float32)
    eps[4:6, :, 3:5] = 2.25
    source = fdtd.Source((6, 5, 4), frequency=3e9, bandwidth=3e9, component="x")
    probes = [(6, 5, 9, "x"), (0, 3, 9, "y"), (7, 6, 9, "z"), (3, 3, 10)]
    return fdtd.simulate(eps, source, 200, probes=probes, spacings=spacings,
                         boundary=("periodic", "periodic", "open"), pec=pec,
                         pml_cells=3, record_energy=True, record_peak=True,
                         frames=fdtd.frame_plan(shape, 200, max_frames=4),
                         **extra)


def _digest(run):
    h = hashlib.sha256()
    for key in ("traces", "dt", "energy", "peak", "frames"):
        value = run[key]
        if isinstance(value, dict):
            for c in sorted(value):
                h.update(np.ascontiguousarray(value[c]).tobytes())
        elif value is not None:
            h.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("bloch", [None, (0.0, 0.0, 0.0)])
def test_k_zero_reproduces_the_6_1a_run_exactly(bloch):
    run = _k0_run(bloch=bloch)
    assert run["traces_sin"] is None
    assert _digest(run) == K0_DIGEST


# --- refusals -----------------------------------------------------------------

def _tiny(**extra):
    return fdtd.simulate(_cell_eps(), _source(SOURCE_AT), 3, spacings=SPACINGS,
                         **extra)


def test_a_bloch_k_on_an_axis_that_is_not_periodic_is_refused():
    with pytest.raises(fdtd.FDTDError, match="along z needs that axis periodic"):
        _tiny(boundary=("periodic", "periodic", "metal"), bloch=(1.0, 0, 5.0))


@pytest.mark.parametrize("bloch", [(1.0, 2.0), "k", (1.0, float("nan"), 0.0)])
def test_a_malformed_bloch_vector_is_refused(bloch):
    with pytest.raises(fdtd.FDTDError, match="wave vector"):
        _tiny(boundary="periodic", bloch=bloch)


def test_a_paired_run_refuses_a_metal():
    region = np.zeros(CELL, bool)
    region[2:4, 1:3, 1:3] = True
    with pytest.raises(fdtd.FDTDError, match="sin copy"):
        _tiny(boundary="periodic", bloch=(30.0, 0.0, 0.0),
              ade_terms=[fdtd.LorentzDrude("Au", region)])


def test_a_paired_run_refuses_a_plane_wave():
    source = fdtd.Source(kind="plane", direction="+z", component="x",
                         box=((0, 0, 5), (5, 4, 14)), frequency=1e9,
                         bandwidth=5e8)
    with pytest.raises(fdtd.FDTDError, match="plane wave"):
        fdtd.simulate(np.ones((6, 5, 20), np.float32), source, 3, dx=DX,
                      boundary=("periodic", "periodic", "open"),
                      bloch=(40.0, 0.0, 0.0))
