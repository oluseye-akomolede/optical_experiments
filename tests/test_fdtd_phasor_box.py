"""The phasor box monitor: ``fdtd.simulate(..., phasor_box=..., phasor_frequencies=...)``.

The monitor records, during a run, the surface ``farfield.py`` transforms: per
face of a closed box of node layers, the complex E and H at every face-cell
centre (the staggered Yee components interpolated there, E and H stamped with
their own half-step-apart sample times). These tests drive it with the one
emitter that has an exact answer, a z-dipole in vacuum, and read it three
independent ways:

* **the power** -- ``farfield.surface_power`` of the monitor's faces against
  the power the source delivers at its own cell (``fdtd.delivered_power``, the
  scheme's own energy identity) and against the existing flux box;
* **the far field** -- ``farfield.far_field`` of the faces against the
  Hertzian ``sin^2(theta)``, and its full-sphere integral against the surface
  power;
* **the near field itself** -- the monitor's E and H on every face against
  ``farfield.hertzian_dipole`` with ONE complex moment fitted to E and one to
  H: the two moments must agree in phase (the half step) and in size with the
  drive's own moment (the normalisation).

MEASURED ON: a 37^3 uniform grid, dx = 10 nm, CPML 6 cells, the dipole
(Ez, CW) at the centre node, 16 cells per wavelength (f = 1.874 PHz,
wavelength 160 nm), a cubic box of half-extent 8 cells (0.5 wavelength,
faces at node layers 10..26), the monitor read over the last 8 drive periods
with a Hann window -- the same tail ``delivered_power`` reads. One run, about
a second with ``OMP_NUM_THREADS=2``; the tolerances below are those
measurements with headroom, and each docstring names the mutation it catches
(``tests/check_discriminating.py phasor-*`` turns each red).

The errors are the grid's and shrink with it: at 20 cells per wavelength
(41^3, box half-extent 10, otherwise the same; not run here, 2 s) the pattern
is 0.0051 off sin^2 (0.0081 at 16), the field residuals 0.63% / 0.72%
(0.94% / 1.11%), the E-H moment phase -0.20 degrees (-0.34), and the surface
power 0.9972 of the delivered power (0.9984).
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
sys.path.insert(0, str(ROOT / "tests"))

import farfield  # noqa: E402
import fdtd  # noqa: E402
from test_fdtd import _DeviceXp  # noqa: E402  (the refusing device fake)

DX = 1e-8
CELLS_PER_WAVELENGTH = 16
HALF = 8                     # box half-extent, cells
PML = 6
GAP = 3                      # cells between the box's H reach and the CPML
N = 2 * (HALF + PML + 1 + GAP) + 1          # 37
CENTRE = N // 2
FREQUENCY = fdtd.C0 / (CELLS_PER_WAVELENGTH * DX)
WINDOW_PERIODS = 8
#: An undriven frequency recorded alongside the drive's: its phasor must be
#: (nearly) empty, which pins that each frequency gets its own phase column.
OFF_FREQUENCY = 1.5 * FREQUENCY


def _rel(expected, *, rel):
    """``pytest.approx`` without its 1e-12 absolute floor (powers are ~1e-10 W)."""
    return pytest.approx(expected, rel=rel, abs=0.0)


@functools.lru_cache(maxsize=None)
def _vacuum_run():
    """The one shared run: delivered power, flux box and phasor box together."""
    dt = fdtd.courant_dt(DX, DX, DX)
    travel = HALF * math.sqrt(3.0) * DX / fdtd.C0
    steps = (int(math.ceil((5.0 + WINDOW_PERIODS + 1.0) / (FREQUENCY * dt)))
             + int(math.ceil(1.5 * travel / dt)))
    tail = int(round(WINDOW_PERIODS / (FREQUENCY * dt)))
    centre = (CENTRE,) * 3
    source = fdtd.Source(centre, frequency=FREQUENCY, profile="cw", component="z")
    run = fdtd.simulate(
        np.ones((N,) * 3, np.float32), source, steps, dx=DX, boundary="open",
        pml_cells=PML, record_drive=True, record_source_field=True,
        flux_box=(centre, (HALF,) * 3),
        phasor_box=(centre, (HALF,) * 3),
        phasor_frequencies=[FREQUENCY, OFF_FREQUENCY],
        phasor_start=steps - tail, phasor_window="hann")
    assert run["dt"] == dt
    return run


def _faces():
    return _vacuum_run()["phasor_faces"][0]


def _delivered():
    run = _vacuum_run()
    return fdtd.delivered_power(run["drive"], run["source_field"], run["dt"],
                                volume=run["source_volume"], eps_r=1.0,
                                frequency=FREQUENCY, periods=WINDOW_PERIODS)


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------

def test_the_faces_are_the_far_field_layout_of_the_box():
    """Six faces per frequency, in DIRECTIONS order, at the box's node layers,
    with the box's own primal cells as edges. Catches a face list out of
    order or a position taken from the wrong layer."""
    run = _vacuum_run()
    assert len(run["phasor_faces"]) == 2
    for faces in run["phasor_faces"]:
        assert [face.normal for face in faces] == list(fdtd.DIRECTIONS)
        for face in faces:
            layer = CENTRE + (HALF if face.normal[0] == "+" else -HALF)
            assert face.position == pytest.approx(layer * DX, rel=1e-12)
            expected = np.arange(CENTRE - HALF, CENTRE + HALF + 1) * DX
            np.testing.assert_allclose(face.u_edges, expected, rtol=1e-12)
            np.testing.assert_allclose(face.v_edges, expected, rtol=1e-12)
            assert face.E.shape == face.H.shape == (3, 2 * HALF, 2 * HALF)


# --------------------------------------------------------------------------
# (a) the power
# --------------------------------------------------------------------------

def test_the_monitors_surface_power_is_the_delivered_power():
    """In a lossless medium the power through any closed surface is the power
    the source delivers. Measured: surface power 0.9984 of the delivered power
    (tolerance 1%), and the existing flux box on the same faces reads 1.0270
    of it -- the flux route's own ``|E|^2 / eta`` wave-zone estimate, which
    reads a little high at half a wavelength (tolerance 4%).

    Catches the DFT normalisation (``1/sum(w)`` for ``2/sum(w)``: the power
    reads 0.2496, a quarter). The far field's normalised pattern is blind to
    it, which is why this test exists. Also catches every frequency phased at
    the first one's: the undriven 1.5 f column then reads the drive."""
    run = _vacuum_run()
    delivered = _delivered()
    surface = farfield.surface_power(_faces())
    flux = fdtd.radiated_power(run["flux"], run["dt"], index=1.0,
                               frequency=FREQUENCY, periods=WINDOW_PERIODS)
    print(f"\nsurface {surface:.6e} W, delivered {delivered:.6e} W "
          f"(ratio {surface / delivered:.5f}), flux {flux:.6e} W "
          f"(ratio {flux / delivered:.5f})")
    assert surface == _rel(delivered, rel=0.01)
    assert surface == _rel(flux, rel=0.04)
    # the undriven frequency's own column holds almost nothing (measured
    # 7.9e-8 of the drive's power: the Hann skirt 4 bins away)
    off = farfield.surface_power(run["phasor_faces"][1])
    assert abs(off) < 1e-3 * surface


# --------------------------------------------------------------------------
# (b) the far field
# --------------------------------------------------------------------------

def test_the_monitors_far_field_is_the_hertzian_sin_squared():
    """``U / max U`` equals ``sin^2(theta)`` on 37 x 24 directions (theta
    0.05..pi-0.05, the full sphere), measured 0.0081 off at worst (tolerance
    0.02), and ``int U dOmega`` over the sphere equals the surface power,
    measured 0.9949 (tolerance 1.5%).

    Catches: dropping the half-step offset (H stamped at (n+1) dt like E:
    pattern 0.046 off, sphere power 1.072 of the surface's); the DFT sign
    (``exp(-i w t)``, which is not a Maxwell solution at w: 0.56 off);
    swapping u and v in the face arrays (0.51); reading H on one side of the
    face instead of interpolating across it (0.39); taking a node sample at
    the cell corner instead of the in-plane mean (0.29)."""
    faces = _faces()
    theta = np.linspace(0.05, math.pi - 0.05, 37)
    phi = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    pattern = farfield.radiation_pattern(faces, FREQUENCY, theta, phi)
    expected = np.sin(theta)[:, None] ** 2 * np.ones_like(phi)[None, :]
    error = float(np.max(np.abs(pattern - expected / expected.max())))
    nodes, phis, weights = farfield.cone_quadrature(math.pi, 32, 64)
    sphere = float(np.sum(farfield.far_field(faces, FREQUENCY, nodes, phis).intensity
                          * weights))
    surface = farfield.surface_power(faces)
    print(f"\npattern off sin^2 by {error:.5f}; sphere / surface = "
          f"{sphere / surface:.5f}")
    assert error < 0.02
    assert sphere == _rel(surface, rel=0.015)


# --------------------------------------------------------------------------
# the near field: what the monitor recorded, sample by sample
# --------------------------------------------------------------------------

def _moments():
    """Least-squares complex moments fitting the faces' E and H separately."""
    pos = np.array([CENTRE * DX, CENTRE * DX, (CENTRE + 0.5) * DX])  # Ez's node
    num_e = num_h = 0.0
    den_e = den_h = 0.0
    pairs = []
    for face in _faces():
        e, h = farfield.hertzian_dipole(face.points(), [0.0, 0.0, 1.0], pos,
                                        FREQUENCY)
        e, h = np.moveaxis(e, -1, 0), np.moveaxis(h, -1, 0)
        num_e += np.vdot(e, face.E)
        den_e += np.vdot(e, e).real
        num_h += np.vdot(h, face.H)
        den_h += np.vdot(h, h).real
        pairs.append((face, e, h))
    p_e, p_h = num_e / den_e, num_h / den_h

    def residual(which, p):
        miss = sum(np.sum(np.abs(getattr(face, which) - p * ref[which == "H"]) ** 2)
                   for face, *ref in pairs)
        size = sum(np.sum(np.abs(p * ref[which == "H"]) ** 2) for face, *ref in pairs)
        return math.sqrt(miss / size)

    return p_e, p_h, residual("E", p_e), residual("H", p_h)


def test_the_monitors_fields_are_one_dipoles_e_and_h():
    """Every face sample against Jackson's exact dipole field: one complex
    moment fitted to all of E and one to all of H. Measured: the field
    residuals are 0.94% (E) and 1.11% (H) of the fitted field (tolerance 3%);
    the two moments differ in phase by 0.34 degrees (tolerance 1.5) and in
    size by 0.49% (tolerance 2%); the E moment is 1.0078 of the drive's own
    (``fdtd.dipole_moment``, tolerance 3%).

    Catches dropping the half-step offset most directly: the H moment then
    leads the E moment by 6.09 degrees (``w dt / 2`` is 6.43 here). Also the
    normalisation (the E moment reads 0.504 of the drive's) and the DFT sign,
    u/v swap and interpolation mutations (residuals 7% to 185%)."""
    run = _vacuum_run()
    p_e, p_h, r_e, r_h = _moments()
    drive = fdtd.dipole_moment(run["drive"], run["dt"],
                               volume=run["source_volume"], eps_r=1.0,
                               frequency=FREQUENCY, periods=WINDOW_PERIODS)
    angle = math.degrees(float(np.angle(p_h / p_e)))
    print(f"\nresidual E {r_e:.4f}, H {r_h:.4f}; p_H / p_E = {abs(p_h / p_e):.5f} "
          f"at {angle:+.3f} deg; |p_E| / drive moment = {abs(p_e) / drive:.5f}")
    assert r_e < 0.03 and r_h < 0.03
    assert abs(angle) < 1.5
    assert abs(p_h / p_e) == _rel(1.0, rel=0.02)
    assert abs(p_e) == _rel(drive, rel=0.03)


# --------------------------------------------------------------------------
# the other backend, and nothing else changes
# --------------------------------------------------------------------------

def _small(flux=True, **extra):
    """A 12^3 metal box and a short pulse: the cheapest run with a field. (The
    device fake has no ``xp.sum``, so its run leaves the flux box out.)"""
    source = fdtd.Source((6, 5, 6), frequency=3e9, bandwidth=3e9, component="z")
    return fdtd.simulate(np.ones((12, 12, 12), np.float32), source, 60,
                         probes=[(6, 6, 6, "z"), (3, 4, 8, "x")],
                         spacings=(0.01, 0.011, 0.009), boundary="metal",
                         flux_box=((6, 6, 6), (3, 3, 3)) if flux else None,
                         **extra)


def test_the_monitor_runs_on_the_device_fake_and_returns_host_faces():
    """Through the refusing device fake (an array that raises on any implicit
    host conversion, as cupy's does), the faces come back as host Faces with
    the numpy run's numbers. Catches a per-step host read in the accumulation
    (it would raise) and a transfer that dropped or reordered samples."""
    box = dict(phasor_box=((6, 6, 6), (3, 3, 2)), phasor_frequencies=[2e9, 3e9])
    device = _small(False, xp=_DeviceXp(), **box)["phasor_faces"]
    host = _small(False, **box)["phasor_faces"]
    assert len(device) == len(host) == 2
    for faces_d, faces_h in zip(device, host):
        for d, h in zip(faces_d, faces_h):
            assert type(d.E) is np.ndarray and d.E.dtype == np.complex128
            assert np.any(h.E != 0) and np.any(h.H != 0)
            np.testing.assert_array_equal(d.E, h.E)
            np.testing.assert_array_equal(d.H, h.H)


def test_without_the_option_nothing_changes():
    """The traces, the flux record and the dt are bit-identical with and
    without a phasor box, and the key is None when not asked for."""
    plain = _small()
    monitored = _small(phasor_box=((6, 6, 6), (3, 3, 2)),
                       phasor_frequencies=3e9)
    assert plain["phasor_faces"] is None
    assert len(monitored["phasor_faces"]) == 1
    np.testing.assert_array_equal(plain["traces"], monitored["traces"])
    np.testing.assert_array_equal(plain["flux"], monitored["flux"])
    assert plain["dt"] == monitored["dt"]


# --------------------------------------------------------------------------
# (c) refusals
# --------------------------------------------------------------------------

def _open_refusal(box, **extra):
    source = fdtd.Source((CENTRE,) * 3, frequency=FREQUENCY, profile="cw",
                         component="z")
    kwargs = dict(phasor_box=box, phasor_frequencies=[FREQUENCY])
    kwargs.update(extra)
    return fdtd.simulate(np.ones((N,) * 3, np.float32), source, 2, dx=DX,
                         boundary="open", pml_cells=PML, **kwargs)


def test_a_phasor_box_in_the_absorber_is_refused():
    """A face must leave the H half a cell outside it clear of the CPML too:
    on this 37-node axis with a 6-cell layer the faces may run 7..29. A box
    whose faces are 6..30 is one a flux box accepts (its E-only faces may sit
    on the first CPML-free node) and the phasor box refuses; 3..33 is deep in
    the layer. Catches the phasor box checked with the flux box's rule (the
    half cell of H reach forgotten)."""
    centre = (CENTRE,) * 3
    primal = [np.full(N - 1, DX)] * 3
    fdtd.flux_faces((centre, (12,) * 3), (N,) * 3, primal, PML)     # 6..30: fine
    fdtd.phasor_box_layers((centre, (11,) * 3), (N,) * 3, PML)      # 7..29: fine
    for half in (12, 15):
        with pytest.raises(fdtd.FDTDError, match="absorbing layer"):
            _open_refusal((centre, (half,) * 3))
    # one axis is enough, and an axis with no absorber may go to layer 1
    with pytest.raises(fdtd.FDTDError, match="axis 2 of 37"):
        _open_refusal((centre, (8, 8, 12)))
    fdtd.phasor_box_layers((centre, (17, 8, 8)), (N,) * 3, [0, PML, PML])


def test_a_phasor_box_off_the_grid_is_refused():
    """Faces past the live layers 1..n-2, or a box nowhere near the grid,
    are refused with the layers named; so are a malformed box, a missing
    half of the option, a frequency at or above Nyquist, and a start step
    outside the run."""
    with pytest.raises(fdtd.FDTDError, match="live layers 1..35"):
        _open_refusal(((CENTRE, CENTRE, 30), (8, 8, 8)))
    with pytest.raises(fdtd.FDTDError, match="live layers"):
        _open_refusal(((100, 100, 100), (2, 2, 2)))
    with pytest.raises(fdtd.FDTDError, match="three integers"):
        _open_refusal("box")
    with pytest.raises(fdtd.FDTDError, match="at least one cell"):
        _open_refusal(((CENTRE,) * 3, (0, 4, 4)))
    with pytest.raises(fdtd.FDTDError, match="only phasor_box"):
        _open_refusal(((CENTRE,) * 3, (8,) * 3), phasor_frequencies=None)
    with pytest.raises(fdtd.FDTDError, match="only phasor_frequencies"):
        _open_refusal(None)
    dt = fdtd.courant_dt(DX, DX, DX)
    for bad in ([0.6 / dt], [-FREQUENCY], [], [float("nan")]):
        with pytest.raises(fdtd.FDTDError, match="phasor"):
            _open_refusal(((CENTRE,) * 3, (8,) * 3), phasor_frequencies=bad)
    for start in (2, -1):
        with pytest.raises(fdtd.FDTDError, match="phasor_start"):
            _open_refusal(((CENTRE,) * 3, (8,) * 3), phasor_start=start)


def test_a_bloch_run_refuses_a_phasor_box():
    """A paired Bloch run (k L neither 0 nor pi) has no physical field to put
    a surface around, as for the flux box; k = 0 and the zone edge run the
    real path and accept it. Catches the refusal dropped (the cos set alone
    would be transformed as if it were the field)."""
    eps = np.ones((9, 9, 9), np.float32)
    source = fdtd.Source((4, 4, 4), frequency=3e9, bandwidth=3e9, component="z")
    period = 8 * 0.01
    box = dict(phasor_box=((4, 4, 4), (2, 2, 2)), phasor_frequencies=[3e9])
    run = dict(dx=0.01, boundary=("periodic", "periodic", "metal"))
    for real in (None, (math.pi / period, 0.0, 0.0)):
        faces = fdtd.simulate(eps, source, 3, bloch=real, **run, **box)["phasor_faces"]
        assert len(faces) == 1
    with pytest.raises(fdtd.FDTDError, match="drop phasor_box or the Bloch phase"):
        fdtd.simulate(eps, source, 3, bloch=(2 * math.pi / (3 * period), 0.0, 0.0),
                      **run, **box)
