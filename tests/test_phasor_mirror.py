"""How a phasor box meets a mirror: the options measured, not argued.

Paper 1's collection fraction (NA 0.8) is read from ``fdtd.simulate``'s phasor
box through ``farfield.py``. The emitter sits over a gold mirror, and the box
has to meet that mirror somehow. These tests measure the ways it can, on tiny
grids, so the choice in ``scripts/paper1_nanowire.py`` (openspec change
``paper1-collection``) rests on numbers:

* **B, the PEC image**: five faces, the bottom on the mirror's top surface,
  ``farfield``'s ``pec_mirror_z`` image of each face;
* **C, a closed box in air**: six faces, the bottom between the mirror and the
  emitter, free space;
* **D, an open bottom**: B's five faces without the image;
* **A, the full enclosure**: six faces around the emitter AND the whole
  mirror, every face in vacuum, free space. Possible only where the mirror is
  finite and has vacuum all round it -- which is the paper-1 scene: a gold
  pad of finite width with the layout's open pad on every side, below too.

Two scenes. Over a PERFECT conductor (an infinite plane, through the
absorber) the image-dipole pattern is the closed-form answer and B is exact.
Over a small Rakić Lorentz-Drude gold slab at 900 nm there is no closed form,
but A is exact for whatever the grid holds -- the equivalence theorem with
every source inside and vacuum outside -- and the test shows it (two boxes of
different size agree to 1e-4), so A is the reference the others are read
against.

The PEC plane is a test-only polarisation term that zeroes the Yee samples a
perfect conductor at node layer ``k_m`` zeroes: Ex and Ey on the layers
``k <= k_m``, Ez below the plane only (``k < k_m``, positions up to
``k_m - 1/2``). The solver's own node mask would also zero the Ez half a cell
ABOVE the plane, which moves the image plane by half a cell (here 0.2 rad of
``k h``) and is a different question.

Every number below was measured on these grids (``OMP_NUM_THREADS=2``); the
tolerances are those measurements with headroom.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import farfield  # noqa: E402
import fdtd  # noqa: E402

NA = 0.8
PML = 6
SETTLE_PERIODS = 4          # after the 5-period ramp; nothing here resonates
WINDOW_PERIODS = 4


class _PECPlane(fdtd.PolarisationTerm):
    """A perfect conductor filling the node layers ``k <= k_m``, Yee-exact."""

    name = "PEC plane"

    def __init__(self, k_m: int):
        self.k_m = int(k_m)

    def advance(self, fields, dt, step):
        k = self.k_m
        return {c: (where, -fields[c][where].copy()) for c, where in
                (("x", np.s_[:, :, :k + 1]), ("y", np.s_[:, :, :k + 1]),
                 ("z", np.s_[:, :, :k]))}


def _cw_run(shape, dx, frequency, cell, box, terms):
    """One x-dipole CW run with a phasor box over its settled tail."""
    dt = fdtd.courant_dt(dx, dx, dx)
    for term in terms:
        stable = getattr(term, "stable_dt", None)
        if stable is not None:
            dt = min(dt, stable(dt, (dx, dx, dx)))
    steps = int(math.ceil((5 + SETTLE_PERIODS + WINDOW_PERIODS) / (frequency * dt)))
    tail = int(round(WINDOW_PERIODS / (frequency * dt)))
    source = fdtd.Source(cell, frequency=frequency, profile="cw", component="x")
    run = fdtd.simulate(np.ones(shape, np.float32), source, steps, dx=dx,
                        boundary="open", pml_cells=PML, ade_terms=list(terms),
                        record_drive=True, record_source_field=True,
                        phasor_box=box, phasor_frequencies=[frequency],
                        phasor_start=steps - tail)
    delivered = fdtd.delivered_power(run["drive"], run["source_field"], run["dt"],
                                     volume=run["source_volume"],
                                     eps_r=run["source_eps"], frequency=frequency,
                                     periods=WINDOW_PERIODS)
    return run["phasor_faces"][0], delivered


def _box(c, z_lo, z_hi, half):
    """``(centre, half_extent)`` with faces at z_lo..z_hi (z_hi raised to even)."""
    if (z_lo + z_hi) % 2:
        z_hi += 1
    return ((c, c, (z_lo + z_hi) // 2), (half, half, (z_hi - z_lo) // 2)), z_lo


def _five(faces):
    return [face for face in faces if face.normal != "-z"]


def _pattern(faces, frequency, **kwargs):
    """U on the upper hemisphere, 19 x 24 directions, theta 0..pi/2."""
    theta = np.linspace(0.0, math.pi / 2.0, 19)
    phi = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    return theta, phi, farfield.far_field(faces, frequency, theta, phi,
                                          **kwargs).intensity


def _worst(u, reference):
    return float(np.max(np.abs(u / u.max() - reference / reference.max())))


# --------------------------------------------------------------------------
# a perfect conductor: the image dipole is the answer
# --------------------------------------------------------------------------

PEC_DX = 1e-8
PEC_CPW = 16                  # cells per wavelength
PEC_FREQUENCY = fdtd.C0 / (PEC_CPW * PEC_DX)
PEC_PLANE = PML + 2           # the conductor's top node layer
PEC_HEIGHT = 4                # cells: h = lambda / 4
PEC_HALF = 8
PEC_KH = 2.0 * math.pi * PEC_HEIGHT / PEC_CPW


def _image_dipole(theta, phi):
    """An x-dipole a height h over a perfect conductor (its image is -p at -h)."""
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return ((np.cos(t) * np.cos(p)) ** 2 + np.sin(p) ** 2) \
        * np.sin(PEC_KH * np.cos(t)) ** 2


def _image_dipole_fraction():
    """The closed-form NA fraction of ``_image_dipole`` (Gauss-Legendre in cos)."""
    mu, w = np.polynomial.legendre.leggauss(200)

    def cone(c0):
        m = 0.5 * (1.0 - c0) * mu + 0.5 * (1.0 + c0)
        # the phi mean of cos^2 t cos^2 p + sin^2 p is (1 + cos^2 t) / 2
        return float(np.sum(0.5 * (1.0 - c0) * w * 0.5 * (1.0 + m ** 2)
                            * np.sin(PEC_KH * m) ** 2))
    return cone(math.sqrt(1.0 - NA ** 2)) / cone(0.0)


@functools.lru_cache(maxsize=None)
def _pec(bottom: str):
    """``bottom`` "plane": the box's bottom face on the conductor (B, D);
    "air": two layers above it (C). 37 x 37 x 31 nodes, dx 10 nm."""
    n = 2 * (PEC_HALF + PML + 1 + 3) + 1
    c = n // 2
    dipole = PEC_PLANE + PEC_HEIGHT
    top = dipole + PEC_HALF
    nz = top + 1 + 3 + PML + 1
    z_lo = PEC_PLANE if bottom == "plane" else PEC_PLANE + 2
    box, _ = _box(c, z_lo, top, PEC_HALF)
    faces, delivered = _cw_run((n, n, nz), PEC_DX, PEC_FREQUENCY, (c, c, dipole),
                               box, [_PECPlane(PEC_PLANE)])
    return faces, delivered, PEC_PLANE * PEC_DX


def test_over_a_perfect_conductor_the_image_is_the_answer():
    """B -- five faces, the bottom on the plane, the PEC image -- against the
    closed-form image dipole (x-dipole, h = lambda/4, 16 cells per wavelength,
    37 x 37 x 31 nodes, f = 1.874 PHz). Measured: NA 0.8 fraction 0.7666
    against 0.7661 closed form (tolerance 0.005), the upper-hemisphere
    pattern 0.0062 off it (tolerance 0.02), the upward far-field power 0.993
    of the delivered power (tolerance 2%: over a perfect conductor every watt
    goes up). The bottom face itself adds nothing under the image (its
    tangential E is zero on the plane and its J cancels its own image)."""
    faces, delivered, plane = _pec("plane")
    kwargs = dict(pec_mirror_z=plane)
    fraction = farfield.collection_fraction(_five(faces), PEC_FREQUENCY, NA, **kwargs)
    up = farfield.hemisphere_power(_five(faces), PEC_FREQUENCY, **kwargs)
    theta, phi, u = _pattern(_five(faces), PEC_FREQUENCY, **kwargs)
    error = _worst(u, _image_dipole(theta, phi))
    with_bottom = farfield.collection_fraction(faces, PEC_FREQUENCY, NA, **kwargs)
    print(f"\nB: fraction {fraction:.4f} against {_image_dipole_fraction():.4f}, "
          f"pattern {error:.4f} off, up / delivered {up / delivered:.4f}, "
          f"with the bottom face {with_bottom:.4f}")
    assert abs(fraction - _image_dipole_fraction()) < 0.005
    assert error < 0.02
    assert abs(up / delivered - 1.0) < 0.02
    assert abs(with_bottom - fraction) < 1e-3


def test_over_a_perfect_conductor_a_box_in_air_misses_the_mirror():
    """What C and D get wrong where the answer is known. C, a closed box in
    air (bottom two layers above the plane), radiates in free space only the
    sources inside it: the dipole without its reflection, so it reads the
    free x-dipole's 0.4960 (measured 0.4996) for the image dipole's 0.7661,
    its pattern 0.97 off and only 0.43 of the delivered power upward. The
    same faces WITH the image are right again (0.7660, pattern 0.002): over a
    perfect conductor the image, not the bottom's height, is what matters. D,
    the open bottom without an image on the plane, keeps 0.951 of the power
    but its pattern is 0.32 off."""
    faces, delivered, plane = _pec("air")
    free = farfield.collection_fraction(faces, PEC_FREQUENCY, NA)
    up = farfield.hemisphere_power(faces, PEC_FREQUENCY)
    theta, phi, u = _pattern(faces, PEC_FREQUENCY)
    imaged = farfield.collection_fraction(faces, PEC_FREQUENCY, NA, pec_mirror_z=plane)
    _, _, u_img = _pattern(faces, PEC_FREQUENCY, pec_mirror_z=plane)
    reference = _image_dipole(theta, phi)
    open_faces, open_delivered, _ = _pec("plane")
    _, _, u_open = _pattern(_five(open_faces), PEC_FREQUENCY)
    up_open = farfield.hemisphere_power(_five(open_faces), PEC_FREQUENCY)
    print(f"\nC: fraction {free:.4f}, pattern {_worst(u, reference):.4f} off, "
          f"up / delivered {up / delivered:.4f}; with the image {imaged:.4f} "
          f"({_worst(u_img, reference):.4f}); D: pattern "
          f"{_worst(u_open, reference):.4f} off, up / delivered "
          f"{up_open / open_delivered:.4f}")
    assert abs(free - 0.4960) < 0.01                        # the free dipole's
    assert _image_dipole_fraction() - free > 0.25
    assert _worst(u, reference) > 0.8
    assert 0.38 < up / delivered < 0.48
    assert abs(imaged - _image_dipole_fraction()) < 0.005
    assert _worst(u_img, reference) < 0.02
    assert _worst(u_open, reference) > 0.25


# --------------------------------------------------------------------------
# a finite Lorentz-Drude gold slab: the full enclosure is the reference
# --------------------------------------------------------------------------

LD_DX = 3e-8
LD_WAVELENGTH = 9e-7
LD_FREQUENCY = fdtd.C0 / LD_WAVELENGTH
LD_SLAB_HALF = 6              # cells: the slab is 360 nm wide (0.4 wavelength)
LD_SLAB_CELLS = 2             # 60 nm thick, 2.2 skin depths
LD_HEIGHT = 7                 # cells above the slab's top: 210 nm, k h = 1.466
LD_CLEAR = 2                  # layers between a face and the slab
LD_ABOVE = 3                  # layers between the dipole and the top face
LD_N = 2 * (LD_SLAB_HALF + LD_CLEAR + 2 + PML + 1 + 1) + 1     # 37: room for A2
LD_C = LD_N // 2
LD_BOTTOM = PML + 1 + 1 + LD_CLEAR + 2                          # slab's lowest layer
LD_TOP = LD_BOTTOM + LD_SLAB_CELLS                              # slab's top layer
LD_DIPOLE = LD_TOP + LD_HEIGHT
LD_TOP_FACE = LD_DIPOLE + LD_ABOVE
LD_NZ = LD_TOP_FACE + 2 + 1 + PML + 2                           # 35

LD_BOXES = {
    # name: (bottom face layer, top face layer, lateral half extent)
    "A": (LD_BOTTOM - LD_CLEAR, LD_TOP_FACE, LD_SLAB_HALF + LD_CLEAR),
    "A2": (LD_BOTTOM - LD_CLEAR - 2, LD_TOP_FACE + 2, LD_SLAB_HALF + LD_CLEAR + 2),
    "B": (LD_TOP, LD_TOP_FACE, LD_SLAB_HALF + LD_CLEAR),
    "C": (LD_TOP + 2, LD_TOP_FACE, LD_SLAB_HALF + LD_CLEAR),
}


@functools.lru_cache(maxsize=None)
def _gold(name: str):
    """One run of the gold-slab scene with box ``name``; the same scene each time.

    37 x 37 x 35 nodes at dx 30 nm, CPML 6, 900 nm (gold's Rakić eps is
    -27.6 + 2.45i there), an x-dipole 210 nm over a 360 x 360 x 60 nm slab
    with vacuum all round it. About 2.5 s a run at ``OMP_NUM_THREADS=2``.
    """
    region = np.zeros((LD_N, LD_N, LD_NZ), dtype=bool)
    lat = slice(LD_C - LD_SLAB_HALF, LD_C + LD_SLAB_HALF + 1)
    region[lat, lat, LD_BOTTOM:LD_TOP + 1] = True
    z_lo, z_hi, half = LD_BOXES[name]
    box, _ = _box(LD_C, z_lo, z_hi, half)
    faces, delivered = _cw_run((LD_N, LD_N, LD_NZ), LD_DX, LD_FREQUENCY,
                               (LD_C, LD_C, LD_DIPOLE), box,
                               [fdtd.LorentzDrude("Au", region)])
    return faces, delivered


def test_the_full_enclosure_does_not_depend_on_its_box():
    """A is exact for the scene the grid holds, and the test of that is two
    different boxes: A (faces 2 layers off the slab) and A2 (4 layers, every
    face moved out by 2) read the same NA 0.8 fraction, measured 0.6121 and
    0.6121 (tolerance 0.002), and each closes on itself -- the full-sphere far
    field over its surface power, measured 0.9999 and 1.0006 (tolerance 0.5%). The
    surface power is below the delivered power by what the gold absorbs
    (measured 0.984 and 0.985 of it; between 0.95 and 1). Any option that leaves a
    source outside the box would move with the box."""
    readings = {}
    for name in ("A", "A2"):
        faces, delivered = _gold(name)
        surface = farfield.surface_power(faces)
        sphere = farfield.cone_power(faces, LD_FREQUENCY, math.pi)
        readings[name] = (farfield.collection_fraction(faces, LD_FREQUENCY, NA),
                          sphere / surface, surface / delivered)
    print("\n" + "; ".join(f"{k}: fraction {v[0]:.4f}, sphere / surface {v[1]:.4f}, "
                           f"surface / delivered {v[2]:.4f}"
                           for k, v in readings.items()))
    assert abs(readings["A"][0] - readings["A2"][0]) < 0.002
    for fraction, closure, absorbed in readings.values():
        assert abs(closure - 1.0) < 0.005
        assert 0.95 < absorbed < 1.0


def test_on_a_finite_gold_pad_the_mirror_models_are_off():
    """B and C against A on the same gold-slab scene. Measured: A 0.6121;
    B (PEC image, bottom on the slab's top) 0.7556, +0.143 -- the image is an
    INFINITE perfect mirror under a 0.4-wavelength pad, and its pattern is
    0.32 off A's; C (closed box in air, free space) 0.4964, -0.116, the free
    dipole again (its upward power 0.48 of A's: the reflection is missing).
    The bands are the measurement +-0.03. On a slab 840 nm wide (53 x 53 x 35
    nodes, not run here) A reads 0.7172 and B 0.7346, +0.017: B's error falls
    with the pad's width but does not reach zero, and it is the error of a
    model of the mirror, where A has none."""
    faces_a, _ = _gold("A")
    reference = farfield.collection_fraction(faces_a, LD_FREQUENCY, NA)
    up_a = farfield.hemisphere_power(faces_a, LD_FREQUENCY)
    theta, phi, u_a = _pattern(faces_a, LD_FREQUENCY)
    faces_b, _ = _gold("B")
    image = dict(pec_mirror_z=LD_TOP * LD_DX)
    pec = farfield.collection_fraction(_five(faces_b), LD_FREQUENCY, NA, **image)
    _, _, u_b = _pattern(_five(faces_b), LD_FREQUENCY, **image)
    faces_c, _ = _gold("C")
    air = farfield.collection_fraction(faces_c, LD_FREQUENCY, NA)
    up_c = farfield.hemisphere_power(faces_c, LD_FREQUENCY)
    print(f"\nA {reference:.4f}; B {pec:.4f} ({pec - reference:+.4f}, pattern "
          f"{_worst(u_b, u_a):.4f} off); C {air:.4f} ({air - reference:+.4f}, "
          f"upward {up_c / up_a:.4f} of A's)")
    assert 0.11 < pec - reference < 0.17
    assert _worst(u_b, u_a) > 0.2
    assert -0.15 < air - reference < -0.085
    assert abs(air - 0.4960) < 0.01
    assert 0.4 < up_c / up_a < 0.56
