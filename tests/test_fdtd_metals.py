"""Dispersive metals: the Rakić 1998 Lorentz-Drude library and its ADE term.

Everything here is tier 1: the library checks are arithmetic, the term is
driven directly on a handful of samples, and the two solver runs are under
16^3 cells.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402

#: Independently published n, k for the same model: refractiveindex.info,
#: database/data/main/<metal>/nk/Rakic-LD.yml (CC0; tabulated from Rakić et
#: al., Appl. Opt. 37, 5271 (1998), LD model). Rows copied verbatim: the
#: tabulated wavelength nearest 0.4, 0.8 and 1.6 um. (wavelength um, n, k)
PUBLISHED = {
    "Au": ((4.0285e-01, 1.4112, 1.7490), (7.9466e-01, 0.23010, 4.4679),
           (1.5931e+00, 0.58993, 9.9355)),
    "Ag": ((3.9746e-01, 0.16007, 1.7990), (8.0658e-01, 0.17457, 5.0710),
           (1.6050e+00, 0.42174, 10.543)),
    "Al": ((4.0143e-01, 0.45683, 4.7236), (7.9962e-01, 2.6719, 8.3382),
           (1.6060e+00, 1.6068, 15.914)),
    "Cu": ((3.9916e-01, 1.2496, 2.2459), (8.0344e-01, 0.24149, 4.9706),
           (1.5842e+00, 0.48830, 10.694)),
    "Cr": ((4.0108e-01, 1.7714, 3.5652), (8.0033e-01, 4.1450, 4.3029),
           (1.5970e+00, 4.2397, 5.0281)),
}


def test_the_library_matches_a_published_rakic_table():
    """n + i k from the library within 2% of the published value, 3 energies
    per metal (the spec's scenario). Both n and k enter, so a conjugated
    convention (k < 0) or a mistyped parameter fails."""
    assert sorted(PUBLISHED) == sorted(fdtd.METALS)
    for metal, rows in PUBLISHED.items():
        for wavelength_um, n, k in rows:
            nk = np.sqrt(complex(fdtd.permittivity(
                metal, fdtd.C0 / (wavelength_um * 1e-6))))
            error = abs(nk - complex(n, k)) / abs(complex(n, k))
            assert error < 0.02, (metal, wavelength_um, nk, (n, k))


def test_a_band_outside_the_fit_is_refused_naming_it():
    to_hz = fdtd.EV / (2 * math.pi)
    fdtd.check_metal_band("Au", 1.0 * to_hz, 3.0 * to_hz)        # inside
    with pytest.raises(fdtd.FDTDError, match=r"Au, 0\.2-5 eV"):
        fdtd.check_metal_band("Au", 0.15 * to_hz, 3.0 * to_hz)
    with pytest.raises(fdtd.FDTDError, match=r"Cr, 0\.1-5 eV"):
        fdtd.check_metal_band("Cr", 1.0 * to_hz, 5.5 * to_hz)
    with pytest.raises(fdtd.FDTDError, match="unknown metal"):
        fdtd.check_metal_band("Pt", 1.0 * to_hz, 2.0 * to_hz)


def driven_susceptibility(metal, electronvolts):
    """Drive the term with E = cos(w t) on its own (no curl) and fit P/eps0.

    ``advance`` returns -(P^{n+1} - P^n)/eps0 when the curl is zero and
    eps_r = 1, so the running sum is P^{n+1}/eps0. The fit has a constant
    column: switching the drive on leaves a Drude pole a permanent offset
    (no restoring force), and without it the fit reads up to 30% wrong."""
    shape = (3, 4, 5)
    region = np.zeros(shape, bool)
    region[1, 1:3, 1:4] = True
    f = electronvolts * fdtd.EV / (2 * math.pi)
    dt = 0.02 / (2 * math.pi * f)
    term = fdtd.LorentzDrude(metal, region)
    term.bind(shape=shape, dt=dt, xp=np,
              inv_eps=np.full(shape, 1 / fdtd.EPS0, np.float32))
    fields = {c: np.zeros(shape, np.float32) for c in "xyz"}
    steps = int(60e-15 / dt)            # 5x the slowest pole's decay time
    p = np.zeros(steps)
    total = 0.0
    for n in range(steps):
        fields["z"][:] = math.cos(2 * math.pi * f * n * dt)
        term.prepare(fields, dt, n)
        total -= float(term.advance(fields, dt, n)["z"][1][0, 0, 0])
        p[n] = total
    t = (np.arange(steps) + 1) * dt
    tail = slice(steps - 3000, steps)
    columns = np.stack([np.cos(2 * math.pi * f * t[tail]),
                        np.sin(2 * math.pi * f * t[tail]),
                        np.ones(3000)], 1)
    c, s, _ = np.linalg.lstsq(columns, p[tail], rcond=None)[0]
    return complex(c, s), complex(fdtd.permittivity(metal, f)) - 1


@pytest.mark.parametrize("metal,electronvolts",
                         [("Au", 1.0), ("Au", 3.0), ("Ag", 2.0), ("Cr", 2.0)])
def test_the_term_steps_the_models_susceptibility(metal, electronvolts):
    """At w dt = 0.02 the central-difference ADE reproduces chi(w) to 7.5e-3
    over all five metals at 1, 2 and 3 eV (measured); a sign or coefficient
    slip is off by tens of percent."""
    chi, model = driven_susceptibility(metal, electronvolts)
    assert abs(chi - model) / abs(model) < 1.5e-2, (chi, model)


def test_a_sample_on_a_metal_face_takes_the_metal():
    """One metal node: each E component takes it at the two samples whose
    segment touches it, on a non-square grid so an axis swap cannot agree."""
    shape = (7, 8, 9)
    region = np.zeros(shape, bool)
    region[3, 4, 5] = True
    term = fdtd.LorentzDrude("Au", region)
    term.bind(shape=shape, dt=1e-18, xp=np,
              inv_eps=np.full(shape, 1 / fdtd.EPS0, np.float32))
    expected = {"x": {(2, 4, 5), (3, 4, 5)}, "y": {(3, 3, 5), (3, 4, 5)},
                "z": {(3, 4, 4), (3, 4, 5)}}
    for component, cells in expected.items():
        part = term.parts[component]
        offset = np.array([s.start for s in part["box"]])
        found = {tuple(int(v) for v in offset + i)
                 for i in np.argwhere(part["keep"] > 0)}
        assert found == cells, component


def test_a_face_sample_takes_eps_inf_one_whatever_eps_r_says():
    """eps_r at a face sample's index can be the neighbouring dielectric's
    (it is shared by three components). The term must undo it: with E* - E^n
    = d from the core's update at eps_r = 4, the sample ends at E^n + 4 d
    (what eps = 1 gives), less the polarisation current."""
    shape = (5, 6, 7)
    region = np.zeros(shape, bool)
    region[2, 3, 3] = True
    eps = np.ones(shape, np.float32)
    eps[1, 3, 3] = 4.0                  # Ex at (1+1/2, 3, 3) touches the metal
    term = fdtd.LorentzDrude("Au", region)
    term.bind(shape=shape, dt=1e-18, xp=np, inv_eps=1 / (fdtd.EPS0 * eps))
    fields = {c: np.zeros(shape, np.float32) for c in "xyz"}
    term.prepare(fields, 1e-18, 0)      # E^n = 0, so P^{n+1} = 0
    fields["x"][1, 3, 3] = 0.25         # the core's curl increment d
    box, increment = term.advance(fields, 1e-18, 0)["x"]
    offset = tuple(s.start for s in box)
    local = (1 - offset[0], 3 - offset[1], 3 - offset[2])
    assert increment[local] == pytest.approx(0.75)      # d (4 - 1)


def test_a_metal_near_the_plane_wave_box_is_refused():
    """eps_r = 1 inside a metal, so only the term's region shows it."""
    shape = (24, 20, 22)
    eps = np.ones(shape, np.float32)
    region = np.zeros(shape, bool)
    region[11, 9, 8] = True             # one node, inside the box's shell
    source = fdtd.Source(frequency=fdtd.EV * 2 / (2 * math.pi),
                         bandwidth=fdtd.EV / (2 * math.pi), kind="plane",
                         direction="+x", component="z",
                         box=((8, 7, 7), (15, 12, 12)))
    with pytest.raises(fdtd.FDTDError, match="a dispersive metal"):
        fdtd.simulate(eps, source, 2, spacings=(5e-9,) * 3, boundary="open",
                      pml_cells=4, ade_terms=[fdtd.LorentzDrude("Au", region)])


def gold_box_run(xp=None, with_gold=True):
    """A 320 nm PEC box (lowest mode ~2.7 eV), a 3 eV pulse, a gold block."""
    shape = (16, 14, 15)
    eps = np.ones(shape, np.float32)
    region = np.zeros(shape, bool)
    region[9:14, 3:11, 4:12] = True
    to_hz = fdtd.EV / (2 * math.pi)
    source = fdtd.Source((4, 7, 7), frequency=3.0 * to_hz,
                         bandwidth=1.5 * to_hz, component="z")
    terms = [fdtd.LorentzDrude("Au", region)] if with_gold else []
    return fdtd.simulate(eps, source, 1500, probes=[(5, 6, 8, "z")],
                         spacings=(20e-9, 21e-9, 22e-9), boundary="metal",
                         ade_terms=terms, xp=xp)


def test_a_gold_block_absorbs_the_box_ringing_and_stays_bounded():
    """Without gold the lossless box rings on; with it the late field is a
    small fraction of the peak and finite (an unstable ADE grows instead)."""
    bare = np.abs(gold_box_run(with_gold=False)["traces"][0])
    gold = np.abs(gold_box_run()["traces"][0])
    assert np.all(np.isfinite(gold))
    late = slice(1200, 1500)
    assert bare[late].max() > 0.2 * bare.max(), "the control box did not ring"
    assert gold[late].max() < 0.05 * gold.max(), (
        gold[late].max() / gold.max())


def test_the_growth_check_sees_courant_and_the_metal():
    """Positive controls: with no poles the check is the Courant limit (1.01
    of it grows); 5 nm gold at 0.99 of Courant grows by about 1.16."""
    h = (5e-9, 5e-9, 5e-9)
    courant = fdtd.courant_limit(*h)
    assert fdtd.lorentz_drude_growth([], 0.99 * courant, h) < 1 + 1e-6
    assert fdtd.lorentz_drude_growth([], 1.01 * courant, h) > 1.2
    dt = fdtd.courant_dt(*h)
    gold = fdtd.lorentz_drude_coefficients("Au", dt)
    assert fdtd.lorentz_drude_growth(gold, dt, h) > 1.1


def test_a_metal_shrinks_dt_only_as_far_as_it_must():
    """5 nm gold needs about 0.98 of the Courant dt; 2 nm gold needs none.
    The 2 nm case pins the rounding tolerance: eigvals reads the Drude
    pole's static mode up to 6e-8 above 1, and a tolerance below that took
    2 nm gold down to 0.20 of the Courant dt for nothing."""
    term = fdtd.LorentzDrude("Au", np.zeros((2, 2, 2), bool))
    for nm, low, high in ((5, 0.9, 0.99), (2, 1.0, 1.0)):
        h = (nm * 1e-9,) * 3
        dt = fdtd.courant_dt(*h)
        reduced = term.stable_dt(dt, h)
        assert low <= reduced / dt <= high, (nm, reduced / dt)
        if reduced < dt:
            coefficients = fdtd.lorentz_drude_coefficients("Au", reduced)
            assert fdtd.lorentz_drude_growth(coefficients, reduced, h) < 1 + 1e-6


def test_simulate_runs_the_metal_at_the_reduced_dt():
    """The 20 nm gold box needs about 0.82 of the Courant dt; the run must
    report the dt it used, not the Courant one."""
    result = gold_box_run()
    courant = fdtd.courant_dt(20e-9, 21e-9, 22e-9)
    assert result["dt"] < 0.9 * courant
    assert np.all(np.isfinite(result["traces"]))
