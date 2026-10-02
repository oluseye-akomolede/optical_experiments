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
#: al., Appl. Opt. 37, 5271 (1998), LD model). Rows copied verbatim: 12 per
#: metal, evenly spaced through each table, both ends included, so the short,
#: mid and long wavelength poles all show. (wavelength um, n, k)
PUBLISHED = {
    "Au": ((2.4797e-01, 1.0941e+00, 1.5478e+00),
            (3.3177e-01, 1.5422e+00, 1.6012e+00),
            (4.4390e-01, 1.3011e+00, 1.6840e+00),
            (5.9393e-01, 3.7366e-01, 2.7930e+00),
            (7.9466e-01, 2.3010e-01, 4.4679e+00),
            (1.0632e+00, 2.7776e-01, 6.4377e+00),
            (1.4458e+00, 5.2226e-01, 9.0142e+00),
            (1.9344e+00, 7.3408e-01, 1.2164e+01),
            (2.5882e+00, 1.3425e+00, 1.6353e+01),
            (3.4629e+00, 2.0620e+00, 2.1457e+01),
            (4.6333e+00, 3.0386e+00, 2.8646e+01),
            (6.1992e+00, 5.0974e+00, 3.8118e+01)),
    "Ag": ((2.4797e-01, 4.4265e-01, 1.1737e+00),
            (3.5324e-01, 2.5903e-01, 1.2030e+00),
            (5.0321e-01, 1.3208e-01, 2.7927e+00),
            (7.1684e-01, 1.5727e-01, 4.4272e+00),
            (1.0212e+00, 2.2512e-01, 6.5716e+00),
            (1.4547e+00, 3.6298e-01, 9.5296e+00),
            (2.1134e+00, 6.6249e-01, 1.3952e+01),
            (3.0107e+00, 1.2442e+00, 1.9905e+01),
            (4.2888e+00, 2.4114e+00, 2.8271e+01),
            (6.1096e+00, 4.7257e+00, 3.9914e+01),
            (8.7034e+00, 9.2038e+00, 5.5787e+01),
            (1.2398e+01, 1.7485e+01, 7.6626e+01)),
    "Al": ((6.1992e-02, 6.6397e-01, 9.4537e-03),
            (1.3196e-01, 5.1091e-02, 1.2422e+00),
            (2.8091e-01, 2.1871e-01, 3.2452e+00),
            (5.9303e-01, 1.1576e+00, 6.9063e+00),
            (1.2624e+00, 1.2839e+00, 1.2255e+01),
            (2.6873e+00, 3.6265e+00, 2.6516e+01),
            (5.7204e+00, 1.1299e+01, 5.1800e+01),
            (1.2177e+01, 2.8360e+01, 9.7682e+01),
            (2.5921e+01, 7.6316e+01, 1.7444e+02),
            (5.4722e+01, 1.6987e+02, 2.6481e+02),
            (1.1649e+02, 3.0004e+02, 3.7233e+02),
            (2.4797e+02, 4.7220e+02, 5.2302e+02)),
    "Cu": ((2.0664e-01, 1.1018e+00, 1.8654e+00),
            (2.9926e-01, 1.4041e+00, 1.6300e+00),
            (4.3340e-01, 1.3036e+00, 2.2223e+00),
            (6.2767e-01, 3.4872e-01, 3.4910e+00),
            (9.0900e-01, 2.4042e-01, 5.7930e+00),
            (1.3164e+00, 3.5714e-01, 8.7934e+00),
            (1.9462e+00, 7.1373e-01, 1.3209e+01),
            (2.8185e+00, 1.4185e+00, 1.9041e+01),
            (4.0818e+00, 2.4775e+00, 2.6983e+01),
            (5.9114e+00, 3.7308e+00, 3.8388e+01),
            (8.5611e+00, 6.2835e+00, 5.5245e+01),
            (1.2398e+01, 1.1985e+01, 7.9048e+01)),
    "Cr": ((2.4797e-01, 8.5637e-01, 2.0847e+00),
            (4.1004e-01, 1.8387e+00, 3.6256e+00),
            (6.7804e-01, 3.6802e+00, 4.3511e+00),
            (1.1150e+00, 4.4855e+00, 4.2849e+00),
            (1.8438e+00, 4.0742e+00, 5.6353e+00),
            (3.0490e+00, 3.5896e+00, 9.3608e+00),
            (5.0418e+00, 3.7733e+00, 1.6234e+01),
            (8.3371e+00, 5.8737e+00, 2.7472e+01),
            (1.3786e+01, 1.2134e+01, 4.4081e+01),
            (2.2671e+01, 2.5404e+01, 6.5421e+01),
            (3.7489e+01, 4.7584e+01, 8.9595e+01),
            (6.1992e+01, 7.7150e+01, 1.1506e+02)),
}


def test_the_library_matches_a_published_rakic_table():
    """n + i k from the library within 1e-3 of the published value over 12
    rows per metal. The library matches all 60 to 6.4e-5 (measured). The
    gate is tight and the rows span the band because a 10% typo in one
    parameter moves n + i k by much less than 10% except near that pole: at
    2% over 3 rows per metal, 51 of 81 such typos passed; here 77 of 81 fail
    (the four that pass move no row by 1e-3). Both n and k enter, so a
    conjugated convention (k < 0) fails too."""
    assert sorted(PUBLISHED) == sorted(fdtd.METALS)
    for metal, rows in PUBLISHED.items():
        for wavelength_um, n, k in rows:
            nk = np.sqrt(complex(fdtd.permittivity(
                metal, fdtd.C0 / (wavelength_um * 1e-6))))
            error = abs(nk - complex(n, k)) / abs(complex(n, k))
            assert error < 1e-3, (metal, wavelength_um, nk, (n, k))


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


def test_a_periodic_slab_is_metal_across_its_whole_cross_section():
    # On a periodic axis the E update runs to row n-1 (the copy of row 0). A
    # metal term bound with the wall ranges left that row vacuum, so every
    # periodic slab had a vacuum slot through it: E at j=1 differed from
    # j=0/j=3 by ~2.5% and V4 read ~0.03 low against Fresnel.
    t, n = 4, 30
    shape = (t, t, n)
    region = np.zeros(shape, dtype=bool)
    region[:, :, 12:18] = True
    term = fdtd.LorentzDrude("Au", region)
    source = fdtd.Source(kind="plane", direction="+z", component="x",
                         box=((0, 0, 8), (t - 1, t - 1, 21)),
                         frequency=3.6e14, bandwidth=1e14)
    spacings = (np.full(t - 1, 5e-9), np.full(t - 1, 5e-9),
                np.full(n - 1, 5e-9))
    run = fdtd.simulate(np.ones(shape, np.float32), source, 400,
                        [(1, j, 14, "x") for j in range(t)],
                        spacings=spacings,
                        boundary=("periodic", "periodic", "open"),
                        pml_cells=5, ade_terms=[term])
    peaks = [float(np.abs(trace).max()) for trace in run["traces"]]
    assert min(peaks) > 0, "the probes inside the slab saw nothing"
    assert (max(peaks) - min(peaks)) / max(peaks) < 1e-6, (
        f"E inside a periodic slab varies across it: {peaks}")


@pytest.mark.parametrize("periodic", [(False, False, False),
                                      (True, True, False)])
def test_a_metal_filling_the_grid_covers_exactly_the_samples_the_e_update_advances(
        periodic):
    """Each component's metal box is its E update range: half positions
    0..n-2 along its own axis, nodes 1..n-2 across a wall or open axis and
    1..n-1 across a periodic one. A sample beyond it is never curl-updated,
    so a wider box changes no field (no run can see it); this pins it."""
    shape = (7, 8, 9)                         # non-square
    eps = np.ones(shape, dtype=np.float32)
    term = fdtd.LorentzDrude("Au", np.ones(shape, dtype=bool))
    term.bind(shape=shape, dt=1e-18, xp=np, inv_eps=1 / (fdtd.EPS0 * eps),
              periodic=periodic)
    for axis, component in enumerate(fdtd.COMPONENTS):
        want = tuple((0, n - 1) if a == axis else
                     (1, n) if periodic[a] else (1, n - 1)
                     for a, n in enumerate(shape))
        box = term.parts[component]["box"]
        assert tuple((s.start, s.stop) for s in box) == want, component
        assert term.parts[component]["keep"].all(), component


def test_energy_can_be_recorded_with_a_metal_in_the_box():
    # A metal's increment is (where, values); unpacking it into the energy
    # sum's own ``region`` name made the second step's sum broadcast the whole
    # grid against the metal's sub-box and raise.
    shape = (10, 6, 6)
    region = np.zeros(shape, bool)
    region[5:7, :, :] = True
    source = fdtd.Source((3, 3, 2), frequency=4e14, bandwidth=3e14,
                         component="z")
    run = fdtd.simulate(np.ones(shape, np.float32), source, 40, dx=1e-8,
                        boundary="metal", record_energy=True,
                        ade_terms=[fdtd.LorentzDrude("Au", region)])
    energy = np.asarray(run["energy"])
    assert energy.shape == (40,) and np.all(np.isfinite(energy))
    assert energy[-1] > 0
