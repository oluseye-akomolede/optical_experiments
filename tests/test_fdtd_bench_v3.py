"""The V3 bench's judgement, without running the solver.

The bench's two gates read the solver's running |E_c| maximum through two
masks and turn the result into a verdict; the run itself is the lead's
Jetson job. Every fixture is non-square, so a mask that swapped axes, or
used one Yee rule for every component, would not line up by accident.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_bench_v3", ROOT / "scripts" / "fdtd_bench_v3.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SHAPE = (14, 16, 18)
PML = 2
BOX = ((5, 6, 7), (9, 10, 11))


def quiet_peak(incident=1.0):
    """A run whose wave stayed in the box: E_z is ``incident`` inside it."""
    peak = {c: np.zeros(SHAPE, np.float32) for c in "xyz"}
    peak["z"][bench.incident_mask(SHAPE, BOX, "z")] = incident
    return peak


def test_a_component_is_inside_on_its_own_half_rule():
    """E_x sits at half indices along x, at nodes along y and z: the sample
    stored at index high along x is at high+1/2, outside; along y it is at
    the node high, inside."""
    (lx, ly, lz), (hx, hy, hz) = BOX
    mask = bench.scattered_mask(SHAPE, BOX, PML, "x")
    assert not mask[hx - 1, ly, lz]          # x = high - 1/2: inside
    assert mask[hx, ly, lz]                  # x = high + 1/2: outside
    assert not mask[lx, hy, lz]              # y = high node: inside
    assert mask[lx, hy + 1, lz]
    assert not mask[lx, ly, hz] and mask[lx, ly, hz + 1]
    z_mask = bench.scattered_mask(SHAPE, BOX, PML, "z")
    assert not z_mask[hx, ly, lz] and z_mask[lx, ly, hz]


def test_the_two_masks_split_the_physical_region_and_skip_the_cpml():
    physical = np.zeros(SHAPE, bool)
    physical[tuple(slice(PML, n - PML) for n in SHAPE)] = True
    for component in "xyz":
        out = bench.scattered_mask(SHAPE, BOX, PML, component)
        inside = bench.incident_mask(SHAPE, BOX, component)
        assert not (out & inside).any()
        assert ((out | inside) == physical).all(), component


def test_a_sample_outside_the_box_is_a_leak_and_one_in_the_cpml_is_not():
    peak = quiet_peak()
    incident, leak, _ = bench.measure(peak, [0.0, 1.0, 0.0], BOX, PML, "z")
    assert incident == 1.0 and leak == 0.0
    peak["y"][0, 0, 0] = 0.5                 # in the CPML: not judged
    assert bench.measure(peak, [1.0], BOX, PML, "z")[1] == 0.0
    peak["y"][PML, 8, 9] = 2e-3              # physical, outside the box
    _, leak, _ = bench.measure(peak, [1.0], BOX, PML, "z")
    assert leak == np.float32(2e-3)
    passed, failures = bench.v3_verdict([("+x:z", incident, leak, 0.0)])
    assert not passed and "scattered-field region" in failures[0]
    assert bench.v3_verdict([("+x:z", 1.0, 5e-4, 0.0)])[0]


def test_the_residual_is_the_last_energy_over_the_peak():
    _, _, residual = bench.measure(quiet_peak(), [0.0, 2.0, 1.0, 1e-4],
                                   BOX, PML, "z")
    assert residual == 5e-5
    assert bench.v3_verdict([("+x:z", 1.0, 0.0, residual)])[0]
    passed, failures = bench.v3_verdict([("+x:z", 1.0, 0.0, 2e-4)])
    assert not passed and "peak energy" in failures[0]


def test_a_run_with_no_wave_or_a_nan_fails():
    """No wave leaks nothing and leaves no energy, so it would pass both
    gates; it must fail. So must NaN, which compares false with any bound."""
    incident, leak, residual = bench.measure(quiet_peak(0.0), [0.0, 0.0],
                                             BOX, PML, "z")
    passed, failures = bench.v3_verdict([("+x:z", incident, leak, residual)])
    assert not passed and "no incident wave" in failures[0]
    assert not bench.v3_verdict([("+x:z", 1.0, math.nan, 0.0)])[0]
    assert not bench.v3_verdict([("+x:z", 1.0, 0.0, math.nan)])[0]
    assert not bench.v3_verdict([])[0]


def test_the_default_pairs_are_the_twelve_transverse_ones():
    assert len(bench.ALL_PAIRS) == 12
    for pair in bench.ALL_PAIRS:
        direction, component = pair.split(":")
        assert component != direction[1]
