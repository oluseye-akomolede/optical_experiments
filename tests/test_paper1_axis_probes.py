"""Tier 1: the axial probe line, its price, and the profile read from it.

``scripts/paper1_nanowire.py --ringdown --save-tail PATH --axis-probes`` puts
one probe per E component on every z node of the wire's axis, and
``--axial-profile PATH`` reads their traces back out of the saved record as
|E(z)|^2. What this file checks is what the arm can get wrong without anything
else noticing: that a probe really lands on the node it is placed from (on the
layout's own path, not just the plan's), that the line starts and stops where
the geometry does and stays out of the absorber, that the profile finds a
standing wave's antinodes where the wavelength and phase put them, that it
reads only the drive-free tail it was told to, that the traces are priced in
the estimate the memory cap reads, and that the flags it cannot read are
refused before a scene is built.

Every grid here is tiny (the ``--tiny`` scene's dimensions), and no solve runs:
the plans build no voxels, the one layout voxelises a 20 nm grid, and the
profile tests read a record written by hand.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "paper1_nanowire.py"


@functools.lru_cache(maxsize=None)
def _load():
    """The script as a module (it puts the solver directory on the path)."""
    spec = importlib.util.spec_from_file_location("paper1_nanowire_axis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The tier-1 scene's own dimensions (``--tiny``'s defaults), spelled out so the
#: arm can be planned without ``--tiny`` (which refuses other flags).
TINY = ["--dx", "20", "--diameter", "60", "--height", "100", "--oxide", "40",
        "--gold", "40", "--mirror-pad", "40", "--qd-below", "60", "--pml", "6"]
AXIS = TINY + ["--profile", "pulse", "--ringdown", "--save-tail", "record.npz",
               "--axis-probes"]


def _lined():
    """The tiny plan and request carrying the axial line."""
    m = _load()
    args = m.parse_args(AXIS)
    plan, request, geom = m.planned_scene(args)
    return m, args, plan, request, geom


# --------------------------------------------------------------------------
# where the probes land
# --------------------------------------------------------------------------

def test_every_probe_is_located_on_the_node_it_is_placed_from():
    """The line is placed from the plan's node list, and the layout places the
    same points on the same nodes: every located probe's node coordinates are
    exactly the requested point, and its component is the one asked for."""
    m, args, plan, request, geom = _lined()
    points = request["probes"]
    assert points, "the line has no points"
    assert len(points) == len(plan["probes"])
    assert {point[3] for point in points} == set(m.AXIS_COMPONENTS)
    for point, (i, j, k, component) in zip(points, plan["probes"]):
        assert component == point[3]
        assert plan["nodes"][0][i] == point[0]
        assert plan["nodes"][1][j] == point[1]
        assert plan["nodes"][2][k] == point[2]
    # and the run's own layout -- the voxeliser path -- finds the same cells
    lay, lined_request, _geom = m.solve_scene(args)
    assert [tuple(probe) for probe in lay["probes"]] == \
        [tuple(probe) for probe in plan["probes"]]
    for point, (i, j, k, _c) in zip(lined_request["probes"], lay["probes"]):
        assert lay["nodes"][2][k] == point[2]


def test_the_line_runs_from_the_gold_top_to_three_cells_above_the_wire():
    """The first z is the first node at or above the gold's top face, the last
    is the last node at or below the wire top plus three coarse cells, and
    neither end is inside the absorbing layer."""
    m, args, plan, request, geom = _lined()
    nodes = plan["nodes"][2]
    zs = sorted({float(point[2]) for point in request["probes"]})
    gold_top = float(plan["part_bounds"][0][1][2])
    wire_lo, wire_hi = plan["part_bounds"][-1]
    dx = float(request["dx_m"])
    pml = int(plan["pml_cells"])
    first = int(np.searchsorted(nodes, gold_top + 1e-6 * dx, side="left"))
    stop = int(np.searchsorted(nodes, float(wire_hi[2]) + m.AXIS_PROBE_ABOVE_CELLS * dx,
                               side="right"))
    first, stop = max(first, pml), min(stop, len(nodes) - pml)
    assert zs == [float(v) for v in nodes[first:stop]]
    assert zs[0] > float(gold_top)
    assert zs[-1] <= float(wire_hi[2]) + m.AXIS_PROBE_ABOVE_CELLS * dx
    assert all(pml <= int(np.argmin(np.abs(nodes - z))) <= len(nodes) - 1 - pml
               for z in zs)


def test_the_line_is_the_four_thirds_of_the_wire_it_measures():
    """Three probes a layer, one per component, in the wire frame: the record's
    own reader needs a component per layer and no magnitude."""
    m, args, plan, request, geom = _lined()
    layers = len({float(point[2]) for point in request["probes"]})
    assert len(request["probes"]) == 3 * layers
    assert layers >= 4     # a wire span that a profile can be read on


# --------------------------------------------------------------------------
# the price the memory cap reads
# --------------------------------------------------------------------------

def test_a_run_pays_exactly_four_bytes_a_sample_a_probe():
    """Two estimates on the same nodes, with and without probes: their bytes
    differ by 4 * probes * steps, and nothing else moves -- probes cost memory,
    not time."""
    m, args, plan, request, geom = _lined()
    # the masks are not the subject here -- the two calls differ only in their
    # probe count, so the same (empty) scene is the same scene for both
    base = m.fdtd.estimate_run(plan["nodes"], duration=request["duration_s"],
                               boundary=request["boundary"], metals={}, pec=None)
    n = 7
    lined = m.fdtd.estimate_run(plan["nodes"], duration=request["duration_s"],
                                boundary=request["boundary"], metals={}, pec=None,
                                probes=n)
    assert lined["bytes"] - base["bytes"] == 4 * n * base["steps"]
    assert lined["dt"] == base["dt"]
    assert lined["steps"] == base["steps"]
    assert lined["cells"] == base["cells"]
    # a direct caller that gives probes but no step count would price them at
    # zero, which is a silent undercount: refused instead
    with pytest.raises(ValueError, match="probe_steps"):
        m.fdtd.memory_estimate((4, 4, 4), probes=3)


def test_the_layout_estimate_prices_the_line_it_planned():
    """``fdtd_layout.estimate`` on the lined plan is the unlined plan's bytes
    plus exactly the traces; this is the number the ladder prints and the caps
    read."""
    m, args, plan, request, geom = _lined()
    plain_args = m.parse_args([a for a in AXIS if a != "--axis-probes"])
    plain, plain_request, _geom = m.planned_scene(plain_args)
    assert [len(n) for n in plain["nodes"]] == [len(n) for n in plan["nodes"]]
    plain_est = m.fdtd_layout.estimate(plain, plain_request, record_frames=False)
    lined_est = m.fdtd_layout.estimate(plan, request, record_frames=False)
    assert lined_est["steps"] == plain_est["steps"]
    assert lined_est["bytes"] - plain_est["bytes"] == \
        4 * len(plan["probes"]) * lined_est["steps"]


# --------------------------------------------------------------------------
# reading the record
# --------------------------------------------------------------------------

WAVELENGTH_NM = 400.0
STEP_NM = 25.0
TAIL = 3000
START = 1000
DT = 1e-17


def _record(path, early=1.0, start=START, probes=True):
    """A standing wave on a 400 nm wavelength, sampled at the line's nodes.

    ``cos(k z + 0) cos(w t)``: antinodes at 0, 200, 400 and 600 nm, nodes at
    100, 300 and 500 -- the line's centre at 300 nm is a node, so a reader
    that reported the centre as a maximum would fail. ``early`` scales the
    part before ``start`` (garbage the tail reader must not use). x, y and z
    components carry the same shape with amplitudes 1, 2 and 0, so the summed
    power is 5 cos^2 and each component's square can be checked against it.
    """
    m = _load()
    zs = np.arange(0.0, 600.0 + STEP_NM, STEP_NM) * m.NM
    f = m.C / (WAVELENGTH_NM * m.NM)
    n_steps = START + TAIL
    t = np.arange(n_steps) * DT
    k = 2.0 * np.pi / (WAVELENGTH_NM * m.NM)
    amp = np.cos(k * zs)
    traces = np.zeros((3 * zs.size, n_steps), dtype=np.float32)
    components = []
    for layer, z in enumerate(zs):
        for row, factor in enumerate((1.0, 2.0, 0.0)):
            trace = factor * amp[layer] * np.cos(2.0 * np.pi * f * t)
            trace[:start] *= early
            traces[3 * layer + row] = trace
            components.append(m.AXIS_COMPONENTS[row])
    extra = ({"probe_traces": traces,
              "probe_z_m": np.repeat(zs, 3),
              "probe_components": np.array(components),
              "wire_z_m": np.array([zs[0], zs[-1]]),
              "emitter_z_m": float(zs[len(zs) // 2])} if probes else {})
    m.save_record(path, np.zeros(n_steps), DT, start,
                  {"frequency_hz": f, "bandwidth": 0.05 * f}, **extra)
    return path, f, zs


def test_a_standing_waves_antinodes_are_where_the_physics_puts_them(tmp_path):
    m = _load()
    path, f, zs = _record(tmp_path / "standing.npz")
    profile = m.axial_profile(path)
    assert profile["frequency_hz"] == pytest.approx(f)
    assert profile["maxima_z_m"] == pytest.approx([200e-9, 400e-9],
                                                  abs=STEP_NM * m.NM / 2)
    total = profile["total"]
    peak = np.max(total)
    centre = int(np.argmin(np.abs(zs - 300e-9)))
    assert total[centre] / peak < 1e-3          # a node, not a maximum
    # components are kept apart and squared: y is twice x in amplitude, so
    # four times the power, and the sum is x + y + z
    assert np.allclose(profile["power"]["y"], 4.0 * profile["power"]["x"])
    assert np.allclose(profile["power"]["z"], 0.0, atol=peak * 1e-6)
    assert np.allclose(total, sum(profile["power"].values()))


def test_the_profile_reads_the_tail_it_was_told_to(tmp_path):
    """The early part of the record is 50x the tail; a reader that used it
    would report a power two orders larger. The record's own ``start`` and an
    explicit one give the same profile, and a ``start`` past the end refuses."""
    m = _load()
    path, f, zs = _record(tmp_path / "early.npz", early=50.0)
    own = m.axial_profile(path)
    explicit = m.axial_profile(path, start=START)
    assert np.allclose(own["total"], explicit["total"])
    whole = m.axial_profile(path, start=0)
    assert np.max(whole["total"]) / np.max(own["total"]) > 10.0
    assert whole["maxima_z_m"] == pytest.approx([200e-9, 400e-9],
                                                abs=STEP_NM * m.NM / 2)
    later = m.axial_profile(path, start=START + 1000)
    assert later["tail_start"] == START + 1000
    assert later["maxima_z_m"] == pytest.approx([200e-9, 400e-9],
                                                abs=STEP_NM * m.NM / 2)
    with pytest.raises(ValueError, match="no decayed part"):
        m.axial_profile(path, start=START + TAIL)


def test_a_record_without_probe_traces_is_refused_naming_them(tmp_path):
    m = _load()
    path, f, zs = _record(tmp_path / "bare.npz", probes=False)
    with pytest.raises(ValueError, match="no probe traces"):
        m.axial_profile(path)
    assert m.main(["--axial-profile", str(path)]) == 2


def test_the_profile_arm_prints_the_record_it_reads(tmp_path, capsys):
    m = _load()
    path, f, zs = _record(tmp_path / "table.npz")
    assert m.main(["--axial-profile", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "25 z layers, 3 components" in printed   # this fixture, not Run C's
    assert "maxima" in printed
    assert "200.00" in printed and "400.00" in printed


# --------------------------------------------------------------------------
# the flags the arm cannot read
# --------------------------------------------------------------------------

def _refusal(argv):
    m = _load()
    return m.arm_conflicts(m.parse_args(argv))


def test_axis_probes_without_a_tail_record_is_refused():
    assert "--ringdown" in _refusal(TINY + ["--axis-probes"])
    assert "--save-tail" in _refusal(
        TINY + ["--profile", "pulse", "--ringdown", "--axis-probes"])


def test_the_ladder_can_price_a_line_it_never_records():
    """`--estimate` builds no scene to record, so the same flag that needs a
    tail record on the run path may stand alone to price the line."""
    m = _load()
    assert _refusal(TINY + ["--estimate", "--axis-probes"]) == ""
    plan, request, geom = m.planned_scene(m.parse_args(
        TINY + ["--estimate", "--axis-probes"]))
    assert request["probes"]
    assert len(plan["probes"]) == len(request["probes"])


def test_the_profile_arm_is_refused_with_every_arm_that_builds_a_scene():
    for flag in (["--ringdown"], ["--estimate"], ["--tiny"],
                 ["--save-series", "x.npz"], ["--from-series", "x.npz"],
                 ["--collection-na", "0.8"], ["--locator-tau", "10"],
                 ["--axis-probes"]):
        why = _refusal(["--axial-profile", "r.npz"] + flag)
        assert "--axial-profile" in why and flag[0] in why, (flag, why)


def test_the_readers_own_flags_need_the_reader():
    assert "--axial-profile" in _refusal(["--axial-frequency-nm", "900"])
    assert "--axial-profile" in _refusal(["--axial-png", "p.png"])
    assert _refusal(["--axial-profile", "r.npz",
                     "--axial-frequency-nm", "900",
                     "--axial-png", "p.png"]) == ""
