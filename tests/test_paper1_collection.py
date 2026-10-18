"""Tier 1: paper 1's collection arm (``--collection-na``), on tiny layouts.

``scripts/paper1_nanowire.py --collection-na 0.8`` adds a phasor box to the CW
arm and reads the fraction of the upward far-field power inside NA 0.8 from it
(``farfield.collection_fraction``). The box goes around the WHOLE scene -- the
wire, the oxide and the entire gold pad -- with every face in the layout's
open vacuum pad, so the free-space transform is exact for the scene as built
(openspec change ``paper1-collection``; ``tests/test_phasor_mirror.py``
measures why that beats a PEC image or a box in air).

What this file checks is what the arm can get wrong without anything else
noticing: where the box lands on a real (graded) layout and that it refuses a
layout with no room for it; that a face through material is refused; the
arm's refusals; one real run in vacuum, where the answer is the free x
dipole's closed form; that without the flag the run is the run it was; and
that the box is priced into the estimate and the ladder the caps read.

Every grid here is tiny: the plans build no voxels, and the one solve is
37 x 37 x 42 nodes (about 5 s at ``OMP_NUM_THREADS=2``).
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "paper1_nanowire.py"


@functools.lru_cache(maxsize=None)
def _load():
    """The script as a module (it puts the solver directory on the path)."""
    spec = importlib.util.spec_from_file_location("paper1_nanowire", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The tier-1 scene's own dimensions (``--tiny``'s defaults), spelled out so a
#: run of the arm can use them: ``--tiny`` itself refuses the flag.
TINY = ["--dx", "20", "--diameter", "60", "--height", "100", "--oxide", "40",
        "--gold", "40", "--mirror-pad", "40", "--qd-below", "60", "--pml", "6"]
#: The vacuum run: the tiny scene with its open pad held at 240 nm (12 cells)
#: and a 16-period record read over its last 4, so the solve stays small.
VACUUM = TINY + ["--pad-nm", "240", "--periods", "16", "--window-periods", "4",
                 "--vacuum"]
NA = 0.8


def _free_x_dipole_fraction(na: float) -> float:
    """The closed form: an x dipole's U ~ 1 - sin^2 t cos^2 p, whose phi mean
    is (1 + cos^2 t) / 2, so the cone to cos t = c holds
    ((1 - c) + (1 - c^3) / 3) / 2 of it; NA 0.8 gives 0.49600."""
    c = math.sqrt(1.0 - na ** 2)
    return ((1.0 - c) + (1.0 - c ** 3) / 3.0) / (1.0 + 1.0 / 3.0)


def _plan(argv):
    m = _load()
    args = m.parse_args(argv)
    plan, request, geom = m.planned_scene(args)
    return m, args, plan, geom


# --------------------------------------------------------------------------
# where the box goes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [["--tiny"], ["--tiny", "--pad-nm", "160"], [],
                                  ["--dx", "20", "--mirror-pad", "400"]],
                         ids=["tiny", "tight", "paper", "run-1"])
def test_the_box_encloses_the_whole_scene_inside_the_open_pad(argv):
    """On the tiny plan, the paper's default plan and run 1's (dx 20, pad
    400), whose pads are graded, and the tiny one with its pad cut to 160 nm,
    where the top faces have exactly the 2 layers of room: the parts' box the arm places from is the
    plan's own part bounds; every face is at least ``COLLECTION_CLEAR_CELLS``
    node layers outside the parts and inside the layers ``simulate`` allows
    beside the absorber (``fdtd.phasor_box_layers`` accepts the box and spells
    it back as the same layers); each face sits within a layer of the middle
    of its free gap. Measured on the paper's plan (212 x 213 x 203 nodes):
    faces 13..197, 13..199, 13..189, 3 to 4 layers clear of the parts and of
    the absorber."""
    m, args, plan, geom = _plan(argv)
    lo_m, hi_m = m.scene_extent(geom)
    bounds = plan["part_bounds"]
    np.testing.assert_allclose(lo_m, np.min([b[0] for b in bounds], axis=0), atol=1e-15)
    np.testing.assert_allclose(hi_m, np.max([b[1] for b in bounds], axis=0), atol=1e-15)
    pml = geom["pml_cells"]
    placed = m.collection_box(plan["nodes"], geom, pml)
    lo, hi = m.fdtd.phasor_box_layers(placed["box"], plan["shape"], pml)
    assert tuple(lo) == placed["lo"] and tuple(hi) == placed["hi"]
    for a in range(3):
        z = np.asarray(plan["nodes"][a])
        n = z.size
        first = int(np.flatnonzero(z <= lo_m[a] + 1e-15)[-1])
        last = int(np.flatnonzero(z >= hi_m[a] - 1e-15)[0])
        assert first - lo[a] >= m.COLLECTION_CLEAR_CELLS, (a, first, lo[a])
        assert hi[a] - last >= m.COLLECTION_CLEAR_CELLS, (a, last, hi[a])
        assert z[lo[a]] < lo_m[a] and z[hi[a]] > hi_m[a]
        assert pml + 1 <= lo[a] and hi[a] <= n - 2 - pml
        assert abs(lo[a] - 0.5 * (pml + 1 + first - m.COLLECTION_CLEAR_CELLS)) <= 1.0
        assert abs(hi[a] - 0.5 * (last + m.COLLECTION_CLEAR_CELLS + n - 2 - pml)) <= 1.0
        assert placed["clear"][a] == (first - lo[a], hi[a] - last)


def test_a_layout_with_no_room_for_the_box_is_refused():
    """The tiny scene with its pad cut to 140 nm (7 cells at dx 20, graded
    finer towards the parts) beside a 6-cell absorber leaves no z layer that
    is 2 outside the parts and clear of the absorber: the placement refuses,
    naming the axis and the layers it had. (160 nm still fits: the grading
    puts 3 layers there.)"""
    m, args, plan, geom = _plan(["--tiny", "--pad-nm", "140"])
    with pytest.raises(m.fdtd.FDTDError, match="no room on axis z: the parts span"
                                                " node layers 10..29 of 38"):
        m.collection_box(plan["nodes"], geom, geom["pml_cells"])


def test_the_ladder_refuses_a_rung_with_no_room(capsys):
    """The same pad through the ladder: the three dx-20 rungs (the asked one
    included) print the refusal instead of a price, and the finer rungs,
    whose 140 nm holds more cells, are priced."""
    m = _load()
    code = m.main(["--tiny", "--estimate", "--collection-na", "0.8", "--pad-nm", "140"])
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith(" 20.0")]
    assert out.count("fits (") == 6, out
    assert len(rows) == 3 and all("refused" in row
                                  and "the collection box has no room" in row
                                  for row in rows), out
    assert code == 0, out


def test_a_face_through_material_is_refused():
    """With the voxelised permittivity and metals in hand the box checks its
    faces' own slabs (the face layer and one each side, across the face's
    extent plus one): a dielectric node or a metal node there is refused, a
    vacuum shell passes."""
    m, args, plan, geom = _plan(["--tiny"])
    pml = geom["pml_cells"]
    placed = m.collection_box(plan["nodes"], geom, pml)
    lo, hi = placed["lo"], placed["hi"]
    centre = placed["box"][0]
    eps = np.ones(plan["shape"])
    m.collection_box(plan["nodes"], geom, pml, eps=eps, metals={})
    eps[lo[0], centre[1], centre[2]] = 2.0
    with pytest.raises(m.fdtd.FDTDError, match="node layer .* on axis x touches"):
        m.collection_box(plan["nodes"], geom, pml, eps=eps, metals={})
    eps[...] = 1.0
    gold = np.zeros(plan["shape"], dtype=bool)
    gold[centre[0], centre[1], hi[2] + 1] = True
    with pytest.raises(m.fdtd.FDTDError, match="axis z touches material"):
        m.collection_box(plan["nodes"], geom, pml, eps=eps, metals={"Au": gold})


# --------------------------------------------------------------------------
# the arm's refusals
# --------------------------------------------------------------------------

def test_the_arm_refuses_what_it_cannot_read():
    """Not an aperture in air (0, 1.2); a pulse record (--spectrum,
    --ringdown, --profile pulse: no one frequency to read at); --tiny without
    --estimate (its controls would leave the flag inert). The CW arm and the
    tiny ladder accept it, and without the flag nothing new is refused."""
    m = _load()

    def why(*argv):
        return m.arm_conflicts(m.parse_args(list(argv)))

    for na in ("0", "1.2", "-0.5"):
        assert "is not a numerical aperture" in why("--collection-na", na)
    for arm in (["--spectrum"], ["--ringdown"], ["--profile", "pulse"]):
        assert "reads the CW arm's settled steady state" in why(
            *arm, "--collection-na", "0.8")
    assert "with --tiny" in why("--tiny", "--collection-na", "0.8")
    assert why("--tiny", "--estimate", "--collection-na", "0.8") == ""
    assert why("--collection-na", "0.8") == ""
    assert why("--collection-na", "1") == ""
    for arm in ([], ["--spectrum"], ["--ringdown"], ["--tiny"]):
        assert why(*arm) == ""


# --------------------------------------------------------------------------
# one real run: the vacuum scene, where the answer is a closed form
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _vacuum_run():
    """The arm through ``main``, in vacuum: (exit code, printed lines)."""
    import contextlib
    import io
    m = _load()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = m.main(VACUUM + ["--collection-na", str(NA)])
    return code, buffer.getvalue()


def _number(pattern, text):
    found = re.search(pattern, text, re.M)
    assert found, text
    return [float(g) for g in found.groups()]


def test_the_arm_reads_the_free_dipole_in_vacuum():
    """The tiny scene with every material stripped (37 x 37 x 42 nodes, dx 20
    nm, 900 nm, an x dipole, 16 periods read over the last 4): the box around
    the (empty) scene must read an x dipole in free space. Measured: NA 0.8
    fraction 0.4960 against the closed form 0.49600 (tolerance 0.003); 0.4974
    of the surface power upward (tolerance 0.02 of a half); the surface power
    0.9894 of the delivered power and the full-sphere far field 0.9947 of the
    surface power (tolerance 2.5% each: the 12-cell pad keeps the absorber's
    echo in the record). The box line names the layers it recorded."""
    code, out = _vacuum_run()
    assert code == 0, out
    assert re.search(r"^box\s+phasor faces at node layers x 9\.\.27, y 9\.\.27,"
                     r" z 9\.\.31: the wire, the oxide and the whole 148 nm mirror"
                     r" inside, all six faces in vacuum", out, re.M), out
    fraction, cone, upward, of_all = _number(
        r"^collect\s+NA 0\.80: ([\d.]+) of the upward far-field power \(cone"
        r" ([\d.e+-]+) W of ([\d.e+-]+) W upward\), ([\d.]+) of all", out)
    up, total, share = _number(r"^surface\s+upward ([\d.e+-]+) W, total ([\d.e+-]+)"
                               r" W through the box \(([\d.]+) of it upward\)", out)
    delivered_ratio, closure = _number(
        r"^sanity\s+box surface power / delivered power = ([\d.]+) .*full-sphere"
        r" far field / surface power = ([\d.]+)$", out)
    print(f"\nfraction {fraction:.4f} (closed form {_free_x_dipole_fraction(NA):.5f}),"
          f" upward share {share:.4f}, surface / delivered {delivered_ratio:.4f},"
          f" sphere / surface {closure:.4f}")
    assert abs(fraction - _free_x_dipole_fraction(NA)) < 0.003
    assert fraction == pytest.approx(cone / upward, abs=1e-4)
    assert up == upward
    assert abs(share - 0.5) < 0.02
    # cone / sphere is the fraction times the upward share over the closure
    assert of_all == pytest.approx(fraction * share / closure, abs=2e-3)
    assert abs(delivered_ratio - 1.0) < 0.025
    assert abs(closure - 1.0) < 0.025
    # the CW arm's own reading is still there, and still F = 1 in vacuum
    (F,) = _number(r"^measured\s+F = ([\d.]+) at 900 nm", out)
    assert abs(F - 1.0) < 0.05


# --------------------------------------------------------------------------
# off by default
# --------------------------------------------------------------------------

#: The keywords the CW arm handed ``fdtd.simulate`` before this arm existed.
PLAIN_SIMULATE = {"spacings", "boundary", "pec", "pml_cells", "ade_terms", "xp",
                  "record_drive", "record_source_field"}
#: Lines only the arm prints, and the run's wall-clock line.
ARM_LINES = ("box ", "collect ", "surface ", "sanity ")


def _fake_main(m, monkeypatch, argv):
    """``main`` with the solve replaced by a canned CW record (and faces).

    The record is a 900 nm tone, so every reading downstream is a number; the
    faces, when asked for, are an x dipole's on a 400 nm box. Returns the
    printed lines (minus the wall-clock one), what ``simulate`` was handed,
    and what ``estimate_run`` was handed."""
    import farfield
    calls = {"simulate": [], "estimate": []}
    real_estimate = m.fdtd.estimate_run

    def estimate_run(nodes, **kwargs):
        calls["estimate"].append(kwargs)
        return real_estimate(nodes, **kwargs)

    def simulate(eps, source, n_steps, probes, **kwargs):
        calls["simulate"].append(kwargs)
        dt = m.fdtd.courant_dt(*(float(np.min(s)) for s in kwargs["spacings"]))
        t = np.arange(n_steps) * dt
        w = 2.0 * math.pi * source.frequency
        faces = None
        if "phasor_box" in kwargs:
            faces = [farfield.sample_box(
                [-2e-7] * 3, [2e-7] * 3, (8, 8, 8),
                lambda p: farfield.hertzian_dipole(p, [1e-30, 0, 0], [0, 0, 0],
                                                   source.frequency))]
        return {"drive": np.cos(w * t), "source_field": -1e-3 * np.sin(w * t),
                "source_eps": 1.0, "phasor_faces": faces}

    import contextlib
    import io
    buffer = io.StringIO()
    with monkeypatch.context() as patch, contextlib.redirect_stdout(buffer):
        patch.setattr(m.fdtd, "simulate", simulate)
        patch.setattr(m.fdtd, "estimate_run", estimate_run)
        code = m.main(argv)
    lines = [line for line in buffer.getvalue().splitlines()
             if not line.startswith("run ")]
    return code, lines, calls


def test_without_the_flag_the_run_is_the_one_it_was(monkeypatch):
    """The same tiny vacuum scene through ``main`` with the solve faked, with
    and without ``--collection-na``. Without it: ``simulate`` is handed exactly
    the keywords it was before the arm existed (no phasor key at all), no
    ``estimate_run`` call carries a phasor keyword, and the arm prints nothing.
    With it, every one of those lines is still printed in the same order and
    unchanged -- the arm only adds its own lines (the memory line too is the
    same at this size) -- and ``simulate`` gets the box over the CW reading's
    own tail."""
    m = _load()
    code_off, off, calls_off = _fake_main(m, monkeypatch, VACUUM)
    code_on, on, calls_on = _fake_main(m, monkeypatch, VACUUM + ["--collection-na", "0.8"])
    assert code_off == code_on == 0, "\n".join(off + on)
    assert set(calls_off["simulate"][0]) == PLAIN_SIMULATE
    assert not any(key.startswith("phasor") for call in calls_off["estimate"]
                   for key in call)
    assert not any(line.startswith(ARM_LINES) for line in off)
    assert [line for line in on if not line.startswith(ARM_LINES)] == off
    on_kwargs = calls_on["simulate"][0]
    assert set(on_kwargs) == PLAIN_SIMULATE | {"phasor_box", "phasor_frequencies",
                                               "phasor_start", "phasor_window"}
    text = "\n".join(on)
    steps = int(re.search(r"^record\s+([\d,]+) steps", text, re.M).group(1).replace(",", ""))
    tail = int(re.search(r"read over the last ([\d,]+) steps$", text, re.M)
               .group(1).replace(",", ""))
    assert on_kwargs["phasor_start"] == steps - tail
    assert on_kwargs["phasor_window"] == "hann"
    assert any(line.startswith("collect   NA 0.80: 0.4960") for line in on), on


def test_the_box_is_priced_into_the_estimate_the_caps_read():
    """``filled`` with the flag places the box on the layout and its estimate
    rises by exactly ``fdtd.phasor_box_bytes`` of that box over the CW tail
    it reads (``window_periods`` of steps, the delivered power's own
    rounding); without the flag there is no plan and the estimate is the
    plain one. Uses the vacuum layout (no gold refinement, about 1 s)."""
    m = _load()
    args = m.parse_args(VACUUM + ["--collection-na", "0.8"])
    lay, request, geom = m.solve_scene(args)
    on = m.filled(args, lay, dict(request), geom)
    plain = m.parse_args(VACUUM)
    off = m.filled(plain, lay, dict(request), geom)
    assert "phasor" not in off
    phasor = on["phasor"]
    dt = on["est"]["dt"]
    tail = int(round(int(args.window_periods) / (request["frequency_hz"] * dt)))
    assert phasor["steps"] == tail and phasor["start"] == on["steps"] - tail
    assert phasor["box"] == m.collection_box(lay["nodes"], geom, geom["pml_cells"])["box"]
    added = m.fdtd.phasor_box_bytes(phasor["box"], on["est"]["shape"],
                                    frequencies=[request["frequency_hz"]], steps=tail)
    assert on["est"]["bytes"] - off["est"]["bytes"] == added > 0
    assert on["steps"] == off["steps"] and on["exceeded"] == off["exceeded"]
    with pytest.raises(ValueError, match="pass geom"):
        m.filled(args, lay, dict(request))


def test_the_ladder_prices_the_box_on_every_rung(monkeypatch):
    """The ladder is plans only, so it is run at the paper's own dimensions:
    every rung's estimate reaches ``check_limits`` larger by exactly the
    box's ``phasor_box_bytes`` on that rung's plan with the flag, and as it
    was without it (8 rungs, the refused dx-10 / 600 one included)."""
    m = _load()
    seen = {"on": [], "off": []}
    real = m.fdtd.check_limits

    def record(into):
        def check(est, **kwargs):
            seen[into].append(est["bytes"])
            return real(est, **kwargs)
        return check

    expected = []
    for dx, pad in [(dx, pad) for dx in (20.0, 15.0, 12.0, 10.0) for pad in (400.0, 600.0)]:
        args = m.parse_args(["--dx", str(dx), "--mirror-pad", str(pad),
                             "--collection-na", "0.8"])
        plan, request, geom = m.planned_scene(args)
        est = m.fdtd_layout.estimate(plan, request, record_frames=False)
        request["duration_s"] = m.steps_for(args, request, est["dt"]) * est["dt"]
        est = m.fdtd_layout.estimate(plan, request, record_frames=False)
        box = m.collection_box(plan["nodes"], geom, geom["pml_cells"])["box"]
        expected.append(m.fdtd.phasor_box_bytes(
            box, est["shape"], frequencies=1, steps=m.tail_steps(args, request, est["dt"])))
    for into, argv in (("off", ["--estimate"]),
                       ("on", ["--estimate", "--collection-na", "0.8"])):
        monkeypatch.setattr(m.fdtd, "check_limits", record(into))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            m.main(argv)
    assert len(seen["on"]) == len(seen["off"]) == 8
    assert [a - b for a, b in zip(seen["on"], seen["off"])] == expected
