"""Tier 1: paper 1's bench script -- the scene's controls and its refusals.

``scripts/paper1_nanowire.py`` is the scene the step-1 measurement arm is pointed
at: a hexagonal InP wire on a 12 nm SiO2 buffer over a gold mirror, a dipole on
the axis, open boundaries. What this file checks is what the script can get
wrong without anything else noticing -- that the layout really builds the stack
it prints (the buffer resolved to its own cells, the gold out of the absorber,
the source in InP and clear of the facet), that a homogeneous scene reads F = 1
through the script's own path, and that the refusals fire.

The controls are checked both ways. Each one is a claim about the scene, so a
fixture that can only pass measures nothing: ``--qd-below`` breaks the facet
control and the prism control is run against a mesh with a face removed, so
both are seen red before they are trusted green.

The refusals are the product's own, reached through ``tiny``: a record shorter
than its tail, a record over ``fdtd.MAX_STEPS``, and a record too short for the
hard-started CW drive to have settled (8 periods is F = 4.61 in vacuum, and the
run says so).

The three per-run ceilings are covered both ways as well: the product's own refuse
a long record, and the bench flags that raise them (``--max-steps``,
``--max-seconds`` and ``--max-cells``, the ceilings only a flagged bench run may
raise) are read by the ladder and the run path through one function and refused
if they would lower a ceiling instead.

The open boundary is covered as a distance (``--pad-nm``/``--pml-nm``), because
``fdtd_layout.OPEN_PAD_CELLS`` is a cell count: 16 cells is 320 nm at dx 20 --
what r20 ran -- and 160 nm at dx 10, so a dx-10 record on the product's own pad
would move the boundary and the resolution at once. The knob's own control is
the coarse rung: at dx 20 the same pair is the product's own 16/10 cells and the
same box, which is what makes the dx-10 record comparable to r20. The tiny run
then holds a named boundary end to end -- the layout builds with it and the
absorber reaches the solver -- and the refusals that keep the distance honest
(a pad under one cell, ``--pml`` and ``--pml-nm`` disagreeing, an absorber as
thick as the pad) are read both directly and through the script's own entry.

The estimate ladder is covered at the tiny dims too -- there it only builds plans
and estimates, never voxelises or solves, so it stays tier 1 -- which is how a
ladder with every rung refused is seen to report that rather than read the last
rung it happened to try.

The tail arm -- a pulse record whose drive is over partway through, fitted to one
decaying sinusoid -- is covered three ways: the fit itself against synthetic
tails with a known tau (and against one with no decay at all, so a fit that
invents a decay is caught), the window it cuts at read against the product's own
pulse envelope, and the refusals a band arm and a too-short span owe it.

The tiny scene is deliberately small -- 20 nm cells, a 100 nm wire -- so it runs
in tier 1. It exercises the same code path the Jetson run does, not the same
physics.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "paper1_nanowire.py"


def _load():
    """The script as a module, so its mesh helpers can be called directly."""
    spec = importlib.util.spec_from_file_location("paper1_nanowire", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*args: str, timeout: float = 900.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT)


def test_the_tiny_scene_holds_every_control():
    done = run("--tiny")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tiny: all controls hold" in done.stdout
    assert "FAIL" not in done.stdout
    # 21 controls: a scene that quietly loses one still has to be caught here. The
    # three past 15 are the fit's window following the drive's band (2e), the saved
    # tail re-fitting to the line it came from (2f), and an unwritable tail path
    # raising the one exception the arm catches (2h); the fourth is --save-tail
    # refused on the arms that have no tail to write (2i); the last two are the
    # emitter's own z being a node at two dx values, and the refusal that fires
    # when a mesh does not put it there (2j).
    assert done.stdout.count("\n  ok   ") == 21, done.stdout
    # and the settled record the small scene runs must not warn about itself
    assert "warning: the record is" not in done.stdout
    # the step loop reports itself, so a run the bench kills leaves more than its
    # plan behind: the step it reached, the seconds, and the rate it was given
    assert re.search(r"^progress  step 1,000 of [\d,]+,\s+[\d.]+ s,"
                     r"\s+[\d.]+ Mcell-steps/s$", done.stdout, re.M), done.stdout


def test_the_tiny_scene_reads_one_across_the_band_in_vacuum():
    """The band arm, through the script's own path, on a homogeneous scene."""
    done = run("--tiny", "--spectrum")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tiny: all controls hold" in done.stdout
    # Every bin, not only the peak: the shape string cannot say how far above 1
    # the band's floor sits, and a peak on the band's edge is itself a reading.
    bins = re.findall(r"^bin\s+(\d+\.\d) nm\s+F =\s+([\d.]+)$", done.stdout, re.M)
    assert len(bins) == 6, done.stdout
    peak = re.search(r"^peak\s+F = ([\d.]+) at ([\d.]+) nm$", done.stdout, re.M)
    assert peak, done.stdout
    values = {wavelength: float(value) for wavelength, value in bins}
    # the peak line is one of the bins, not its own reading of the record
    assert float(peak.group(2)) in [float(w) for w, _ in bins], done.stdout
    assert float(peak.group(1)) == max(values.values()), done.stdout
    # the reading and its own prediction, both on the record of the run
    assert re.search(r"^run\s+\d+\.\d s wall\s+\(\d+\.\d\d x the"
                     r" \d+\.\d s estimate\)$", done.stdout, re.M), done.stdout
    # the dipole line says where the node is and what was asked for it
    dipole = re.search(r"^dipole\s+node \((\d+), (\d+), (\d+)\): z\s+([\d.]+) nm,"
                       r"\s+([\d.]+) nm below the top facet at ([\d.]+) nm"
                       r"\s+\(--qd-below (\S+) asked\); in plane\s+([+\-][\d.]+),"
                       r"\s+([+\-][\d.]+) nm off the wire's axis, (\w)-polarised$",
                       done.stdout, re.M)
    assert dipole, done.stdout
    node_z, depth, top = (float(dipole.group(4)), float(dipole.group(5)),
                          float(dipole.group(6)))
    # the line's own arithmetic: the printed depth is the two printed z values
    assert abs((top - node_z) - depth) < 1e-9, done.stdout
    # 2j pins the source's own z as a grid node, so the printed depth is not
    # merely inside a cell of the request, as it was while this read only
    # "within one finest cell". At the 2i revision this same line read 57.11 nm
    # against the 60 nm asked (node (31, 31, 34), its own log): 2.89 nm, inside
    # that grid's 6.218 nm finest cell, so the weak rule passed it -- and 2.89 nm
    # outside the 0.005 below, so this one fails it. Group(7) is the
    # `--qd-below` field the line itself prints, so the two numbers compared
    # are the line's own rather than a 60 retyped here.
    assert abs(depth - float(dipole.group(7))) < 0.005, done.stdout


def test_every_rung_over_the_cap_is_reported_not_skipped():
    """A ladder with no rung left must say so, not read the last one it tried.

    The tiny dims make the rungs cheap (they only build plans and estimate, never
    voxelise or solve) and a 4,000-period record puts every rung over the step cap.
    """
    done = run("--tiny", "--estimate", "--periods", "4000")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "every rung above was refused" in done.stdout
    assert done.stdout.count("refused:") == 8, done.stdout


def test_the_source_control_can_fail():
    """--qd-below 25 nm leaves the dipole under the two cells the control asks."""
    done = run("--tiny", "--qd-below", "25")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "FAIL" in done.stdout
    assert "clear of the facet" in done.stdout


def test_the_mesh_control_can_fail():
    """The prism control's discriminating power, on a mesh with a face removed."""
    module = _load()
    good = module._hex_prism(0.0, 1.0, 100.0, 0.0)
    broken = trimesh.Trimesh(vertices=good.vertices, faces=good.faces[:-1],
                             process=False)
    exact = 3.0 * module.math.sqrt(3.0) / 2.0 * 100.0 ** 2
    assert good.is_watertight and not broken.is_watertight
    assert abs(good.volume - exact) / exact < 1e-9
    assert abs(broken.volume - exact) / exact > 0.1


def test_the_record_shorter_than_its_window_is_refused():
    """A record read over a longer tail than it holds cannot be averaged."""
    done = run("--tiny", "--periods", "2", "--window-periods", "8")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "refused" in done.stdout


def test_a_run_over_the_step_cap_is_refused():
    """The product's own cap, through the same refusal the Jetson arm answers.

    4000 periods is 1,013,479 steps at the tiny scene's 6.22 nm finest cell, over
    ``fdtd.MAX_STEPS``. The point of running it here is that ``tiny`` and
    ``main`` share one refusal function, so a scene the tier-1 arm accepts is a
    scene the Jetson arm would accept.
    """
    done = run("--tiny", "--periods", "4000")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "refused" in done.stdout
    assert "over the limit of 20,000" in done.stdout


LONG = ["--tiny", "--periods", "500"]           # 126,685 steps at 6.22 nm
RAISED = ["--max-steps", "200000", "--max-seconds", "7200"]


def test_the_same_long_record_is_refused_without_a_flag_and_accepted_with_one():
    """One record, both verdicts: the product's ceilings, then the bench's.

    500 periods on the tiny scene is 126,685 steps at its 6.22 nm finest cell --
    over ``fdtd.MAX_STEPS`` before any solve starts. Without a flag the run path
    refuses it, and ``run_limits`` is the product's own three constants; with the
    flags, the same call accepts it. This is the pair the owner's 2026-10-16
    decision turns on: only a flagged bench run may go longer, and the flag is
    what the ladder and the run path both read.
    """
    m = _load()
    plain = m.parse_args(list(LONG))
    assert m.run_limits(plain) == {"max_cells": m.fdtd.MAX_CELLS,
                                   "max_steps": m.fdtd.MAX_STEPS,
                                   "max_seconds": m.fdtd.MAX_SECONDS,
                                   "max_bytes": m.fdtd.MAX_BYTES}
    lay, request, geom = m.solve_scene(plain)
    measured = m.filled(plain, lay, request)
    assert measured["exceeded"], measured
    assert "over the limit of 20,000" in measured["exceeded"], measured["exceeded"]

    args = m.parse_args(LONG + RAISED)
    assert m.run_limits(args) == {"max_cells": m.fdtd.MAX_CELLS,
                                  "max_steps": 200_000, "max_seconds": 7200.0,
                                  "max_bytes": m.fdtd.MAX_BYTES}
    lay, request, geom = m.solve_scene(args)
    measured = m.filled(args, lay, request)
    assert measured["exceeded"] is None, measured["exceeded"]


def test_the_cell_ceiling_is_the_third_thing_the_bench_flags_can_raise(capsys):
    """The product's own pricing arm on the manifest's own record: refused, then priced.

    The ladder is the cheap half of the product's own layout -- ``planned_scene``
    is documented as "the product's own plan, and its cheap estimate (no
    voxeliser)" -- so this exercises the whole path the manifest's args take,
    without voxelising 14 M cells, which is not tier-1 work. The manifest's
    record holds the open boundary at r20's own distance (``--pad-nm 320
    --pml-nm 200``, the product's own 16/10 cells at dx 20): that is what makes
    the dx-10 record r20's own box at a finer dx -- 14,048,300 cells against
    r20's 2,903,040, over ``fdtd.MAX_CELLS`` by 75.6 % -- because only the three
    pad layers take ``--dx`` while the oxide's ``thickness / MIN_CELLS`` and the
    gold's ``depth / CELLS_PER_SKIN`` are the same capped requests at either dx.
    """
    m = _load()
    record = ["--estimate", "--hex", "paper", "--wavelength", "993", "--dx",
              "10", "--mirror-pad", "600", "--periods", "100",
              "--pad-nm", "320", "--pml-nm", "200"]
    # Without a bench flag not one rung is priced, and the mechanism is per rung:
    # the coarse ones are over the STEP cap, the finer ones (dx 12 at pad 600,
    # dx 10 at either pad) over the CELL cap. The ladder prints only the first 64
    # characters of a refusal, so each row shows the limit that leads its message
    # -- which is why this comment names two ceilings and the assertions below
    # read a coarse row and a fine one, not just the record's own.
    assert m.ladder(m.parse_args(list(record))) == 2
    out = capsys.readouterr().out
    assert ("refused: The grid has 14,048,300 cells, over the limit of 8,000,000"
            in out), out
    assert "every rung above was refused" in out, out
    by_rung = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "600" and parts[0] in ("20.0", "10.0"):
            by_rung[parts[0]] = line
    assert "time steps, over the limit of 20,000" in by_rung["20.0"], out
    assert "14,048,300 cells, over the limit of 8,000,000" in by_rung["10.0"], out

    # The cell ceiling on its own does not admit the record: its 39,650 steps are
    # over the product's 20,000 whatever the cell cap says, so the refusal only
    # moves to the next ceiling. All three flags are needed, as the manifest has.
    assert m.ladder(m.parse_args(record + ["--max-cells", "15000000"])) == 2
    out = capsys.readouterr().out
    assert ("refused: The run needs 39,650 time steps, over the limit of 20,000"
            in out), out

    # With the three flags the manifest passes, the same rung is priced, and the
    # price is the one tasks.md quotes: cells, steps, the estimator's seconds and
    # GiB. The table's row for dx 10.0 / pad 600, read by its own first two
    # columns -- this rung is one of the table's own, so it carries no "(as asked)"
    # marker and a substring search over the whole table could match another row.
    assert m.ladder(m.parse_args(record + ["--max-cells", "15000000",
                                           "--max-steps", "200000",
                                           "--max-seconds", "18000"])) == 0
    row = None
    for line in capsys.readouterr().out.splitlines():
        cells = line.split()
        if len(cells) >= 5 and cells[0] == "10.0" and cells[1] == "600":
            row = line
    assert row, "the dx-10/pad-600 row is missing from the table"
    assert "14,048,300" in row and "39,650" in row, row
    assert "13662.4" in row and "1.63" in row, row
    assert "fits" in row and "refused" not in row, row


def _extents_nm(plan):
    """The planned box in nm per axis: the nodes' own span in the owner frame."""
    return [round((float(max(axis)) - float(min(axis))) / 1e-9, 1)
            for axis in plan["nodes"]]


def test_the_held_boundary_is_the_product_s_own_at_r20_s_dx():
    """``--pad-nm 320 --pml-nm 200`` at dx 20 IS the product's own 16/10 cells.

    That is the whole reason the flags exist: the same pair holds r20's box at dx
    10, so a difference from r20 is the resolution and not the boundary. If the
    knob moved the coarse grid at all, the dx-10 record it is meant to make
    comparable to r20 would be a different box, and nothing in the log would say
    so. Both plans are the product's own, so nothing is voxelised here.
    """
    m = _load()
    base = ["--hex", "paper", "--wavelength", "993", "--mirror-pad", "600",
            "--periods", "100"]
    plain = m.parse_args(base + ["--dx", "20"])
    held = m.parse_args(base + ["--dx", "20", "--pad-nm", "320", "--pml-nm", "200"])
    a, _, geom_a = m.planned_scene(plain)
    b, _, geom_b = m.planned_scene(held)
    assert (geom_b["pad_cells"], geom_b["pml_cells"]) == (m.fdtd_layout.OPEN_PAD_CELLS,
                                                          m.fdtd.DEFAULT_PML_CELLS)
    assert (geom_a["pad_cells"], geom_a["pml_cells"]) == (geom_b["pad_cells"],
                                                          geom_b["pml_cells"])
    assert a["shape"] == b["shape"], (a["shape"], b["shape"])
    assert math.prod(a["shape"]) == math.prod(b["shape"]) == 2_903_040
    assert _extents_nm(a) == _extents_nm(b) == [2318.8, 2318.8, 2177.0]


def test_the_dx_10_record_holds_r20_s_box_and_only_the_resolution_moves():
    """Held at dx 10: r20's own box, 4.84x the cells, the same steps and the same dt.

    The count is what decides the bench ceiling, and the steps are what decide the
    record's own cost: a uniform grid halved per axis would be 8x the cells and,
    at a halved dt, 2x the steps. Neither is what happens, and the reason is in
    the plan -- the capped layers keep their own spacing, so they take up node
    count per axis without refining (244/144 = 1.694, 245/144 = 1.701,
    235/140 = 1.679, which is the 4.839x cells), while ``dt`` is set by the
    oxide's own 3.000 nm and does not move at all.
    """
    m = _load()
    base = ["--hex", "paper", "--wavelength", "993", "--mirror-pad", "600",
            "--periods", "100"]
    coarse, _, _ = m.planned_scene(m.parse_args(base + ["--dx", "20"]))
    fine, _, geom = m.planned_scene(m.parse_args(
        base + ["--dx", "10", "--pad-nm", "320", "--pml-nm", "200"]))
    assert (geom["pad_cells"], geom["pml_cells"]) == (32, 20)
    assert (geom["pad_nm"], geom["pml_nm"]) == (320.0, 200.0)
    assert fine["shape"] == (244, 245, 235), fine["shape"]
    assert math.prod(fine["shape"]) == 14_048_300
    assert _extents_nm(fine) == _extents_nm(coarse)
    # the same dt, because the finest spacing is the oxide's own 3.000 nm at either
    # dx -- read off the plans rather than retyped
    assert float(min(fine["spacings"][2])) == float(min(coarse["spacings"][2]))


def test_the_boundary_distance_is_rounded_to_the_nearest_cell(capsys):
    """A rung whose dx does not divide 320 nm cannot hold it exactly, and says what it held.

    dx 15 nm is the nearest rung that fits the product's own 8 M cells with the
    box held; 320/15 and 200/15 are not whole cells, so the plan holds 21 and 13
    (315 nm and 195 nm) and reports those. Rounded up instead, the same rung
    would hold 22/14 -- 330/210 nm -- and the two are different boundaries, which
    is why the rounding is pinned rather than assumed.
    """
    m = _load()
    args = m.parse_args(["--hex", "paper", "--wavelength", "993", "--mirror-pad",
                         "600", "--dx", "15", "--pad-nm", "320", "--pml-nm", "200"])
    cells = m.boundary_cells(args)
    assert (cells["pad_cells"], cells["pml_cells"]) == (21, 13)
    plan, _, geom = m.planned_scene(args)
    assert (geom["pad_nm"], geom["pml_nm"]) == (315.0, 195.0)
    assert _extents_nm(plan) != [2318.8, 2318.8, 2177.0], _extents_nm(plan)
    # and the nearest-fitting rung really is under the product's own ceiling: this
    # is the rung the manifest's comment offers as the alternative
    assert math.prod(plan["shape"]) == 5_234_775
    out = _ladder_rows(m, ["--dx", "12", "--pad-nm", "320", "--pml-nm", "200"])
    assert out["12.0"][0] == 8_952_300, out


def _ladder_rows(m, extra):
    """The ladder's own table over one record: ``{"dx": [cells, seconds]}`` for pad 600."""
    import contextlib
    import io
    held = io.StringIO()
    record = ["--estimate", "--hex", "paper", "--wavelength", "993",
              "--mirror-pad", "600", "--periods", "100", "--max-cells", "15000000",
              "--max-steps", "200000", "--max-seconds", "18000"] + list(extra)
    with contextlib.redirect_stdout(held):
        assert m.ladder(m.parse_args(record)) == 0
    rows = {}
    for line in held.getvalue().splitlines():
        cells = line.split()
        if len(cells) >= 5 and cells[1] == "600" and cells[0].replace(".", "").isdigit():
            rows[cells[0]] = [int(cells[2].replace(",", "")), float(cells[4])]
    return rows


def test_the_three_boundary_refusals(capsys):
    """A pad under one cell, two absorbers that disagree, and an absorber outside the pad.

    Every one of them is argument-level, so it fires before a scene exists: a
    pad of no cells is no boundary at all, ``--pml`` and ``--pml-nm`` naming
    different absorbers would leave the run guessing which the log meant, and an
    absorber as thick as the pad puts the scene inside it.
    """
    m = _load()
    base = ["--hex", "paper", "--dx", "10"]
    thin = m.parse_args(base + ["--pad-nm", "8"])
    assert "under one cell at dx 10 nm" in m.arm_conflicts(thin), m.arm_conflicts(thin)
    two = m.parse_args(base + ["--pml", "20", "--pml-nm", "100"])
    assert "different absorbers at dx 10 nm (10 cells)" in m.arm_conflicts(two)
    deep = m.parse_args(base + ["--pad-nm", "100"])
    assert ("the 10-cell absorber is not inside the 10-cell pad (100 nm in 100 nm)"
            in m.arm_conflicts(deep)), m.arm_conflicts(deep)
    # reached through the script's own entry, each is an exit 2 and no scene is
    # built: `--tiny` would print its controls if the run had got that far. The
    # tiny arm's own dx is 20 nm and its own absorber 6 cells (120 nm), which is
    # why the nm flags below are the ones they are: the pair that disagrees with
    # its absorber is refused, and a pad that cannot hold that absorber is too.
    for extra in (["--pad-nm", "10"], ["--pml-nm", "20"],
                  ["--pad-nm", "100", "--pml-nm", "120"]):
        done = run("--tiny", *extra)
        assert done.returncode == 2, (extra, done.stdout, done.stderr)
        assert "refused:" in done.stdout + done.stderr, (extra, done.stdout, done.stderr)
        assert "tiny: all controls hold" not in done.stdout, extra


def test_a_tiny_run_holds_the_boundary_it_names():
    """The run path's own boundary line, and the controls still holding at it.

    The tiny scene runs at dx 20 nm with a 6-cell (120 nm) absorber of its own,
    so naming 200/120 nm is 10 and 6 cells: the same absorber in the same
    distance, a smaller pad. What this pins is the pair of readings a log has to
    carry for a dx-10 record to be comparable to r20 at all -- the cells the grid
    was built with and the distance they are -- and that the run really built
    with them: `solve_scene` hands the pair to the layout and the absorber to the
    solver, so the controls' own verdict at this boundary is what says the
    reading is not just a printout. A cell count is not a distance: 16 cells is
    320 nm at dx 20 and 160 nm at dx 10.
    """
    done = run("--tiny", "--pad-nm", "200", "--pml-nm", "120")
    assert done.returncode == 0, done.stdout + done.stderr
    assert ("boundary  open: pad 10 cells = 200 nm, absorber 6 cells = 120 nm,"
            " 80 nm of open space (--pad-nm 200 --pml-nm 120)" in done.stdout), done.stdout
    assert "tiny: all controls hold" in done.stdout, done.stdout


def test_no_product_path_names_the_bench_boundary():
    """The boundary flags are the bench script's own, and its two builders pass them.

    ``plan`` reads ``pad_cells``/``pml_cells`` only from its own caller, so the
    product's boundary stays the product's. The two files that plan a product run
    are the worker and the web form, and the bench script's two scene builders --
    the ones a run, the tiny arm and the estimator's ladder all go through -- hand
    over both, so the box a record is priced in is the box it is built in. The
    tiny arm's three emitter-z probes call the planner with neither on purpose:
    they are asking where the emitter lands, not what the boundary is.
    """
    for product in (ROOT / "worker" / "td_solve.py", ROOT / "web" / "td_form.py"):
        text = product.read_text()
        assert "pad_cells" not in text, product
        assert "pml_cells" not in text, product
    src = SCRIPT.read_text()
    for builder in ("fdtd_layout.plan(names, parts, request, length_scale=1.0,",
                    "fdtd_layout.layout(names, parts, request, length_scale=1.0,"):
        assert builder in src, builder
        window = src[src.index(builder):][:400]
        assert 'pad_cells=geom["pad_cells"]' in window, window
        assert 'pml_cells=geom["pml_cells"]' in window, window


def test_the_estimator_and_the_run_hold_the_same_boundary():
    """A record must be priced in the box it is run in, at the real dx.

    ``plan`` is what the ladder prices (node counts); ``layout`` is what a run
    builds (the nodes). They are handed the same pair of counts from the same
    args, and the two geoms have to come out equal -- otherwise the record's
    price would be for one boundary and its run for another, and the run's own
    log would be the only place that could show it.

    This is the tiny arm on purpose: `solve_scene` voxelises, so the dx-10 record
    (14,048,300 cells) is the Jetson's or the Job's work, not tier 1's.
    """
    m = _load()
    args = m.parse_args(["--tiny", "--pad-nm", "200", "--pml-nm", "120"])
    plan, _, geoms_plan = m.planned_scene(args)
    lay, _, geoms_run = m.solve_scene(args)
    assert geoms_plan == geoms_run
    assert (geoms_run["pad_cells"], geoms_run["pml_cells"]) == (10, 6)
    assert (lay["pad_cells"], lay["pml_cells"]) == (10, 6)
    assert (plan["pad_cells"], plan["pml_cells"]) == (10, 6)



def test_the_run_path_hands_its_raised_ceiling_to_the_cap_check(monkeypatch):
    """The raised ceiling reaches ``fdtd.check_limits``, not just this module.

    ``filled`` builds the estimate and the verdict; the limit that verdict is
    measured against has to be the one handed over, or a run priced against the
    bench ceiling would be refused by the product's own at the last step.
    """
    m = _load()
    calls = []
    real = m.fdtd.check_limits
    monkeypatch.setattr(m.fdtd, "check_limits",
                        lambda est, **kw: (calls.append(kw), real(est, **kw))[1])
    args = m.parse_args(LONG + RAISED)
    lay, request, geom = m.solve_scene(args)
    m.filled(args, lay, request)
    assert calls == [{"max_cells": m.fdtd.MAX_CELLS,
                      "max_steps": 200_000, "max_seconds": 7200.0,
                      "max_bytes": m.fdtd.MAX_BYTES}], calls


def test_no_cap_call_site_escapes_the_bench_ceiling():
    """Every ``check_limits`` call in the bench script hands one over.

    The estimator and the run path must read the SAME ceiling, so a bare
    ``fdtd.check_limits(est)`` -- a second call site quietly taking the product's
    own cap while a rung or a run was priced against the raised one -- is the
    failure this looks for, and the ladder's ``run_limits(sub)`` is the one the
    spy test above cannot reach without sweeping scenes. The product half, where a
    bare call is correct, is ``test_td_solve``.
    """
    src = SCRIPT.read_text()
    calls = re.findall(r"fdtd\.check_limits\([^)]*\)", src)
    assert len(calls) == 2, calls
    assert all("**" in c for c in calls), calls


def test_a_flag_that_would_lower_a_cap_is_refused_before_any_scene(capsys):
    """The flags raise a ceiling; a value under the product's own is an error.

    Two ways, because the parser is not the only way to reach ``run_limits``:
    argparse refuses the flag at the command line, and the ``max()`` inside
    ``run_limits`` makes a namespace that never met the parser read the product's
    own ceiling rather than a lowered one.
    """
    m = _load()
    for flag, value in (("--max-cells", "1000000"), ("--max-steps", "1000"),
                        ("--max-seconds", "60"), ("--max-bytes", "2")):
        with pytest.raises(SystemExit) as info:
            m.parse_args(["--tiny", flag, value])
        assert info.value.code == 2
        err = capsys.readouterr().err
        assert "may only raise a cap" in err, err
    lowered = argparse.Namespace(max_cells=1_000_000, max_steps=1000,
                                 max_seconds=60.0, max_bytes=1e9)
    assert m.run_limits(lowered) == {"max_cells": m.fdtd.MAX_CELLS,
                                     "max_steps": m.fdtd.MAX_STEPS,
                                     "max_seconds": m.fdtd.MAX_SECONDS,
                                     "max_bytes": m.fdtd.MAX_BYTES}


def test_the_memory_ceiling_is_the_fourth_thing_the_bench_flags_can_raise():
    """``--max-bytes`` names GB: ``14`` is 14e9 bytes, and it only raises.

    dx 5 nm of the paper's own scene estimates 11.55 GB against the 3.0 GB the
    product may assume; this is the flag that lets a measured card price it.
    """
    m = _load()
    args = m.parse_args(["--tiny", "--max-bytes", "14"])
    assert args.max_bytes == 14e9
    assert m.run_limits(args)["max_bytes"] == 14e9
    plain = m.parse_args(["--tiny"])
    assert plain.max_bytes is None
    assert m.run_limits(plain)["max_bytes"] == m.fdtd.MAX_BYTES


def test_a_raised_memory_ceiling_is_reported_on_the_caps_line(capsys):
    """A memory-only raise still prints the bench-ceiling parenthetical.

    The dx-5 nm record needs 11.55 GB and no more steps, cells or seconds than
    the product allows, so the parenthetical's gate must read ``max_bytes`` too
    -- otherwise the one ceiling the run raised is the one the log line omits.
    ``--tiny`` bypasses ``report`` (it has its own ``tiny()`` arm), so this calls
    ``report`` on a cheap scene, the same way the run path does, and the plain
    call is the control: no flag, no parenthetical.
    """
    m = _load()
    for extra, wanted in ((["--max-bytes", "14"], True), ([], False)):
        args = m.parse_args(["--dx", "400", *extra])
        lay, request, geom = m.solve_scene(args)
        measured = m.filled(args, lay, request)
        m.report(args, lay, request, geom, measured)
        caps = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("caps")]
        assert len(caps) == 1, caps
        assert ("bench ceiling:" in caps[0]) is wanted, caps
        if wanted:
            assert "14.0 GB" in caps[0], caps


def test_the_gold_flag_selects_what_the_scene_prices_and_reports(capsys):
    """--gold-data picks the library name in one place, and every reader uses it.

    The scene's mirror material, the estimate's metals table, the skin depth the
    report prints and the material the run names all come from ``gold_metal``; a
    run whose estimate priced Rakić while the solver built Palik would report a
    stack it did not run. Rakić is the default, so earlier records reproduce.
    """
    m = _load()
    plain = m.parse_args([])
    palik = m.parse_args(["--gold-data", "palik"])
    assert plain.gold_data == "rakic"
    assert m.scene(plain)[2]["mirror"] == "Au"
    assert m.scene(palik)[2]["mirror"] == "Au-palik"
    args = m.parse_args(["--dx", "400", "--gold-data", "palik"])
    lay, request, geom = m.solve_scene(args)
    assert "Au-palik" in lay["metals"], sorted(lay["metals"])
    measured = m.filled(args, lay, request)
    m.report(args, lay, request, geom, measured)
    stack = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("stack")]
    assert len(stack) == 1, stack
    assert "Au-palik" in stack[0], stack[0]


def test_a_record_that_is_not_settled_is_flagged():
    """8 periods with a 4-period tail: the drive has not settled, F reads 4.61.

    The warning is the guard against the trap rather than the trap itself, so
    what this pins is that the run says so and does not call itself good.
    """
    done = run("--tiny", "--periods", "8", "--window-periods", "4")
    assert "warning: the record is" in done.stdout, done.stdout + done.stderr
    assert "Raise --periods." in done.stdout
    assert "tiny: all controls hold" not in done.stdout


def test_the_fitter_recovers_a_decay_and_refuses_to_invent_one():
    """The fit both ways, on synthetic tails whose tau is known.

    Two fixtures, because one would measure nothing: a decaying sinusoid at a tau
    the window is long enough to see, and a sinusoid that does not decay at all
    inside the window. A fit that reports a decay either way fails the second,
    and a fit that reports none fails the first.
    """
    module = _load()
    f0, dt, n = 3.331e14, 5.0e-18, 6000
    t = np.arange(n) * dt
    rng = np.random.default_rng(7)
    tau = 1.0e-14                                  # 10 fs over a 30 fs window
    values = 1.0 + np.exp(-t / tau) * np.cos(2.0 * np.pi * f0 * t)
    values = values + 1e-4 * rng.standard_normal(n)
    fit = module.fit_decay(values, dt, 0, frequency=f0)
    assert abs(fit["tau"] - tau) / tau < 0.02, fit
    assert abs(fit["f"] - f0) / f0 < 0.02, fit
    assert abs(fit["q"] - math.pi * f0 * tau) / (math.pi * f0 * tau) < 0.05, fit
    assert fit["resolved"]

    flat = 1.0 + np.cos(2.0 * np.pi * f0 * t)      # no decay this window can see
    nope = module.fit_decay(flat, dt, 0, frequency=f0)
    assert not nope["resolved"], nope
    assert nope["at_edge"], nope


def test_the_drive_cut_needs_a_pulse_to_cut():
    """An all-zero drive has no end, and the cut says so instead of returning 0."""
    module = _load()
    with pytest.raises(ValueError):
        module.drive_done(np.zeros(100))


def test_the_tail_arm_reads_the_decay_of_a_drive_free_record():
    """--tiny --ringdown: the window is the drive's own end, and the arm answers.

    Vacuum has no cavity, so either verdict is a pass here -- what this pins is
    that the arm reaches one through the script's own path, over a tail of the
    length the record was asked for, and that ``tiny``'s own control sees it.
    """
    done = run("--tiny", "--ringdown")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tiny: all controls hold" in done.stdout
    assert "the tail arm runs on a homogeneous scene" in done.stdout
    assert "tail      the drive is under" in done.stdout
    found = re.search(r"= ([0-9.]+) periods\n", done.stdout)
    assert found and float(found.group(1)) >= 6.0, done.stdout
    assert "\nQ         " in done.stdout or "unresolved:" in done.stdout, done.stdout


def test_the_band_arm_refuses_a_tail():
    """A tail moves the pulse off the centre of its own record: -94 dB."""
    done = run("--tiny", "--spectrum", "--tail-periods", "10")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "-94 dB" in done.stdout


def test_a_span_that_ends_inside_the_drive_leaves_no_tail():
    """--pulse-span 5 with the tail's own 8 periods: the record ends first."""
    done = run("--tiny", "--ringdown", "--pulse-span", "5", "--tail-periods", "8")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "refused: the drive is over only" in done.stdout, done.stdout
    assert "FAIL" in done.stdout


def test_the_ladder_says_the_tail_arm_does_not_wait_for_the_record():
    """Both ways: the 4 nm line is a bar the tail arm is not held to, and only when it runs.

    The ladder prices a record long enough to resolve 4 nm; the tail arm reads a decay
    instead, so the bar has to be marked as not applying to it -- and marked only when
    it is the arm in play, or the line says it of a CW run the reader is about to make.
    """
    plain = run("--tiny", "--estimate")
    assert "resolving the paper's 4 nm mode needs" in plain.stdout, plain.stdout
    assert "the tail arm does not wait for that" not in plain.stdout, plain.stdout
    tail = run("--tiny", "--estimate", "--ringdown")
    assert "the tail arm does not wait for that" in tail.stdout, tail.stdout


def test_the_ladder_prices_the_rung_that_was_asked_for():
    """--dx 5 is not one of the ladder's own rungs, so it has to be priced on request.

    Left alone, --estimate --dx 5 prints a table of other rungs and says nothing about
    the one the caller named -- which reads exactly like the asked rung was priced and
    fitted. Both directions: the marker means something only if a standard run lacks it.
    """
    done = run("--estimate", "--dx", "5", "--mirror-pad", "400")
    row = [line for line in done.stdout.splitlines()
           if line.strip().startswith("5.0")]
    assert row, done.stdout + done.stderr
    assert "as asked" in row[0], done.stdout          # priced, and marked as the ask
    assert "over the limit of 8,000,000" in row[0], done.stdout
    plain = run("--estimate", "--dx", "20", "--mirror-pad", "400")
    assert "as asked" not in plain.stdout, plain.stdout


def test_the_unresolved_outcome_is_reported_with_its_bound(capsys):
    """The lead's second accepted outcome, forced rather than hoped for.

    No scene produces an unresolved tail on demand -- the tiny vacuum tail resolved to
    Q = 2 when it was run -- so the fit is stubbed and the arm asked directly. Two
    things this deliberately does not claim: `--tiny` returns its controls' verdict (0,
    the control line naming the answer the arm gave) rather than the arm's code, and the
    non-tiny path's pass-through of that code is a tier-2 reading, not a tier-1 one. In
    process, because a stub cannot reach a subprocess.
    """
    m = _load()
    args = m.parse_args(["--tiny", "--ringdown"])
    lay, request, geom = m.solve_scene(args)
    measured = m.filled(args, lay, request)
    real = m.fit_decay

    def stub(values, dt, start, **kwargs):
        return {"f": 3.331e14, "f_err": 1.0e12, "tau": 9.7e-14, "tau_err": 3.0e-14,
                "q": 97.0, "q_err": 30.0, "resolved": False, "at_edge": False,
                "rms": 1.0e-6, "peak": 1.0e-2, "n": 5_000, "window": 9.7e-14,
                "dof": 4_995, "f_edge": False}

    m.fit_decay = stub
    try:
        _, code = m.ringdown(args, lay, request, measured)
    finally:
        m.fit_decay = real
    out = capsys.readouterr().out
    assert code == m.UNRESOLVED == 3, out
    assert "unresolved" in out, out
    assert "Q >" in out, out                         # the bound is what it reports
    assert "Q         " not in out, out              # and not a Q it did not measure
    assert "AT THE EDGE" not in out, out             # a fit inside its window says so

    # The same arm, one flag apart: a fit sitting on its window's edge must not
    # print an f that reads like a measured line. This is the case the window
    # exists for -- it is what a mode outside the drive's band looks like.
    def edge_stub(values, dt, start, **kwargs):
        return {**stub(values, dt, start, **kwargs), "f_edge": True}

    m.fit_decay = edge_stub
    try:
        m.ringdown(args, lay, request, measured)
    finally:
        m.fit_decay = real
    edge = capsys.readouterr().out
    assert "AT THE EDGE" in edge, edge


def test_a_degenerate_covariance_is_reported_not_rounded_to_zero(capsys):
    """A constant tail: the covariance's tau and omega columns both vanish.

    fit_decay is called for real here -- the fixture is the diagnosis, not a
    stub. With no signal the optimum sits at a = b = 0, and that kills both the
    tau and the omega Jacobian column, so cov[3,3] and cov[4,4] are not
    positive. The clamp this was written against called them 0.0, and a zero
    error passes every "more than three of its own errors" test there is: a
    record with nothing in it reported a resolved Q with no error bar at all.
    """
    m = _load()
    # r17's own record: dt 8.431110137e-18 s and a 5,474-sample tail, so this
    # fixture is the shape the arm actually printed "+- 0" for on 2026-10-16.
    f0, dt, n = 3.019e14, 8.431110137e-18, 5474
    fit = m.fit_decay(np.full(n, 1e-9), dt, 0, frequency=f0)
    assert fit["tau_err"] is None and fit["f_err"] is None, fit
    assert fit["q_err"] is None, fit
    assert not fit["resolved"], fit                  # a zero error must not pass
    assert m._err(None, 1e-15) == m.UNDETERMINED
    assert m._err(1.0e-16, 1e-15) == "0.1"           # a real error still prints

    # And the arm prints the word where the number would have gone. An error is
    # not reachable on demand from a scene, so the fit is stubbed here -- but a
    # good tau with a degenerate omega is the shape r17's tail actually had.
    # The scene is a carrier for the arm's own report, not a measurement: the fit
    # is stubbed, so what the tail holds does not matter, only that the arm runs.
    # It still costs real stepping, so it is asked for the shortest record the arm
    # accepts -- 1,525 steps = 18.5 fs = 6.2 periods, against 11,487 for the
    # parser's own --tail-periods 32. The word is what is read, and it is the same
    # word at either length: 82.4 s for this sequence against 482.61 s.
    args = m.parse_args(["--tiny", "--ringdown", "--band-half", "200",
                         "--tail-periods", "6"])
    lay, request, geom = m.solve_scene(args)
    measured = m.filled(args, lay, request)
    real = m.fit_decay
    bad_f = {"f": 3.331e14, "f_err": None, "tau": 9.7e-14, "tau_err": 3.0e-14,
             "q": 97.0, "q_err": None, "resolved": True, "at_edge": False,
             "rms": 1.0e-6, "peak": 1.0e-2, "n": 5_000, "window": 9.7e-14,
             "dof": 4_995, "f_edge": False}
    m.fit_decay = lambda *a, **k: bad_f
    try:
        _, code = m.ringdown(args, lay, request, measured)
    finally:
        m.fit_decay = real
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.count(m.UNDETERMINED) == 2, out       # the f error and the Q error
    assert "+- 0," not in out and "+- 0 " not in out, out

    # A tau error that is undetermined answers in the unresolved branch, and it
    # must not print a three-sigma floor it cannot form.
    m.fit_decay = lambda *a, **k: {**bad_f, "tau_err": None, "resolved": False}
    try:
        _, code = m.ringdown(args, lay, request, measured)
    finally:
        m.fit_decay = real
    out = capsys.readouterr().out
    assert code == m.UNRESOLVED == 3, out
    assert m.UNDETERMINED in out, out
    assert "three-sigma floor" in out and "Q >" not in out, out


def test_a_positive_error_below_its_own_parameter_is_the_pinv_too():
    """The other shape of the same thing: a positive diagonal, and no error.

    The clamp this arm was written against caught a non-positive diagonal. Run A
    (2026-10-16, the corners wire at 898.3 nm, tau at the fit grid's top) came
    back with both errors positive and neither one a number -- 4.1e-65 THz
    beside a 333.7 THz centre, 6.3e-14 fs beside a 2033 fs decay, tens of
    orders below their own values (67 decades for the frequency, 16.5 for the
    decay). A `<= 0` test passes both, and the
    three-sigma test then reads the pinv's noise as a resolved mode.

    Two fixtures, because the floor has to drop one shape and keep the other. A
    noiseless decaying sinusoid reproduces it: the residual is at the machine's
    own floor, so sigma2 is, and the normal matrix is ill-conditioned enough
    that the pinv answers with noise. Beside it stands the record the fitter is
    *supposed* to measure, and the two are separated by the size of the error
    next to its own parameter -- 6.5e-6 for the good tau against 1e-9, three
    decades of daylight. Without that control the fix could be `return None`
    and this test would not notice.
    """
    m = _load()
    f0, dt, n = 3.331e14, 5.0e-18, 6000
    t = np.arange(n) * dt
    quiet = 1.0 + np.exp(-t / 1.0e-14) * np.cos(2.0 * np.pi * f0 * t)
    fit = m.fit_decay(quiet, dt, 0, frequency=f0)
    assert fit["f_err"] is None and fit["tau_err"] is None, fit
    assert fit["q_err"] is None and not fit["resolved"], fit
    assert m._err(fit["f_err"], 1e12) == m.UNDETERMINED

    # The control: the same tail with 1e-4 of noise, which is the fixture the
    # fitter is expected to resolve. One call keeps a tau error and drops an f
    # error, which is the whole distinction the floor is meant to draw.
    rng = np.random.default_rng(7)
    got = m.fit_decay(quiet + 1.0e-4 * rng.standard_normal(n), dt, 0, frequency=f0)
    assert got["tau_err"] is not None and got["tau_err"] / got["tau"] > 1.0e-7, got
    assert got["resolved"], got
    assert got["f_err"] is None, got


def test_the_settling_line_reads_the_record_before_its_end(capsys):
    """The CW arm reads one record twice: at two thirds, and at its end.

    One number cannot say whether the window it was read from had stopped rising,
    and that is the whole question a settled-record claim rests on, so the arm
    prints the same mean at both places. What makes the earlier number is
    `fdtd.delivered_power` and `fdtd.dipole_moment` taking the LAST `periods` of
    the series they are handed (`delivered_power` slices `drive[size - steps:]`
    itself): the record cut at the two-thirds step is that window, ended early.
    `_run` is stubbed because the claim is about the arm's reading of a record
    whose shape is known, not about a scene; every helper below it is real, and
    both readings come out of the same ones, which is what keeps this from being
    a second estimator. Two-sided on purpose: a rising record must read different
    at the two places, and a settled one must not.
    """
    m = _load()
    args = m.parse_args(["--tiny"])                  # the CW arm, 8-period windows
    f0, dt, n = 3.019e14, 8.431110137e-18, 20_000   # 993 nm, 20,000 steps
    t = np.arange(n) * dt                            # 50.9 drive periods
    drive = np.sin(2.0 * np.pi * f0 * t)
    real_run = m._run

    def run(sim):
        m._run = lambda *a, **k: (sim, 1.0e-27, 1.0, 1.0)
        capsys.readouterr()
        m.measure(args, None, {"frequency_hz": f0}, {"est": {"dt": dt}})
        return capsys.readouterr().out

    try:
        # tau of 300 fs against a 50.9-period record: at two thirds the envelope
        # is 0.73 of its value at the end (0.3125 against 0.4299; 0.53 in
        # power), so a reading that says "settled" here
        # would be wrong by construction rather than by tolerance.
        out = run({"drive": drive,
                   "source_field": (1.0 - np.exp(-t / 3.0e-13)) * drive})
        line = next(row for row in out.splitlines() if row.startswith("settling"))
        word = line.replace(",", " ").split()
        early = float(word[word.index("F") + 2])
        end = float(word[word.index("against") + 1])
        assert end / early > 1.2, out                    # it has NOT settled
        # ... and the second number is the arm's own F, not a third estimate.
        measured = next(row for row in out.splitlines()
                        if row.startswith("measured"))
        assert f"F = {end:.4f}" in measured, out

        flat = run({"drive": drive, "source_field": drive.copy()})
        line = next(row for row in flat.splitlines() if row.startswith("settling"))
        word = line.replace(",", " ").split()
        settled = (float(word[word.index("against") + 1])
                   / float(word[word.index("F") + 2]))
        assert abs(settled - 1.0) < 0.01, flat           # a tone reads the same
    finally:
        m._run = real_run


def test_the_progress_line_survives_a_killed_run(tmp_path):
    """Why the line exists: rung 6 hit the bench's cap and left only its plan.

    The child here runs 3,000 steps with stdout on a file -- so the interpreter
    block-buffers it, which is what the tier-2 runner does -- and leaves through
    its own check hook with ``os._exit``: no flush, no atexit, nothing. That is
    the kill the bench delivered, minus the 30 minutes. The lines for steps 1,000
    and 2,000 have to be in the file anyway, which they are only if the loop
    flushed them itself.

    The box is 8x the one this started on (24x22x22 = 11,616 cells, measured here
    at 0.7 s and 1.5 s for the two lines) because the check on the number below
    needs the loop's own seconds to be long enough that the tenth the line prints
    is a small part of them.
    """
    program = tmp_path / "progress_probe.py"
    program.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'dockerfiles' / 'maxwell_solver')!r})\n"
        "import numpy as np, fdtd\n"
        "def stop(step):\n"
        "    if step >= 2048:\n"
        "        os._exit(9)\n"
        "eps = np.ones((24, 22, 22), np.float32)\n"
        "src = fdtd.Source((12, 11, 11), frequency=3e14, bandwidth=1e14,"
        " component='z')\n"
        "fdtd.simulate(eps, src, 3000, probes=[(13, 11, 11, 'z')],\n"
        "              spacings=(20e-9,) * 3, boundary='metal', check=stop)\n")
    log = tmp_path / "run.log"
    with log.open("w") as handle:
        done = subprocess.run([sys.executable, str(program)], stdout=handle,
                              stderr=subprocess.STDOUT, cwd=ROOT, timeout=300)
    assert done.returncode == 9, done.returncode
    text = log.read_text()
    # The two lines the run had reached, with the rate the loop was actually
    # given.  Both numbers on a line are rounded -- the seconds to a tenth, the
    # rate to a tenth -- so the exact division is not the thing to compare
    # against: the first draft did that within 0.1 and was red on this box
    # (0.2 s / 6.4 Mcell-steps/s, whose exact rate is 7.26) while a slow machine
    # made the same line pass.  A control whose verdict is the machine's speed
    # proves nothing, so the number is asked to sit in the band the printed
    # seconds can support: the slowest and the fastest true time that could have
    # rounded to it, each side widened by the rate's own last digit.
    lines = re.findall(r"^progress  step ([\d,]+) of 3,000,\s+([\d.]+) s,"
                       r"\s+([\d.]+) Mcell-steps/s$", text, re.M)
    assert [step for step, _, _ in lines] == ["1,000", "2,000"], text
    for step, seconds, rate in lines:
        cells = 24 * 22 * 22
        t = float(seconds)
        assert t > 0.1, lines
        implied = cells * int(step.replace(",", "")) / 1e6
        assert (implied / (t + 0.05) - 0.05
                <= float(rate) <= implied / (t - 0.05) + 0.05), lines


def test_save_tail_without_the_tail_arm_is_refused(tmp_path):
    """2i: the flag is read in the ringdown arm and nowhere else, and it is not inert.

    Both arms that have no tail to save are asked through the command line the
    bench uses, and the tail arm is asked to keep its write -- a refusal that
    also refused the armed run would be worse than the silence it replaces.
    """
    for extra in ([], ["--spectrum"]):
        done = run("--tiny", *extra, "--save-tail", str(tmp_path / "tail"))
        assert done.returncode == 2, done.stdout + done.stderr
        assert "refused" in done.stdout and "--save-tail" in done.stdout, done.stdout
    saved = tmp_path / "tail-ok"
    done = run("--tiny", "--ringdown", "--save-tail", str(saved))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (saved.parent / (saved.name + ".npz")).exists(), done.stdout


def test_a_tail_that_cannot_be_written_still_leaves_the_number(tmp_path):
    """2h, driven end to end: the record is a convenience, the fit is the number.

    ``tiny``'s own control asks the raise site for the type of what it raises,
    which is how the arm's ``except OSError`` is kept honest -- but it never puts
    a failing path through the arm, so it cannot see whether the arm survives one.
    The child here is the same command the bench runs, with the path pointing
    where no file can be written (a file where a directory belongs), and the
    asserts are the run's own: exit 0, the fit and Q printed, the warning naming
    the exception, and no record left at the path. A fix that dropped the number
    instead of the file fails on the first two.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x\n")
    target = blocker / "tail"
    done = run("--tiny", "--ringdown", "--save-tail", str(target))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "fit       f =" in done.stdout, done.stdout
    assert "\nQ         " in done.stdout or "unresolved:" in done.stdout, done.stdout
    assert re.search(r"^warning   the tail was not written \(NotADirectoryError:",
                     done.stdout, re.M), done.stdout
    # the write's own line is what says the record is on disk (rung 7's log has it:
    # "17,948 samples and the dt they were taken with in logs/r7_tail.npz"), so a run
    # whose path could not be written must not be printing it -- an assertion that can
    # fail, which `not Path(...).exists()` here could not: the path's parent is a file
    assert "samples and the dt they were taken with in" not in done.stdout, done.stdout


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_source_line_prints_the_cell_and_the_applied_permittivity(capsys):
    """The `source` line carries both permittivities, and the arm reads the applied one.

    The readings (`delivered_power`, `dipole_moment`, `purcell_spectrum`) take
    the permittivity the update applies at the source's own E sample
    (`fdtd.source_permittivity`), not the layout's node value; the two differ
    only where a Lorentz-Drude metal owns the sample, which it updates with
    eps_inf = 1. A scene with a gold node next to the source along its own axis
    (x) makes them differ, so a log shows it and the returned value is 1; the
    same scene with the gold beside it along y makes them the same. The host
    index stays the material's: sqrt of the node value.
    """
    m = _load()
    args = m.parse_args(["--tiny"])                  # the CW arm, --axis x
    eps_inp = m.N_INP ** 2
    cell = (4, 4, 4)
    request = {"frequency_hz": 3.3e14, "boundary": "open"}
    for metal_node, applied in (((5, 4, 4), 1.0), ((4, 5, 4), eps_inp)):
        region = np.zeros((9, 9, 9), dtype=bool)
        region[metal_node] = True
        lay = {"cell": cell, "eps": np.full((9, 9, 9), eps_inp),
               "metals": {"Au": region}, "spacings": [np.full(8, 2.0e-8)] * 3}
        capsys.readouterr()
        _source, _terms, _volume, index, eps_r = m.source_and_terms(
            args, lay, request)
        line = next(row for row in capsys.readouterr().out.splitlines()
                    if row.startswith("source"))
        assert eps_r == pytest.approx(applied, rel=1e-12, abs=0.0), line
        assert index == pytest.approx(m.N_INP, rel=1e-12), line
        assert (f"eps_r {eps_inp:.4f} (cell), {applied:.4f} (applied at Ex)"
                in line), line
