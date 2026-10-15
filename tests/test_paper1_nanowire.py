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

The estimate ladder is covered at the tiny dims too -- there it only builds plans
and estimates, never voxelises or solves, so it stays tier 1 -- which is how a
ladder with every rung refused is seen to report that rather than read the last
rung it happened to try.

The tiny scene is deliberately small -- 20 nm cells, a 100 nm wire -- so it runs
in tier 1. It exercises the same code path the Jetson run does, not the same
physics.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

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
    # 13 controls: a scene that quietly loses one still has to be caught here
    assert done.stdout.count("\n  ok   ") == 13, done.stdout
    # and the settled record the small scene runs must not warn about itself
    assert "warning: the record is" not in done.stdout


def test_the_tiny_scene_reads_one_across_the_band_in_vacuum():
    """The band arm, through the script's own path, on a homogeneous scene."""
    done = run("--tiny", "--spectrum")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tiny: all controls hold" in done.stdout


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

    4000 periods is 995,162 steps at the tiny scene's 6.29 nm finest cell, over
    ``fdtd.MAX_STEPS``. The point of running it here is that ``tiny`` and
    ``main`` share one refusal function, so a scene the tier-1 arm accepts is a
    scene the Jetson arm would accept.
    """
    done = run("--tiny", "--periods", "4000")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "refused" in done.stdout
    assert "over the limit of 20,000" in done.stdout


def test_a_record_that_is_not_settled_is_flagged():
    """8 periods with a 4-period tail: the drive has not settled, F reads 4.61.

    The warning is the guard against the trap rather than the trap itself, so
    what this pins is that the run says so and does not call itself good.
    """
    done = run("--tiny", "--periods", "8", "--window-periods", "4")
    assert "warning: the record is" in done.stdout, done.stdout + done.stderr
    assert "Raise --periods." in done.stdout
    assert "tiny: all controls hold" not in done.stdout


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
