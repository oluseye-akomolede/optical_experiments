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
    # 19 controls: a scene that quietly loses one still has to be caught here. The
    # three past 15 are the fit's window following the drive's band (2e), the saved
    # tail re-fitting to the line it came from (2f), and an unwritable tail path
    # raising the one exception the arm catches (2h); the fourth is --save-tail
    # refused on the arms that have no tail to write (2i).
    assert done.stdout.count("\n  ok   ") == 19, done.stdout
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
    # the dipole line says where the node is and what was asked for it, so the
    # gap between them is a quantisation and must be inside one finest cell
    dipole = re.search(r"^dipole\s+node \((\d+), (\d+), (\d+)\): z\s+([\d.]+) nm,"
                       r"\s+([\d.]+) nm below the top facet at ([\d.]+) nm"
                       r"\s+\(--qd-below (\S+) asked\); in plane\s+([+\-][\d.]+),"
                       r"\s+([+\-][\d.]+) nm off the wire's axis, (\w)-polarised$",
                       done.stdout, re.M)
    assert dipole, done.stdout
    finest = float(re.search(r"z ([\d.]+) nm finest", done.stdout).group(1))
    node_z, depth, top = (float(dipole.group(4)), float(dipole.group(5)),
                          float(dipole.group(6)))
    # the line's own arithmetic: the printed depth is the two printed z values
    assert abs((top - node_z) - depth) < 1e-9, done.stdout
    # and the node it picked is within one finest cell of the depth asked for
    assert abs(depth - float(dipole.group(7))) <= finest, done.stdout


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


def test_the_progress_line_survives_a_killed_run(tmp_path):
    """Why the line exists: rung 6 hit the bench's cap and left only its plan.

    The child here runs 3,000 steps with stdout on a file -- so the interpreter
    block-buffers it, which is what the tier-2 runner does -- and leaves through
    its own check hook with ``os._exit``: no flush, no atexit, nothing. That is
    the kill the bench delivered, minus the 30 minutes. The lines for steps 1,000
    and 2,000 have to be in the file anyway, which they are only if the loop
    flushed them itself.
    """
    program = tmp_path / "progress_probe.py"
    program.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'dockerfiles' / 'maxwell_solver')!r})\n"
        "import numpy as np, fdtd\n"
        "def stop(step):\n"
        "    if step >= 2048:\n"
        "        os._exit(9)\n"
        "eps = np.ones((12, 11, 11), np.float32)\n"
        "src = fdtd.Source((6, 5, 5), frequency=3e14, bandwidth=1e14,"
        " component='z')\n"
        "fdtd.simulate(eps, src, 3000, probes=[(7, 5, 5, 'z')],\n"
        "              spacings=(20e-9,) * 3, boundary='metal', check=stop)\n")
    log = tmp_path / "run.log"
    with log.open("w") as handle:
        done = subprocess.run([sys.executable, str(program)], stdout=handle,
                              stderr=subprocess.STDOUT, cwd=ROOT, timeout=300)
    assert done.returncode == 9, done.returncode
    text = log.read_text()
    # the two lines the run had reached, with the rate it was actually given --
    # 1452 cells over the loop's own seconds, so the number is the machine's
    lines = re.findall(r"^progress  step ([\d,]+) of 3,000,\s+([\d.]+) s,"
                       r"\s+([\d.]+) Mcell-steps/s$", text, re.M)
    assert [step for step, _, _ in lines] == ["1,000", "2,000"], text
    for step, seconds, rate in lines:
        cells = 12 * 11 * 11
        assert abs(float(rate) - cells * int(step.replace(",", ""))
                   / float(seconds) / 1e6) < 0.1, lines


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
