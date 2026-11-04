"""`scripts/stark_bench.py` (1.9): the depth sweep, and that it cannot flatter itself.

The bench's whole claim is that `beta` is a property of an UNMEASURED well
depth. A bench that printed one comfortable number would quietly turn an
unmeasured parameter into a result, so what is pinned here is the sweep's
discrimination: the depth must actually move beta, and the shallow-but-present
well must be visible in `<z^2>` rather than buried in an energy.

The trap these tests guard is the one this work already fell into once: a well
whose half-width equals the box half-width lands its barrier on the Dirichlet
walls, the potential is then unused, and the solve silently reports the
INFINITE-well energy at every resolution. That looks converged and is wrong.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOLVER = ROOT / "dockerfiles" / "maxwell_solver"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SOLVER))
sys.path.insert(0, str(SCRIPTS))

import dot_material as DM  # noqa: E402
import stark as S  # noqa: E402
import stark_bench as B  # noqa: E402


def test_bench_runs_as_shipped():
    """The script's own main, with its defaults, exits clean."""
    assert B.main([]) == 0


def test_bench_refuses_a_well_as_tall_as_its_box():
    """Height == extent is the infinite-well trap, so it must be refused.

    Refused rather than silently reinterpreted: the caller who passes it means
    one of two different problems and the script cannot tell which.
    """
    with pytest.raises(SystemExit):
        B.main(["--height", "16.0", "--z-extent", "16.0"])


def test_bench_refuses_a_sweep_with_no_zero_field_point():
    """Without F = 0 the shift has no anchor and p/beta are meaningless."""
    with pytest.raises(SystemExit):
        B.main(["--fields", "0.1", "0.2"])


def test_the_bare_box_reproduces_the_analytic_z_squared():
    """Depth 0 means no well, and `<z^2>` must then be the bare-box value.

    This is the calibration the whole diagnostic rests on: if `<z^2>` at depth
    0 were something else, every "it shrank, so it is confined" comparison
    below would be against the wrong ceiling.
    """
    material = B._material()
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    bare = B._z_squared(box, material.m_h_ratio, 0.0, 4.0)
    extent = box.size[2]
    want = extent**2 * (1.0 / 12.0 - 1.0 / (2.0 * np.pi**2))
    assert bare == pytest.approx(want, rel=5e-3), (bare, want)


def test_a_deeper_well_confines_the_hole_more():
    """`<z^2>` falls monotonically with depth; the diagnostic must resolve depth.

    If this failed, the sweep could not distinguish a confined carrier from a
    free one and its beta column would be a table of numbers about nothing.
    """
    material = B._material()
    box = S.Box((0.0, 0.0, -8.0), (12.0, 12.0, 16.0), (6, 6, 40))
    widths = [B._z_squared(box, material.m_h_ratio, d, 4.0)
              for d in (0.1, 0.2, 0.5)]
    assert widths[0] > widths[1] > widths[2], widths


def test_the_sweep_shows_beta_is_not_a_property_of_the_dot():
    """beta must MOVE with depth, and the report must say so.

    This is the bench's headline. Pinned as a test so a later "tidy-up" that
    picks one representative depth cannot pass silently: the assertion is that
    the spread is large, not that it is small.
    """
    rows, _box = B.sweep((0.1, 0.2, 0.5), (0.0, 0.1, 0.2), 4.0, 12.0, 16.0, 6, 40)
    betas = [r["beta"] for r in rows]
    assert all(np.isfinite(betas)), betas
    spread = (max(betas) - min(betas)) / abs(min(betas))
    assert spread > 0.25, (betas, spread)


def test_the_report_names_the_unconfined_hole():
    """At zero depth the hole is NOT a dot, and the report must say so."""
    rows, box = B.sweep((0.0, 0.5), (0.0, 0.2), 4.0, 12.0, 16.0, 6, 40)
    bare = box.size[2] ** 2 * (1.0 / 12.0 - 1.0 / (2.0 * np.pi**2))
    assert rows[0]["z2h"] == pytest.approx(bare, rel=5e-3)
    assert rows[1]["z2h"] < 0.5 * bare
