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


def test_report_flags_non_perturbative_fields(capsys):
    """The default sweep's fields push the shift past the level spacing.

    A 4.9 eV shift on a 0.08 eV excitation is not a perturbation, and the
    quadratic p/beta form does not describe it. Reporting beta from those
    rows without saying so would be the same class of error as quoting a
    number nobody measured, so the bench must say it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import stark_bench as SB
    rows, box = SB.sweep(depths=(0.0, 0.3), fields=(0.0, 0.1, 0.2, 0.4),
                         height_nm=4.0, transverse_nm=12.0, z_extent_nm=16.0,
                         n_xy=6, n_z=40)
    SB.report(rows, (0.0, 0.1, 0.2, 0.4), box, 4.0)
    out = capsys.readouterr().out
    assert "outside the regime" in out
    # and the units a reader needs
    assert "kV/cm" in out


def test_report_uses_the_well_ceiling_not_the_box_ceiling(capsys):
    """The confinement criterion must be h^2/12, a LENGTH, not a fraction of
    the box's bare value. With the box-relative rule the zero-depth row -- a
    free particle sitting exactly at the box ceiling -- reads as confined,
    which is backwards."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import stark_bench as SB
    rows, box = SB.sweep(depths=(0.0,), fields=(0.0, 0.1),
                         height_nm=4.0, transverse_nm=12.0, z_extent_nm=16.0,
                         n_xy=6, n_z=40)
    SB.report(rows, (0.0, 0.1), box, 4.0)
    out = capsys.readouterr().out
    assert "1.333 nm^2" in out
    # at depth 0 both carriers sit at the bare box ceiling and must be
    # reported NOT confined
    assert "HOLE is NOT confined" in out
    assert "ELECTRON is NOT confined" in out


def _beta_cell(out: str, depth: float) -> str:
    """Return the beta cell of the emitted table row for ``depth``.

    The table is what a reader screenshots, so the test parses the printed
    row rather than the row dict -- the defect being pinned lives in the
    emission, not the computation.
    """
    for line in out.splitlines():
        toks = line.split()
        if len(toks) >= 5 and toks[0] == f"{depth:.2f}":
            return toks[4]
    raise AssertionError(f"no emitted table row for depth {depth}")


def test_beta_cell_is_not_a_bare_number_on_an_electron_limited_row(capsys):
    """A beta from a row whose electron is loose describes the box, not the dot.

    At depth 0.2 the hole is held (<z2>h = 1.086 nm^2) but the electron is not
    (<z2>e = 3.244 nm^2 against the 1.333 nm^2 ceiling), so the exciton extent
    is set by the box. The caveat must sit IN the cell: a number with its
    warning sixteen lines below is a number that gets lifted bare.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import stark_bench as SB
    rows, box = SB.sweep(depths=(0.2,), fields=(0.0, 0.1),
                         height_nm=4.0, transverse_nm=12.0, z_extent_nm=16.0,
                         n_xy=6, n_z=40)
    SB.report(rows, (0.0, 0.1), box, 4.0)
    out = capsys.readouterr().out
    cell = _beta_cell(out, 0.2)
    # a plain float is exactly the failure mode
    assert not cell.removeprefix("-").replace(".", "").isdigit(), cell
    assert "*" in cell
    assert "electron-limited" in out


def test_beta_cell_is_a_number_when_the_electron_is_held(capsys):
    """The marker must be earned, not painted on every row.

    If the flag were unconditional, blanking beta everywhere would pass the
    test above while destroying the table's information. Use a depth deep
    enough to hold the electron and check the cell is a real number with no
    asterisk.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import stark_bench as SB
    rows, box = SB.sweep(depths=(3.0,), fields=(0.0, 0.1),
                         height_nm=4.0, transverse_nm=12.0, z_extent_nm=16.0,
                         n_xy=6, n_z=40)
    SB.report(rows, (0.0, 0.1), box, 4.0)
    out = capsys.readouterr().out
    cell = _beta_cell(out, 3.0)
    assert "*" not in cell
    float(cell)  # must parse as a number
