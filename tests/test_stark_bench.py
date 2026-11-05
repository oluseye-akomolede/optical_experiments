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


def test_the_electron_needs_an_unrealistically_deep_well_to_bind():
    """The z-only electron is unbound wherever a real offset would put it.

    The reviewer measured the electron above the tight-well ceiling at EVERY
    swept depth. This pins the threshold, because "unbound at our depths"
    invites the reply "sweep deeper". The electron first satisfies h^2/12 at a
    z-well depth just above 0.6 eV, and the 4 nm infinite-well electron level
    is already 0.294 eV at m_e = 0.08 -- so a realistic conduction offset sits
    well short of the depth needed.

    This is the honest negative, not a tuning target: the assertion is that
    the threshold is HIGH, so shrinking it to make the dot bind would fail.
    """
    n = 161
    zs = np.linspace(-8.0, 8.0, n)
    h = 16.0 / (n - 1)
    ceiling = 4.0**2 / 12.0

    def confined(depth):
        V = np.where(np.abs(zs) < 2.0, 0.0, depth)
        e, full = S.lowest_states_1d(n, h, mass_ratio=0.08,
                                     potential_eV=V, n_states=1)
        v = full[:, 0] / np.linalg.norm(full[:, 0])
        return float(np.sum(v**2 * zs**2)) <= ceiling

    assert not confined(0.30), "electron must still be loose at 0.30 eV"
    assert not confined(0.50), "electron must still be loose at 0.50 eV"
    assert confined(0.80), "electron must bind by 0.80 eV"


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


# ---------------------------------------------- lateral confinement (1.9 follow-on)

def test_lateral_default_is_the_equal_area_square_of_the_cited_diameter():
    """The side is derived from dot_material, never typed into the bench."""
    import math
    assert B.EQUAL_AREA_SIDE_NM == pytest.approx(
        DM.DOT_DIAMETER_NM.value * math.sqrt(math.pi) / 2.0, rel=1e-15)
    assert B.EQUAL_AREA_SIDE_NM ** 2 == pytest.approx(
        math.pi * (DM.DOT_DIAMETER_NM.value / 2.0) ** 2, rel=1e-12)


def test_bench_refuses_a_lateral_dot_as_wide_as_its_box():
    """The default 12 nm box is narrower than the 15.95 nm dot: refused, not
    silently turned into the hard box wall."""
    with pytest.raises(SystemExit):
        B.main(["--lateral"])
    with pytest.raises(SystemExit):
        B.main(["--lateral", "20", "--transverse", "20"])


def _lateral_electron(depth):
    material = B._material()
    box = S.Box((0.0, 0.0, -8.0), (32.0, 32.0, 16.0), (16, 16, 40))
    return B._confinement(box, material.m_e_ratio, depth, 4.0,
                          B.EQUAL_AREA_SIDE_NM, "cell")


def test_with_lateral_confinement_the_electron_is_held_from_0_75_eV_not_0_7():
    """The follow-on's answer, at the bench's lateral grid (2 nm x 0.4 nm).

    On the 1.9 length criterion (<z^2>e <= h^2/12 = 1.333 nm^2): NOT held at
    0.7 eV (1.344), held at 0.75 (1.294) and 0.8 (1.251). Below the barrier
    (E_e < V0, a bound state) already at 0.1 eV (0.0875 eV). Pinned both ways
    so neither verdict can drift silently.
    """
    tight = 16.0 / 12.0
    at = {d: _lateral_electron(d) for d in (0.1, 0.7, 0.75, 0.8)}
    assert at[0.7]["z2"] > tight, at[0.7]
    assert at[0.75]["z2"] < tight and at[0.8]["z2"] < tight, (at[0.75], at[0.8])
    assert at[0.1]["energy"] < 0.1, at[0.1]
    assert 0.5 < at[0.1]["p_in"] < at[0.8]["p_in"] < 1.0


def test_lateral_report_prints_the_binding_table_against_V0(capsys):
    """With a lateral dot the continuum edge is V0 itself, and the table says so."""
    rows, box = B.sweep((0.8,), (0.0, 0.1), 4.0, 20.0, 16.0, 10, 40,
                        lateral_nm=B.EQUAL_AREA_SIDE_NM, mask="cell")
    B.report(rows, (0.0, 0.1), box, 4.0, B.EQUAL_AREA_SIDE_NM, "cell")
    out = capsys.readouterr().out
    assert "electron binding" in out
    assert "continuum edge = V0," in out
    assert "box dot 15.952 x 15.952 nm" in out
    assert rows[0]["bound_e"]
