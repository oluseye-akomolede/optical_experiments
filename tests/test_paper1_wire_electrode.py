"""Tests for the paper-1 wire-with-top-electrode fixture (task 1.6b).

The fixture is UNVALIDATED and its field is NOT converged, so these tests assert
the things that ARE established and, where the field is involved, assert the
NEGATIVE -- that it moves with the mesh. A test that pinned a field value here
would pin an artefact of the element count.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "dockerfiles", "maxwell_solver"))

import paper1_wire_electrode as F  # noqa: E402


# --------------------------------------------------------------------------
# geometry: exact, and checked against closed forms rather than eyeballed
# --------------------------------------------------------------------------

def test_circumradius_comes_from_the_papers_own_hex_to_cylinder_relation():
    """2s = 1.14 D (paper 1 p. 10, S2), so the circumradius is s."""
    side = 1.14 * F.DIAMETER_NM / 2.0
    assert F.circumradius_nm() == pytest.approx(side, rel=1e-12)
    # 0.2 recorded the corner-to-corner width as 478.8 nm and the flat-to-flat
    # as 414.6 nm; both derive from this one number.
    radius = F.circumradius_nm()
    assert 2 * radius == pytest.approx(478.8, abs=0.1)
    assert 2 * radius * math.cos(math.pi / 6) == pytest.approx(414.6, abs=0.1)


def test_prism_volume_matches_the_closed_form_exactly():
    """A hexagon prism's volume is (3 sqrt(3)/2) r^2 H -- no approximation."""
    solid, _label, zs, _labels = F.stacked_solid_with_labels()
    radius = F.circumradius_nm() * F.NM
    expected = 3 * math.sqrt(3) / 2 * radius ** 2 * (zs[-1] - zs[0])
    assert solid.volume == pytest.approx(expected, rel=1e-9)


def test_the_solid_is_closed_and_orientable():
    """A PLC TetGen accepts must be watertight with one shell."""
    solid, _label, _zs, _labels = F.stacked_solid_with_labels()
    assert solid.is_watertight
    assert solid.euler_number == 2


def test_the_layer_boundaries_are_the_paper_thicknesses():
    """Assert the LITERAL paper numbers, not ``F.OXIDE_NM``.

    An earlier version compared each layer against the same module constant it
    was derived from, so changing ``OXIDE_NM`` from 12 to 50 still passed -- a
    test that cannot fail is not a test. The paper's numbers are written out
    here so a change to the constant has to argue with the paper.
    """
    _solid, _label, zs, labels = F.stacked_solid_with_labels()
    nm = [z / F.NM for z in zs]
    assert labels == ["mirror", "oxide", "wire", "top"]
    assert nm[1] - nm[0] == pytest.approx(120.0, abs=1e-6)   # gold mirror
    assert nm[2] - nm[1] == pytest.approx(12.0, abs=1e-6)    # "a 12 nm thick oxide" (p. 4)
    assert nm[3] - nm[2] == pytest.approx(1375.0, abs=1e-6)  # H (p. 3)
    assert nm[4] - nm[3] == pytest.approx(60.0, abs=1e-6)    # top electrode


def test_footprint_error_reports_the_pad_margin_rather_than_hiding_it():
    """The fixture drops the square pad; this is the number that says so."""
    hexagon, square, ratio = F.footprint_error()
    radius = F.circumradius_nm() * F.NM
    pad = radius + F.ELECTRODE_PAD_NM * F.NM
    assert hexagon == pytest.approx(3 * math.sqrt(3) / 2 * radius ** 2)
    assert square == pytest.approx((2 * pad) ** 2)
    assert ratio == pytest.approx(2.408, rel=1e-3)
    assert ratio > 1.0, "the square pad is larger than the hexagon, by definition"


# --------------------------------------------------------------------------
# permittivities: static, and deliberately NOT n^2
# --------------------------------------------------------------------------

def test_permittivities_are_static_and_not_the_optical_index_squared():
    mats = F.materials()
    assert mats["wire"] == pytest.approx(12.5)      # InP eps_s, Ioffe 300 K
    assert mats["oxide"] == pytest.approx(3.9)      # SiO2 eps_s
    # n = 3.44 (paper 1 p. 9, S1) is an OPTICAL index; n^2 = 11.83 is not eps_s.
    # The two are 5.6% apart and must never be substituted for one another.
    assert mats["wire"] != pytest.approx(3.44 ** 2, rel=1e-3)
    assert mats["wire"] > 3.44 ** 2


def test_every_part_has_a_permittivity_the_solver_can_use():
    """``solve`` refuses an eps_static below 1, so no part may be omitted."""
    mats = F.materials()
    _solid, _label, _zs, labels = F.stacked_solid_with_labels()
    assert set(mats) == set(labels)
    for name, value in mats.items():
        assert value >= 1.0, f"{name} has eps_s {value} < 1, which is not a material"


# --------------------------------------------------------------------------
# the mesh and the solve
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_mesh():
    return F.mesh_fixture(max_tets=8_000)


def test_mesh_covers_all_four_layers(coarse_mesh):
    parts = coarse_mesh["parts"]
    region = np.asarray(coarse_mesh["region_of_tet"])
    assert parts == ["mirror", "oxide", "wire", "top"]
    # Every layer must be present; an absent layer means label_fn's z-splits
    # do not match the geometry and the solve below is meaningless.
    for i, name in enumerate(parts):
        assert (region == i).sum() > 0, f"no tets in layer {name!r}"


def test_tets_straddle_layer_boundaries(coarse_mesh):
    # The labelling is per centroid, so it is only exact if no element spans a
    # boundary. It is NOT exact: this is one solid with no internal facets, so
    # TetGen spans elements across the layer planes. Pinning straddle > 0 is
    # what stops a future reader (or a future "the label is exact" comment)
    # from assuming the label is safe. If this ever fails because the mesher
    # gained internal facets, that is good news: revisit the UNVALIDATED status
    # in the module docstring rather than deleting the test.
    s = F.straddle_counts(coarse_mesh)
    assert s["n_tets"] > 0
    assert s["n_straddle"] > 0, (
        "no tet straddles a layer boundary; if the mesher now constrains to "
        "internal facets, the fixture may be mesh-convergent -- re-measure "
        "the spread in test_the_field_is_not_converged_in_the_mesh")
    # Both oxide faces are crossed, so the effect is on the layer that matters.
    # Boundaries sit near 120 nm and at 132 nm (mirror 120 + oxide 12). Keys are
    # float nm out of the dict, so match with a tolerance rather than ==.
    crossings = s["per_boundary_nm"]
    near = lambda target: [n for z, n in crossings.items() if abs(z - target) < 0.5]  # noqa: E731
    assert near(120.0) and near(120.0)[0] > 0, crossings
    assert near(132.0) and near(132.0)[0] > 0, crossings
    assert 0.0 < s["fraction"] < 1.0


def test_region_ids_are_integers_into_parts(coarse_mesh):
    region = np.asarray(coarse_mesh["region_of_tet"])
    assert region.dtype.kind == "i"
    assert region.min() >= 0
    assert region.max() < len(coarse_mesh["parts"])


def test_probe_sits_inside_the_mesh_and_outside_an_electrode(coarse_mesh):
    """``field_at`` refuses a point in an electrode: phi there is imposed."""
    probe = F.probe_point()
    assert probe[2] < (F.MIRROR_THICKNESS_NM + F.OXIDE_NM + F.HEIGHT_NM) * F.NM
    sol = F.solve_wire(coarse_mesh, 1.0)
    # This raises if the probe is outside the mesh or inside an electrode.
    field = F.E.field_at(sol, probe)
    assert np.all(np.isfinite(field))


def test_the_field_is_exactly_linear_in_the_applied_volts(coarse_mesh):
    """The problem is linear, so a departure is a solver bug, not physics."""
    report = F.field_report(coarse_mesh)
    lin = report["linearity"]
    assert lin["two_v_over_two_vs_one_v"] == pytest.approx(0.0, abs=1e-9)
    assert lin["minus_one_v_vs_one_v"] == pytest.approx(0.0, abs=1e-9)


def test_the_solver_runs_on_the_fixture(coarse_mesh):
    result = F.wire_field_per_volt(coarse_mesh)
    assert result["method"] in F.E.METHODS
    assert result["n_free"] > 0
    assert np.isfinite(result["magnitude_per_volt"])
    assert result["magnitude_per_volt"] > 0


def test_the_direct_and_amg_paths_agree_on_this_fixture(coarse_mesh):
    """Whichever path ``auto`` chose, the other one must land in the same place."""
    direct = F.wire_field_per_volt(coarse_mesh, method="direct")
    amg = F.wire_field_per_volt(coarse_mesh, method="amg")
    assert amg["magnitude_per_volt"] == pytest.approx(
        direct["magnitude_per_volt"], rel=1e-5)


# --------------------------------------------------------------------------
# the negative result: the field is NOT converged, and that must stay testable
# --------------------------------------------------------------------------

def test_the_field_is_not_converged_in_the_mesh():
    """The measured headline, pinned so it cannot be quietly 'fixed' by
    pegging a resolution and calling the answer converged.

    If this test starts failing because the spread DROPPED, the mesher gained a
    way to resolve the oxide and the fixture can be re-evaluated. It must not be
    relaxed to make a field value quotable.
    """
    values = []
    for n_modes in (20, 60):
        mesh = F.mesh_fixture(max_tets=400_000, n_modes=n_modes)
        values.append(F.wire_field_per_volt(mesh)["magnitude_per_volt"])
    spread = (max(values) - min(values)) / min(values)
    assert spread > 0.20, (
        f"field spread across mesh sizes is only {spread:.1%}; if the mesher "
        f"can now resolve the 12 nm oxide, revisit the fixture's UNVALIDATED "
        f"status rather than deleting this test"
    )
