"""Conforming layered meshes for the electrostatics solver (task 1.6c).

Three claims, each with a control that shows the check can fail:

1. No tet straddles a material interface (``layer_straddles`` is zero), on the
   thin-layer capacitor and on the paper-1 stack. Control: the same PLC without
   its internal facets straddles, and the counter sees it.
2. A thin layer has at least ``n_across`` elements across it, and the mesh is
   graded: the thin layer's elements are finer than the bulk's.
3. The answers: the capacitor matches the exact 1-D series capacitor at every
   step of a refinement ladder, and the paper-1 field at the dot converges on
   the layered mesh. Controls: the centroid-labelled one-solid mesh misses the
   capacitor by percent, and the 1.6b one-part negative test is left exactly as
   it was (``test_paper1_wire_electrode.py``).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "dockerfiles", "maxwell_solver"))

import electrostatics_bench as BENCH  # noqa: E402
import mse_bench_fixtures as BF  # noqa: E402
import mse_electrostatics as E  # noqa: E402
import mse_meshing as MESH  # noqa: E402
import paper1_wire_electrode as F  # noqa: E402

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _capacitance(problem) -> float:
    C, _ = E.capacitance_matrix(
        problem["points_m"], problem["tets"], problem["parts"],
        problem["region_of_tet"], problem["eps_static"], problem["electrodes"],
        method="direct")
    return float(abs(C[0, 0]))


@pytest.fixture(scope="module")
def capacitor():
    return BF.thin_layer_capacitor(20e-9)


@pytest.fixture(scope="module")
def paper_layered():
    edge_nm, n_across = F.LAYERED_LADDER[0]
    return F.mesh_fixture_layered(edge_nm=edge_nm, n_across=3)


# --------------------------------------------------------------------------
# 1. no tet straddles an interface
# --------------------------------------------------------------------------

def test_no_tet_straddles_an_interface_in_the_capacitor(capacitor):
    problem, _ = capacitor
    s = MESH.layer_straddles(problem["points_m"], problem["tets"],
                             problem["region_of_tet"], problem["planes_m"])
    assert s["n_tets"] > 0
    assert s["crossing"] == 0, s
    assert s["mislabelled"] == 0, s
    # Every layer, electrodes included, is its own non-empty region.
    counts = np.bincount(problem["region_of_tet"], minlength=len(problem["parts"]))
    assert np.all(counts > 0), counts


def test_no_tet_straddles_an_interface_in_the_paper1_stack(paper_layered):
    s = MESH.layer_straddles(paper_layered["points_m"], paper_layered["tets"],
                             paper_layered["region_of_tet"], paper_layered["planes_m"])
    assert s["crossing"] == 0, s
    assert s["mislabelled"] == 0, s
    assert len(s["crossing_per_plane"]) == 3   # mirror|oxide, oxide|wire, wire|top


def test_the_straddle_counter_sees_a_straddle():
    """Control: without internal facets the same stack straddles, and both counts say so."""
    problem, _ = BF.thin_layer_capacitor(20e-9, conforming=False)
    s = MESH.layer_straddles(problem["points_m"], problem["tets"],
                             problem["region_of_tet"], problem["planes_m"])
    assert s["crossing"] > 0
    assert s["mislabelled"] > 0


def test_the_label_check_catches_a_conforming_mesh_tagged_one_layer_off(capacitor):
    """``mislabelled`` reads the label against the geometry, so a shifted label fails
    even though no tet crosses a plane."""
    problem, _ = capacitor
    shifted = (problem["region_of_tet"] + 1) % len(problem["parts"])
    s = MESH.layer_straddles(problem["points_m"], problem["tets"], shifted,
                             problem["planes_m"])
    assert s["crossing"] == 0
    assert s["mislabelled"] == s["n_tets"]


def test_region_ids_are_tetgen_attributes_not_centroids(capacitor):
    """Every tet's centroid lies in the layer its id names (a consequence, checked)."""
    problem, _ = capacitor
    cz = problem["points_m"][problem["tets"]][:, :, 2].mean(axis=1)
    planes = problem["planes_m"]
    by_centroid = np.searchsorted(planes, cz, side="right") - 1
    assert np.array_equal(by_centroid, problem["region_of_tet"])


# --------------------------------------------------------------------------
# 2. elements across a thin layer, and grading
# --------------------------------------------------------------------------

def test_a_thin_layer_has_at_least_n_across_elements(capacitor):
    problem, _ = capacitor
    report = {r["layer"]: r for r in MESH.layer_mesh_report(
        problem["points_m"], problem["tets"], problem["region_of_tet"],
        problem["layers"])}
    n = MESH.LAYER_ELEMENTS_ACROSS
    assert report["thin"]["elements_across"] >= n * (1 - 1e-9), report["thin"]


def test_the_paper1_oxide_has_three_elements_across_its_12_nm(paper_layered):
    report = {r["layer"]: r for r in MESH.layer_mesh_report(
        paper_layered["points_m"], paper_layered["tets"],
        paper_layered["region_of_tet"], paper_layered["layers"])}
    oxide = report["oxide"]
    assert oxide["thickness_m"] == pytest.approx(12e-9)
    assert oxide["elements_across"] >= 3 * (1 - 1e-9), oxide
    # 1.6b measured the one-part mesh's MEDIAN z-span at 20.4 nm (n_modes 200).
    # Here the WORST oxide element spans one sub-slab.
    assert oxide["max_zspan_m"] <= 4e-9 * (1 + 1e-9)


def test_the_mesh_is_graded_toward_the_thin_layer(paper_layered):
    """Element size grows with distance from the oxide: grading, not a uniform mesh.

    Measured by the longest edge per tet, binned by the centroid's height above
    the oxide -- NOT by a per-layer median, which the many small tets next to
    the oxide dominate (the wire's median edge is 12 nm for exactly that reason).
    Measured at this mesh: 10.5 nm in the oxide, 52 nm at 170-270 nm above it,
    ~300 nm at 670-1070 nm above it.
    """
    pts, tets = paper_layered["points_m"], paper_layered["tets"]
    xyz = pts[tets]
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    longest = np.linalg.norm(xyz[:, pairs[:, 0]] - xyz[:, pairs[:, 1]], axis=2).max(axis=1)
    planes = paper_layered["planes_m"]
    above = xyz[:, :, 2].mean(axis=1) - planes[2]           # height above the oxide
    oxide = np.median(longest[paper_layered["region_of_tet"] == 1])
    near = np.median(longest[(above > 0) & (above < 50e-9)])
    far = np.median(longest[(above > 600e-9) & (above < 1100e-9)])
    assert oxide < near < far
    assert far > 10 * oxide, (oxide, near, far)


def test_a_thick_layer_is_not_subdivided():
    layers = [("a", 10.0), ("thin", 0.1), ("b", 10.0)]
    assert MESH.layer_subdivisions(layers, 1.0, 3) == [1, 3, 1]


# --------------------------------------------------------------------------
# 3a. the capacitor against the exact 1-D answer, over a refinement ladder
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ladder_rows():
    return BENCH.layered_rows()


def test_the_capacitor_matches_the_exact_series_answer_at_every_step(ladder_rows):
    conforming = [r for r in ladder_rows if r["conforming"]]
    assert len(conforming) == len(BENCH.LAYERED_LADDER)
    tets = [r["n_tets"] for r in conforming]
    assert tets == sorted(tets) and tets[0] < tets[-1], tets   # it IS a refinement
    for r in conforming:
        assert r["straddling"] == 0
        assert r["c_err"] < BENCH.TOL_LAYERED, r
        assert r["e_thin_err"] < BENCH.TOL_LAYERED, r


def test_the_centroid_labelled_control_misses_by_percent(ladder_rows):
    """The control the conforming mesh replaces: C off by > 1 % at every step."""
    control = [r for r in ladder_rows if not r["conforming"]]
    assert len(control) == len(BENCH.LAYERED_LADDER)
    for r in control:
        assert r["straddling"] > 0
        assert r["c_err"] > 1e-2, r


def test_the_capacitor_sees_the_thin_layers_permittivity():
    """A wrong eps in the 10 nm layer must move C by far more than TOL_LAYERED."""
    problem, expected = BF.thin_layer_capacitor(40e-9)
    problem = dict(problem, eps_static=dict(problem["eps_static"]))
    problem["eps_static"]["thin"] *= 1.01
    err = abs(_capacitance(problem) - expected["C"]) / expected["C"]
    assert err > 1e3 * BENCH.TOL_LAYERED


# --------------------------------------------------------------------------
# 3b. the paper-1 field at the dot, on the layered mesh (the 1.6b check, rerun)
# --------------------------------------------------------------------------

def test_the_paper1_field_converges_on_the_layered_mesh():
    """1.6b's check on the LAYERED mesh: two mesh sizes, the same 20 % bar.

    1.6b's own test (``test_the_field_is_not_converged_in_the_mesh``) asserts
    the ONE-PART mesh's spread is above 20 %; it is unchanged and still runs on
    that mesh. Here the same bar is held from the other side, and the measured
    spread (7.9e-15 over the three-step ladder) is far inside it, so the value
    is also checked against the fixture's exact 1-D answer.
    """
    values = []
    for edge_nm, n_across in F.LAYERED_LADDER[:2]:
        mesh = F.mesh_fixture_layered(edge_nm=edge_nm, n_across=n_across)
        values.append(F.wire_field_per_volt(mesh)["magnitude_per_volt"])
    spread = (max(values) - min(values)) / min(values)
    assert spread < 0.20
    for v in values:
        assert v == pytest.approx(F.field_per_volt_1d(), rel=1e-9)


def test_the_paper1_1d_value_is_the_series_formula():
    """Typed from the formula, not from the fixture's helper."""
    s = 12e-9 / 3.9 + 1375e-9 / 12.5
    assert F.field_per_volt_1d() == pytest.approx(1.0 / (12.5 * s), rel=1e-12)


def test_the_fixture_reads_inp_eps_from_the_materials_module():
    import dot_material as DOT
    assert F.EPS_INP == DOT.INP_EPS_S.value


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_a_non_convex_outline_is_refused():
    notch = [(0, 0), (2, 0), (2, 2), (1, 0.5), (0, 2)]
    with pytest.raises(MESH.MeshingError, match="convex"):
        MESH.layer_stack_plc(notch, [("a", 1.0)], 0.5)


@pytest.mark.parametrize("layers, match", [
    ([("a", 1.0), ("a", 1.0)], "unique"),
    ([("a", 0.0)], "thickness"),
    ([("a", -1.0)], "thickness"),
])
def test_bad_layers_are_refused(layers, match):
    with pytest.raises(MESH.MeshingError, match=match):
        MESH.layer_stack_plc(SQUARE, layers, 0.5)


def test_a_layered_mesh_over_the_cap_is_refused():
    with pytest.raises(MESH.MeshingError, match="limit"):
        MESH.tetrahedralize_layers(SQUARE, [("a", 1.0), ("t", 0.05), ("b", 1.0)],
                                   0.5, max_tets=100)


def test_a_clockwise_outline_meshes_the_same_as_counter_clockwise():
    layers = [("a", 0.5), ("t", 0.1), ("b", 0.5)]
    ccw = MESH.tetrahedralize_layers(SQUARE, layers, 0.3)
    cw = MESH.tetrahedralize_layers(SQUARE[::-1], layers, 0.3)
    for mesh in (ccw, cw):
        s = MESH.layer_straddles(*mesh, MESH.layer_planes(layers))
        assert s["crossing"] == 0 and s["mislabelled"] == 0
