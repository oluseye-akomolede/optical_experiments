"""Laying a td upload out on its grid (spec: "A td upload SHALL be laid out on
its grid by one fixed rule", after the dca89b2 review fixes).

Fixtures are tiny boxes in mesh units with a length scale, so every grid stays
a few tens of nodes per axis. Expected positions come from the spec's rule
(16 padding cells, quarter-extent spacing, conductor refusal), not from the
layout's own constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))
import fdtd_layout as L  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    return box


def _request(**over):
    req = {"boundary": "metal", "source_kind": "point", "profile": "pulse",
           "frequency_hz": 3.0e14, "bandwidth": 1.0e14, "position": [0, 0, 0],
           "direction": None, "polarisation": "z", "samples": None,
           "sample_duration_s": None, "duration_s": 1e-13, "probes": [],
           "dx_m": 1.0}
    req.update(over)
    return req


def test_the_frame_is_the_bounding_box_corner_in_metres():
    part = _box((2, 0, 0), (6, 3, 2))           # mm
    dx = 1e-3
    req = _request(dx_m=dx, position=[1e-3, 1e-3, 1e-3])
    metal = L.layout(["a"], [part], req, length_scale=1e-3, n_index=1.5)
    x = metal["nodes"][0]
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[-1] == pytest.approx(4e-3)
    opened = L.layout(["a"], [part], dict(req, boundary="open"),
                      length_scale=1e-3, n_index=1.5)
    x = opened["nodes"][0]
    assert x[0] == pytest.approx(-16 * dx)
    assert x[-1] == pytest.approx(4e-3 + 16 * dx)


def test_a_part_thinner_than_four_cells_gets_four_cells():
    # a 1.5 dx plate in z: at least 4 cells across it, none wider than a quarter
    dx = 1e-3
    plate = _box((0, 0, 0), (6, 5, 1.5))
    out = L.layout(["p"], [plate], _request(dx_m=dx, position=[1e-3, 1e-3, 0.5e-3]),
                   length_scale=1e-3, n_index=2.0)
    z = out["nodes"][2]
    across = np.diff(z[(z >= -1e-12) & (z <= 1.5e-3 + 1e-12)])
    assert len(across) >= 4
    assert across.max() <= 1.5e-3 / 4 * (1 + 1e-9)


def test_a_curved_part_carries_the_caveat_and_a_box_does_not():
    dx = 0.5e-3
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
    sphere.apply_translation((1.5, 1.5, 1.5))
    req = _request(dx_m=dx, position=[1.5e-3, 1.5e-3, 1.5e-3])
    curved = L.layout(["s"], [sphere], req, length_scale=1e-3, n_index=1.5)
    assert any(L.THIN_CAVEAT in c and "s" in c for c in curved["caveats"])
    boxed = L.layout(["b"], [_box((0, 0, 0), (3, 3, 3))], req,
                     length_scale=1e-3, n_index=1.5)
    assert boxed["caveats"] == []


def _two_boxes():
    # a gap between two dielectric boxes is the metal boundary's conductor
    return ["left", "right"], [_box((0, 0, 0), (2, 3, 3)), _box((4, 0, 0), (6, 3, 3))]


def test_a_source_in_the_metal_boundary_is_refused_naming_it():
    names, parts = _two_boxes()
    req = _request(dx_m=0.5e-3, position=[3e-3, 1.5e-3, 1.5e-3])
    with pytest.raises(L.LayoutError, match=r"source position.*metal boundary"):
        L.layout(names, parts, req, length_scale=1e-3,
                 materials={"left": 1.5, "right": 1.5})


def test_a_source_inside_a_pec_part_is_refused_naming_the_part():
    names = ["shell", "core"]
    parts = [_box((0, 0, 0), (6, 4, 4)), _box((2, 1, 1), (4, 3, 3))]
    req = _request(dx_m=0.5e-3, position=[3e-3, 2e-3, 2e-3])
    with pytest.raises(L.LayoutError, match=r"source position.*part core"):
        L.layout(names, parts, req, length_scale=1e-3,
                 materials={"shell": 1.5, "core": "pec"})


def test_a_probe_inside_a_rakic_metal_is_refused_naming_the_part():
    # nm-scale: a 60 nm gold cube inside a 200 nm dielectric cube
    names = ["host", "gold"]
    parts = [_box((0, 0, 0), (200, 200, 200)), _box((70, 70, 70), (130, 130, 130))]
    req = _request(dx_m=50e-9, position=[30e-9, 30e-9, 30e-9],
                   probes=[[100e-9, 100e-9, 100e-9, "z"]])
    with pytest.raises(L.LayoutError, match=r"probe.*part gold"):
        L.layout(names, parts, req, length_scale=1e-9,
                 materials={"host": 1.5, "gold": "Au"})


def test_a_source_and_probe_in_a_dielectric_are_laid_out():
    # the positive control for the three refusals above: same scenes, legal spots
    names, parts = _two_boxes()
    out = L.layout(names, parts,
                   _request(dx_m=0.5e-3, position=[1e-3, 1.5e-3, 1.5e-3],
                            probes=[[5e-3, 1.5e-3, 1.5e-3, "x"]]),
                   length_scale=1e-3, materials={"left": 1.5, "right": 1.5})
    assert out["cell"] is not None and len(out["probes"]) == 1


def test_a_thin_gold_part_keeps_the_skin_depth_spacing():
    # 60 nm of gold is under 4 coarse cells (thin rule: 15 nm) and under 6 skin
    # depths (skin rule: depth/4, ~6 nm here): the smaller spacing must win
    import fdtd_grid
    names = ["host", "gold"]
    parts = [_box((0, 0, 0), (200, 200, 200)), _box((70, 70, 70), (130, 130, 130))]
    req = _request(dx_m=50e-9, position=[30e-9, 30e-9, 30e-9])
    out = L.layout(names, parts, req, length_scale=1e-9,
                   materials={"host": 1.5, "gold": "Au"})
    cap = fdtd_grid.skin_depth("Au", 4.0e14) / 4
    assert cap < 60e-9 / 4                       # the fixture tests the min
    for axis in range(3):
        n = out["nodes"][axis]
        inside = np.diff(n[(n >= 70e-9 - 1e-15) & (n <= 130e-9 + 1e-15)])
        assert inside.max() <= cap * (1 + 1e-9), axis


@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize("what", ["source", "probe"])
def test_a_source_or_probe_inside_the_cpml_is_refused(what, side):
    # open, dx 1 mm: the padding is 16 cells and the CPML its outer 10, so node
    # 9 (-7 mm) is the layer's last node and node 10 (-6 mm) the first legal one
    part = _box((0, 0, 0), (3, 3, 3))
    inside_pml, legal = [-7e-3, 1e-3, 1e-3], [-6e-3, 1e-3, 1e-3]
    if side == "high":                 # mirrored past the part's 3 mm far face
        inside_pml, legal = [10e-3, 1e-3, 1e-3], [9e-3, 1e-3, 1e-3]
    req = _request(boundary="open", dx_m=1e-3, position=legal)
    if what == "source":
        bad = dict(req, position=inside_pml)
    else:
        bad = dict(req, probes=[inside_pml + ["z"]])
    L.layout(["a"], [part], dict(req, probes=[legal + ["z"]]),
             length_scale=1e-3, n_index=1.5)            # positive control
    with pytest.raises(L.LayoutError, match=r"absorbing layer.*x must lie"):
        L.layout(["a"], [part], bad, length_scale=1e-3, n_index=1.5)


def test_gold_thinner_than_its_skin_depth_takes_the_thickness_rule():
    # 10 nm of gold in z: extent/4 (2.5 nm) is under skin depth/4 here, so the
    # min must pick the thickness rule; the 60 nm fixture above tests the
    # other order
    import fdtd_grid
    names = ["host", "gold"]
    parts = [_box((0, 0, 0), (200, 200, 200)), _box((70, 70, 95), (130, 130, 105))]
    req = _request(dx_m=50e-9, position=[30e-9, 30e-9, 30e-9])
    out = L.layout(names, parts, req, length_scale=1e-9,
                   materials={"host": 1.5, "gold": "Au"})
    cap = 10e-9 / 4
    assert cap < fdtd_grid.skin_depth("Au", 4.0e14) / 4   # the fixture tests the min
    z = out["nodes"][2]
    inside = np.diff(z[(z >= 95e-9 - 1e-15) & (z <= 105e-9 + 1e-15)])
    assert inside.max() <= cap * (1 + 1e-9), (
        f"a 10 nm gold film got a {inside.max():.3g} m cell; the thickness rule wants {cap:.3g}")



def test_a_film_two_coarse_cells_thick_is_resolved_by_the_layout_rule():
    # 20 nm of gold at dx 10 nm: two coarse cells, so build_grid's own
    # thin-layer floor (which only acts below ONE coarse cell) does not apply,
    # and only the layout's min(extent/4, skin/4) = 5 nm resolves it
    import fdtd_grid
    names = ["host", "gold"]
    parts = [_box((0, 0, 0), (200, 200, 200)), _box((70, 70, 90), (130, 130, 110))]
    req = _request(dx_m=10e-9, position=[30e-9, 30e-9, 30e-9])
    out = L.layout(names, parts, req, length_scale=1e-9,
                   materials={"host": 1.5, "gold": "Au"})
    cap = 20e-9 / 4
    assert 20e-9 >= 10e-9                  # not under one coarse cell
    assert cap < fdtd_grid.skin_depth("Au", 4.0e14) / 4   # the min picks extent/4
    z = out["nodes"][2]
    inside = np.diff(z[(z >= 90e-9 - 1e-15) & (z <= 110e-9 + 1e-15)])
    assert inside.max() <= cap * (1 + 1e-9), (
        f"a 20 nm gold film at dx 10 nm got a {inside.max():.3g} m cell; "
        f"the layout's rule wants {cap:.3g}")

@pytest.mark.parametrize("boundary,far", [("metal", 5e-3), ("open", 25e-3)])
@pytest.mark.parametrize("what", ["source", "probe"])
def test_a_source_or_probe_outside_the_grid_is_refused_naming_the_extent(
        what, boundary, far):
    # metal: the grid is the part's 0..3 mm; open adds 16 mm of padding, so
    # 25 mm is off the grid either way
    part = _box((0, 0, 0), (3, 3, 3))
    legal, off = [1e-3, 1e-3, 1e-3], [far, 1e-3, 1e-3]
    req = _request(boundary=boundary, dx_m=1e-3, position=legal)
    L.layout(["a"], [part], dict(req, probes=[legal + ["z"]]),
             length_scale=1e-3, n_index=1.5)            # positive control
    bad = (dict(req, position=off) if what == "source"
           else dict(req, probes=[off + ["z"]]))
    with pytest.raises(L.LayoutError, match=r"outside the grid: x spans -?[0-9.e-]+\.\.[0-9.e-]+ m"):
        L.layout(["a"], [part], bad, length_scale=1e-3, n_index=1.5)


@pytest.mark.parametrize("profile,top", [("pulse", 4.0e14), ("cw", 3.0e14),
                                         ("drawn", 3.0e14)])
def test_the_band_top_adds_the_bandwidth_only_for_a_pulse(profile, top):
    assert L.band_top(profile, 3.0e14, 1.0e14) == pytest.approx(top, rel=1e-12)


def test_thick_gold_gets_a_skin_layer_inside_both_faces():
    # 400 nm of gold is over 6 skin depths: each bounding face gets a layer 3
    # skin depths deep at depth/4, the low face AND the high face
    import fdtd_grid
    names = ["host", "gold"]
    parts = [_box((0, 0, 0), (800, 800, 800)), _box((200, 200, 200), (600, 600, 600))]
    req = _request(dx_m=100e-9, position=[100e-9, 100e-9, 100e-9])
    out = L.layout(names, parts, req, length_scale=1e-9,
                   materials={"host": 1.5, "gold": "Au"})
    depth = fdtd_grid.skin_depth("Au", 4.0e14)
    assert 400e-9 > 2 * L.SKIN_DEPTHS * depth            # the fixture is thick
    fine = depth / L.CELLS_PER_SKIN
    for axis in range(3):
        n = out["nodes"][axis]
        for face, (a, b) in (("low", (200e-9, 200e-9 + 3 * depth)),
                             ("high", (600e-9 - 3 * depth, 600e-9))):
            cells = np.diff(n[(n >= a - 1e-15) & (n <= b + 1e-15)])
            assert len(cells) and cells.max() <= fine * (1 + 1e-9), (
                f"{'xyz'[axis]} {face} face of the gold: {cells.max():.3g} m "
                f"cells where the skin layer wants {fine:.3g}")


def test_the_plane_wave_box_is_the_bounding_box_grown_by_the_clearance():
    import fdtd
    part = _box((0, 0, 0), (4, 3, 2))
    req = _request(boundary="open", dx_m=1e-3, source_kind="plane",
                   direction="+z", polarisation="x", position=None)
    out = L.layout(["a"], [part], req, length_scale=1e-3, n_index=1.5)
    size = (4e-3, 3e-3, 2e-3)
    lo = tuple(int(np.argmin(np.abs(n))) for n in out["nodes"])
    hi = tuple(int(np.argmin(np.abs(n - s))) for n, s in zip(out["nodes"], size))
    c = fdtd.BOX_CLEARANCE
    assert c == 3
    assert out["box"] == (tuple(i - c for i in lo), tuple(i + c for i in hi)), (
        f"box {out['box']} is not the faces {lo}..{hi} grown by {c} cells")
    assert out["cell"] is None


def test_an_unknown_boundary_and_a_plane_wave_in_metal_are_refused():
    part = _box((0, 0, 0), (3, 3, 3))
    with pytest.raises(L.LayoutError, match=r"boundary 'periodic' must be one of metal, open"):
        L.layout(["a"], [part], _request(boundary="periodic", dx_m=1e-3),
                 length_scale=1e-3, n_index=1.5)
    with pytest.raises(L.LayoutError, match=r"plane wave needs 'open'"):
        L.layout(["a"], [part], _request(source_kind="plane", direction="+z",
                                         polarisation="x", position=None, dx_m=1e-3),
                 length_scale=1e-3, n_index=1.5)


# --- plan / estimate / refuse_conductor_points: the form's cheap half ------

def _gold_scene(shape="box"):
    names = ["host", "gold"]
    gold = (_box((70, 70, 70), (130, 130, 130)) if shape == "box"
            else trimesh.creation.icosphere(subdivisions=2, radius=30.0)
            .apply_translation((100, 100, 100)))
    return names, [_box((0, 0, 0), (200, 200, 200)), gold]


@pytest.mark.parametrize("boundary", ["metal", "open"])
def test_the_plan_lays_out_the_nodes_the_layout_solves_on(boundary):
    names, parts = _gold_scene()
    req = _request(boundary=boundary, dx_m=50e-9, position=[30e-9, 30e-9, 30e-9],
                   probes=[[170e-9, 30e-9, 30e-9, "x"]])
    kw = dict(length_scale=1e-9, materials={"host": 1.5, "gold": "Au"})
    planned, laid = L.plan(names, parts, req, **kw), L.layout(names, parts, req, **kw)
    for a in range(3):
        np.testing.assert_array_equal(planned["nodes"][a], laid["nodes"][a])
    assert planned["cell"] == laid["cell"] and planned["probes"] == laid["probes"]
    assert planned["box"] == laid["box"]


@pytest.mark.parametrize("shape", ["box", "sphere"])
@pytest.mark.parametrize("boundary", ["metal", "open"])
def test_the_form_estimate_counts_the_solved_steps_and_never_less_memory(
        shape, boundary):
    import fdtd
    names, parts = _gold_scene(shape)
    req = _request(boundary=boundary, dx_m=50e-9, position=[30e-9, 30e-9, 30e-9],
                   duration_s=2e-15)
    kw = dict(length_scale=1e-9, materials={"host": 1.5, "gold": "Au"})
    form = L.estimate(L.plan(names, parts, req, **kw), req)
    laid = L.layout(names, parts, req, **kw)
    solved = fdtd.estimate_run(laid["nodes"], duration=req["duration_s"],
                               boundary=boundary, metals=laid["metals"],
                               pec=laid["pec"], record_frames=True)
    assert laid["metals"], "the gold must voxelise, or this compares nothing"
    for key in ("cells", "dt", "steps", "setup"):
        assert form[key] == solved[key], key
    assert form["bytes"] >= solved["bytes"]
    assert form["bytes"] <= 1.05 * solved["bytes"], "the bound is loose"


def test_the_form_refuses_a_source_in_the_metal_boundary_naming_it():
    names, parts = _two_boxes()
    req = _request(dx_m=0.5e-3, position=[3e-3, 1.5e-3, 1.5e-3])
    planned = L.plan(names, parts, req, length_scale=1e-3,
                     materials={"left": 1.5, "right": 1.5})
    with pytest.raises(L.LayoutError, match=r"source position.*metal boundary"):
        L.refuse_conductor_points(planned, parts, length_scale=1e-3)


def test_the_form_refuses_a_source_inside_a_pec_part_naming_the_part():
    names = ["shell", "core"]
    parts = [_box((0, 0, 0), (6, 4, 4)), _box((2, 1, 1), (4, 3, 3))]
    req = _request(boundary="open", dx_m=0.5e-3, position=[3e-3, 2e-3, 2e-3])
    planned = L.plan(names, parts, req, length_scale=1e-3,
                     materials={"shell": 1.5, "core": "pec"})
    with pytest.raises(L.LayoutError, match=r"source position.*part core"):
        L.refuse_conductor_points(planned, parts, length_scale=1e-3)


def test_the_form_refuses_a_probe_inside_a_rakic_metal_naming_the_part():
    names, parts = _gold_scene()
    req = _request(boundary="open", dx_m=50e-9, position=[30e-9, 30e-9, 30e-9],
                   probes=[[100e-9, 100e-9, 100e-9, "z"]])
    planned = L.plan(names, parts, req, length_scale=1e-9,
                     materials={"host": 1.5, "gold": "Au"})
    with pytest.raises(L.LayoutError, match=r"probe.*part gold"):
        L.refuse_conductor_points(planned, parts, length_scale=1e-9)


def test_the_form_passes_a_source_and_probe_in_a_dielectric():
    # the positive control for the three refusals above
    names, parts = _two_boxes()
    req = _request(dx_m=0.5e-3, position=[1e-3, 1.5e-3, 1.5e-3],
                   probes=[[5e-3, 1.5e-3, 1.5e-3, "x"]])
    planned = L.plan(names, parts, req, length_scale=1e-3,
                     materials={"left": 1.5, "right": 1.5})
    assert L.refuse_conductor_points(planned, parts, length_scale=1e-3) is None


def test_the_form_estimate_counts_a_pec_parts_conductor_mask():
    import fdtd
    names = ["shell", "core"]
    parts = [_box((0, 0, 0), (6, 4, 4)), _box((2, 1, 1), (4, 3, 3))]
    req = _request(boundary="open", dx_m=0.5e-3, position=[1e-3, 2e-3, 2e-3],
                   duration_s=1e-12)
    kw = dict(length_scale=1e-3, materials={"shell": 1.5, "core": "pec"})
    form = L.estimate(L.plan(names, parts, req, **kw), req)
    laid = L.layout(names, parts, req, **kw)
    assert laid["pec"] is not None, "the core must voxelise as pec"
    solved = fdtd.estimate_run(laid["nodes"], duration=req["duration_s"],
                               boundary="open", metals=laid["metals"],
                               pec=laid["pec"], record_frames=True)
    assert form["bytes"] == solved["bytes"]
