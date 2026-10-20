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


def test_a_slab_just_over_six_skin_depths_gets_no_sliver():
    # 180 nm of gold at 4e14 Hz is just over 6 skin depths (173.9 nm). As two
    # face layers it leaves a 6.1 nm bulk strip in the middle, under one 50 nm
    # coarse cell, and the grid's sub-coarse gap rule (graded_axis, MIN_CELLS=4)
    # then splits that strip into 4 cells of 1.53 nm -- finer than the skin
    # layers themselves. That is what Run C's 150 nm gold mirror at dx 10 nm
    # did under Au-palik: a 3.21 nm strip, 0.801 nm cells (7.6x under the
    # 6.116 nm skin spacing), and 378,896 steps for a record that needs
    # 121,121. The part must get one layer across its whole extent instead,
    # and no cell across it under depth/4.
    #
    # The second case is the band the coarse-cell threshold alone leaves open:
    # 190 nm at dx 10 nm leaves a 16.1 nm middle, wider than a coarse cell but
    # under one skin depth, and the two-layer branch fills that middle with 4
    # cells of 4.028 nm -- 0.56x the 7.245 nm spacing.
    import fdtd_grid
    depth = fdtd_grid.skin_depth("Au", 4.0e14)
    cap = depth / L.CELLS_PER_SKIN
    for thickness, dx, above_dx in ((180e-9, 50e-9, False),
                                    (190e-9, 10e-9, True)):
        names = ["host", "gold"]
        parts = [_box((0, 0, 0), (400, 400, 400)),
                 _box((100, 100, 110), (300, 300, 110 + thickness / 1e-9))]
        req = _request(dx_m=dx, position=[30e-9, 30e-9, 30e-9])
        middle = thickness - 2 * L.SKIN_DEPTHS * depth
        assert 0 < middle < max(dx, depth)   # the one-layer branch's band
        assert bool(middle > dx) == above_dx  # this fixture is what it says
        out = L.layout(names, parts, req, length_scale=1e-9,
                       materials={"host": 1.5, "gold": "Au"})
        z = out["nodes"][2]
        inside = np.diff(z[(z >= 110e-9 - 1e-15)
                           & (z <= (110 + thickness / 1e-9) * 1e-9 + 1e-15)])
        # one layer of thickness/cap cells, each within the count's rounding
        # of cap; a split part instead leaves 4 cells under a fifth of cap
        assert inside.size <= int(np.ceil(thickness / cap)), inside.size
        assert inside.min() >= 0.5 * cap, (
            f"{thickness / 1e-9:.0f} nm of gold at dx {dx / 1e-9:.0f} nm left "
            f"a {middle * 1e9:.2g} nm middle and got a {inside.min():.3g} m "
            f"cell; one layer across the part wants {cap:.3g}")


# plate thicknesses (um) where, before 5.4d, the 0.2 um film read 1.19, 1.22
# and 1.43 times its thickness: its faces were off the grid's nodes
@pytest.mark.parametrize("lo,hi", [(0.1973, 0.2285), (0.2459, 0.3846),
                                   (0.2846, 0.3275)])
def test_a_film_flush_between_pec_plates_keeps_its_thickness_on_the_layouts_grid(
        lo, hi):
    t, z1 = 0.2, lo
    z2 = z1 + t
    parts = [_box((0, 0, 0), (1, 1, z1)), _box((0, 0, z1), (1, 1, z2)),
             _box((0, 0, z2), (1, 1, z2 + hi))]
    req = _request(dx_m=0.05e-6, position=[0.5e-6, 0.5e-6, (z1 + t / 2) * 1e-6])
    out = L.layout(["lo", "film", "hi"], parts, req, length_scale=1e-6,
                   materials={"lo": "pec", "film": 1.5, "hi": "pec"})
    x, y, z = (n / 1e-6 for n in out["nodes"])
    col = out["pec"][int(np.argmin(abs(x - 0.5))), int(np.argmin(abs(y - 0.5))), :]
    free = np.flatnonzero(~col)
    k = free[np.argmin(abs(z[free] - (z1 + t / 2)))]
    below = k - np.flatnonzero(col[k::-1])[0]
    above = k + np.flatnonzero(col[k:])[0]
    assert z[above] - z[below] == pytest.approx(t, rel=1e-6), (
        f"the 0.2 um film reads {z[above] - z[below]:.4g} um between its walls")


def test_a_curved_part_adds_no_face_line():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
    plate = _box((0, 0, 0), (3, 3, 0.4))
    assert all(L._flat_faces(sphere, axis, 1.0) == [] for axis in range(3))
    assert L._flat_faces(plate, 2, 1.0) == [0.0, 0.4]


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


# --- coatings (task 5.4b) ---------------------------------------------------

def _coated_core(boundary="open", **over):
    """A 120 x 80 x 100 nm core of n = 1.5 coated in 10 nm of gold."""
    req = _request(**{"boundary": boundary, "dx_m": 40e-9,
                      "position": [60e-9, 40e-9, -30e-9],
                      "coatings": {"core": ["Au", 10e-9]}, **over})
    return ["core"], [_box((0, 0, 0), (120, 80, 100))], req


def _at(nodes, value):
    i = int(np.argmin(np.abs(nodes - value)))
    assert abs(nodes[i] - value) < 1e-15, f"no node at {value}"
    return i


def test_a_gold_coated_core_lays_out_as_a_gold_shell_over_its_core():
    names, parts, req = _coated_core()
    out = L.layout(names, parts, req, length_scale=1e-9, materials={"core": 1.5})
    x, y, z = out["nodes"]
    # the shell's faces are grid planes on every axis (flat faces)
    for n, (a, b) in zip(out["nodes"], ((0, 120e-9), (0, 80e-9), (0, 100e-9))):
        _at(n, a - 10e-9), _at(n, a), _at(n, b), _at(n, b + 10e-9)
    gold = out["metals"]["Au"]
    j, k = int(np.argmin(np.abs(y - 40e-9))), int(np.argmin(np.abs(z - 50e-9)))
    # along x through the middle: vacuum, gold shell, core, gold shell, vacuum
    inside_shell = (x > -10e-9 + 1e-15) & (x < -1e-15)
    assert inside_shell.any() and gold[inside_shell, j, k].all()
    core = (x > 1e-15) & (x < 120e-9 - 1e-15)
    assert not gold[core, j, k].any()
    assert np.allclose(out["eps"][core, j, k], 2.25)
    far = (x < -20e-9) | (x > 140e-9)
    assert not gold[far, j, k].any()
    assert np.allclose(out["eps"][far, j, k], 1.0)


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (12.5, 48.4, 58.3),
                                    (-731.1, 3.7e3, 0.29)])
def test_every_face_of_a_coating_is_the_same_number_of_nodes_thick(offset):
    """The reviewer's 12 face planes: 50 nm of gold on a 306 x 333 x 504 nm
    n = 1.5 box. Every node on the core's six faces and on the shell's six
    outer faces is gold (the conductor wins a face over the dielectric core
    and over vacuum), so the shell is the same number of nodes thick on every
    face. A float comparison decided those nodes face by face: 6, 7 or 8."""
    size, t = np.array([306.0, 333.0, 504.0]), 50.0
    lo = np.asarray(offset)
    names, parts = ["core"], [_box(lo, lo + size)]
    req = _request(boundary="open", dx_m=100e-9, position=list(size / 2 * 1e-9),
                   duration_s=2e-15, coatings={"core": ["Au", t * 1e-9]})
    out = L.layout(names, parts, req, length_scale=1e-9, materials={"core": 1.5})
    gold, nodes = out["metals"]["Au"], out["nodes"]
    thick = []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        # the face's footprint: nodes strictly inside the core's extent
        span = [(nodes[i] > 1e-15) & (nodes[i] < size[i] * 1e-9 - 1e-15)
                for i in (b, c)]
        for core_face, outer_face in ((0.0, -t), (size[a], size[a] + t)):
            planes = {}
            for what, at in (("core", core_face), ("outer", outer_face)):
                i = _at(nodes[a], at * 1e-9)
                plane = np.take(gold, i, axis=a)[np.ix_(*span)]
                planes[what] = int(plane.sum()), plane.size
                assert plane.all(), (offset, a, at, what, planes[what])
            lo_i, hi_i = sorted((_at(nodes[a], core_face * 1e-9),
                                 _at(nodes[a], outer_face * 1e-9)))
            thick.append(hi_i - lo_i + 1)
            mid = [int(np.argmin(np.abs(nodes[i] - size[i] / 2 * 1e-9))) for i in (b, c)]
            line = [slice(None)] * 3
            line[b], line[c] = mid
            column = gold[tuple(line)][lo_i:hi_i + 1]
            assert column.all(), (offset, a, core_face, column)
            # the nodes just beyond the shell's two faces are not gold
            assert not gold[tuple(line)][[lo_i - 1, hi_i + 1]].any(), (offset, a)
    assert len(set(thick)) == 1, (offset, thick)


def test_a_dielectric_coating_gets_its_own_thin_layers():
    """No skin depth here, so only the coating rule can put the shell's outer
    faces on the grid and 4 cells across it (a gold coating's skin-depth
    layers cover the same spans and would hide a missing coating rule). Inside
    a host, so the shell's outer faces are not the grid's own edges."""
    _n, parts, req = _coated_core("metal", coatings={"core": [3.0, 10e-9]},
                                  position=[20e-9, 20e-9, 20e-9])
    names, parts = ["host", "core"], [_box((-200, -200, -200), (320, 280, 300)),
                                      parts[0]]
    planned = L.plan(names, parts, req, length_scale=1e-9,
                     materials={"host": 1.0, "core": 1.5})
    a = 200e-9                    # the frame starts at the host's corner
    for n, b in zip(planned["nodes"], (a + 120e-9, a + 80e-9, a + 100e-9)):
        for lo, hi in ((a - 10e-9, a), (b, b + 10e-9)):
            cells = _at(n, hi) - _at(n, lo)
            assert cells >= 4, (lo, hi, cells)


def test_a_plane_wave_box_encloses_the_coating():
    import fdtd
    # a dielectric coating: no skin-depth cells, so a smaller grid
    names, parts, req = _coated_core(source_kind="plane", direction="+z",
                                     polarisation="x", position=None,
                                     coatings={"core": [3.0, 10e-9]})
    out = L.layout(names, parts, req, length_scale=1e-9, materials={"core": 1.5})
    c = fdtd.BOX_CLEARANCE
    lo = tuple(_at(n, -10e-9) - c for n in out["nodes"])
    hi = tuple(_at(n, s + 10e-9) + c
               for n, s in zip(out["nodes"], (120e-9, 80e-9, 100e-9)))
    assert out["box"] == (lo, hi)
    # the frame's origin stays at the parts' corner
    assert _at(out["nodes"][0], 0.0) > 0


def test_a_coating_on_the_cavitys_own_wall_is_refused():
    names, parts, req = _coated_core("metal", position=[60e-9, 40e-9, 50e-9])
    with pytest.raises(L.LayoutError, match=r"'core' enclose the cavity.*metal boundary's wall"):
        L.plan(names, parts, req, length_scale=1e-9, materials={"core": 1.5})


def test_the_form_refuses_a_source_in_a_gold_coating_naming_it():
    names, parts, req = _coated_core(position=[-5e-9, 40e-9, 50e-9],
                                     coatings={"core": ["pec", 10e-9]})
    planned = L.plan(names, parts, req, length_scale=1e-9, materials={"core": 1.5})
    with pytest.raises(L.LayoutError, match=r"sits inside the coating of part core"):
        L.refuse_conductor_points(planned, parts, length_scale=1e-9)
    with pytest.raises(L.LayoutError, match=r"sits inside the coating of part core"):
        L.layout(names, parts, req, length_scale=1e-9, materials={"core": 1.5})


@pytest.mark.parametrize("boundary", ["metal", "open"])
def test_the_form_estimate_counts_a_gold_coating(boundary):
    """A gold-coated dielectric core inside a host: the estimate counts the
    coating's metal (else it would admit a run the solved scene refuses)."""
    import fdtd
    names = ["host", "core"]
    parts = [_box((0, 0, 0), (200, 200, 200)), _box((70, 70, 70), (130, 130, 130))]
    req = _request(boundary=boundary, dx_m=50e-9, position=[30e-9, 30e-9, 30e-9],
                   duration_s=2e-15, coatings={"core": ["Au", 10e-9]})
    kw = dict(length_scale=1e-9, materials={"host": 1.5, "core": 1.2})
    planned = L.plan(names, parts, req, **kw)
    form = L.estimate(planned, req)
    laid = L.layout(names, parts, req, **kw)
    for a in range(3):
        np.testing.assert_array_equal(planned["nodes"][a], laid["nodes"][a])
    solved = fdtd.estimate_run(laid["nodes"], duration=req["duration_s"],
                               boundary=boundary, metals=laid["metals"],
                               pec=laid["pec"], record_frames=True)
    assert laid["metals"], "the coating must voxelise, or this compares nothing"
    for key in ("cells", "dt", "steps", "setup"):
        assert form[key] == solved[key], key
    assert solved["bytes"] <= form["bytes"] <= 1.05 * solved["bytes"]


# ---- the movie window (change td-movie-window) ------------------------------

def _est(steps=2000, dt=1e-16, shape=(8, 6, 5)):
    """The numbers ``movie_window`` reads: a 200 fs run of 2000 steps of 0.1 fs,
    so a whole number of fs is a whole number of steps (25 fs = step 250)."""
    return {"steps": steps, "dt": dt, "shape": shape}


def test_a_window_records_only_its_own_steps_and_ends_at_its_end():
    """The window is the run's steps 250..1000 (25 fs to 100 fs); the frames are
    spread over it by the rule a whole run already uses -- evenly spaced, the
    last one at the window's end -- and nothing outside it is recorded."""
    est = _est()
    plan = L.movie_window({"movie_start_s": 25e-15, "movie_end_s": 100e-15,
                           "frame_count": 40}, est)
    steps = plan["steps"]
    assert len(steps) == 40
    assert steps[-1] == 1000
    assert all(250 <= s <= 1000 for s in steps)
    assert all(a < b for a, b in zip(steps, steps[1:]))
    gaps = np.diff([249] + steps)
    assert gaps.max() - gaps.min() <= 1, "evenly spread over the window"
    # ... and the same rule as the whole run: the window's own end is the last
    # frame's step, whatever the width.
    for window, count in (((0, 1999), 60), ((250, 1000), 40), ((900, 1999), 7)):
        got = L.movie_window({"movie_start_s": window[0] * 1e-16,
                              "movie_end_s": window[1] * 1e-16,
                              "frame_count": count}, est)["steps"]
        assert got[-1] == window[1] and all(window[0] <= s <= window[1] for s in got)
        assert len(got) == count


def test_a_window_closing_at_the_runs_end_lands_on_its_last_step():
    """An end that is a hair past the run (float noise on the duration) is the
    run's own duration, not an out-of-run refusal, and its last frame is the
    last step the solver takes."""
    est = _est()
    duration = est["steps"] * est["dt"]
    for end in (duration, duration * (1 + 1e-12)):
        plan = L.movie_window({"movie_start_s": duration / 2, "movie_end_s": end,
                               "frame_count": 7}, est)
        assert plan["steps"][-1] == 1999
        assert all(1000 <= s <= 1999 for s in plan["steps"])


def test_a_window_before_the_first_step_is_the_runs_start():
    """Rounding is toward the window: its first frame is the first step at or
    after the start, its last the last at or before the end, and an end past
    the run's own last step is that last step."""
    plan = L.movie_window({"movie_start_s": 1e-17, "movie_end_s": 1e-15}, _est())
    assert plan["steps"][0] == 1 and plan["steps"][-1] == 10


def test_a_window_of_the_whole_run_is_todays_plan_exactly():
    """There is one plan function, so a request that says nothing about the
    window, or spells the whole run out, is the plan of a run with no window
    at all -- the frames an existing job recorded, step for step."""
    import fdtd
    est = _est(steps=1000, dt=1e-16, shape=(130, 64, 65))
    want = fdtd.frame_plan((130, 64, 65), 1000)
    for req in ({}, {"movie_start_s": None, "movie_end_s": None, "frame_count": None},
                {"movie_start_s": 0.0, "movie_end_s": 1000 * 1e-16,
                 "frame_count": 60}):
        assert L.movie_window(req, est) == want, req


@pytest.mark.parametrize("over, needle, also", [
    # a window outside the run names the run's own length
    ({"movie_start_s": -1e-16}, "outside the run", "200 fs"),
    ({"movie_end_s": 201e-15}, "outside the run", "200 fs"),
    ({"movie_end_s": 200e-15 * (1 + 1e-6)}, "outside the run", "200 fs"),
    ({"movie_start_s": 5e-15, "movie_end_s": 5e-15}, "at or before its start", "5 fs"),
    ({"movie_start_s": 10e-15, "movie_end_s": 5e-15}, "at or before its start", "10 fs"),
    # a window narrower than one step names the run's own dt
    ({"movie_start_s": 12e-17, "movie_end_s": 16e-17}, "holds no time steps", "0.1 fs"),
    # more frames than the window holds steps names the window's own step count
    ({"movie_end_s": 10e-15, "frame_count": 120}, "holds only 101 time steps", None),
    ({"movie_start_s": 100e-15, "movie_end_s": 101e-15, "frame_count": 120},
     "holds only 11 time steps", None),
    # a count over the cap is refused as a count, before the window is read
    ({"movie_start_s": 100e-15, "movie_end_s": 101e-15, "frame_count": 121},
     "at most 120", None),
    ({"frame_count": 121}, "at most 120", None),
    ({"frame_count": 0}, "at least 1", None),
    ({"frame_count": -3}, "at least 1", None),
    ({"frame_count": 2.5}, "whole number", None),
    ({"frame_count": "many"}, "whole number", None),
])
def test_a_window_the_run_cannot_honour_is_refused_in_words(over, needle, also):
    with pytest.raises(L.LayoutError) as info:
        L.movie_window(over, _est())
    message = str(info.value)
    assert needle in message, message
    if also is not None:
        assert also in message, f"the refusal must name {also}: {message}"
    # a refusal is a sentence for a person, not a traceback
    assert message[0].isupper() and message.endswith("."), message
