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
