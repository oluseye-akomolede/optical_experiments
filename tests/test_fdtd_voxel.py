"""Voxelising an upload onto the FDTD grid (task 3.1).

The reference for inside/outside is the cavity tool's winding-number test
(``mse_geometry.point_in_mesh``), an independent algorithm. The sample sets
include lattice points that lie exactly on box faces' edges and grid lines,
because an axis-aligned upload puts mesh edges on the grid and a ray through
an edge is where a crossing-count voxeliser goes wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402
import fdtd_voxel as V  # noqa: E402
import mse_geometry as GEO  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def _box(extents, centre=(0.0, 0.0, 0.0)):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(centre)
    return box


def _points():
    rng = np.random.default_rng(7)
    loose = rng.uniform(-1.2, 1.2, (1500, 3))
    g = np.linspace(-1.0, 1.0, 9)          # hits every box edge and face line
    lattice = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return np.vstack([loose, lattice])


def _reference(parts, depths, pts):
    """Innermost containing part by winding number, at the same jittered points."""
    ref = np.zeros(len(pts), dtype=int)
    best = np.full(len(pts), -1)
    for p, part in enumerate(parts):
        inside, _on = GEO.point_in_mesh(part, pts)
        take = inside & (depths[p] > best)
        ref[take] = p + 1
        best[take] = depths[p]
    return ref


@pytest.mark.parametrize("order", ["wall-first", "core-first"])
def test_the_innermost_part_wins_whatever_the_file_order(order):
    wall, core = _box((2.0, 2.0, 2.0)), _box((1.0, 0.5, 1.5), (0.25, 0.0, 0.0))
    parts = [wall, core] if order == "wall-first" else [core, wall]
    finder = V.PartIndex(parts)
    assert sorted(finder.depths) == [0, 1]
    pts = _points()
    index = finder(pts)
    ref = _reference(parts, finder.depths, pts + finder.shift)
    assert np.array_equal(index, ref), int((index != ref).sum())
    core_id = 1 + parts.index(core)
    assert (index == core_id).sum() > 50          # the core is really seen


def test_points_on_box_edges_are_classified_like_the_winding_test():
    """The lattice points sit on the rays through the box's edges."""
    box = _box((1.5, 1.0, 0.5))
    finder = V.PartIndex([box])
    g = np.linspace(-1.0, 1.0, 9)
    pts = np.stack(np.meshgrid(g, g, [0.0, 0.1], indexing="ij"), -1).reshape(-1, 3)
    pts = pts[np.isclose(np.abs(pts[:, 0]), 0.75) | np.isclose(np.abs(pts[:, 1]), 0.5)
              | (np.abs(pts[:, 0]) < 0.75)]
    index = finder(pts)
    ref = _reference([box], finder.depths, pts + finder.shift)
    assert np.array_equal(index, ref)
    assert 0 < index.sum() < len(pts)


def test_a_torus_hole_is_outside():
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
    finder = V.PartIndex([torus])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.1],
                    [1.0, 0.0, 0.5]])
    assert finder(pts).tolist() == [0, 1, 1, 0]


def _scene_nodes():
    # non-square, and in metres: the parts are written in micrometres
    return [np.linspace(-1.2e-6, 1.2e-6, 13), np.linspace(-1.2e-6, 1.2e-6, 9),
            np.linspace(-1.2e-6, 1.2e-6, 11)]


def test_a_cavity_is_walled_by_metal_outside_its_shape():
    wall = _box((2.0, 1.6, 2.0))
    out = V.voxelise(["cavity"], [wall], {"cavity": 1.5}, _scene_nodes(),
                     boundary="metal", length_scale=1e-6)
    x, y, z = np.meshgrid(*_scene_nodes(), indexing="ij")
    deep_out = (np.abs(x) > 1.1e-6) | (np.abs(y) > 0.9e-6) | (np.abs(z) > 1.1e-6)
    deep_in = (np.abs(x) < 0.9e-6) & (np.abs(y) < 0.7e-6) & (np.abs(z) < 0.9e-6)
    assert out["pec"] is not None
    assert out["pec"][deep_out].all() and not out["pec"][deep_in].any()
    assert np.allclose(out["eps"][deep_in], 2.25)
    assert out["metals"] == {}


def test_open_space_is_vacuum_and_each_material_lands_in_its_part():
    shell = _box((1.6, 1.2, 1.6))
    core = _box((0.6, 0.4, 0.8), (0.3, 0.0, 0.0))
    out = V.voxelise(["shell", "core"], [shell, core],
                     {"shell": 2.0, "core": "Au"}, _scene_nodes(),
                     boundary="open", length_scale=1e-6)
    x, y, z = np.meshgrid(*_scene_nodes(), indexing="ij")
    outside = (np.abs(x) > 0.9e-6) | (np.abs(y) > 0.7e-6) | (np.abs(z) > 0.9e-6)
    assert out["pec"] is None
    assert np.allclose(out["eps"][outside], 1.0)
    assert set(out["metals"]) == {"Au"}
    gold = out["metals"]["Au"]
    in_core = (np.abs(x - 0.3e-6) < 0.25e-6) & (np.abs(y) < 0.15e-6) & (np.abs(z) < 0.35e-6)
    assert gold[in_core].all() and in_core.any()
    assert not gold[outside].any()
    in_shell = (np.abs(x + 0.5e-6) < 0.2e-6) & (np.abs(y) < 0.5e-6) & (np.abs(z) < 0.7e-6)
    assert np.allclose(out["eps"][in_shell], 4.0) and in_shell.any()


def test_a_face_through_a_node_gives_the_cut_cell_average():
    """The slab face x = 0 runs through the middle node column: half n=2."""
    nodes = [np.linspace(-1.0e-6, 1.0e-6, 11), np.linspace(-0.4e-6, 0.4e-6, 5),
             np.linspace(-0.6e-6, 0.6e-6, 7)]
    slab = _box((2.0, 2.0, 2.0), (1.0, 0.0, 0.0))     # fills x > 0
    out = V.voxelise(["slab"], [slab], {"slab": 2.0}, nodes, boundary="open",
                     length_scale=1e-6, samples=4)
    column = out["eps"][5]
    assert np.allclose(column, 2.5), column


def test_a_pec_part_joins_the_conductor_mask():
    wall = _box((2.0, 2.0, 2.0))
    post = _box((0.4, 0.4, 2.0))
    out = V.voxelise(["wall", "post"], [wall, post], {"wall": 1.0, "post": "pec"},
                     _scene_nodes(), boundary="metal", length_scale=1e-6)
    x, y, z = np.meshgrid(*_scene_nodes(), indexing="ij")
    on_axis = (np.abs(x) < 0.1e-6) & (np.abs(y) < 0.1e-6) & (np.abs(z) < 0.9e-6)
    between = (np.abs(x) > 0.4e-6) & (np.abs(x) < 0.9e-6) & (np.abs(y) < 0.9e-6) \
        & (np.abs(z) < 0.9e-6)
    assert out["pec"][on_axis].all() and on_axis.any()
    assert not out["pec"][between].any() and between.any()


@pytest.mark.parametrize("materials, needle", [
    ({}, "no material given for part(s) 'a'"),
    ({"a": 1.0, "b": 1.0}, "names part(s) 'b' that are not in this file"),
    ({"a": "unobtainium"}, "unknown material 'unobtainium'"),
    ({"a": 0.5}, "must be a finite number of at least 1"),
    ({"a": float("nan")}, "must be a finite number of at least 1"),
    ({"a": None}, "material None is not a refractive index"),
    ({"a": [1.5]}, "material [1.5] is not a refractive index"),
    ({"a": True}, "material True is not a refractive index"),
])
def test_a_bad_materials_table_is_refused_by_name(materials, needle):
    with pytest.raises(V.VoxelError, match=None) as err:
        V.voxelise(["a"], [_box((1.0, 1.0, 1.0))], materials, _scene_nodes(),
                   length_scale=1e-6)
    assert needle in str(err.value)
    assert isinstance(err.value, fdtd.FDTDError)


def test_an_unknown_boundary_is_refused():
    with pytest.raises(V.VoxelError, match="boundary 'periodic'"):
        V.voxelise(["a"], [_box((1.0, 1.0, 1.0))], {"a": 1.0}, _scene_nodes(),
                   boundary="periodic", length_scale=1e-6)


def test_every_conductor_part_and_the_cavity_wall_join_one_mask():
    """Two PEC posts in a metal cavity: both posts AND the outside are walls."""
    # posts sit on node lines (x nodes step 0.2 um) so each region samples them
    wall = _box((2.0, 2.0, 2.0))
    left = _box((0.4, 0.4, 2.0), (-0.6, 0.0, 0.0))
    right = _box((0.4, 0.4, 2.0), (0.6, 0.0, 0.0))
    out = V.voxelise(["wall", "left", "right"], [wall, left, right],
                     {"wall": 1.0, "left": "pec", "right": "pec"},
                     _scene_nodes(), boundary="metal", length_scale=1e-6)
    x, y, z = np.meshgrid(*_scene_nodes(), indexing="ij")
    inner = (np.abs(y) < 0.1e-6) & (np.abs(z) < 0.9e-6)
    on_left = inner & (np.abs(x + 0.6e-6) < 0.1e-6)
    on_right = inner & (np.abs(x - 0.6e-6) < 0.1e-6)
    gap = inner & (np.abs(x) < 0.2e-6)
    outside = (np.abs(x) > 1.1e-6) | (np.abs(y) > 1.1e-6) | (np.abs(z) > 1.1e-6)
    for name, region in [("left", on_left), ("right", on_right),
                         ("outside", outside)]:
        assert region.any() and out["pec"][region].all(), name
    assert gap.any() and not out["pec"][gap].any()
