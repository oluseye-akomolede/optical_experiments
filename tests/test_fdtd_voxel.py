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


def _flush_pair():
    """A slab below z=0 and a slab above, sharing the face z=0. The jitter
    shifts points up, so without the rule the face goes to the upper slab."""
    return _box((1.0, 1.0, 0.5), (0, 0, -0.25)), _box((1.0, 1.0, 0.5), (0, 0, 0.25))


_ON_FACE = np.array([[x, y, 0.0] for x in (-0.3, 0.0, 0.2) for y in (-0.2, 0.1)])


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("below, above", [("pec", 1.5), ("Au", 3.0), (2.0, 1.5)])
def test_a_shared_face_goes_to_the_conductor_else_the_higher_index(flip, below, above):
    low, high = _flush_pair()
    parts, mats = [low, high], [below, above]
    if flip:
        parts, mats = parts[::-1], mats[::-1]
    finder = V.PartIndex(parts, priority=[V.priority_of(m) for m in mats])
    want = parts.index(low) + 1
    assert (finder(_ON_FACE) == want).all()
    # Off the face, each slab still owns its own interior.
    assert (finder(_ON_FACE + [0, 0, 0.1]) == parts.index(high) + 1).all()
    assert (finder(_ON_FACE - [0, 0, 0.1]) == want).all()


def test_the_shared_face_rule_reaches_the_conductor_mask():
    low, high = _flush_pair()
    nodes = tuple(np.linspace(-0.6e-6, 0.6e-6, 7) for _ in range(3))
    out = V.voxelise(["film", "plate"], [high, low], {"film": 1.5, "plate": "pec"},
                     nodes, boundary="open", length_scale=1e-6)
    x, y, z = np.meshgrid(*nodes, indexing="ij")
    face = (np.abs(z) < 1e-12) & (np.abs(x) < 0.45e-6) & (np.abs(y) < 0.45e-6)
    assert face.any() and out["pec"][face].all()


@pytest.mark.parametrize("plate", ["pec", "Au"])
def test_a_film_flush_between_two_plates_keeps_its_full_thickness(plate):
    """The owner's stack, with both faces off zero: plates on z in [0.1, 0.5]
    and [0.7, 0.9] um and a film between them. On a grid with nodes on the
    faces, the conductor walls sit on both faces, so the film spans
    thickness / dz = 0.2 / 0.05 = 4 cells (3 interior nodes)."""
    lo = _box((1.0, 1.0, 0.4), (0, 0, 0.3))
    film = _box((1.0, 1.0, 0.2), (0, 0, 0.6))
    hi = _box((1.0, 1.0, 0.2), (0, 0, 0.8))
    xy = np.linspace(-0.6e-6, 0.6e-6, 7)
    z = np.round(np.arange(21) * 0.05, 12) * 1e-6
    out = V.voxelise(["lo", "film", "hi"], [lo, film, hi],
                     {"lo": plate, "film": 1.5, "hi": plate},
                     (xy, xy, z), boundary="open", length_scale=1e-6)
    wall = out["pec"] if plate == "pec" else out["metals"][plate]
    column = wall[3, 3, :]                      # x = y = 0, inside every part
    k_lo, k_hi = 10, 14                         # z = 0.5 and 0.7 um
    assert column[k_lo] and column[k_hi]
    assert not column[k_lo + 1:k_hi].any()
    assert (k_hi - k_lo) == round(0.2 / 0.05)
    assert np.allclose(out["eps"][3, 3, k_lo + 1:k_hi], 1.5 ** 2)


def test_parts_that_share_volume_are_refused_naming_both():
    a = _box((1.0, 1.0, 1.0), (0, 0, 0))
    b = _box((1.0, 1.0, 1.0), (0.5, 0, 0))
    finder = V.PartIndex([a, b], priority=[1.5, 2.0], names=["lens", "prism"])
    with pytest.raises(V.OverlapError, match=r"'lens' and 'prism' overlap"):
        finder(np.array([[0.25, 0.0, 0.0]]))


def test_touching_and_nested_parts_are_not_overlap():
    low, high = _flush_pair()
    V.PartIndex([low, high])(_ON_FACE)
    outer, inner = _box((2.0, 2.0, 2.0)), _box((0.5, 0.5, 0.5))
    assert (V.PartIndex([outer, inner])(np.zeros((1, 3))) == 2).all()


# --- coatings (task 5.4b) ---------------------------------------------------

def _box_distance(lo, hi, pts):
    """Exact distance from points outside an axis-aligned box to the box."""
    return np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1)


def test_surface_distance_matches_the_exact_box_distance_outside():
    box = _box((1.0, 0.6, 0.8), (0.1, 0.0, -0.1))
    lo, hi = box.bounds
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1.5, 1.5, (400, 3))
    pts = pts[_box_distance(lo, hi, pts) > 0]
    got = V.surface_distance(GEO.of_faces(box.vertices, box.faces), pts)
    assert np.allclose(got, _box_distance(lo, hi, pts), atol=1e-12)
    # and with a reach: inside it exact, beyond it inf
    near = V.surface_distance(GEO.of_faces(box.vertices, box.faces), pts, 0.3)
    want = _box_distance(lo, hi, pts)
    assert np.allclose(near[want <= 0.3], want[want <= 0.3], atol=1e-12)
    assert np.all(near[want > 0.3 + 1e-9] > 0.3)


def _coated_scene():
    """A 1 x 0.6 x 0.8 um core of n = 1.5 at the origin, open space, on a
    0.05 um grid that puts nodes on the core's faces and on a 0.1 um shell's
    outer faces. Non-square, so an axis mix-up moves the shell."""
    nodes = [np.round(np.arange(-16, 17) * 0.05, 12) * 1e-6,
             np.round(np.arange(-12, 13) * 0.05, 12) * 1e-6,
             np.round(np.arange(-14, 15) * 0.05, 12) * 1e-6]
    return _box((1.0, 0.6, 0.8)), nodes


def _shell_masks(nodes, half, t):
    """Nodes strictly inside the core; nodes in the shell over one face (not
    over an edge or corner, where the rounded shell and its box differ); and
    nodes beyond the shell."""
    x, y, z = np.meshgrid(*nodes, indexing="ij")
    r = np.stack([np.abs(x), np.abs(y), np.abs(z)], -1)
    h = np.asarray(half) * 1e-6
    tm, eps = t * 1e-6, 1e-12
    core = np.all(r < h - eps, axis=-1)
    outside_core = np.any(r > h + eps, axis=-1)
    # within the shell along exactly one axis: the face slabs, no rounded edges
    over = np.maximum(r - h, 0.0)
    one_axis = (over > eps).sum(axis=-1) == 1
    shell = outside_core & one_axis & (over.max(axis=-1) < tm - eps)
    # a whole cell clear of the shell (diagonal 0.05 um x sqrt 3), so no
    # node's dual cell reaches into it
    beyond = np.linalg.norm(over, axis=-1) > tm + 0.09e-6
    return core, shell, beyond


@pytest.mark.parametrize("coat", ["Au", "pec", 3.0])
def test_a_coated_part_lays_out_as_a_shell_of_the_coating_over_its_core(coat):
    """The 5.4b check: the part's cells read as its own material, a shell one
    thickness deep over its surface reads as the coating, and beyond it is
    vacuum. On cbfff8a ``voxelise`` takes no coating at all."""
    core_part, nodes = _coated_scene()
    out = V.voxelise(["core"], [core_part], {"core": 1.5}, nodes,
                     boundary="open", length_scale=1e-6,
                     coatings={"core": (coat, 0.1e-6)})
    core, shell, beyond = _shell_masks(nodes, (0.5, 0.3, 0.4), 0.1)
    assert core.any() and shell.any() and beyond.any()
    assert np.allclose(out["eps"][core], 2.25)
    assert np.allclose(out["eps"][beyond], 1.0)
    if coat == "pec":
        assert out["pec"][shell].all() and not out["pec"][core | beyond].any()
        assert out["metals"] == {}
    elif coat == "Au":
        gold = out["metals"]["Au"]
        assert gold[shell].all() and not gold[core | beyond].any()
        assert out["pec"] is None
    else:
        assert np.allclose(out["eps"][shell], 9.0)
        assert out["pec"] is None and out["metals"] == {}


def test_a_coating_on_a_curved_part_follows_its_surface():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    nodes = [np.linspace(-0.8e-6, 0.8e-6, 17)] * 3
    out = V.voxelise(["ball"], [sphere], {"ball": 1.5}, nodes, boundary="open",
                     length_scale=1e-6, coatings={"ball": ("Au", 0.2e-6)})
    x, y, z = np.meshgrid(*nodes, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) * 1e6
    gold = out["metals"]["Au"]
    # the icosphere's facets sit up to ~1% inside the radius; keep clear of both faces
    assert gold[(r > 0.53) & (r < 0.67)].all()
    assert not gold[r < 0.47].any() and not gold[r > 0.72].any()
    assert np.allclose(out["eps"][r < 0.4], 2.25)


def test_a_coating_paints_its_enclosing_host_but_never_a_neighbour():
    host = _box((2.0, 2.0, 2.0))
    core = _box((0.4, 0.4, 0.4), (-0.4, 0.0, 0.0))
    # a neighbour flush against the core's +x face
    side = _box((0.4, 0.4, 0.4), (0.0, 0.0, 0.0))
    nodes = [np.round(np.arange(-20, 13) * 0.05, 12) * 1e-6,
             np.round(np.arange(-8, 9) * 0.05, 12) * 1e-6,
             np.round(np.arange(-8, 9) * 0.05, 12) * 1e-6]
    out = V.voxelise(["host", "core", "side"], [host, core, side],
                     {"host": 1.2, "core": 1.5, "side": 2.0}, nodes,
                     boundary="open", length_scale=1e-6,
                     coatings={"core": (3.0, 0.1e-6)})
    i = {round(float(v) * 1e6, 3): k for k, v in enumerate(nodes[0])}
    j = len(nodes[1]) // 2
    eps = out["eps"][:, j, j]
    # -x of the core (x in -0.7..-0.6): the shell displaces the host
    assert np.isclose(eps[i[-0.65]], 9.0)
    assert np.isclose(eps[i[-0.8]], 1.44)
    # +x of the core is the neighbour (from -0.2): within the shell's reach
    # of the core's face it keeps its own material
    assert np.isclose(eps[i[-0.15]], 4.0) and np.isclose(eps[i[0.05]], 4.0)


def test_a_metal_boundary_wall_is_not_painted():
    """Outside every part of a cavity is the conductor wall; a coating does
    not paint into it (the layout refuses an outermost coated part)."""
    wall = _box((2.0, 1.6, 2.0))
    nodes = _scene_nodes()
    plain = V.voxelise(["cavity"], [wall], {"cavity": 1.5}, nodes,
                       boundary="metal", length_scale=1e-6)
    coated = V.voxelise(["cavity"], [wall], {"cavity": 1.5}, nodes,
                        boundary="metal", length_scale=1e-6,
                        coatings={"cavity": (3.0, 0.2e-6)})
    assert np.array_equal(plain["pec"], coated["pec"])
    assert np.array_equal(plain["eps"], coated["eps"])


def test_where_two_shells_meet_the_conductor_wins():
    a = _box((0.4, 0.4, 0.4), (-0.3, 0.0, 0.0))
    b = _box((0.4, 0.4, 0.4), (0.3, 0.0, 0.0))
    gap = np.array([[0.0, 0.0, 0.0]])          # 0.1 um from both
    for coats in ({"a": (3.0, 0.15e-6), "b": ("Au", 0.15e-6)},
                  {"a": ("Au", 0.15e-6), "b": (3.0, 0.15e-6)}):
        finder, table = V.coated_index(["a", "b"], [a, b], {"a": 1.5, "b": 1.5},
                                       coats, boundary="open", length_scale=1e-6)
        gold = 3 + [m for _p, m, _t in table].index("Au")
        assert int(finder(gap)[0]) == gold


@pytest.mark.parametrize("coatings, needle", [
    ({"lid": ("Au", 1e-8)}, r"coating names part\(s\) 'lid'"),
    ({"core": ("Xx", 1e-8)}, r"coating of part 'core': unknown material 'Xx'"),
    ({"core": ("Au", 0.0)}, r"coating of part 'core' needs a positive thickness"),
    ({"core": ("Au", "10nm")}, r"coating of part 'core' needs a positive thickness"),
    ({"core": "Au"}, r"coating of part 'core' must be a material and a thickness"),
])
def test_a_bad_coating_is_refused_by_name(coatings, needle):
    core_part, nodes = _coated_scene()
    with pytest.raises(V.VoxelError, match=needle):
        V.voxelise(["core"], [core_part], {"core": 1.5}, nodes, boundary="open",
                   length_scale=1e-6, coatings=coatings)
