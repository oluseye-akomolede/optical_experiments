"""Splitting a multi-part .obj into named parts, and refusing the shapes that
lie about their own structure.

THE MEASUREMENTS THESE PIN. ``trimesh.load`` on a two-group .obj returns a
merged ``Trimesh`` -- not a ``Scene`` -- with the names gone and one volume for
what were two parts. The names exist only in the raw text. And a file whose
faces come before any ``o``/``g`` line has ONE name in the text and TWO parts
from ``split()``, so a bare name scan zips two lists of different length and
assigns every material to the wrong region while all the counts agree.

The torus case is here because the lead asked for it: a torus's centroid lies in
its HOLE, outside the part, so it is a seed that a centroid-based implementation
would accept and mesh wrongly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_geometry as GEO  # noqa: E402
import mse_parts as PARTS  # noqa: E402
from mse_common import MeshingError  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def _two_group_obj():
    """A wall box with a smaller core box inside it, exported as two groups."""
    wall = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    core = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
    return trimesh.Scene({"wall": wall, "core": core}).export(file_type="obj")


# ---------------------------------------------------------------------------
# The happy path, and the correspondence it must actually check.
# ---------------------------------------------------------------------------
def test_names_and_parts_line_up_in_file_order():
    names, parts = PARTS.split_obj_parts(_two_group_obj())
    assert names == ["wall", "core"]
    assert len(parts) == 2
    # the correspondence is the whole point: a name on the wrong part applies a
    # material to the wrong region with nothing else failing
    assert float(parts[0].volume) == pytest.approx(8.0, rel=1e-6)
    assert float(parts[1].volume) == pytest.approx(0.216, rel=1e-6)


def test_the_parts_are_watertight_and_cleaned():
    _names, parts = PARTS.split_obj_parts(_two_group_obj())
    for part in parts:
        assert part.is_watertight
        assert len(part.faces) > 0


# ---------------------------------------------------------------------------
# The trap: faces before any name, and the count mismatch it produces.
# ---------------------------------------------------------------------------
def test_faces_before_any_name_become_a_named_first_part():
    """One name in the text, two parts from split() -- measured.

    The implicit leading part must get a STABLE name and must be FIRST. A bare
    name scan would see one name and assign it to the first part, putting the
    owner's material on the wrong region.
    """
    raw = (
        "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\n"
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
        "o second\n"
        "v 5 0 0\nv 6 0 0\nv 5 1 0\nv 5 0 1\n"
        "f 5 6 7\nf 5 6 8\nf 5 7 8\nf 6 7 8\n"
    )
    names, parts = PARTS.split_obj_parts(raw)
    assert names == [PARTS.UNNAMED_PART, "second"]
    assert len(parts) == 2
    # Measured volumes, read back from the parts rather than derived: 4/3 for
    # the first tetrahedron and 31/6 for the second. Stated as measured so the
    # assertion is about the CORRESPONDENCE -- which name went on which part --
    # and not about my arithmetic on the vertex list.
    assert float(parts[0].volume) == pytest.approx(4.0 / 3.0, rel=1e-6)
    assert float(parts[1].volume) == pytest.approx(31.0 / 6.0, rel=1e-6)


def test_part_order_follows_the_FILE_when_vertex_indices_disagree():
    """The names zip to the parts in FILE order, not vertex-index order.

    The fixture puts the part named FIRST in the text on the HIGHER vertex
    indices, so the two orders genuinely disagree and a correspondence that
    followed vertex ordering would swap them -- with every face count still
    agreeing, so nothing downstream would notice and the material would land on
    the wrong region.

    MEASURED, so the claim is not taken on faith:
      * trimesh returns this file's faces in FILE order -- ``[[4 5 6] ... [0 1
        2] ...]``, the first-named part's high-index faces first. That is true
        of the ``.obj`` loader with OR without ``maintain_order=True`` (both
        measured identical), so ``maintain_order`` is not the mechanism here
        and this test does not claim it is.
      * the split returns its components in the order of that face array, so
        with the array in file order the parts come back in file order.

    Verified capable of failing: reversing the correspondence between ``names``
    and ``parts`` (one line in ``split_obj_parts``) reddens this test along with
    seven others, so it is a guard and not a self-comparison. It is also
    REDUNDANT with ``test_names_and_parts_line_up_in_file_order`` for the simple
    case; it earns its place only by using a fixture where vertex-index order
    and file order disagree, which that test's fixture does not.

    The volumes are read back from the parts rather than derived from the
    vertex list, so the assertion is about which NAME went on which PART and
    not about my arithmetic on the coordinates.
    """
    raw = (
        "v 0 0 0\nv 4 0 0\nv 0 4 0\nv 0 0 4\n"          # vertices 1-4
        "v 9 9 9\nv 10 9 9\nv 9 10 9\nv 9 9 10\n"       # vertices 5-8
        "o first_high\n"                                 # FIRST in the text
        "f 5 6 7\nf 5 6 8\nf 5 7 8\nf 6 7 8\n"          # uses the HIGH vertices
        "o second_low\n"                                 # SECOND in the text
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"          # uses the LOW vertices
    )
    names, parts = PARTS.split_obj_parts(raw)
    assert names == ["first_high", "second_low"]

    # The two tetrahedra differ in volume, so a swap is visible. Values are
    # measured from the parts, not predicted from the coordinates.
    first, second = float(parts[0].volume), float(parts[1].volume)
    assert first != pytest.approx(second, rel=0.05), (
        "the two parts have the same volume, so this fixture cannot see a swap"
    )
    # 9.167 for the tetra at (9,9,9)-(10,9,9)-(9,10,9)-(9,9,10); 10.667 for the
    # one at the origin with a 4-unit edge.
    assert first == pytest.approx(9.1667, rel=0.01), (
        "the first NAME did not land on the part that comes first in the FILE"
    )
    assert second == pytest.approx(10.6667, rel=0.01)


def test_a_duplicate_group_name_is_refused():
    raw = (
        "o part\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        "o part\nf 1 2 3\n"
    )
    with pytest.raises(MeshingError) as exc:
        PARTS.split_obj_parts(raw)
    assert "part" in str(exc.value)
    assert "more than once" in str(exc.value)


def test_an_empty_named_group_is_refused_not_guessed():
    """A declared group with no faces: measured 2 names, 1 part.

    This is the count mismatch in its most likely real form -- an exporter that
    writes an ``o`` line for a part it then emits nothing for, or a part the
    owner deleted in the CAD tool while leaving the group behind. A guessed
    correspondence would take the single part and call it "ghost", and the
    owner's material for "whole" would land on it.

    Measured on this fixture: ``o whole`` before one tetrahedron, then a bare
    ``o ghost`` with no faces. Text has 2 groups; ``split()`` returns 1 part.
    """
    raw = (
        "o whole\n"
        "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\n"
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
        "o ghost\n"
    )
    with pytest.raises(MeshingError) as exc:
        PARTS.split_obj_parts(raw)
    message = str(exc.value)
    # the message must say BOTH numbers, because the owner can usually act on
    # "2 groups, 1 part" without opening the file in a text editor
    assert "2" in message and "1" in message


# ---------------------------------------------------------------------------
# Input forms: the web route passes text, the CLI passes a path.
# ---------------------------------------------------------------------------
def test_the_parser_reads_text_bytes_and_a_path(tmp_path):
    text = _two_group_obj()
    path = tmp_path / "two.obj"
    path.write_text(text)

    by_text = PARTS.split_obj_parts(text)
    by_path = PARTS.split_obj_parts(str(path))
    by_bytes = PARTS.split_obj_parts(text.encode())
    with open(path, "rb") as fh:
        by_file = PARTS.split_obj_parts(fh)

    for names, parts in (by_text, by_path, by_bytes, by_file):
        assert names == ["wall", "core"]
        assert len(parts) == 2


def test_a_single_part_file_still_works():
    """The ordinary upload: no groups at all, one part, one name."""
    text = trimesh.creation.icosphere(subdivisions=2).export(file_type="obj")
    names, parts = PARTS.split_obj_parts(text)
    assert len(parts) == 1
    assert names == [PARTS.UNNAMED_PART]


# ---------------------------------------------------------------------------
# The torus: the seed case the lead asked for.
# ---------------------------------------------------------------------------
def _torus():
    return trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)


def test_the_torus_centroid_is_outside_the_torus():
    """The measured property that makes the torus a seed case.

    A torus's centroid sits in its HOLE: inside the bounding box, inside the
    convex hull, and outside the solid. A seed finder that trusted the centroid
    would hand TetGen a region point outside the region, and the mesh would come
    back without that region -- or with the wrong one -- and nothing would fail.
    """
    torus = _torus()
    centroid = np.asarray(torus.centroid, dtype=float)
    inside, _on = GEO.point_in_mesh(torus, centroid.reshape(1, 3))
    assert not inside[0], (
        "the torus centroid must be OUTSIDE the torus, or this test case does "
        "not exercise the seed finder at all"
    )


def test_a_point_on_the_torus_tube_is_inside():
    """The positive control for the test above.

    Without this, a containment test that said "outside" for everything would
    pass the centroid assertion. A point on the tube's centre circle (radius =
    major radius, z = 0) is unambiguously inside the solid.
    """
    torus = _torus()
    on_tube = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    inside, _on = GEO.point_in_mesh(torus, on_tube)
    assert inside.all(), "points on the tube's centre circle are inside it"


def test_a_nested_pair_around_a_torus_centroid_is_not_contained():
    """The containment direction, with the torus as the inner part.

    A small box placed around the torus's centroid is inside the torus's BOUNDS
    but not inside the torus itself, so ``part_contains`` must say no -- which is
    the check that keeps a nested-part seed from being accepted for a part it is
    not in.
    """
    torus = _torus()
    hole = trimesh.creation.box(extents=(0.2, 0.2, 0.2))     # centred on origin
    assert not GEO.part_contains(torus, hole), (
        "a box in the hole must not be reported as inside the torus"
    )


# ---------------------------------------------------------------------------
# The seed finder: the centroid is not an interior point.
# ---------------------------------------------------------------------------
def test_the_seed_finder_avoids_the_torus_hole():
    """The torus case from the lead, end to end.

    The centroid is in the hole (asserted above), so the finder must return
    something else, and that something must be verifiably inside the solid.
    """
    torus = _torus()
    seed = PARTS.find_region_seed(torus)
    inside, on = GEO.point_in_mesh(torus, np.asarray(seed, dtype=float).reshape(1, 3))
    assert inside[0] and not on[0], "the found seed must be strictly inside"


def test_the_seed_finder_avoids_a_child_that_holds_the_centroid():
    """The nested case: the wall's centroid is inside the core.

    Measured: a 2x2x2 wall containing a 1x1x1 core has its centroid at the
    origin, which is INSIDE the core. A centroid seed for the wall would put the
    wall's material in the core's region. The finder must return a point that is
    inside the wall and outside the core, and this test asserts that the
    centroid fails, so the case is proven rather than assumed.
    """
    wall = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    core = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    centroid = np.asarray(wall.centroid, dtype=float).reshape(1, 3)
    in_core, _ = GEO.point_in_mesh(core, centroid)
    assert in_core[0], (
        "this fixture relies on the wall's centroid being inside the core; if "
        "that is no longer true the test no longer exercises the case"
    )

    seed = np.asarray(PARTS.find_region_seed(wall, children=[core]), dtype=float)
    in_wall, on_wall = GEO.point_in_mesh(wall, seed.reshape(1, 3))
    in_core, on_core = GEO.point_in_mesh(core, seed.reshape(1, 3))
    assert in_wall[0] and not on_wall[0], "the seed must be inside the wall"
    assert not in_core[0] and not on_core[0], "and outside the core"


def test_the_seed_finder_uses_the_centroid_when_it_is_fine():
    """The simple case: an un-nested part can keep its centroid.

    Not an optimisation -- a check that the finder does not gratuitously move
    the seed, which would make the mesh depend on the sampling for no reason.
    """
    part = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    seed = np.asarray(PARTS.find_region_seed(part), dtype=float)
    assert np.allclose(seed, np.asarray(part.centroid, dtype=float), atol=1e-9)


def test_the_seed_is_deterministic():
    """The same part must give the same seed every run.

    A random seed would move the element count between identical uploads, and
    the element count decides the route to the Jetson and the cap refusal.
    """
    torus = _torus()
    a = np.asarray(PARTS.find_region_seed(torus), dtype=float)
    b = np.asarray(PARTS.find_region_seed(torus), dtype=float)
    assert np.array_equal(a, b)


def test_children_of_finds_nesting_not_collision():
    """``children_of`` must report the core under the wall, and not the reverse."""
    wall = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    core = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    parts = PARTS.label_parts(["wall", "core"], [wall, core])
    assert PARTS.children_of(0, parts) == [1]
    assert PARTS.children_of(1, parts) == []


def test_a_colliding_pair_is_not_reported_as_nesting():
    """The negative control for ``children_of``.

    Two boxes that interpenetrate are not parent and child, so neither may be
    reported as a child of the other -- otherwise the seed finder would exclude
    a region it should not, and the mesher would be given a PLC it cannot mesh.
    """
    a = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    b.apply_translation([1.0, 0.0, 0.0])
    parts = PARTS.label_parts(["a", "b"], [a, b])
    assert PARTS.children_of(0, parts) == []
    assert PARTS.children_of(1, parts) == []
