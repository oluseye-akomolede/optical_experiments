"""Split a multi-part .obj into named parts, and refuse the shapes that lie.

WHY THIS EXISTS. A multi-material cavity is uploaded as ONE .obj with several
named groups -- the PEC wall, a silicon core, maybe a rod or two. The names are
what the owner's material table addresses, so the pair (part, name) is the thing
this module has to get right, and getting it wrong applies a refractive index to
the wrong region with every count still agreeing.

THE NAME TRAP, MEASURED. ``trimesh.load`` returns a merged ``Trimesh`` for a
multi-group .obj -- not a ``Scene`` -- with ``body_count`` set and **no name
information at all**. Measured on a two-group box pair: the loaded object has
one volume (8.216 for parts of 8.0 and 0.216), so the groups are welded, and
``scene.geometry`` does not exist to read them from. ``split()`` does recover the
parts geometrically (2 parts, extents 2.0 and 0.6, in file order), but with no
names. So the geometry comes from trimesh and the names MUST come from the raw
text.

A BARE NAME SCAN IS NOT ENOUGH. Measured on a file with four unnamed faces and
then ``o second``: the text has **1** name while ``split()`` returns **2** parts.
A scan that collected names and zipped them against the parts would assign
"second" to the FIRST part -- the wrong material on both regions, with the counts
agreeing. The scan below therefore walks the file line by line and records, for
every ``f`` line, which group it belongs to.

Trimesh's own reader is used for the geometry (it is already a dependency and it
handles the format properly); only the GROUP STRUCTURE is re-derived here,
because that is the one thing it discards.
"""

from __future__ import annotations

import io

import numpy as np

import mse_geometry as GEO
from mse_common import MeshingError, unit_scale

#: The name given to a leading group of faces that no ``o``/``g`` line opened.
#: Documented on the upload page and in the refusal messages, because the owner
#: has to be able to address it: it is a real part with a real material, and
#: silently dropping it or numbering it would make the material table address
#: something the owner never named.
UNNAMED_PART = "(unnamed)"


class ObjGroup:
    """One ``o``/``g`` group of faces, in file order."""

    __slots__ = ("name", "face_count")

    def __init__(self, name: str):
        self.name = name
        self.face_count = 0


def parse_obj_groups(obj_text: str) -> list[ObjGroup]:
    """Group structure of an .obj: one :class:`ObjGroup` per group, file order.

    Records the number of ``f`` lines each group owns, so the caller can compare
    the *faces* per group against the parts trimesh returns -- and not merely
    the count of names.

    Handles the ``o`` and ``g`` forms, and the leading implicit group. It does
    NOT try to interpret any other line: only ``o``, ``g`` and ``f`` matter for
    the assignment, and every other line is skipped, so a vertex line between
    two face lines cannot end a group.

    Raises :class:`MeshingError` on a duplicate group name, because two parts
    the owner cannot tell apart are two parts the material table cannot address
    and the failure downstream would be a material on the wrong region.
    """
    groups: list[ObjGroup] = []
    current: ObjGroup | None = None
    seen: dict[str, int] = {}

    for lineno, raw_line in enumerate(obj_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line[0] == "#":
            continue
        head = line.split(None, 1)
        keyword = head[0]

        if keyword in ("o", "g"):
            # ``g`` with no name is legal and means "the default group"; treat
            # it as unnamed rather than inventing a name for it.
            name = head[1].strip() if len(head) > 1 else ""
            if name == "":
                name = UNNAMED_PART
            if name in seen:
                raise MeshingError(
                    f"this .obj names the group {name!r} more than once (the "
                    f"second at line {lineno}). Give each part a distinct name, "
                    "then assign a material to each."
                )
            seen[name] = lineno
            current = ObjGroup(name)
            groups.append(current)
            continue

        if keyword == "f":
            if current is None:
                # Faces before any ``o``/``g``: the implicit leading part. It is
                # always FIRST, matching split()'s file order.
                current = ObjGroup(UNNAMED_PART)
                seen[UNNAMED_PART] = lineno
                groups.append(current)
            current.face_count += 1

    return groups


def _obj_text(obj_bytes_or_path) -> str:
    """The raw .obj text, from text, bytes, or a path.

    A ``str`` is ambiguous -- it could be the text or a path -- so it is read as
    TEXT when it contains a newline or any character an .obj line would need,
    and as a path otherwise. Guessing the other way is what made a text argument
    raise ``FileNotFoundError`` naming the whole file as the filename, which is
    at least loud; the quieter failure would be a path read as text. Both are
    avoided by testing for the shape of the content rather than the type.
    """
    if isinstance(obj_bytes_or_path, (bytes, bytearray)):
        return bytes(obj_bytes_or_path).decode("utf-8", errors="replace")
    if isinstance(obj_bytes_or_path, str):
        if "\n" in obj_bytes_or_path or obj_bytes_or_path.lstrip()[:1] in ("v", "o", "g", "f", "#"):
            return obj_bytes_or_path
        with open(obj_bytes_or_path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    # Anything else with a ``read`` (an open file, a BytesIO) is read directly.
    read = getattr(obj_bytes_or_path, "read", None)
    if read is not None:
        data = read()
        if isinstance(data, str):
            return data
        return bytes(data).decode("utf-8", errors="replace")
    raise MeshingError(
        f"cannot read an .obj from {type(obj_bytes_or_path).__name__}; pass "
        "the file's bytes, its text, a path, or an open binary file."
    )


def _clean_part(mesh):
    """Weld, drop degenerate/duplicate faces, drop unreferenced vertices.

    The same cleanup ``load_surface`` applies to a single-part upload, applied
    per part. A part that is not cleaned the same way would be watertight-tested
    on different triangles than the ones that get meshed.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _split_by_components(mesh):
    """``mesh``'s connected face components, as cleaned Trimesh objects.

    Replaces ``trimesh.Trimesh.split()``, which cannot run in the web image:
    its ``connected_components`` step requires a graph engine (networkx or
    scipy) and neither is installed there (measured). ``face_adjacency`` is
    pure numpy, so this does the labelling itself with a union-find and works
    identically in the venv, the solver image and the web image.

    Faces of one component share vertices transitively, which is exactly the
    relation ``split()`` uses, so the grouping is the same one -- and the
    caller checks the face counts against the parsed groups, so a disagreement
    is refused rather than silently mis-assigned.
    """
    import trimesh

    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []

    parent = np.arange(n_faces)

    def find(i: int) -> int:
        # Path-halving union-find. Iterative: a recursive one would hit the
        # interpreter's recursion limit on a mesh with a long thin component,
        # which is exactly the degenerate input an upload can contain.
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n_faces)], dtype=np.int64)
    # Ordered by first appearance, which is file order -- the same order the
    # names come back in, and the correspondence this function exists to keep.
    labels = {root: index for index, root in enumerate(
        dict.fromkeys(roots.tolist())
    )}

    parts = []
    for root in labels:
        face_index = np.flatnonzero(roots == root)
        # ``process=False`` keeps vertex order, so a re-exported part is
        # byte-stable and the one-part path stays byte-identical.
        parts.append(_clean_part(trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[face_index],
            process=False,
        )))
    return parts


def split_obj_parts(obj_bytes_or_path):
    """``(names, parts)`` for a multi-part .obj, in file order.

    ``parts`` are cleaned ``trimesh.Trimesh`` objects, one per named group.
    ``names`` is the matching list of group names, with :data:`UNNAMED_PART`
    for the leading implicit group when there is one.

    The correspondence between the two lists is the whole point of this
    function, so it is CHECKED rather than assumed: the number of groups parsed
    from the text must equal the number of parts trimesh returns, and the faces
    per group must match the faces per part where the counts are knowable. A
    mismatch is refused by name, never guessed at -- an assumed correspondence
    is how a material lands on the wrong region while everything else agrees.
    """
    import trimesh

    text = _obj_text(obj_bytes_or_path)
    groups = parse_obj_groups(text)

    try:
        loaded = trimesh.load(
            io.BytesIO(text.encode("utf-8")), file_type="obj",
            process=False, maintain_order=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface the parser's own words
        raise MeshingError(f"could not parse this .obj file: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        # trimesh 5.1.0 does not take this branch for a multi-group .obj (it
        # returns a merged Trimesh, measured), but another version or a truly
        # multi-object file could. Handle it by concatenating rather than
        # taking the biggest -- taking the biggest is the bug this whole
        # feature is replacing.
        parts = [
            _clean_part(g) for g in loaded.geometry.values() if len(g.faces)
        ]
    else:
        # Split by connected components WITHOUT trimesh.split(), which cannot
        # run in the web image at all. Measured, both images and the venv:
        #
        #   trimesh's split() -> submesh() -> connected_components() needs a
        #   "graph engine" (networkx, or scipy as a fallback). networkx is
        #   ABSENT everywhere. scipy is PRESENT in the solver image and the
        #   venv, but ABSENT from the web image -- so in the web image, which is
        #   where the upload's "which part is open?" refusal runs, split()
        #   raises "no graph engines available!" outright. With repair=True (the
        #   default) it dies earlier still, inside fill_holes.
        #
        # face_adjacency, by contrast, is pure numpy and works in every one of
        # the three environments (measured). So the components are labelled here
        # by a union-find over that adjacency -- about twenty lines, numpy only,
        # no new dependency for either image, and identical results in all
        # three. This is not an optimisation; it is the difference between the
        # refusal working in the deployed web image and not working at all.
        parts = _split_by_components(loaded)
        parts = [p for p in parts if len(p.faces) > 0]

    if not parts:
        raise MeshingError("this .obj contains no faces")

    names = [g.name for g in groups]

    if not groups and len(parts) == 1:
        # A single-part file with no o/g lines at all: the ordinary upload.
        # Name it, so the one-part path has a name to report like any other.
        names = [UNNAMED_PART]
    elif len(names) != len(parts):
        # The count mismatch the trap produces. Refuse, and say both numbers,
        # because the owner can usually see which grouping is wrong from them.
        raise MeshingError(
            f"this .obj has {len(groups)} named group(s) but {len(parts)} "
            "closed part(s). Every face must belong to exactly one named "
            "group: add an 'o <name>' line before the faces that have none, "
            "and give each part a distinct name."
        )

    # The weak check above catches the common case; this one catches the rest.
    # Where the text's face count for a group is a SUBSET relation (trimesh may
    # duplicate a shared vertex, not a face), the totals must still line up, so
    # compare the sum of the group face counts against the total faces trimesh
    # saw. A mismatch means the two readings of the file disagree.
    if groups:
        text_faces = sum(g.face_count for g in groups)
        mesh_faces = sum(len(p.faces) for p in parts)
        if text_faces != mesh_faces:
            raise MeshingError(
                f"this .obj does not read consistently: the file lists "
                f"{text_faces} face(s) across its groups but {mesh_faces} "
                "triangle(s) came back from it. Re-export the .obj from your "
                "CAD tool, one named group per part."
            )

    return names, parts


# ---------------------------------------------------------------------------
# Region seeds: a point inside the part and outside every part nested in it.
# ---------------------------------------------------------------------------
#: How many samples to try before giving up on a part. The candidates below are
#: tried in order and the first that satisfies the containment predicate wins;
#: the count only bounds the stratified fallback, which is tried last.
SEED_SAMPLES = 20000


def find_region_seed(part, children=(), *, samples: int = SEED_SAMPLES):
    """A point inside ``part`` and outside every part in ``children``.

    THE CENTROID IS NOT AN INTERIOR POINT, and this function exists because of
    it. Measured: a torus's centroid lies in its HOLE -- inside its bounding box,
    inside its convex hull, and outside the solid. A dielectric torus in a
    cavity, the case the lead named, would get a region point outside its own
    region: TetGen would take the attribute request, mesh something, and the
    material would be somewhere the owner did not ask for with nothing failing.

    For a NESTED part the failure is the mirror image: the outer part's centroid
    can lie inside the inner part, so the wall's seed lands in the core's region.

    Candidates, in the order they are tried:

    1. the centroid -- correct most of the time, so it is tried first;
    2. the centroid of the largest tetrahedron trimesh can find, which is pulled
       toward the mass rather than the mean of the vertices;
    3. a deterministic stratified sample of the bounding box, bounded by
       ``samples``, taking the first point that is inside ``part`` and outside
       every child.

    Deterministic is a requirement, not tidiness: the same upload must produce
    the same mesh every time, and a random seed would make the element count --
    and so the route decision and the cap -- vary between identical uploads.

    Raises :class:`MeshingError` naming the part when no candidate works. That is
    a real possibility for a thin shell with no room for a sample to land, and
    the owner can act on "this part is too thin to place a material in it" in a
    way they cannot act on a mesher crash.
    """
    verts = np.asarray(part.vertices, dtype=np.float64)
    if verts.size == 0:
        raise MeshingError(f"part has no geometry to place a seed in")

    def ok(point: np.ndarray) -> bool:
        pts = np.asarray(point, dtype=np.float64).reshape(1, 3)
        inside, on = GEO.point_in_mesh(part, pts)
        if not bool(inside[0]) or bool(on[0]):
            return False
        for child in children:
            c_in, c_on = GEO.point_in_mesh(child, pts)
            # the seed must be OUTSIDE the child, so being on the child's
            # surface is not acceptable either
            if bool(c_in[0]) or bool(c_on[0]):
                return False
        return True

    centroid = np.asarray(part.centroid, dtype=np.float64)
    if ok(centroid):
        return centroid

    # The largest tetrahedron trimesh can inscribe gives a point pulled toward
    # the interior mass. It fails too for a torus (its largest tetrahedron is
    # off the hole, but a symmetric torus can put it anywhere), so it is one
    # candidate among several rather than the answer.
    try:
        tet = part.convex_hull
        # prefer a vertex of the hull that is genuinely interior to the surface
        for candidate in np.asarray(tet.vertices, dtype=np.float64)[:64]:
            if ok(candidate):
                return candidate
    except Exception:  # noqa: BLE001 - hull is a convenience, not a requirement
        pass

    lo, hi = np.asarray(part.bounds, dtype=np.float64)
    # Stratified by a low-discrepancy sequence (a shuffled grid), so the samples
    # cover the box rather than clustering, and are identical every run.
    n = max(int(samples), 1)
    side = max(int(round(n ** (1.0 / 3.0))), 1)
    axis = (np.arange(side) + 0.5) / side
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)[:n]
    # a fixed permutation of the grid order, seeded by nothing
    order = np.argsort((grid @ np.array([1.0, 7.0, 53.0])) % 1.0)
    candidates = lo + grid[order] * (hi - lo)

    # Test the interior candidates in chunks; ``point_in_mesh`` is already
    # chunked, but the loop keeps the first-hit logic simple and bounds how many
    # points are held at once for a large part.
    step = 512
    for start in range(0, candidates.shape[0], step):
        block = candidates[start:start + step]
        inside, on = GEO.point_in_mesh(part, block)
        good = inside & ~on
        for child in children:
            c_in, c_on = GEO.point_in_mesh(child, block)
            good &= ~(c_in | c_on)
        if good.any():
            return block[int(np.flatnonzero(good)[0])]

    raise MeshingError(
        f"could not find a point inside part {part_name_of(part)!r} that is "
        "outside the parts nested in it. The part may be too thin or too "
        "folded to place a material in it; simplify it or give it more "
        "thickness."
    )


class PartSet:
    """A validated multi-part upload: named parts, nesting, seeds and regions.

    Built by :func:`load_parts`, and the ONLY supported way to get one. Every
    field here is the result of a check that either passed or raised, so a
    ``PartSet`` in hand means: the names and the parts correspond one for one,
    every part is closed, no two parts intersect or touch, the PEC wall is
    unique, and every region has a seed that is provably inside it.

    Nothing here is derived lazily. The whole file is validated before any of it
    is meshed, so a refusal happens while the message can still name the part --
    once TetGen has run, the same problem surfaces as a triangulation failure
    with no part name attached, which is the could-not-ask shape.
    """

    def __init__(self, names, parts, region_of_part, seeds, wall_index):
        self.names = list(names)
        self.parts = list(parts)
        #: index into ``parts`` of the part that encloses every other, or None
        #: for a single-part upload -- which has no wall part, because the one
        #: part IS the cavity.
        self.wall_index = wall_index
        #: region id per part. Region 0 is the FILL of the wall (the wall's
        #: enclosed volume minus everything nested in it); every other part gets
        #: its own id. This is what ``tetrahedralize`` asks TetGen for and what
        #: comes back as ``region_of_tet``.
        self.region_of_part = list(region_of_part)
        #: a point strictly inside each part's own region (inside the part,
        #: outside every part nested in it) -- never a centroid. See
        #: :func:`find_region_seed` for why.
        self.seeds = list(seeds)

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return iter(self.parts)

    def __getitem__(self, index):
        return self.parts[index]

    @property
    def is_multi(self) -> bool:
        return len(self.parts) > 1

    def children_of(self, part_index: int) -> list:
        """Indices of the parts nested inside part ``part_index``.

        The direction matters: a nested part's vertices are inside its parent,
        but not the other way round, so only that asymmetry tells a nested part
        from a collision. ``load_parts`` has already refused collisions, so what
        is left here is exactly the nesting.
        """
        mine = self.parts[part_index]
        out = []
        for j, other in enumerate(self.parts):
            if j == part_index:
                continue
            if GEO.part_contains(mine, other) and not GEO.part_contains(other, mine):
                out.append(j)
        return out

    def filled_volume_m3(self, part_index: int, units: str) -> float:
        """The volume part ``part_index`` actually FILLS, in cubic metres.

        ``units`` has NO DEFAULT, deliberately. It was ``units="mm"`` and that
        default is how a real defect hid: ``mse_common.region_plan`` called this
        without the argument, so every multi-part upload's volume was converted
        as if written in millimetres no matter what the owner chose. Beside it
        ``h_min`` was true metres, so the estimate was wrong by the cube of the
        unit ratio -- measured through the real upload on ONE physical object:
        mm 51,187 (GPU) but in 3 (cluster, 17,000x under) and um 5.1e13, nm
        5.1e19 (refused). A defaulted unit argument is a place for this to hide
        again, so the caller must say which unit it means.

        Its own enclosed volume MINUS everything nested in it. A nested part's
        every point is inside its parent, so the parent's enclosed volume counts
        the child twice: once as "parent" and once as "child". Measured on a
        6mm box with a 2mm core, ``surface.volume`` of the merged PLC is 224 mm3
        where the object fills 216 -- the core, counted in both.

        This matters because the element count goes like ``V / h^3`` and the
        estimate is what ROUTES the job. The wall region is where the coarse
        elements are, so inflating its volume inflates the estimate, and a job
        that fits on the Jetson is sent to the cluster instead.

        One expression, used by both the mesher's per-region plan and the web
        app's estimate, so the two cannot disagree about what a region holds --
        the "one quantity, two spaces" shape this feature keeps meeting.
        """
        parts_volume = abs(float(self.parts[part_index].volume))
        for child in self.children_of(part_index):
            parts_volume -= abs(float(self.parts[child].volume))
        return max(parts_volume, 0.0) * unit_scale(units) ** 3

    def part_of_region(self, region_id: int) -> str:
        """The part name a region id belongs to. Names, never indices, in
        messages: the owner assigned materials by name."""
        return self.names[self.region_of_part.index(int(region_id))]

    def region_materials(self, materials):
        """``{region_id: n}`` for every region, from a name -> n table.

        Refuses a table that does not name every part, and a table naming a part
        that is not in the file. Both are mismatches between the upload and the
        geometry, and either one would otherwise apply a material to a region it
        was not meant for -- or leave a region at whatever the default happened
        to be.
        """
        by_name = {name: i for i, name in enumerate(self.names)}
        missing = [n for n in self.names if n not in materials]
        unknown = [k for k in materials if k not in by_name]
        if missing:
            raise MeshingError(
                f"no refractive index given for part(s) "
                f"{', '.join(repr(m) for m in missing)}. Every part needs one."
            )
        if unknown:
            raise MeshingError(
                f"the materials table names part(s) "
                f"{', '.join(repr(u) for u in unknown)} that are not in this "
                f"file. The file's parts are "
                f"{', '.join(repr(n) for n in self.names)}."
            )
        out = {}
        for name, n_part in materials.items():
            part_index = by_name[name]
            out[self.region_of_part[part_index]] = float(n_part)
        return out


def load_parts(obj_bytes_or_path, units: str = "mm"):
    """Split, validate and label a multi-part .obj. Returns a :class:`PartSet`.

    This is ``load_surface``'s multi-part counterpart, and it is deliberately
    STRICTER: ``load_surface`` takes the biggest part and ignores the rest, which
    for a multi-part file silently solves a different object than the one the
    owner uploaded. Here every part must be accounted for or the upload is
    refused.

    The checks, in the order they run, each naming the part it is about:

    1. **split** -- :func:`split_obj_parts`, which already refuses a name/part
       count mismatch and a duplicate or empty group.
    2. **closed** -- every part watertight. The row's "not closed" case, and it
       is currently masked for multi-part files because only the biggest part
       was ever checked.
    3. **no overlap** -- decided by TetGen's own ``docheck`` at meshing time,
       not here; see :func:`mse_meshing.tetrahedralize`. Overlap is the one
       case where the mesher's own view of the PLC is a better instrument than
       a second geometric opinion.
    4. **no touching** -- :func:`mse_geometry.touching_vertices`, with a
       tolerance. ``docheck`` does NOT catch this (measured: two boxes sharing
       a face plane mesh as 12 tets with no error), so it needs its own test.
    5. **one wall** -- exactly one part encloses every other, or the file is a
       single part with no wall. Two candidates, or none with several parts, is
       a refusal: the PEC boundary has to be somewhere specific.

    The seed for each region is found by :func:`find_region_seed` and then
    ASSERTED with a containment test, so a seed that slipped through the finder
    fails here by name rather than producing a mesh with a material in the
    wrong place.
    """
    names, parts = split_obj_parts(obj_bytes_or_path)
    parts = label_parts(names, parts)
    n = len(parts)

    # 2. every part closed
    for name, part in zip(names, parts):
        if not part.is_watertight:
            raise MeshingError(
                f"part {name!r} is not a closed surface (it has holes or open "
                "edges), so it cannot be a region of the cavity. Please upload "
                "a watertight .obj."
            )

    if n == 1:
        # The ordinary one-part upload. No regions to place and no wall to find,
        # and the seed is irrelevant. Returned with region 0 and wall None so
        # that callers have one shape to handle.
        return PartSet(names, parts, region_of_part=[0],
                       seeds=[None], wall_index=None)

    # 4. touching pairs, with the pre-filter first so the O(n_faces) winding
    #    test runs only on pairs whose bounding boxes are near each other.
    for i, j in GEO.parts_may_interact(parts):
        a_on_b, b_on_a = GEO.touching_vertices(parts[i], parts[j])
        # Assert BOTH directions separately. They are not symmetric on a given
        # fixture -- a summed count stays green when either side is dead, and
        # this exact case cost a mutation round to find.
        if a_on_b.size or b_on_a.size:
            raise MeshingError(
                f"parts {names[i]!r} and {names[j]!r} touch, so they cannot be "
                "separated into materials. Leave a gap between them, or merge "
                "them into one part."
            )

    # 5. the unique enclosing wall
    children = [children_of(i, parts) for i in range(n)]
    walls = [i for i in range(n) if len(children[i]) == n - 1]
    if len(walls) != 1:
        if not walls:
            raise MeshingError(
                "none of these parts encloses the others, so there is no PEC "
                "outer wall. The first part must contain every other part."
            )
        raise MeshingError(
            f"parts {', '.join(repr(names[w]) for w in walls)} each enclose "
            "every other part, so there is more than one candidate for the PEC "
            "outer wall. Nested parts must form a single chain, not a set of "
            "siblings."
        )
    wall_index = walls[0]

    # Region ids: 0 is the wall's FILL, then each non-wall part in file order.
    # The wall's fill is its enclosed volume minus every child, which is why a
    # region is not simply "the part" -- a wall region is a shell.
    region_of_part = [None] * n
    region_of_part[wall_index] = 0
    next_region = 1
    for i in range(n):
        if i == wall_index:
            continue
        region_of_part[i] = next_region
        next_region += 1

    seeds = []
    for i, name in enumerate(names):
        seed = find_region_seed(parts[i], [parts[c] for c in children[i]])
        # Assert the seed really is in its own region. The finder already
        # tested this, but the assertion is what makes a future change to the
        # finder fail HERE with the part named, rather than at mesh time with
        # no name at all.
        pts = np.asarray(seed, dtype=np.float64).reshape(1, 3)
        inside, on = GEO.point_in_mesh(parts[i], pts)
        if not bool(inside[0]) or bool(on[0]):
            raise MeshingError(
                f"internal error: the seed found for part {name!r} is not "
                f"inside it. This is a bug, not a problem with your upload."
            )
        for c in children[i]:
            c_in, c_on = GEO.point_in_mesh(parts[c], pts)
            if bool(c_in[0]) or bool(c_on[0]):
                raise MeshingError(
                    f"internal error: the seed found for part {name!r} lies "
                    f"inside the part {names[c]!r} nested in it. This is a bug, "
                    "not a problem with your upload."
                )
        seeds.append(seed)

    return PartSet(names, parts, region_of_part=region_of_part,
                   seeds=seeds, wall_index=wall_index)


def merged_surface(parts, units: str = "mm"):
    """Every part's geometry as ONE watertight ``trimesh.Trimesh``.

    This is the PLC TetGen needs: a single closed surface whose facets bound
    every region. Each part is already closed and they do not touch or overlap
    (``load_parts`` proved all three), so concatenating their triangles gives a
    valid outer boundary -- there is no boolean to perform and none of the
    missing boolean backends (``manifold3d``, ``rtree``, ``embreex`` are all
    absent from the solver image) is required.

    The vertices are COPIED and laid out in file order, so a seed found on a
    part (a point in that part's own coordinates) is still a point in this
    surface's coordinates. That is the property the mesher relies on, and it is
    the reason this returns a plain concatenation rather than a merged mesh that
    welds coincident vertices: welding would renumber, and the seeds are
    expressed in the pre-weld numbering.

    ``units`` is accepted for symmetry with the other loaders and is not used --
    no scaling happens here, because the parts are already in the user's units
    and the caller scales once.
    """
    import trimesh

    vs, fs = [], []
    v_at = 0
    for part in parts:
        pv = np.asarray(part.vertices, dtype=np.float64)
        pf = np.asarray(part.faces, dtype=np.int64)
        vs.append(pv)
        fs.append(pf + v_at)
        v_at += pv.shape[0]
    merged = trimesh.Trimesh(
        vertices=np.concatenate(vs, axis=0),
        faces=np.concatenate(fs, axis=0),
        process=False,
    )
    return merged


def part_name_of(part) -> str:
    """The name a part carries, if it was labelled, else a stable fallback.

    ``trimesh.Trimesh`` has no name attribute of its own -- names live in the
    group structure this module parses -- so callers that need a name in a
    message should carry the name alongside the part. This exists so a message
    built deep inside a helper still says something the owner can look up
    instead of ``<Trimesh>``.
    """
    for attr in ("part_name", "name"):
        value = getattr(part, attr, None)
        if isinstance(value, str) and value:
            return value
    return "<unnamed>"


def label_parts(names, parts) -> list:
    """Attach each name to its part as ``part_name``, for messages and debugging.

    A binding, not a copy: the same ``Trimesh`` objects come back, so a caller
    holding either reference sees the label.
    """
    if len(names) != len(parts):
        raise MeshingError(
            f"internal error: {len(names)} name(s) for {len(parts)} part(s)"
        )
    for name, part in zip(names, parts):
        try:
            part.part_name = name
        except Exception:  # noqa: BLE001 - a frozen mesh just keeps no label
            pass
    return parts


def children_of(index: int, parts) -> list:
    """Indices of parts strictly inside part ``index``.

    Containment is decided by :func:`mse_geometry.part_contains`, which is the
    direction-sensitive check: a nested part's vertices are inside its parent,
    but the parent's are not inside it, and only that asymmetry tells a nested
    part from a collision.
    """
    mine = parts[index]
    out = []
    for j, other in enumerate(parts):
        if j == index:
            continue
        if GEO.part_contains(mine, other) and not GEO.part_contains(other, mine):
            out.append(j)
    return out
