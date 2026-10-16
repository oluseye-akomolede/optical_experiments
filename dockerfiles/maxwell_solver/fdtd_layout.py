"""Lay a ``td`` upload out on its grid: the spec's one fixed rule.

The worker solves this layout and phase 5's estimate counts it, so the run that
is estimated is the run that is solved. Nothing here allocates a field array;
every refusal comes before ``simulate`` is called.

Frames. The owner's frame has its origin at the minimum corner of the parts'
combined bounding box, in metres (mesh units x ``length_scale``); ``position``
and probes are given in it, and ``nodes`` are returned in it. The voxeliser
works in mesh coordinates, so it is handed the same nodes shifted back by that
corner.

Film thickness. Every flat face that bounds a part's box (a plate's face, a
film's face, a cylinder's end cap) is a grid line along its normal axis, and a
node on a conductor face is a conductor node (``average_eps`` samples on the
node; the voxeliser gives a shared face to the conductor). So a film flush
between two conductor plates keeps exactly its thickness wherever the parts
sit. Without the lines, the film read one cell too thick on almost every grid
offset from its faces (the reviewer measured 5 cells for a 4-cell film at 19 of
20 offsets). A curved part has no flat face there, so its box adds no line.

Coatings (``request["coatings"]``: part name -> [material, thickness in m]).
A coating is a shell outside its part's surface (``fdtd_voxel.CoatedIndex``).
Each axis of a coated part gets ``fdtd_grid.coating_layers``: two thin layers
at its faces where both are flat, else its whole span resolved at thickness /
4, plus the skin-depth spacing for a metal coating. In open space the grid and
a plane wave's TF/SF box cover the shells; the frame's origin stays at the
parts' corner. In a metal cavity an outermost part's shell would lie in the
wall, so it is refused.
"""
from __future__ import annotations

import numpy as np

import fdtd
import fdtd_grid
import fdtd_voxel

#: Cells of padding on every side of an ``open`` scene: the 3-cell TF/SF
#: clearance, 3 more, and the CPML (the spec's 16).
OPEN_PAD_CELLS = fdtd.BOX_CLEARANCE + 3 + fdtd.DEFAULT_PML_CELLS
#: A part thinner than this many coarse cells on an axis gets a layer there,
#: with this many cells across it.
THIN_CELLS = 4
#: A part filling less than this share of its bounding box is not a box: its
#: thin walls may be unrefined, so the result says so.
BOX_FILL = 0.99
THIN_CAVEAT = "thin curved features are resolved only at the coarse cell"

#: A metal face is resolved this many skin depths deep ...
SKIN_DEPTHS = 3
#: ... at this many cells per skin depth.
CELLS_PER_SKIN = 4
BOUNDARIES = ("metal", "open")


class LayoutError(fdtd.FDTDError):
    """The upload and request cannot be laid out (named in the message)."""


def band_top(profile: str, frequency: float, bandwidth) -> float:
    """The top of the source band the skin depth is taken at (Hz)."""
    if profile == "pulse":
        return float(frequency) + float(bandwidth)
    return float(frequency)


def _bounds(parts, length_scale):
    pts = np.concatenate([np.asarray(p.vertices, dtype=float).reshape(-1, 3)
                          for p in parts]) * float(length_scale)
    return pts.min(axis=0), pts.max(axis=0)


def _flat_faces(part, axis, length_scale):
    """The part's box faces on ``axis`` that are flat: some triangle of the
    part lies in that plane (to 1e-6 of the part's size). Metres."""
    v = np.asarray(part.vertices, dtype=float).reshape(-1, 3) * float(length_scale)
    tri = v[np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)][:, :, axis]
    lo, hi = v[:, axis].min(), v[:, axis].max()
    tol = 1e-6 * float(np.ptp(v, axis=0).max())
    return [face for face in (lo, hi)
            if np.any(np.all(np.abs(tri - face) <= tol, axis=1))]


def _materials(names, materials, n_index):
    if materials:
        return dict(materials)
    if len(names) == 1 and n_index is not None:
        return {names[0]: float(n_index)}
    raise LayoutError(f"{len(names)} parts need a materials table; got none")


def _volume(part):
    """Enclosed volume of a closed triangle mesh (divergence theorem)."""
    v = np.asarray(part.vertices, dtype=float).reshape(-1, 3)
    t = v[np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)]
    return abs(float(np.einsum("ij,ij->i", t[:, 0],
                               np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


def _node_index(nodes, value):
    """The index of the node at ``value`` (a layer edge, so present)."""
    i = int(np.argmin(np.abs(nodes - value)))
    if not np.isclose(nodes[i], value, rtol=0, atol=1e-9 * max(nodes[-1], 1e-300)):
        raise LayoutError(f"no grid node at {value:.6g} m")
    return i


def _locate(nodes, point, what, pml=0):
    """Nearest node indices for a frame point, refused outside the grid or
    inside the ``pml`` absorbing cells at either end of an axis (a source there
    is absorbed as it is injected, a probe there reads a damped field)."""
    cell = []
    for axis, (n, v) in enumerate(zip(nodes, point)):
        v = float(v)
        if not (n[0] <= v <= n[-1]):
            raise LayoutError(
                f"{what} {list(point)} lies outside the grid: "
                f"{'xyz'[axis]} spans {n[0]:.6g}..{n[-1]:.6g} m")
        i = int(np.argmin(np.abs(n - v)))
        if pml and not (pml <= i <= len(n) - 1 - pml):
            raise LayoutError(
                f"{what} {list(point)} lies inside the absorbing layer: "
                f"{'xyz'[axis]} must lie within {n[pml]:.6g}..{n[-1 - pml]:.6g} m")
        cell.append(i)
    return tuple(cell)


def _owner(index, names, table, coats):
    """``(words, material)`` for a lookup index: a part, a coating or 0."""
    if 0 < index <= len(names):
        return f"part {names[index - 1]}", table.get(names[index - 1])
    if index > len(names):
        p, material, _t = coats[index - len(names) - 1]
        return f"the coating of part {names[p]}", material
    return "the metal boundary's conductor", None


def _refuse_conductor_nodes(names, parts, table, scene, nodes, lo, length_scale,
                            where, finder):
    """A source or probe on a conductor node injects or reads nothing, which
    would read as "no response": refuse it, naming the position and part."""
    masks = ([scene["pec"]] if scene["pec"] is not None else []) + list(
        scene["metals"].values())
    if not masks:
        return
    for what, cell in where:
        if not any(bool(m[cell]) for m in masks):
            continue
        point = np.array([nodes[a][cell[a]] for a in range(3)])
        index_of, coats = finder()
        index = int(index_of((point + lo) / float(length_scale))[0])
        owner = _owner(index, names, table, coats)[0]
        raise LayoutError(
            f"{what} at {[round(float(x), 12) for x in point]} m sits inside "
            f"{owner}: a conductor carries no field there, so the run would "
            f"read as no response")


def plan(names, parts, request, *, length_scale: float, materials=None,
         n_index=None, pad_cells=None, pml_cells=None) -> dict:
    """``layout`` without the voxeliser: the grid, source placement and probes.

    Cheap enough for the web form (no per-node point-in-mesh test), so the
    form refuses a bad grid, position or probe before the job is queued, and
    ``estimate`` counts the same nodes ``layout`` will solve on. Returns
    ``nodes``, ``spacings``, ``shape``, ``box``, ``cell``, ``probes``,
    ``origin_m`` and ``band_top_hz`` as ``layout`` does, plus ``names``,
    ``table`` (part name -> material) and ``part_bounds`` (owner-frame
    metres, per part).

    ``request["pin_source_z"]`` (default off) puts a point source's own z on
    the z axis as a node, so the source sits where it was asked to rather than
    at the nearest node ``_locate`` finds.

    The open boundary's own geometry is ``pad_cells`` and ``pml_cells``, and
    passing them is the ONLY way to it: both default to this module's
    ``OPEN_PAD_CELLS`` and ``fdtd.DEFAULT_PML_CELLS`` read at call time, and
    ``request`` is not consulted for either. The web form and the worker build
    their own request and call this with neither, so the product's boundary is
    the product's. The bench script is the one caller that passes them
    (``--pad-nm``/``--pml-nm``): the pad is a cell COUNT, so at ``--dx`` 10 the
    product's own 16-cell pad is 160 nm against the 320 nm r20 ran with, and a
    dx-10 record priced without them would move the boundary's distance at the
    same time as the resolution, leaving the two indistinguishable. A
    ``pml_cells`` that is not inside ``pad_cells`` is refused by the caller
    (``scripts/paper1_nanowire.py``'s own guard), not here.
    """
    boundary = request["boundary"]
    if boundary not in BOUNDARIES:
        raise LayoutError(f"boundary {boundary!r} must be one of "
                          f"{', '.join(BOUNDARIES)}; periodic is a phase 6 option")
    plane = request["source_kind"] == "plane"
    if plane and boundary == "metal":
        raise LayoutError("a plane wave needs 'open' boundaries; a metal "
                          "cavity has nowhere for it to come from")
    dx = float(request["dx_m"])
    if not dx > 0:
        raise LayoutError(f"dx_m must be positive, got {request['dx_m']!r}")
    names = list(names)
    table = _materials(names, materials, n_index)
    coats = fdtd_voxel.coating_table(names, request.get("coatings"))
    if coats and boundary == "metal":
        depths = fdtd_voxel.nesting_depths(parts) if len(parts) > 1 else [0]
        outer = [names[p] for p, _m, _t in coats if depths[p] == 0]
        if outer:
            raise LayoutError(
                f"part(s) {', '.join(map(repr, outer))} enclose the cavity, so "
                f"a coating on them would lie in the metal boundary's wall. Coat "
                f"a part inside the cavity, or use 'open' boundaries.")
    thick = {p: t for p, _m, t in coats}

    lo, hi = _bounds(parts, length_scale)
    # the grid's extent: the parts, and in open space their coatings' shells
    olo, ohi = lo.copy(), hi.copy()
    if boundary == "open":
        for p, t in thick.items():
            plo, phi = _bounds([parts[p]], length_scale)
            olo, ohi = np.minimum(olo, plo - t), np.maximum(ohi, phi + t)
    size, off = ohi - olo, lo - olo
    pad_cells = OPEN_PAD_CELLS if pad_cells is None else int(pad_cells)
    pad = pad_cells * dx if boundary == "open" else 0.0
    f_top = band_top(request["profile"], request["frequency_hz"],
                     request["bandwidth"])

    # layers in grid coordinates: 0 is the padded minimum corner of the extent
    layers, lines = [[], [], []], [[], [], []]
    for axis in range(3):
        if pad:
            layers[axis].append(fdtd_grid.Layer(pad, pad + size[axis]))
    coat_bounds = []
    for p, material, t in coats:
        plo, phi = _bounds([parts[p]], length_scale)
        coat_bounds.append((material, plo - t - lo, phi + t - lo))
        depth = (fdtd_grid.skin_depth(material, f_top)
                 if isinstance(material, str) and material in fdtd.METALS else None)
        for axis in range(3):
            a, b = float(plo[axis] - olo[axis] + pad), float(phi[axis] - olo[axis] + pad)
            flat = len(_flat_faces(parts[p], axis, length_scale)) == 2
            shell = fdtd_grid.coating_layers(a, b, t, flat=flat)
            layers[axis] += shell
            if depth is not None:
                # where the thickness rule covers the same span, build_grid
                # keeps the smaller spacing of the two
                layers[axis] += [fdtd_grid.Layer(s.lo, s.hi, depth / CELLS_PER_SKIN)
                                 for s in shell]
    for name, part in zip(names, parts):
        plo, phi = _bounds([part], length_scale)
        plo, phi = plo - olo + pad, phi - olo + pad
        kind = table.get(name)
        metal = kind if isinstance(kind, str) and kind in fdtd.METALS else None
        depth = fdtd_grid.skin_depth(metal, f_top) if metal else None
        for axis in range(3):
            a, b = float(plo[axis]), float(phi[axis])
            if not b > a:
                continue           # the voxeliser refuses a flat part by name
            lines[axis] += [face - olo[axis] + pad
                            for face in _flat_faces(part, axis, length_scale)]
            thin = (b - a) / THIN_CELLS if b - a < THIN_CELLS * dx else None
            if thin is not None:
                layers[axis].append(fdtd_grid.Layer(a, b, thin))
            if depth is not None:
                fine = depth / CELLS_PER_SKIN
                if b - a < 2 * SKIN_DEPTHS * depth:
                    # where the thin rule's layer covers the same span,
                    # build_grid keeps the smaller spacing of the two
                    layers[axis].append(fdtd_grid.Layer(a, b, fine))
                else:
                    layers[axis] += [fdtd_grid.Layer(a, a + SKIN_DEPTHS * depth, fine),
                                     fdtd_grid.Layer(b - SKIN_DEPTHS * depth, b, fine)]

    if request.get("pin_source_z") and not plane:
        # A point source's own z becomes a node, so it sits at the depth it was
        # asked for. Without this the source is SNAPPED to the nearest node
        # (``_locate``), which is a different geometry and not a coarser one: on
        # this scene's own meshes a dipole asked 30 nm under the top facet
        # landed 39.99 nm there at dx 20 and 29.89 nm at dx 30, and the two
        # depths are two different Purcell factors. Opt-in, because pinning an
        # axis that would otherwise be uniform makes it graded and costs cells;
        # the line is still dropped by ``graded_axis`` if it is out of range or
        # within ``LINE_SNAP`` of the extent's edges, so a caller that cares
        # reads the planned node back -- see ``scripts/paper1_nanowire.py``,
        # which refuses a run whose realised depth is not the depth it asked
        # for. The in-plane coordinates are NOT pinned: x and y are uniform at
        # ``dx`` for every part here, so a line there would grade them.
        lines[2] = lines[2] + [float(request["position"][2]) - olo[2] + pad]
    grid = fdtd_grid.build_grid(tuple(size + 2 * pad), dx, layers, lines)
    nodes = tuple(n - pad - off[a]                         # owner frame
                  for a, n in enumerate(grid["nodes"]))

    pml = ((fdtd.DEFAULT_PML_CELLS if pml_cells is None else int(pml_cells))
           if boundary == "open" else 0)
    box = None
    if plane:
        faces = [(_node_index(n, float(-o)), _node_index(n, float(s - o)))
                 for n, s, o in zip(nodes, size, off)]
        box = (tuple(f[0] - fdtd.BOX_CLEARANCE for f in faces),
               tuple(f[1] + fdtd.BOX_CLEARANCE for f in faces))
    cell = None
    if not plane:
        cell = _locate(nodes, request["position"], "the source position", pml)
    probes = []
    for probe in request["probes"] or ():
        if len(probe) != 4 or probe[3] not in fdtd.COMPONENTS:
            raise LayoutError(f"a probe is [x, y, z, component], got {probe!r}")
        probes.append(_locate(nodes, probe[:3], "probe", pml) + (probe[3],))
    part_bounds = [tuple(b - lo for b in _bounds([part], length_scale))
                   for part in parts]
    return {"names": names, "table": table, "nodes": nodes,
            "spacings": grid["spacings"], "shape": grid["shape"], "box": box,
            "cell": cell, "probes": probes, "origin_m": lo,
            "band_top_hz": f_top, "part_bounds": part_bounds,
            "boundary": boundary, "coatings": coats, "coat_bounds": coat_bounds,
            "pad_cells": pad_cells if boundary == "open" else 0,
            "pml_cells": pml}


def _finder(names, parts, table, coatings, boundary, length_scale, nodes):
    """The voxeliser's own lookup, shared-face rule, coatings and face
    tolerance included, so a point is named after the part or coating the
    scene gave it. Returns ``(index_of, coats)`` as
    ``fdtd_voxel.coated_index`` does."""
    return fdtd_voxel.coated_index(names, parts, table, coatings,
                                   boundary=boundary, length_scale=length_scale,
                                   snap=fdtd_voxel.grid_snap(nodes, length_scale))


def _conductor(kind) -> bool:
    return kind == "pec" or (isinstance(kind, str) and kind in fdtd.METALS)


def _fs(seconds: float) -> str:
    """Seconds as a readable number of fs (the form's own unit)."""
    return f"{float(seconds) / 1e-15:.6g} fs"


def _step_at(seconds: float, dt: float, *, up: bool) -> int:
    """The step index at ``seconds``: the first at or after it (``up``), else
    the last at or before it. The epsilon absorbs the float noise of a time
    that is a whole number of steps, so 3 steps is step 3 and not 4."""
    ratio = float(seconds) / float(dt)
    return int(np.ceil(ratio - 1e-9) if up else np.floor(ratio + 1e-9))


def movie_frames(request):
    """The requested frame count as a whole number, or None for the default.

    Refused here, before any estimate: the count alone can be too large or not
    a count, and the reader should hear that rather than the memory limit it
    would have hit. Null (the form's blank field, or a job written before the
    window existed) means the solver's default, ``fdtd.MAX_FRAMES``.
    """
    frames = request.get("frame_count")
    if frames is None:
        return None
    try:
        whole = float(frames).is_integer()
    except (TypeError, ValueError):
        whole = False
    if not whole or int(frames) < 1:
        raise LayoutError(
            f"The movie's frame count must be a whole number of at least 1, "
            f"got {frames!r}.")
    if int(frames) > fdtd.MAX_MOVIE_FRAMES:
        raise LayoutError(
            f"The movie asks for {int(frames)} frames; at most "
            f"{fdtd.MAX_MOVIE_FRAMES} can be recorded. Ask for "
            f"{fdtd.MAX_MOVIE_FRAMES} or fewer.")
    return int(frames)


def movie_window(request, est, frames=None) -> dict:
    """``fdtd.frame_plan`` for the request's movie window, or a refusal.

    The form's ``movie_start_s``/``movie_end_s`` are seconds (null: the whole
    run) and ``frame_count`` is null for the default 60. The window is turned
    into step indices here, once, so the form and the worker spread the frames
    over the same steps and refuse the same windows in the same words. Every
    number in a message is the run's own: its duration, its step count, its dt.
    """
    steps, dt = int(est["steps"]), float(est["dt"])
    duration = steps * dt
    first_s, last_s = request.get("movie_start_s"), request.get("movie_end_s")
    frames = movie_frames(request) if frames is None else frames
    start = 0.0 if first_s is None else float(first_s)
    end = duration if last_s is None else float(last_s)
    if start < 0 or end > duration * (1 + 1e-9):
        raise LayoutError(
            f"The movie window runs from {_fs(start)} to {_fs(end)}, outside "
            f"the run's own {_fs(duration)} ({steps:,} steps). Set the window "
            f"inside the run, or leave both times blank for the whole run.")
    if end <= start:
        raise LayoutError(
            f"The movie window ends at {_fs(end)}, at or before its start of "
            f"{_fs(start)}. Set the end after the start, or leave both blank "
            f"for the whole run.")
    first_step, last_step = _step_at(start, dt, up=True), _step_at(end, dt, up=False)
    last_step = min(last_step, steps - 1)
    if last_step < first_step:
        raise LayoutError(
            f"The movie window from {_fs(start)} to {_fs(end)} holds no time "
            f"steps: the run's dt is {_fs(dt)}. Widen the window.")
    span = last_step - first_step + 1
    if frames is not None and frames > span:
        raise LayoutError(
            f"The movie asks for {frames} frames, but its window from "
            f"{_fs(start)} to {_fs(end)} holds only {span:,} time steps (the "
            f"run's dt is {_fs(dt)}). Ask for {span:,} frames or fewer, or "
            f"widen the window.")
    return fdtd.frame_plan(tuple(est["shape"]), steps,
                           max_frames=frames or fdtd.MAX_FRAMES,
                           first_step=first_step, last_step=last_step)


def estimate(planned, request, *, record_frames=True) -> dict:
    """``fdtd.estimate_run`` for a ``plan``, before anything is voxelised.

    Each metal's region is every node whose dual cell overlaps one of its
    parts' bounding boxes, or a coating's (its part's box grown by the
    thickness). Only those nodes can have any of their cell in the
    part, so no other node can be marked, whatever the grading or the float
    noise in a node that sits on a face (``searchsorted`` on the box itself
    dropped those). The memory errs high (by that ring and the corners a curved
    part leaves empty) and never admits a run the voxelised estimate would
    refuse. dt does not depend on the region
    (``stable_dt`` reads the metal and the spacings), so cells, dt and steps
    are the solved run's exactly.
    """
    nodes = planned["nodes"]
    shape = tuple(len(n) for n in nodes)
    metals, pec = {}, planned["boundary"] == "metal"
    duals = [fdtd_grid._dual_bounds(np.asarray(n, dtype=float)) for n in nodes]
    regions = [(planned["table"].get(name), plo, phi)
               for name, (plo, phi) in zip(planned["names"], planned["part_bounds"])]
    for kind, plo, phi in regions + list(planned.get("coat_bounds", ())):
        if kind == "pec":
            pec = True
        elif _conductor(kind):
            region = metals.setdefault(kind, np.zeros(shape, dtype=bool))
            # dual hi > a and dual lo < b: the cell overlaps [a, b]
            region[tuple(slice(int(np.searchsorted(hi, a, "right")),
                               int(np.searchsorted(lo, b, "left")))
                         for (lo, hi), a, b in zip(duals, plo, phi))] = True
    metals = {k: v for k, v in metals.items() if v.any()}
    # The movie is counted at the frame count the run will record: the request
    # is checked against this estimate, so a 120-frame movie must be refused
    # here rather than after it has been queued.
    frames = movie_frames(request)
    return fdtd.estimate_run(nodes, duration=float(request["duration_s"]),
                             boundary=planned["boundary"], metals=metals,
                             pec=True if pec else None,
                             record_frames=record_frames,
                             max_frames=fdtd.MAX_FRAMES if frames is None
                             else frames)


def refuse_conductor_points(planned, parts, *, length_scale: float) -> None:
    """The form's version of ``layout``'s conductor-node refusal.

    Tests only the source and probe nodes (a handful of points, not the grid):
    a node inside a pec or metal part, or outside every part of a metal-walled
    cavity, is refused naming the position and part. ``layout`` still makes
    the exact check on the voxelised masks when the job runs.
    """
    nodes, lo = planned["nodes"], planned["origin_m"]
    where = (([("the source position", planned["cell"])] if planned["cell"] else [])
             + [("probe", p[:3]) for p in planned["probes"]])
    names, table = planned["names"], planned["table"]
    coats = planned.get("coatings") or []
    if not where or not (planned["boundary"] == "metal"
                         or any(_conductor(table.get(n)) for n in names)
                         or any(_conductor(m) for _p, m, _t in coats)):
        return
    points = np.array([[nodes[a][c[a]] for a in range(3)] for _, c in where])
    index_of, coats = _finder(names, parts, table,
                              {names[p]: (m, t) for p, m, t in coats},
                              planned["boundary"], length_scale, nodes)
    index = index_of((points + lo) / float(length_scale))
    for (what, _), point, i in zip(where, points, index):
        i = int(i)
        owner, kind = _owner(i, names, table, coats)
        if not (_conductor(kind) or (i == 0 and planned["boundary"] == "metal")):
            continue
        raise LayoutError(
            f"{what} at {[round(float(x), 12) for x in point]} m sits inside "
            f"{owner}: a conductor carries no field there, so the run would "
            f"read as no response")


def layout(names, parts, request, *, length_scale: float, materials=None,
           n_index=None, pad_cells=None, pml_cells=None) -> dict:
    """The grid, voxelised scene, source placement and probes for a td job.

    ``request`` is ``td_solve.request_from``'s dict. Returns ``nodes`` (owner
    frame, metres), ``spacings``, ``shape``, ``eps``, ``pec``, ``metals``,
    ``box`` (TF/SF node-index corners, plane wave only), ``cell`` (point
    source node), ``probes`` ([(i, j, k, component)]), ``origin_m`` (the
    bounding box's minimum corner in mesh-frame metres), ``pad_cells`` and
    ``pml_cells`` (the open boundary's own geometry, as planned -- see
    ``plan``, which is the only place either is read) and ``caveats``
    (strings for the result, e.g. a non-box part's thin-feature caveat).
    """
    planned = plan(names, parts, request, length_scale=length_scale,
                   materials=materials, n_index=n_index, pad_cells=pad_cells,
                   pml_cells=pml_cells)
    names, table, nodes = planned["names"], planned["table"], planned["nodes"]
    lo, cell, probes = planned["origin_m"], planned["cell"], planned["probes"]
    coatings = request.get("coatings")
    scene = fdtd_voxel.voxelise(
        names, parts, table, tuple(n + lo[i] for i, n in enumerate(nodes)),
        boundary=planned["boundary"], length_scale=length_scale,
        coatings=coatings)
    _refuse_conductor_nodes(
        names, parts, table, scene, nodes, lo, length_scale,
        ([("the source position", cell)] if cell else [])
        + [("probe", p[:3]) for p in probes],
        lambda: _finder(names, parts, table, coatings, planned["boundary"],
                        length_scale, nodes))
    caveats = []
    for name, part in zip(names, parts):
        plo, phi = _bounds([part], 1.0)
        if _volume(part) < BOX_FILL * float(np.prod(phi - plo)):
            caveats.append(f"{THIN_CAVEAT} (part {name} is not a box)")
    return {"caveats": caveats, "nodes": nodes, "spacings": planned["spacings"],
            "shape": planned["shape"], "eps": scene["eps"], "pec": scene["pec"],
            "metals": scene["metals"], "box": planned["box"], "cell": cell,
            "probes": probes, "origin_m": lo, "band_top_hz": planned["band_top_hz"],
            "pad_cells": planned["pad_cells"], "pml_cells": planned["pml_cells"]}
