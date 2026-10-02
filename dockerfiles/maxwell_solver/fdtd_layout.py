"""Lay a ``td`` upload out on its grid: the spec's one fixed rule.

The worker solves this layout and phase 5's estimate counts it, so the run that
is estimated is the run that is solved. Nothing here allocates a field array;
every refusal comes before ``simulate`` is called.

Frames. The owner's frame has its origin at the minimum corner of the parts'
combined bounding box, in metres (mesh units x ``length_scale``); ``position``
and probes are given in it, and ``nodes`` are returned in it. The voxeliser
works in mesh coordinates, so it is handed the same nodes shifted back by that
corner.
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


def _refuse_conductor_nodes(names, parts, table, scene, nodes, lo, length_scale,
                            where):
    """A source or probe on a conductor node injects or reads nothing, which
    would read as "no response": refuse it, naming the position and part."""
    masks = ([scene["pec"]] if scene["pec"] is not None else []) + list(
        scene["metals"].values())
    if not masks:
        return
    finder = None
    for what, cell in where:
        if not any(bool(m[cell]) for m in masks):
            continue
        point = np.array([nodes[a][cell[a]] for a in range(3)])
        if finder is None:
            finder = _finder(names, parts, table)
        index = int(finder((point + lo) / float(length_scale))[0])
        owner = (f"part {names[index - 1]}" if index > 0
                 else "the metal boundary's conductor")
        raise LayoutError(
            f"{what} at {[round(float(x), 12) for x in point]} m sits inside "
            f"{owner}: a conductor carries no field there, so the run would "
            f"read as no response")


def plan(names, parts, request, *, length_scale: float, materials=None,
         n_index=None) -> dict:
    """``layout`` without the voxeliser: the grid, source placement and probes.

    Cheap enough for the web form (no per-node point-in-mesh test), so the
    form refuses a bad grid, position or probe before the job is queued, and
    ``estimate`` counts the same nodes ``layout`` will solve on. Returns
    ``nodes``, ``spacings``, ``shape``, ``box``, ``cell``, ``probes``,
    ``origin_m`` and ``band_top_hz`` as ``layout`` does, plus ``names``,
    ``table`` (part name -> material) and ``part_bounds`` (owner-frame
    metres, per part).
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

    lo, hi = _bounds(parts, length_scale)
    size = hi - lo
    pad = OPEN_PAD_CELLS * dx if boundary == "open" else 0.0
    f_top = band_top(request["profile"], request["frequency_hz"],
                     request["bandwidth"])

    # layers in grid coordinates: 0 is the padded minimum corner
    layers = [[], [], []]
    for axis in range(3):
        if pad:
            layers[axis].append(fdtd_grid.Layer(pad, pad + size[axis]))
    for name, part in zip(names, parts):
        plo, phi = _bounds([part], length_scale)
        plo, phi = plo - lo + pad, phi - lo + pad
        kind = table.get(name)
        metal = kind if isinstance(kind, str) and kind in fdtd.METALS else None
        depth = fdtd_grid.skin_depth(metal, f_top) if metal else None
        for axis in range(3):
            a, b = float(plo[axis]), float(phi[axis])
            if not b > a:
                continue           # the voxeliser refuses a flat part by name
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

    grid = fdtd_grid.build_grid(tuple(size + 2 * pad), dx, layers)
    nodes = tuple(n - pad for n in grid["nodes"])          # owner frame

    pml = fdtd.DEFAULT_PML_CELLS if boundary == "open" else 0
    box = None
    if plane:
        faces = [(_node_index(n, 0.0), _node_index(n, float(s)))
                 for n, s in zip(nodes, size)]
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
            "boundary": boundary}


def _finder(names, parts, table):
    """The voxeliser's own part lookup, shared-face rule included, so a point
    is named after the part the scene gave it."""
    return fdtd_voxel.PartIndex(
        parts, priority=[fdtd_voxel.priority_of(table[n]) for n in names],
        names=names)


def _conductor(kind) -> bool:
    return kind == "pec" or (isinstance(kind, str) and kind in fdtd.METALS)


def estimate(planned, request, *, record_frames=True) -> dict:
    """``fdtd.estimate_run`` for a ``plan``, before anything is voxelised.

    Each metal's region is taken as its parts' bounding boxes on the grid, a
    box holding every node the voxeliser can mark, so the memory errs high (by
    the corners a curved part leaves empty) and never admits a run the
    voxelised estimate would refuse. dt does not depend on the region
    (``stable_dt`` reads the metal and the spacings), so cells, dt and steps
    are the solved run's exactly.
    """
    nodes = planned["nodes"]
    shape = tuple(len(n) for n in nodes)
    metals, pec = {}, planned["boundary"] == "metal"
    for name, (plo, phi) in zip(planned["names"], planned["part_bounds"]):
        kind = planned["table"].get(name)
        if kind == "pec":
            pec = True
        elif _conductor(kind):
            region = metals.setdefault(kind, np.zeros(shape, dtype=bool))
            region[tuple(slice(int(np.searchsorted(n, a, "left")),
                               int(np.searchsorted(n, b, "right")))
                         for n, a, b in zip(nodes, plo, phi))] = True
    metals = {k: v for k, v in metals.items() if v.any()}
    return fdtd.estimate_run(nodes, duration=float(request["duration_s"]),
                             boundary=planned["boundary"], metals=metals,
                             pec=True if pec else None,
                             record_frames=record_frames)


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
    if not where or not (planned["boundary"] == "metal"
                         or any(_conductor(table.get(n)) for n in names)):
        return
    points = np.array([[nodes[a][c[a]] for a in range(3)] for _, c in where])
    index = _finder(names, parts, table)((points + lo) / float(length_scale))
    for (what, _), point, i in zip(where, points, index):
        i = int(i)
        if i > 0 and _conductor(table.get(names[i - 1])):
            owner = f"part {names[i - 1]}"
        elif i == 0 and planned["boundary"] == "metal":
            owner = "the metal boundary's conductor"
        else:
            continue
        raise LayoutError(
            f"{what} at {[round(float(x), 12) for x in point]} m sits inside "
            f"{owner}: a conductor carries no field there, so the run would "
            f"read as no response")


def layout(names, parts, request, *, length_scale: float, materials=None,
           n_index=None) -> dict:
    """The grid, voxelised scene, source placement and probes for a td job.

    ``request`` is ``td_solve.request_from``'s dict. Returns ``nodes`` (owner
    frame, metres), ``spacings``, ``shape``, ``eps``, ``pec``, ``metals``,
    ``box`` (TF/SF node-index corners, plane wave only), ``cell`` (point
    source node), ``probes`` ([(i, j, k, component)]), ``origin_m`` (the
    bounding box's minimum corner in mesh-frame metres) and ``caveats``
    (strings for the result, e.g. a non-box part's thin-feature caveat).
    """
    planned = plan(names, parts, request, length_scale=length_scale,
                   materials=materials, n_index=n_index)
    names, table, nodes = planned["names"], planned["table"], planned["nodes"]
    lo, cell, probes = planned["origin_m"], planned["cell"], planned["probes"]
    scene = fdtd_voxel.voxelise(
        names, parts, table, tuple(n + lo[i] for i, n in enumerate(nodes)),
        boundary=planned["boundary"], length_scale=length_scale)
    _refuse_conductor_nodes(names, parts, table, scene, nodes, lo, length_scale,
                            ([("the source position", cell)] if cell else [])
                            + [("probe", p[:3]) for p in probes])
    caveats = []
    for name, part in zip(names, parts):
        plo, phi = _bounds([part], 1.0)
        if _volume(part) < BOX_FILL * float(np.prod(phi - plo)):
            caveats.append(f"{THIN_CAVEAT} (part {name} is not a box)")
    return {"caveats": caveats, "nodes": nodes, "spacings": planned["spacings"],
            "shape": planned["shape"], "eps": scene["eps"], "pec": scene["pec"],
            "metals": scene["metals"], "box": planned["box"], "cell": cell,
            "probes": probes, "origin_m": lo, "band_top_hz": planned["band_top_hz"]}
