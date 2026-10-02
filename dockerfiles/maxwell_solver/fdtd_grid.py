"""The graded FDTD grid: thin layers resolved, coarse elsewhere.

A tensor-product grid is three node lists, one per axis. Each axis is built
from LAYERS: intervals that must be resolved finer than the coarse cell (a
film, a coating, a metal skin). Their edges become nodes, so a flat interface
lies on a grid plane, and

* every interval between two consecutive edges that is thinner than the
  coarse cell gets at least ``MIN_CELLS`` cells across it;
* a layer that asks for a spacing (a metal's skin depth / 4) gets it;
* neighbouring spacings differ by at most ``MAX_RATIO`` (the spec's 1.3),
  growing geometrically away from the fine layers until they reach the
  coarse cell.

Interfaces that are not grid planes (curved or oblique surfaces) are handled
by ``average_eps``: each node's permittivity is the volume average over its
dual cell of the materials sampled in it, so a layer thinner than one cell
still counts.

Coatings: a coating is a material plus a thickness on a region's surface.
On an axis-aligned flat face it is a thin layer at that face; on a curved
surface the shell crosses every grid line of the region's span at a different
place, so the whole span (widened by the thickness) is resolved at
thickness / 4. That is why a curved coated object can cost far more cells
than a flat film of the same thickness, and the estimate says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fdtd

MAX_RATIO = 1.3          # the spec's neighbour spacing ratio
MIN_CELLS = 4            # cells across a layer thinner than the coarse cell
GROWTH = 0.2             # target growth per cell; leaves margin under 1.3
_SAMPLES = 200           # at least this many quadrature points per interval
_SUBSTEPS = 8            # quadrature points per cell of the envelope
_REPAIRS = 60
_GROWTH_LIMIT = 4       # repairs may at most quadruple the first attempt


class GridError(fdtd.FDTDError):
    """The grid cannot be built as asked (bad extent, coarse or layer)."""


@dataclass(frozen=True)
class Layer:
    """An interval on one axis to resolve: ``spacing`` None means only the
    thin-layer rule (``MIN_CELLS`` across it when thinner than the coarse
    cell); a number caps the spacing inside it."""
    lo: float
    hi: float
    spacing: float | None = None


def skin_depth(metal, frequency: float) -> float:
    """The field's 1/e decay length in ``metal`` at ``frequency`` (m)."""
    eps = complex(fdtd.permittivity(metal, frequency))
    kappa = abs(np.sqrt(eps).imag)
    if kappa == 0:
        raise GridError(f"{metal} has no loss at {frequency:.3g} Hz")
    return fdtd.C0 / (2 * np.pi * frequency * kappa)


def coating_layers(lo: float, hi: float, thickness: float, *,
                   flat: bool = True) -> list:
    """The layers one axis needs for a coating on a region spanning lo..hi.

    ``flat``: the region's faces normal to this axis are planes, so the
    coating is two thin layers at them. Otherwise the shell crosses the
    whole span, which is resolved at thickness / ``MIN_CELLS``.
    """
    if not thickness > 0:
        raise GridError(f"a coating needs a positive thickness, got {thickness!r}")
    if flat:
        return [Layer(lo - thickness, lo), Layer(hi, hi + thickness)]
    return [Layer(lo - thickness, hi + thickness, thickness / MIN_CELLS)]


def _envelope(x, lo, hi, h_lo, h_hi, cap):
    """Largest spacing allowed at x inside one interval."""
    return np.minimum(np.minimum(h_lo + GROWTH * (x - lo),
                                 h_hi + GROWTH * (hi - x)), cap)


def _fill(lo, hi, h_lo, h_hi, cap, minimum):
    """Interior nodes of lo..hi: spacings follow the envelope, the slack is
    taken from the largest cells (capped at c) so the edge cells keep the
    spacing their fixed node asks for."""
    # quadrature points at an eighth of the local spacing: evenly spaced
    # points coarser than the fine end never resolve the grading there, and
    # every repair then shrinks that end further (seen: 288 M nodes)
    x = [lo]
    while x[-1] < hi:
        x.append(x[-1] + _envelope(x[-1], lo, hi, h_lo, h_hi, cap) / _SUBSTEPS)
    x = np.asarray(x)
    x[-1] = hi
    if x.size < _SAMPLES:
        x = np.unique(np.concatenate([x, np.linspace(lo, hi, _SAMPLES)]))
    env = _envelope(x, lo, hi, h_lo, h_hi, cap)

    def count(c):
        inv = 1.0 / np.minimum(env, c)
        return float(np.sum(0.5 * (inv[1:] + inv[:-1]) * np.diff(x)))

    cells = max(int(np.ceil(count(env.max()) - 1e-9)), minimum)
    top, bottom = env.max(), env.max()
    while count(bottom) < cells:
        bottom *= 0.5
    for _ in range(80):                 # count(c) falls as c rises
        mid = 0.5 * (top + bottom)
        top, bottom = (mid, bottom) if count(mid) < cells else (top, mid)
    inv = 1.0 / np.minimum(env, 0.5 * (top + bottom))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(x))])
    s *= cells / s[-1]
    return np.interp(np.arange(1, cells), s, x)


def graded_axis(length: float, coarse: float, layers=(), *,
                ratio: float = MAX_RATIO, min_cells: int = MIN_CELLS):
    """Node positions 0..length for one axis (layer edges included)."""
    if not length > 0 or not coarse > 0:
        raise GridError(f"length and coarse cell must be positive, got "
                        f"{length!r}, {coarse!r}")
    layers = [layer if isinstance(layer, Layer) else Layer(*layer)
              for layer in layers]
    for layer in layers:
        if not layer.hi > layer.lo:
            raise GridError(f"a layer must have hi > lo, got {layer}")
        if layer.spacing is not None and not layer.spacing > 0:
            raise GridError(f"a layer's spacing must be positive, got {layer}")
    edges = [0.0, float(length)]
    for layer in layers:
        edges += [min(max(e, 0.0), length) for e in (layer.lo, layer.hi)]
    edges = np.unique(np.round(np.asarray(edges) / length, 12)) * length
    gaps = np.diff(edges)

    need = np.full(gaps.size, float(coarse))      # spacing cap per interval
    for k, gap in enumerate(gaps):
        if gap < coarse:
            need[k] = gap / min_cells
        middle = edges[k] + 0.5 * gap
        for layer in layers:
            if layer.spacing is not None and layer.lo <= middle <= layer.hi:
                need[k] = min(need[k], layer.spacing)

    node_h = np.empty(edges.size)                 # spacing asked at each edge
    node_h[0], node_h[-1] = need[0], need[-1]
    node_h[1:-1] = np.minimum(need[:-1], need[1:])

    first = None
    for _ in range(_REPAIRS):
        # the growth limit, carried between edges: h_i <= h_j + g |x_i - x_j|
        h = node_h.copy()
        for i in range(1, h.size):
            h[i] = min(h[i], h[i - 1] + GROWTH * gaps[i - 1])
        for i in range(h.size - 2, -1, -1):
            h[i] = min(h[i], h[i + 1] + GROWTH * gaps[i])
        nodes = [edges[:1]]
        for k, gap in enumerate(gaps):
            minimum = min_cells if gap < coarse else 1
            nodes.append(_fill(edges[k], edges[k + 1], h[k], h[k + 1],
                               need[k], minimum))
            nodes.append(edges[k + 1:k + 2])
        nodes = np.concatenate(nodes)
        first = nodes.size if first is None else first
        if nodes.size > _GROWTH_LIMIT * first:
            raise GridError(f"grading did not converge: repairs grew the axis "
                            f"from {first} to {nodes.size} nodes")
        widths = np.diff(nodes)
        steps = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
        bad = np.flatnonzero(steps > ratio)
        if bad.size == 0:
            return nodes
        # tighten the edge nearest each violation and rebuild
        for b in bad:
            node_h[np.argmin(np.abs(edges - nodes[b + 1]))] *= 0.7
    raise GridError(f"could not grade the axis within ratio {ratio} after "
                    f"{_REPAIRS} repairs")


def check_axis(nodes, coarse: float, layers=(), *, ratio: float = MAX_RATIO,
               min_cells: int = MIN_CELLS) -> list:
    """The builder's rules, checked on finished nodes: the problems found."""
    nodes = np.asarray(nodes, dtype=float)
    widths = np.diff(nodes)
    problems = []
    if np.any(widths <= 0):
        problems.append("nodes are not strictly increasing")
        return problems
    steps = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
    if steps.size and steps.max() > ratio * (1 + 1e-9):
        problems.append(f"a neighbour ratio is {steps.max():.4f} > {ratio}")
    if widths.max() > coarse * (1 + 1e-9):
        problems.append(f"a cell is {widths.max():.4g} > coarse {coarse:.4g}")
    tol = 1e-9 * (nodes[-1] - nodes[0])
    for layer in layers:
        layer = layer if isinstance(layer, Layer) else Layer(*layer)
        lo = min(max(layer.lo, nodes[0]), nodes[-1])
        hi = min(max(layer.hi, nodes[0]), nodes[-1])
        inside = nodes[(nodes >= lo - tol) & (nodes <= hi + tol)]
        if hi - lo < coarse and inside.size - 1 < min_cells:
            problems.append(f"layer {lo:.4g}..{hi:.4g} has {inside.size - 1} "
                            f"cells, fewer than {min_cells}")
        if layer.spacing is not None:
            local = np.diff(inside)
            if local.size and local.max() > layer.spacing * (1 + 1e-9):
                problems.append(f"layer {lo:.4g}..{hi:.4g} has a cell of "
                                f"{local.max():.4g} > {layer.spacing:.4g}")
        for edge in (lo, hi):
            if np.min(np.abs(nodes - edge)) > tol:
                problems.append(f"layer edge {edge:.6g} is not a node")
    return problems


def build_grid(extent, coarse: float, layers=((), (), ())) -> dict:
    """Nodes and simulate's ``spacings`` for a box of ``extent`` (x, y, z)."""
    if len(tuple(extent)) != 3 or len(tuple(layers)) != 3:
        raise GridError("extent and layers are one entry per axis (x, y, z)")
    nodes = tuple(graded_axis(float(length), coarse, axis_layers)
                  for length, axis_layers in zip(extent, layers))
    return {"nodes": nodes,
            "spacings": tuple(np.diff(n) for n in nodes),
            "shape": tuple(n.size for n in nodes)}


def _dual_bounds(nodes):
    """Each node's dual cell: halfway to its neighbours, clipped at the ends."""
    mid = 0.5 * (nodes[1:] + nodes[:-1])
    return np.concatenate([[nodes[0]], mid]), np.concatenate([mid, [nodes[-1]]])


def average_eps(nodes, index_of, eps_of, *, samples: int = 3,
                metals=()) -> tuple:
    """Sub-cell averaged permittivity on the grid's nodes.

    ``index_of(points)`` maps an (m, 3) array of positions to region indices;
    ``eps_of`` is a sequence giving each index's relative permittivity
    (metals may use any value: their nodes are reported separately).
    Each node's dual cell is sampled at ``samples``^3 midpoints; eps is the
    volume average over the dielectric samples, and a node is a metal node
    when at least half of its samples are that metal. Returns (eps, masks)
    with one boolean mask per index in ``metals``.
    """
    nodes = [np.asarray(n, dtype=float) for n in nodes]
    eps_of = np.asarray(eps_of, dtype=float)
    shape = tuple(n.size for n in nodes)
    frac = (np.arange(samples) + 0.5) / samples
    sub = []
    for n in nodes:
        lo, hi = _dual_bounds(n)
        sub.append(lo[:, None] + (hi - lo)[:, None] * frac[None, :])
    eps = np.empty(shape)
    masks = {m: np.zeros(shape, dtype=bool) for m in metals}
    sy, sz = sub[1].reshape(-1), sub[2].reshape(-1)
    for i in range(shape[0]):            # one x slab at a time bounds memory
        px, py, pz = np.meshgrid(sub[0][i], sy, sz, indexing="ij")
        index = np.asarray(index_of(np.stack([px.ravel(), py.ravel(),
                                              pz.ravel()], axis=1)))
        index = index.reshape(samples, shape[1], samples, shape[2], samples)
        index = index.transpose(1, 3, 0, 2, 4).reshape(shape[1], shape[2], -1)
        is_metal = np.isin(index, list(metals))
        for m in metals:
            masks[m][i] = (index == m).mean(axis=-1) >= 0.5
        values = np.where(is_metal, 0.0, eps_of[index])
        count = (~is_metal).sum(axis=-1)
        # an all-metal cell has no dielectric part; its eps is never read
        # through the metal term, so 1 keeps the Courant check honest
        eps[i] = np.where(count > 0,
                          values.sum(axis=-1) / np.maximum(count, 1), 1.0)
    return eps, masks
