"""Turning a solver field file into canvas data for the browser.

The solver writes ``mode_XX.bin`` as ``float32``, C order, shaped
``[z][y][x][3]`` (the three components are Ex, Ey, Ez in metres^-1, each mode
normalised to ``max |E| = 1``, NaN outside the object). ``modes.json`` carries
the grid geometry that indexes it.

The viewer needs two things fast: a scalar ``|E|`` volume to colormap, and the
three components on a slice to draw arrows. Doing this here in numpy is far
cheaper than shipping 9 MB per mode to a phone, so the browser gets a small
gzipped payload instead.

The axis order is the trap in this file. ``grid.nx/ny/nz`` are counts along the
**x/y/z axes**, but the array is indexed ``[z][y][x]``, so the shape is
``(nz, ny, nx, 3)``. Every function below states which convention it is in.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np


class FieldError(Exception):
    """The field file is missing or does not match the manifest."""


def load_modes_json(data: bytes) -> dict:
    """Parse ``modes.json`` and sanity-check the grid block."""
    try:
        manifest = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise FieldError("the solver's mode table could not be read") from exc

    grid = manifest.get("grid") or {}
    for key in ("nx", "ny", "nz", "origin", "spacing"):
        if key not in grid:
            raise FieldError(f"the solver's mode table is missing grid.{key}")
    if len(grid["origin"]) != 3 or len(grid["spacing"]) != 3:
        raise FieldError("the solver's grid origin and spacing must have 3 values")
    return manifest


def shape_of(grid: dict) -> tuple[int, int, int]:
    """Array shape of one mode: ``(nz, ny, nx)`` -- note the reversal."""
    return int(grid["nz"]), int(grid["ny"]), int(grid["nx"])


def n_samples(grid: dict) -> int:
    nz, ny, nx = shape_of(grid)
    return nz * ny * nx


def decode_mode(data: bytes, grid: dict) -> np.ndarray:
    """Decode one ``mode_XX.bin`` into a ``(nz, ny, nx, 3)`` float32 array."""
    expected = n_samples(grid) * 3
    if len(data) != expected * 4:
        raise FieldError(
            "the field file does not match the grid in the mode table "
            f"({len(data)} bytes, expected {expected * 4})"
        )
    # '<f4' explicitly: the solver writes little-endian in practice, but naming
    # it keeps the reader honest if that ever changes.
    flat = np.frombuffer(data, dtype="<f4")
    field = flat.reshape((-1, 3)).copy()
    return field.reshape(shape_of(grid) + (3,))


def field_is_readable(grid: dict, size_bytes: int | None) -> bool:
    """Could ``decode_mode`` read a blob of this length, without fetching it?

    The index route needs this answer for every mode, and reading twenty
    multi-megabyte blobs to ask would be most of the cost the payload split
    exists to avoid. The only thing that makes a field unreadable is its length,
    so the check is arithmetic against a size the caller got cheaply (a HEAD, or
    a stat). ``None`` means the file is not there at all.
    """
    if size_bytes is None:
        return False
    return size_bytes == n_samples(grid) * 3 * 4


def magnitude(field: np.ndarray) -> np.ndarray:
    """``|E|`` as float32, NaN preserved where the solver marked outside."""
    return np.linalg.norm(field, axis=-1).astype(np.float32)


def _array_axis(axis: str) -> int:
    """Position of a named axis in the field's array shape ``(nz, ny, nx)``.

    The two orderings in this file are different and both are needed, so they
    are named once here rather than derived at each call site -- deriving them
    twice is how the slices came out at the wrong physical coordinate.

    The solver's ``sample_grid`` builds ``shape`` and ``origin``/``spacing`` in
    **(x, y, z)** order, while the field array it fills is ``(nz, ny, nx, 3)``
    -- the array axes are the *reverse* of the metadata order. So "z" is array
    axis 0 but metadata index 2, and conflating the two silently reads z's
    count against x's origin.
    """
    return ("z", "y", "x").index(axis)


def _meta_axis(axis: str) -> int:
    """Position of a named axis in ``grid["origin"]``/``grid["spacing"]``.

    The solver writes those in **(x, y, z)** order (see ``sample_grid``:
    ``xs = linspace(lo[0], hi[0], nx)``, and ``lo`` is
    ``[x, y, z]``). This is the other half of :func:`_array_axis`.
    """
    return ("x", "y", "z").index(axis)


def slice_axes(grid: dict) -> list[dict]:
    """Descriptions of the three orthogonal slice families.

    Each entry says which array axis the family varies along, how many samples
    it has, its physical coordinate in **metres**, and the labels for a slider.
    """
    nz, ny, nx = shape_of(grid)
    origin = [float(v) for v in grid["origin"]]
    spacing = [float(v) for v in grid["spacing"]]
    counts = {"x": nx, "y": ny, "z": nz}
    return [
        {
            "name": axis,
            "axis": _array_axis(axis),       # index into the (nz, ny, nx) shape
            "count": counts[axis],
            "origin_m": origin[_meta_axis(axis)],
            "spacing_m": spacing[_meta_axis(axis)],
            "label": f"{axis} = {{v}} m",
        }
        for axis in ("z", "y", "x")
    ]


def _index_for(grid: dict, axis: str, position_m: float) -> int:
    """Nearest array index along one named axis for a physical position.

    ``position_m`` is a physical coordinate and ``origin``/``spacing`` are the
    grid metadata, so both are read in the metadata's **(x, y, z)** order. Only
    ``counts`` is an array fact -- it is keyed by name, which is why this
    function looked correct at a glance.
    """
    nz, ny, nx = shape_of(grid)
    counts = {"x": nx, "y": ny, "z": nz}
    origin = float(grid["origin"][_meta_axis(axis)])
    spacing = float(grid["spacing"][_meta_axis(axis)])
    if spacing == 0:
        return 0
    idx = int(round((position_m - origin) / spacing))
    return max(0, min(counts[axis] - 1, idx))


def _round(values):
    """JSON-safe list: non-finite to ``None``, everything else rounded.

    The one place that decides how a float reaches the browser. The solver marks
    outside-the-object as ``NaN``, and ``NaN``/``Infinity`` are not JSON -- a
    single bare token makes ``JSON.parse`` reject the whole payload, which is
    how the 3-D view broke for the owner. ``None`` becomes ``null``, which the
    canvas reads as "nothing here".
    """
    return [None if not math.isfinite(v) else round(float(v), 6) for v in values]


def slice_payload(field: np.ndarray, grid: dict, axis: str,
                  position_m: float, stride: int = 1) -> dict:
    """One slice: ``|E|`` for the colour map plus the vectors for arrows.

    ``axis`` is the *normal* of the plane ('x' gives a y-z plane). The result is
    oriented so that ``u`` runs along the first in-plane axis and ``v`` along the
    second, which is what the canvas treats as (horizontal, vertical).
    """
    nz, ny, nx = shape_of(grid)
    idx = _index_for(grid, axis, position_m)

    # Indexing away the normal axis leaves the other two in their original
    # relative order, which out of (z, y, x) is always (v, u): picking one axis
    # out of a three-axis sequence cannot reorder the two that remain.
    if axis == "x":
        # array (nz, ny) -> v=z (vertical), u=y (horizontal)
        plane, u_axis, v_axis, u_n, v_n = field[:, :, idx, :], "y", "z", ny, nz
    elif axis == "y":
        # array (nz, nx) -> v=z, u=x
        plane, u_axis, v_axis, u_n, v_n = field[:, idx, :, :], "x", "z", nx, nz
    else:
        # array (ny, nx) -> v=y, u=x
        plane, u_axis, v_axis, u_n, v_n = field[idx, :, :, :], "x", "y", nx, ny

    u_m = float(grid["origin"][_meta_axis(u_axis)])
    u_d = float(grid["spacing"][_meta_axis(u_axis)])
    v_m = float(grid["origin"][_meta_axis(v_axis)])
    v_d = float(grid["spacing"][_meta_axis(v_axis)])

    mag = np.linalg.norm(plane, axis=-1)

    normal_m = float(grid["origin"][_meta_axis(axis)])
    normal_d = float(grid["spacing"][_meta_axis(axis)])

    return {
        "axis": axis,
        "plane_index": idx,
        "plane_position_m": round(normal_m + idx * normal_d, 9),
        "u": {"name": u_axis, "count": int(u_n), "origin_m": u_m, "spacing_m": u_d},
        "v": {"name": v_axis, "count": int(v_n), "origin_m": v_m, "spacing_m": v_d},
        "mag": {"data": _round(mag.ravel()), "min": _finite_min(mag), "max": _finite_max(mag)},
        "vec": {"data": _round(plane.reshape(-1, 3).ravel())},
    }


def _finite_min(a: np.ndarray) -> float:
    good = a[np.isfinite(a)]
    return round(float(good.min()), 6) if good.size else 0.0


def _finite_max(a: np.ndarray) -> float:
    good = a[np.isfinite(a)]
    return round(float(good.max()), 6) if good.size else 0.0


def downsample(field: np.ndarray, grid: dict, longest: int) -> dict:
    """Shrink a full mode volume to a browseable one for the 3-D point cloud.

    The full ``[z][y][x]`` grid loses nothing physical when sampled at a
    coarser stride for display; the arrow slice view keeps full resolution.

    Non-finite samples become ``None``, via the same :func:`_round` the slice
    path uses. This is not tidiness: the solver marks everything outside the
    object as ``NaN``, so a real mode is *mostly* NaN, and ``NaN`` is not JSON.
    ``json.dumps`` writes the bare token ``NaN`` by default, and the browser's
    ``JSON.parse`` rejects it -- one bad token and the viewer loses the entire
    payload, which is exactly how the 3-D view broke for the owner on
    2026-09-27. ``None`` serialises as ``null``, which is what "no value here"
    means to the canvas.
    """
    nz, ny, nx = shape_of(grid)
    step = max(1, int(round(max(nz, ny, nx) / max(longest, 1))))
    blunt = field[::step, ::step, ::step]
    return {
        "stride": step,
        "shape": list(blunt.shape[:3]),
        "mag": _round(magnitude(blunt).ravel()),
        "vec": _round(blunt.reshape(-1, 3).ravel()),
    }


__all__ = [
    "FieldError", "load_modes_json", "shape_of", "n_samples", "decode_mode",
    "magnitude", "slice_axes", "slice_payload", "downsample",
]
