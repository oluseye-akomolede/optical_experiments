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
            "axis": i,                       # index into the (nz, ny, nx) shape
            "count": counts[axis],
            "origin_m": origin[i],
            "spacing_m": spacing[i],
            "label": f"{axis} = {{v}} m",
        }
        for i, axis in enumerate(("z", "y", "x"))
    ]


def _index_for(grid: dict, axis: str, position_m: float) -> int:
    """Nearest array index along one named axis for a physical position."""
    nz, ny, nx = shape_of(grid)
    counts = {"x": nx, "y": ny, "z": nz}
    i = ("z", "y", "x").index(axis)
    origin = float(grid["origin"][i])
    spacing = float(grid["spacing"][i])
    if spacing == 0:
        return 0
    idx = int(round((position_m - origin) / spacing))
    return max(0, min(counts[axis] - 1, idx))


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

    u_m = float(grid["origin"][("z", "y", "x").index(u_axis)])
    u_d = float(grid["spacing"][("z", "y", "x").index(u_axis)])
    v_m = float(grid["origin"][("z", "y", "x").index(v_axis)])
    v_d = float(grid["spacing"][("z", "y", "x").index(v_axis)])

    mag = np.linalg.norm(plane, axis=-1)

    def _round(values):
        return [None if not math.isfinite(v) else round(float(v), 6) for v in values]

    return {
        "axis": axis,
        "plane_index": idx,
        "plane_position_m": round(float(grid["origin"][("z", "y", "x").index(axis)])
                                  + idx * float(grid["spacing"][("z", "y", "x").index(axis)]), 9),
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
    """
    nz, ny, nx = shape_of(grid)
    step = max(1, int(round(max(nz, ny, nx) / max(longest, 1))))
    blunt = field[::step, ::step, ::step]
    return {
        "stride": step,
        "shape": list(blunt.shape[:3]),
        "mag": magnitude(blunt).ravel().tolist(),
        "vec": blunt.reshape(-1, 3).ravel().tolist(),
    }


def ui_scale(grid: dict) -> dict:
    """Metres-per-unit for display plus a sensible unit label.

    The owner uploads in mm/cm/m/in; the viewer shows the numbers back in the
    unit they chose, which is what they will compare against their CAD model.
    """
    origin = [float(v) for v in grid["origin"]]
    spacing = [float(v) for v in grid["spacing"]]
    lo = origin
    hi = [origin[i] + spacing[i] * (shape_of(grid)[2 - i] - 1) for i in range(3)]
    return {
        "origin_m": lo,
        "corner_m": hi,
        "extent_m": [hi[i] - lo[i] for i in range(3)],
    }


__all__ = [
    "FieldError", "load_modes_json", "shape_of", "n_samples", "decode_mode",
    "magnitude", "slice_axes", "slice_payload", "downsample", "ui_scale",
]
