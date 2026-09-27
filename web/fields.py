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


def encode_field16(field: np.ndarray, grid: dict) -> bytes:
    """Encode a vector field as little-endian ``float16``, C order, NaN kept.

    This is what the volume plot downloads. Two things it must not do:

    - **Lose the NaN.** The solver marks everything outside the object NaN, and
      the browser renders NaN as fully transparent -- that is how the object's
      shape appears at all. A NaN written as some finite number would fill the
      outside with coloured fog. IEEE half precision has NaN, so ``astype``
      carries it through; ``nan`` is written with the same payload, so no
      comparison downstream can be confused by payload bits.
    - **Re-order the samples.** The array is already ``(nz, ny, nx, 3)``, which
      is the layout the texture wants; only the dtype changes. Reshaping here
      would risk the x/z swap this module has been bitten by before.

    Half precision is a product decision, not a shortcut: the wire size is
    halved and the solver has already normalised every mode to ``max |E| = 1``,
    so ~3 significant digits resolve more finely than the colour map that will
    read them. The exact values stay available in the ``.vtu`` download.
    """
    if field.shape != shape_of(grid) + (3,):
        raise FieldError(
            f"field shape {field.shape} does not match the grid {shape_of(grid)}")
    half = field.astype("<f2", copy=True)
    # np.nan is not guaranteed to survive a cast with a canonical payload; write
    # the canonical quiet NaN explicitly so the bytes are predictable.
    half[np.isnan(field)] = np.float16("nan")
    return half.tobytes(order="C")


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


__all__ = [
    "FieldError", "load_modes_json", "shape_of", "n_samples", "decode_mode",
    "encode_field16", "field_is_readable",
]
