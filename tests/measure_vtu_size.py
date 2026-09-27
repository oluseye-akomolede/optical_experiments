"""Size of mode_NN.vtu at the production grid: ascii (old) vs binary (new).

This is the size half of the .vtu row's ask. The round-trip half -- that the
binary file reads back to the same values, NaN included -- is checked at tier 1
by the test suite on a tiny mesh (see test_maxwell_solver.py). This script is for
the *number*, which needs the production grid and is therefore a tier-2 run on
the Jetson, not something to do on the VM.

Why it reads a real job rather than synthesising a field: the number the owner
cares about is "how big is a mode's .vtu", and that depends on the real grid
(64x45x32 at the default n_index) and the real NaN fraction (16%, measured). A
synthetic field would answer a question nobody asked.

Usage, with a job's files already on disk:

    python tests/measure_vtu_size.py <job-dir>

where <job-dir> holds modes.json and mode_01.bin as the solver wrote them -- the
same bytes the web app serves. Every mode_*.bin found is measured.

`bench_box.py --write` produces exactly this directory for a box run, which is
the path the lead uses on the Jetson; no object-store fetch is needed.

It prints the ascii size the OLD writer produced, the binary size the NEW writer
produces, and confirms the binary file reads back to the same float32 values and
the same NaN pattern as the .bin.
"""

import json
import os
import sys

import numpy as np

# mse_fields lives beside the solver image's other modules, not on the path;
# this script is run from the repository root by convention but nothing should
# depend on that.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "dockerfiles", "maxwell_solver"))

import mse_fields as FLD  # noqa: E402


def ascii_size(field, shape, origin, spacing):
    """What the old `%.6g` ascii writer would have produced, in bytes.

    Recomputed rather than remembered, so the comparison is against *this*
    field and not against a number that was true for a different one. It is the
    same element sequence the old writer emitted, which is what makes the two
    sizes comparable at all.
    """
    nz, ny, nx = shape
    zs = origin[2] + np.arange(nz) * spacing[2]
    ys = origin[1] + np.arange(ny) * spacing[1]
    xs = origin[0] + np.arange(nx) * spacing[0]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    ex = field[:, :, :, 0].ravel()
    ey = field[:, :, :, 1].ravel()
    ez = field[:, :, :, 2].ravel()
    mag = np.linalg.norm(field, axis=-1).ravel()

    total = 0
    for row in coords:
        total += len(" ".join(f"{v:.6g}" for v in row)) + 1
    for a, b, c in zip(ex, ey, ez, strict=False):
        total += len(f"{a:.6g} {b:.6g} {c:.6g}") + 1
    for v in mag:
        total += len(f"{v:.6g}") + 1
    # the fixed XML scaffolding the old writer emitted, measured once
    total += 900
    return total


def main(job_dir):
    with open(os.path.join(job_dir, "modes.json")) as handle:
        manifest = json.load(handle)
    grid = manifest["grid"]
    shape = (grid["nz"], grid["ny"], grid["nx"])
    origin = grid["origin"]
    spacing = grid["spacing"]

    print(f"grid {grid['nx']}x{grid['ny']}x{grid['nz']} "
          f"({grid['nx'] * grid['ny'] * grid['nz']:,} samples)")

    for name in sorted(os.listdir(job_dir)):
        if not (name.startswith("mode_") and name.endswith(".bin")):
            continue
        field = np.fromfile(os.path.join(job_dir, name), dtype=np.float32)
        field = field.reshape(shape + (3,))

        import io
        buf = io.BytesIO()
        FLD.write_vtu(buf, field, shape, origin, spacing)
        binary = buf.getvalue()
        ascii_bytes = ascii_size(field, shape, origin, spacing)

        # Read the binary file back and compare to the .bin it came from.
        import tempfile
        import pyvista as pv
        with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as tmp:
            tmp.write(binary)
            path = tmp.name
        back = np.array(pv.read(path)["E"])
        os.unlink(path)
        src = field.reshape(-1, 3)
        nan_ok = (np.isnan(back).any(axis=1) == np.isnan(src).any(axis=1)).all()
        finite = np.isfinite(src).all(axis=1)
        max_diff = float(np.abs(back[finite] - src[finite]).max()) if finite.any() else 0.0

        print(f"\n{name}")
        print(f"  ascii  (old writer): {ascii_bytes / 1e6:8.2f} MB")
        print(f"  binary (new writer): {len(binary) / 1e6:8.2f} MB  "
              f"({ascii_bytes / len(binary):.1f}x smaller)")
        print(f"  reads back to the same float32: max abs diff {max_diff:.3g}")
        print(f"  NaN pattern preserved: {nan_ok}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
