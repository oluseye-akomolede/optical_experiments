#!/usr/bin/env python3
"""How much a CPML reflects, by angle, measured on the solver's own layer.

For a plane wave with transverse wavenumber K along a face, the 3-D Yee update
with a CPML along the face normal x reduces EXACTLY to a 1-D line: Ez at the
nodes, Hy at the half nodes (the normal derivative, CPML-corrected) and Hx at
the nodes (the transverse derivative, a factor iK, never corrected). The line
here takes its (b, c) from ``fdtd._pml_profiles`` and corrects the derivatives
the way ``simulate`` does, so what it measures is the layer the solver builds,
not a retyped one. One line is a few hundred cells, so a whole sweep is a
tier-1 run (seconds, well under 100 MB).

The measurement: a pulse from a soft source crosses a gap, hits the layer and
comes back past a probe. The same run on a line long enough that nothing comes
back inside the window is the reference; the reflection is
max|test - reference| / max|reference| at the probe. A bare wall (``--pml-cells
0``) must read ~1, which is the instrument's positive control.

K is set from the angle at the pulse's carrier, so at a large angle the
pulse's low-frequency tail is evanescent along x, as it is at a real face.

    python3 scripts/fdtd_cpml_reflection.py                 # 10 and 20 cells
    python3 scripts/fdtd_cpml_reflection.py --pml-cells 0 10 --angles 0 60
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402


def run_line(n, pml, dx, dt, K, src, probe, steps, source):
    """The 1-D reduction, complex fields for the transverse phase e^{iKy}."""
    primal = np.full(n - 1, dx)
    b_e, c_e = (np.asarray(v, dtype=float)[1:-1]
                for v in fdtd._pml_profiles(primal, pml, dt, np, "node"))
    b_h, c_h = (np.asarray(v, dtype=float)
                for v in fdtd._pml_profiles(primal, pml, dt, np, "cell"))
    ez = np.zeros(n, complex)
    hy = np.zeros(n - 1, complex)
    hx = np.zeros(n, complex)
    psi_e = np.zeros(n - 2, complex)
    psi_h = np.zeros(n - 1, complex)
    out = np.zeros(steps, complex)
    for step in range(steps):
        raw = (ez[1:] - ez[:-1]) / dx
        psi_h = b_h * psi_h + c_h * raw
        hy += dt / fdtd.MU0 * (raw + psi_h)
        hx -= dt / fdtd.MU0 * 1j * K * ez
        raw = (hy[1:] - hy[:-1]) / dx
        psi_e = b_e * psi_e + c_e * raw
        ez[1:-1] += dt / fdtd.EPS0 * ((raw + psi_e) - 1j * K * hx[1:-1])
        ez[src] += dt * source.value(step * dt)
        out[step] = ez[probe]
    return out


def reflection(pml: int, angle_deg: float, cells_per_wavelength: float = 20.0,
               gap: int = 30, probe_depth: int = 10) -> float:
    """Peak reflected amplitude over peak incident, at the probe."""
    dx = 1e-3
    wavelength = cells_per_wavelength * dx
    carrier = fdtd.C0 / wavelength
    bandwidth = 0.8 * carrier
    dt = fdtd.courant_dt(dx, dx, dx)
    K = 2.0 * math.pi / wavelength * math.sin(math.radians(angle_deg))
    source = fdtd.Source((0, 0, 0), frequency=carrier, profile="pulse",
                         bandwidth=bandwidth)
    # wall | layer | gap | layer | wall: _pml_profiles puts a layer at both
    # ends, so the source sits in the interior, 3 cells clear of the far one
    n = pml + gap + pml + 1
    probe, src = pml + probe_depth, n - 1 - pml - 3
    slow = max(math.cos(math.radians(angle_deg)), 0.2)
    steps = int(1.4 * 2 * n / slow * dx / fdtd.C0 / dt
                + 10.0 / (math.pi * bandwidth) / dt)
    test = run_line(n, pml, dx, dt, K, src, probe, steps, source)
    extra = steps + 50            # the echo cannot get back inside the window
    reference = run_line(n + extra, pml, dx, dt, K, src + extra,
                         probe + extra, steps, source)
    return float(np.abs(test - reference).max() / np.abs(reference).max())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pml-cells", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[0.0, 30.0, 45.0, 60.0, 75.0])
    parser.add_argument("--cells-per-wavelength", type=float, nargs="+",
                        default=[10.0, 20.0, 40.0])
    args = parser.parse_args(argv)
    print(f"CPML order {fdtd.PML_ORDER}, sigma_max from R = "
          f"{fdtd.PML_REFLECTION:g}, kappa 1, alpha 0")
    print(f"{'cells':>5} {'c/wl':>5} " + " ".join(f"{a:>8g}d" for a in args.angles))
    for pml in args.pml_cells:
        for cpw in args.cells_per_wavelength:
            row = [reflection(pml, a, cpw) for a in args.angles]
            print(f"{pml:5d} {cpw:5g} " + " ".join(f"{r:9.2e}" for r in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
