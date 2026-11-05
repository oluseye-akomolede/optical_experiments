"""Print the roundness figure of every tests/test_roundness.py fixture (task 2.6).

    python scripts/roundness_check.py

Each line is the worst loop's circumradius R, inradius r and R/r about the
detected axis, and whether the 2 % limit (mse_geometry.ROUNDNESS_TOLERANCE)
passes it. These are the figures tasks.md 2.6 quotes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("tests", "dockerfiles/maxwell_solver", "web")]

import mse_geometry as G  # noqa: E402
import test_roundness as T  # noqa: E402

v, f = T._prism(n=256, height=4.0)
rot = T._rotation([1, 2, 0.5], 0.7)
CASES = {
    "cylinder n=256 R=2": T._prism(a=2.0, n=256),
    "faceted n=24": T._prism(n=24),
    "faceted n=16": T._prism(n=16),
    "faceted n=12": T._prism(n=12),
    "hex R=239.4": T._prism(a=239.4, n=6, height=1500.0),
    "ellipse a/b=1.03": T._prism(a=1.03, b=1.0, n=256),
    "ellipse a/b=1.015": T._prism(a=1.015, b=1.0, n=256),
    "squat L=2R": T._prism(n=256, height=2.0),
    "tilted+offset": (v @ rot.T + [5, -3, 2], f),
}

for name, (verts, faces) in CASES.items():
    m = G.measure_roundness(verts, faces)
    verdict = "PASS" if m["ratio"] <= 1 + G.ROUNDNESS_TOLERANCE else "REFUSE"
    print(f"{name:20s} R={m['circumradius']:.6g} r={m['inradius']:.6g} "
          f"ratio={m['ratio']:.6f} {verdict}")

m = G.measure_roundness(*CASES["tilted+offset"])
z = G.roundness_about(*CASES["tilted+offset"], m["axis_point"], [0, 0, 1])
print(f"tilted+offset about the coordinate z axis: ratio={z['ratio']:.4f}")
print(G.roundness_refusal(G.measure_roundness(*CASES["hex R=239.4"]), "nm"))
