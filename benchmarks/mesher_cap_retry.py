"""What does the cap's coarsening retry do to the mesh the user gets?

``tetrahedralize`` has a second response to the element cap besides refusing:
when the mesh is OVER the cap but NOT too coarse (``mse_meshing.py``, the final
``else`` of the attempt loop), it multiplies ``maxvolume`` by
``(n_tets / max_tets) * 1.15`` and meshes again. ``maxvolume`` is the only
lever, so the delivered edge grows -- below the resolution rule the user asked
for.

WHICH RULE THE COARSENED MESH IS THEN JUDGED BY has changed since the table in
``openspec/changes/mesher-region-guard/proposal.md`` was recorded: the guard is
now PER REGION (``MAX_EDGE_OVERSHOOT`` times each region's OWN required edge),
where it used to be the whole mesh's median against the WALL's. The recorded
table is therefore a pre-fix measurement, and this script prints BOTH readings
per case -- the global one it has always printed, and the per-region one the
mesher now applies -- so the before/after is visible on the same fixtures.

So for an upload the cap retry coerces, three questions:

  1. How much coarser than the rule is the delivered mesh?
  2. Does the guard catch it, and if so what does the caller get -- a refusal
     or a coarse mesh?
  3. Is the user told anywhere that the mesh was coarsened?  (Answered by code
     reading in ``openspec/changes/mesher-region-guard/proposal.md``, item (d);
     the only thing this script adds is whether the LOG says it, which is what
     a user can reach with ``kubectl logs``.)

FIXTURE CAP: the caps below are DELIBERATELY LOWER than the product's
``CFG.max_tets()``. The retry is only reachable for an upload whose
unconstrained count exceeds the cap, and a tier-1 fixture cannot get past
600,000 elements; a lowered cap exercises the same code path on a small mesh.
Every such cap is printed and labelled. The numbers to transfer to the product
are the RATIOS (delivered vs rule, and how far over the cap), not the caps.

    (ulimit -v 4194304; OMP_NUM_THREADS=1 nice -n 19 .venv-mse/bin/python \
        benchmarks/mesher_cap_retry.py)
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "dockerfiles" / "maxwell_solver", _ROOT / "web"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mse_meshing as MESHSOLVER  # noqa: E402
import mse_common as COMMON  # noqa: E402
from benchmarks import estimate_vs_built as BENCH  # noqa: E402
from benchmarks.mesher_region_guard import prepare  # noqa: E402

#: (family label EXACTLY as in ``BENCH.FAMILIES``, fixture caps to try). Two
#: groups: HARD caps, several times under the unconstrained count, where the
#: retry has a lot of work to do; and MILD caps just under it, which is the
#: regime the PRODUCT is in -- an accepted upload whose real count overshoots
#: 600,000 by a few percent, because the estimate under-reads by at most ~5%
#: at the accept threshold. The mechanism (tets scale as 1/maxvolume) is
#: scale-free, so the mild group is what transfers to the product's cap.
CASES = [
    ("one-part sphere r3", (5_000, 2_500, 1_200)),
    ("2-region core 1.5mm n=2", (20_000, 22_500, 23_500, 15_000, 8_000, 4_000)),
    ("2-region dense core n=3.5", (120_000, 128_000, 60_000, 20_000)),
    ("3-region shell+core n=2/3", (380_000, 400_000, 100_000, 50_000)),
]
_FAMILY = {label: (ps, n) for label, ps, n in BENCH.FAMILIES}

_MAXVOL = re.compile(r"maxvolume ([0-9.eE+-]+)")
_OVER = re.compile(r"(\d[\d,]*) tets is over the (\d[\d,]*) cap")


def _family(label: str):
    return _FAMILY[label]


def run_case(label: str, caps) -> None:
    parts_spec, declared_n = _family(label)
    n_modes = max(1, int(round(declared_n * BENCH.FAMILY_N_SCALE)))
    c = prepare(label, parts_spec, n_modes)
    unit_m = c.unit_m
    h_wall_units = c.h_wall_units
    allowed_m = c.allowed_m
    print(f"\n{label}  (n_modes={n_modes}, product cap "
          f"{BENCH.MAX_TETS:,})", flush=True)
    print(f"  rule: h_wall {h_wall_units:.4f} mm -> allowed (guard) "
          f"{allowed_m / unit_m:.4f} mm; h_min {c.h_min_units:.4f} mm")

    for cap in caps:
        print(f"  --- FIXTURE CAP {cap:,} (NOT the product's "
              f"{BENCH.MAX_TETS:,}: lowered so the retry is reachable in "
              f"tier 1)", flush=True)
        buf = io.StringIO()
        outcome, err = "accepted", ""
        try:
            with contextlib.redirect_stdout(buf):
                points, tets, regions = MESHSOLVER.tetrahedralize(
                    c.surface, "mm", n_modes=n_modes, parts=c.parts,
                    materials=c.materials, max_tets=cap, verbose=True,
                )
        except Exception as exc:  # noqa: BLE001 - a refusal is a row
            outcome, err = type(exc).__name__, str(exc)
            points = tets = regions = None
        log = buf.getvalue()
        # The mesher's own words (``mse_meshing.py``): the refine arm prints
        # "too coarse; refining with maxvolume V", the cap arm prints
        # "N tets is over the M cap; retrying with maxvolume V". Count the
        # cap arm by ITS text -- counting "raising maxvolume" (a string the
        # mesher never prints) reported 0 coarsens on a run whose count
        # visibly fell 24,376 -> 8,992.
        coarsens = log.count("retrying with maxvolume")
        refines = log.count("too coarse; refining")
        maxvols = [float(v) for v in _MAXVOL.findall(log)]
        overs = _OVER.findall(log)
        # ``maxvolume`` in the log is in TetGen's CANONICAL frame (the mesher
        # centres the object and divides by its longest side, so a 6 mm box
        # divides volumes by 6**3 = 216). It is NOT comparable to a rule value
        # in mm**3, so it is reported raw, as a trail, and only for the count.
        print(f"    retries: coarsen {coarsens}, refine {refines}; "
              f"maxvolume trail {['%.4g' % v for v in maxvols]} "
              f"(canonical frame)")
        if overs:
            n1, cap1 = (int(v.replace(",", "")) for v in overs[0])
            print(f"    the log did say: {n1:,} tets over the {cap1:,} cap "
                  f"(bite {n1 / cap1:.2f}x)")
        if points is None:
            print(f"    OUTCOME {outcome}: {err.splitlines()[0][:110]}")
            continue

        achieved_m = MESHSOLVER.achieved_edge_length(points, tets)
        verdict = "PASS" if achieved_m <= allowed_m else "FAIL"
        print(f"    n={len(tets):,}  OUTCOME accepted  "
              f"GUARD(global, the pre-fix reading) {verdict} "
              f"(whole-mesh achieved {achieved_m / unit_m:.4f} mm "
              f"= {achieved_m / allowed_m:.3f}x the WALL's allowance)")
        if err:
            print(f"    (warning: {err.splitlines()[0][:90]})")
        reg = (np.asarray(regions) if regions is not None
               else np.ones(len(tets), dtype=int))
        by_region = c.by_region or {int(r): 1.0 for r in np.unique(reg)}
        n_max = max(float(v) for v in by_region.values())
        worst = 0.0
        for rid in sorted(by_region):
            n_r = float(by_region[rid])
            mask = reg == rid
            if not mask.any():
                continue
            h_r_units = h_wall_units * c.n_wall / n_r
            ach = MESHSOLVER.achieved_edge_length(points, tets[mask])
            ratio = (ach / unit_m) / h_r_units
            worst = max(worst, ratio)
            print(f"      region {rid:g} ({c.part_of_region.get(rid, '?')}): "
                  f"h_r {h_r_units:.4f} -> ach {ach / unit_m:.4f} mm  "
                  f"ach/h_r {ratio:6.3f}")
        print(f"    WORST region delivered at {worst:.3f}x its own required "
              f"edge; the fixed guard allows {COMMON.MAX_EDGE_OVERSHOOT:g}x "
              f"each region's OWN, so this row was ACCEPTED, while the pre-fix "
              f"WALL-based allowance amounted to "
              f"{COMMON.MAX_EDGE_OVERSHOOT * n_max / c.n_wall:.2f}x its own")


def main() -> int:
    print("THE CAP-COARSENING RETRY: what the user gets")
    print(f"product cap = {BENCH.MAX_TETS:,} "
          f"(bench MAX_TETS; lowered fixture caps are labelled below)")
    only = sys.argv[1:] or None
    for label, caps in CASES:
        if only and label not in only:
            continue
        try:
            run_case(label, caps)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
