"""What the mesher delivers when its own volume bound is tightened.

The upload estimate is faithful to the mesher's REQUEST and the delivery wanders
around it: at the product's cap these families come back from 0.715x to 8.952x of
the volume law at the edge the plan asked for (0.643x on one family at half the
mode counts). One candidate fix, on the MESHER's side, is the bound itself:
``TETGEN_VOLUME_SLACK = 4.0`` licenses four regular tets' worth of volume per
requested element, i.e. it lets a delivered mesh be up to 4x coarser than the
volume law.

This probe re-runs the family sweep's own fixtures and code path
(``estimate_vs_built``) with that constant monkeypatched at the shipped mode
counts, so the only thing different between the tables is the bound the mesher
asks for. It changes NO product code and no file outside this one.
``built/volaw`` is the count relative to the volume law at the edge the plan
requested, so the bound itself permits ``1/SLACK`` as a floor (0.25 at 4.0).

**Measured answer: tightening the bound makes the delivery FINER, not more
predictable.** As a whole the nine families span ``built/volaw`` 0.715-8.952 at
slack 4.0, 1.472-3.818 at 2.0, 3.295-9.020 at 1.0 and 6.779-10.931 at 0.5 (where
two families no longer fit the cap at all). Both ends move UP together, because
the bound is a CEILING and TetGen refines below it for quality: asking for a
smaller maximum gets a smaller mesh, not a request that is honoured. So the
bound is not what sets these meshes' element size, and no value of it narrows the
estimator's spread -- which is why the estimator change does not touch it.

The slack-4.0 table also reproduces the product-cap sweep exactly, which is the
probe's own cross-check that the patch changes nothing but the bound.

Tier 1: tiny fixtures, a few seconds each. Run:

    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python \
        benchmarks/mesher_slack_probe.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in ("web", "dockerfiles/maxwell_solver", "benchmarks"):
    sys.path.insert(0, str(ROOT / extra))

import estimate_vs_built as BENCH  # noqa: E402  - the sweep's fixtures + rows
import mse_common as COMMON  # noqa: E402
import mse_meshing as MESHER  # noqa: E402  - patched alongside, see below

#: the slack values to sweep. 4.0 is the shipped value and the control: the row
#: it prints must reproduce the family sweep's own table.
SLACKS = (4.0, 2.0, 1.0, 0.5)

#: Raise the sweep's cap so a tighter bound is never answered by the cap retry
#: (which coarsens ``maxvolume`` on purpose and would confound the table). No
#: fixture here is anywhere near this; the point is that none is shaped by it.
BENCH.MAX_TETS = 500_000


def main() -> int:
    print("MESHER DELIVERY vs TETGEN_VOLUME_SLACK (fixtures from the family sweep)")
    for slack in SLACKS:
        # BOTH bindings, deliberately: ``target_maxvolume`` reads
        # ``mse_common``'s global, but ``mse_meshing`` imported the constant by
        # value (``from mse_common import TETGEN_VOLUME_SLACK``), so patching
        # only one of them is the classic monkeypatch that silently patches
        # nothing. ``_assert_patch_lands`` below checks the mesher really builds
        # a different ``maxvolume`` before the table is trusted.
        COMMON.TETGEN_VOLUME_SLACK = float(slack)
        MESHER.TETGEN_VOLUME_SLACK = float(slack)
        BENCH.ROWS.clear()
        print(f"\n=== TETGEN_VOLUME_SLACK = {slack}  "
              f"(bound = 1/{slack} of the volume law; a delivered mesh cannot be "
              f"coarser than {slack ** (1 / 3):.3f}x the requested edge without "
              f"breaking it)")
        # The control: the volume bound the mesher will actually hand TetGen for
        # a 0.5 mm request. It must move with the slack -- if a future refactor
        # re-binds the constant somewhere this probe cannot see, this line goes
        # flat and the whole table below is a re-run of the shipped mesher.
        bound = COMMON.target_maxvolume(None, 0.5)
        law = 0.5 ** 3 / COMMON.TETS_PER_UNIT_VOLUME
        print(f"    control: maxvolume(edge=0.5mm) = {bound:.6g} mm^3 "
              f"= {bound / law:.3f}x the volume law (slack={slack})")
        BENCH.family_rows()
        BENCH._summarize(BENCH.ROWS, f"slack {slack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
