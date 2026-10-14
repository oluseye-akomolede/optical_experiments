"""A validation matrix for per-region mesh grading, as a runnable bench mode.

WHY THIS EXISTS. The owner's question (2026-10-13): *"Is this tightening just
for one job? The meshing needs to work generally for any mesh supplied."* The
other bench's fixtures are one concentric two-region family of spheres; grading
can hold on that family and still be wrong for the next upload. This file is
the answer in a form that can be RUN, not prose: a case list laid out over the
axes an arbitrary upload actually varies, one case per invocation, each sized to
fit the tier-2 budget.

THE AXES, and which case covers each (``_coverage`` exposes this as data so a
test can fail when a table edit drops one):

* index contrast 1.5, 2, 3, 4             -- c1.5-thick .. c4-thick
* shell thickness, thin (the thickness     -- c2-thin, c3-thin, c4-thin
  clamp BITES) and thick (it does not)       vs the *-thick cases
* number of regions 2 and 3                -- *-thick vs three-region
* shape: non-spherical, non-concentric     -- box-sphere, offset-sphere
* reference: analytic and uniform-refined  -- concentric vs the shape cases

THE CRITERION, per case, is the owner's: the graded mesh's error is no worse
than the uniform mesh's, both solved on the product's own pipeline and scored
against the SAME reference. UNMEASURED is its own outcome (the uniform solve
raised, or the reference could not be built) and is never a pass -- the Orin's
n3 uniform reference OOMed once at 199,062 elements, which is exactly the
"could not ask" shape this rig treats as its signature defect.

THE REFERENCE, per case, is one of two things and the case says which:

* ``analytic`` for concentric two-index spheres -- the closed-form spectrum from
  ``sphere_dielectric``, the same reference the other bench scores against.
* ``uniform-refined`` where no closed form exists (a box, an off-centre
  inclusion, three distinct indices). The reference is the SAME geometry solved
  on a uniform mesh at ``h_min/K`` for K in ``REFINE_FACTORS``; the coarser
  level is scored against the finest, so the reference's OWN convergence is
  printed beside every error it is used to judge. A reference whose convergence
  is not shown cannot support a no-worse-than claim.

WHAT THIS FILE DOES NOT DO. It does not change the grading rule; it measures it.
And nothing here is a constant fitted to a case: the per-region edges come from
``mse_common.region_plan`` and the predicted counts from
``web.meshing.estimate_tets``, both production functions, and the estimator's
constant from ``web.config.CFG.est_tets_c()``, the value ``web/app.py`` routes
on.

SIZING, AND WHAT IT COSTS. Each case is sized so every mesh it builds fits tier
2: one solve at a time on the Orin, under 5 GB and 30 minutes. The binding cost
is the UNIFORM reference -- every region meshed at the finest region's edge --
and the measured law is steep: it costs ``contrast**3 * n_modes`` elements, and
nothing else moves it (the outer size cancels, because the target edge scales
with it; the shell thickness does not enter it at all, since ``h_min`` comes
from the densest region and not from the wall). So a contrast-4 sphere predicts
611,826 elements at the other bench's ten modes, 3.1x the mark, and the two
contrast-4 cases therefore run at three modes -- the first complete degenerate
triplet -- saying so in their notes. The mode count is the only lever that moves
the uniform reference; on the thin cases it moves only the core, because the
clamp pins the wall's edge near its thickness, so c4-thin needs three modes to
bring its graded side down too, while c3-thin is left at ten modes carrying a
graded side of 220,160 -- flagged, not quietly resized, because the row asks
which cases the Orin cannot solve. THE MARK is the row's own: the n3 uniform
mesh at 199,062 elements OOMed once and solved once (built at the probe floor,
the one the Orin solved; at the corrected floor the same case's uniform side is
258,114 and is flagged). Predicted counts come from
the product's estimator, which reads 1.2-3.3x HIGH across the six points where
the estimate and the built mesh are both known (274,415 predicted against
199,062 built, 128,225 against 105,321, 45,829 against 22,947) and 1.06x on one
of them, so a flagged case may still solve; the flag means expect UNMEASURED,
never a pass.

USAGE
    python3 benchmarks/run_grading_matrix.py --list
    python3 benchmarks/run_grading_matrix.py --case c3-thick
    python3 benchmarks/run_grading_matrix.py --case box-sphere --bands

EXIT CODES are the other bench's: 0 every criterion measured and passed, 2 bad
arguments, 3 a criterion FAILED, 4 a criterion could not be measured.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
_SOLVER_DIR = _BENCH_DIR.parent / "dockerfiles" / "maxwell_solver"
_WEB_DIR = _BENCH_DIR.parent / "web"
for _d in (_BENCH_DIR, _SOLVER_DIR, _WEB_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import run_multimaterial_bench as BENCH  # noqa: E402
import sphere_dielectric as REF  # noqa: E402
from mse_common import region_plan as common_region_plan  # noqa: E402

#: The scorer's own matching tolerance, READ from its signature instead of
#: retyped: the missing-side message quotes this number, and a message quoting
#: a different tolerance than the scorer matched with would mislead exactly the
#: diagnosis it exists to give. It is not a knob -- see ``score_multiplets``.
SCORE_TOL = inspect.signature(REF.score_multiplets).parameters["tol"].default

#: Same exit vocabulary as ``run_multimaterial_bench``: a criterion that could
#: not be measured is NOT the same as one that passed.
MATRIX_OK = 0
MATRIX_USAGE = 2
MATRIX_FAIL = 3
MATRIX_INCOMPLETE = 4
# Not a verdict. They are separate codes from PASS and FAIL on purpose: a
# script that treated "no verdict" as "not a failure" (or as "fine") is the
# defect this bench exists to catch, so neither may share its exit status with
# anything that reads as an answer.
MATRIX_SAME_MESH = 5
MATRIX_UNRELIABLE = 6
MATRIX_NO_VERDICT = 7

#: The tier-2 budget: one solve at a time on the Orin, under the 5 GB limit and
#: 30 minutes. The element mark is empirical -- the n3 uniform reference at
#: 199,062 elements OOMed once and solved once -- and is used to WARN at
#: ``--list`` time, not to refuse: a case whose uniform reference cannot be
#: solved is a case the criterion reports as UNMEASURED, which is the outcome
#: the row asks to be made visible.
TIER2_MAX_TETS = 200_000

#: The uniform reference's refinement levels. The finest is the reference the
#: errors are scored against; the coarser one measures the reference's own
#: convergence. Both are uniform meshes at ``h_min/K``, so each level costs
#: ``K**3`` times the comparison mesh -- K = 2.0 is 8x, which no case whose
#: outside is a 10 mm sphere can afford on tier 2: the offset-sphere case at its
#: first sizing predicted 1,097,656 elements there, five times the mark. The
#: pair is one constant for every case rather than a per-case choice, because a
#: factor picked to make one case fit is the fixture-fitted tuning this row
#: exists to rule out. The cost of that choice is stated where it is used: the
#: reference is 1.5x finer than the mesh it judges, and the convergence check
#: between the two levels is what has to carry its trust.
REFINE_FACTORS = (1.25, 1.5)

#: ``--fine-reference K`` (item 1, step 2c): one extra rung FINER than the
#: default finest, and the reference itself for a uniform-refined case. Two
#: uniform-refined cases (centred-core, offset-sphere) were undecidable because
#: the gap being judged (0.038%) sat under the reference's own convergence at
#: 1.25 vs 1.5 (0.043%); a reference one rung finer shrinks that convergence.
#: K is required rather than defaulted, because the cost is real and per-case:
#: the rung is BUILT, not estimated, and no single value fits every case. The
#: measured price of the two obvious choices (tier 1, ``probe_item1.py``,
#: meshing only -- rerun it before quoting: it is a measurement, not a spec):
#:
#:     k     centred-core   offset-sphere   c4-thin
#:     1.75     130,447         142,555      355,400
#:     2.0      232,242         278,569      595,791   <- k=2.0 is OVER the
#:                                                       200,000 tier-2 mark
#:                                                       for both sphere cases
#:
#: So k=1.75 is the affordable tightening for those two; k=2.0 fits neither, and
#: c4-thin fits neither at any of these levels. ``refine_factors``/
#: ``richardson_scale`` keep the run's rungs and the Richardson ratio derived
#: from the SAME tuple, so the correction always matches the ladder that ran.


def refine_factors(fine=None):
    """The ladder this run uses: the default rungs, plus a finer one if asked.

    ``fine`` is the k of one extra rung FINER than the default finest. It is the
    reference itself for a uniform-refined case, so a finer one is a tighter
    reference -- and it also changes the coarse:fine ratio the Richardson
    correction is derived from, which is why the scale is computed from the
    returned tuple and never from a constant.
    """
    if fine is None:
        return tuple(REFINE_FACTORS)
    if float(fine) <= REFINE_FACTORS[-1]:
        raise ValueError(
            f"a finer reference rung must be > {REFINE_FACTORS[-1]} (the "
            f"current finest), got {fine}")
    return tuple(REFINE_FACTORS) + (float(fine),)


def richardson_scale(factors):
    """``1 / (r**p - 1)`` for the run's own coarse:fine ratio at p = 2."""
    return 1.0 / ((factors[-1] / factors[0]) ** 2 - 1.0)


#: The coarse rung, built only under ``--ladder`` (item 2). Below 1.0 on purpose:
#: the refinement levels are all finer than the comparison mesh, so on their own
#: they can only ever show the error FALLING towards the reference -- a case
#: whose error passes through zero on the way is invisible from above it. One
#: mesh coarser than the comparison (edge ``h_min / 0.8``, 0.51x the elements)
#: puts a rung on the other side, which is what makes a zero crossing
#: measurable rather than argued. It is off by default because it is one more
#: solve per case and the matrix's budget is the lead's.
COARSE_RUNG = 0.8


# ---------------------------------------------------------------- the fixtures
#
# Each builder returns ``(named_meshes, layers, scale_mm)``:
#   named_meshes -- (name, trimesh) OUTERMOST FIRST, the order ``PartSet``
#                   requires (the first part contains every other)
#   layers       -- outermost first; each {"n", "kind", "centre", "r"/"half"},
#                   which assigns eps per tet by GEOMETRY -- the nesting rule a
#                   dielectric follows -- rather than by a region id the mesher
#                   happened to keep
#   scale_mm     -- the length ``kb`` is normalised by for this case
#
# ``target_edge`` is the whole-object edge, so no part's SURFACE is finer than
# the interior it bounds (the bench's surface-floors-the-mesh trap).


def _mesh_lines(parts) -> bytes:
    """OBJ bytes in mm from ``(name, trimesh)`` pairs.

    The writer rule of ``run_multimaterial_bench._parts_obj`` -- one ``o`` line
    per part, face indices offset by the vertices written before them --
    generalised to any mesh, because the shape axis needs a box and a
    translated sphere. Sphere parts here still go through the bench's own
    faceting rule (``_facet_subdivisions``), so a sphere part is identical
    whichever builder wrote it.
    """
    lines: list[str] = []
    n_written = 0
    for name, mesh in parts:
        verts = np.asarray(mesh.vertices, float)
        faces = np.asarray(mesh.faces, np.int64)
        lines.append(f"o {name}")
        lines.extend(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in verts)
        lines.extend(f"f {f[0] + 1 + n_written} {f[1] + 1 + n_written} "
                     f"{f[2] + 1 + n_written}" for f in faces)
        n_written += len(verts)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sphere(name, radius, target_edge, centre=(0.0, 0.0, 0.0)):
    import trimesh
    s = BENCH._facet_subdivisions(radius, target_edge)
    mesh = trimesh.creation.icosphere(subdivisions=s, radius=radius)
    if any(centre):
        mesh = mesh.copy()
        mesh.apply_translation(centre)
    return (name, mesh)


def _box(name, half_extents):
    import trimesh
    return (name, trimesh.creation.box(extents=[2.0 * h for h in half_extents]))


def _concentric(radii, materials, target_edge):
    order = sorted(materials, key=lambda nm: radii[nm], reverse=True)
    meshes = [_sphere(nm, radii[nm], target_edge) for nm in order]
    layers = [{"n": float(materials[nm]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(radii[nm])} for nm in order]
    return meshes, layers, float(radii[order[0]])


def _offset(b_mm, a_mm, offset_mm, materials, target_edge):
    meshes = [_sphere("wall", b_mm, target_edge),
              _sphere("core", a_mm, target_edge, centre=(0.0, 0.0, offset_mm))]
    layers = [{"n": float(materials["wall"]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(b_mm)},
              {"n": float(materials["core"]), "kind": "sphere",
               "centre": (0.0, 0.0, float(offset_mm)), "r": float(a_mm)}]
    return meshes, layers, float(b_mm)


def _box_sphere(half_extents, a_mm, materials, target_edge):
    meshes = [_box("wall", half_extents), _sphere("core", a_mm, target_edge)]
    layers = [{"n": float(materials["wall"]), "kind": "box",
               "centre": (0.0, 0.0, 0.0),
               "half": tuple(float(h) for h in half_extents)},
              {"n": float(materials["core"]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(a_mm)}]
    return meshes, layers, float(min(half_extents))


# --------------------------------------------------------------- the case list

#: The other bench's outer radius, so ``c3-thick`` is that bench's n3 fixture.
B_MM = 10.0
A_THICK = B_MM / 3.0     # wall 6.67 mm thick: the clamp cannot bite
A_THIN = 0.9 * B_MM      # wall 1.0 mm thick: the clamp bites from n = 2 up


def _case(label, build, materials, reference, axes, n_modes=10, note=""):
    return {"label": label, "build": build, "materials": materials,
            "reference": reference, "axes": axes, "n_modes": n_modes,
            "note": note}


MATRIX = [
    _case("c1.5-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 1.5}, te),
          {"wall": 1.0, "core": 1.5}, "analytic",
          {"contrast": 1.5, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          note="the lowest contrast the feature is meant to help at"),
    _case("c2-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "analytic",
          {"contrast": 2.0, "shell": "thick", "regions": 2,
           "shape": "concentric"}),
    _case("c3-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "analytic",
          {"contrast": 3.0, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          note="the other bench's n3 fixture, reproduced here"),
    _case("c4-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 4.0}, te),
          {"wall": 1.0, "core": 4.0}, "analytic",
          {"contrast": 4.0, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          n_modes=3,
          note="contrast 4 costs 19x the contrast-1.5 case: at the sibling's ten "
               "modes its uniform side is 611,826 elements, 3.1x the tier-2 "
               "mark. Three modes -- the first complete degenerate triplet -- "
               "brings the pair to 16,215 graded and 184,163 uniform, both "
               "under the mark."),
    _case("c2-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "analytic",
          {"contrast": 2.0, "shell": "thin", "regions": 2,
           "shape": "concentric"},
          note="wall 1.0 mm thick vs the 1.19 mm the rule asks: clamp bites"),
    _case("c3-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "analytic",
          {"contrast": 3.0, "shell": "thin", "regions": 2,
           "shape": "concentric"},
          note="the clamp pins the wall near 1.0 mm, so the graded side is "
               "220,160 elements at ten modes against c3-thick's 35,314 on the "
               "same rule. Its uniform side is NOT c3-thick's mesh, which is "
               "what the corrected baseline changed: the thin wall's clamp "
               "raises the case's own base edge (0.7217 mm against c3-thick's "
               "0.6084), so the two uniform meshes come out different "
               "(154,673 estimated against 258,114)."),
    _case("c4-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 4.0}, te),
          {"wall": 1.0, "core": 4.0}, "analytic",
          {"contrast": 4.0, "shell": "thin", "regions": 2,
           "shape": "concentric"},
          n_modes=3,
          note="contrast 4 AND a wall the clamp pins near 1.0 mm: at ten modes "
               "the pair is 494,772 graded against 366,633 uniform, both over "
               "the mark, and three modes brings them to 161,389 and 109,990 -- "
               "the graded side 19% under it."),
    _case("three-region",
          lambda te: _concentric({"wall": B_MM, "mid": 0.6 * B_MM,
                                  "core": 0.25 * B_MM},
                                 {"wall": 1.0, "mid": 1.5, "core": 2.0}, te),
          {"wall": 1.0, "mid": 1.5, "core": 2.0}, "uniform-refined",
          {"contrast": 2.0, "shell": "thick", "regions": 3,
           "shape": "concentric"},
          n_modes=5,
          note="three distinct indices: no closed-form spectrum. The ramp is "
               "mild (1/1.5/2) because the reference's own finer level costs "
               "contrast**3: the same three regions at 1/2/4 predict 873,764 "
               "elements at the 1.5 level, 4.4x the mark, and 258,893 for the "
               "uniform side alone (the 1/1.5/2 case's finest level is "
               "109,218)."),
    _case("box-sphere",
          lambda te: _box_sphere((5.0, 3.5, 2.5), 1.6,
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "uniform-refined",
          {"contrast": 3.0, "shell": "thick", "regions": 2, "shape": "box"},
          n_modes=5,
          note="non-spherical cavity: a 10x7x5 box. Its own surface edge is "
               "1.0105 mm, 3x the core's 0.3368 mm floor, so the box pays for "
               "its shape on EVERY side: on the corrected baseline the uniform "
               "mesh predicts 128,225 elements (under the mark, so it would "
               "solve) and the two reference levels 250,439 and 432,759 (both "
               "over it). As written this case's REFERENCE is what cannot be "
               "built on tier 2, and the row's Orin run returned it "
               "UNMEASURED."),
    _case("offset-sphere",
          lambda te: _offset(B_MM, 2.5, 4.0, {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "uniform-refined",
          {"contrast": 2.0, "shell": "thick", "regions": 2, "shape": "offset"},
          n_modes=5,
          note="non-concentric: the core is 4 mm off centre. What this case "
               "carries is the SHAPE -- contrast 3 is box-sphere's -- because "
               "the reference costs contrast**3: at contrast 3 and five modes "
               "its levels predict 257,615 and 445,159 elements (2.2x the "
               "mark), and at contrast 3 and three modes still 267,094 at the "
               "1.5 level, over the mark. At contrast 2 the whole pair fits."),
    # offset-sphere's CONTROL, identical in every argument but the offset. The
    # lead's item 4 asks why an off-centre core loses 2.27x where a centred one
    # passes, and the only way to answer with a number rather than a story is to
    # hold everything else fixed: same builder, same radii, same materials, same
    # n_modes, same reference kind. If the offset is the cause, this case passes
    # and the core's built edge lands on (or near) its asked edge; if it also
    # fails, the cause is the CONTRAST and the offset is a coincidence.
    _case("centred-core",
          lambda te: _offset(B_MM, 2.5, 0.0, {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "uniform-refined",
          {"contrast": 2.0, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          n_modes=5,
          note="offset-sphere with the core moved to the centre: same 10 mm "
               "cavity, same 2.5 mm core, same contrast 2, same five modes, the "
               "same `_offset` builder with offset 0. It is the control for the "
               "off-centre FAIL (graded 0.098% against uniform 0.043% on the "
               "Orin at 1935511), and its job is to be an A/B, not a product "
               "case. PREDICTION: it PASSES, and its core's built edge lands "
               "within a few percent of its asked 1.1416 mm, because a centred "
               "core is surrounded by the fine background field on all sides "
               "while the off-centre one is pushed by a coarse outer region "
               "from one side. FALSIFIER: if this case FAILS the same way, the "
               "offset is not the cause and the contrast is."),
    _case("shell-dense",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 2.0, "core": 1.0}, te),
          {"wall": 2.0, "core": 1.0}, "uniform-refined",
          {"contrast": 2.0, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          n_modes=5,
          note="A DENSE SHELL AROUND A LIGHTER CORE: the index decreases "
               "inward, so no region is denser than the wall. Every other "
               "graded case has the ramp rising inward and n_wall=1. The rule's "
               "raw edge is h*n_wall/n_r, and its outer min(h, .) is what "
               "reaches here: the core's raw ask is 2h, capped to h, so the "
               "plan it produces IS uniform -- the prediction this case exists "
               "to test. Measured on the corrected baseline (tier 1, meshing "
               "only): both regions ask 2.2972 mm and the two meshes come out "
               "IDENTICAL at 2,753 tets each, so this case cannot show the "
               "rule doing anything at all. (The estimator reads 4,952 for the "
               "same two meshes -- 1.8x -- which is the instrument difference "
               "`--list` carries; the built mesh is the authority.)"),
    _case("ramp-outward",
          lambda te: _concentric({"wall": B_MM, "mid": 0.6 * B_MM,
                                  "core": 0.25 * B_MM},
                                 {"wall": 2.0, "mid": 1.5, "core": 3.0}, te),
          {"wall": 2.0, "mid": 1.5, "core": 3.0}, "uniform-refined",
          {"contrast": 2.0, "shell": "thick", "regions": 3,
           "shape": "concentric"},
          n_modes=5,
          note="AN OUTER REGION WITH INDEX ABOVE 1: n_wall=2.0 where every "
               "other case has 1.0, so the rule divides by a wall term that "
               "has never been exercised. Only the core (3.0) is denser than "
               "the wall, and its raw edge is h*2/3 -- a milder grading than "
               "the same ramp gives at n_wall=1. Measured on the corrected "
               "baseline (tier 1, meshing only): both non-core regions ask "
               "2.4313 mm -- the middle one capped by min(h, .) at the wall's "
               "own edge, though its material ratio asked for more -- and the "
               "core asks 1.6209 mm. Built: 6,746 graded tets against 7,350 "
               "for the region-aware uniform at the same floor, 8% cheaper -- "
               "and the ledger is NOT one-sided: the wall accounts for the "
               "saving (5,494 tets against 6,710) while the middle region "
               "GAINS 612 (1,172 against 560), because the uniform build's "
               "middle came out 3.5000 mm against an asked 1.6209 (2.16x "
               "coarse, item 2's other direction) where the graded build "
               "reached 1.9155. The core holds 80 tets of the 6,746 (1.2%). "
               "(Estimator: 7,167 and 13,652 -- 1.9x. The built mesh is the "
               "authority.)"),
]


def _coverage(cases=None) -> dict:
    """Which value each axis has a case for. Read by a test."""
    cov: dict[str, set] = {k: set() for k in
                           ("contrast", "shell", "regions", "shape", "reference")}
    for c in (cases if cases is not None else MATRIX):
        for k in ("contrast", "shell", "regions", "shape"):
            cov[k].add(c["axes"][k])
        cov["reference"].add(c["reference"])
    return cov


# ------------------------------------------------------------------- geometry


def _contains(layer, p) -> bool:
    cx, cy, cz = layer["centre"]
    if layer["kind"] == "sphere":
        return ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
                <= layer["r"] ** 2)
    hx, hy, hz = layer["half"]
    return (abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy
            and abs(p[2] - cz) <= hz)


def _eps_by_layers(points_m, tets, layers) -> np.ndarray:
    """``eps_r`` per tet: the innermost layer containing its centroid."""
    cent = np.asarray(points_m, float)[np.asarray(tets, np.int64)].mean(axis=1)
    cent = cent * 1e3                                    # mm, like ``layers``
    eps = np.ones(cent.shape[0], float)
    for layer in layers:                    # outermost first: inner ones win
        eps[np.array([_contains(layer, p) for p in cent])] = layer["n"] ** 2
    return eps


def _outer_surface(parts):
    return parts.parts[parts.wall_index if parts.wall_index is not None else 0]


# ------------------------------------------------------------------ the runner


def _build(case, n_modes: int):
    """The plan and the base edge the PRODUCT derives for this case, as a dict.

    Keys: ``obj, parts, plan, h_min_mm, probe_h_min_mm, layers, scale_mm``.

    The per-region edges come from ``mse_common.region_plan``, and the base edge
    fed to it comes from ``mse_common.effective_edge_m`` measured on **this
    case's own merged surface** -- the two production calls the mesher itself
    makes (``mse_meshing.mse_tetrahedralize``). An earlier version of this
    function measured the base edge on a fixed icosphere PROBE of radius
    ``B_MM`` instead, and for a shape that is not that icosphere the two edges
    differ: measured on 2026-10-14, box-sphere's own surface gives 1.0105 mm
    where the probe gives 2.2533 mm, so the probe's floor came out 0.7511 mm
    against the product's 0.3368 mm -- and the whole uniform side of that case
    was then built and scored at a resolution 2.23x coarser than the mesh the
    product actually used. ``probe_h_min_mm`` is kept so the runner can report
    the old number beside the corrected one instead of silently dropping it.

    ``probe_h_min_mm`` is computed from the probe target the case builders still
    take; the case's own ``build`` is called with the case's own target so any
    builder that sizes a fixture from the edge gets the product's number too.
    """
    import trimesh
    import mse_common as C
    import mse_meshing as MESH
    import mse_maxwell_solver as SOLVER

    probe = trimesh.creation.icosphere(subdivisions=2, radius=B_MM)
    probe_target = float(MESH.target_edge_length(probe, n_modes=n_modes,
                                                 units="mm", n_index=1.0))
    meshes, layers, scale_mm = case["build"](probe_target)
    obj = _mesh_lines(meshes)
    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    surface = SOLVER.PARTS.merged_surface(parts)
    # the product's own base edge, for THIS surface (mse_meshing.py:~517)
    edge_m = float(C.effective_edge_m(
        surface, n_modes=n_modes, units="mm", n_index=1.0, parts=parts,
        materials=case["materials"]))
    plan = common_region_plan(parts, case["materials"], edge_m, "mm")
    h_min = min(i["edge_m"] for i in plan.values()) * 1e3
    probe_plan = common_region_plan(parts, case["materials"],
                                    probe_target * 1e-3, "mm")
    probe_h_min = min(i["edge_m"] for i in probe_plan.values()) * 1e3
    return {"obj": obj, "parts": parts, "plan": plan, "h_min_mm": h_min,
            "probe_h_min_mm": probe_h_min, "layers": layers,
            "scale_mm": scale_mm}


def _kb(freqs, scale_mm):
    """``kb = 2 pi f L / c`` with L the case's own scale, in metres."""
    return 2.0 * np.pi * np.asarray(freqs, float) * (scale_mm * 1e-3) / 299_792_458.0


def _graded(case, obj, work, *, grid, backend, verbose):
    """The GRADED run, through the product's own entry point."""
    import mse_maxwell_solver as SOLVER

    store_dir = work / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    store = SOLVER.STORE.LocalStore(str(store_dir))
    job_id = f"matrix-{case['label']}"
    store.write(SOLVER.STORE.job_key(job_id, "input.obj"), obj)
    t0 = time.time()
    summary = SOLVER.solve(job_id, "mm", 1.0, case["n_modes"], store, store,
                           max_tets=600_000, grid=grid, verbose=verbose,
                           backend=backend, materials=case["materials"])
    modes = json.loads(store.read(SOLVER.STORE.job_key(job_id, "modes.json")))
    freqs = np.array([m["freq_hz"] for m in modes["modes"]], float)
    return summary, modes, freqs, time.time() - t0


def _uniform(materials, obj, case, h_min_mm, *, k: float, layers, backend,
             np_verbose=False):
    """One uniform mesh+solve at ``h_min/k``, the same bytes.

    Mirrors ``run_multimaterial_bench._run_case``'s reference half: the mesh is
    ``tetrahedralize`` with ``min_feature_m`` forcing the fine edge onto a
    one-part mesh (``materials=None``), and the spectrum comes from
    ``_solve_mesh`` -- the product's own five solver calls -- with eps supplied
    per tet by geometry, because a uniform mesh carries no regions.
    """
    import mse_meshing as MESH
    import mse_maxwell_solver as SOLVER

    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    merged = SOLVER.PARTS.merged_surface(parts)
    points, tets, _region = MESH.tetrahedralize(
        merged, "mm", n_modes=case["n_modes"], n_index=1.0, max_tets=600_000,
        min_feature_m=(h_min_mm / k) * 1e-3, verbose=np_verbose,
        materials=None, parts=None)
    eps = _eps_by_layers(points, tets, layers)
    kb = BENCH._solve_mesh(points, tets, eps, case["n_modes"], backend)
    return points, tets, kb


def _uniform_regions(case, obj, h_min_mm, verbose=False):
    """A UNIFORM mesh through the product's own region-aware path. NO SOLVE.

    ``_uniform`` meshes one part with ``materials=None``, so its mesh carries no
    region attribute and no per-region report: it cannot say what edge any one
    region came out at, which is what two of the Orin diagnosis items ask. This
    runs the same ``tetrahedralize`` call the graded side runs -- the product's
    multi-part path, region attributes and all -- with every region's target
    edge pinned to ``h_min``. That is the rule the product meshed by BEFORE the
    grading feature (``mse_meshing._region_plan``: "the previous rule meshed
    EVERY region at h_min"), so the per-region table here is the pre-grading
    build's own report.

    The pin is a bench-side patch of ``mse_common.region_edges``: the product's
    rule is NOT changed and the graded side in the same case is unaffected,
    because the patch is installed and removed inside this call.
    """
    import mse_common as C
    import mse_meshing as MESH
    import mse_maxwell_solver as SOLVER

    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    merged = SOLVER.PARTS.merged_surface(parts)
    pinned_m = h_min_mm * 1e-3
    real_edges = C.region_edges

    def _pinned(h, n_wall, materials, prts, units, _real=real_edges,
                _h=pinned_m):
        out = _real(h, n_wall, materials, prts, units)
        for region_id in out:
            out[region_id]["edge"] = _h
            out[region_id]["unclamped"] = _h
        return out

    C.region_edges = _pinned
    try:
        _points, tets, region_of_tet = MESH.tetrahedralize(
            merged, "mm", n_modes=case["n_modes"], n_index=1.0,
            max_tets=600_000, verbose=verbose, materials=case["materials"],
            parts=parts)
    finally:
        C.region_edges = real_edges
    return tets, region_of_tet, MESH.LAST_MESH_REPORT


def _clusters_pairs(kb):
    """A measured spectrum in ``score_multiplets``' reference shape.

    ``l`` and ``family`` are report labels: this reference carries no
    degeneracy labels, so the matcher is used for its clustering and one-to-one
    matching, not for its physics.
    """
    return [(float(c), 0, "uniform-ref", int(n))
            for c, n in REF.cluster_modes(np.asarray(kb, float), 0.02)]


def _full_spectrum(reference, n_modes):
    """The part of a reference spectrum the run can actually score against.

    An ANALYTIC reference is the whole spectrum; the run asks for ``n_modes``
    modes, so only the multiplets covering the lowest ``n_modes`` are reachable
    and the rest are noise for "did the reference build" purposes. A
    uniform-refined reference is built for exactly those clusters, so it is
    already the part in question.
    """
    covered, out = 0, []
    for entry in sorted(reference, key=lambda r: float(r[0])):
        if covered >= n_modes:
            break
        out.append(entry)
        covered += int(entry[3])
    return out or list(reference)


def _looks_like_a_mesh(value) -> bool:
    """True when ``value`` is an ``(n, 3)`` array of finite float coordinates.

    The failure this exists for is not an exception: a build that returns a
    partial or degenerate array on a refused refinement is still truthy and
    still array-shaped, and the only thing detectable from the outside is that
    its shape is not a node cloud. Counting elements comes after, so a truthy
    empty array cannot pass for a mesh either.
    """
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return False
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return False
    return bool(np.isfinite(arr.astype(float)).all())


def _last_residual():
    """The solver's measured worst refined residual, or ``None``.

    Read from ``mse_eigen``'s module state -- the same global the solver copies
    into ``modes.json`` as ``solver_worst_residual`` -- because a uniform level's
    solve goes through ``_solve_mesh``'s five calls rather than ``solve()`` and
    so writes no ``modes.json`` of its own. It is ``None`` on a backend that
    does not refine (the CPU factorisation records no ``worst_residual``), and
    ``None`` must stay ``None``: a tolerance defaulted to a number here would be
    a picked number, which is the one thing the tolerance may not be.
    """
    try:
        import mse_eigen as EIG
        return EIG.LAST_WORST_RESIDUAL
    except Exception:  # noqa: BLE001 - a missing reader is a missing number
        return None


def _num(x):
    """A number for a report line, or ``None`` printed as ``None``.

    ``None`` must stay visible: a missing per-region measurement that printed as
    ``0`` would read as a mesh with no edge.
    """
    return None if x is None else round(float(x), 4)


def missing_side_reason(tag, score, reference):
    """WHY a side has no worst error -- as a sentence, or ``None``.

    ``score_multiplets`` returns ``worst_rel_error=None`` exactly when its
    ``matched`` list is empty, and it already carries every fact needed to say
    what that means: how many clusters were found, how many of the reference's
    multiplets went unmatched, and how far the nearest cluster sat from a
    reference. The bench's previous message -- "one of the two worst errors is
    missing" -- named neither the side nor a single number, which is how a
    UNMEASURED row could be read as "nothing there" instead of "nothing
    matched". A side that produced no number MUST say which side it is and what
    it did produce.
    """
    if score.get("worst_rel_error") is not None:
        return None
    n_clusters = score.get("n_clusters")
    matched = score.get("n_matched")
    spurious = score.get("spurious") or []
    nearest = min((s["nearest_rel"] for s in spurious
                   if s.get("nearest_rel") is not None), default=None)
    unmatched = score.get("unmatched_reference") or []
    if nearest is None:
        tail = f"no cluster came within the {SCORE_TOL * 100:.2f}% tolerance"
    else:
        tail = (f"nearest cluster-to-reference distance "
                f"{nearest * 100:.2f}% against the {SCORE_TOL * 100:.2f}% "
                f"tolerance")
    return (f"{tag} matched {matched} of its {n_clusters} clusters to the "
            f"reference ({len(unmatched)} of {len(reference)} reference "
            f"multiplets unmatched); {tail}")


def plan_is_uniform(plan) -> bool:
    """True when every region's TARGET edge came out the same.

    That is the plan-level statement of "grading did nothing": the rule asks for
    a different edge per region, and if the clamping floor and the region
    thicknesses have flattened them to one value, the graded build and the
    uniform build are the same mesh by construction. c2-thin is the real
    instance -- both regions asked 1.0825 mm, so both sides came out 22,947
    elements.

    A single-region plan is not uniform-and-graded: it has no contrast to
    flatten, so it cannot be evidence about grading either way and is not
    reported as SAME MESH.
    """
    edges = [v.get("edge_m") for v in (plan or {}).values()
             if isinstance(v, dict) and isinstance(v.get("edge_m"), float)]
    return len(edges) >= 2 and max(edges) == min(edges)


def signed_pairs(score):
    """One row per MATCHED multiplet, with the error's SIGN kept.

    ``score_multiplets`` reports ``rel_error = |computed - reference| /
    reference``: a magnitude, so two runs whose errors sit on opposite sides of
    the reference are indistinguishable from two runs on the same side. That is
    the same trap as reading ``|E|`` where ``E`` was meant (it rectifies, and
    the sign is exactly what a zero crossing lives in), and it is why this
    recomputes the sign from the pair the scorer already kept -- ``centre`` and
    ``reference_kb`` -- rather than from the magnitude it returned.

    ``reference_kb`` of 0 would divide by zero; a mode with no reference energy
    has no relative error to report, so it comes back ``None`` and the report
    prints that rather than inf.
    """
    rows = []
    for m in (score.get("matched") or []):
        ref = m.get("reference_kb")
        centre = m.get("centre")
        rows.append({
            "l": m.get("l"), "family": m.get("family"),
            "reference_kb": ref, "centre_kb": centre,
            "count": m.get("count"),
            "reference_multiplicity": m.get("reference_multiplicity"),
            "signed_rel": (None if not ref or centre is None
                           else (centre - ref) / ref),
        })
    return sorted(rows, key=lambda r: (r["reference_kb"] or 0.0))


def per_mode_table(graded, uniform):
    """Both sides' SIGNED per-multiplet errors, joined on the reference.

    Keyed by the reference multiplet, because that is the object both sides are
    measured against; a multiplet one side did not match is printed as a missing
    entry, not dropped, so a table built from a partial match cannot read as a
    complete one.
    """
    by_ref = {}
    for tag, score in (("graded", graded), ("uniform", uniform)):
        for row in signed_pairs(score):
            key = ("ref", round(row["reference_kb"], 9)
                   if row["reference_kb"] is not None else None,
                   row["l"], row["family"])
            by_ref.setdefault(key, {"l": row["l"], "family": row["family"],
                                    "reference_kb": row["reference_kb"]})
            by_ref[key][f"{tag}_signed_rel"] = row["signed_rel"]
    return [by_ref[k] for k in sorted(by_ref, key=lambda k: (k[1] or 0.0))]


def zero_crossings(rows):
    """Multiplets whose signed error changes SIGN somewhere in this table.

    A crossing is the one case where magnitude comparison is not merely noisy
    but wrong: on either side of the crossing the magnitude can be made as small
    as the crossing is well reached, so a mesh that is COARSER than another can
    score a smaller magnitude. Reported rather than silently averaged away.
    """
    out = []
    for row in rows:
        a, b = row.get("graded_signed_rel"), row.get("uniform_signed_rel")
        if a is None or b is None:
            continue
        if a == 0.0 or b == 0.0 or (a > 0) != (b > 0):
            out.append(row)
    return out


def level_errors(levels, reference, n_modes):
    """Each refinement level's error against the reference, finest last.

    ``levels`` is the map ``run_case`` already builds -- ``{k: kb}`` -- so the
    monotonicity check costs no extra solve. A level that failed to build is
    absent from ``levels`` and therefore absent here; the caller must not read a
    short list as a monotone one.
    """
    out = []
    for k in sorted(levels):
        score = REF.score_multiplets(levels[k], reference, n_requested=n_modes)
        out.append({"k": k, "worst_rel_error": score.get("worst_rel_error"),
                    "n_matched": score.get("n_matched")})
    return out


def self_ladder(levels, n_modes):
    """Each uniform rung scored against the FINEST rung's own spectrum.

    The ladder with the reference taken out of it (item 2, step 2c). A
    non-falling error ladder has two causes with opposite verdicts: the mesh is
    not converging, or the mesh IS converging and the reference is what moves.
    Scoring every rung against the finest rung's spectra separates them and
    costs no solve, because ``levels`` already holds them.

    The comparison is legitimate, not a proxy: for a FIXED reference the change
    in the error against it between two rungs equals the change between those
    two rungs' own spectra, both taken through ``score_multiplets`` as the same
    ``worst_rel_error``. When the two differ, the reference is the one moving.
    """
    if len(levels) < 2:
        return []
    finest = max(levels)
    reference = _clusters_pairs(levels[finest])
    out = []
    for k in sorted(levels):
        if k == finest:
            continue
        score = REF.score_multiplets(levels[k], reference, n_requested=n_modes)
        out.append({"k": k, "against": finest,
                    "self_rel_error": score.get("worst_rel_error"),
                    "n_matched": score.get("n_matched")})
    return out


def reference_multiplets_in_reach(reference, n_modes):
    """Distinct reference multiplets the ASKED-FOR modes can reach.

    Not the length of the reference list: an analytic reference carries the whole
    spectrum (c4-thin's has tens of multiplets), while the run asks for three
    modes and therefore only ever sees the lowest ones. The count that matters
    for a verdict is how many independent targets the comparison actually has,
    so the spectrum is walked from the lowest multiplet up until the multiplicities
    cover ``n_modes``. c4-thin: three modes, one triplet -- one target, and one
    signed number deciding the verdict.
    """
    covered, distinct = 0, set()
    for kb, _l, _fam, m in sorted(reference, key=lambda r: float(r[0])):
        if covered >= n_modes:
            break
        distinct.add(round(float(kb), 9))
        covered += int(m)
    return max(len(distinct), 1)


def _self_ladder_note(selflad, ref_step):
    """One clause naming what the mesh did on its own, or ``None``.

    ``selflad`` is :func:`self_ladder`; ``ref_step`` is how much the error
    against the reference moved over the same finest step (a magnitude). Both
    are ``worst_rel_error`` from the same scorer, so they are directly
    comparable and no factor is picked: when the mesh's own step is smaller,
    the reference is what moved.
    """
    entries = [e for e in (selflad or []) if e.get("self_rel_error") is not None]
    if not entries or ref_step is None:
        return None
    e = entries[-1]                      # the finest step the ladder has
    own = e["self_rel_error"]
    if own < ref_step:
        return (f"whereas the same two rungs differ from each other by only "
                f"{own * 100:.3f}% (k = {e['k']:g} -> {e['against']:g}), so the "
                "mesh has converged and the REFERENCE is what moved")
    return (f"and the same two rungs differ from each other by {own * 100:.3f}% "
            f"(k = {e['k']:g} -> {e['against']:g}) against "
            f"{ref_step * 100:.3f}% against the reference, so the mesh itself is "
            "still moving and no reference at these levels can decide it")


def reliability(errors, n_multiplets, rows, selflad=None, reference_kind=None):
    """``None`` if this case can carry a verdict, else the reason it cannot.

    Two failure modes, both of which make a PASS or a FAIL read as information
    that is not there:

    * the uniform side's error does not fall as its mesh is refined (a
      non-monotone ladder means the errors are dominated by something other
      than discretisation -- a zero crossing, a spurious cluster, a sign flip
      between levels -- and the comparison is against a curve, not a level);
    * the case has a single reference multiplet, so one signed number decides
      everything and a crossing has nowhere else to show itself (c4-thin: three
      modes are one triplet, and one triplet is one number).

    ``reference_kind`` marks the further consequence for an ``analytic``
    reference: a closed-form spectrum has no finer level of its own to converge
    into, so when the ladder does not fall there is no knob left to tighten
    it -- the case is unusable as a grading fixture, not merely unjudged.
    """
    worsts = [e["worst_rel_error"] for e in errors
              if e.get("worst_rel_error") is not None]
    if len(worsts) < len(errors):
        return ("a refinement level of the uniform side did not match the "
                "reference, so its ladder is incomplete")
    ladder = ", ".join(f"{w * 100:.3f}%" for w in worsts)
    ks = ", ".join(f"{e['k']:g}" for e in errors)
    if len(worsts) >= 2 and any(b > a for a, b in zip(worsts, worsts[1:])):
        ref_step = abs(worsts[-1] - worsts[-2])
        note = _self_ladder_note(selflad, ref_step)
        analytic = ("; the reference is ANALYTIC, a closed-form spectrum with no "
                    "finer level of its own to converge into, so it cannot be "
                    "tightened -- this case is unusable as a grading fixture "
                    "until it has a meshed reference"
                    if reference_kind == "analytic" else "")
        return ("the uniform side's error does not fall as its mesh is refined "
                f"({ladder} at k = {ks}), so it is not tracking "
                "discretisation" + (f"; {note}" if note else "") + analytic)
    if n_multiplets <= 1:
        return (f"the reference has {n_multiplets} multiplet, so one signed "
                "number decides the verdict and a zero crossing has nowhere "
                "else to appear")
    if zero_crossings(rows):
        return (f"{len(zero_crossings(rows))} multiplet(s) change sign between "
                "the two meshes, so their magnitudes cannot be compared")
    return None


def criterion(graded_worst, uniform_worst, unmeasured_reason, *,
              same_plan=False, tol=0.0, sides_over_mark=(),
              reference_convergence=None):
    """The owner's bar as one pure function of the two numbers and their limits.

    Returns ``(verdict, detail)``. A verdict is one of:

    ``UNMEASURED``
        something the bar needs is missing or could not be built -- and it never
        returns PASS: a mesh whose reference could not be solved is not a mesh
        that beat it. A side over the tier-2 mark on its BUILT count is here
        too, because a mesh that cannot be built on the tier cannot be judged on
        it (the lead's item 3: judge on built counts, keep the mark).
    ``SAME MESH``
        the graded and uniform builds are the same mesh: the plan was uniform,
        so grading did nothing and the case says nothing about grading. Calling
        that a PASS would report a rule that did nothing as a rule that held.
    ``NO VERDICT``
        the gap being judged is smaller than the reference's own convergence.
        The comparison then measures the reference, not the two meshes: with a
        reference error of ``c`` and a gap below it, both sides' numbers move by
        ``c`` and the sign of the difference is not determined (box-sphere: the
        gap is 0.023% against the reference's 1.007%).
    ``PASS`` / ``FAIL``
        otherwise, and PASS needs ``graded <= uniform + tol`` where ``tol`` is
        the solver's own measured residual: two meshes whose scores differ by
        less than the solver's noise have not been shown to differ.
    """
    if unmeasured_reason:
        return "UNMEASURED", unmeasured_reason
    if same_plan:
        return "SAME MESH", ("the graded plan is uniform, so the graded and "
                             "uniform builds are the same mesh: grading did "
                             "nothing in this case, and it cannot pass or fail")
    if graded_worst is None or uniform_worst is None:
        return "UNMEASURED", "one of the two worst errors is missing"
    if sides_over_mark:
        return "UNMEASURED", ("over the tier-2 mark on BUILT counts: "
                              + ", ".join(sorted(sides_over_mark))
                              + f" -- the mark is {TIER2_MAX_TETS:,} elements")
    gap = abs(graded_worst - uniform_worst)
    if reference_convergence is not None and gap < reference_convergence:
        return "NO VERDICT", (
            f"the gap {gap * 100:.3f}% is smaller than the reference's own "
            f"convergence {reference_convergence * 100:.3f}%, so this "
            f"comparison is measuring the reference: graded "
            f"{graded_worst * 100:.3f}% vs uniform "
            f"{uniform_worst * 100:.3f}%")
    ratio = (graded_worst / uniform_worst) if uniform_worst else float("inf")
    within_tol = bool(tol) and gap <= tol
    verdict = "PASS" if graded_worst <= uniform_worst + tol else "FAIL"
    return verdict, (f"graded {graded_worst * 100:.3f}% vs uniform "
                     f"{uniform_worst * 100:.3f}% ({ratio:.2f}x the uniform mesh)"
                     + (f"; within the solver's measured residual {tol:.2e} "
                        "-- NOT measurably worse" if within_tol else ""))


def favourable_bound(uniform_worst, graded_worst, reference_convergence,
                     richardson: float = 1.0):
    """The uniform side's number with the reference's own error ADDED to it, and
    what is left of the verdict -- or ``None`` when an operand is missing.

    Item 5 of the Orin diagnosis: on a ``uniform-refined`` reference both sides
    are measured against a finer mesh of the SAME family as the uniform one, so
    the comparison leans the uniform side's way, and the reference's own
    convergence between its two levels is the only bound available for how much.
    A bound built from a gap would read as a measurement, so all three operands
    are required. The raw bound adds the whole coarse-vs-finest convergence,
    which is a FLOOR on the correction rather than an upper bound on it -- see
    ``richardson``.

    ``richardson`` scales that difference to an estimate of the REFERENCE's own
    error, which is the quantity that has to come off the uniform side: the
    convergence measured is ``|e_coarse - e_fine|``, and for a method of order p
    refined by ``K_fine / K_coarse`` that is ``(1 - r**-p)`` of ``e_fine``, so
    ``richardson = 1 / (r**p - 1)``. For this bench's 1.25 -> 1.5 pair at p = 2
    it is 2.27. It is an ASSUMPTION -- one order, one constant -- so the raw and
    the scaled bounds are both reported rather than one being chosen here.
    """
    if uniform_worst is None or graded_worst is None or reference_convergence is None:
        return None
    p = {}
    for tag, factor in (("", 1.0), ("richardson_", richardson)):
        bound = uniform_worst + factor * reference_convergence
        p[f"{tag}uniform"] = bound
        p[f"{tag}ratio"] = (graded_worst / bound) if bound else None
        p[f"{tag}still_fail"] = bool(graded_worst > bound)
    p.update({"uniform_worst": uniform_worst, "graded_worst": graded_worst,
              "reference_convergence": reference_convergence,
              "richardson": richardson})
    return p


#: Richardson scale for the bench's own refinement pair, from the order of the
#: eigenproblem's discretisation (p = 2 for linear tets with a consistent mass),
#: applied only to a reference's coarse-vs-finest convergence. See
#: ``favourable_bound``.
RICHARDSON_SCALE = richardson_scale(REFINE_FACTORS)


def run_case(case, *, grid, backend, work: Path, verbose, bands=False,
             baseline: str = "product", ladder=False,
             fine_reference: float | None = None) -> dict:
    """Build, mesh and solve ONE matrix case, and judge it.

    ``baseline`` picks the resolution of the UNIFORM side: ``"product"`` builds
    it at the finest edge the product's own plan asks for (the rule the product
    meshed by before grading: "every region at h_min"), ``"probe"`` reproduces
    the older probe-derived floor so the numbers already measured on the Orin
    can still be reproduced. Both floors and both counts are reported either
    way -- the point of the pair is that they differ.

    ``ladder`` adds the coarse rung (see ``COARSE_RUNG``) so the uniform side's
    error can be seen from both sides of the reference; without it a crossing is
    invisible. It costs one extra build and its solve.

    ``fine_reference`` adds one rung FINER than the default finest (see
    ``refine_factors``), which tightens the reference's own convergence for a
    uniform-refined case -- the knob for a gap that sits under the reference's
    error. It costs one extra build and its solve, and the price grows ~K^3.
    """
    import mse_maxwell_solver as SOLVER

    n_modes = case["n_modes"]
    out: dict = {"label": case["label"], "n_modes": n_modes,
                 "reference_kind": case["reference"], "axes": case["axes"],
                 "note": case.get("note", "")}
    try:
        built = _build(case, n_modes)
    except Exception as exc:  # noqa: BLE001 - a fixture the producer rejects
        out["build_error"] = f"{type(exc).__name__}: {exc}"
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the fixture could not be built: {out['build_error']}",
                "case": out}
    obj, parts, plan = built["obj"], built["parts"], built["plan"]
    layers, scale_mm = built["layers"], built["scale_mm"]
    h_min = built["h_min_mm"] if baseline == "product" else built["probe_h_min_mm"]
    out.update(h_min_mm=h_min, h_min_mm_product=built["h_min_mm"],
               h_min_mm_probe=built["probe_h_min_mm"], baseline=baseline,
               scale_mm=scale_mm,
               plan={str(k): {kk: round(vv, 9) if isinstance(vv, float) else vv
                              for kk, vv in v.items()}
                     for k, v in plan.items()})

    try:
        summary, modes, freqs, elapsed = _graded(case, obj, work, grid=grid,
                                                 backend=backend,
                                                 verbose=verbose)
        kb_graded = _kb(freqs, scale_mm)
        out["n_tets_graded"] = int(summary["n_tets"])
        out["elapsed_s_graded"] = round(elapsed, 1)
        mesh_report = modes.get("mesh") or {}
        out["graded_regions"] = (mesh_report.get("regions") or {})
    except Exception as exc:  # noqa: BLE001 - a failed graded solve is data
        out["graded_error"] = f"{type(exc).__name__}: {exc}"
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the GRADED solve raised: {out['graded_error']}",
                "case": out}

    # The uniform reference's spectra, measured, one level per factor.
    levels: dict[float, np.ndarray] = {}
    uniform_error = None
    out["uniform"] = {}
    # LADDER (item 2). The refinement ladder has a structural blind spot: for a
    # uniform-refined case the reference IS the uniform mesh at the finest rung,
    # so that rung's own error is 0 by construction -- a rung that cannot fail.
    # Two rungs are therefore the minimum that can show a crossing, and with only
    # k = 1.0 and k = 1.25 below the reference, a case has ONE informative rung.
    # The coarse rung adds the second, at k < 1 (a coarser mesh: the target edge
    # is h_min / k), and it costs one extra build and no solve of the reference.
    # The run's own ladder (item 1): the default rungs, plus one finer rung if
    # asked. Everything downstream -- the reference, the convergence estimate
    # and the Richardson ratio the favourable bound is scaled by -- is derived
    # from THIS tuple, so the correction always matches the ladder that ran.
    factors = refine_factors(fine_reference)
    out["refine_factors"] = list(factors)
    rungs = {1.0} | set(factors) | ({COARSE_RUNG} if ladder else set())
    for k in rungs:
        try:
            _p, tets_u, kb_u = _uniform(case["materials"], obj, case, h_min,
                                        k=k, layers=layers, backend=backend)
            # The mesh this level actually produced. For a uniform-refined case
            # the reference IS this build, so it is checked HERE, while the
            # coordinates are in hand: a partially built or degenerate cloud is
            # visible in the array and nowhere else, and the check costs one
            # pass over the coordinates. Only the RESULT travels in the JSON --
            # the node cloud itself would be megabytes per level for nothing.
            mesh_ok = _looks_like_a_mesh(_p)
            if not mesh_ok:
                raise RuntimeError(
                    f"the k={k} build returned something that is not a finite "
                    f"(n, 3) node cloud ({type(_p).__name__}), so a reference "
                    f"built from it would be partial or degenerate")
            n_nodes = int(np.asarray(_p).shape[0])
            levels[k] = kb_u
            # The solver's own measured residual for THIS level, read from the
            # module state the pipeline itself reads (mse_eigen sets it at the
            # end of every solve). Captured per level, because the next level's
            # solve overwrites it.
            out["uniform"][str(k)] = {
                "n_tets": int(tets_u.shape[0]),
                "n_nodes": n_nodes,
                "mesh_ok": mesh_ok,
                "worst_residual": _last_residual()}
        except Exception as exc:  # noqa: BLE001 - a failed reference is data
            uniform_error = f"at h_min/{k}: {type(exc).__name__}: {exc}"
            out["uniform"][str(k)] = {"error": f"{type(exc).__name__}: {exc}"}
    if 1.0 not in levels:
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the uniform solve raised: {uniform_error}",
                "case": out}

    # THE REFERENCE.
    meta: dict = {"reference_levels": {k: v for k, v in out["uniform"].items()}}
    if case["reference"] == "analytic":
        a_eff, _b_eff, ab_eff = BENCH._effective_radii(
            parts, case["materials"].get("core"), units="mm")
        _n, spectrum = BENCH._analytic(case["materials"].get("core"), ab_eff)
        reference = [(float(k), int(l), f, int(m)) for k, l, f, m in spectrum]
        out["a_over_b_eff"] = ab_eff
    else:
        finest = factors[-1]
        if finest not in levels:
            return {"label": case["label"], "verdict": "UNMEASURED",
                    "detail": f"the reference could not be built: {uniform_error}",
                    "case": out, "meta": meta}
        reference = _clusters_pairs(levels[finest])
        coarser = factors[0]
        if coarser in levels:
            conv = REF.score_multiplets(levels[coarser], reference,
                                        n_requested=n_modes)
            meta["reference_convergence"] = conv.get("worst_rel_error")

    # THE REFERENCE GATE. A reference that does not cover the asked-for modes is
    # not a weaker reference, it is a different question: the scorer matches
    # what it can and reports the rest as unmatched, so the verdict would read
    # "graded beat uniform against one target" while the case was set up for
    # five. Counting the covered modes catches that for both kinds of reference
    # (a half-built uniform mesh and a truncated spectrum look the same to the
    # scorer), and for the built kind the mesh itself is checked as well -- a
    # partial build that still yields enough clusters would otherwise pass.
    reachable = _full_spectrum(reference, n_modes)
    covered = sum(int(m) for _k, _l, _f, m in reachable)
    gate = {"kind": case["reference"], "n_entries": len(reference),
            "n_reachable": len(reachable), "modes_covered": covered,
            "modes_asked": n_modes}
    if case["reference"] != "analytic":
        gate["reference_mesh_ok"] = (out["uniform"]
                                     .get(str(factors[-1])) or {}) \
            .get("mesh_ok")
        if gate["reference_mesh_ok"] is False:
            gate["incomplete"] = ("the build the reference came from is not a "
                                  "finite (n, 3) node cloud")
    if covered < n_modes:
        gate["incomplete"] = (
            f"the reference covers {covered} of the {n_modes} asked-for modes"
            f"{' -- a partially built reference' if case['reference'] != 'analytic' else ''}")
    meta["reference_gate"] = gate
    if gate.get("incomplete"):
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": "the reference is incomplete: " + gate["incomplete"],
                "case": out, "meta": meta}

    graded = REF.score_multiplets(kb_graded, reference, n_requested=n_modes)
    uniform = REF.score_multiplets(levels[1.0], reference, n_requested=n_modes)
    # ``matched`` is kept as well as the summary counts: the per-multiplet
    # SIGNED errors (item 2) are recomputed from these pairs, and a derived
    # table whose source pairs were dropped from the artifact cannot be checked
    # against its own evidence. The pairs cost a handful of floats.
    out["graded_score"] = {k: graded[k] for k in
                           ("n_matched", "n_clusters", "n_returned",
                            "worst_rel_error", "matched") if k in graded}
    out["uniform_score"] = {k: uniform[k] for k in
                            ("n_matched", "n_clusters", "n_returned",
                             "worst_rel_error", "matched") if k in uniform}

    # The uniform side's PER-REGION build, beside the graded side's. The bench's
    # uniform mesh is one part with no regions (see ``_uniform``), so on its own
    # it cannot say what edge each region came out at, and two of the six Orin
    # diagnosis items are exactly that question. This is a SECOND mesh and no
    # solve: the same region-aware call the graded side makes, with every
    # region's target pinned to h_min -- the rule the product meshed by BEFORE
    # grading (mse_meshing._region_plan's docstring). It is a bench-side patch
    # of ``mse_common.region_edges``; the product's rule is not changed.
    try:
        u_tets, u_reg, u_report = _uniform_regions(case, obj, h_min)
        out["uniform_regions"] = u_report or {}
        out["n_tets_uniform_region_aware"] = int(u_tets.shape[0])
    except Exception as exc:  # noqa: BLE001 - a failed side build is data
        out["uniform_regions_error"] = f"{type(exc).__name__}: {exc}"

    if bands and case["axes"]["shape"] == "concentric":
        try:
            import mse_meshing as MESH
            merged = SOLVER.PARTS.merged_surface(parts)
            gp, gt, gr = MESH.tetrahedralize(
                merged, "mm", n_modes=n_modes, n_index=1.0, max_tets=600_000,
                min_feature_m=None, verbose=False, materials=case["materials"],
                parts=parts)
            fine_region = int(((modes.get("mesh") or {}).get("fine_region")))
            a_eff, _b, _ab = BENCH._effective_radii(
                parts, case["materials"].get("core"), units="mm")
            out["bands"] = BENCH._band_edges_mm(gp, gt, float(a_eff),
                                                region_of_tet=gr,
                                                fine_region=fine_region)
        except Exception as exc:  # noqa: BLE001
            out["bands_error"] = f"{type(exc).__name__}: {exc}"

    # SIGNED per-mode errors for BOTH sides, and the zero crossings among them.
    # The scorer returns magnitudes; two errors on opposite sides of the
    # reference are indistinguishable in a magnitude, so the sign is recovered
    # from the pairs it kept. See ``signed_pairs``.
    rows = per_mode_table(graded, uniform)
    out["modes_errors"] = rows
    crossings = zero_crossings(rows)
    if crossings:
        out["zero_crossings"] = [
            {"l": r["l"], "family": r["family"], "reference_kb": r["reference_kb"],
             "graded_signed_rel": r.get("graded_signed_rel"),
             "uniform_signed_rel": r.get("uniform_signed_rel")}
            for r in crossings]

    # The refinement ladder the uniform side ALREADY builds, scored. It is the
    # only check on whether these errors track discretisation at all, and it
    # costs no solve -- every level was built for the reference.
    errors = level_errors(levels, reference, n_modes)
    out["uniform_level_errors"] = errors
    # The same ladder with the reference taken out (item 2, step 2c). An
    # analytic reference cannot be tightened by refining it, so a non-falling
    # error ladder has to be attributed: either the mesh is not converging, or
    # the mesh IS converging and the reference describes a different object.
    # This separates the two from the spectra already in hand.
    out["uniform_self_ladder"] = self_ladder(levels, n_modes)

    # The tolerance, from the solver's own measured residual on each side: two
    # meshes whose scores differ by less than the solver's noise have not been
    # shown to differ. The sum is used because the comparison subtracts two
    # independent solves, so their errors add; a missing residual on EITHER
    # side leaves the tolerance unavailable rather than zero, and the run says
    # so, because a strict comparison presented as a justified one is worse than
    # no tolerance at all.
    residuals = [graded_res := modes.get("solver_worst_residual"),
                 out["uniform"].get("1.0", {}).get("worst_residual")]
    tol = (sum(r for r in residuals) if all(isinstance(r, float) for r in residuals)
           else 0.0)
    out["tolerance"] = tol
    out["tolerance_from_residuals"] = {
        "graded": graded_res,
        "uniform": out["uniform"].get("1.0", {}).get("worst_residual")}
    if not tol:
        out["tolerance_unavailable"] = (
            "no measured residual on "
            + ", ".join(t for t, r in zip(("graded", "uniform"), residuals)
                        if not isinstance(r, float))
            + " (the CPU factorisation records none), so the comparison is "
              "strict")

    # Item 3: judge on BUILT counts, at the printed mark. The estimator
    # overshoots (c4-thin: 161,389 predicted against 61,244 built), so a case
    # that only the estimate puts over the mark is measured, not refused.
    built_sides = {"graded": out.get("n_tets_graded"),
                   "uniform": out["uniform"].get("1.0", {}).get("n_tets")}
    over = [tag for tag, n in built_sides.items()
            if n is not None and n > TIER2_MAX_TETS]
    out["built_sides"] = built_sides

    # Item 1: same mesh. The lead's rule is "same plan AND same tet count", and
    # both halves are needed. A uniform PLAN with different counts is a real
    # difference (the two build paths disagree even at one edge -- the estimator
    # finding), while a graded plan that happened to land on the same count is
    # still graded. Only both together mean grading did nothing here.
    out["plan_uniform"] = plan_is_uniform(plan)
    graded_tets = built_sides.get("graded")
    uniform_tets = built_sides.get("uniform")
    same_mesh = bool(out["plan_uniform"]) and graded_tets is not None \
        and graded_tets == uniform_tets
    out["same_mesh"] = same_mesh

    # A side with no number says which side it is and what it produced, rather
    # than "one of the two worst errors is missing".
    reason = (missing_side_reason("the GRADED mesh", graded, reference)
              or missing_side_reason("the UNIFORM mesh", uniform, reference))
    verdict, detail = criterion(graded.get("worst_rel_error"),
                                uniform.get("worst_rel_error"), reason,
                                same_plan=same_mesh,
                                tol=tol, sides_over_mark=over,
                                reference_convergence=meta.get("reference_convergence"))
    # A verdict that cannot be trusted is not a verdict. It is kept as the
    # provisional reading and NOT reported as PASS: the case is UNRELIABLE.
    n_targets = reference_multiplets_in_reach(reference, n_modes)
    out["reference_multiplets_in_reach"] = n_targets
    unreliable = reliability(errors, n_targets, rows,
                             selflad=out["uniform_self_ladder"],
                             reference_kind=case["reference"])
    out["reliability"] = unreliable
    if unreliable and verdict in ("PASS", "FAIL"):
        out["provisional_verdict"] = verdict
        verdict, detail = ("UNRELIABLE",
                           f"the comparison reads {verdict} ({detail}) but the "
                           f"case cannot carry it: {unreliable}")
    # Item 5's bound, recorded with the verdict it is a bound on. ``None`` for
    # an analytic reference (an exact one has no convergence to give away).
    bound = favourable_bound(uniform.get("worst_rel_error"),
                             graded.get("worst_rel_error"),
                             meta.get("reference_convergence"),
                             richardson=richardson_scale(factors))
    if bound is not None:
        out["favourable_uniform_bound"] = bound
    return {"label": case["label"], "verdict": verdict, "detail": detail,
            "graded_worst": graded.get("worst_rel_error"),
            "uniform_worst": uniform.get("worst_rel_error"),
            "meta": meta, "case": out}


# ---------------------------------------------------------------------- --list


def predicted_counts(case) -> dict:
    """Predicted element count for every mesh a case would build.

    The numbers are the product's own estimator (``web.meshing.estimate_tets``,
    the function that ROUTES an upload), not a re-derivation, so a case that
    would route to the GPU here is the case that would route there in
    production. Its constant comes from ``web.config.CFG.est_tets_c()`` -- the
    same value ``web/app.py`` routes on -- rather than a number written here,
    because a constant chosen in a benchmark is the fixture-fitted tuning this
    row exists to rule out. The refined levels are that estimate scaled by
    ``k**3`` -- the exact factor when the only thing changing is the edge.
    """
    import meshing as WEBMESH
    import config as CFG      # app.py:34 imports it the same way

    built = _build(case, case["n_modes"])
    parts, plan = built["parts"], built["plan"]
    h_min, probe_h_min = built["h_min_mm"], built["probe_h_min_mm"]
    outer = _outer_surface(parts)
    c = CFG.est_tets_c()
    counts = {
        "graded": int(WEBMESH.estimate_tets(outer, case["n_modes"], "mm", c=c,
                                            parts=parts,
                                            materials=case["materials"])),
        "uniform": int(WEBMESH.estimate_tets(outer, case["n_modes"], "mm", c=c,
                                             min_feature_m=h_min * 1e-3)),
    }
    probe_uniform = int(WEBMESH.estimate_tets(
        outer, case["n_modes"], "mm", c=c, min_feature_m=probe_h_min * 1e-3))
    if case["reference"] == "uniform-refined":
        for k in REFINE_FACTORS:
            counts[f"ref{k}"] = int(round(counts["uniform"] * k ** 3))
    return {"plan": plan, "h_min_mm": h_min, "probe_h_min_mm": probe_h_min,
            "probe_uniform_tets": probe_uniform,
            "baseline_ratio": probe_uniform / max(counts["uniform"], 1),
            "counts": counts,
            "over_ceiling": sorted(k for k, n in counts.items()
                                   if n > TIER2_MAX_TETS)}


def _read_built(results_dir):
    """``{label: {"graded": n, "uniform": n, "source": path}}`` from run JSONs.

    Only ``run_case``'s own output counts, and only counts that came from a
    BUILD (``n_tets_graded`` / ``uniform['1.0'].n_tets``) -- an estimate read
    back as a built count would be the estimator checked against itself, which
    is exactly the comparison this table exists to replace.
    """
    built = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - an unreadable result is a gap
            continue
        cases = d if isinstance(d, list) else [d]
        for c in cases:
            if not isinstance(c, dict):
                continue
            label = c.get("label") or (c.get("case") or {}).get("label")
            body = c.get("case") if isinstance(c.get("case"), dict) else c
            if not label or not isinstance(body, dict):
                continue
            n_g = body.get("n_tets_graded")
            n_u = (body.get("uniform") or {}).get("1.0", {}).get("n_tets")
            if n_g is None and n_u is None:
                continue
            built[label] = {"graded": n_g, "uniform": n_u,
                            "source": path.name}
    return built


def estimator_table(cases, results_dir=None) -> int:
    """Item 5: the estimator's two paths against BUILT counts, one row a case.

    ``web.meshing.estimate_tets`` is what the live site ROUTES on (``web/app.py``:
    GPU above 15,000, refuse above 1,200,000), so a case where the estimate and
    the mesher disagree by a large factor is a routing finding about the product,
    not a bench artefact. Reported, not changed: the lead's instruction was to
    report it.

    The two estimate columns are the two API paths, and they are different
    functions of the input -- ``materials=<parts>`` routes through the
    region-aware plan, ``min_feature_m=`` through the legacy single-feature
    floor -- so where they disagree the estimate does not depend on the mesh
    alone but on which call the caller happens to make.
    """
    import meshing as WEBMESH
    import config as CFG

    built = _read_built(results_dir) if results_dir else {}
    c = CFG.est_tets_c()
    # The thresholds are READ from the product, never retyped: web/app.py:1136
    # routes on CFG.gpu_min_tets() and web/app.py:1127 refuses above
    # CFG.upload_refuse_tets(), so a bench constant here would be a second
    # source of truth for a routing decision the product makes.
    #
    # The app's own comment (web/config.py:248) says the refuse threshold sits at
    # twice the cap "because the estimate can be up to 2x high". The table below
    # is the test of that assumption, and the warning line reports every case
    # that breaks it.
    print(f"live routing (read from config): route to GPU at or above "
          f"{CFG.gpu_min_tets():,}; refuse an upload above "
          f"{CFG.upload_refuse_tets():,} (= 2x the real cap "
          f"{CFG.max_tets():,}); tier-2 mark {TIER2_MAX_TETS:,} (built counts "
          f"judge that, not estimates)")
    print(f"{'case':>13} {'est(materials=)':>15} {'est(min_feature_m=)':>19} "
          f"{'built graded':>13} {'built uniform':>14} {'built/est':>9}  source")
    incomplete = 0
    for case in cases:
        try:
            built_case = _build(case, case["n_modes"])
            outer = _outer_surface(built_case["parts"])
            est_m = int(WEBMESH.estimate_tets(
                outer, case["n_modes"], "mm", c=c,
                parts=built_case["parts"], materials=case["materials"]))
            est_f = int(WEBMESH.estimate_tets(
                outer, case["n_modes"], "mm", c=c,
                min_feature_m=built_case["h_min_mm"] * 1e-3))
        except Exception as exc:  # noqa: BLE001 - a failed estimate is data
            print(f"{case['label']:>13}   --- ESTIMATE FAILED: "
                  f"{type(exc).__name__}: {exc}")
            incomplete += 1
            continue
        b = built.get(case["label"]) or {}
        g, u = b.get("graded"), b.get("uniform")
        ratio = (f"{g / est_m:.2f}x" if g and est_m else "----")
        over = []
        if g and g > TIER2_MAX_TETS:
            over.append("built graded")
        if u and u > TIER2_MAX_TETS:
            over.append("built uniform")
        print(f"{case['label']:>13} {est_m:>15,} {est_f:>19,} "
              f"{(f'{g:,}' if g else 'not built'):>13} "
              f"{(f'{u:,}' if u else 'not built'):>14} {ratio:>9}  "
              f"{b.get('source', 'no built counts here')}")
        if over:
            print(f"{'':>13}   ! over the tier-2 mark {TIER2_MAX_TETS:,} on "
                  f"{' and '.join(over)} -- UNMEASURED whatever the estimate "
                  f"said ({est_m:,} materials=, {est_f:,} min_feature_m=)")
        if u and est_f and abs(est_f / u - 1.0) > 0.25:
            print(f"{'':>13}   ! the min_feature_m= estimate is "
                  f"{est_f / u:.2f}x the BUILT uniform count ({est_f:,} vs "
                  f"{u:,}); the bench builds its uniform side by this path")
        # The documented bound is 2x (web/config.py:248). Anything above it is a
        # case the refuse threshold was not sized for.
        for tag, est_v, built_v in (("materials=", est_m, g),
                                    ("min_feature_m=", est_f, u)):
            if built_v and est_v > 2.0 * built_v:
                print(f"{'':>13}   ! the {tag} estimate is "
                      f"{est_v / built_v:.2f}x the BUILT count ({est_v:,} vs "
                      f"{built_v:,}) -- above the 2x the refuse threshold "
                      f"(web/config.py:248) is documented to allow")
    if not built:
        print("\nno built counts: pass --results DIR (a directory of this "
              "bench's own --json output) to compare the estimate with what "
              "the mesher actually built. The two estimate columns need no "
              "solve; the built columns do, so they come from a run's files.")
    return MATRIX_INCOMPLETE if incomplete else MATRIX_OK


def list_cases(cases, as_json=False) -> int:
    rows, incomplete = [], 0
    for case in cases:
        try:
            pred = predicted_counts(case)
        except Exception as exc:  # noqa: BLE001 - a failed estimate is data
            pred = {"error": f"{type(exc).__name__}: {exc}"}
            incomplete += 1
        rows.append({"label": case["label"], "reference": case["reference"],
                     "n_modes": case["n_modes"], "axes": case["axes"],
                     "note": case.get("note", ""), **pred})
    if as_json:
        print(json.dumps(rows, indent=2))
        return MATRIX_INCOMPLETE if incomplete else MATRIX_OK
    # BOTH floors, in mm, next to each other: item 3 is exactly "which base edge
    # is each side on", and a floor printed in a count's place (or a count in a
    # floor's) is the error this row is about. The old probe's uniform COUNT
    # stays in the JSON and in the warning line below, where it is labelled.
    print(f"{'case':>13} {'ref':>6} {'reg':>4} {'modes':>5} "
          f"{'h_min(prod)':>11} {'h_min(probe)':>12} {'graded':>9} "
          f"{'uniform':>9} {'len-rat':>7}  axes")
    deviating = []
    for r in rows:
        if "error" in r:
            print(f"{r['label']:>13} {r['reference']:>6}   --- COUNT FAILED: "
                  f"{r['error']}")
            continue
        c = r["counts"]
        extra = ("" if r["reference"] == "analytic"
                 else f"  refs {[c[f'ref{k}'] for k in REFINE_FACTORS]}")
        # The floors are LENGTHS; the counts are volumes, so a count ratio is
        # the length ratio CUBED. The column and the warning are in length,
        # because that is the quantity the two meshers are given -- an earlier
        # version printed the count ratio in a length column and read 0.09x
        # where the length is 2.23x (box-sphere).
        lr = (r["probe_h_min_mm"] / r["h_min_mm"]) if r["h_min_mm"] else float("nan")
        print(f"{r['label']:>13} {r['reference']:>6} {r['axes']['regions']:>4} "
              f"{r['n_modes']:>5} "
              f"{r['h_min_mm']:>11.4f} {r['probe_h_min_mm']:>12.4f} "
              f"{c['graded']:>9,} {c['uniform']:>9,} {lr:>7.2f}  "
              f"{r['axes']['shape']}, {r['axes']['shell']}, "
              f"n={r['axes']['contrast']}{extra}")
        if r["over_ceiling"]:
            print(f"{'':>13}   ! over the tier-2 mark {TIER2_MAX_TETS:,}: "
                  f"{r['over_ceiling']} -- expect UNMEASURED, not a pass")
        if abs(lr - 1.0) > 0.05:
            deviating.append((r["label"], lr))
            print(f"{'':>13}   ! base edge {r['probe_h_min_mm']:.4f}mm is "
                  f"{lr:.2f}x the product's {r['h_min_mm']:.4f}mm here, so this "
                  f"case's uniform side is {r['probe_uniform_tets']:,} tets on "
                  f"the old baseline and {c['uniform']:,} on the corrected one")
    if deviating:
        print(f"\n{len(deviating)} of {len(rows)} cases: the old probe's base "
              f"edge differs from the product's by more than 5% "
              f"({min(x[1] for x in deviating):.2f}x to "
              f"{max(x[1] for x in deviating):.2f}x). Verdicts measured before "
              f"2026-10-14 are against that other baseline.")
    print("\naxes covered: "
          + "; ".join(f"{k}={sorted(v, key=str)}" for k, v in
                      _coverage(cases).items()))
    print(f"tier-2 mark is {TIER2_MAX_TETS:,} elements. " + _routing_note())
    return MATRIX_INCOMPLETE if incomplete else MATRIX_OK


def _routing_note() -> str:
    """Production's own ceilings, so the list says which cases can never ship.

    Above ``upload_refuse_tets`` the upload route refuses the job outright, so a
    case past it would never be graded in production no matter how it scores.
    Read from the same config the route reads; a failure to read it is printed
    rather than skipped, because a missing ceiling must not read as no ceiling.
    """
    import config as CFG
    try:
        return (f"production refuses an upload above {CFG.upload_refuse_tets():,} "
                f"and sends one above {CFG.gpu_min_tets():,} to the GPU")
    except Exception as exc:  # noqa: BLE001 - report the read failure
        return f"PRODUCTION CEILINGS UNAVAILABLE ({type(exc).__name__}: {exc})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="print the case list with predicted element counts")
    p.add_argument("--case", default=None, help="run ONE case by label")
    p.add_argument("--estimator", action="store_true",
                   help="item 5: the estimator's two paths vs BUILT counts, "
                        "one row per case (no solves; pair with --results)")
    p.add_argument("--results", default=None,
                   help="a directory of this bench's own --json output; its "
                        "build counts fill the BUILT columns of --estimator")
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--backend", default="gpu", choices=("cpu", "gpu"))
    p.add_argument("--json", default=None, help="write the case result here")
    p.add_argument("--work", default=None)
    p.add_argument("--ladder", action="store_true",
                   help="also build and solve ONE coarser rung (h_min/0.8) per "
                        "case, so the uniform side's error can be seen from both "
                        "sides of the reference and a zero crossing is "
                        "measurable rather than argued (item 2)")
    p.add_argument("--fine-reference", type=float, default=None,
                   metavar="K", help=("add ONE reference rung at h_min/K, finer "
                                      "than the default finest (a tighter "
                                      "reference for a uniform-refined case; "
                                      "must be > 1.5). The rung is BUILT, and "
                                      "cost scales ~K^3: on the two sphere cases "
                                      "k=1.75 is ~130k/143k tets but k=2.0 is "
                                      "~232k/279k, over the 200k tier-2 mark, "
                                      "so 1.75 is the affordable tightening. "
                                      "Applies to a case RUN; --list sizes the "
                                      "default ladder and says so"))
    p.add_argument("--baseline", default="product", choices=("product", "probe"),
                   help="resolution of the uniform side: 'product' = the "
                        "finest edge the product's own plan asks for (the rule "
                        "it meshed by before grading; the corrected baseline), "
                        "'probe' = the older probe-derived floor, kept so the "
                        "numbers measured before 2026-10-14 reproduce")
    p.add_argument("--bands", action="store_true",
                   help="also report the per-band achieved edges (concentric "
                        "cases; costs one extra mesh build)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        if args.fine_reference is not None:
            # --list sizes the DEFAULT ladder only, and says so rather than
            # absorbing the flag: its refs column comes from REFINE_FACTORS, so a
            # silently ignored --fine-reference would let a reader size the
            # requested rung off the wrong row -- and an ESTIMATED count for it
            # would be worse than none (the estimator misses by up to 1.4x, 5.8).
            print(f"note: --list sizes the default ladder "
                  f"({', '.join(f'{k:g}' for k in REFINE_FACTORS)}); "
                  f"--fine-reference {args.fine_reference:g} is a case-run rung "
                  f"and is NOT in this table -- its BUILT count comes from the "
                  f"run, not the estimate")
        return list_cases(MATRIX)
    if args.estimator:
        return estimator_table(MATRIX, results_dir=args.results)
    labels = [c["label"] for c in MATRIX]
    if not args.case:
        print("nothing to do: pass --list, --estimator or --case <label>",
              file=sys.stderr)
        return MATRIX_USAGE
    if args.case not in labels:
        print(f"unknown case {args.case!r}; known: {labels}", file=sys.stderr)
        return MATRIX_USAGE
    case = next(c for c in MATRIX if c["label"] == args.case)
    work = (Path(args.work) if args.work
            else Path(tempfile.mkdtemp(prefix=f"mse-matrix-{args.case}-")))
    print(f"[matrix] work root: {work}")
    print(f"[matrix] case {case['label']}: {case['note'] or case['axes']}")
    try:
        result = run_case(case, grid=args.grid, backend=args.backend, work=work,
                          verbose=not args.quiet, bands=args.bands,
                          baseline=args.baseline, ladder=args.ladder,
                          fine_reference=args.fine_reference)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[matrix] case RAISED: {type(exc).__name__}: {exc}")
        return MATRIX_INCOMPLETE
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"[matrix] {result['verdict']}: {result['detail']}")
    out = result.get("case") or {}
    if out.get("n_tets_graded"):
        us = out.get("uniform", {}).get("1.0", {}) or {}
        print(f"[matrix] graded {out['n_tets_graded']:,} tets; uniform "
              f"{us.get('n_tets') or 'unmeasured'} tets"
              + ("" if out.get("n_tets_uniform_region_aware") is None else
                 f"; uniform side rebuilt region-aware "
                 f"{out['n_tets_uniform_region_aware']:,} tets"))
        print(f"[matrix] baseline {out.get('baseline')}: h_min "
              f"{out.get('h_min_mm', float('nan')):.4f} mm "
              f"(product {out.get('h_min_mm_product', float('nan')):.4f}, "
              f"probe {out.get('h_min_mm_probe', float('nan')):.4f})")
        for tag, regions in (("graded", out.get("graded_regions") or {}),
                             ("uniform",
                              (out.get("uniform_regions") or {}).get("regions")
                              or {})):
            for rid, reg in (regions or {}).items():
                tgt = reg.get("target_edge_units")
                ach = reg.get("achieved_edge_units")
                print(f"[matrix]   {tag} region {rid}: asked "
                      f"{_num(tgt)} units, built {_num(ach)}"
                      + (", CLAMPED" if reg.get("clamped") else ""))
        if out.get("uniform_regions_error"):
            print(f"[matrix]   uniform per-region build FAILED: "
                  f"{out['uniform_regions_error']}")
    # Item 1: say it in words, not only as an exit code -- "the plan was uniform
    # so both sides are the same mesh" is the finding, and a verdict string
    # alone does not carry it.
    if out.get("plan_uniform") is not None:
        print(f"[matrix] plan uniform: {bool(out['plan_uniform'])}"
              + (f"; same mesh as the uniform side: {out['same_mesh']} "
                 f"({out.get('built_sides', {}).get('graded'):,} tets both)"
                 if out.get("same_mesh") else ""))
    # Item 2: the signed per-multiplet errors, both sides, and the crossings.
    rows_m = out.get("modes_errors") or []
    if rows_m:
        print("[matrix] per-multiplet SIGNED error (graded / uniform):")
        for r in rows_m:
            g = r.get("graded_signed_rel")
            u = r.get("uniform_signed_rel")
            print(f"[matrix]   l={r['l']} {r['family']} ref "
                  f"{_num(r['reference_kb'])} GHz: "
                  + (f"graded {g * 100:+.4f}%" if g is not None
                     else "graded not matched")
                  + "  |  "
                  + (f"uniform {u * 100:+.4f}%" if u is not None
                     else "uniform not matched"))
        for c in (out.get("zero_crossings") or []):
            print(f"[matrix]   ZERO CROSSING at ref {_num(c['reference_kb'])} GHz "
                  f"(l={c['l']} {c['family']}): graded "
                  f"{_num(c['graded_signed_rel'])} vs uniform "
                  f"{_num(c['uniform_signed_rel'])} -- magnitudes here are not "
                  "comparable")
    ladder = out.get("uniform_level_errors") or []
    if ladder:
        print("[matrix] uniform ladder (worst error vs the reference): "
              + ", ".join(f"k={e['k']:g} "
                          + (f"{_num(e['worst_rel_error'])}"
                             if e.get('worst_rel_error') is not None
                             else f"unmatched({e.get('n_matched')})")
                          for e in ladder))
    # Item 2, step 2c: the same ladder with the reference removed, so a
    # non-falling error can be attributed to the mesh or to the reference.
    selflad = out.get("uniform_self_ladder") or []
    if selflad:
        print("[matrix] uniform SELF-ladder (worst error against the finest "
              "rung's own spectrum): "
              + ", ".join(f"k={e['k']:g}->{e['against']:g} "
                          + (f"{_num(e['self_rel_error'])}"
                             if e.get('self_rel_error') is not None
                             else f"unmatched({e.get('n_matched')})")
                          for e in selflad))
    if out.get("tolerance") is not None and out.get("n_tets_graded"):
        if out.get("tolerance"):
            print(f"[matrix] tolerance from the solver's measured residual: "
                  f"{out['tolerance']:.3e} (graded "
                  f"{_num(out['tolerance_from_residuals']['graded'])}, uniform "
                  f"{_num(out['tolerance_from_residuals']['uniform'])})")
        else:
            print(f"[matrix] tolerance: none available -- "
                  f"{out.get('tolerance_unavailable')}")
    # The reason a verdict cannot be trusted prints ONCE. On UNRELIABLE it is
    # already the tail of the verdict line above (the verdict string carries it
    # to the JSON too, where a lone "reliability" field would be a second place
    # to look); on NO VERDICT, where the verdict line has a different reason,
    # this line is its only carrier.
    if out.get("reliability") and result.get("verdict") != "UNRELIABLE":
        print(f"[matrix] verdict reliability: {out['reliability']}")
    fb = out.get("favourable_uniform_bound")
    # A duplicated number reads as two measurements. The favourable line below
    # states this same convergence inside its arithmetic, so the standalone
    # line is only printed when there is no such line to state it (a missing
    # worst error leaves the bound with an operand missing).
    if (fb is None
            and result.get("meta", {}).get("reference_convergence") is not None):
        print("[matrix] reference's own convergence (coarse vs finest): "
              f"{result['meta']['reference_convergence'] * 100:.3f}%")
    if fb:
        print("[matrix] uniform side's most favourable reading: "
              f"{fb['uniform_worst'] * 100:.3f}% + the reference's whole "
              f"convergence {fb['reference_convergence'] * 100:.3f}% = "
              f"{fb['uniform'] * 100:.3f}%, against graded "
              f"{fb['graded_worst'] * 100:.3f}%"
              + (" -- STILL FAIL" if fb["still_fail"] else " -- the verdict "
                 "does depend on the reference's quality"))
        # The coarse-vs-finest difference UNDERSTATES the finest level's own
        # error by 1 - r**-p, so the line above is a floor on the correction,
        # not the correction. Both are printed; only the run's author chooses.
        _rf = out.get("refine_factors") or list(REFINE_FACTORS)
        print(f"[matrix]   scaled by Richardson (p = 2, ratio "
              f"{_rf[-1] / _rf[0]:g} -> x{fb['richardson']:.2f}): "
              f"{fb['richardson_uniform'] * 100:.3f}%, against graded "
              f"{fb['graded_worst'] * 100:.3f}% = "
              f"{fb['richardson_ratio']:.2f}x"
              + (" -- STILL FAIL" if fb["richardson_still_fail"]
                 else " -- FLIPS: not decidable from these runs"))
    if out.get("bands"):
        print("[matrix] bands: " + "  ".join(
            f"{k} {v['edge_mm']:.4f} mm" for k, v in out["bands"].items()))
    return {"PASS": MATRIX_OK, "FAIL": MATRIX_FAIL,
            "UNMEASURED": MATRIX_INCOMPLETE, "SAME MESH": MATRIX_SAME_MESH,
            "UNRELIABLE": MATRIX_UNRELIABLE,
            "NO VERDICT": MATRIX_NO_VERDICT}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
