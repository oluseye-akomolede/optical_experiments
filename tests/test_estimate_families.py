"""The upload estimate against REAL meshes, across the shape families.

Claim under test, in two parts, because the measurement does not support the
one-part version:

  * on the six families whose meshes land at or near the volume law,
    ``estimate_tets`` reads AT OR ABOVE the mesh TetGen really builds, and never
    more than ``config.EST_TETS_WORST_OVER`` times it;
  * on the two that do not -- nested thin sheets, where TetGen over-refines 3x
    below the edge it was asked for, and a bumpy sphere at half the mode counts,
    where the surface's own facets floor the delivery -- the estimate reads BELOW
    the real mesh, and the test pins each under-read with its measured number
    instead of hiding it, because that is the honest state of the product: the
    first is the finding the lead parked as a mesher issue, the second is the
    counterexample to ``EST_TETS_MARGIN`` (see the change's proposal). If a
    mesher fix lands, these redden and say to re-measure: an under-read going away
    is not a green here, it is a change to the constants.

The families are the ones the fix was scoped to: a single part, two and three
regions, a thin and a thick wall, an off-centre core and a non-spherical part.

Why these fixtures are MESHED and not doubles: the estimate's whole job is to
predict a number that exists only after TetGen runs, and the quantity that moves
that number (how far TetGen lands below the volume law, through
``TETGEN_VOLUME_SLACK`` and its quality constraints) is a property of the
MESHER. A test against a hand-written count would keep passing while the mesher
changed under it. These redden when the mesher does -- which is the point: the
two constants they pin are measurements OF the mesher, so a mesher change means
they have to be re-measured, and a red test is how that gets noticed.

What the old formula did on these meshes is asserted here too, recomputed from
the SAME real mesh (``_replaced_formula_estimate``), so the before/after differ
by the formula and by nothing else.

The fixtures are meshed AT THE PRODUCT'S OWN CAP (``config.max_tets()``), not at
a smaller literal: the mesher's over-cap retry coarsens ``maxvolume`` and
rebuilds, so a lower cap measures a different mesh than a real upload gets. That
is not hypothetical -- measuring these same fixtures at 60,000 made two families
look ~8x better than they are and produced a step-1 result (a 3.5x finer core
reading 10.86x) that the product never sees.

Tier 1: seven of the eight fixtures are thousands of elements each (the largest
of those is the 2-region core at ~24,000); the eighth (nested thin sheets, the
under-read) is the one family that builds hundreds of thousands -- 419,176,
measured -- and it is here because it is the family that matters. Each is meshed
once per run and cached. The sweep that measured the constants is
``benchmarks/estimate_vs_built.py --families``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import config as CFG  # noqa: E402
import meshing as WEB  # noqa: E402
import mse_common as COMMON  # noqa: E402
import mse_meshing as MESHSOLVER  # noqa: E402
import mse_parts as PARTS  # noqa: E402

#: the fitted coefficient this change replaced, kept here ONLY to recompute what
#: the old formula would have said on the same real mesh. It is not read by the
#: product any more; a test that reads the old constant from the product would
#: stop being able to state that, which is the whole point of the comparison.
REPLACED_C = 14.0

#: The cap the PRODUCT meshes under (``web.app`` passes ``CFG.max_tets()`` to the
#: solver, which passes it to ``tetrahedralize``). NOT a smaller literal: below a
#: fixture's unconstrained count the mesher's over-cap retry coarsens the
#: request and rebuilds, so a lower cap measures a mesh the product never makes.
MAX_TETS = CFG.max_tets()

#: family label -> its built row, so a family is meshed once per run.
_BUILT: dict[str, dict] = {}


#: The families whose mesh TetGen over-refines, so the estimate reads below it.
#: Each carries the two numbers measured at the product's cap: the ratio the
#: estimate reads (``est / built``, below 1) and the delivery in the margin's own
#: units (``built / volaw``, which the margin has to cover and does not). These
#: are kept out of the at-or-above assertion and pinned one at a time.
UNDER_READ_FAMILIES = {
    # ``three-region-shell-core`` was here under the UNIFORM mesher: TetGen
    # refined 3x below the requested edge, built ~9x the law, estimate read
    # 0.15x. Since graded-meshing-default it is meshed GRADED and moved to the
    # other side; its numbers are in GRADED_FAMILIES.
    # an icosphere with bumps at HALF the mode counts (a legitimate upload -- the
    # user picks the mode count): the surface's own facets floor the delivery at
    # 1.868x the law. The counterexample to the margin, and the worst over-read
    # in the other direction (mounting the same family the estimate reads 2.098x).
    # ONE part, so grading does not touch it: these are today's numbers too.
    "one-part-bumpy-sphere": (0.72, 1.868),
}

#: The nested families re-measured under the GRADED mesher, the product rule
#: since ``graded-meshing-default``. ``(est/built, built/law)``. The uniform
#: entries above are kept because ``MSE_GRADED_MESHING=0`` still builds those.
GRADED_FAMILIES = {
    # The found-issue family, graded. The sizing field holds the refinement at
    # the requested edges instead of 3x below them, so the delivery drops below
    # the law rather than 9x above it (8.952 -> 0.287) and the estimate now
    # reads ABOVE the real mesh (0.15 -> 1.177) -- one-sided on the cheap side,
    # which is the direction the margin is for. The estimate itself moved
    # 63,214 -> 15,791 on the same upload (both measured on 2026-11-02): the
    # uniform plan meshes the coarse wall at the core's fine edge, so its sum
    # prices ~4x a mesh nobody builds. That is the pricing bug this change
    # exists to fix, not noise.
    "three-region-shell-core": (1.177, 0.287),
}


def _box(extents):
    import trimesh

    return trimesh.creation.box(extents=extents)


def _sphere(radius, subdivisions=3, bump=0.0):
    import numpy as np
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    if bump:
        # The same perturbed sphere the sweep's ``_sphere(bump=...)`` builds, so
        # this fixture is the bench's row and not a lookalike. The radial factor
        # is one number per VERTEX, so it multiplies along axis 1 (3 coordinates).
        v = mesh.vertices
        factor = 1.0 + bump * np.sin(3.0 * v[:, 0]) * np.cos(2.0 * v[:, 1])
        mesh.vertices = v * factor[:, None]
    return mesh


def _obj(parts):
    """OBJ bytes for a named set of surfaces -- the upload's own convention."""
    import trimesh

    return trimesh.Scene(parts).export(file_type="obj").encode("utf-8")


#: (label, n_modes, {part name: (n_index, surface builder)}). One part is
#: meshed at 10 modes across the object; a nested one at 4, because a 2x finer
#: core is 8x the elements and these fixtures exist to be cheap. The bumpy
#: sphere is at 5: that is ``--family-scale-n 0.5`` on the bench, the setting
#: where its under-read (and, on the plate-shaped families, the worst over-read)
#: is largest, and it is a mode count the user can pick.
FAMILIES = {
    "one-part-cube": (10, {"body": (1.0, lambda: _box((6.0, 6.0, 6.0)))}),
    "one-part-plate": (10, {"body": (1.0, lambda: _box((8.0, 4.0, 2.0)))}),
    "one-part-sphere": (10, {"body": (1.0, lambda: _sphere(3.0))}),
    "one-part-bumpy-sphere": (5, {"body": (1.0, lambda: _sphere(3.0, bump=0.15))}),
    "two-region-core": (4, {"wall": (1.0, lambda: _box((6.0, 6.0, 6.0))),
                            "core": (2.0, lambda: _box((1.5, 1.5, 1.5)))}),
    "two-region-offcentre": (4, {"wall": (1.0, lambda: _box((6.0, 6.0, 6.0))),
                                 "core": (2.0, lambda: _box((2.0, 2.0, 2.0)).apply_translation(
                                     (1.2, 0.8, -1.0)))}),
    "two-region-thin-wall": (4, {"wall": (1.0, lambda: _box((6.0, 6.0, 6.0))),
                                 "core": (2.0, lambda: _box((5.2, 5.2, 5.2)))}),
    # The found-issue family: nested thin sheets, ~0.15x (see UNDER_READ_FAMILIES
    # for the mechanism). Measured at the product's cap; NOT a small fixture (see
    # the module docstring).
    "three-region-shell-core": (4, {"wall": (1.0, lambda: _box((6.0, 6.0, 6.0))),
                                    "shell": (2.0, lambda: _box((4.0, 4.0, 4.0))),
                                    "core": (3.0, lambda: _box((2.0, 2.0, 2.0)))}),
}

#: the families whose meshes land at or near the volume law -- the ones the
#: one-sided requirement is stated over. The under-read families are not here.
COMPLIANT_FAMILIES = sorted(f for f in FAMILIES if f not in UNDER_READ_FAMILIES)


def _build_family(label):
    """Estimate and mesh one family through the product's own calls.

    Cached: the 3-region fixture builds hundreds of thousands of tetrahedra and
    two tests ask for the same family, so the mesh is built once per run. The
    rows are read-only, which is what makes sharing them safe.

    The cache key carries WHICH mesher built the row. ``web.meshing`` reads
    grading from the environment, so a row built with grading ON and a row built
    with it OFF are different meshes of the same fixture -- and the graded
    estimate (15,791 on the nested fixture) is not the uniform one (63,214).
    Keying on the label alone would hand whichever test ran first to both.
    """
    graded = bool(COMMON.graded_meshing_enabled())
    key = (label, graded)
    if key in _BUILT:
        return _BUILT[key]

    row = _build_row(label)
    row["graded"] = graded
    _BUILT[key] = row
    return row


def _build_row(label):
    """Estimate and mesh one family; ``_build_family`` owns the cache."""
    import numpy as np

    n_modes, spec = FAMILIES[label]
    materials = {name: n_index for name, (n_index, _) in spec.items()}
    surfaces = {name: build() for name, (_, build) in spec.items()}
    surface_bytes = _obj(surfaces)

    if len(spec) == 1:
        surface, _ = MESHSOLVER.load_surface(surface_bytes, units="mm")
        est = WEB.estimate_tets(surface, n_modes, "mm", margin=CFG.est_tets_margin())
        volume_m3 = COMMON.enclosure_volume_m3(surface, "mm")
        h_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm")
        _points, tets, _regions = MESHSOLVER.tetrahedralize(
            surface, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False)
        return dict(n_modes=n_modes, est=est, built=int(np.asarray(tets).shape[0]),
                    h_m=h_m, volume_m3=volume_m3)

    parts = PARTS.load_parts(surface_bytes, units="mm")
    merged = PARTS.merged_surface(parts)
    est = WEB.estimate_tets(merged, n_modes, "mm", margin=CFG.est_tets_margin(),
                            parts=parts, materials=materials)
    edge_m = COMMON.effective_edge_m(merged, n_modes=n_modes, units="mm",
                                     parts=parts, materials=materials)
    plan = COMMON.region_plan(parts, materials, edge_m, "mm")
    h_m = min(float(info["edge_units"]) for info in plan.values())
    volume_m3 = sum(float(info["filled_volume_m3"]) for info in plan.values())
    _points, tets, _regions = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False,
        materials=materials, parts=parts)
    return dict(n_modes=n_modes, est=est, built=int(np.asarray(tets).shape[0]),
                h_m=h_m, volume_m3=volume_m3, parts=parts, materials=materials)


def _built_over_law(row):
    """``built / (6*sqrt(2) * V / h^3)`` -- where the delivery landed vs the law."""
    return row["built"] / (COMMON.TETS_PER_UNIT_VOLUME * row["volume_m3"] / row["h_m"] ** 3)


def _replaced_formula_estimate(h_m, volume_m3):
    """``C * V / h^3`` -- what the estimate said before this change."""
    return int(REPLACED_C * volume_m3 / (h_m ** 3))


@pytest.mark.parametrize("label", COMPLIANT_FAMILIES)
def test_the_estimate_never_reads_below_the_real_mesh(label):
    """The one-sided requirement: under-reading sends a job to the wrong queue.

    An under-estimate is the expensive direction -- the estimate chooses the
    machine, so a mesh built far larger than its estimate lands on the queue
    chosen for a smaller job -- which is why the margin is the worst measured
    under-estimate rather than a median fit. So on every family whose mesh lands
    near the law the estimate must be at or above the count TetGen really
    returns, and the same run also has to stay inside the widest over-estimate
    the sweep measured, or the band has moved and the margin needs re-measuring.
    """
    row = _build_family(label)
    ratio = row["est"] / row["built"]
    assert row["built"] > 0, f"{label}: the fixture built no elements"

    assert ratio >= 1.0, (
        f"{label}: the estimate reads {ratio:.3f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,} elements). An under-estimate sends "
        f"the job to the slower machine; the margin ("
        f"{CFG.EST_TETS_MARGIN}) has to cover this family too."
    )
    assert ratio <= CFG.EST_TETS_WORST_OVER, (
        f"{label}: the estimate reads {ratio:.2f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,}), over the "
        f"{CFG.EST_TETS_WORST_OVER}x worst the sweep measured -- the band has "
        f"moved, so EST_TETS_MARGIN (and the refusal trade written against "
        f"EST_TETS_WORST_OVER) rest on numbers that no longer hold. If the "
        f"mesher changed, re-measure with "
        f"`python benchmarks/estimate_vs_built.py --families` and re-derive."
    )


def test_the_replaced_formula_would_refuse_an_upload_that_fits():
    """The harm the change removes, on one real mesh -- the reddening assertion.

    The family here is the plate: a coarse delivery (0.715x the volume law), the
    worst over-read the shipped sweep measured at its own setting. Two claims:

    1. on this real mesh the product's estimate stays inside the band, while the
       fitted ``C = 14`` this change replaced -- recomputed from the SAME mesh at
       the SAME edge -- does not;
    2. therefore on an upload whose real mesh is exactly the cap, main refuses it
       (its estimate crosses the refusal threshold) and this change does not.

    Reverting the constant reddens (1) on the product's own ratio. (2) is
    arithmetic on the measured ratio, which is why the mesh is built rather than
    the whole case being made up.
    """
    row = _build_family("one-part-plate")
    ratio = row["est"] / row["built"]
    old_est = _replaced_formula_estimate(row["h_m"], row["volume_m3"])
    old_ratio = old_est / row["built"]

    assert ratio <= CFG.EST_TETS_WORST_OVER, (
        f"the product's estimate reads {ratio:.2f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,}), outside the "
        f"{CFG.EST_TETS_WORST_OVER}x band the sweep measured"
    )
    assert old_ratio > ratio, "the replaced formula must read higher on any mesh"
    assert old_ratio > CFG.EST_TETS_WORST_OVER, (
        f"the replaced formula reads only {old_ratio:.2f}x the real mesh "
        f"({old_est:,} vs {row['built']:,}) -- inside the band, so this test no "
        f"longer demonstrates that the old formula was the problem. If the "
        f"mesher now lands near the volume law, re-measure the sweep and "
        f"re-derive EST_TETS_MARGIN, EST_TETS_WORST_OVER and the refusal factor "
        f"from it rather than widening this assertion."
    )
    # And the two really are different formulas, not the same one twice: the
    # replaced constant is 14/(6*sqrt(2)*margin) times the new estimate.
    assert old_est == pytest.approx(
        row["est"] * REPLACED_C / (COMMON.TETS_PER_UNIT_VOLUME * CFG.EST_TETS_MARGIN),
        rel=1e-3,
    ), "the recomputed old estimate does not match the ratio of the two constants"

    # (2) the product consequence, at the cap: a real mesh of exactly max_tets()
    # scales by this family's measured ratio on each formula.
    cap = CFG.max_tets()
    at_cap_new = ratio * cap
    at_cap_old = old_ratio * cap
    assert at_cap_old > CFG.upload_refuse_tets() >= at_cap_new, (
        f"on an upload whose real mesh is the {cap:,} cap this family's measured "
        f"ratios give {at_cap_old:,.0f} elements by the replaced formula and "
        f"{at_cap_new:,.0f} by this one, against a refusal threshold of "
        f"{CFG.upload_refuse_tets():,}: the pair no longer shows the old formula "
        f"refusing an upload that fits while this one accepts it"
    )


@pytest.mark.parametrize("label,recorded", sorted(UNDER_READ_FAMILIES.items()),
                         ids=[label for label, _ in sorted(UNDER_READ_FAMILIES.items())])
def test_the_over_refined_families_read_below_the_real_mesh(label, recorded):
    """The families the estimate misses on the LOW side, pinned with their numbers.

    Two mechanisms, one direction. The 3-region nested-thin-sheets fixture is the
    mesher found issue: under the UNIFORM mesher it asked for one edge, TetGen
    refined 3x below it and built ~9x the count the law predicts, so the estimate
    -- which can only read the request -- read ~0.15x of the real mesh. The bumpy
    sphere at half the mode counts is the same direction from a different cause:
    the surface's own facets floor the delivery at 1.868x the law, leaving the
    estimate at 0.72x, which is the direct counterexample to ``EST_TETS_MARGIN``.

    WHICH mesher built each family is the gate: the nested one is multi-part, so
    grading (the product rule since ``graded-meshing-default``) now shapes it and
    its numbers live in GRADED_FAMILIES; the bumpy sphere is one part, so grading
    never touches it and these are the live numbers. Running with
    ``MSE_GRADED_MESHING=0`` puts the nested family back under this test.

    Both recorded numbers are asserted -- the ratio AND the delivery in the
    margin's own units -- so neither can drift while the other is checked. A
    mesher fix, or a margin raised to cover them, reddens this test, and that is
    the intended signal: the constants were measured against a mesher that no
    longer exists.
    """
    ratio_recorded, delivery_recorded = recorded
    # Pin the UNIFORM mesher for this test. The uniform record is what
    # ``MSE_GRADED_MESHING=0`` builds, and it has to stay checked -- otherwise
    # the rollback is an untested second code path rather than a rollback.
    with pytest.MonkeyPatch.context() as off:
        off.setattr(COMMON, "graded_meshing_enabled", lambda: False)
        row = _build_family(label)
    assert not row["graded"], (
        f"{label} was built GRADED; the uniform record did not take. Its graded "
        f"numbers are pinned by test_the_graded_families_read_against_the_real_mesh."
    )
    ratio = row["est"] / row["built"]
    bv = _built_over_law(row)

    assert ratio < 1.0, (
        f"{label}: the estimate now reads {ratio:.2f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,}). If the mesher has been fixed to "
        f"deliver the edge it asks for, this is the good news and the constants "
        f"are now WRONG in the other direction: re-measure with "
        f"`python benchmarks/estimate_vs_built.py --families` and re-derive "
        f"EST_TETS_MARGIN and EST_TETS_WORST_OVER (and the refusal trade written "
        f"against them)."
    )
    assert ratio == pytest.approx(ratio_recorded, abs=0.05), (
        f"{label}: the estimate reads {ratio:.3f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,}, delivered {bv:.2f}x the law), not "
        f"the {ratio_recorded} measured when this under-read was recorded. It has "
        f"changed shape -- re-measure the sweep before trusting either constant."
    )
    assert bv == pytest.approx(delivery_recorded, rel=0.05), (
        f"{label}: the delivery is {bv:.3f}x the law, not the "
        f"{delivery_recorded} recorded. The margin is stated over this quantity, "
        f"so this is the number the record has to hold."
    )
    # In the margin's OWN units: a family whose delivery exceeds the margin is by
    # construction not covered by it. (Comparing the est/built ratio against the
    # margin would be an estimate-only error -- the two are different quantities.)
    assert bv > CFG.EST_TETS_MARGIN, (
        f"{label}: the delivery ({bv:.3f}x the law) is now inside the margin "
        f"({CFG.EST_TETS_MARGIN}), so this family is no longer an under-read and "
        f"does not belong in UNDER_READ_FAMILIES"
    )


@pytest.mark.parametrize("label,recorded", sorted(GRADED_FAMILIES.items()),
                         ids=[label for label, _ in sorted(GRADED_FAMILIES.items())])
def test_the_graded_families_read_against_the_real_mesh(label, recorded):
    """The same families under the GRADED mesher, pinned with their numbers.

    This is the product rule since ``graded-meshing-default``, so these are the
    numbers a real upload of this shape now gets. The sizing field holds each
    region at the edge it was asked for, so the 3x-below-request refinement that
    made the uniform mesher build 8.95x the law is gone: the delivery is 0.29x
    the law and the estimate reads above the real mesh, which is the direction
    ``EST_TETS_MARGIN`` is one-sided for.

    Pinned as a pair for the same reason the uniform entry is: a mesher change
    that moves either number reddens this, and that is the signal to re-measure
    the sweep before trusting the margin on the graded path.
    """
    ratio_recorded, delivery_recorded = recorded
    assert label in GRADED_FAMILIES, (
        f"{label} has no GRADED_FAMILIES entry, so this test cannot pin it"
    )
    row = _build_family(label)
    assert row["graded"], (
        f"{label} is parametrized as a GRADED family but was built with grading "
        f"OFF (MSE_GRADED_MESHING). Unset it, or run the uniform record."
    )
    ratio = row["est"] / row["built"]
    bv = _built_over_law(row)
    assert ratio == pytest.approx(ratio_recorded, abs=0.05), (
        f"{label}: graded, the estimate reads {ratio:.3f}x the real mesh "
        f"({row['est']:,} vs {row['built']:,}, delivered {bv:.2f}x the law), not "
        f"the {ratio_recorded} recorded. Re-measure the sweep before trusting "
        f"the margin on this path."
    )
    assert bv == pytest.approx(delivery_recorded, rel=0.05), (
        f"{label}: graded, the delivery is {bv:.3f}x the law, not the "
        f"{delivery_recorded} recorded."
    )
    # The change's whole point, asserted as a direction on the family the
    # uniform mesher over-refined by ~9x: grading must land it BELOW the law.
    assert bv < 1.0, (
        f"{label}: graded, the delivery is {bv:.3f}x the law, not under it -- "
        f"the sizing field is no longer holding the refinement at the requested "
        f"edges, so the graded path is not doing what it is priced for"
    )


def test_the_constants_are_the_measured_numbers():
    """The margin and the refusal factor against the sweep they came from.

    Both are measurements of the mesher, so both carry a provenance that can be
    checked without running anything: the worst under-estimate the sweep saw
    (``built / volaw``) is what the margin covers, and the worst over-estimate
    (``est / built``) is the width of the band the refusal rule has to live with.
    The refusal factor itself stays where main ships it, and that is checked here
    too.

    The margin covers every family in the shipped sweep EXCEPT the two recorded
    under-reads, which are stated here with their numbers rather than dropped:
    the margin is a measurement of one point in the mode-count x geometry space,
    not a bound over it, and the alternative to leaving those uncovered is a
    margin that reads the dense-cored families 6-12x high.
    """
    # bench: `python benchmarks/estimate_vs_built.py --families` at the product's
    # cap, and the same with --family-scale-n 0.5 for the half-mode-count rows.
    covered_worst_under = 1.298   # 2-region dense core n=3.5, built/volaw
    # the two under-reads the margin does NOT cover (see UNDER_READ_FAMILIES):
    uncovered_under = {
        1.414: "one-part 6mm cube with a 3.5x finer core (estimate 0.95x)",
        1.868: "bumpy sphere at half the mode counts (estimate 0.72x)",
        8.952: "3-region nested thin sheets (estimate 0.15x), the mesher issue",
    }
    worst_over = 2.098   # bumpy sphere at half the mode counts, est/built

    assert CFG.EST_TETS_MARGIN > covered_worst_under, (
        f"the margin {CFG.EST_TETS_MARGIN} does not cover the worst under-"
        f"estimate in the shipped sweep it was measured on ({covered_worst_under}) "
        f"-- a family that lands there would be routed on a count below its mesh"
    )
    assert CFG.EST_TETS_MARGIN < covered_worst_under * 1.25, (
        f"the margin {CFG.EST_TETS_MARGIN} is far above the measured worst "
        f"({covered_worst_under}); headroom is not free -- every point of it is "
        f"an over-read on every family"
    )
    for value, what in sorted(uncovered_under.items()):
        assert value > CFG.EST_TETS_MARGIN, (
            f"{what} measured {value} of the law, at or BELOW the margin "
            f"{CFG.EST_TETS_MARGIN}, so it is not an uncovered under-read any "
            f"more -- re-measure the sweep and correct the margin's provenance "
            f"rather than leaving this stale"
        )
    assert CFG.EST_TETS_WORST_OVER >= worst_over, (
        f"the band's top ({CFG.EST_TETS_WORST_OVER}) is below the worst "
        f"over-estimate measured ({worst_over}), so the refusal rule written "
        f"against it is not a bound"
    )
    # The refusal factor is deliberately NOT the band's top, and this change
    # leaves it where main ships it. A factor at the band's top would accept an
    # upload whose estimate is tight with up to 2.1 x 600,000 = 1.26M real
    # elements -- the mesher's own cap then coarsens that to fit rather than the
    # Orin building it, a quality loss rather than the OOM main's would have
    # risked, but still the worse of the two failures the lead weighed
    # (web/config.py, UPLOAD_REFUSE_FACTOR). Pinning the shipped value here makes
    # "this change did not move the refusal rule" a checked fact rather than a
    # promise: whoever re-derives the factor has to change this too.
    assert CFG.UPLOAD_REFUSE_FACTOR == 2, (
        f"the refusal factor is {CFG.UPLOAD_REFUSE_FACTOR}, not the 2 main "
        f"ships. Widening it is a product decision -- it buys fewer wrongful "
        f"refusals at the price of accepting meshes up to factor x cap real "
        f"elements -- not something a formula change may do on its own."
    )
    assert CFG.UPLOAD_REFUSE_FACTOR < CFG.EST_TETS_WORST_OVER, (
        "the factor has grown to the band's top; see web/config.py for why the "
        "two numbers are not the same and what accepting that costs"
    )
    assert CFG.upload_refuse_tets() == CFG.UPLOAD_REFUSE_FACTOR * CFG.max_tets(), (
        "the refusal threshold is no longer the factor times the cap"
    )


def test_the_two_estimate_paths_agree_on_a_nested_object(monkeypatch):
    """The multi-part sum against the one-part law at ``h_min``, grading OFF.

    This is the uniformity that ``MSE_GRADED_MESHING=0`` buys: with grading off
    every region is meshed at the same edge, so the per-region sum IS the
    one-part expression on the filled volume -- algebraically, and to within the
    float noise of summing before dividing rather than after. Pinning it on the
    rollback side is what makes the switch a rollback rather than a second,
    untested code path.

    It is NOT asserted on the graded side, and that asymmetry is the change:
    graded, each region is meshed at its own edge, so the sum differs from the
    ``h_min`` law by construction. See
    ``test_grading_moves_the_estimate_with_the_mesh``.
    """
    monkeypatch.setattr(COMMON, "graded_meshing_enabled", lambda: False)
    row = _build_family("two-region-core")
    margin = CFG.est_tets_margin()
    one_part = int(margin * COMMON.TETS_PER_UNIT_VOLUME
                   * row["volume_m3"] / row["h_m"] ** 3)
    assert abs(row["est"] - one_part) <= 1, (
        f"the multi-part estimate {row['est']:,} and the one-part law at h_min "
        f"{one_part:,} disagree by more than the float noise of the sum's order"
    )


def test_grading_moves_the_estimate_with_the_mesh():
    """The graded estimate is the mesher's plan, and it is not the uniform sum.

    Two facts, and the second is why the change exists. (1) With grading ON --
    the product rule -- ``web.meshing.estimate_tets`` must return exactly
    ``estimate_tets_graded`` on the same plan the mesher enumerates, so the
    upload is priced against the mesh it will get. (2) On a nested fixture whose
    core is finer than its wall, that number must be MATERIALLY DIFFERENT from
    the uniform-at-``h_min`` sum -- whichever way the fixture goes.

    It goes DOWN here, and that is the useful direction: the uniform plan meshes
    the coarse wall at the core's fine edge, so its sum prices a mesh nobody
    builds. Measured on ``two-region-core``: graded 4,907, uniform 18,755 (0.26x).
    An earlier version of this test asserted graded was ABOVE uniform; that was
    an assumption about fixture shape, and this fixture refutes it. What must
    hold either way is that the two disagree by more than noise -- equal numbers
    would mean the fixture stopped exercising grading.
    """
    row = _build_family("two-region-core")
    parts = row["parts"]
    materials = row["materials"]
    margin = CFG.est_tets_margin()
    edge_units = COMMON.effective_edge_m(
        PARTS.merged_surface(parts), n_modes=row["n_modes"], units="mm",
        parts=parts, materials=materials,
    )
    graded = COMMON.estimate_tets_graded(
        parts, materials, edge_units, PARTS.merged_surface(parts), "mm", margin
    )
    # ``estimate_tets_by_region`` takes no switch -- it IS the uniform sum -- so
    # this is the number grading is compared against, not a second switch path.
    uniform = COMMON.estimate_tets_by_region(
        parts, materials, edge_units, PARTS.merged_surface(parts), "mm", margin
    )
    assert row["est"] == graded, (
        f"with grading ON the product estimate is {row['est']:,}, not the "
        f"graded plan's {graded:,}: the route is pricing a different mesh than "
        f"the mesher builds"
    )
    assert graded != uniform and abs(graded - uniform) > 0.05 * uniform, (
        f"the graded estimate {graded:,} and the uniform sum {uniform:,} agree "
        f"to within 5% on a nested fixture. If the regions have genuinely "
        f"landed on one edge this fixture no longer exercises grading, and the "
        f"switch's ON side is untested -- fix the fixture, not this assertion."
    )


def test_the_web_fallback_margin_matches_the_config_one():
    """``meshing.EST_TETS_MARGIN`` is the fallback for callers that pass none.

    The upload route passes ``config.est_tets_margin()``; everything else (the
    bench, a test, a future caller) gets the module constant. Two numbers for
    one measurement is exactly how a route and a mesh drift apart, so they are
    held equal here.
    """
    assert WEB.EST_TETS_MARGIN == CFG.EST_TETS_MARGIN, (
        f"meshing.EST_TETS_MARGIN is {WEB.EST_TETS_MARGIN} and "
        f"config.EST_TETS_MARGIN is {CFG.EST_TETS_MARGIN}"
    )
