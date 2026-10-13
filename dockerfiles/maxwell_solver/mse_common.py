"""The little that BOTH images need, with dependencies light enough for both.

WHY THIS FILE EXISTS. It is not a tidy-up; it is a crashloop that reached
production. The web image and the solver image are built from two Dockerfiles
that COPY different directories. ``web/config.py`` imported
``mse_maxwell_solver`` for two integer defaults, and ``web/meshing.py`` imported
``mse_meshing`` for the sizing functions -- both living in
``dockerfiles/maxwell_solver/``, which the web Dockerfile does not copy. The
image built, pushed, and died at import:

    /opt/app/config.py:112
    ModuleNotFoundError: No module named 'mse_maxwell_solver'

Every test passed, because the tests put ``dockerfiles/maxwell_solver`` on
``sys.path``. Nothing in the suite started the web IMAGE, so nothing asked the
question the container asks first. The site was down for about six minutes.

THE RULE THIS FILE ENCODES. Anything both images import lives HERE, and this
module may import only what both images already have: numpy and (inside the
functions that need it) trimesh. Neither Dockerfile gains a dependency to share
this, which is what keeps the web image small enough to stay free of the native
toolchain.

WHAT MUST NOT COME HERE. Anything that pulls in the heavy solver stack -- the
tetrahedralizer, pyvista/vtk, cuDSS, the scipy factorisations, the fields and
storage modules. If a helper needs one of those, it belongs in
:mod:`mse_meshing` and the web app must reach it some other way. The one test
that would catch a violation is ``tests/test_web_imports.py``, which checks that
every import reachable from ``web/`` resolves to either a stdlib module, a
pinned requirement, or a file the web Dockerfile actually copies.

IMPORTS GO ONE WAY. ``mse_meshing`` and ``mse_maxwell_solver`` import from this
module; this module imports from neither. That is what makes it safe for the
web image to import directly, and it is why the solver modules re-export these
names rather than defining their own -- there is exactly one definition of each,
and the re-export is a binding, not a copy. A second definition would be two
sources of truth for the element cap, which is the defect class the cap's own
docstring already describes.
"""

from __future__ import annotations

import os

import numpy as np

#: unit name -> metres per unit
#:
#: ``um`` and ``nm`` are here because optical structures are routinely specified
#: in them: a 1550 nm telecom cavity is 1.55 um across, which in mm is a number
#: nobody wants to type (0.00155). Adding a unit is not cosmetic -- the solver
#: works in metres throughout, so a wrong scale is a wrong mode frequency by
#: exactly that factor.
#:
#: ``in`` is the odd one out: it is not a metric prefix, and a 1-inch cavity is
#: 0.0254 m rather than 1e-3 of something. It is here because CAD exports use it.
UNIT_SCALE = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
}

#: VTK cell id for a linear tetrahedron
VTK_TETRA = 10

#: TetGen's ``maxvolume`` is a volume bound, but the elements it emits with
#: ``minratio``/``mindihedral`` run larger than the bound. Calibrated so that
#: ``target_maxvolume`` actually lands near the requested edge length.
TETGEN_VOLUME_SLACK = 4.0

#: How much larger than the requested edge length the achieved median edge may
#: be before we refuse to solve. A mesh this coarse no longer resolves the
#: requested modes, and the frequencies would be wrong without anything failing.
MAX_EDGE_OVERSHOOT = 1.6

#: A coarse region's target edge may not approach the THICKNESS of the region it
#: describes. Past that cliff, TetGen's mesh-sizing function stops being honoured
#: and it fails the WRONG way: the fine region loses its resolution too, so the
#: whole mesh collapses and the job fails a check it should never have met.
#:
#: The length used is the region's own thickness proxy ``2 * V_filled /
#: A_boundary`` (see ``region_boundary_area``): a slab's volume is thickness x
#: area, so for a thin shell this IS the thickness, and for a fat region it
#: under-estimates it, which clamps sooner and errs toward the finer, safer mesh.
#: MEASURED on nested-cube fixtures (wall 2.0, canonical frame, tetgen 0.8.4):
#: shell 0.375 held at h_wall 0.70 and collapsed at 1.00; shell 0.15 held at 0.30
#: and collapsed at 0.45 -- a cliff at 1.9-3x the shell's thickness, and
#: independent of the background mesh's resolution (0.22 vs 0.08/0.05/0.04
#: canonical gave an identical 128-element collapse, so that is not the cause).
#: This proxy is 0.309 and 0.147 for those two fixtures, so 1.0 x it sits under
#: both cliffs with a 2-3x margin. A region's edge never falls below ``h_min``
#: for it, so the finest region -- the one the cavity is solved at -- is never
#: clamped and no run is silently made coarser.
CLAMP_THICKNESS_FRACTION = 1.0

#: default element cap. Chosen so a worst-case solve stays well under the 14Gi
#: container limit; the lead calibrates the real figure on the Jetson.
#:
#: THE ONE SOURCE OF TRUTH for this number. ``web/config.py`` imports it rather
#: than keeping its own copy, and both the cluster Job env and the worker's
#: claim payload carry the value this function resolves. It used to be 80_000
#: here while the app capped uploads at 600_000, with nothing passing the app's
#: number to the solver: every job the app accepted above 80k tets -- exactly
#: the large jobs the GPU exists for -- died in the solver on a limit the app
#: had already told the owner was fine. Two constants that must agree and are
#: never compared is the whole defect class; there is now one constant.
#:
#: It lives in THIS module rather than in the solver's entrypoint for the same
#: reason it exists at all: the app must be able to read it without importing
#: the solver, and the solver must not keep a second copy for the app to drift
#: from.
DEFAULT_MAX_TETS = 600_000

#: elements above which the CLUSTER cannot solve a job, but the GPU still can.
#: The upload-time estimate decides the route, but an estimate can be wrong in
#: either direction; this is the exact count re-deciding, at the first point the
#: exact count exists. Deliberately lower than DEFAULT_MAX_TETS: it is a
#: routing boundary, not a capability limit.
DEFAULT_CPU_MAX_TETS = 150_000

#: what a ``TooBigForProcessor`` writes into its termination message. The app
#: reads this string back out of the pod's last state to tell "this job is on
#: the wrong machine" from "this job is broken" -- see the class docstring in
#: ``mse_maxwell_solver``.
#:
#: Defined here because the APP has to match it and the app cannot import the
#: solver. A token the two sides spell out separately is a token that can drift
#: to a string nobody writes and nobody reads.
REROUTE_TOKEN = "MSE_REROUTE"

#: How much headroom over Weyl's law to allow. Weyl's count is asymptotic and
#: mildly under-predicts for a small cavity (for the 10x7x5 box at N=20 it gives
#: k = 1.19/mm against a true 1.262/mm, about 6%), so the geometric estimate is
#: scaled by this before it becomes a wavelength.
WEYL_SAFETY = 1.2


class MeshingError(Exception):
    """A failure the user can act on (bad file, open surface, mesh too big)."""


def unit_scale(units: str) -> float:
    """Metres per unit for one of the supported unit names."""
    try:
        return UNIT_SCALE[units]
    except KeyError:
        raise MeshingError(
            f"unsupported units {units!r}; expected one of {sorted(UNIT_SCALE)}"
        ) from None


def apply_scale(surface, scale: float):
    """Return ``surface`` with every vertex multiplied by ``scale``.

    This is the whole of the free scale factor: the object is scaled before
    anything else looks at it, so the volume, the surface edge, the target edge
    length and the decimation all see the same size -- which is the only way
    ``est_tets`` computed in the app can agree with the mesh the solver builds.

    Scaling leaves the shape exactly similar, so the ``scale``-free quantities
    (the *ratio* of surface edge to target edge, the face count decimation
    chooses) are unchanged; only the quantities with a length dimension move,
    and they move together. That is why this can be a single multiply at the
    edge and not a parameter threaded through the sizing functions.
    """
    if surface is None:
        return None
    factor = float(scale)
    if not np.isfinite(factor) or factor <= 0:
        raise MeshingError(f"scale must be a positive number, not {scale!r}")
    if factor == 1.0:
        # Deliberately the same object: an unscaled upload must behave exactly
        # as it did before this parameter existed, copy included.
        return surface
    scaled = surface.copy()
    scaled.vertices = np.asarray(scaled.vertices, dtype=float) * factor
    return scaled


def enclosure_volume_m3(surface, units: str) -> float:
    """Volume actually enclosed by the surface, in cubic metres.

    NOT the bounding box: Weyl's law counts modes per unit *enclosed* volume, so
    a hollow or non-convex object needs its own volume or the mesh comes out far
    too fine. Falls back to the box volume when trimesh cannot produce a signed
    volume (a degenerate or nearly-closed surface), because a coarse estimate
    only costs accuracy, while raising here would refuse a usable upload.
    """
    scale = unit_scale(units)
    try:
        volume = float(abs(surface.volume)) * scale ** 3
    except Exception:  # noqa: BLE001 - trimesh raises several things here
        volume = 0.0
    if not np.isfinite(volume) or volume <= 0:
        bounds = np.asarray(surface.bounds, dtype=float)
        extent = bounds[1] - bounds[0]
        volume = float(np.prod(extent)) * scale ** 3
    if not np.isfinite(volume) or volume <= 0:
        raise MeshingError("this object has no enclosed volume to mesh")
    return volume


def target_edge_length(
    surface,
    n_modes: int,
    units: str,
    n_index: float = 1.0,
) -> float:
    """Element edge length (in the user's units) that resolves the top mode.

    The mode *count* in three dimensions follows Weyl's law,
    ``N(k) ~ V k^3 / (3 pi^2)``, so the k of the N-th mode is

        k_N ~ (3 pi^2 N / V)^(1/3)

    with V the enclosed volume. k grows like the CUBE ROOT of the mode number,
    not linearly -- an earlier version of this function used
    ``k_high = n_modes * pi / D`` (D = bbox diagonal), which over-refines badly:
    for the 10x7x5 box at 20 modes it asked for k = 4.76/mm when the true 20th
    mode is at 1.262/mm, about 4x too fine in edge length and ~60x too many
    elements.

    n_index deliberately does not enter: the geometric k is independent of the
    filling medium (the in-medium wavelength is what changes, and that is
    handled downstream). The parameter is kept for call compatibility.
    """
    scale = unit_scale(units)
    volume_m3 = enclosure_volume_m3(surface, units)
    count = max(int(n_modes), 1)

    k_high = WEYL_SAFETY * (3.0 * np.pi ** 2 * count / volume_m3) ** (1.0 / 3.0)
    wavelength_m = 2.0 * np.pi / max(k_high, 1e-30)

    edges_per_wavelength = float(os.getenv("MSE_EDGES_PER_WAVELENGTH", "7"))
    edge_m = wavelength_m / edges_per_wavelength
    return edge_m / scale                            # back to the user's units


def effective_edge_m(
    scaled_surface,
    n_modes: int,
    units: str,
    min_feature_m: float | None = None,
    n_index: float = 1.0,
    physics_m: float | None = None,
    parts=None,
    materials: dict | None = None,
) -> float:
    """The edge length the mesh is built to, in METRES on the scaled surface.

    This is the ONE place the physics edge and the owner's declared smallest
    feature are compared. Both callers -- ``tetrahedralize`` and the web app's
    ``estimate_tets`` -- call it and neither re-derives the comparison, because
    the whole class of bug this replaces was two call sites combining the same
    two numbers in two different length spaces.

    The arguments are already in the one space that matters:

    * ``scaled_surface`` has been through :func:`apply_scale`, so every length
      measured on it is the object the solver was actually asked to solve.
    * ``min_feature_m`` is METRES, converted once at the edge of the system
      (the upload form, from its own unit selector) and carried in metres from
      there. It is NOT converted here -- that is deliberate. The parameter used
      to be called ``min_feature_mm`` and arrived as a bare number in the
      object's units, which meant every reader had to know *which* unit it was
      and every one of them could get it wrong. A value that is already metres
      cannot be converted wrongly.

    ``target_edge_length`` returns the user's units, so it is converted here --
    ``physics_m`` is the only length in this function that needs a factor, and
    it needs exactly one.

    ``parts`` and ``materials`` describe a MULTI-PART upload. With them this
    still returns the WEYL edge, unchanged -- and NOT a function of the wall's
    index. In a uniform cavity the mesh is independent of n: ``target_edge_
    length`` computes the geometric k from the mode count and the volume, and
    says so ("n_index deliberately does not enter"). The index enters
    DOWNSTREAM, in ``region_edges``, which fines each region relative to the
    wall. So this value is the wall's edge by construction -- the wall is the
    region that keeps it -- and the multi-part branch below does not shrink it.

    ``None`` for either is the one-part upload, which is byte-identical to
    before this existed.

    ``physics_m`` may be passed in by a caller that already computed it. The
    solver's tetrahedralizer does, because it needs the same number for its own
    verbose line AND needs the call to go through ``mse_meshing``'s module
    global so the resolution the guard tests can be scripted. Passing it keeps
    one evaluation instead of two that could disagree at the last bit -- and the
    ``min`` below is exactly where a last-bit disagreement would show up as two
    different edges for one object.
    """
    if physics_m is None:
        physics_m = target_edge_length(
            scaled_surface, n_modes=n_modes, units=units, n_index=n_index
        ) * unit_scale(units)

    if min_feature_m is None:
        base_m = physics_m
    else:
        feature_m = float(min_feature_m)
        if not np.isfinite(feature_m) or feature_m <= 0:
            # A nonsense feature is ignored rather than divided by. Same rule as
            # the app's validation, applied here too because this function is
            # reachable from the solver's own CLI.
            base_m = physics_m
        else:
            # Whichever is SMALLER wins: a feature left unresolved and a top
            # mode left unresolved both give frequencies that are confidently
            # wrong.
            base_m = min(physics_m, feature_m)

    # --- the per-region rule ------------------------------------------------
    #
    # ``parts``/``materials`` are a multi-part upload, and the GLOBAL edge above
    # is the WALL's edge. It is NOT shrunk to the densest region's edge: the
    # per-region edges are a RATIO of it (``h_r = base_m * n_wall / n_r``, see
    # ``region_edges``), so shrinking the base would drag every region down --
    # including the wall, which is the reference.
    #
    # An earlier version took ``min(base_m / n_r)`` here and applied it as the
    # global. For {wall 2.0, core 3.5} that meshed the WALL at base/2 and
    # over-predicted the count by more than 10x, refusing an upload the GPU
    # could run. The wall's edge is the base; the regions divide it.
    return base_m


def target_maxvolume(surface, edge_length: float) -> float:
    """TetGen ``maxvolume`` that gives roughly the requested edge length.

    TetGen's ``maxvolume`` is a bound on element volume, not edge length. For an
    equilateral-ish tet of edge ``h`` the volume is ``h^3 / (6 sqrt(2))``; TetGen
    with ``minratio``/``mindihedral`` produces elements somewhat larger than that
    bound, so we scale by ``TETGEN_VOLUME_SLACK``. Calibrated against real
    tetrahedralizations: the exact factor only needs to be within ~2x, because
    the achieved edge length is what the accuracy depends on and the hard
    ``MSE_MAX_TETS`` cap is enforced on the count TetGen actually returns.
    """
    h = float(edge_length)
    return TETGEN_VOLUME_SLACK * (h ** 3) / (6.0 * np.sqrt(2.0))


def estimate_tet_count(surface, edge_length: float) -> tuple[int, float]:
    """Rough ``(n_tets, volume)`` estimate for a mesh of the given edge length.

    The volume is the bounding-box volume, which overestimates a non-convex
    object but is far better behaved than an ``A^(3/2)`` guess (that formula is
    off by ~2.6x for a plain box). It only needs to be the right order of
    magnitude: the real count is read back from TetGen and capped there.
    """
    bounds = np.asarray(surface.bounds, dtype=float)
    volume = max(float(np.prod(bounds[1] - bounds[0])), 1e-18)
    maxvolume = target_maxvolume(surface, edge_length)
    return int(volume / max(maxvolume, 1e-30)), volume


def region_boundary_area(parts, part_index: int) -> float:
    """Area of the REGION's boundary, in the parts' own units squared.

    The region ``part_index`` bounds is the part MINUS everything nested inside
    it, so its boundary is the part's own closed surface plus every nested
    part's -- the SAME set ``PartSet.filled_volume_m3`` subtracts volumes for,
    because both describe that one region. ``trimesh`` computes the area the
    same way it computes the volume used there, so the pair cannot disagree
    about the object.

    It exists for one use: the thickness proxy ``2 * V_filled / A_boundary``
    that ``region_edges`` clamps a coarse target under. A slab's volume is
    thickness x area, so for a thin shell this is the thickness itself; for a
    fat region it under-estimates the thickness (a cube of side ``s`` gives
    ``s/3``), which clamps sooner and errs toward the finer, safer mesh.
    """
    total = 0.0
    for index in [part_index, *parts.children_of(part_index)]:
        total += abs(float(parts.parts[index].area))
    return float(total)


def region_edges(h: float, n_wall: float, materials: dict, parts,
                 units: str) -> dict:
    """``{region_id: {...}}`` -- the per-region target edge for a multi-part mesh.

    THE ONE HELPER. The mesher and the web estimate both call this, so the
    edge a region is built to and the edge the route was decided on cannot
    drift apart. ``min`` of ``edge`` over the values is ``h_min``: the edge the
    whole cavity used to be meshed at, and still the resolution FLOOR.

    THE RULE: EACH REGION AT ITS OWN EDGE, CLAMPED UNDER ITS OWN THICKNESS.

        raw_r = min(h, h * n_wall / n_r)            -- the physics, per region
        thickness_r = CLAMP_THICKNESS_FRACTION * 2 * V_filled_r / A_boundary_r
        edge_r = max(min(raw_r, thickness_r), h_min)

    ``min(h, ...)`` keeps a region LESS dense than the wall from coarsening it
    (``h_min`` is a resolution floor, not a function of the densest region
    alone). ``max(..., h_min)`` is what the floor guarantees the callers: no
    region is ever given an edge finer than ``h_min``, so the finest region --
    the one the cavity is SOLVED at -- is never clamped and ``h_min`` does not
    move from the uniform rule. A region with no denser neighbour keeps ``h``.

    WHY THE CLAMP. A region's target is delivered by TetGen's mesh-sizing
    FUNCTION (``-m``), a background mesh whose nodes carry the target edge, and
    the field stops being honoured when a target approaches the thickness of
    the region it describes -- failing the WRONG way: the fine region loses its
    resolution too, so the job fails its achieved-edge check rather than
    shipping a coarse answer. Measured on nested cubes (canonical frame, tetgen
    0.8.4): shell 0.375 held at h_wall 0.70, collapsed at 1.00; shell 0.15 held
    at 0.30, collapsed at 0.45 -- a cliff at 1.9-3x the thickness, and
    background-mesh resolution was tested as the cause and REFUTED (identical
    collapse at 0.22 and at 0.08/0.05/0.04). ``CLAMP_THICKNESS_FRACTION``
    carries the margin.

    WHAT THIS REPLACED. The previous rule meshed every region at ``h_min``
    because the mechanism to grade them was believed unavailable: TetGen's
    per-region volume, through ``add_region``'s ``max_vol``, was measured INERT
    on a nested PLC -- a nine-order sweep of the core's region volume moved the
    core's own element count by exactly zero, with identical totals under
    ``varvolume=True``, ``fixedvolume=True``, and with neither flag. THAT
    MEASUREMENT STANDS and its negative is still guarded (that call is not
    made here). What it does not cover is the sizing function, a DIFFERENT
    mechanism, which does grade: measured on the nested cube, 2,813 tets
    against 39,454 uniform with the core's median edge 0.0432 against 0.0433.
    So a region denser than the wall now gets a finer edge, the wall keeps
    ``h``, and the estimate follows the mesh instead of pricing a mesh nobody
    builds.

    ``units`` is the UPLOAD's unit and has no default, for the same reason
    ``filled_volume_m3`` has none: it converts the volume the clamp's thickness
    proxy is built from. ``h`` is in METRES, and so is everything this
    function returns -- the caller converts at its own boundary.

    ``n_wall`` is still required: it is what converts a region's index into a
    length relative to the wall's medium, and without it ``h * n_wall / n_r``
    would be ``h / n_r``, whose meaning depends on which medium ``h`` was
    derived in. The wall is the reference; the ratio to it is the physics.
    """
    by_region = parts.region_materials(materials)
    n_max = 0.0
    for region_id, n_region in by_region.items():
        n_region = float(n_region)
        if not (n_region > 0) or not np.isfinite(n_region):
            raise MeshingError(
                f"region {region_id} has refractive index {n_region:g}; it "
                f"must be finite and greater than zero."
            )
        n_max = max(n_max, n_region)
    if n_max <= 0:
        raise MeshingError(
            "a multi-part mesh needs at least one region with a refractive "
            "index to mesh to, and this upload has none."
        )
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        # The clamp is the first thing in this function that MIXES two length
        # spaces -- the edge and the region's thickness -- so the space the edge
        # arrives in is now a behaviour, not a formality. Refusing a
        # non-length is the cheap half of that; the callers' docstrings are the
        # other half, and ``test_routing`` pins the metres.
        raise MeshingError(
            f"region_edges needs the base edge in METRES and got {h:g}."
        )
    scale = unit_scale(units)
    # The floor: the finest region's own raw edge. `min(h, ...)` rather than
    # the raw ratio so a region LESS dense than the wall cannot coarsen the
    # cavity past h.
    h_min = min(h, h * float(n_wall) / n_max)
    # Per REGION, not per part: a region is a material and two parts may share
    # one, so the region's fill and boundary are the sums over its parts. The
    # edge itself is uniform within a region (same n, same material), so only
    # the geometry needs summing.
    volume_m3: dict = {}
    area_units2: dict = {}
    for part_index, region_id in enumerate(parts.region_of_part):
        region_id = int(region_id)
        volume_m3[region_id] = (
            volume_m3.get(region_id, 0.0)
            + parts.filled_volume_m3(part_index, units)
        )
        area_units2[region_id] = (
            area_units2.get(region_id, 0.0)
            + region_boundary_area(parts, part_index)
        )
    out = {}
    for region_id in by_region:
        region_id = int(region_id)
        n_region = float(by_region[region_id])
        raw = min(h, h * float(n_wall) / n_region)
        area = area_units2.get(region_id, 0.0)
        # 2*V/A in metres: the volume is cubic metres by construction, the area
        # is the parts' own units squared -- scaled by scale^2 to match.
        thickness_m = (
            CLAMP_THICKNESS_FRACTION * 2.0 * volume_m3.get(region_id, 0.0)
            / (area * scale * scale)
            if area > 0 else float("inf")
        )
        out[region_id] = {
            "edge": max(min(raw, thickness_m), h_min),
            "unclamped": raw,
            "thickness_m": thickness_m,
        }
    return out


def region_plan(parts, materials, edge_m: float, units: str,
                n_wall: float | None = None) -> dict:
    """Per-region ``(edge_m, filled_volume_m3)`` for a multi-part upload.

    ONE expression, called by BOTH the mesher and the web app's ``estimate_tets``.
    They answer the same question -- how fine the mesh is, and how much material
    is in each region -- in two processes, and the row that asked for this
    feature names the trap exactly: if the estimate and the mesher derive the
    mesh parameters from different places, the app's route decision and the mesh
    the solver builds disagree, and the estimate is what routes.

    ``edge_m`` IS METRES, and the name says so because it used to say
    ``edge_units`` while holding metres: the mesher fed it UPLOAD UNITS and the
    web fed it metres, which was harmless only while the rule divided by a bare
    index ratio. The thickness clamp mixes the edge with a thickness, so the two
    spaces stopped being interchangeable. The mesher now passes metres and
    converts at its own boundary; see ``estimate_tets_by_region`` for what the
    old ambiguity cost.

    UNDER THE GRADED RULE the two quantities are:

    * ``edge_m`` is the region's OWN target edge, ``min(h, h * n_wall /
      n_r)`` clamped under the region's thickness -- ``region_edges`` is the one
      place that rule lives. ``min`` over the plan's edges is ``h_min``, the
      finest region's edge, and it does not move from the uniform rule: the
      finest region is never clamped.
    * the volume is the region's FILLED volume, not its enclosed one:
      ``PartSet.filled_volume_m3`` subtracts everything nested inside it. The
      wall's enclosed volume counts its core twice. The estimate sums
      ``V_r / h_r^3`` over the regions, which is the one-part expression when
      the edges are equal.
    * ``unclamped_m`` and ``thickness_m`` are what makes the clamp
      REPORTABLE: the edge the ramp asked for before the clamp, and the region's
      thickness proxy. ``edge_m`` is below ``unclamped_m`` exactly when
      the clamp bit, which is when the job's caveats name the region.

    ``n_region`` comes from ``materials`` (part name -> n, naming every part).
    A region with no material is REFUSED by ``PartSet.region_materials`` rather
    than defaulted: the default would be vacuum, and the owner would get
    frequencies for a cavity that is partly empty with nothing saying so.

    ``units`` is the UPLOAD's unit and has no default. It reaches
    ``PartSet.filled_volume_m3``, which converts the region volume to cubic
    metres. There is no default here for the same reason there is none there:
    with one, a caller that forgot the argument silently got millimetres while
    the edge beside it stayed true metres, making the estimate wrong by
    the cube of the unit ratio -- for inches, 17,000x UNDER, which routes a job
    to the cluster that the GPU would have solved. Measured through the real
    upload on one physical object: mm 51,187 (GPU), in 3 (cluster), um 5.1e13
    and nm 5.1e19 (refused). The unit-invariance test pins this.
    """
    by_region = parts.region_materials(materials)
    # The wall's index is the reference the other regions are measured against.
    # ``n_wall`` comes from the CALLER when it has one (the solver knows which
    # part is the wall); otherwise it is the wall region's own material, which
    # is what ``parts`` records. Either way it is one number, resolved once.
    if n_wall is None:
        wall_index = getattr(parts, "wall_index", None)
        if wall_index is None:
            raise MeshingError(
                "a multi-part mesh needs a wall part to measure the other "
                "regions against, and this upload has none. The first part "
                "must contain every other part."
            )
        n_wall = float(by_region[int(parts.region_of_part[wall_index])])
    edges = region_edges(edge_m, n_wall, materials, parts, units)
    out = {}
    for part_index, region_id in enumerate(parts.region_of_part):
        info = edges[int(region_id)]
        out[int(region_id)] = {
            "part": parts.names[part_index],
            "edge_m": info["edge"],
            "filled_volume_m3": parts.filled_volume_m3(part_index, units),
            "n": float(by_region[region_id]),
            # For the record the job carries: what the ramp asked for before the
            # thickness clamp, and the thickness it was clamped under. A region
            # whose ``unclamped_m`` is bigger than ``edge_m`` was
            # clamped, and the job's caveats say so.
            "unclamped_m": info["unclamped"],
            "thickness_m": info["thickness_m"],
        }
    return out


#: How far, in log-edge space, the sizing field's transition moves a COARSE
#: region below its own target edge. ``estimate_tets_by_region`` prices a coarse
#: region at ``h_r * (h_min/h_r) ** this``.
#:
#: MEASURED through the product's own path -- the routing estimate against
#: TetGen's element count for the SAME plan (est/built; "plain" is the sum with
#: no correction) -- on the five graded plans ``tests/test_routing.py``'s
#: nested-pair matrix builds (the 6/2 box at n_modes 3) and the two
#: ``tests/test_meshing_guard.py::test_the_graded_estimate_lands_on_the_mesh_the_mesher_builds``
#: builds, so every row is reproducible by running them:
#:
#:     plan (6/2 box, index ratios)  plain   0.3     0.5
#:     n=1.0/1.0  (ungraded)         2.99   2.99    2.99
#:     n=1.0/2.0                     1.17   1.95    2.81
#:     n=1.0/3.5                     1.19   2.14    3.70
#:     n=2.0/3.5                     1.29   2.00    2.70
#:     n=3.0/3.5                     1.97   2.24    2.45
#:     6/2 wall/core 1.0/3.0         0.57   1.04    1.74
#:     6/4 wall/core 1.0/3.0         0.74   0.85    1.00
#:
#: The plain per-region sum is a bound on NEITHER side, and the mesher's own
#: per-region report says why: two effects move a graded mesh in opposite
#: directions.
#:
#:   * A coarse region carries a TRANSITION the plain formula does not pay for,
#:     and its tets come out FINER than its target -- achieved/target 0.71 on
#:     the n=1.0/3.5 plan, 0.64 on the thick fixture and 0.80 on the pinned
#:     test's own plan -- so it always holds more than ``c * V / h_r^3``:
#:     1.27x, 2.26x and 1.06x its plain pricing. This pushes the sum DOWN, and
#:     it is the expensive direction: the estimate ROUTES the job, so an
#:     under-count sends it onto a machine it does not fit and the failure
#:     lands after the machine has been paid for.
#:   * A FINE region only a few cells across is not described by the volume law
#:     at all -- its tets are dominated by the boundary it shares with the
#:     transition. The same 8 mm^3 core delivered an effective coefficient of
#:     8.0, 17.7 and 5.5 on those three plans, against the 14 fitted on whole
#:     objects. That pushes the sum UP on two of the three.
#:
#: The exponent pays the first effect and cannot pay the second, so it is a
#: single-parameter compromise and that spread is its accuracy limit. 0.3 is the
#: largest inflation that keeps EVERY measured plan clear of the estimator's own
#: ceiling (the tests bound est/built at 3.0, itself the UNGRADED case measured
#: at 2.99); 0.5 measured 3.70 on the 1.0/3.5 plan and broke that contract. An
#: earlier revision used the geometric mean of the two edges (0.5) on the two
#: fixtures alone; the nested-pair matrix is what falsified it.
TRANSITION_EDGE_EXPONENT = 0.3


def estimate_tets_by_region(parts, materials, edge_m: float,
                            surface, units: str, c: float) -> int:
    """Weyl-free element estimate for a multi-part upload.

    THE ONE-PART FORMULA, PER REGION:

        est = c * sum_r V_r / h_r^3

    with every region at its own planned edge, from the same ``region_edges``
    the mesher builds to. The estimate is what ROUTES the job, so one that kept
    pricing a uniform mesh would send a graded job to the cluster for elements
    it no longer has.

    WHEN EVERY REGION LANDS ON ONE EDGE -- the one-part case, and any multi-part
    upload whose regions share an index -- the code takes the one-part
    expression ``c * V_cavity / h_min^3`` BIT FOR BIT rather than summing. The
    two are algebraically identical there but round differently, and this is the
    number that routes a job, so it must not move by a rounding step when
    nothing was graded. ``V_cavity`` is the sum of the regions' FILLED volumes,
    which is the cavity's own volume with no double counting (a nested part's
    every point is inside its parent, so summing enclosed volumes would count
    the core twice).

    UNITS -- NO CONVERSION HAPPENS HERE, and that is a measured claim, not a
    stylistic one. This function produced a 2.3e13-element estimate for a 19k
    mesh on its first attempt because it multiplied by ``unit_scale(units)``
    "to get metres". Measured for the 6 mm fixture (wall n=2, core n=3.5, mm):

        effective_edge_m(...)        -> 1.0181282863016227e-03   [= 1.018 mm]
        region_edges(= h*2/3.5)      -> 5.817875921723558e-04    [= 1.018/1.75]
        filled_volume_m3 (sum)       -> 2.16e-07                 [cubic metres]
        unit_scale('mm')             -> 1.0e-03

    ``effective_edge_m`` returns METRES (1.018e-3 m is 1.018 mm, the right edge
    for a 6 mm box), and ``region_edges`` divides by a bare index ratio, so the
    plan's ``edge_m`` is in the SAME metres -- the argument is named for the
    space it holds, which it was not while this bug lived. ``filled_volume_m3``
    is cubic metres by construction. So the pair is already coherent and
    ``unit_scale`` here applies the factor a SECOND time, making the count wrong
    by ``1/scale^3`` = 1e9 for mm. That is the "one quantity, two spaces" shape
    this rig keeps meeting: the quantity was already in the target space and a
    second conversion was invisible to reading.

    ``c`` is the SAME constant the one-part estimate uses -- passed in, never
    re-derived here, so the two paths cannot disagree about it.

    A COARSE REGION IS PRICED AT THE EDGE THE TRANSITION DELIVERS, not at its
    target. The sizing field ramps from the fine edge to the coarse one across
    every boundary between regions, and elements near that boundary are finer
    than the coarse region's own target, so pricing each region at its own
    target UNDER-counts the mesh the mesher then builds -- measured, 1.35x on
    the thin wall fixture and 1.75x on the thick one, and the estimate is what
    routes the job. Each region above ``h_min`` is therefore priced at

        h_eff = h_r * (h_min / h_r) ** TRANSITION_EDGE_EXPONENT

    which is ``h_r`` exactly when nothing is graded (the exponent's own limit,
    preserving the one-edge branch below bit for bit) and moves toward the fine
    edge as the ratio grows. The correction is ONE number, so it pays the coarse
    region's transition and cannot also represent the fine region's own boundary
    effects; ``TRANSITION_EDGE_EXPONENT`` carries the measured table and states
    that as the model's accuracy limit, rather than claiming a bound.
    """
    plan = region_plan(parts, materials, edge_m, units)
    if not plan:
        # No regions at all -- not a multi-part upload. Refuse rather than
        # return 0, which would look like "a tiny job" and route it to the GPU.
        raise MeshingError(
            "the multi-part estimate was asked for a mesh with no regions; "
            "this is the one-part path's job and calling it here would route "
            "a job on a number of zero."
        )
    edges = set()
    v_m3_total = 0.0
    for info in plan.values():
        h_r = float(info["edge_m"])
        v_r = float(info["filled_volume_m3"])
        if not (h_r > 0) or not (v_r > 0):
            # Could not ask. A zero here would route a real job to the fast
            # machine on the strength of a number nothing computed -- refuse.
            raise MeshingError(
                f"the multi-part estimate got h={h_r:g} m and V={v_r:g} m3 for "
                f"region {info['part']!r}; neither can be used to size a mesh."
            )
        edges.add(h_r)
        v_m3_total += v_r
    if len(edges) == 1:
        # EVERY region at one edge: the one-part expression, bit for bit. Not a
        # nicety -- this is the number that routes the job, and it must not move
        # by a rounding step when nothing was graded. The per-region sum is
        # algebraically the same but rounds differently.
        return int(float(c) * v_m3_total / (edges.pop() ** 3))
    h_min = min(edges)
    delivered = 0.0
    for info in plan.values():
        h_r = float(info["edge_m"])
        v_r = float(info["filled_volume_m3"])
        if h_r > h_min:
            h_r *= (h_min / h_r) ** TRANSITION_EDGE_EXPONENT
        delivered += v_r / (h_r ** 3)
    return int(float(c) * delivered)
