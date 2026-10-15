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

#: THE VOLUME LAW: how many elements a volume holds at a given edge length.
#:
#: A regular tetrahedron of edge ``h`` has volume ``h^3 / (6 sqrt(2))``, so a
#: tetrahedralisation of a volume ``V`` whose elements all sat at edge ``h``
#: holds ``6 sqrt(2) * V / h^3`` = ``8.4853 * V / h^3`` elements. This is the
#: quantity ``target_maxvolume`` inverts (it asks TetGen for
#: ``TETGEN_VOLUME_SLACK * h^3 / TETS_PER_UNIT_VOLUME``), and it is the law the
#: upload-time estimate is built on (``web.meshing.estimate_tets`` multiplies it
#: by a measured margin). ONE law with two users: the estimate is what ROUTES an
#: upload and the mesher is what runs it, so the two must not each keep their own
#: version of the geometry.
TETS_PER_UNIT_VOLUME = 6.0 * np.sqrt(2.0)

#: How much larger than the requested edge length the achieved median edge may
#: be before we refuse to solve. A mesh this coarse no longer resolves the
#: requested modes, and the frequencies would be wrong without anything failing.
MAX_EDGE_OVERSHOOT = 1.6

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
    # Bit-identical to the ``(6.0 * np.sqrt(2.0))`` this used to divide by: the
    # constant IS that expression, so the one-part mesh stays byte-identical.
    return TETGEN_VOLUME_SLACK * (h ** 3) / TETS_PER_UNIT_VOLUME


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


def region_edges(h: float, n_wall: float, materials: dict, parts) -> dict:
    """``{region_id: h_r}`` -- each region's OWN required edge, multi-part mesh.

    THE ONE HELPER. The mesher and the web estimate both call this, so the edge
    a region is MEASURED against and the edge the route was decided on cannot
    drift apart.

    THE PER-REGION RULE. Every region is held to

        h_r = h * n_wall / n_r

    -- a denser region must be FINER, because the wavelength it has to resolve
    is shorter, so ``h_r`` falls as ``n_r`` rises. This is the requirement the
    resolution GUARD reads, region by region: no region may come out coarser
    than ``MAX_EDGE_OVERSHOOT * h_r``.

    ``min`` over these values, floored at ``h``, is ``h_min`` -- the SINGLE edge
    the whole cavity is MESHED at, which :func:`region_plan` stores as a
    region's ``edge_units``. The two are different questions and both are real:
    ``h_r`` is what a region's own resolution forbids exceeding, ``h_min`` is
    what the mesher builds to. A dense core's ``h_r`` is far below ``h_min`` and
    the core only reaches it because ``h_min`` is finer still -- which is why a
    guard that read ``h_min``'s allowance for every region could pass a core
    that its own rule refuses.

    WHY THE MESHER STILL BUILDS UNIFORM AT ``h_min``. A per-region MESHING
    target means handing ``h * n_wall / n_r`` to TetGen as that region's own
    size, on the theory that a region denser than the wall should be finer and a
    less dense one coarser. That theory is sound and the mechanism to implement
    it is NOT AVAILABLE in this build:
    TetGen's per-region volume, supplied through ``add_region``'s ``max_vol``,
    was measured INERT on a nested PLC -- a nine-order sweep of the core's
    region volume moved the core's own element count by exactly zero, with
    identical totals under ``varvolume=True``, ``fixedvolume=True``, and with
    neither flag. (Hand-built PLC, no application code involved; the numbers
    and the two instrument errors that hid this for a session are on the row.)
    So a per-region TARGET that the mesher cannot realise is a target that
    misleads the estimate -- which is the whole defect this function exists to
    prevent.

    Meshing everything at ``h_min`` realises the resolution requirement with a
    mechanism that provably works (a single global ``maxvolume``, the same one
    the one-part path uses), and it keeps the estimate on the SAME calibrated
    constant as the one-part path -- one formula, one calibration, no new
    fitting.

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
    # The raw per-region ratio, NOT a min: the floor that keeps ``h_min`` at or
    # below the wall's edge is applied by the CALLER (``region_plan``), so this
    # stays the single expression of the rule and the value a region is guarded
    # against is the same arithmetic the estimate routes on.
    return {
        int(region_id): float(h) * float(n_wall) / float(n_region)
        for region_id, n_region in by_region.items()
    }


def region_plan(parts, materials, edge_units: float, units: str,
                n_wall: float | None = None) -> dict:
    """Per-region ``(edge_units, filled_volume_m3)`` for a multi-part upload.

    ONE expression, called by BOTH the mesher and the web app's ``estimate_tets``.
    They answer the same question -- how fine the mesh is, and how much material
    is in each region -- in two processes, and the row that asked for this
    feature names the trap exactly: if the estimate and the mesher derive the
    mesh parameters from different places, the app's route decision and the mesh
    the solver builds disagree, and the estimate is what routes.

    UNDER THE UNIFORM RULE the two quantities are:

    * ``edge_units`` is the SAME for every region: ``h_min``, the finest region
      edge (``edge_units * n_wall / max n_r``, floored at ``edge_units``). The
      whole cavity is meshed at it -- see ``region_edges`` for why per-region
      targets are not realisable in this TetGen build, which is why the plan
      reports one MESHING edge and not a distinct one per region.
    * ``own_edge_units`` is DIFFERENT per region: ``h_r = edge_units * n_wall /
      n_r``, the edge that region's OWN resolution forbids exceeding. It is what
      the resolution GUARD reads -- a region built coarser than
      ``MAX_EDGE_OVERSHOOT * own_edge_units`` is refused -- and it is NOT what
      the mesher builds to. A dense core's ``own_edge_units`` is far below
      ``edge_units``, so reading the guard's allowance from ``edge_units`` for
      every region lets a core be built several times coarser than its own rule
      while the guard passes. That is the defect ``own_edge_units`` closes.
    * the volume is the region's FILLED volume, not its enclosed one:
      ``PartSet.filled_volume_m3`` subtracts everything nested inside it. The
      wall's enclosed volume counts its core twice. (The estimate now works on
      the total cavity volume rather than summing ``V_r / h_r^3``, but the
      filled volume is still what the estimate must convert, and the per-region
      figure is what the mesher's own verbose line reports.)

    ``n_region`` comes from ``materials`` (part name -> n, naming every part).
    A region with no material is REFUSED by ``PartSet.region_materials`` rather
    than defaulted: the default would be vacuum, and the owner would get
    frequencies for a cavity that is partly empty with nothing saying so.

    ``units`` is the UPLOAD's unit and has no default. It reaches
    ``PartSet.filled_volume_m3``, which converts the region volume to cubic
    metres. There is no default here for the same reason there is none there:
    with one, a caller that forgot the argument silently got millimetres while
    ``edge_units`` beside it stayed true metres, making the estimate wrong by
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
    own_edges = region_edges(edge_units, n_wall, materials, parts)
    # ONE edge for every region: the finest region's own requirement, floored at
    # the wall's. Resolved as a MIN over the plan rather than recomputed from the
    # materials, so this is the value the estimate also routed on. `min(...,
    # edge_units)` rather than the raw ratio so a region LESS dense than the wall
    # cannot coarsen the wall: h_min is a resolution floor for the cavity, not a
    # function of the densest region alone.
    h_min = min([float(edge_units), *(float(e) for e in own_edges.values())])
    out = {}
    for part_index, region_id in enumerate(parts.region_of_part):
        out[int(region_id)] = {
            "part": parts.names[part_index],
            "edge_units": h_min,
            "own_edge_units": float(own_edges[int(region_id)]),
            "filled_volume_m3": parts.filled_volume_m3(part_index, units),
            "n": float(by_region[region_id]),
        }
    return out


def estimate_tets_by_region(parts, materials, edge_units: float,
                            surface, units: str, margin: float) -> int:
    """Element estimate for a multi-part upload: the VOLUME LAW, summed per region.

        est = margin * TETS_PER_UNIT_VOLUME * sum_r filled_V_r / h_r^3

    ``h_r`` is the edge the mesher applies IN REGION ``r`` -- the plan's own
    ``edge_units`` for that region, the same number the mesher hands
    ``target_maxvolume``. On today's mesher that number is the SAME for every
    region (the plan's uniform rule, applied in ``region_plan``: one cavity, one
    edge, ``h_min``), so the sum
    collapses to the one-part expression at ``h_min`` on the filled total.
    Written per region it follows the plan if a per-region target ever becomes
    realisable in this TetGen build, instead of silently describing a mesher
    that no longer exists. (The per-region mechanism was measured INERT -- a
    nine-order sweep of ``add_region``'s ``max_vol`` moved a region's own count
    by zero -- so that is the shape to expect if it is ever revisited.)

    ``filled_V_r`` is the region's FILLED volume, not its enclosed one: the
    wall's enclosed volume counts everything nested in it a second time, and the
    count is what ROUTES the job.

    ``margin`` is the MEASURED factor the volume law needs to stay at or above
    the mesh TetGen actually builds. TetGen's delivery is not the request:
    ``maxvolume`` (see ``TETGEN_VOLUME_SLACK``) is a CEILING on the element size,
    so a compliant mesh measures at or below the law in count, and TetGen refines
    well below it wherever its own quality rules demand -- measured at the
    product's cap, the delivered count runs 0.715x to 8.952x the law across the
    shape families. The estimate can only read the request, so the margin has to
    cover the worst case measured, not the average one, and no single margin can
    cover the whole measured range (the 8.952 end would make every other family
    read 6-12x its mesh). It is passed in, from the app's own config, and is
    deliberately NOT derived here: the sweep that measures it is
    ``benchmarks/estimate_vs_built.py --families`` and the tables it produced are
    in ``openspec/changes/fix-upload-tet-estimate/``.

    UNITS -- NO CONVERSION HAPPENS HERE, and that is a measured claim, not a
    stylistic one. This function produced a 2.3e13-element estimate for a 19k
    mesh on its first attempt because it multiplied by ``unit_scale(units)``
    "to get metres". Measured for the 6 mm fixture (wall n=2, core n=3.5, mm):

        effective_edge_m(...)        -> 1.0181282863016227e-03   [= 1.018 mm]
        region_edges(= h*2/3.5)      -> 5.817875921723558e-04    [= 1.018/1.75]
        filled_volume_m3 (sum)       -> 2.16e-07                 [cubic metres]
        unit_scale('mm')             -> 1.0e-03

    ``effective_edge_m`` returns METRES (1.018e-3 m is 1.018 mm, the right edge
    for a 6 mm box), ``region_edges`` divides by a bare index ratio, and the
    plan's ``h_min`` is a MIN over that same space (the raw edge and the
    per-region edges), so the plan's ``edge_units`` is in the SAME metres -- the
    name says "units" but the
    value is metres, because that is what the caller feeds it. ``filled_volume_m3``
    is cubic metres by construction. So the pair is already coherent and
    ``unit_scale`` here applies the factor a SECOND time, making the count wrong
    by ``1/scale^3`` = 1e9 for mm. That is the "one quantity, two spaces" shape
    this rig keeps meeting: the quantity was already in the target space and a
    second conversion was invisible to reading.
    """
    plan = region_plan(parts, materials, edge_units, units)
    if not plan:
        # No regions at all -- not a multi-part upload. Refuse rather than
        # return 0, which would look like "a tiny job" and route it to the GPU.
        raise MeshingError(
            "the multi-part estimate was asked for a mesh with no regions; "
            "this is the one-part path's job and calling it here would route "
            "a job on a number of zero."
        )
    law = 0.0
    by_edge: dict[float, float] = {}
    for region_id, info in plan.items():
        h_r = float(info["edge_units"])
        v_r = float(info["filled_volume_m3"])
        if not (h_r > 0) or not (v_r > 0):
            # Could not ask. A zero here would route a real job to the fast
            # machine on the strength of a number nothing computed -- refuse.
            raise MeshingError(
                f"region {region_id} of the plan gave h={h_r:g} m and "
                f"V={v_r:g} m3; neither can be used to size a mesh."
            )
        # Grouped by edge so that regions sharing an edge are summed before the
        # division: on today's uniform plan that makes this expression EXACTLY
        # the one-part count at ``h_min`` on the filled total (the identity the
        # suite pins), and a graded plan would still sum each edge's own volume.
        by_edge[h_r] = by_edge.get(h_r, 0.0) + v_r
    law = sum(v / (h ** 3) for h, v in by_edge.items())
    return int(float(margin) * TETS_PER_UNIT_VOLUME * law)
