"""Environment-driven configuration for the MSE web app.

Secrets and endpoints arrive as environment variables only (from the
``mse-minio`` / ``mse-db`` Secrets); nothing here is ever written to disk or
baked into the image. Every default matches the solver's own defaults so the app
and the Job cannot drift apart.
"""

from __future__ import annotations

import os

#: name of the namespace everything of ours lives in
NAMESPACE = os.environ.get("MSE_NAMESPACE", "mode-search-engine")

#: image the solver Job runs. Deliberately has NO default. The obvious fallback
#: (the old ``spark-fem-runner`` image) is the *scalar* solver, so a missing
#: variable would silently produce the wrong physics with no error at all --
#: worse than refusing. Set MSE_SOLVER_IMAGE in the Deployment (it is set in
#: manifests/mse/deployment.yaml); ``solver_image()`` raises if it is not.
DEFAULT_SOLVER_IMAGE = None

#: secret used to pull that image (the Harbor project is private)
DEFAULT_PULL_SECRET = "mse-harbor-pull"

#: object storage
DEFAULT_MINIO_ENDPOINT = "http://minio.minio.svc.cluster.local:9000"
DEFAULT_BUCKET = "mse-web"

#: uploads are capped at 50 MB by the task body
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

#: what the solver supports
UNIT_CHOICES = ("mm", "cm", "m", "um", "nm", "in")
DEFAULT_UNITS = "mm"
DEFAULT_N_MODES = 20

#: The smallest-feature control's OWN units, deliberately shorter than
#: ``UNIT_CHOICES`` and INDEPENDENT of the object's units. The feature is a
#: physical length: the owner says "0.2 mm" and means 2e-4 m whatever the object
#: was uploaded in. Sharing the object's widget invited the coupling that caused
#: the length-space bug this replaces, so the two lists are separate on purpose
#: and a test asserts this one is exactly these three.
FEATURE_UNIT_CHOICES = ("nm", "um", "mm")
DEFAULT_FEATURE_UNIT = "mm"

#: bounds on the free scale factor. Not a technical limit -- a sanity rail, so a
#: mistyped \"1e12\" is refused at upload rather than meshed into oblivion. The
#: upper bound is generous because a metre-scale object in nm units is a real
#: case (a 1 m model in nm is 1e9); the lower bound only has to exclude zero and
#: the negatives.
MAX_SCALE = 1e9
MIN_SCALE = 1e-9

#: bounds on a per-part refractive index, for the multi-material upload.
#:
#: The same acceptance as ``n_index`` (finite, >= 1) except for the top, which
#: ``n_index`` does not have: this is a form field repeated once per part, and
#: an accidental extra digit in one row is a much easier mistake to make than in
#: a single field. 10 is far above any real dielectric at these frequencies
#: (silicon is 3.5, and the roadmap's materials are all below 4), so a value
#: over it is a typo rather than an exotic material -- and a typo that reached
#: the mesher would refine its region by that factor, cubing the element count.
#: It is a rail, like MIN_SCALE, not a physical limit.
MAX_MATERIAL_N = 10.0
MIN_MATERIAL_N = 1.0

#: how many modes the owner may ask for. A picklist, not a free field: the mode
#: count is the single biggest lever on both the mesh size and the solve time
#: (it sets the target edge length, and the element count goes like h^-3), so an
#: accidental extra zero here is a much bigger mistake than a typo elsewhere.
MODE_CHOICES = (10, 20, 40)

#: settings for the solver Job (see the task-2 amendment)
JOB_CPU = "4"
JOB_MEMORY = "14Gi"
JOB_ACTIVE_DEADLINE_SECONDS = 1800
JOB_TTL_SECONDS_AFTER_FINISHED = 30

#: how often the background tick looks for queued uploads, in seconds
TICK_SECONDS = float(os.environ.get("MSE_TICK_SECONDS", "5"))

# --------------------------------------------------------------------------
# Routing: which machine an upload goes to (dispatch 2/3)
# --------------------------------------------------------------------------

#: how much margin the upload-time estimate keeps over the VOLUME LAW -- the
#: geometric count of regular tets that fit in the object's volume.
#:
#: ``est_tets = EST_TETS_MARGIN * 6*sqrt(2) * V / h^3``. The law itself is exact
#: (``mse_common.TETS_PER_UNIT_VOLUME``, the same constant the mesher inverts to
#: ask TetGen for an edge); the margin covers the fact that TetGen does not land
#: on the law. It is one-sided on purpose: the estimate ROUTES the job, so a job
#: routed as small is then built far larger than predicted on the machine chosen
#: for the predicted size, while over-estimating costs a small job a second of
#: queue. So the margin is meant to be the WORST under-estimate (the highest
#: ``built / law``) measured across the shape families, measured rather than
#: fitted -- except that the sweep found under-estimates ABOVE it, which no
#: margin can cover without wrecking the other direction (see below). See
#: ``benchmarks/estimate_vs_built.py`` and the tables in
#: ``openspec/changes/fix-upload-tet-estimate/proposal.md``.
#:
#: Measured at the PRODUCT's own cap, the worst ``built / law`` is 1.414 (a
#: one-part 6 mm cube with a 3.5x finer core: 358,307 elements built against an
#: estimate of 342,129, so the estimate reads 0.95x of the mesh). 1.35 covers the
#: other ten builds in the shipped sweep, but not that one, and not the same
#: families at HALF their mode counts (a legitimate upload: the user picks the
#: mode count), where a bumpy sphere's fine facets floor the delivery at 1.868 of
#: the law and the estimate reads 0.72x of the real mesh.
#:
#: One family is far outside all of this: a 3-region object of nested thin
#: sheets, where TetGen over-refines 3x below the edge it was asked for (94.3% of
#: its tetrahedra are more than an octave finer than the request), builds 419,176
#: elements where the law says 46,825, and leaves the estimate reading 0.15x of
#: the real mesh. No margin over the law reaches that: covering 8.952 would make
#: the ten other families read 6-12x their meshes and turn their routing into
#: guesswork. It is a MESHER defect -- recorded in the change's proposal with its
#: fixture numbers, and out of scope for an estimator change (the lead,
#: 2026-10-15).
#:
#: So 1.35 is a measurement of one point in the (mode count, geometry) space and
#: not a bound over it, and it is left there rather than raised: raising it to 1.9
#: (enough to cover the swept 1.868) scales every estimate by 1.9/1.35 = 1.41, so
#: the worst over-estimate becomes 2.1 * 1.41 = 2.95 -- and it drags the refusal
#: threshold down with it, to 2 * 600,000 * 0.715 / 1.9 = 451,579 real elements on
#: the plate family, i.e. it would refuse an upload at 75% of the cap that fits.
#: Which failure the product prefers is the open decision recorded in the change's
#: proposal; a constant cannot remove both.
EST_TETS_MARGIN = 1.35

#: at or above this estimate the job is queued for the Jetson GPU instead of the
#: cluster. This is the cluster's *routing* boundary, not its capability limit;
#: the solver re-decides on the exact count (MSE_CPU_MAX_TETS).
DEFAULT_GPU_MIN_TETS = 15_000

#: the largest mesh either machine can solve (the lead's measured largest
#: success in single precision). The solver enforces it on the exact count.
#:
#: IMPORTED from the solver, not restated. This number existed in two places and
#: the two disagreed: the app capped uploads at 600_000 while the solver's own
#: default was 80_000, and nothing ever passed the app's value to the solver, so
#: every job between the two -- exactly the large jobs the GPU exists for --
#: failed inside the solver with a limit the app had already approved. One
#: import makes that impossible rather than merely fixed.
#:
#: The values come from ``mse_common``, NOT from ``mse_maxwell_solver``, and the
#: difference is not stylistic -- it is a production outage. The web image is
#: built from ``dockerfiles/web/Dockerfile``, which copies ``web/``, ``db/`` and
#: this one shared module. It does NOT copy the solver package, so importing the
#: solver here built, pushed, and then died at container start:
#:
#:     /opt/app/config.py:112
#:     ModuleNotFoundError: No module named 'mse_maxwell_solver'
#:
#: Every test passed, because the tests put ``dockerfiles/maxwell_solver`` on
#: ``sys.path``; nothing in the suite started the web IMAGE. ``mse_common`` is
#: the part both images share and imports only numpy, so both can have it. If
#: the module cannot be found the app FAILS TO IMPORT rather than falling back
#: to a restated number -- a silent fallback is the defect this import removes.
def _common_dir() -> str:
    """Put the shared module's directory on the path, wherever it lives here.

    Two layouts, one line each: in the repo the shared module sits in
    ``dockerfiles/maxwell_solver/`` beside the solver, and in the image both are
    flattened into ``/opt/app/``. The Dockerfile sets ``PYTHONPATH`` for the
    image case, so this is the checkout case; both are needed because the same
    source tree is imported both ways and a path that works in only one of them
    is exactly how the crashloop got in.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return str(root)


_common_dir()
from mse_common import DEFAULT_CPU_MAX_TETS, DEFAULT_MAX_TETS  # noqa: E402

#: the widest the estimate has been measured to read ABOVE the real mesh, as
#: ``est_tets / built``. A RECORD of the sweep, not an input to a derivation.
#:
#: This is a property of the MESHER, not of the formula. Across nine shape
#: families (single part, 2 and 3 regions, thin and thick shells, off-centre,
#: sphere, plate) the delivered mesh came back anywhere from 0.715x to 8.952x
#: the volume law: one global request per object, and TetGen's own quality
#: refinement deciding what it actually builds, nowhere near the request. The
#: estimate can only read the request, so it misses in BOTH directions:
#:
#:   * coarse deliveries (plate, cube, thin wall: 0.643-0.805 of the law) make
#:     the estimate read high -- 1.89x the real mesh on the plate family at the
#:     shipped setting, 2.098x at half the mode counts, which is the number
#:     recorded here.
#:   * fine deliveries make it read low, and one family does so by 6.7x; see
#:     EST_TETS_MARGIN and the proposal's found issue.
#:
#: 2.1 is 2.098 rounded to a tenth: a high-water mark of the measured sweep, not
#: a bound over the product. The 14x between the two ends IS the mesher's
#: delivery spread, and it is invariant to the estimate -- the constant moves
#: both ends together. Tightening the mesher's volume bound does not narrow it
#: either: ``benchmarks/mesher_slack_probe.py`` re-runs these families at
#: ``TETGEN_VOLUME_SLACK`` 4.0/2.0/1.0/0.5 and the delivery gets FINER as the
#: bound tightens (built/volaw 0.715-8.952 at 4.0, 1.472-3.818 at 2.0), because
#: the bound is a ceiling and TetGen refines below it for quality. The tables are
#: in ``openspec/changes/fix-upload-tet-estimate/proposal.md``.
#:
#: This is DELIBERATELY not the refusal factor; see below.
EST_TETS_WORST_OVER = 2.1

#: an upload is refused outright only above this many times the real cap.
#:
#: UNCHANGED by this change: 2, the value main ships. The two-sided derivation
#: belongs here rather than in the change, because its answer is that no single
#: factor is both safe and tight over the measured spread:
#:
#:   * never wrongly refuse -- accept every upload whose real mesh fits -- needs
#:     a factor at or above the worst over-estimate, 2.1 (the half-mode-count
#:     cube). The lead ruled wrongly accepting an oversized mesh the worse
#:     failure (2026-10-15), and widening the factor widens that arm.
#:   * never wrongly accept needs the factor at or below 1/2.1 = 0.48, which
#:     would refuse almost every upload.
#:   * 2 sits just inside the first bound. The refusal threshold is
#:     2 x 600,000 x built/volaw / 1.35 = 888,889 x built/volaw real elements:
#:     at the shipped setting the coarsest family measured (0.715) puts it at
#:     635,556, above the cap, and only the half-mode-count cube (0.643) reaches
#:     below it, to 571,000 -- 5% under the cap, and the whole of this change's
#:     wrongful-refusal exposure. Main's fitted constant put the same threshold
#:     at 557,000 for the plate and the cube, so this change refuses strictly
#:     less than main and never more.
#:   * the other arm, measured the same way: acceptance needs
#:     ``est <= 1.2M``, so the largest real mesh that slips through at a family's
#:     ratio r = est/real is 1.2M/r -- nothing above 1.26M at the shipped setting
#:     (smallest r 0.95), and up to 1.67M at half the mode counts (smallest r
#:     0.72), i.e. 2.8x the cap, which the mesher's own over-cap retry then
#:     coarsens to fit. A quality loss, not an OOM.
#:
#: The spread is not a property of the formula, so no choice of factor narrows
#: it: the delivered element size varies 0.34-0.71 mm across family plans that
#: are near-identical, and the mesher has no per-region lever to steer it
#: (per-region ``max_vol`` is measured inert -- see ``mse_meshing.py``).
#: Narrowing it is a MESHER row. Re-deriving this factor is the lead's decision.
UPLOAD_REFUSE_FACTOR = 2

#: how long a GPU job may sit unclaimed before the app gives up and puts it back
#: on the cluster queue, in seconds.
DEFAULT_GPU_WAIT_SECONDS = 600

#: how long a worker's lease is valid; a worker renews it on every progress
#: call, so an expired lease means the worker died, not that the job is slow.
DEFAULT_LEASE_SECONDS = 300

#: set MSE_DRY_RUN=1 to log what would be launched instead of launching it.
#: Used by the tests; never set in the cluster.
DRY_RUN = os.environ.get("MSE_DRY_RUN", "") not in ("", "0", "false", "False")


class ConfigError(Exception):
    """The app is misconfigured in a way the operator must fix."""


def solver_image() -> str:
    """Image reference for the solver Job.

    Raises rather than falling back: the only historical default is the old
    scalar solver, and launching that would report confidently wrong modes.
    """
    image = os.environ.get("MSE_SOLVER_IMAGE") or DEFAULT_SOLVER_IMAGE
    if not image:
        raise ConfigError(
            "MSE_SOLVER_IMAGE is not set, so there is no solver image to run. "
            "Set it in the Deployment (see manifests/mse/deployment.yaml). It "
            "is not defaulted on purpose: the old default was the scalar "
            "solver, which would return wrong modes silently."
        )
    return image


def pull_secret() -> str:
    """Name of the imagePullSecret; the Harbor project is private."""
    return os.environ.get("MSE_PULL_SECRET") or DEFAULT_PULL_SECRET


def bucket() -> str:
    return os.environ.get("MSE_BUCKET") or DEFAULT_BUCKET


def minio_endpoint() -> str:
    return os.environ.get("MINIO_ENDPOINT") or DEFAULT_MINIO_ENDPOINT


def _positive_int(env: str, default: int) -> int:
    """An int from the environment, falling back on anything unusable.

    A typo in a threshold must not take the app down at import: the defaults are
    the values the lead calibrated, and a bad override is a misconfiguration
    worth ignoring rather than a reason to refuse every upload.
    """
    try:
        value = int(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def est_tets_margin() -> float:
    """Margin over the volume law. See :data:`EST_TETS_MARGIN`."""
    try:
        value = float(os.environ.get("MSE_EST_TETS_MARGIN", EST_TETS_MARGIN))
    except (TypeError, ValueError):
        return EST_TETS_MARGIN
    return value if value > 0 else EST_TETS_MARGIN


def gpu_min_tets() -> int:
    """Estimate at or above which a job is queued for the Jetson."""
    return _positive_int("MSE_GPU_MIN_TETS", DEFAULT_GPU_MIN_TETS)


def max_tets() -> int:
    """The absolute element cap, mirrored from the solver's own limit.

    The default is ``mse_common.DEFAULT_MAX_TETS`` -- the same object the solver
    reads, not a copy of it. See the import comment above for why the web app
    must reach it without importing the solver package.
    """
    return _positive_int("MSE_MAX_TETS", DEFAULT_MAX_TETS)


def cpu_max_tets() -> int:
    """Element count above which the CLUSTER cannot solve a job but the GPU can.

    A routing boundary, not a capability limit -- the same distinction the
    solver makes at ``mse_common.DEFAULT_CPU_MAX_TETS``. The app sends it
    to every cluster Job so the solver's re-decision on the exact count lands in
    the same place as the app's upload-time decision, instead of re-deciding with
    a constant the app never saw. ``MSE_CPU_MAX_TETS`` overrides it.
    """
    return _positive_int("MSE_CPU_MAX_TETS", DEFAULT_CPU_MAX_TETS)


def upload_refuse_tets() -> int:
    """Estimate above which an upload is refused before it is ever queued.

    ``UPLOAD_REFUSE_FACTOR`` times the real cap, unchanged by this change.
    Refusing here is the only guard that costs the owner nothing -- nothing has
    been meshed yet -- but it is also the only one that can refuse something
    that would have worked, and the measured spread does not admit a factor that
    avoids both sides of that trade. See :data:`UPLOAD_REFUSE_FACTOR` for the
    two-sided derivation, the numbers, and which failure it prefers.
    """
    return UPLOAD_REFUSE_FACTOR * max_tets()


def gpu_wait_seconds() -> float:
    """How long an unclaimed GPU job waits before falling back to the cluster."""
    try:
        return max(1.0, float(os.environ.get("MSE_GPU_WAIT_S",
                                             DEFAULT_GPU_WAIT_SECONDS)))
    except (TypeError, ValueError):
        return float(DEFAULT_GPU_WAIT_SECONDS)


def lease_seconds() -> float:
    """Validity of a worker lease, in seconds."""
    try:
        return max(1.0, float(os.environ.get("MSE_LEASE_S",
                                             DEFAULT_LEASE_SECONDS)))
    except (TypeError, ValueError):
        return float(DEFAULT_LEASE_SECONDS)


def max_running_jobs() -> int:
    """Concurrency for solver Jobs.

    The namespace ResourceQuota allows exactly one ``batch/v1`` Job, but the app
    keeps the knob so a raised quota needs only an env change.
    """
    try:
        return max(1, int(os.environ.get("MSE_MAX_RUNNING", "1")))
    except ValueError:
        return 1
