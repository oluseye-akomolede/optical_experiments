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

#: coefficient of the upload-time tet estimate, ``est_tets = C * V / h^3``.
#: Fitted on 13 measured meshes (the lead's runs); C = 14 is the CONSERVATIVE
#: fit, chosen so the estimate is at or above the measured count on nearly
#: every point. Under-estimating sends a big job to the CPU where it dies;
#: over-estimating costs a small job about a second on the GPU. The asymmetry
#: is the whole reason for the constant.
EST_TETS_C = 14.0

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

#: an upload is refused outright only above TWICE that, because the estimate can
#: be up to 2x high (worst over-estimate measured) and refusing at 1x would turn
#: away jobs that fit.
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


def est_tets_c() -> float:
    """Coefficient of the upload-time estimate. See :data:`EST_TETS_C`."""
    try:
        value = float(os.environ.get("MSE_EST_TETS_C", EST_TETS_C))
    except (TypeError, ValueError):
        return EST_TETS_C
    return value if value > 0 else EST_TETS_C


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

    Twice the real cap, because the estimate can be up to 2x high: refusing at
    1x would turn away jobs that fit. Refusing here is the only guard that
    costs the owner nothing -- nothing has been meshed yet.
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
