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
UNIT_CHOICES = ("mm", "cm", "m", "in")
DEFAULT_UNITS = "mm"
DEFAULT_N_MODES = 20

#: settings for the solver Job (see the task-2 amendment)
JOB_CPU = "4"
JOB_MEMORY = "14Gi"
JOB_ACTIVE_DEADLINE_SECONDS = 1800
JOB_TTL_SECONDS_AFTER_FINISHED = 30

#: how often the background tick looks for queued uploads, in seconds
TICK_SECONDS = float(os.environ.get("MSE_TICK_SECONDS", "5"))

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


def max_running_jobs() -> int:
    """Concurrency for solver Jobs.

    The namespace ResourceQuota allows exactly one ``batch/v1`` Job, but the app
    keeps the knob so a raised quota needs only an env change.
    """
    try:
        return max(1, int(os.environ.get("MSE_MAX_RUNNING", "1")))
    except ValueError:
        return 1
