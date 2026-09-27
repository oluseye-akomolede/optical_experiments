"""The solver Job: its manifest, and launching/deleting it.

This is a plain ``batch/v1`` Job in the ``mode-search-engine`` namespace -- *not*
a SparkApplication, and nothing is ever created in ``default`` (see the task-2
amendment; the Spark operator only watches ``default``, which is outside the
owner's power cap).

``build_job_manifest`` is a **pure function**: same arguments in, same dict out,
no environment, no clock, no cluster. That is what the tests pin.

Concurrency is one. The namespace quota counts finished Jobs too, so a finished
Job is deleted as soon as its result is in the database rather than waiting for
``ttlSecondsAfterFinished``.
"""

from __future__ import annotations

import re

#: label values must be DNS-1123-ish; a job id is a uuid but be defensive
_UNSAFE = re.compile(r"[^a-z0-9.-]")

#: the solver CLI takes n-modes; the owner-facing default lives in config
DEFAULT_N_MODES = 20


def job_name(job_id) -> str:
    """Deterministic, DNS-1123-safe Kubernetes object name for a job id."""
    clean = _UNSAFE.sub("-", str(job_id).lower()).strip("-.")
    return f"mse-solver-{clean[:48]}"


def build_job_manifest(job_id, units: str, n_index: float, n_modes: int,
                       solver_image: str, pull_secret: str, namespace: str,
                       cpu: str = "4", memory: str = "14Gi",
                       active_deadline_seconds: int = 1800,
                       ttl_seconds_after_finished: int = 30,
                       cpu_max_tets: int | None = None,
                       min_feature_mm: float | None = None,
                       scale: float = 1.0):
    """Return the ``batch/v1`` Job manifest that solves one upload.

    Pure: no environment lookups, no cluster calls, no time.

    The shape is dictated by the amendment:
    ``envFrom`` the ``mse-minio``/``mse-db`` Secrets, ``imagePullSecrets``
    (the Harbor project is private, so the pull secret is required),
    ``restartPolicy: Never``, ``backoffLimit: 0`` (a failed solve must not be
    retried -- it would spend another ten minutes reaching the same error),
    and ``requests == limits`` so the Job is Guaranteed and the quota, not the
    scheduler, decides admission.
    """
    name = job_name(job_id)
    labels = {"app": "mse-solver", "job-id": str(job_id)}
    args = [
        "--job-id", str(job_id),
        "--units", str(units),
        "--n-index", repr(float(n_index)),
        "--n-modes", str(int(n_modes)),
    ]
    env = [
        # Named explicitly, not left to the solver's default. This image has no
        # CUDA stack, so "cpu" is the only value that can work here -- but a
        # default that agrees with the image is a default that hides a mistake,
        # and the backend is the first thing to check when a run is slow.
        {"name": "MSE_SOLVER_BACKEND", "value": "cpu"},
    ]
    if cpu_max_tets is not None:
        # The solver re-decides the route on the exact tetrahedron count, at the
        # first point that count exists. Handing it the same boundary the app
        # routed on is what makes that second decision agree with the first
        # instead of inventing a different one.
        env.append({"name": "MSE_CPU_MAX_TETS", "value": str(int(cpu_max_tets))})
    if min_feature_mm is not None:
        # The owner's declared smallest feature. Passed as a flag rather than an
        # env var because it is a property of THIS upload, not of the machine --
        # two jobs on one node have different feature sizes, and an env var is
        # the wrong shape for a per-job value.
        args += ["--min-feature", repr(float(min_feature_mm))]
    if scale is not None and float(scale) != 1.0:
        # Omitted when 1.0 so a job with no scaling is byte-identical to what
        # this function produced before the argument existed -- which is what
        # the pre-existing tests assert against.
        args += ["--scale", repr(float(scale))]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            # The quota tallies finished Jobs, so the TTL is only a backstop for
            # the tick having died; the app deletes the Job itself once the
            # result is recorded.
            "ttlSecondsAfterFinished": int(ttl_seconds_after_finished),
            "activeDeadlineSeconds": int(active_deadline_seconds),
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    # The solver only talks to MinIO and Postgres through env
                    # credentials; it never calls the Kubernetes API, so do not
                    # hand it a token that could be used to reach the cluster.
                    "automountServiceAccountToken": False,
                    "imagePullSecrets": [{"name": pull_secret}],
                    "containers": [
                        {
                            "name": "solver",
                            "image": solver_image,
                            "args": args,
                            "env": env,
                            "envFrom": [
                                {"secretRef": {"name": "mse-minio"}},
                                {"secretRef": {"name": "mse-db"}},
                            ],
                            "resources": {
                                "requests": {"cpu": cpu, "memory": memory},
                                "limits": {"cpu": cpu, "memory": memory},
                            },
                        }
                    ],
                },
            },
        },
    }


class K8sJobLauncher:
    """Create/delete the solver Job through the in-cluster Kubernetes API.

    Everything the request needs is passed in, so the object can be built in
    tests (and with ``dry_run``) without a cluster.
    """

    def __init__(self, namespace: str, image: str, pull_secret: str,
                 cpu: str = "4", memory: str = "14Gi",
                 active_deadline_seconds: int = 1800,
                 ttl_seconds_after_finished: int = 30, dry_run: bool = False,
                 cpu_max_tets: int | None = None):
        self.namespace = namespace
        self.image = image
        self.pull_secret = pull_secret
        self.cpu = cpu
        self.memory = memory
        self.cpu_max_tets = cpu_max_tets
        self.active_deadline_seconds = active_deadline_seconds
        self.ttl_seconds_after_finished = ttl_seconds_after_finished
        self.dry_run = dry_run
        self._client = None

    def _batch(self):
        # imported lazily so the app imports (and the tests run) with no kube
        # configuration present at all
        if self._client is None:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except Exception:  # noqa: BLE001 - fall back for local/dev use
                config.load_kube_config()
            self._client = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
        return self._client

    def manifest(self, job_id, units: str, n_index: float, n_modes: int,
                 min_feature_mm: float | None = None,
                 scale: float = 1.0) -> dict:
        return build_job_manifest(
            job_id, units, n_index, n_modes, self.image, self.pull_secret,
            self.namespace, cpu=self.cpu, memory=self.memory,
            active_deadline_seconds=self.active_deadline_seconds,
            ttl_seconds_after_finished=self.ttl_seconds_after_finished,
            cpu_max_tets=self.cpu_max_tets,
            min_feature_mm=min_feature_mm, scale=scale,
        )

    # -- cluster state -----------------------------------------------------
    def active_job_names(self) -> list[str]:
        """Names of Jobs with our labels that still occupy the quota slot.

        A Job counts against ``count/jobs.batch`` from creation until deletion,
        including while it is finished, so "occupying" means "exists".

        Raises on 403 rather than returning ``[]``. An empty list is a *claim*
        that nothing is there, and both of this method's callers act on that
        claim: the tick launches, and ``_fail_dead_jobs`` fails every running
        row. Swallowing a 403 into ``[]`` therefore starts a Job against a quota
        we could not read, or -- worse -- fails a live solve whose only problem
        was that we were not allowed to look at it. Letting the exception out
        makes each caller see "could not ask", which they already handle.
        """
        if self.dry_run:
            return []
        from kubernetes import client

        api = self._batch()
        try:
            jobs = api.list_namespaced_job(
                self.namespace, label_selector="app=mse-solver"
            )
        except client.ApiException as exc:
            if exc.status == 403:
                raise CannotAskCluster(
                    "not allowed to list solver jobs"
                ) from exc
            raise
        return [j.metadata.name for j in jobs.items]

    def is_quota_exceeded(self, exc: Exception) -> bool:
        """A 403 'exceeded quota' means wait, not fail.

        Distinct from :func:`_as_launch_error`, which turns the same exception
        into ``QuotaExceeded`` for the launch path; this is the predicate form
        the tests and any caller holding a bare exception use.
        """
        status = getattr(exc, "status", None)
        body = str(getattr(exc, "body", "") or str(exc))
        return status == 403 and "exceeded quota" in body.lower()

    def job_states(self) -> dict[str, str]:
        """``{job name: failure reason}`` for our Jobs that have FAILED.

        Only failures are reported, so a name absent from the result means
        "exists and has not failed". A caller distinguishes the three cases it
        cares about by crossing this with ``active_job_names``: running (present
        in both), dead-and-failed (in this dict), and gone (in neither).

        The reason is a best-effort plain word -- ``DeadlineExceeded`` from the
        Job's own condition, or a container's ``OOMKilled``/``Error`` -- and may
        be ``""`` when Kubernetes does not say. Callers must not depend on it
        being one of a fixed set.

        Raises on 403, for the same reason ``active_job_names`` does: an empty
        dict is indistinguishable from "every Job is healthy", and callers act
        on it.
        """
        if self.dry_run:
            return {}
        from kubernetes import client

        api = self._batch()
        try:
            jobs = api.list_namespaced_job(
                self.namespace, label_selector="app=mse-solver"
            )
        except client.ApiException as exc:
            if exc.status == 403:
                raise CannotAskCluster(
                    "not allowed to list solver jobs"
                ) from exc
            raise

        states: dict[str, str] = {}
        for job in jobs.items:
            job_status = getattr(job, "status", None)
            pods = None if _names_the_failure_itself(
                _job_condition_reason(job_status)) else self._job_pods(job)
            reason = _failed_reason(job, pods)
            if reason is None:
                continue
            # A job that died asking to be re-routed says so in its pod's last
            # state, which is the only channel a dead container has. `_failed_reason`
            # reads the pods for OOMKilled and would not recognise this, so the
            # pods are inspected once more here rather than in a second list call.
            if _REROUTE_REASON is not None and wants_reroute(pods):
                states[job.metadata.name] = _REROUTE_REASON
            else:
                states[job.metadata.name] = reason
        return states

    def _job_pods(self, job) -> list:
        """The pods belonging to ``job``, or ``[]`` if we cannot list them.

        A pod list is how a container's ``OOMKilled`` becomes visible -- the
        Job's status holds no pods. Failing to list them must not sink the whole
        tick: the Job's own condition still says *that* it failed, so the caller
        falls back to a reason it does have rather than losing the failure.
        """
        from kubernetes import client

        name = getattr(getattr(job, "metadata", None), "name", None)
        if not name:
            return []
        try:
            return list(self._core().list_namespaced_pod(
                self.namespace, label_selector=f"job-name={name}"
            ).items)
        except client.ApiException:
            # Not allowed, or the pod is already gone. The Job's own condition
            # still says *that* it failed, so the caller keeps that reason rather
            # than losing the failure to a failed lookup.
            return []
        except Exception as exc:  # noqa: BLE001
            # Nothing about naming an OOMKilled container is worth failing a
            # scheduler tick for. If the core API is unreachable for any reason
            # at all, fall back to the reason we already have -- but say so, so
            # a swallowed lookup cannot pass for "this Job has no pods".
            print(f"[mse-web] could not list pods for {name}: {exc}", flush=True)
            return []

    def _core(self):
        """The CoreV1 API, built alongside the batch one.

        ``_batch()`` short-circuits when a client is already assigned (the tests
        do that to run without a cluster), so the core client is built here if
        that path skipped it.
        """
        self._batch()
        if getattr(self, "_core_api", None) is None:
            from kubernetes import client

            self._core_api = client.CoreV1Api()
        return self._core_api

    # -- mutation ----------------------------------------------------------
    def create(self, job_id, units: str, n_index: float, n_modes: int,
               min_feature_mm: float | None = None, scale: float = 1.0) -> str:
        """Create the Job and return its name."""
        manifest = self.manifest(job_id, units, n_index, n_modes,
                                 min_feature_mm=min_feature_mm, scale=scale)
        if self.dry_run:
            return manifest["metadata"]["name"]
        from kubernetes import client

        api = self._batch()
        try:
            api.create_namespaced_job(self.namespace, manifest)
        except client.ApiException as exc:
            raise _as_launch_error(exc) from exc
        return manifest["metadata"]["name"]

    def delete(self, name: str) -> None:
        """Delete a Job (and its pods) once its result is recorded."""
        if self.dry_run or not name:
            return
        from kubernetes import client

        api = self._batch()
        try:
            api.delete_namespaced_job(
                name, self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
        except client.ApiException as exc:
            if exc.status != 404:  # already gone is success
                raise


def _job_condition_reason(status) -> str | None:
    """The ``Failed`` condition's reason, or ``None`` if it is not set yet.

    Split out so a caller can ask "does the Job itself already say why?" without
    paying for a pod list. Returns ``""`` (not ``None``) when the condition is
    set but carries no reason -- that is still a failure, so the caller must not
    read it as "keep looking".
    """
    for cond in (getattr(status, "conditions", None) or []):
        if getattr(cond, "type", "") == "Failed" and \
                getattr(cond, "status", "") == "True":
            return getattr(cond, "reason", "") or ""
    return None


def _names_the_failure_itself(reason: str | None) -> bool:
    """Whether a Job condition reason is the best answer we will get.

    Only ``DeadlineExceeded`` qualifies. It says the solver overran its own
    budget, and no pod can sharpen that.

    Every other failure under ``backoffLimit: 0`` arrives as
    ``BackoffLimitExceeded`` -- including the OOM this app most needs to name.
    Treating *any* condition as the whole answer is what hid ``OOMKilled``:
    the condition appears within seconds and then persists, so the pod list was
    skipped for exactly the steady state an owner would be looking at, and the
    container's ``OOMKilled`` was only ever visible in the brief window before
    the condition landed. So an unhelpful condition must fall through to the
    pods, which is why this is a check on the *reason* and not on
    ``reason is not None``.
    """
    return reason == "DeadlineExceeded"


def _failed_reason(job, pods=None) -> str | None:
    """A short reason string if the Job has FAILED, else ``None``.

    Kubernetes signals this two ways and we check both: the Job's own
    ``Failed`` condition (``DeadlineExceeded`` when ``activeDeadlineSeconds``
    ran out, ``BackoffLimitExceeded``), and a container's termination reason
    (``OOMKilled`` when the cgroup killed it). A Job with neither is not
    treated as failed -- it is either running or finished successfully.

    ``pods`` are the Job's pods, which the caller must list separately: a
    ``V1JobStatus`` describes the Job, not its pods -- ``status.failed`` is a
    *count*, not a list. Passing the Job alone therefore cannot name an
    ``OOMKilled`` container, and asking this function to try was the bug.
    """
    status = getattr(job, "status", None)
    if status is None:
        return None

    own = _job_condition_reason(status)
    if _names_the_failure_itself(own):
        return own
    if own is not None and not pods:
        # The Job says it failed and says nothing useful; without the pods there
        # is nothing better to report, so report what we have.
        return own

    # A specific container reason beats an unhelpful condition: ``BackoffLimitExceeded``
    # is what every failure under ``backoffLimit: 0`` gets, while ``OOMKilled``
    # tells the owner what to actually change. This is the case the owner most
    # needs named, which is why it is allowed to overwrite the condition.
    for reason in ("OOMKilled", "Error"):
        if _pods_have_reason(pods or [], reason):
            return reason
    return own if own is not None else None


def _pods_have_reason(pods, reason: str) -> bool:
    """Whether any container of any of these pods terminated with ``reason``.

    Reads the shape the Python client returns; a plain dict is accepted too so
    the tests can pass either. The ``state`` of a container is either a dict or
    an object depending on how it was built, so both are handled once, in
    ``_attr``, rather than guessed at per call site.
    """
    for pod in pods or []:
        pod_status = _attr(pod, "status") or pod
        for cs in (_attr(pod_status, "container_statuses") or []):
            state = _attr(cs, "state")
            if state is None:
                continue
            terminated = _attr(state, "terminated")
            if terminated is None and isinstance(state, dict):
                terminated = state.get("terminated")
            if _attr(terminated, "reason", "") == reason:
                return True
    return False


#: the token the solver writes to its termination log when it is on the wrong
#: machine. Kept in step with ``mse_maxwell_solver.REROUTE_TOKEN`` by the
#: tier-1 test that imports both and asserts they are equal -- a copy of a
#: constant that can drift is worse than no constant.
REROUTE_TOKEN = "MSE_REROUTE"


def wants_reroute(pods, token: str = REROUTE_TOKEN) -> bool:
    """Whether any of these pods asked to be re-routed to another machine.

    **Only a positive read counts.** A missing pod, a pod with no termination
    message, a message that was truncated, an unreadable field -- every one of
    them returns False, because the cost of the two mistakes is not symmetric:

    * Wrongly rerouting a job that genuinely failed (a bad mesh, a bug) puts it
      back in the queue, where it fails again on the other machine -- and, if the
      other machine also reroutes it, could ping-pong.
    * Wrongly *not* rerouting a job that asked costs the owner one upload, with
      "too large for the processors available" on the job page, which is honest
      and actionable.

    This is the same rule the rig calls *could-not-ask must not look like
    nothing-there*: "we could not read the message" is not evidence of anything
    and must not be treated as "yes".

    The token is matched as a substring because the solver writes
    ``"<token>: <explanation>"``; the explanation is for a human reading the pod
    and is deliberately not parsed.
    """
    for pod in pods or []:
        pod_status = _attr(pod, "status") or pod
        for cs in (_attr(pod_status, "container_statuses") or []):
            for state_name in ("last_state", "state"):
                state = _attr(cs, state_name)
                if state is None and isinstance(cs, dict):
                    state = cs.get(state_name)
                terminated = _attr(state, "terminated")
                if terminated is None and isinstance(state, dict):
                    terminated = state.get("terminated")
                if terminated is None:
                    continue
                message = _attr(terminated, "message", "") or ""
                if token in str(message):
                    return True
    return False


def _attr(obj, name, default=None):
    """``getattr`` that also reads a plain dict, for fakes and real objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


#: owner-facing text per known failure reason. Kept deliberately plain: this is
#: what appears on the job page, and the owner is not a Kubernetes operator.
FAILURE_MESSAGES = {
    "DeadlineExceeded": (
        "The solver took too long and was stopped. Try a simpler object, or "
        "ask for fewer modes."
    ),
    "OOMKilled": (
        "The solver ran out of memory. Try a simpler object, or ask for fewer "
        "modes."
    ),
}

#: used when the Job is gone or died without Kubernetes giving a reason
FAILURE_DEFAULT = (
    "The solver stopped unexpectedly. Please try a simpler object, or ask for "
    "fewer modes."
)


#: the reason string that means "this job is on the wrong machine, move it".
#: Deliberately not a Kubernetes reason and deliberately not in
#: ``FAILURE_MESSAGES``: it is never shown to the owner, it is acted on. It
#: exists so the reroute decision survives the trip through ``job_states``
#: (a dict of name -> string) without widening that contract.
_REROUTE_REASON = "MseReroute"


def failure_message(reason: str | None) -> str:
    """Plain English for the owner, from a Kubernetes reason.

    Unknown reasons fall back rather than leaking a Kubernetes token into the
    UI, and the reason is never interpolated verbatim.

    ``MseReroute`` deliberately falls back to the generic message: it is an
    instruction to the scheduler, not a diagnosis for a human, and if it ever
    reaches here the job has already failed to reroute (or rerouted and failed
    again) -- the owner then gets the actionable "too large for the processors
    available", not the word ``MseReroute``.
    """
    if reason:
        return FAILURE_MESSAGES.get(reason, FAILURE_DEFAULT)
    return FAILURE_DEFAULT


class JobLaunchError(Exception):
    """The cluster refused the Job for a reason worth telling the owner."""


class CannotAskCluster(Exception):
    """We could not read cluster state -- distinct from reading it and finding nothing.

    Raised when Kubernetes answers 403 to a list call. The two are not the same
    thing and callers must not treat them as one an empty result would: an empty
    list of Jobs means "there is nothing there", which is a licence to launch and
    to fail every running row; "we were not allowed to look" is a licence for
    neither. This used to be swallowed into an empty result, and the tick then
    failed live solves whose only fault was being invisible to us.
    """


def _as_launch_error(exc: Exception) -> Exception:
    """Translate a Kubernetes API error into something the app can handle."""
    status = getattr(exc, "status", None)
    body = str(getattr(exc, "body", "") or "")
    if status == 403 and "exceeded quota" in body.lower():
        return QuotaExceeded(body)
    if status == 403:
        return JobLaunchError(
            "The service account is not allowed to start solver jobs."
        )
    return JobLaunchError(f"the solver job could not be started: {body or exc}")


class QuotaExceeded(Exception):
    """The namespace is full. This is a 'wait', not a 'fail'."""
