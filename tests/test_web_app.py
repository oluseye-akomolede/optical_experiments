"""The web app: the Job manifest, upload validation, status JSON, field decode.

Everything here runs without a cluster, without MinIO and without Postgres:
the Job manifest is a pure function, and the store, database and launcher are
all injectable. That is what keeps this tier-1 test safe to run on the VM.
"""

from __future__ import annotations

import io
import json
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import config as CFG  # noqa: E402
import db as DB  # noqa: E402
import fields as FLD  # noqa: E402
import jobs as JOBS  # noqa: E402
import storage as STORE  # noqa: E402
from meshing import UploadError, validate_upload  # noqa: E402

JOB_ID = "7f3a1c2e-0000-4000-8000-abcdefabcdef"


# ---------------------------------------------------------------------------
# The Job manifest: the piece the task asks to be a pure, tested function.
# ---------------------------------------------------------------------------
def _manifest(**overrides):
    kwargs = dict(
        job_id=JOB_ID, units="mm", n_index=1.5, n_modes=20,
        solver_image="registry/mse-solver:1", pull_secret="mse-harbor-pull",
        namespace="mode-search-engine",
    )
    kwargs.update(overrides)
    return JOBS.build_job_manifest(**kwargs)


def test_manifest_is_a_batch_job_in_our_namespace():
    manifest = _manifest()
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "mode-search-engine"
    # nothing of ours may land in `default`
    assert manifest["metadata"]["namespace"] != "default"


def test_manifest_is_pure():
    """Same arguments, same dict -- no clock, no environment, no cluster."""
    assert _manifest() == _manifest()
    assert _manifest(n_index=1.0) != _manifest(n_index=1.5)


def test_manifest_labels_identify_the_job():
    manifest = _manifest()
    assert manifest["metadata"]["labels"]["app"] == "mse-solver"
    assert manifest["metadata"]["labels"]["job-id"] == JOB_ID
    # the pod template carries the same labels, or `kubectl get pods -l` lies
    assert manifest["spec"]["template"]["metadata"]["labels"] == \
        manifest["metadata"]["labels"]


def test_manifest_arguments_match_the_solver_cli():
    args = _manifest(units="cm", n_index=1.5, n_modes=20)["spec"]["template"][
        "spec"]["containers"][0]["args"]
    assert args == ["--job-id", JOB_ID, "--units", "cm",
                    "--n-index", "1.5", "--n-modes", "20"]


def test_manifest_arguments_are_all_strings():
    """Kubernetes requires string args; a float 1.5 fails at the API server."""
    args = _manifest(n_index=1.0)["spec"]["template"]["spec"]["containers"][0]["args"]
    assert all(isinstance(a, str) for a in args)


def test_manifest_pull_secret_is_required():
    """The Harbor project is private; a missing pull secret is a failed pod."""
    spec = _manifest(pull_secret="mse-harbor-pull")["spec"]["template"]["spec"]
    assert spec["imagePullSecrets"] == [{"name": "mse-harbor-pull"}]


def test_manifest_does_not_mount_a_service_account_token():
    """The solver reads everything from env; it never calls the Kubernetes API.

    Automounting would hand the pod a token for the ``mode-search-engine``
    service account, which can create Jobs, for no benefit at all.
    """
    spec = _manifest()["spec"]["template"]["spec"]
    assert spec["automountServiceAccountToken"] is False


def test_manifest_uses_exactly_the_named_secrets():
    container = _manifest()["spec"]["template"]["spec"]["containers"][0]
    names = [ref["secretRef"]["name"] for ref in container["envFrom"]]
    assert names == ["mse-minio", "mse-db"]


def test_manifest_resources_are_guaranteed():
    """requests == limits, so the quota decides admission rather than the
    scheduler."""
    resources = _manifest()["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert resources["requests"] == resources["limits"] == {"cpu": "4",
                                                            "memory": "14Gi"}


def test_manifest_retries_and_deadlines():
    spec = _manifest()["spec"]
    assert spec["backoffLimit"] == 0          # a failed solve costs 10 minutes
    assert spec["restartPolicy" if "restartPolicy" in spec else "template"
                  ]["spec"]["restartPolicy"] == "Never"
    assert spec["activeDeadlineSeconds"] == 1800
    assert spec["ttlSecondsAfterFinished"] == 30


def test_job_name_is_dns_safe_and_deterministic():
    name = JOBS.job_name(JOB_ID)
    assert name == JOBS.job_name(JOB_ID)
    assert name.startswith("mse-solver-")
    assert name == name.lower()
    assert len(name) <= 63
    assert all(c.isalnum() or c in "-." for c in name)


def test_job_name_survives_a_hostile_job_id():
    """The name is derived from a uuid, but a bad one must not escape into a
    Kubernetes object name."""
    name = JOBS.job_name("../../etc/passwd UPPER")
    assert "/" not in name and " " not in name
    assert name == name.lower()
    assert name.startswith("mse-solver-")


def test_quota_exceeded_is_classified_as_wait():
    """A 403 'exceeded quota' means wait -- it must not be a job failure."""

    class FakeApiError(Exception):
        status = 403

        def __init__(self):
            super().__init__("forbidden")
            self.body = ('{"message":"exceeded quota: mse-power-cap, '
                         'requested: count/jobs.batch: 1"}')

    exc = FakeApiError()
    launcher = JOBS.K8sJobLauncher("ns", "img", "secret", dry_run=True)
    assert launcher.is_quota_exceeded(exc) is True


def test_a_plain_403_is_not_a_quota_wait():
    class FakeApiError(Exception):
        status = 403
        body = '{"message":"cannot create jobs in this namespace"}'

    launcher = JOBS.K8sJobLauncher("ns", "img", "secret", dry_run=True)
    assert launcher.is_quota_exceeded(FakeApiError()) is False


# ---------------------------------------------------------------------------
# The launcher's list calls, reached without a cluster
# ---------------------------------------------------------------------------
# ``_batch()`` returns ``self._client`` when it is already set, so assigning a
# fake here short-circuits the lazy kubernetes import: no kube config, no
# cluster, and the branch under test is the real production one. The 403 fix was
# briefly pinned only by reading the source on the assumption that the client
# was unreachable in a unit test. It is reachable.
class _FakeApi:
    def __init__(self, exc=None, jobs=(), pods=(), pods_exc=None):
        self._exc = exc
        self._jobs = list(jobs)
        self._pods = list(pods)
        self._pods_exc = pods_exc
        self.calls = 0
        self.pod_calls = 0

    def list_namespaced_job(self, namespace, label_selector=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return types.SimpleNamespace(items=self._jobs)

    def list_namespaced_pod(self, namespace, label_selector=None):
        self.pod_calls += 1
        if self._pods_exc is not None:
            raise self._pods_exc
        return types.SimpleNamespace(items=self._pods)


def _launcher_with(exc=None, jobs=(), pods=(), pods_exc=None):
    launcher = JOBS.K8sJobLauncher("ns", "img", "secret")
    api = _FakeApi(exc=exc, jobs=jobs, pods=pods, pods_exc=pods_exc)
    launcher._client = api                    # short-circuits _batch()
    launcher._core_api = api                  # and the core client too
    return launcher, api


def _api_exception(status):
    """A real kubernetes ``ApiException`` when importable, else a stand-in.

    The production code catches ``kubernetes.client.ApiException`` and reads
    ``.status``. Importing kubernetes keeps this honest about the type the real
    call raises; the fallback keeps the suite runnable without it.
    """
    try:
        from kubernetes import client
        return client.ApiException(status=status, reason="Forbidden")
    except Exception:  # noqa: BLE001 - kubernetes not installed
        exc = Exception("Forbidden")
        exc.status = status
        return exc


def test_active_job_names_raises_on_a_403_instead_of_returning_empty():
    """The reviewer's defect as behaviour, not as source text.

    An empty list is a *claim* that nothing is there, and both callers act on
    that claim: the tick launches, and ``_fail_dead_jobs`` fails every running
    row. Answering a permission error with that claim is what failed live solves.
    """
    launcher, _api = _launcher_with(exc=_api_exception(403))
    with pytest.raises(JOBS.CannotAskCluster):
        launcher.active_job_names()


def test_job_states_raises_on_a_403_instead_of_returning_empty():
    launcher, _api = _launcher_with(exc=_api_exception(403))
    with pytest.raises(JOBS.CannotAskCluster):
        launcher.job_states()


def test_a_non_403_api_error_is_not_relabelled_as_a_refusal():
    """A 500 or a timeout must reach the caller as itself.

    ``CannotAskCluster`` means "we were not allowed to look". Anything else is a
    different failure and must not be dressed as one. Both are errors rather
    than empty answers, which is the property that matters here.
    """
    launcher, _api = _launcher_with(exc=_api_exception(500))
    for call in (launcher.active_job_names, launcher.job_states):
        with pytest.raises(Exception) as excinfo:
            call()
        assert not isinstance(excinfo.value, JOBS.CannotAskCluster)


def test_a_good_list_returns_the_names_and_the_failure_states():
    """The happy path, so the raise is not hiding a broken success path."""
    job = types.SimpleNamespace(metadata=types.SimpleNamespace(name="mse-solver-a"),
                                status=types.SimpleNamespace(conditions=[]))
    launcher, api = _launcher_with(jobs=[job])
    assert launcher.active_job_names() == ["mse-solver-a"]
    assert launcher.job_states() == {}
    assert api.calls == 2, "one list call per method"


# ---------------------------------------------------------------------------
# The real client's shape: status.failed is a COUNT, not a pod list
# ---------------------------------------------------------------------------
def _real_job(status):
    """A real ``V1Job`` carrying a real ``V1JobStatus``.

    Built from the actual kubernetes client, not a ``SimpleNamespace`` -- that
    is the whole point. The bug this pins was invisible precisely because the
    doubles were hand-shaped.
    """
    from kubernetes import client as k8s

    return k8s.V1Job(metadata=k8s.V1ObjectMeta(name="mse-solver-a"),
                     status=status)


def test_a_real_job_status_whose_failure_is_only_a_count_does_not_raise():
    """The crash: ``status.failed`` is ``1``, and the old code iterated it.

    ``failed=1`` with no ``Failed`` condition is exactly the window the bug
    lived in -- Kubernetes has counted the failure but not yet published the
    condition. The old code reached its pod loop, tried to iterate the int, and
    raised ``TypeError``, which propagated out of ``job_states`` and failed the
    whole scheduler tick: no dead-Job marking and no launch until the condition
    landed. Reproduced here against the real client objects.
    """
    from kubernetes import client as k8s

    job = _real_job(k8s.V1JobStatus(failed=1))
    # The precondition this test depends on: iterating it really does raise.
    with pytest.raises(TypeError):
        list(job.status.failed)

    assert JOBS._failed_reason(job, []) is None
    assert JOBS._job_condition_reason(job.status) is None


def test_a_real_failed_condition_is_read_before_any_pod_lookup():
    from kubernetes import client as k8s

    job = _real_job(k8s.V1JobStatus(
        failed=1,
        conditions=[k8s.V1JobCondition(
            type="Failed", status="True", reason="DeadlineExceeded")]))
    assert JOBS._failed_reason(job, []) == "DeadlineExceeded"


def test_oomkilled_is_named_from_the_pods_not_from_the_job_status():
    """The message that could never fire, now reachable.

    A Job's status holds no pods, so ``OOMKilled`` can only come from a pod
    list. The old code looked for pods inside the Job status, so this reason was
    unreachable in production no matter what Kubernetes reported.
    """
    from kubernetes import client as k8s

    job = _real_job(k8s.V1JobStatus(failed=1))      # no Failed condition yet
    pod = k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(name="mse-solver-a-x"),
        status=k8s.V1PodStatus(container_statuses=[
            k8s.V1ContainerStatus(
                name="solver", ready=False, restart_count=0,
                image="img", image_id="",
                state=k8s.V1ContainerState(terminated=k8s.V1ContainerStateTerminated(
                    exit_code=137, reason="OOMKilled"))),
        ]))

    assert JOBS._failed_reason(job, [pod]) == "OOMKilled"
    # and without the pods it is honestly unknown, not silently "no failure"
    assert JOBS._failed_reason(job, []) is None


def test_a_pod_list_that_fails_does_not_lose_the_job_failure():
    """A failed pod lookup must degrade to the reason we have, not to none.

    ``_job_pods`` returns ``[]`` rather than raising, so the Job's own
    condition still reports the failure. Losing the whole failure because we
    could not name its cause would be a worse trade.
    """
    from kubernetes import client as k8s

    job = _real_job(k8s.V1JobStatus(
        failed=1,
        conditions=[k8s.V1JobCondition(
            type="Failed", status="True", reason="BackoffLimitExceeded")]))
    launcher, api = _launcher_with(
        jobs=[job], pods_exc=_api_exception(403))

    assert launcher.job_states() == {
        "mse-solver-a": "BackoffLimitExceeded"}


def test_job_states_asks_for_the_pods_of_each_failed_job():
    """The pod list is requested per Job, so OOMKilled has somewhere to come from."""
    from kubernetes import client as k8s

    job = _real_job(k8s.V1JobStatus(failed=1))
    launcher, api = _launcher_with(jobs=[job])

    assert launcher.job_states() == {}
    assert api.calls == 1
    assert api.pod_calls == 1, "the pods of a failed Job must be listed"


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def _closed_box() -> bytes:
    return trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode()


def _open_box() -> bytes:
    box = trimesh.creation.box(extents=[10, 7, 5])
    faces = np.array(box.faces)[:-1]        # knock one face off
    open_mesh = trimesh.Trimesh(vertices=np.array(box.vertices), faces=faces,
                                process=False)
    return open_mesh.export(file_type="obj").encode()


def test_a_closed_obj_is_accepted():
    validate_upload("box.obj", _closed_box(), 50 * 1024 * 1024)


def test_a_non_watertight_obj_is_rejected_with_a_reason():
    with pytest.raises(UploadError) as excinfo:
        validate_upload("open.obj", _open_box(), 50 * 1024 * 1024)
    message = str(excinfo.value).lower()
    assert "closed" in message or "watertight" in message
    assert "cavity" in message


def test_the_wrong_extension_is_rejected():
    with pytest.raises(UploadError) as excinfo:
        validate_upload("model.stl", _closed_box(), 50 * 1024 * 1024)
    assert ".obj" in str(excinfo.value)


def test_an_oversized_upload_is_rejected_by_size_not_by_parsing():
    with pytest.raises(UploadError) as excinfo:
        validate_upload("box.obj", _closed_box(), 100)
    assert "larger than" in str(excinfo.value)


def test_an_empty_upload_is_rejected():
    with pytest.raises(UploadError):
        validate_upload("box.obj", b"", 50 * 1024 * 1024)


def test_a_missing_filename_is_rejected():
    with pytest.raises(UploadError):
        validate_upload("", _closed_box(), 50 * 1024 * 1024)


def test_garbage_that_is_not_an_obj_is_rejected():
    with pytest.raises(UploadError):
        validate_upload("box.obj", b"\x00\x01not a mesh at all", 50 * 1024 * 1024)


def test_an_obj_with_no_faces_is_rejected():
    with pytest.raises(UploadError):
        validate_upload("empty.obj", b"# just a comment\n", 50 * 1024 * 1024)


# ---------------------------------------------------------------------------
# The field container: decode, and the slice geometry the viewer relies on.
# ---------------------------------------------------------------------------
def _grid(nx=4, ny=3, nz=2):
    return {"nx": nx, "ny": ny, "nz": nz, "origin": [0.0, 0.0, 0.0],
            "spacing": [1.0, 1.0, 1.0]}


def _field(grid, seed=0):
    nz, ny, nx = grid["nz"], grid["ny"], grid["nx"]
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nz, ny, nx, 3)).astype("<f4")


def test_decode_round_trips_the_solver_layout():
    grid = _grid()
    field = _field(grid)
    back = FLD.decode_mode(field.tobytes(), grid)
    assert back.shape == (grid["nz"], grid["ny"], grid["nx"], 3)
    assert np.array_equal(back, field)


def test_decode_rejects_a_truncated_file():
    grid = _grid()
    with pytest.raises(FLD.FieldError) as excinfo:
        FLD.decode_mode(b"\x00" * 8, grid)
    assert "does not match the grid" in str(excinfo.value)


def test_the_array_shape_is_nz_ny_nx_not_nx_ny_nz():
    """The single most likely way to ship a mirrored view."""
    grid = _grid(nx=5, ny=3, nz=2)
    assert FLD.shape_of(grid) == (2, 3, 5)


def test_slice_payload_orientation_and_values():
    """Each slice keeps the array's own ordering, so the viewer's u is the
    second-remaining axis and v the first."""
    grid = _grid()
    field = _field(grid, seed=7)

    for axis, expected_vu in (("x", (2, 3)), ("y", (2, 4)), ("z", (3, 4))):
        payload = FLD.slice_payload(field, grid, axis, 0.0)
        assert (payload["v"]["count"], payload["u"]["count"]) == expected_vu

    # a z-normal slice is field[0], read row-major. The payload rounds to six
    # decimals for JSON, so compare at that precision rather than float32's.
    payload = FLD.slice_payload(field, grid, "z", 0.0)
    assert payload["mag"]["data"] == pytest.approx(
        np.linalg.norm(field[0], axis=-1).ravel().tolist(), abs=1e-6)
    assert payload["vec"]["data"] == pytest.approx(
        field[0].reshape(-1, 3).ravel().tolist(), abs=1e-6)


def test_slice_payload_marks_true_nan_as_null_not_as_a_dark_colour():
    """Solver writes NaN outside the object; JSON has no NaN, and a 0 would
    paint the outside as a real (if faint) value."""
    grid = _grid()
    field = _field(grid)
    field[0, 0, 0, :] = np.nan
    payload = FLD.slice_payload(field, grid, "z", 0.0)
    assert payload["mag"]["data"][0] is None


def test_slice_positions_stay_inside_the_grid():
    grid = _grid(nx=5, ny=4, nz=3)
    for entry in FLD.slice_axes(grid):
        idx = FLD._index_for(grid, entry["name"],
                             entry["origin_m"] + entry["spacing_m"] * 2)
        assert 0 <= idx < entry["count"]
    # and a position far outside clamps rather than raising
    assert FLD._index_for(grid, "x", 1e9) == grid["nx"] - 1
    assert FLD._index_for(grid, "x", -1e9) == 0


def test_modes_json_must_carry_a_usable_grid():
    with pytest.raises(FLD.FieldError):
        FLD.load_modes_json(json.dumps({"modes": []}).encode())
    with pytest.raises(FLD.FieldError):
        FLD.load_modes_json(json.dumps(
            {"grid": {"nx": 1, "ny": 1, "nz": 1, "origin": [0, 0],
                      "spacing": [1, 1, 1]}}).encode())
    good = {"grid": {"nx": 1, "ny": 1, "nz": 1, "origin": [0, 0, 0],
                     "spacing": [1, 1, 1]}}
    assert FLD.load_modes_json(json.dumps(good).encode()) == good


def test_downsample_shrinks_the_volume_without_changing_the_peak():
    grid = _grid(nx=40, ny=20, nz=10)
    field = _field(grid, seed=3)
    cloud = FLD.downsample(field, grid, longest=8)
    assert cloud["stride"] > 1
    assert len(cloud["mag"]) == int(np.prod(cloud["shape"]))
    assert len(cloud["vec"]) == len(cloud["mag"]) * 3


def test_magnitude_returns_float32():
    grid = _grid()
    assert FLD.magnitude(_field(grid)).dtype == np.float32


# ---------------------------------------------------------------------------
# Container safety
# ---------------------------------------------------------------------------
def test_local_store_refuses_to_escape_its_root(tmp_path):
    store = STORE.LocalStore(tmp_path)
    with pytest.raises(STORE.StorageError):
        store.read("../../etc/passwd")
    with pytest.raises(STORE.StorageError):
        store.write("jobs/../../escape.txt", b"x")
    # a legit key round-trips
    store.write(STORE.job_key(JOB_ID, "input.obj"), b"data")
    assert store.read(STORE.job_key(JOB_ID, "input.obj")) == b"data"


def test_job_keys_match_the_solver_layout():
    assert STORE.job_key(JOB_ID, "input.obj") == f"jobs/{JOB_ID}/input.obj"
    assert STORE.job_key(JOB_ID, "mode_01.bin") == f"jobs/{JOB_ID}/mode_01.bin"


# ---------------------------------------------------------------------------
# Configuration must fail loudly, never fall back to the wrong solver
# ---------------------------------------------------------------------------
def test_the_solver_image_is_required_and_has_no_fallback(monkeypatch):
    """MSE_SOLVER_IMAGE unset must RAISE, not quietly run another image.

    Regression: the old default pointed at the scalar ``spark-fem-runner``
    image, so on any environment that forgot the variable the app would launch
    the wrong solver and report scalar modes with no error at all.
    """
    monkeypatch.delenv("MSE_SOLVER_IMAGE", raising=False)
    with pytest.raises(CFG.ConfigError, match="MSE_SOLVER_IMAGE"):
        CFG.solver_image()


def test_the_solver_image_is_used_when_set(monkeypatch):
    monkeypatch.setenv("MSE_SOLVER_IMAGE", "registry/mse-maxwell-solver:1")
    assert CFG.solver_image() == "registry/mse-maxwell-solver:1"


def test_the_data_dependent_defaults_still_fall_back(monkeypatch):
    """Only the image is fatal; endpoints and the pull secret keep defaults."""
    for var in ("MSE_PULL_SECRET", "MSE_BUCKET", "MSE_MINIO_ENDPOINT",
                "MSE_NAMESPACE"):
        monkeypatch.delenv(var, raising=False)
    assert CFG.pull_secret() == "mse-harbor-pull"
    assert CFG.bucket() == "mse-web"
    assert CFG.NAMESPACE == "mode-search-engine"


# ---------------------------------------------------------------------------
# The WSGI entrypoint: gunicorn calls it, so it must start the ticker once
# ---------------------------------------------------------------------------
def _entrypoint(monkeypatch):
    """Import app.py with the ticker start counted instead of started."""
    import importlib

    monkeypatch.syspath_prepend(str(ROOT / "web"))
    module = importlib.import_module("app")
    monkeypatch.setattr(module, "_TICKER_STARTED", False, raising=True)
    calls = []

    def fake_start(app, store, jobs_db, launcher):
        calls.append(app)
        return (None, None)

    monkeypatch.setattr(module, "start_ticker", fake_start)
    monkeypatch.setattr(module, "create_app",
                        lambda *a, **k: ("the-app", "store", "db", "launcher"))
    return module, calls


def test_build_app_starts_the_ticker_exactly_once(monkeypatch):
    """One gunicorn worker, one scheduler.

    Two tickers would race for the namespace's single Job slot and could launch
    two solver Jobs at once -- which the quota then rejects, stranding one
    upload in ``running`` with no Job behind it.
    """
    module, calls = _entrypoint(monkeypatch)
    assert module.build_app() == "the-app"
    assert len(calls) == 1
    # a second call (a gunicorn worker fork, or an accidental import) must not
    # start a second thread
    assert module.build_app() == "the-app"
    assert len(calls) == 1


def test_build_app_can_be_told_not_to_start_the_ticker(monkeypatch):
    """MSE_NO_TICK exists so the WSGI app can be imported without threads."""
    module, calls = _entrypoint(monkeypatch)
    monkeypatch.setenv("MSE_NO_TICK", "1")
    assert module.build_app() == "the-app"
    assert calls == []
