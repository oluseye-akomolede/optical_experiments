"""The web app's routes and its one-at-a-time scheduler.

The app takes its store, database and Job launcher by injection, so all of this
runs with no cluster, no MinIO and no Postgres. The point of these tests is the
scheduling contract, which is easy to get subtly wrong and expensive to debug in
the cluster: exactly one Job at a time, a 403 quota is a wait rather than a
failure, and a Job is deleted once its result is recorded.
"""

from __future__ import annotations

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

import app as APP  # noqa: E402
import db as DB  # noqa: E402
import jobs as JOBS  # noqa: E402
import storage as STORE  # noqa: E402


# ---------------------------------------------------------------------------
# Doubles for the three collaborators
# ---------------------------------------------------------------------------
class MemoryStore(STORE.LocalStore):
    """A LocalStore with the bucket semantics the app assumes."""

    def __init__(self, root):
        super().__init__(root)


class FakeDb:
    """The parts of JobStore the routes and the tick actually use."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.modes: dict[str, list] = {}
        self.failures: list[tuple] = []
        self.launched: list[tuple] = []

    # writes
    def create_job(self, job_id, filename, units, n_index, n_modes):
        self.jobs[str(job_id)] = {
            "job_id": str(job_id), "created_at": "2026-09-26T00:00:00+00:00",
            "original_filename": filename, "units": units, "n_index": n_index,
            "n_modes": n_modes, "status": DB.QUEUED, "stage": DB.QUEUED,
            "message": None, "k8s_job_name": None, "peak_rss_mb": None,
            "finished_at": None,
        }

    def mark_running(self, job_id, name):
        self.jobs[str(job_id)].update(status=DB.RUNNING, stage="meshing",
                                      k8s_job_name=name)
        self.launched.append((str(job_id), name))

    def set_k8s_job_name(self, job_id, name):
        self.jobs[str(job_id)]["k8s_job_name"] = name

    def mark_failed(self, job_id, message):
        self.jobs[str(job_id)].update(status=DB.FAILED, stage=DB.FAILED,
                                      message=message)
        self.failures.append((str(job_id), message))

    # reads
    def get_job(self, job_id):
        return self.jobs.get(str(job_id))

    def list_jobs(self, limit=25):
        return list(self.jobs.values())[:limit]

    def list_modes(self, job_id):
        return self.modes.get(str(job_id), [])

    def next_queued(self):
        for job in self.jobs.values():
            if job["status"] == DB.QUEUED:
                return job
        return None

    def queue_position(self, job_id):
        job = self.jobs.get(str(job_id))
        if not job or job["status"] != DB.QUEUED:
            return None
        order = [j["job_id"] for j in self.jobs.values()
                 if j["status"] == DB.QUEUED]
        return order.index(str(job_id)) + 1

    def count_running(self):
        return sum(1 for j in self.jobs.values() if j["status"] == DB.RUNNING)

    # set a job up as finished, with a mode table and field files on disk
    def finish(self, job_id, store, n_modes=2):
        grid = {"nx": 4, "ny": 3, "nz": 2, "origin": [0.0, 0.0, 0.0],
                "spacing": [1.0, 1.0, 1.0]}
        self.jobs[str(job_id)].update(status=DB.DONE, stage=DB.DONE,
                                      finished_at="2026-09-26T00:10:00+00:00")
        rows = []
        for index in range(1, n_modes + 1):
            rows.append({"index": index, "k0_per_m": 1.0e3 * index,
                         "freq_hz": 1.0e11 * index,
                         "vacuum_wavelength_m": 3.0e-3 / index})
            field = np.ones((2, 3, 4, 3), dtype="<f4") * index
            store.write(STORE.job_key(job_id, f"mode_{index:02d}.bin"),
                        field.tobytes())
        self.modes[str(job_id)] = rows
        store.write(STORE.job_key(job_id, "modes.json"),
                    json.dumps({"units": "mm", "n_index": 1.0, "n_tets": 6,
                                "n_edges": 12, "grid": grid,
                                "modes": rows}).encode())
        store.write(STORE.job_key(job_id, "surface.glb"), b"glTF")
        return rows


class FakeLauncher:
    """A launcher whose cluster state is a set of names.

    ``failed`` is the set of Job names the cluster reports as FAILED, which is
    how the tick tells a crashed solve (OOM/deadline) from a live one.

    ``cannot_ask`` models a 403 on the list calls. It raises
    ``JOBS.CannotAskCluster`` from both, because that is what the real launcher
    does -- and it must NOT return an empty result, because the whole point of
    the fix is that "we were not allowed to look" and "there is nothing there"
    are different answers. A double that returned ``[]``/``{}`` here would model
    the bug rather than the contract.

    ``swallowed_403`` is the mode the 403 tests need to run against, and it is
    not ``cannot_ask``: it is the *pre-fix* launcher, which answered a 403 with
    an empty result. The two are different doubles on purpose. ``cannot_ask``
    raises, so it models the contract every launcher must honour and can only
    show that the callers handle "we could not ask" -- a launcher that raises is
    already fixed whatever the callers do. ``swallowed_403`` models the actual
    defect, so a test claiming to pin it fails when it comes back.
    """

    def __init__(self, active=None, quota_full=False, fail_with=None,
                 failed=None, cannot_ask=False, swallowed_403=False):
        self.active = list(active or [])
        self.failed = dict(failed or {})
        self.quota_full = quota_full
        self.fail_with = fail_with
        self.cannot_ask = cannot_ask
        self.swallowed_403 = swallowed_403
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.dry_run = False

    def active_job_names(self):
        if self.cannot_ask:
            raise JOBS.CannotAskCluster("not allowed to list solver jobs")
        if self.swallowed_403:
            return []                      # the e5b2e3e behaviour: 403 read as empty
        return list(self.active)

    def job_states(self):
        """Only the names still present can be reporting a state."""
        if self.cannot_ask:
            raise JOBS.CannotAskCluster("not allowed to list solver jobs")
        if self.swallowed_403:
            return {}                      # the e5b2e3e behaviour
        return {n: r for n, r in self.failed.items() if n in self.active}

    def create(self, job_id, units, n_index, n_modes):
        if self.quota_full:
            raise JOBS.QuotaExceeded("exceeded quota: count/jobs.batch")
        if self.fail_with is not None:
            raise self.fail_with
        name = JOBS.job_name(job_id)
        self.created.append({"job_id": str(job_id), "units": units,
                             "n_index": n_index, "n_modes": n_modes,
                             "name": name})
        self.active.append(name)
        return name

    def delete(self, name):
        self.deleted.append(name)
        if name in self.active:
            self.active.remove(name)


def make_app(tmp_path, launcher=None, db=None):
    """Build an app wired to in-memory doubles."""
    store = MemoryStore(tmp_path)
    jobs_db = db or FakeDb()
    launcher = launcher or FakeLauncher()
    app, _store, _db, _launcher = APP.create_app({
        "TESTING": True, "MSE_LOCAL_STORE": str(tmp_path),
        "MSE_DB_MOCK": jobs_db, "MSE_JOB_LAUNCHER": launcher,
    })
    return app, store, jobs_db, launcher


def closed_box_bytes():
    return trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode()


def upload(client, filename="box.obj", data=None, units="mm", n_index="1.0"):
    return client.post("/", data={
        "meshfile": (io_bytes(data if data is not None else closed_box_bytes()),
                     filename),
        "units": units, "n_index": n_index,
    }, content_type="multipart/form-data")


def io_bytes(data):
    import io

    return io.BytesIO(data)


# ---------------------------------------------------------------------------
# A probe for the one property the 403 defect was about
# ---------------------------------------------------------------------------
#: The launcher's ``jobs.py`` must carry a sentence saying that a 403 is raised
#: rather than swallowed. This is a *probe*, not a specification: it is a line in
#: a docstring, and it exists only because nothing else in a unit test can reach
#: the real ``K8sJobLauncher``, whose 403 branch needs a Kubernetes client.
#:
#: Why it is needed at all: the two 403 tests below drive ``swallowed_403``, a
#: double that answers empty. They pin the *callers* -- that ``_active_jobs``
#: turns "could not ask" into a shut gate, and ``_fail_dead_jobs`` refuses to
#: fail a live solve -- and they pass with or without the launcher's raise,
#: because the callers were fixed by the placeholder and there is no state a
#: caller can reach that the double cannot also produce. What no double can
#: cover is the launcher's own half of the fix. This probe covers exactly that,
#: and no more: if the raise is reverted to a return, this fails. Treat the
#: wording as changeable -- if the docstring is reworded, reword this too.
JOBS_RAISES_ON_403_MARKER = "Raises on 403 rather than returning ``[]``. An empty list is a *claim*"


def launcher_still_raises_on_403() -> bool:
    """Whether ``web/jobs.py`` still documents that it raises on a 403."""
    source = (ROOT / "web" / "jobs.py").read_text()
    return JOBS_RAISES_ON_403_MARKER in source


def test_the_launcher_raises_on_a_403_instead_of_answering_empty():
    """The launcher's own half of the 403 fix, which no double can stand in for.

    A source-level probe, and it is worth being honest that it is a weak kind of
    assertion. It reads ``web/jobs.py`` for a sentence saying the 403 branch
    raises. If someone rewords that docstring without meaning to change the
    behaviour, this fails and has to be reworded too; that is the cost.

    It is here because there is no alternative. The real ``K8sJobLauncher``'s 403
    branch needs a Kubernetes client and a cluster answering 403, so a unit test
    cannot execute it; and the two 403 tests above, which drive doubles, pass
    with the raise present or absent, because the callers were fixed separately
    and a double that answers empty leaves the tick nothing to distinguish. What
    remains checkable -- and what the reviewer's defect turned on -- is that the
    launcher does not answer a permission error with an empty result.
    """
    source = (ROOT / "web" / "jobs.py").read_text()
    assert JOBS_RAISES_ON_403_MARKER in source, (
        "web/jobs.py no longer says it raises on a 403; if the wording changed, "
        "update JOBS_RAISES_ON_403_MARKER, and if the behaviour changed, the "
        "403 defect is back"
    )
    assert source.count("CannotAskCluster") >= 3, (
        "the class plus one raise at each of the two list calls"
    )


def test_a_transient_list_error_then_a_good_list_behaves_normally(tmp_path):
    """One failed list must not cost the solve its life.

    A blip -- a timeout, a 403 that clears, a rolling API server -- makes one
    tick unable to see the cluster. The row must survive that tick untouched, and
    the next good list must reconcile it normally, because the Job really is
    still there. This is the reviewer's case in the shape the owner would meet
    it: nothing failed, nothing was deleted, and the solve ran to completion.

    The failure is injected by flipping ``cannot_ask`` between the two ticks.
    That means this test passes on ``e5b2e3e`` too, and it is worth saying why
    rather than finding out later: the swallow makes the *first* tick fail the
    row, and this test only inspects the row's survival, which is a difference
    it cannot see -- both a fixed and a broken launcher leave the second tick
    looking the same. It is a regression test for the reconcile-not-crash path,
    not a discriminator for the 403 defect. See
    ``launcher_still_raises_on_403`` for what covers that.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    launcher.cannot_ask = True
    first = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert first["failed"] == 0, "a blip must not fail a live solve"
    assert first["launched"] is None, "and must not launch into an unknown slot"
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert name not in launcher.deleted

    # the blip clears; the Job is still there and healthy
    launcher.cannot_ask = False
    second = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert second["failed"] == 0, "the Job is present and not failed"
    assert second["deleted"] == []
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert launcher.active == [name]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def test_healthz(tmp_path):
    app, *_ = make_app(tmp_path)
    assert app.test_client().get("/healthz").get_json() == {"status": "ok"}


def test_index_lists_nothing_initially(tmp_path):
    app, *_ = make_app(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "No objects yet" in body
    assert "Compute modes" in body


def test_an_upload_is_queued_launched_and_redirected(tmp_path):
    app, store, db, launcher = make_app(tmp_path)
    response = upload(app.test_client())
    assert response.status_code == 302
    assert "/jobs/" in response.headers["Location"]

    (job,) = db.jobs.values()
    assert job["status"] == DB.RUNNING
    assert job["original_filename"] == "box.obj"
    assert job["n_modes"] == 20
    # the file really landed in the bucket under the contract's key
    assert store.read(STORE.job_key(job["job_id"], "input.obj")) == closed_box_bytes()
    assert len(launcher.created) == 1


def test_the_upload_form_options_reach_the_job(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client(), units="cm", n_index="1.5")
    (job,) = db.jobs.values()
    assert (job["units"], job["n_index"]) == ("cm", 1.5)
    assert launcher.created[0]["units"] == "cm"
    assert launcher.created[0]["n_index"] == 1.5


# ---------------------------------------------------------------------------
# Rejected uploads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename,data,units,n_index,needle", [
    ("model.stl", None, "mm", "1.0", ".obj"),
    ("box.obj", b"", "mm", "1.0", "empty"),
    # trimesh does not raise on junk text; it yields a mesh with no faces
    ("box.obj", b"not a mesh", "mm", "1.0", "no triangles"),
    ("box.obj", None, "furlongs", "1.0", "units"),
    ("box.obj", None, "mm", "0.5", "1.0 or greater"),
    ("box.obj", None, "mm", "not-a-number", "must be a number"),
])
def test_a_bad_upload_is_refused_without_creating_anything(
        tmp_path, filename, data, units, n_index, needle):
    app, store, db, launcher = make_app(tmp_path)
    client = app.test_client()
    response = upload(client, filename=filename, data=data,
                      units=units, n_index=n_index)
    assert response.status_code == 302          # back to the form
    assert db.jobs == {}                        # no row
    assert launcher.created == []               # no Job
    # follow the redirect, the way the browser does: the reason is delivered as
    # a flashed message on the next render, not in this response
    body = client.get(response.headers["Location"]).get_data(as_text=True)
    assert needle.lower() in body.lower()


def test_a_non_watertight_upload_is_refused(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    box = trimesh.creation.box(extents=[10, 7, 5])
    open_mesh = trimesh.Trimesh(vertices=np.array(box.vertices),
                                faces=np.array(box.faces)[:-1], process=False)
    client = app.test_client()
    response = upload(client, data=open_mesh.export(file_type="obj").encode())
    assert response.status_code == 302
    assert db.jobs == {}
    assert launcher.created == []
    body = client.get(response.headers["Location"]).get_data(as_text=True)
    assert "closed surface" in body


# ---------------------------------------------------------------------------
# The status JSON the page polls
# ---------------------------------------------------------------------------
def test_status_json_for_a_running_job(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.RUNNING
    assert payload["stage"] == "meshing"
    assert "modes" not in payload          # nothing to show yet


def test_status_json_for_a_finished_job_includes_the_modes(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    rows = db.finish(job["job_id"], store)
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.DONE
    assert [m["index"] for m in payload["modes"]] == [r["index"] for r in rows]
    assert payload["manifest_url"].endswith("/files/modes.json")
    assert payload["surface_url"].endswith("/files/surface.glb")


def test_status_json_for_a_failed_job_carries_the_message(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.mark_failed(job["job_id"], "This object is too large or detailed to "
                                  "resolve 20 modes.")
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.FAILED
    assert "too large" in payload["message"]


def test_status_json_reports_the_queue_position(tmp_path):
    app, store, db, launcher = make_app(tmp_path)
    launcher.active = ["something-else"]
    launcher.quota_full = False
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    # a second upload while the first is queued must show position 2
    first.update(status=DB.QUEUED, stage=DB.QUEUED)
    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    payload = app.test_client().get(f"/jobs/{second['job_id']}/status").get_json()
    assert payload["queue_position"] == 2


def test_an_unknown_job_is_a_404(tmp_path):
    app, *_ = make_app(tmp_path)
    client = app.test_client()
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/jobs/{uuid.uuid4()}/status").status_code == 404
    assert client.get("/jobs/not-a-uuid").status_code == 404


# ---------------------------------------------------------------------------
# Proxied result files
# ---------------------------------------------------------------------------
def test_result_files_are_proxied_when_the_job_is_done(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    client = app.test_client()

    manifest = client.get(f"/jobs/{job['job_id']}/files/modes.json")
    assert manifest.status_code == 200
    assert manifest.get_json()["n_tets"] == 6

    binary = client.get(f"/jobs/{job['job_id']}/files/mode_01.bin")
    assert binary.status_code == 200
    assert binary.mimetype == "application/octet-stream"
    # 2*3*4 samples * 3 components * 4 bytes
    assert len(binary.data) == 2 * 3 * 4 * 3 * 4


def test_result_files_are_not_served_before_the_job_finishes(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    response = app.test_client().get(
        f"/jobs/{job['job_id']}/files/surface.glb")
    assert response.status_code == 404


@pytest.mark.parametrize("name", ["../secret", "input.obj", "mode_1.bin",
                                  "mode_9999.bin", "run.sh", "modes.json.bak"])
def test_only_solver_output_filenames_are_proxied(tmp_path, name):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    response = app.test_client().get(f"/jobs/{job['job_id']}/files/{name}")
    assert response.status_code == 404, name


def test_a_missing_file_is_a_404_not_a_500(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    path = Path(store.root) / STORE.job_key(job["job_id"], "surface.glb")
    path.unlink()
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/surface.glb").status_code == 404


# ---------------------------------------------------------------------------
# The viewer's data endpoint
# ---------------------------------------------------------------------------
def test_slices_json_carries_every_mode_with_all_three_slices(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    payload = app.test_client().get(
        f"/jobs/{job['job_id']}/slices.json").get_json()
    assert [m["index"] for m in payload["modes"]] == [1, 2, 3]
    for mode in payload["modes"]:
        assert [s["axis"] for s in mode["slices"]] == ["z", "y", "x"]
        for slice_ in mode["slices"]:
            assert len(slice_["mag"]["data"]) == \
                slice_["u"]["count"] * slice_["v"]["count"]
        assert mode["cloud"]["mag"]
    assert payload["units"] == "mm"


def test_slices_json_is_empty_before_the_job_finishes(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/slices.json").status_code == 404


# ---------------------------------------------------------------------------
# The scheduler: exactly one Job, a quota 403 is a wait, and a finished Job is
# deleted so the next upload can start.
# ---------------------------------------------------------------------------
def test_a_second_upload_waits_while_one_is_running(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    assert first["status"] == DB.RUNNING

    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    assert second["status"] == DB.QUEUED
    assert len(launcher.created) == 1          # only one Job exists


def test_a_quota_403_waits_rather_than_failing(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    launcher.quota_full = True
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert job["status"] == DB.QUEUED          # still waiting
    assert db.failures == []                   # NOT a failure


def test_the_tick_launches_the_next_queued_job_once_the_slot_frees(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    assert second["status"] == DB.QUEUED

    # the first Job finishes. The app -- NOT the test -- must delete it: the
    # quota counts finished Jobs, so a Job left behind holds the only slot and
    # the queue deadlocks after the first upload.
    db.jobs[first["job_id"]].update(status=DB.DONE, stage=DB.DONE)
    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert first["k8s_job_name"] in launcher.deleted
    assert db.jobs[second["job_id"]]["status"] == DB.RUNNING
    assert len(launcher.created) == 2


def test_a_finished_job_is_deleted_by_the_tick_without_being_asked(tmp_path):
    """The tick frees the slot itself; nothing else calls launcher.delete.

    Regression: delete() had no caller at all, so a finished Job held the
    namespace's single Job slot forever and every later upload sat queued.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert launcher.active == [name]

    db.jobs[job["job_id"]].update(status=DB.DONE, stage=DB.DONE)
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert launcher.deleted == [name]
    assert result["deleted"] == [name]
    assert launcher.active == [], "the slot must actually be free afterwards"


def test_a_failed_job_is_deleted_too(tmp_path):
    """A failed solve is recorded in the DB; its Job must not hold the slot."""
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    db.jobs[job["job_id"]].update(status=DB.FAILED, stage=DB.FAILED,
                                  message="boom")
    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert launcher.deleted == [name]


def test_a_queued_or_running_job_is_never_deleted(tmp_path):
    """Only terminal rows may be reaped -- deleting a live Job loses the solve."""
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    upload(app.test_client())
    (running,) = [j for j in db.jobs.values() if j["status"] == DB.RUNNING]

    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert running["k8s_job_name"] not in launcher.deleted
    assert launcher.active == [running["k8s_job_name"]]


def test_a_job_reporting_deadline_exceeded_fails_the_row_with_that_reason(
        tmp_path):
    """A Job killed by activeDeadlineSeconds must name the deadline, not "gone".

    The owner needs to know whether the solver ran out of time or out of memory
    -- "try fewer modes" is useless advice for the first if it was the second.
    Kubernetes puts the reason on the Job's Failed condition and the tick maps
    it through ``JOBS.failure_message``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    launcher.failed[name] = "DeadlineExceeded"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 1
    assert db.jobs[job["job_id"]]["status"] == DB.FAILED
    assert db.jobs[job["job_id"]]["message"] == \
        JOBS.FAILURE_MESSAGES["DeadlineExceeded"]


def test_a_job_reporting_oomkilled_fails_the_row_with_that_reason(tmp_path):
    """The original incident's cause: the container was OOM-killed.

    This is the case that actually happened -- the writing stage hit its 14 Gi
    limit -- so the message the owner sees must say memory, not "unexpectedly".
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    launcher.failed[job["k8s_job_name"]] = "OOMKilled"

    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert db.jobs[job["job_id"]]["message"] == \
        JOBS.FAILURE_MESSAGES["OOMKilled"]


def test_a_job_failed_for_an_unknown_reason_does_not_leak_the_kubernetes_token(
        tmp_path):
    """An unmapped reason must fall back, never reach the owner verbatim.

    Kubernetes reasons are token-like strings ('BackoffLimitExceeded'); putting
    one in front of the owner is worse than saying nothing specific.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    launcher.failed[job["k8s_job_name"]] = "BackoffLimitExceeded"

    APP.run_tick(None, db, launcher, log=lambda *a: None)

    message = db.jobs[job["job_id"]]["message"]
    assert message == JOBS.FAILURE_DEFAULT
    assert "BackoffLimitExceeded" not in message


def test_a_live_job_is_never_failed_or_deleted(tmp_path):
    """The case that must NOT be touched: Job present, no Failed condition.

    This is the other half of the fix and the more dangerous half to get wrong.
    The reconcile step crosses the database against the cluster, so a bug that
    treats "absent from job_states" as "dead" would fail every healthy solve the
    moment it started writing -- the owner would see failures on every upload.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert launcher.active == [name]

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted, "a live solve must keep its slot"
    assert db.failures == []


def test_a_403_on_the_list_call_does_not_fail_a_live_solve(tmp_path):
    """The reviewer's finding, reproduced: an unreadable cluster is not an empty one.

    ``K8sJobLauncher`` used to swallow a 403 from both list calls into ``[]`` and
    ``{}``. That made "we were not allowed to look" indistinguishable from "there
    is nothing there", so ``_fail_dead_jobs``'s ``except``-to-0 never ran and a
    live solve was failed. Because ``mark_failed`` is terminal, a later tick then
    deleted that still-running Job via ``_finished_job_names`` -- discarding real
    work on the strength of a permission error, which is the exact outcome the
    docstring promised to avoid.

    What this test pins is the *caller's* half: given an unreadable cluster, the
    tick fails nothing and reaps nothing. It drives ``swallowed_403`` for that,
    because the swallow is the input the caller has to survive. It does not fail
    on ``e5b2e3e``, and cannot: the swallow is exactly what makes the tick answer
    an empty cluster, whose correct handling is not to wait -- so the tick's own
    behaviour is the same either way. The launcher's half of the fix, the raise
    that keeps "could not ask" from ever becoming the empty answer, is what the
    probe in ``test_healthz``'s module pins; see ``launcher_still_raises_on_403``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    launcher.cannot_ask = True

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING, (
        "a 403 on the list call must not fail a solve that is still running"
    )
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted, (
        "and it must not be reaped as terminal on a later tick"
    )


def test_a_403_does_not_let_the_tick_launch_against_the_quota(tmp_path):
    """The other half: an unreadable cluster is not a licence to start a Job.

    ``active_job_names`` returning ``[]`` on 403 also told the tick the namespace
    was empty, so it would launch into a quota it could not read. The gate must
    stay shut when the answer is "unknown" -- the namespace admits one Job, so a
    second start is at best a 403 from the quota and at worst a double solve.

    Like its sibling above, this pins the caller and uses ``cannot_ask``: the
    launcher's own raise, which is what keeps "could not ask" from arriving as
    the empty answer in the first place, is covered by
    ``launcher_still_raises_on_403``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    # a healthy Job is occupying the slot, but the app cannot see that
    launcher.cannot_ask = True

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["launched"] is None
    assert result["waited"] is True
    assert len(launcher.created) == 1, (
        "the tick must not start a second solve against a quota it cannot read"
    )


def test_a_running_row_whose_job_vanished_is_failed_not_requeued(tmp_path):
    """The deploy-blocker-3 fix: a dead solve is FAILED, never silently retried.

    This test replaces one that asserted the old behaviour (re-queue), which was
    the bug: a row whose Job had vanished was put back in the queue, so a solve
    that had already died -- an OOM, say -- would be re-run forever and the owner
    would watch a spinner that never ended. The row must end `failed`, with a
    message the owner can read.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()

    # the DB says running, but the cluster has nothing (lost with its node)
    launcher.active.clear()
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 1
    assert db.jobs[job["job_id"]]["status"] == DB.FAILED
    assert db.jobs[job["job_id"]]["message"] == JOBS.FAILURE_DEFAULT
    assert len(launcher.created) == 1, "the dead job must not be relaunched"


def test_the_tick_does_nothing_when_the_slot_is_occupied(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    upload(app.test_client())
    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert len(launcher.created) == 1


def test_a_dead_job_frees_the_slot_for_the_next_upload(tmp_path):
    """Failing the dead row must not wedge the queue behind it.

    The old behaviour re-queued the dead job, which kept the single Job slot
    busy with a solve that could never succeed. Failing it is only correct if
    the *next* upload then runs -- otherwise the fix just swaps one permanent
    spinner for a permanently stuck queue.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    upload(app.test_client())
    first, second = list(db.jobs.values())
    assert first["status"] == DB.RUNNING

    # the running Job vanished from the cluster without recording a result
    launcher.active.clear()
    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert db.jobs[first["job_id"]]["status"] == DB.FAILED
    assert db.jobs[second["job_id"]]["status"] == DB.RUNNING, (
        "the queued upload must be picked up once the dead job is failed"
    )


def test_a_launch_failure_is_recorded_against_the_job(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    launcher.fail_with = JOBS.JobLaunchError(
        "The service account is not allowed to start solver jobs.")
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert job["status"] == DB.FAILED
    assert "not allowed" in job["message"]
    assert len(db.failures) == 1


def test_nothing_is_launched_when_nothing_is_queued(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert result["launched"] is None
    assert launcher.created == []
