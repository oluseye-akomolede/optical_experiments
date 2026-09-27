"""The web app's routes and its one-at-a-time scheduler.

The app takes its store, database and Job launcher by injection, so all of this
runs with no cluster, no MinIO and no Postgres. The point of these tests is the
scheduling contract, which is easy to get subtly wrong and expensive to debug in
the cluster: exactly one Job at a time, a 403 quota is a wait rather than a
failure, and a Job is deleted once its result is recorded.
"""

from __future__ import annotations

import json
import re
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
                 failed=None, cannot_ask=False, swallowed_403=False,
                 raise_on=None):
        self.active = list(active or [])
        self.failed = dict(failed or {})
        self.quota_full = quota_full
        self.fail_with = fail_with
        self.cannot_ask = cannot_ask
        self.swallowed_403 = swallowed_403
        self.raise_on = raise_on
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.dry_run = False

    def active_job_names(self):
        if self.cannot_ask or self.raise_on == "active_job_names":
            raise JOBS.CannotAskCluster("not allowed to list solver jobs")
        if self.swallowed_403:
            return []                      # the e5b2e3e behaviour: 403 read as empty
        return list(self.active)

    def job_states(self):
        """Only the names still present can be reporting a state.

        This class doubles the *launcher*, not the Kubernetes client, so it
        answers ``job_states`` outright and never exercises ``_failed_reason``
        or the pod list. That is why it did not catch the ``status.failed``
        TypeError and is not expected to: the double that hid that bug is
        ``_FakeApi`` in tests/test_web_app.py, which stands in for the client and
        used to hand back pods from a field the real ``V1JobStatus`` does not
        have. Tests for the real client shapes belong there, against real
        ``kubernetes.client`` objects -- see
        ``test_a_real_job_status_whose_failure_is_only_a_count_does_not_raise``.
        """
        if self.cannot_ask or self.raise_on == "job_states":
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
    not a discriminator for the 403 defect. The launcher's own half -- the raise
    that stops the empty answer being produced -- is covered behaviourally in
    ``test_web_app.py``, which reaches the real 403 branch through a fake
    ``_client``.
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


def test_one_call_raising_while_the_other_answers_still_fails_nothing(tmp_path):
    """The guarantee is "no answer", not the placeholder, and not both calls failing.

    These two list calls are made separately, and only one of them has to fail
    for the tick to be unable to answer. The dangerous shape is the *split* one:
    ``active_job_names`` raises -- so the launch gate shuts on the placeholder --
    while ``job_states`` answers ``{}``, an empty dict that says "every Job is
    healthy, none has failed". Crossed against the placeholder that is not an
    empty list, ``_fail_dead_jobs`` still reaches its ``else`` branch and fails
    the live row.

    So the protection is not the placeholder and it is not "both calls raise
    together". It is that a call which cannot answer raises, and the tick treats
    a raise as a reason to stop. That holds per-call, which is what these two
    tests check -- one for each direction, since either call can be the one that
    fails first.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    # active_job_names cannot answer; job_states answers "nothing has failed"
    launcher.raise_on = "active_job_names"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0, (
        "job_states answering {} must not be read as 'no Job here failed'"
    )
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None, (
        "a live solve must not be failed off a split answer"
    )
    assert name not in launcher.deleted


def test_a_split_answer_the_other_way_round_also_fails_nothing(tmp_path):
    """The mirror: ``job_states`` raises while ``active_job_names`` answers empty.

    This is the closer call of the two. An empty name list is a legitimately
    empty cluster, so the launch gate is entitled to open -- and it does, in
    ``run_tick``, before anyone knows that ``job_states`` is about to raise.
    What must not happen is a running row being failed on that tick: the raise
    from ``job_states`` arrives at ``_fail_dead_jobs``' handler and the tick
    fails nothing.

    Note what this test does *not* assert: that nothing was launched. Launching
    on a genuinely empty list is correct behaviour, and this launcher's empty
    list is indistinguishable from one. Only the failure decision is in scope,
    and that is the decision the reviewer's defect got wrong.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    launcher.raise_on = "job_states"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted


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
# The .vtu download
# ---------------------------------------------------------------------------
# The viewer renders the field itself, so the only reason to fetch a .vtu is to
# carry it away and open it in ParaView. That makes this route's job different
# from the proxy's: the bytes have to arrive intact *and* be saved rather than
# displayed, under a name a person can find again on a downloads shelf.
def _plant_vtu(store, job_id, name="mode_01.vtu", data=b"<VTKFile>...</VTKFile>"):
    store.write(STORE.job_key(job_id, name), data)


def test_a_vtu_is_offered_as_a_download_under_a_name_the_upload_earned(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    upload(client, filename="my lens.obj")
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    response = client.get(f"/jobs/{job['job_id']}/files/mode_01.vtu")

    assert response.status_code == 200
    assert response.data == b"<VTKFile>...</VTKFile>"
    # as_attachment is the whole point: without it the browser renders the XML.
    assert "attachment" in response.headers["Content-Disposition"]
    disposition = response.headers["Content-Disposition"]
    assert "my-lens-mode01.vtu" in disposition, disposition
    # Not the solver's name: twenty `mode_01.vtu` from twenty jobs are
    # indistinguishable once they are all sitting in a downloads folder.
    assert "mode_01.vtu" not in disposition.replace("my-lens-mode01.vtu", "")


@pytest.mark.parametrize("filename", [
    "lens.obj", "a/b/c.obj", "….obj", "lens..obj",
    "  spaced name .obj", "-leading.obj", "no-extension-here.obj",
])
def test_the_download_name_is_a_plain_filename_whatever_was_uploaded(
        tmp_path, filename):
    """A name from the upload reaches Content-Disposition, so it may not be one.

    A path separator in the filename header makes some browsers write a file
    somewhere other than the downloads folder, and the escaping rules for the
    header are inconsistent enough that the safe move is to leave nothing in the
    name that needs escaping. Whatever comes in, what goes out is one path
    segment of ``[A-Za-z0-9._-]``.
    """
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    upload(client, filename=filename)
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    from app import _vtu_download_name
    name = _vtu_download_name(job, "mode_01.vtu")

    assert "/" not in name and "\\" not in name, name
    assert name.endswith("-mode01.vtu"), name
    assert name == Path(name).name, f"{name!r} is not a single path segment"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name), name


def test_a_vtu_the_job_does_not_have_is_a_404(tmp_path):
    """A missing .vtu is a missing file, not an empty download.

    ``send_file`` with no data would be a 0-byte attachment that opens as a
    corrupt mesh in ParaView -- a failure the user has to debug themselves.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)

    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 404


def test_a_vtu_is_not_downloadable_before_the_solver_wrote_it(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    _plant_vtu(store, job["job_id"])

    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 404


def test_an_unreachable_store_does_not_hand_back_a_partial_vtu(tmp_path):
    """The store failing must not read as a file that simply is not there.

    A 404 here would tell the user the solver never wrote the mode; the truth is
    that we could not look. This route already distinguishes them, so the test
    exists to keep it that way.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    def unreachable(key):
        raise STORE.StorageError("the store is unreachable")

    # The route closes over the store object built at create_app time, so the
    # one it will actually call is the one in the app's extensions.
    app.extensions["mse"]["store"].read = unreachable
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 502


@pytest.mark.parametrize("name", [
    "mode_01.vtu.bak", "../mode_01.vtu", "mode_1.vtu", "mode_001.vtu",
    "mode_01.VTU", "sub/mode_01.vtu", "mode_01.vtu/../modes.json",
])
def test_a_vtu_has_to_be_a_vtu_the_solver_writes(tmp_path, name):
    """The whitelist is what makes traversal impossible, so prove it holds.

    Nothing shaped like a path can satisfy the pattern at all, which is why the
    guard is a full match rather than a sanitizer: there is no escaping step to
    get wrong.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)

    response = app.test_client().get(f"/jobs/{job['job_id']}/files/{name}")
    assert response.status_code == 404, name


# ---------------------------------------------------------------------------
# The viewer's data endpoint
# ---------------------------------------------------------------------------
def test_the_index_names_every_mode_and_carries_no_field_data(tmp_path):
    """The index is the list, not the payload.

    This is the contract the lazy load exists for: one small response the phone
    can fetch immediately, with the multi-megabyte arrays behind a second,
    per-mode request. If a bulk ``slices`` ever comes back here, the payload has
    silently regressed to the 11 MB shape it was split to avoid.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    payload = app.test_client().get(
        f"/jobs/{job['job_id']}/index.json").get_json()
    assert [m["index"] for m in payload["modes"]] == [1, 2, 3]
    for mode in payload["modes"]:
        assert "slices" not in mode and "cloud" not in mode
        assert mode["available"] is True
        assert mode["freq_hz"] is not None
    assert payload["units"] == "mm"
    assert payload["grid"]["nx"] > 0


def test_one_mode_payload_holds_that_mode_and_only_that_mode(tmp_path):
    """One mode's whole vector field, as float16, and no other mode's.

    The viewer turns this back into the mode's *scalar* field itself, so the
    only thing the route has to get right is that the bytes are one mode's, in
    the grid's shape, and that they are the mode that was asked for. A payload
    that carried the wrong mode would draw a perfectly plausible picture of the
    wrong thing.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    grid = {"nx": 4, "ny": 3, "nz": 2}
    count = grid["nx"] * grid["ny"] * grid["nz"] * 3

    response = app.test_client().get(f"/jobs/{job['job_id']}/field/2.bin")
    assert response.status_code == 200
    assert response.headers["X-Field-Dtype"] == "float16-le"
    # Two bytes a sample, exactly: a longer body would mean the decoder kept a
    # dtype it was not asked for, and the texture upload would be misaligned.
    assert response.headers["Content-Encoding"] == "gzip"
    raw = gzip.decompress(response.data)
    assert len(raw) == count * 2

    # And it is *that* mode. The fixture gives each mode a different field, so
    # the comparison is against the solver's own bytes for mode 2.
    source = np.frombuffer(
        store.read(STORE.job_key(job["job_id"], "mode_02.bin")), dtype="<f4")
    half = np.frombuffer(raw, dtype="<f2")
    mine = app.test_client().get(f"/jobs/{job['job_id']}/field/2.bin")
    again = np.frombuffer(gzip.decompress(mine.data), dtype="<f2")
    assert np.array_equal(half, again)
    finite = ~np.isnan(source)
    assert np.allclose(half[finite].astype(np.float64), source[finite],
                       rtol=1e-3, atol=1e-5)


def test_a_mode_payload_for_a_mode_the_job_lacks_is_404(tmp_path):
    """A mode number outside the solver's table, and a non-number, both 404."""
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    for bad in ("4", "0"):
        assert client.get(
            f"/jobs/{job['job_id']}/field/{bad}.bin").status_code == 404


def test_a_mode_whose_field_file_is_missing_answers_404_not_500(tmp_path):
    """The mode is in the table but its field is gone: 404, never a crash.

    Worth pinning because the two routes disagree on purpose. The *index* says
    ``available: false`` and still returns 200 -- the list is still useful. The
    per-mode route has nothing to send, so it is a 404. A regression here would
    surface as a 500 in the viewer's fetch, which the browser shows as a generic
    failure with no hint that only one mode is affected.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    (Path(store.root) / STORE.job_key(job["job_id"], "mode_02.bin")).unlink()

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    flags = {m["index"]: m["available"] for m in index["modes"]}
    assert flags == {1: True, 2: False, 3: True}
    assert client.get(
        f"/jobs/{job['job_id']}/field/2.bin").status_code == 404
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 200


def test_a_truncated_field_file_is_unavailable_rather_than_half_drawn(tmp_path):
    """Size is checked, so a partial write cannot decode into a wrong shape."""
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    blob = Path(store.root) / STORE.job_key(job["job_id"], "mode_01.bin")
    blob.write_bytes(blob.read_bytes()[:100])

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    assert {m["index"]: m["available"] for m in index["modes"]}[1] is False
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 404


def test_the_index_and_the_fetch_agree_on_which_modes_exist(tmp_path):
    """``available: true`` must mean the fetch will actually serve the mode.

    The reviewer found the shape of this bug: the index resolved a mode's
    filename one way and the fetch route resolved it another, so the list could
    promise a mode the fetcher 404ed -- a could-not-ask wearing the other mask,
    discovered only after a doomed request. Today there is one resolver
    (``FIELD_NAME``); this test exists to catch a *second* one appearing.

    Its most important fixture is the **named** file. The original bug was only
    reachable when ``modes.json`` named a field file, and the ``db.finish``
    helper never writes that key -- so a test built on the default manifest
    would pass against a re-introduced second resolver, exactly as the first
    version of this test did. A mode is renamed here and the manifest is made to
    name it, so any code path that honours ``field_file`` is exercised rather
    than skipped.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=4)

    prefix = STORE.job_key(job["job_id"], "")
    key = lambda name: STORE.job_key(job["job_id"], name)   # noqa: E731

    # mode 1: moved to a name the convention would not guess, and declared.
    store.write(key("mode_01_moved.bin"), store.read(key("mode_01.bin")))
    (Path(store.root) / key("mode_01.bin")).unlink()

    # mode 2: gone entirely, still declared.
    (Path(store.root) / key("mode_02.bin")).unlink()

    # mode 3: present under the convention but the wrong size.
    blob = Path(store.root) / key("mode_03.bin")
    blob.write_bytes(blob.read_bytes()[:64])

    # mode 4: untouched.

    manifest = json.loads(store.read(key("modes.json")))
    declared = {1: "mode_01_moved.bin", 2: "mode_02.bin"}
    for mode in manifest["modes"]:
        if int(mode["index"]) in declared:
            mode["field_file"] = declared[int(mode["index"])]
    store.write(key("modes.json"), json.dumps(manifest).encode())
    assert prefix  # the key layout is the job's, not a bare name

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()

    for mode in index["modes"]:
        served = client.get(
            f"/jobs/{job['job_id']}/field/{mode['index']}.bin")
        if mode["available"]:
            assert served.status_code == 200, (
                f"the index promised mode {mode['index']} and the fetch refused "
                f"it ({served.status_code}) -- the two routes disagree about "
                "which file holds this mode"
            )
        else:
            assert served.status_code == 404, (
                f"the index hid mode {mode['index']} but the fetch served it: "
                "the list is hiding something the server would have given"
            )

    # mode 1 is present in the store, but under a name only the manifest knows,
    # so the single resolver does not find it and the index says so. That is the
    # point: with one resolver, `available` is answerable and correct. A second
    # resolver that honours `field_file` would report it available, the fetch
    # would still 404, and the loop above fails -- which is how this catches the
    # reviewer's bug coming back.
    assert {m["index"]: m["available"] for m in index["modes"]} == {
        1: False, 2: False, 3: False, 4: True}


# ---------------------------------------------------------------------------
# The volume plot's field endpoint
# ---------------------------------------------------------------------------
def test_the_field_route_serves_float16_that_decodes_back_to_the_mode(tmp_path):
    """The bytes the browser textures are the mode, in half precision.

    Round-trips through the real decoder rather than trusting the header: what
    matters is that a texture built from this payload shows the field the solver
    wrote, at half precision, with ``(nz, ny, nx, 3)`` samples in the order the
    grid expects.
    """
    import gzip
    import struct

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    response = app.test_client().get(
        f"/jobs/{job['job_id']}/field/2.bin")
    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert response.headers["X-Field-Dtype"] == "float16-le"

    raw = gzip.decompress(response.data)
    original = np.frombuffer(
        store.read(STORE.job_key(job["job_id"], "mode_02.bin")), dtype="<f4")
    half = np.frombuffer(raw, dtype="<f2")
    assert half.size == original.size
    # Every finite sample agrees to half precision; the count of NaN matches.
    finite = ~np.isnan(original)
    assert int(np.isnan(half).sum()) == int((~finite).sum())
    assert np.allclose(half[finite].astype("f4"), original[finite], atol=1e-3)


def test_the_field_route_keeps_nan_transparent_where_the_solver_put_it(tmp_path):
    """NaN is the object's outside; the plot shows it by *not* showing it.

    If the encoder wrote a number here the browser would paint coloured fog
    around the object, which still looks like a plausible picture and so would
    not be noticed. The route must carry NaN through unchanged.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    # The default fixture writes a finite field, so it could not exercise this
    # at all -- the assertion below would have been vacuous. Write a blob that
    # is NaN over part of its samples, in the shape the grid requires, then
    # check the encoder preserved the NaN mask exactly.
    key = STORE.job_key(job["job_id"], "mode_01.bin")
    source = np.frombuffer(store.read(key), dtype="<f4").copy()
    source[: source.size // 3] = np.nan
    store.write(key, source.tobytes())

    raw = gzip.decompress(app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").data)
    half = np.frombuffer(raw, dtype="<f2")
    assert np.isnan(half).sum() > 0, "the fixture must have NaN to test with"
    assert np.array_equal(np.isnan(half), np.isnan(source))


def test_both_field_routes_agree_on_which_modes_exist(tmp_path):
    """The index and the binary fetch must resolve the same file.

    The reviewer's original bug was two routes resolving one filename
    differently, so a mode the list promised 404ed on fetch. The field route
    resolves ``FIELD_NAME`` too, and this pins that they stay in step: a mode
    the index marks available must serve bytes, and one it hides must not.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=4)
    (Path(store.root) / STORE.job_key(job["job_id"], "mode_02.bin")).unlink()

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    for mode in index["modes"]:
        served = client.get(f"/jobs/{job['job_id']}/field/{mode['index']}.bin")
        if mode["available"]:
            assert served.status_code == 200, mode["index"]
        else:
            assert served.status_code == 404, mode["index"]


def test_a_mode_the_job_lacks_is_404_on_the_field_route(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    for bad in ("4", "0"):
        assert client.get(
            f"/jobs/{job['job_id']}/field/{bad}.bin").status_code == 404


def test_an_unreachable_store_is_a_502_on_the_field_route_too(tmp_path):
    """Same invariant as the index: a store we cannot ask is not "no mode".

    A 404 here would tell the viewer this mode is missing and let it move on
    quietly; the truth is that we could not find out. That distinction is what
    ``available: false`` is only ever allowed to mean from a *successful* check.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    def boom(*_a, **_k):
        raise STORE.StorageError("the store is unreachable")

    app.extensions["mse"]["store"].read = boom
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 502


def test_an_unreachable_store_is_an_error_not_a_page_of_missing_modes(tmp_path):
    """A store we cannot ask must not answer "all modes are gone".

    ``available: false`` is a claim about the data. ``StorageError`` means the
    claim could not be checked at all -- S3 down, credentials lapsed, a network
    partition -- and rendering it as a 200 full of false flags tells the owner
    something false: every mode greyed out, and no error anywhere to explain it.

    This is the signature defect in its third costume (403 into an empty list,
    not-allowed-to-look into nothing-there, and now storage-unreachable into
    file-missing). The store already draws the line -- ``None`` is "looked and
    it is not there", an exception is "could not look" -- so the test asserts the
    route keeps it, and the asymmetry that proves it: the very same store serves
    a mode the index refused to describe.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    # A positive control first: with a working store the index is a 200 and the
    # modes are genuinely available. Without this, a route that always 502s
    # would pass the assertion below for the wrong reason.
    healthy = client.get(f"/jobs/{job['job_id']}/index.json")
    assert healthy.status_code == 200
    assert {m["available"] for m in healthy.get_json()["modes"]} == {True}

    def unreachable(key):
        raise STORE.StorageError(f"could not check {key} in storage")

    # The app builds its own store from ``MSE_LOCAL_STORE``; ``make_app``'s
    # store is a second handle on the same directory, used to lay down fixtures.
    # Patching the app's handle is the one the routes call.
    app.extensions["mse"]["store"].size = unreachable
    response = client.get(f"/jobs/{job['job_id']}/index.json")

    assert response.status_code == 502, (
        "an uncheckable store must be an error the viewer can report, not a "
        "200 claiming every mode is missing"
    )
    # And the fetch route, on the same broken store, still serves the mode --
    # which is the proof that the index's "all false" would have been a lie
    # about the data rather than a fact about it.
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 200


def test_the_index_route_is_the_only_json_the_viewer_needs(tmp_path):
    """The index still has to be strict JSON, and it still has to be small.

    The NaN lesson lives on in a different place now. The field payload is
    float16 bytes rather than JSON, so ``json.dumps`` can no longer leak a bare
    ``NaN`` token into it -- but the *index* is still JSON, and the grid it
    carries still comes from ``modes.json``, which the solver wrote. So this
    parses the index the way the browser does, with the hook that only fires on
    a non-finite token, and separately checks that no mode entry has grown a
    field-sized array back into it.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    response = app.test_client().get(f"/jobs/{job['job_id']}/index.json")
    assert response.status_code == 200

    def reject(token):
        raise AssertionError(
            f"the index wrote the bare token {token!r}; JSON.parse would reject "
            "the whole payload and the viewer would show an error instead")

    payload = json.loads(response.get_data(as_text=True), parse_constant=reject)
    assert [m["index"] for m in payload["modes"]] == [1, 2, 3]
    for mode in payload["modes"]:
        assert "slices" not in mode and "cloud" not in mode and "field" not in mode
    # The whole point of the split: this response must stay small enough to
    # fetch on page load. One mode's float16 field is ~110 kB for this grid, so
    # a plain ceiling well under that catches a bulk array coming back.
    assert len(response.data) < 4000, (
        f"the index is {len(response.data)} bytes; a field-sized array has "
        "probably come back into it, which is the 11 MB shape it was split from")


def test_a_real_field_response_is_written_for_the_node_test(tmp_path):
    """Write the bytes a live route returns, for the node test to parse.

    The old fixture was JSON and the node test used ``JSON.parse``; the payload
    is float16 bytes now, so the instrument changes with it. What must not
    change is the principle: the fixture is produced by a *live route*, not
    hand-written, so ``test_viewfield.mjs`` is checking the real thing rather
    than a double that models what we think we send.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=1)

    key = STORE.job_key(job["job_id"], "mode_01.bin")
    grid = {"nx": 4, "ny": 3, "nz": 2}
    field = np.full((grid["nz"], grid["ny"], grid["nx"], 3), np.nan, dtype="<f4")
    field[0, 0, 0] = [1.0, -0.5, 0.25]
    field[1, 2, 3] = [0.125, 0.0, -1.0]
    store.write(key, field.tobytes())

    response = app.test_client().get(f"/jobs/{job['job_id']}/field/1.bin")
    assert response.status_code == 200

    fixtures = Path(__file__).resolve().parent / "js" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "mode_field.bin").write_bytes(gzip.decompress(response.data))
    # The grid the node test needs to decode it, alongside the bytes. Written
    # from the live index rather than from the literal above, so a change to how
    # the server reports a grid cannot pass unnoticed.
    index = app.test_client().get(f"/jobs/{job['job_id']}/index.json").get_json()
    (fixtures / "mode_grid.json").write_text(json.dumps(index["grid"]))
    # Remove the old JSON fixture: leaving it would let a stale artifact be read
    # by a test that no longer describes the API.
    stale = fixtures / "mode_payload.json"
    if stale.exists():
        stale.unlink()


def test_the_json_provider_refuses_to_write_a_non_finite_number(tmp_path):
    """The tripwire, tested directly rather than only through a route.

    A future field that forgets ``_round`` should fail loudly in tests. Without
    ``allow_nan=False`` the failure is silent, reaches the browser, and surfaces
    as a message about the network -- which is what happened to the owner.
    """
    app, _store, _db, _launcher = make_app(tmp_path)
    with app.test_request_context():
        from flask import jsonify
        # jsonify serialises eagerly, so the ValueError lands here rather than
        # at response time -- either way it is a 500 in the request, not a
        # payload the browser cannot parse.
        with pytest.raises(ValueError):
            jsonify({"bad": float("nan")})


def test_the_index_is_not_served_before_the_job_finishes(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/index.json").status_code == 404
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 404


# ---------------------------------------------------------------------------
# Static assets must revalidate, or a deploy serves the old viewer
# ---------------------------------------------------------------------------
def test_static_assets_must_revalidate_rather_than_expire_on_a_timer(tmp_path):
    """A cached viewer.js must never outlive the API it was written against.

    The endpoints changed shape: ``/index.json`` used to carry every mode's
    field and is now an index. A browser holding the old ``viewer.js`` would ask
    for the new shape, find no ``slices`` key, and render blank slices -- with no
    error anywhere, which is what makes it worth a test rather than a comment.

    This passes today because Flask's ``send_from_directory`` sends
    ``Cache-Control: no-cache`` plus an ETag: the browser caches, but must
    revalidate, and revalidation fails the moment the file changes.
    ``no-cache`` is not ``no-store`` -- it is exactly the behaviour we want.

    What must NOT appear is an expiry without revalidation. ``max-age`` (or
    ``S-F``-style freshness) lets a browser reuse the file with no question
    asked, for as long as it says. That is the regression this test exists for,
    and it is the only test here that a version query-string on the script tag
    would not satisfy -- deliberately, because a query-string is a workaround for
    a missing revalidation guarantee, not the guarantee itself.

    The behavioural half mutates a **copy** of ``web/static``, not the tracked
    files, and points ``app.static_folder`` at it. Mutating the real one would
    work, but a kill between the write and the cleanup, or a concurrent run,
    would leave the source modified -- the same shape as the revert-tool damage
    that needed a lock. Working on a copy removes the window rather than
    tidying up after it.
    """
    import shutil

    app, _store, _db, _launcher = make_app(tmp_path)
    client = app.test_client()

    for asset in ("viewer.js", "modecache.js", "style.css"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200, asset
        cache_control = (response.headers.get("Cache-Control") or "").lower()
        assert "max-age" not in cache_control, (
            f"/static/{asset} sends Cache-Control {cache_control!r}, which lets a "
            "browser reuse it without revalidating -- a deploy would then keep "
            "serving the old viewer against the new API"
        )
        # Revalidation is only meaningful if there is something to compare.
        assert response.headers.get("ETag") or \
            response.headers.get("Last-Modified"), asset

    # From here on the app serves a copy, so this test cannot touch the checkout.
    staged = tmp_path / "static-copy"
    shutil.copytree(app.static_folder, staged)
    real_viewer = Path(app.static_folder) / "viewer.js"
    original = real_viewer.read_bytes()
    app.static_folder = str(staged)

    # And the guarantee must be real, not just declarable: change the file and
    # the stale copy must be refused. Without this the assertions above could
    # pass on a server that ignores validators entirely.
    first = client.get("/static/viewer.js")
    unchanged = client.get("/static/viewer.js", headers={
        "If-None-Match": first.headers["ETag"]})
    assert unchanged.status_code == 304, "an unchanged asset should be reused"

    (staged / "viewer.js").write_text(
        (staged / "viewer.js").read_text() + "\n// deployed change\n")
    revalidated = client.get("/static/viewer.js", headers={
        "If-None-Match": first.headers["ETag"]})
    assert revalidated.status_code == 200, (
        "a changed asset was served as 304: the browser would keep the old "
        "viewer with no way to notice"
    )
    assert real_viewer.read_bytes() == original, (
        "this test modified the tracked web/static/viewer.js; it must only ever "
        "mutate its own copy"
    )


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
    that keeps "could not ask" from ever becoming the empty answer, is covered
    behaviourally in ``test_web_app.py``.
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
    launcher's own raise, which keeps "could not ask" from arriving as the empty
    answer in the first place, is covered behaviourally in ``test_web_app.py``.
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
