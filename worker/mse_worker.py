#!/usr/bin/env python3
"""The Jetson GPU worker: claim one job at a time, solve it, hand back results.

This process lives on the Jetson and is the only thing there that talks to the
app. It is deliberately *warm*: the solver's imports and the CUDA context are
built once and kept, because CUDA context initialisation is the startup cost that
makes one-process-per-job useless, and re-importing the solver for every job
throws away the only batching win this machine has.

Design rules, in the order they are load-bearing:

**One job at a time, and the bench lock is why.** The lead's own benchmarks take
``~/mse-bench/.gpu.lock``; so does this worker, while it solves. The lock is
checked BEFORE claiming, so a job is never taken and then left sitting -- a
claimed job the worker cannot run is worse than an unclaimed one, because it is
invisible at the cluster end and the owner is waiting on it.

**No store client.** The Jetson cannot reach MinIO, so the only way in is the
worker API's input route and the only way out is its file route. This file never
constructs an S3Store and never reads MINIO_* anything.

**A failure must not look like a success.** Every path that cannot finish a job
reports ``failed``, which releases the lease and puts the row back in the pool.
The one path that must never run after a failure is ``done``: the server would
then believe a job finished whose results do not exist.

**Nothing here logs the token.** It arrives from a mode-600 file, is held in the
client, and is never formatted into a log line, an exception message, or a repr.

Run under systemd -- see ``mse-worker.service`` beside this file.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import sys
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path

#: the Jetson's own checkout of the solver package, alongside this file
_SOLVER_DIR = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"

#: default work root: created mode 700 under the worker's home. It holds the
#: downloaded input and the solver's output for one job, and nothing secret --
#: which is what makes deleting it safe.
DEFAULT_WORK_ROOT = Path.home() / "mse-worker" / "work"

#: the lock the lead's benchmarks also take. Not a worker-internal lock: it
#: exists so two things that each want the whole GPU (this worker and a bench
#: run) cannot overlap.
DEFAULT_LOCK_PATH = Path.home() / "mse-bench" / ".gpu.lock"


def _redact_token(text: str, token: str) -> str:
    """Replace ``token`` (and its ``Bearer`` form) with a placeholder.

    Applied to every line of a logged traceback. Returns ``text`` unchanged when
    there is no token, which is the normal case in a test.
    """
    if not token:
        return text
    return text.replace(f"Bearer {token}", "Bearer <redacted>").replace(
        token, "<redacted>")

#: where the token and URL come from. Mode 600; the worker refuses to read a
#: file anyone else can read.
DEFAULT_ENV_FILE = Path.home() / ".config" / "mse-worker" / "env"

#: files the worker API accepts, mirrored from web/worker_api.py. Kept as a
#: local allow-list as well so a future solver output that nobody added to the
#: API fails here, loudly, instead of being silently dropped on upload.
_UPLOAD_EXACT = ("modes.json", "surface.glb")


class WorkerError(Exception):
    """Something the operator must fix before the worker can run."""


class APIError(Exception):
    """The app refused a call. Carries the server's own words.

    ``method`` and ``path`` are carried, not just the status, because the
    status alone does not say WHICH call failed and a job makes five of them in
    a row (input, progress, files, done, failed). "HTTP 500" with no path is the
    difference between an upload route that is broken and a ``/done`` route that
    is broken, and that is the one thing a reader of the journal most needs to
    know.
    """

    def __init__(self, status: int, body: str, method: str = "", path: str = ""):
        self.status = status
        self.body = body
        self.method = method
        self.path = path
        where = f"{method} {path}" if method else "request"
        super().__init__(f"HTTP {status} on {where}: {body}")

    def describe(self) -> str:
        """A one-line, journal-ready form. Never includes the token.

        The token travels in an ``Authorization`` header, which this never
        reads, so nothing here can leak it -- and the URL logged is ``self.url``
        with no credentials, because the worker's config keeps the token
        separate from the base URL.
        """
        where = f"{self.method} {self.path}" if self.method else "(unknown call)"
        return f"HTTP {self.status} on {where}: {self.body}"


class LeaseLost(Exception):
    """This worker no longer owns the job it is solving.

    Fatal for the job, not for the process: the lease expired or another worker
    took over, and continuing would mean doing work nobody will collect. The
    worker abandons the job without reporting anything -- reporting ``failed``
    would release a lease it does not hold, and ``done`` would claim a job it
    does not own.
    """


# The words that mean "this GPU would not come up", as opposed to "this GPU
# tried and could not do THIS". Matched case-insensitively on the message, which
# is the only evidence available: cuDSS raises its status through nvmath, whose
# exception carries the status name as text.
def _solver_word_list(name: str, fallback: tuple) -> tuple:
    """The solver's own word list for this distinction, or a local copy.

    Both places need to read the same words: the worker decides whether to
    requeue, and ``mse_maxwell_solver.owner_message`` decides what to TELL the
    owner. Two hand-maintained copies would drift, and the drift would be
    invisible -- a word added to one and not the other produces a job that is
    requeued while its page blames the object, which is the false sentence this
    pair of lists exists to prevent. The import is late because the worker must
    start on a machine where the solver's heavy dependencies are absent.
    """
    try:
        if str(_SOLVER_DIR) not in sys.path:
            sys.path.insert(0, str(_SOLVER_DIR))
        import mse_maxwell_solver as SOLVER  # noqa: PLC0415 - deliberately late
    except Exception:  # noqa: BLE001 - no solver means classify() has no table
        return fallback
    return tuple(getattr(SOLVER, name, fallback))


# The words that mean "this GPU would not come up", as opposed to "this GPU
# tried and could not do THIS". Matched case-insensitively on the message, which
# is the only evidence available: cuDSS raises its status through nvmath, whose
# exception carries the status name as text.
_GPU_INIT_WORDS = _solver_word_list("GPU_INIT_WORDS", (
    "initializ", "initialis",     # "initialization failed", "could not initialise"
    "no device", "no cuda", "no gpu",
    "device not found", "driver",
    "context creation", "context init",
    "nvrtc", "cuda_driver",
))
# The words that mean the OPPOSITE -- a real answer from a working GPU about
# THIS job. Checked FIRST, so a message containing both ('initialization failed
# after ALLOC_FAILED') is read as terminal, which is the safe direction: a job
# wrongly failed once can be re-uploaded, a job wrongly requeued loops forever.
# The allocation family is spelled several ways depending on whether the message
# comes from cuDSS (CUDSS_STATUS_ALLOC_FAILED), nvmath's prose ("allocation
# failed") or the driver ("out of memory"). All of them are the GPU reporting on
# THIS job, and all of them must beat an init word in the same message --
# "initialization succeeded but allocation failed" is a GPU that came up and then
# ran out of room, so requeueing it is the loop this list exists to prevent.
# (Found by a test: the two spellings of "alloc failed" that were here did not
# cover the third, and the init word won.) The allocation words are the solver's
# own list; the rest are the worker's, because a cuDSS STATUS error reaches the
# worker as text and the solver's owner_message never sees it
# (releases are not passed through owner_message at all).
_GPU_STATUS_WORDS = _solver_word_list("ALLOC_FAILURE_WORDS", (
    "alloc_failed", "alloc failed", "allocation failed", "allocation_failed",
    "out of memory", "outofmemory", "insufficient memory",
)) + (
    "singular", "not positive definite", "invalid value",
    "execution failed", "wrong parameter",
)


def _wraps_a_gpu_init_failure(exc: BaseException) -> bool:
    """Does the ``__cause__``/``__context__`` chain hold a GPU INIT failure?

    ``mse_eigen`` wraps a cuDSS failure as ``EigenError(msg) from exc``, so the
    worker never sees the cuDSS exception itself -- it must look through the
    chain. Walking it is also what keeps this narrow: the top-level type is
    ``EigenError`` for BOTH "the GPU would not initialise" (transient) and "the
    factorisation hit ALLOC_FAILED" (terminal), and only the chain tells them
    apart.

    The line is drawn on the MESSAGE, because that is all the chain carries. A
    status error wins over an init word when both appear: a GPU that came up,
    ran, and then said ALLOC_FAILED is giving a verdict about the job, and
    re-running it would produce the same verdict. Getting that backwards is the
    loop this function exists to prevent.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    depth = 0
    while node is not None and depth < 10 and id(node) not in seen:
        seen.add(id(node))
        depth += 1
        name = type(node).__name__
        text = f"{name}: {node}".lower()

        # A status error anywhere in the chain means the GPU ran and answered.
        if any(word in text for word in _GPU_STATUS_WORDS):
            return False
        # A CUDA TYPE is the strongest evidence: cupy raises these only when the
        # runtime or driver itself is the problem.
        if name in ("CUDARuntimeError", "CUDADriverError"):
            return True
        if any(word in text for word in _GPU_INIT_WORDS):
            return True

        node = node.__cause__ or node.__context__
    return False


def classify(exc: BaseException) -> str:
    """Is this failure TERMINAL (this job can never succeed) or TRANSIENT?

    The whole reason the app has two verbs instead of one. A terminal failure is
    a verdict about the JOB -- a mesh the mesher cannot build, a box too detailed
    for the machine, an eigenproblem with no solution, a bad upload -- and
    re-running it produces the same failure forever, so it must go to
    ``/failed`` and end there. A transient failure says nothing about the job: it
    says this WORKER, right now, could not finish -- a shutdown, a GPU that
    would not initialise, a network that dropped, a disk that is full. Those go
    to ``/release``, which hands the job back for another attempt.

    Getting this backwards is worse in both directions, which is why it is a
    function with a table instead of a guess at each call site: calling a
    terminal failure transient loops forever, and calling a transient failure
    terminal throws away a job that would have worked on the next attempt.

    The default is TERMINAL. An unrecognised exception is more likely to be a
    deterministic solver fault than a blip, and the costly mistake is the loop:
    a job wrongly failed once can be re-uploaded, while a job that loops forever
    pins the GPU and no worker can get past it.
    """
    # Imported here, not at module scope: the mesher and the solver pull numpy
    # and scipy, and the tier-1 tests must not pay for that to test the table.
    try:
        from mse_meshing import MeshingError
        terminal_types: tuple = (MeshingError,)
    except ImportError:                       # pragma: no cover - solver absent
        terminal_types = ()
    try:
        from mse_maxwell_solver import TooBigForProcessor
        terminal_types += (TooBigForProcessor,)
    except ImportError:                       # pragma: no cover
        pass
    try:
        from mse_eigen import EigenError
        terminal_types += (EigenError,)
    except ImportError:                       # pragma: no cover
        pass

    name = type(exc).__name__

    # --- transient: this worker could not, another one might ---
    if isinstance(exc, LeaseLost):
        # Not a verdict at all: the job was taken from us. Reporting it failed
        # would be a lie about a job someone else is now running.
        return "lease-lost"
    if isinstance(exc, WorkerError) and "asked to stop" in str(exc):
        # SIGTERM mid-solve: the operator is restarting the service. The job is
        # fine and must be requeued, not failed.
        return "transient"
    if isinstance(exc, ResultDirUnavailable):
        # A LocalStore that could not enumerate the results it just wrote.
        # TRANSIENT, against the default, because it is not a verdict about the
        # job: nothing about this mesh or this solve is implicated, and the job
        # is worth another attempt. It must be caught BEFORE the bare ``OSError``
        # branch below, since ``FileNotFoundError`` is an ``OSError`` and would
        # otherwise land in the same place for the wrong reason.
        return "transient"
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
        # A dropped socket, a full disk, a permission blip on the work dir.
        return "transient"
    if name in ("OutOfMemoryError", "MemoryError"):
        # The box, not the job: the same job may fit when nothing else is
        # running. (A job too big for the MACHINE arrives as
        # TooBigForProcessor, which is terminal -- different event.)
        return "transient"
    if name.startswith("CUDA") or name in ("CudaError", "GpuInitError"):
        # cuDSS/CUDA initialisation. Documented on this box as a state the
        # process recovers from on restart, not a property of the job.
        return "transient"
    if _wraps_a_gpu_init_failure(exc):
        # The SAME event as the branch above, one wrapper deeper. ``mse_eigen``
        # raises ``EigenError(...) from exc`` when cuDSS fails, so a GPU that
        # would not initialise reaches here as an ``EigenError`` and the name
        # check above cannot see it. Without this the job is failed forever for
        # a fault that clears on the next worker.
        #
        # Deliberately narrow: ONLY initialisation-type GPU faults. A cuDSS
        # STATUS error (ALLOC_FAILED, a singular factor) is a verdict about the
        # data in this job and must stay terminal -- see the docstring of
        # ``_wraps_a_gpu_init_failure`` for where that line is drawn.
        return "transient"
    if isinstance(exc, APIError):
        # The APP could not be reached -- the job never ran. A 4xx that means
        # "you no longer hold the lease" is LeaseLost-shaped (handled above via
        # its own type); anything else is the app being unavailable.
        return "transient"

    # --- terminal: a verdict about the job ---
    if terminal_types and isinstance(exc, terminal_types):
        return "terminal"
    if isinstance(exc, ValueError):
        # The solver raises ValueError for malformed inputs (a bad index, a
        # negative size). Deterministic: the same upload fails the same way.
        return "terminal"

    return "terminal"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """The worker's settings. **Never reprs the token.**

    ``__repr__`` is overridden because a dataclass-style repr is how a token
    ends up in a log: somebody prints the config while debugging, and the one
    secret in the process is now in the journal.
    """

    def __init__(self, url: str, token: str, *, work_root: Path,
                 lock_path: Path, poll_seconds: float = 5.0,
                 keep_failed: int = 1, beat_seconds: float = 20.0):
        self.url = url.rstrip("/")
        self.token = token
        self.work_root = Path(work_root)
        self.lock_path = Path(lock_path)
        self.poll_seconds = float(poll_seconds)
        self.keep_failed = int(keep_failed)
        #: How often the heartbeat renews the lease WHILE A JOB IS SOLVING.
        #: This is the worker's own number, not the app's lease length, and the
        #: distinction is the whole point: the app's lease is how long the row
        #: survives a worker that has died, which wants to be generous, while
        #: the beat interval only has to be short enough that a transient
        #: network failure still leaves a beat inside the window. Deriving it
        #: from the lease (``lease / 3``) made it 100 s for the 300 s default --
        #: longer than many real solves, so the FIRST beat arrived after the
        #: work was done, and a job that ran long with no beat between looked
        #: to the app like a worker that had died.
        self.beat_seconds = float(beat_seconds)

    def __repr__(self) -> str:
        return (f"Config(url={self.url!r}, token=<{len(self.token)} chars>, "
                f"work_root={str(self.work_root)!r}, lock_path={str(self.lock_path)!r})")


def parse_env_file(path: Path) -> dict:
    """Read ``KEY=VALUE`` lines from ``path``. Comments and blanks are skipped.

    Deliberately not ``os.environ.update``: the file is the worker's whole
    configuration surface, and inheriting the process environment would let a
    stray ``MSE_URL`` win silently over what the operator wrote down.
    """
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise WorkerError(f"{path}: {line!r} is not a KEY=VALUE line")
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def check_env_file_mode(path: Path) -> None:
    """Refuse to use an env file that other users can read.

    The file holds the worker token. A 0644 token file on a host with more than
    one account is a leak that this process can simply refuse to participate in,
    and the operator gets a clear error instead of a silent exposure.
    """
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise WorkerError(
            f"{path} is mode {mode:04o}; it holds the worker token and must not "
            f"be readable by anyone else. Fix it with: chmod 600 {path}"
        )


def load_config(env_file: Path | None = None, **overrides) -> Config:
    """Build the Config from the env file, with command-line overrides.

    A missing URL or token is a startup refusal rather than an idle loop: a
    worker that starts, silently claims nothing and reports healthy is the
    failure mode where the Jetson sits idle for a day and nobody notices.
    """
    path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if not path.is_file():
        raise WorkerError(f"{path} does not exist; see worker/README.md")
    check_env_file_mode(path)
    values = parse_env_file(path)

    url = overrides.get("url") or values.get("MSE_URL") or ""
    token = values.get("MSE_WORKER_TOKEN") or ""
    if not url:
        raise WorkerError(f"{path}: MSE_URL is not set")
    if not token:
        raise WorkerError(f"{path}: MSE_WORKER_TOKEN is not set")

    def _path(name: str, default: Path) -> Path:
        raw = overrides.get(name) or values.get(name.upper())
        return Path(raw).expanduser() if raw else default

    def _float(name: str, default: float) -> float:
        raw = overrides.get(name) or values.get(name.upper())
        return float(raw) if raw else default

    return Config(
        url,
        token,
        work_root=_path("work_root", DEFAULT_WORK_ROOT),
        lock_path=_path("lock_path", DEFAULT_LOCK_PATH),
        poll_seconds=_float("poll_seconds", 5.0),
        keep_failed=int(_float("keep_failed", 1)),
        beat_seconds=_float("beat_seconds", 20.0),
    )


# ---------------------------------------------------------------------------
# The HTTP client
# ---------------------------------------------------------------------------

class WorkerAPI:
    """The worker's half of the API in ``web/worker_api.py``.

    Stdlib only: the Jetson venv carries the CUDA stack and does not carry
    ``requests``, and adding a dependency for six calls is not worth the risk of
    disturbing a venv that took a day to get working.
    """

    def __init__(self, url: str, token: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self._token = token
        self.timeout = timeout
        #: every request this client made, for tests and for a smoke diagnosis.
        #: Paths only -- never headers, because the header is where the token is.
        self.requests: list[tuple[str, str]] = []

    def __repr__(self) -> str:
        return f"WorkerAPI(url={self.url!r}, token=<{len(self._token)} chars>)"

    def _request(self, method: str, path: str, data: bytes | None = None,
                 json_body: dict | None = None,
                 content_type: str = "application/octet-stream"):
        body = data
        headers = {"Authorization": f"Bearer {self._token}"}
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif body is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.url + path, data=body,
                                         headers=headers, method=method)
        self.requests.append((method, path))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = resp.read()
                return resp.status, payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise APIError(exc.code, detail, method, path) from None
        except urllib.error.URLError as exc:
            # The app is unreachable. Its own message can name this host; the
            # token is not in it, and this is the operator's clue that the
            # tailnet or the deployment is the problem.
            raise APIError(0, f"could not reach {self.url}: {exc.reason}",
                           method, path) from None

    def claim(self) -> dict | None:
        """One job, or ``None`` when the queue is empty (204)."""
        status, payload = self._request("POST", "/worker/claim")
        if status == 204 or not payload:
            return None
        return json.loads(payload)

    def fetch_input(self, job_id: str) -> bytes:
        _status, payload = self._request("GET", f"/worker/jobs/{job_id}/input.obj")
        return payload

    def progress(self, job_id: str, stage: str) -> None:
        """The heartbeat. Renews the lease; 409 means it was lost."""
        try:
            self._request("POST", f"/worker/jobs/{job_id}/progress",
                          json_body={"stage": stage})
        except APIError as exc:
            if exc.status == 409:
                raise LeaseLost(job_id) from None
            raise

    def upload(self, job_id: str, name: str, data: bytes) -> None:
        self._request("PUT", f"/worker/jobs/{job_id}/files/{name}", data=data)

    def done(self, job_id: str, modes) -> None:
        """Claim the job done, carrying the mode rows the app must write.

        The rows are not decoration: the app writes them to ``mse.modes`` in the
        same transaction that marks the job DONE, so a ``/done`` that reached it
        without rows would be refused (400) rather than producing a
        finished job with an empty mode list.
        """
        try:
            self._request("POST", f"/worker/jobs/{job_id}/done",
                          json_body={"modes": [
                              {"index": int(i), "k0_per_m": float(k0),
                               "freq_hz": float(f),
                               "vacuum_wavelength_m": float(wl)}
                              for (i, k0, f, wl) in modes
                          ]})
        except APIError as exc:
            if exc.status == 409:
                raise LeaseLost(job_id) from None
            raise

    def done_bands(self, job_id: str, bands: dict) -> None:
        """Claim a pc job done, carrying the band diagram the app must write.

        The pc sibling of :meth:`done`, and the same argument holds: the app
        merges the diagram into ``pcs`` in the same transaction that marks the
        job DONE, so a ``/done`` without a diagram is refused (400) rather than
        leaving a finished job that displays as blank.
        """
        try:
            self._request("POST", f"/worker/jobs/{job_id}/done",
                          json_body={"bands": bands})
        except APIError as exc:
            if exc.status == 409:
                raise LeaseLost(job_id) from None
            raise

    def failed(self, job_id: str, message: str) -> None:
        """Give up on this job for good. TERMINAL.

        Only for a DETERMINISTIC failure -- see ``classify``. The app marks the
        row FAILED with this message and the owner reads it on the job page; the
        job is not requeued. Sending a transient failure here would kill a job
        that would have succeeded on the next attempt, which is why the choice
        between this and ``release`` is made in one tested place rather than
        here.
        """
        try:
            self._request("POST", f"/worker/jobs/{job_id}/failed",
                          json_body={"message": message[:2000]})
        except APIError as exc:
            if exc.status == 409:
                # Not our lease any more: the app has already given the job to
                # somebody else, so there is nothing left for us to report and
                # nothing to undo.
                return
            raise

    def release(self, job_id: str, message: str = "") -> dict:
        """Hand the job back with NO verdict, because the reason was transient.

        The app requeues it and counts the release; after ``MAX_RELEASES`` it
        marks the job FAILED itself, so a worker that can never finish a job
        still cannot loop. The response says which happened
        (``{"released": True|False, "failed": ...}``) -- a return value, not an
        exception, because "it was requeued" is the outcome this call asked for
        and only "the app failed it for us" is worth the caller's attention.
        """
        try:
            _status, payload = self._request("POST",
                                             f"/worker/jobs/{job_id}/release",
                                             json_body={"message": message[:2000]})
        except APIError as exc:
            if exc.status == 409:
                # Not our lease any more; the app owns the decision now.
                return {"ok": False, "released": False, "failed": False}
            raise
        if not payload:
            return {"ok": True}
        try:
            return json.loads(payload)
        except ValueError:
            return {"ok": True}


# ---------------------------------------------------------------------------
# The bench lock
# ---------------------------------------------------------------------------

class BenchLock:
    """An exclusive advisory lock on the GPU, shared with the lead's benchmarks.

    ``flock`` rather than a sentinel file: the kernel releases it when the
    process dies, so a worker that is SIGKILLed cannot leave the bench locked
    forever -- which a lockfile whose owner forgot to delete it always can.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock, or return False if someone else holds it."""
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        if not self.acquire():
            raise WorkerError(f"{self.path} is held by another process")
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# ---------------------------------------------------------------------------
# Disk hygiene
# ---------------------------------------------------------------------------

def job_dir(work_root: Path, job_id: str) -> Path:
    """The work directory for one job, under the root and never outside it.

    ``job_id`` comes off the wire, so it is checked rather than trusted: a id
    containing a separator or ``..`` would otherwise let the app decide where
    this process writes and deletes.
    """
    safe = "".join(c for c in job_id if c.isalnum() or c in "-_")
    if safe != job_id or not safe:
        raise WorkerError(f"refusing to use {job_id!r} as a directory name")
    return Path(work_root) / safe


def sweep_failed(work_root: Path, keep: int) -> list[str]:
    """Delete all but the ``keep`` newest ``*.failed`` directories.

    The lead's rule, and the reason it exists: a 0.2 mm job writes ~500 MB, so
    without this the Jetson's disk fills over a few days and the failure looks
    like the GPU breaking rather than a full disk.

    Newest is by mtime. Returns the names deleted, so the caller can log them and
    a test can assert on exactly which ones went.
    """
    if keep < 0:
        return []
    candidates = [p for p in Path(work_root).glob("*.failed") if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for path in candidates[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed


def mark_failed(work_dir: Path) -> Path:
    """Rename a finished-with-failure dir so the sweeper can find it.

    A rename rather than a flag file: it is atomic, and it keeps "this is the
    wreckage of the newest failure" visible in a plain ``ls`` on the box, which
    is how the operator will actually look at it.
    """
    target = work_dir.with_name(work_dir.name + ".failed")
    try:
        work_dir.rename(target)
    except OSError:
        return work_dir
    return target


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

class Worker:
    """Claim, solve, upload. One job at a time, forever, until SIGTERM.

    ``solve`` is injected rather than imported here so the tests can drive every
    path -- including the ones that only fire when the solve raises -- without
    TetGen, an eigen solve, or a GPU.
    """

    def __init__(self, config: Config, api: WorkerAPI, lock: BenchLock,
                 solve, store_factory, log=print, solver_dir: Path = _SOLVER_DIR,
                 log_exception=None):
        self.config = config
        self.api = api
        self.lock = lock
        self._solve = solve
        self._store_factory = store_factory
        self.log = log
        #: the traceback sink. Injected so a test can assert the traceback was
        #: written WITHOUT capturing a real stdio journal, and defaulted so every
        #: existing constructor call keeps working. See ``log_exception``.
        self.log_exception = log_exception or self._default_log_exception
        self.solver_dir = Path(solver_dir)
        self.stop = threading.Event()
        self._stage = "starting"
        self._heartbeat: threading.Thread | None = None
        #: set by the heartbeat thread when it discovers the lease is gone, read
        #: by the main thread. Initialised HERE, not in ``run_job``: the beat
        #: thread starts before the first sleep and would otherwise read an
        #: attribute that does not exist yet on its very first lost lease.
        self._lease_lost = False
        #: the ids solved (or attempted) this process has seen, for the log and
        #: for a test that wants to know the loop really moved on.
        self.processed: list[str] = []

    # -- logging -----------------------------------------------------------

    def _log(self, message: str) -> None:
        self.log(f"[mse-worker] {message}", flush=True)

    def _default_log_exception(self, context: str, exc: BaseException) -> None:
        """Write a traceback to the journal, with the token redacted.

        ``traceback.format_exception`` renders the whole chain, which is the
        point: on deploy 3 the worker logged a generic owner sentence and threw
        the real one away, so "the problem has been recorded" was a lie and the
        failure was undiagnosable. The owner-facing message stays generic; THIS
        is where the truth goes.

        Redaction is not optional even though the token is not normally in a
        traceback: an ``APIError`` body is the app's own text, and a URL with a
        query string or a header echoed into an error page can carry it. The
        token is read from the config at call time and replaced wherever it
        appears, so a leak has to get past an exact-string match AND the
        ``Bearer`` prefix to show up.
        """
        self.log(context, flush=True)
        for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
            self.log(_redact_token(line.rstrip("\n"), getattr(self.config, "token", "")),
                     flush=True)

    # -- the heartbeat -----------------------------------------------------

    def _start_heartbeat(self, job_id: str, lease_seconds: float,
                         reporter: "StageReporter | None" = None) -> None:
        """Renew the lease on a thread until the job ends.

        The interval is the worker's own ``beat_seconds``, NOT a fraction of
        the app's lease. It must be short enough that beats land *inside* a
        solve: deriving it from the lease gave 100 s against the 300 s default,
        so a solve that took less than that was finished before the first beat,
        and the app's only evidence that the worker was alive was the claim.
        A beat is one small POST; the cost of a short interval is negligible
        next to a lease that lapses under a perfectly healthy job.

        ``lease_seconds`` is still taken, and still used: a beat interval longer
        than the lease it renews could never hold the lease, which is a
        configuration error worth refusing loudly rather than solving into.
        A lost lease is recorded and the main thread picks it up -- raising here
        would only kill this thread and leave the solve running on a job nobody
        owns.

        The stage sent is read *live* from ``reporter`` on every beat, so the
        lease renewal carries what the solver is actually doing rather than
        whichever stage it had when the thread started.
        """
        # The floor is 0.01 s, not 1 s. It was 1 s when this was derived from
        # the lease, and the inherited ``max(1.0, ...)`` then silently overrode
        # any shorter configured beat -- a 0.02 s beat became a 1 s one, so a
        # solve shorter than a second was finished before the first beat and
        # the app saw no sign of life at all. A floor only has to stop a
        # mistyped zero from spinning the thread; it must not silently replace
        # the interval the operator asked for.
        lease = float(lease_seconds or 0.0)
        interval = max(0.01, self.config.beat_seconds)
        if lease and interval >= lease:
            self._log(f"beat_seconds={interval} is not shorter than the lease "
                      f"({lease}s); the heartbeat could never keep this job -- "
                      f"set beat_seconds below the app's lease")
            interval = max(0.01, lease / 3.0)

        def current_stage() -> str:
            return reporter.stage if reporter is not None else self._stage

        def beat():
            while not self._beat_done.wait(interval):
                # NOTE: checks the PER-JOB event, never ``self.stop``. ``stop``
                # means "the process is shutting down" and is only set by a
                # signal; a job finishing is not a shutdown, and a beat thread
                # waiting on ``stop`` would keep renewing a lease for a job that
                # is over until the process itself was asked to stop.
                try:
                    self.api.progress(job_id, current_stage())
                except LeaseLost:
                    self._log(f"lease on {job_id} expired; abandoning the job")
                    self._lease_lost = True
                    return
                except APIError as exc:
                    # A transient app problem is not a reason to abandon a
                    # solve that is minutes in; the lease has two more beats
                    # before it expires.
                    self._log(f"heartbeat for {job_id} failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    # Anything else must not kill this thread silently: a dead
                    # heartbeat means the lease expires with nobody watching,
                    # and the main thread would keep solving a job it no longer
                    # owns. Record it as a lost lease, which is the safe read.
                    self._log(f"heartbeat for {job_id} broke: "
                              f"{type(exc).__name__}: {exc}")
                    self._lease_lost = True
                    return

        self._lease_lost = False
        self._beat_done = threading.Event()
        self._heartbeat = threading.Thread(target=beat, name="heartbeat",
                                           daemon=True)
        self._heartbeat.start()

    def _stop_heartbeat(self) -> None:
        """Stop the beat thread and WAIT for it, so no thread outlives its job.

        The event is set first and the join is generous: a thread that is mid
        HTTP call finishes that call and then exits, rather than being abandoned
        to keep renewing a lease for a finished job. The earlier version joined
        for two seconds against the process-wide ``stop`` event (which is not
        set when a job merely ends), so every finished job leaked a live beat
        thread for the rest of the process's life.
        """
        if self._heartbeat is not None:
            self._beat_done.set()
            self._heartbeat.join(timeout=10.0)
            if self._heartbeat.is_alive():  # pragma: no cover - a stuck socket
                self._log("the heartbeat thread did not stop; it will die with "
                          "the process")
            self._heartbeat = None

    # -- one job -----------------------------------------------------------

    def run_job(self, job: dict) -> str:
        """Solve one claimed job. Returns 'done', 'failed' or 'abandoned'.

        Never raises: the loop above must survive any job, because the process
        staying up is the whole point of it being warm.
        """
        job_id = str(job["job_id"])
        self.processed.append(job_id)
        #: The lease the app actually granted at claim time. The claim route
        #: puts it in the payload for this reason (the job row cannot carry it:
        #: the SELECT the row is read through never returns that column), so the
        #: ``or 300.0`` here is a last-resort default for a payload from an older
        #: app -- NOT the value to expect. It is used only to sanity-check the
        #: beat interval; the beating itself is on the worker's own clock.
        lease_seconds = float(job.get("lease_seconds") or 300.0)
        work_dir = job_dir(self.config.work_root, job_id)
        reporter = StageReporter()

        self._log(f"claimed {job_id}: {job.get('n_modes')} modes, "
                  f"units {job.get('units')}, scale {job.get('scale')}")
        self._start_heartbeat(job_id, lease_seconds, reporter)
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(work_dir, 0o700)

            # A pc job has NO upload: its mesh is built inside the solve, and
            # its result is the band diagram, which travels on its own route
            # (``_done`` -> ``done_bands``) rather than as result files. Fetching
            # here would 404 -- the app has no input.obj for a pc row -- and the
            # result scan after the solve would then find no files to upload, so
            # BOTH steps are skipped from ONE decision, made on the claim's kind,
            # which is the same field ``real_solve`` dispatches on.
            is_pc = job.get("kind") == "pc"
            store = self._store_factory(work_dir, job_id)
            if not is_pc:
                input_bytes = self.api.fetch_input(job_id)
                store.write(_input_key(job_id), input_bytes)

            # The stage names the phase the owner is shown. "meshing" would be a
            # lie for a pc job, whose mesh is built inside the solve -- the same
            # reason its solve branch reports "solving".
            self._stage = "solving" if is_pc else "meshing"
            reporter.stage = self._stage
            summary = self._solve(job, store, store, stage=reporter)
            if self.stop.is_set():
                # SIGTERM arrived while solving. HAND IT BACK rather than
                # finishing it: systemd's TimeoutStopSec is a bounded window and
                # a job that keeps computing past it gets SIGKILLed, which leaves
                # a lease dangling until it expires and looks to the owner like a
                # hung job. Uploading a result the app has already re-queued
                # would be worse still.
                #
                # A TRANSIENT reason, not a terminal one -- nothing is wrong with
                # the job, an operator stopped the service -- so this goes to
                # ``/release`` (via ``classify``) and the job is claimable again
                # immediately. Reporting it FAILED would burn a perfectly good
                # job every time the service is restarted.
                raise WorkerError(
                    "the worker was asked to stop; this check runs after "
                    "solve() returned, so it sees a stop that landed while the "
                    "worker was between operations -- handing the job back to "
                    "the queue rather than reporting a result nobody is "
                    "waiting for"
                )
            self._stage = "uploading"
            reporter.stage = "uploading"

            if not is_pc:
                for name in _result_names(store, job_id):
                    self.api.upload(job_id, name, store.read(_key(job_id, name)))
            self._done(job_id, work_dir, summary, reporter)
            return "done"
        except LeaseLost:
            self._log(f"stopped working on {job_id}: the lease is gone")
            shutil.rmtree(work_dir, ignore_errors=True)
            return "abandoned"
        except BaseException as exc:  # noqa: BLE001 - any job failure is survivable
            self._fail(job_id, work_dir, exc)
            return "failed"
        finally:
            self._stop_heartbeat()
            # Unconditionally. The earlier form only released when the job had
            # NOT failed, on the theory that the failure path would release it
            # itself -- it did not, so one failed job left the bench locked for
            # the rest of the process's life, and every later job (and the
            # lead's own benchmarks) waited on a lock whose owner was idle.
            # ``release`` is idempotent: the loop calls it again after a claim
            # comes back empty, and that second call is a no-op.
            self.lock.release()
            # Free the GPU memory this job held, on success AND failure. This
            # process is deliberately WARM -- the CUDA context and the solver's
            # imports are built once and kept -- but the FACTORISATIONS are
            # per-job and must not accumulate. On the Jetson the GPU has unified
            # memory, so a leak here is system RAM, and the ``MemoryMax`` cgroup
            # does not reliably account for cuDSS/nvmap allocations: two large
            # jobs back to back would climb until the OS OOM-killed the worker.
            # A host OOM kill arrives as SIGKILL with no exception, so it is not
            # a ``classify()`` case -- the lease simply expires and the app
            # requeues the row. This is what stops that happening.
            _release_gpu_memory()

    def _done(self, job_id: str, work_dir: Path, summary, reporter) -> None:
        if self._lease_lost:
            # The heartbeat saw the lease go. Uploading was still correct (the
            # files are the job's), but claiming 'done' would be a lie.
            raise LeaseLost(job_id)
        # The ROWS come from the reporter, not from ``summary``. Both carry the
        # same modes today -- ``summary["modes"]`` is the list ``write_results``
        # wrote to modes.json, and it is what the solver handed ``insert_modes``
        # -- but the reporter is the recorded copy of exactly what the solve
        # reported, so there is one producer and the tier-1 test
        # ``test_the_reported_rows_agree_with_the_summary`` fails if they drift.
        # A mismatch here would ship a job whose table rows disagree with its
        # own modes.json, which is worse than either alone.
        # The pc job's result is a band diagram, and it travels under its own
        # key. Branched on the SUMMARY's shape, not on the row's kind again:
        # the solve that just ran is what says what it produced, so the two
        # cannot disagree. ``summary`` is a dict for both paths (cavity's
        # ``SOLVER.solve`` returns one too).
        bands = summary.get("bands") if isinstance(summary, dict) else None
        if bands is not None:
            self.api.done_bands(job_id, bands)
        else:
            self.api.done(job_id, reporter.modes)
        self._log(f"finished {job_id}: {summary}")
        shutil.rmtree(work_dir, ignore_errors=True)

    def _fail(self, job_id: str, work_dir: Path, exc: Exception) -> None:
        """Tell the app why this job did not finish -- with the right VERB.

        The two verbs are not interchangeable and picking the wrong one is the
        bug this whole dispatch exists to fix:

        * TERMINAL (``/failed``) -- the solver, the mesher or the job row is at
          fault, so the same worker, or the next one, will fail it the same way.
          The app marks the row FAILED and the owner sees the message on the job
          page. Requeueing it means an infinite fail/claim/fail loop with the
          reason visible only in a log nobody reads.
        * TRANSIENT (``/release``) -- the machine or the wire is at fault, and
          the very next attempt may well succeed. The job goes back to the queue
          with no verdict. The app counts releases and fails the job after
          ``MAX_RELEASES``, so even a worker flapping forever cannot loop.
        * LEASE-LOST -- not ours to report on at all; the app has already given
          it to somebody else.

        The message is the one the owner will read, so it is the solver's own
        plain-English mapping where there is one (see ``owner_message``), not
        ``repr(exc)``: a traceback line is exactly what the owner cannot act on.
        """
        kind = classify(exc)
        message = _owner_message(exc)
        # The OWNER gets the generic sentence; the JOURNAL gets the truth. These
        # two are deliberately different and the failure mode is when only the
        # first is written -- which is what happened on deploy 3: every Jetson
        # job failed right after a good solve, and the only record was "Something
        # went wrong while computing the modes. The problem has been recorded",
        # which was FALSE because nothing was recorded. A failure nobody can
        # diagnose is a failure that will keep happening.
        #
        # The type, the exception's own message and a full traceback, at ERROR.
        # ``log_exception`` redacts the token, so this is safe to journal.
        self._log(f"job {job_id} failed ({kind}): {message}")
        self.log_exception(
            f"[mse-worker] job {job_id} failed ({kind}); the owner was told: "
            f"{message!r}", exc)
        if isinstance(exc, APIError):
            # The HTTP half in its own line, because it carries fields the
            # traceback does not: the status, the method, the path and the
            # app's own response body. On a 5xx the body is where the app's
            # traceback would be, truncated to 500 chars by ``_request``.
            self._log(f"job {job_id}: the app refused a call -- {exc.describe()}")
        if kind == "lease-lost":
            # ``_done`` raises LeaseLost rather than claiming a job it no longer
            # owns, and the app has already reassigned the row. Reporting
            # anything would be a write by a worker that does not hold the lease
            # (the route answers 409), so the only honest action is silence.
            self._log(f"not reporting {job_id}: this worker no longer holds it")
        else:
            try:
                if kind == "terminal":
                    self.api.failed(job_id, message)
                else:
                    self.api.release(job_id, message)
            except APIError as api_exc:
                # Could not tell the app. The lease will expire and the app's
                # own sweep will requeue (and count) the row; say so rather than
                # pretending the report landed.
                self._log(f"could not report {job_id} ({kind}) to the app "
                          f"({api_exc}); the lease will expire and the app will "
                          f"requeue it")
        mark_failed(work_dir)
        for name in sweep_failed(self.config.work_root, self.config.keep_failed):
            self._log(f"removed old failure directory {name}")

    # -- the loop ----------------------------------------------------------

    def claim_loop(self, max_jobs: int | None = None) -> int:
        """Poll, claim, solve, repeat. Returns the number of jobs attempted.

        ``max_jobs`` exists for the tests and for a one-shot diagnostic run; the
        service runs with no limit until it is signalled.

        A job that fails goes straight back into the pool, and ``claim_jetson``
        takes the OLDEST claimable row -- so the loop is handed the job it just
        gave up on, immediately. That is no longer a problem to work around
        here: ``/failed`` is TERMINAL (the row ends FAILED and stops being
        claimable) and ``/release`` is bounded by the app, so a job this worker
        cannot finish stops coming back on its own. There is nothing to decline
        and nothing to remember between iterations.
        """
        attempted = 0
        #: how many times the loop has gone round without attempting a job.
        #: A loop that only DECLINES work -- the lock is held, or the queue is
        #: empty -- would otherwise spin here forever; with a short poll
        #: interval that is a hot loop hammering the server, which is how the
        #: first version was found (it grew until the test was OOM-killed).
        idle_rounds = 0
        max_idle_rounds = 200 if max_jobs is not None else None
        while not self.stop.is_set():
            if max_jobs is not None and attempted >= max_jobs:
                break
            if max_idle_rounds is not None and idle_rounds >= max_idle_rounds:
                # Only for bounded (test/diagnostic) runs: an unbounded service
                # is SUPPOSED to keep polling, but a ``max_jobs`` run that has
                # spent 200 rounds with nothing to do is finished, not busy.
                self.log(f"[mse-worker] stopping: {idle_rounds} rounds with no "
                         f"job to attempt and only {attempted}/{max_jobs} done")
                break
            if not self.lock.acquire():
                # Someone else has the GPU. Do not claim at all: a claimed job
                # this worker cannot run is a job that has vanished from the
                # cluster's point of view while the owner waits. The lock is
                # taken and HELD across the poll interval, so "this worker only
                # takes work while the bench is free" holds for the whole
                # interval and not just at the instant of a claim -- which is
                # also why every path below that gives up has to release it.
                self.log(f"[mse-worker] {self.config.lock_path} is held; "
                         f"not claiming")
                idle_rounds += 1
                self.stop.wait(self.config.poll_seconds)
                self.lock.release()
                continue
            try:
                job = self.api.claim()
            except APIError as exc:
                self.log(f"[mse-worker] claim failed: {exc}")
                idle_rounds += 1
                self.stop.wait(self.config.poll_seconds)
                self.lock.release()
                continue
            if job is None:
                idle_rounds += 1
                self.stop.wait(self.config.poll_seconds)
                self.lock.release()
                continue
            self.run_job(job)
            attempted += 1
            idle_rounds = 0
        self.log("[mse-worker] stopped")
        return attempted

    def request_stop(self, *_args) -> None:
        self.stop.set()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _key(job_id: str, name: str) -> str:
    return f"jobs/{job_id}/{name}"


def _input_key(job_id: str) -> str:
    return _key(job_id, "input.obj")


class ResultDirUnavailable(WorkerError):
    """``_result_names`` could not enumerate the results, and here is where.

    Carries the paths it tried. The point is not the message -- it is that a
    directory that was never found is reported as ABSENT-BY-CHECK, with the
    paths named, instead of as "the solver wrote no result files". Those two
    readings are opposite and only one of them is a defect in the solver.
    """

    def __init__(self, tried: list[Path], why: str):
        listed = "; ".join(str(p) for p in tried)
        super().__init__(
            f"the worker could not list this job's result files ({why}). It "
            f"looked in: {listed}. The listing itself is at fault -- retryable, "
            f"and not a verdict on the solve."
        )


def _result_names(store, job_id: str) -> list[str]:
    """The result files the solver actually wrote, in a stable order.

    Enumerated from the store rather than from a list written here: a hardcoded
    list would silently stop uploading a file the day the solver adds one, and
    the job would still be reported as done. Only names the API accepts are
    uploaded, and a name outside the allow-list is a loud failure rather than a
    silent skip -- the two cases mean opposite things.

    WHERE IT LOOKS IS DERIVED. This walked up from ``store.root`` and appended a
    literal ````jobs/```` segment, which is a second copy of the key scheme
    ``_key`` already owns. It happens to agree with ``_key`` today (that is
    checked, both here and in the rig suite), so it is not the deploy-3
    failure -- that was attributed here first and the attribution was WRONG.
    It is still a copy, and a copy that goes stale produces exactly the failure
    this harness exists for, so it now asks the store for one path per key.

    The two readings the caller must not confuse: a directory this could not
    find is reported as ``ResultDirUnavailable`` with the paths named, because
    only a directory that WAS found and holds nothing may be reported as an
    empty solve.
    """
    def path_for(key: str) -> Path:
        try:
            return Path(store._path(key))
        except AttributeError:               # a store that does not expose it
            return Path(store.root) / key

    dirs: list[Path] = []
    for key in (f"jobs/{job_id}", job_id):
        candidate = path_for(key)
        if candidate not in dirs:
            dirs.append(candidate)
    #: EVERY place the results can be, named deliberately rather than by the
    #: accident that ``store.root`` happens to be the work directory today.
    #: The staging directory is the one that caught a genuine bug in this
    #: harness: a solve that takes its input from ``work/<id>`` and stages its
    #: output under ``work/<id>.out/`` (the ``--local-in``/``--local-out``
    #: shape the solver announces) puts its results where ``store.root`` cannot
    #: see them at all, and every job would then fail AFTER a successful solve
    #: with the generic owner sentence -- the deploy-3 symptom.
    staging = Path(f"{store.root}.out")
    if staging not in dirs:
        dirs.append(staging)

    #: A directory that is FOUND and is readable settles the question. The
    #: purpose of the later candidates is the case where an earlier one does not
    #: exist at all -- NOT the case where it exists and is empty. Falling
    #: through on "empty" made the honest empty verdict unreachable and
    #: mislabelled every empty result directory as "no directory was found",
    #: which is false -- the same absent-vs-empty confusion this function exists
    #: to fix, inside the fix.
    found_a_directory = False
    for path in dirs:
        try:
            entries = sorted(path.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:  # exists but unreadable -- also not the solver
            raise ResultDirUnavailable([path], str(exc)) from exc
        found_a_directory = True
        names = []
        for entry in entries:
            if not entry.is_file() or entry.name == "input.obj":
                continue
            if entry.name in _UPLOAD_EXACT or _is_mode_file(entry.name):
                names.append(entry.name)
            else:
                raise WorkerError(
                    f"the solver wrote {entry.name!r}, which the worker API does "
                    f"not accept; add it to web/worker_api.py and here, or it "
                    f"will be lost while the job reports success"
                )
        if names:
            return names

    if found_a_directory:
        raise WorkerError(
            f"the solver wrote no result files for {job_id}; refusing to report "
            f"a job as done when there is nothing to show for it"
        )
    raise ResultDirUnavailable(
        dirs, "no directory holding this job's results was found"
    )


def _is_mode_file(name: str) -> bool:
    if not name.startswith("mode_") or "." not in name:
        return False
    stem, _, suffix = name.rpartition(".")
    if suffix not in ("bin", "vtu"):
        return False
    number = stem[len("mode_"):]
    return len(number) == 2 and number.isdigit()


class StageReporter:
    """The ``db`` object ``solve()`` reports stages and modes to.

    ``solve()`` writes progress with ``db.set_stage(job_id, status, stage=...)``
    when it is given a ``db``, and it has no other channel for "which stage am I
    in". The worker needs exactly that: the heartbeat thread renews the lease
    with the *real* stage, and a heartbeat that always says "starting" tells the
    operator nothing about a job that has been meshing for ten minutes.

    It also RECORDS the mode rows. ``solve()`` calls ``delete_modes`` and
    ``insert_modes`` for a ``db`` that can take them, and this used to implement
    only ``set_stage`` -- so on the Jetson the solve completed all nine thousand
    tets, wrote all its result files, and then died with ``AttributeError:
    'StageReporter' object has no attribute 'delete_modes'``. The recorded rows
    are what the worker sends to ``/done``; the app writes them to ``mse.modes``
    in the same transaction that marks the job done, because a DONE with no
    modes is indistinguishable from a job that has none.

    This is a duck-type of the methods ``solve`` calls, not a second database
    client. It deliberately does NOT talk to the database: the Jetson has no
    credentials and no route, and both the stage and the rows reach the app
    through its own authenticated calls -- ``/progress`` and ``/done``.
    """

    def __init__(self):
        self.stage = "starting"
        #: The rows as ``solve()`` handed them to ``insert_modes``:
        #: ``(index, k0_per_m, freq_hz, vacuum_wavelength_m)``.
        self.modes: list[tuple] = []

    def set_stage(self, _job_id, _status, stage=None, **_ignored):
        if stage:
            self.stage = str(stage)

    def delete_modes(self, _job_id):
        """Drop the rows already recorded, mirroring the table's own DELETE.

        A solve reports its modes once, but this is the same contract the
        cluster path's ``JobStore.delete_modes`` implements, and a re-solve
        under the same reporter must not accumulate duplicate rows.
        """
        self.modes = []

    def insert_modes(self, _job_id, rows):
        """Record the mode rows ``solve()`` produced."""
        self.modes = [tuple(row) for row in rows]


def _release_gpu_memory() -> None:
    """Hand the backend's GPU memory back between jobs. Never raises.

    Lives here rather than inline so the tier-1 tests can assert the ORDER
    (factors closed before the pools are emptied) and that it is called on both
    the success and the failure path, without a GPU present. A failure to free
    is logged and swallowed: this runs in a ``finally``, and turning it into a
    job failure would report an infrastructure cleanup problem as a problem with
    the owner's object.
    """
    try:
        if str(_SOLVER_DIR) not in sys.path:
            sys.path.insert(0, str(_SOLVER_DIR))
        import mse_backend  # noqa: PLC0415 - deliberately late, as elsewhere

        closed = mse_backend.release_gpu_memory()
        if closed:
            print(f"[mse-worker] freed {closed} GPU factorisation(s)", flush=True)
    except Exception as exc:  # noqa: BLE001 - cleanup must not fail a job
        print(f"[mse-worker] could not release GPU memory: {exc!r}", flush=True)


def _owner_message(exc: BaseException) -> str:
    """What the OWNER reads on the job page for this failure.

    Delegates to the solver's own mapping wherever the solver is importable,
    because that function is the one the cluster path already uses
    (``mse_maxwell_solver.owner_message``, called from its failure handler) --
    one definition of "what the owner is told", not two that drift. Only the
    failures the owner can act on are shown verbatim; everything else gets the
    generic apology.

    The fallbacks below are for the cases where the solver cannot be imported at
    all (a unit test with a fake solve, an early startup failure): the answer is
    still plain English, and it still never contains a traceback. ``WorkerError``
    is our own type and carries an already-written message, so it passes through
    -- that is how the SIGTERM hand-back and the missing-cap refusal read to the
    owner.
    """
    if isinstance(exc, WorkerError) and str(exc):
        return str(exc)
    try:
        if str(_SOLVER_DIR) not in sys.path:
            sys.path.insert(0, str(_SOLVER_DIR))
        import mse_maxwell_solver as SOLVER  # noqa: PLC0415 - deliberately late
    except Exception:  # noqa: BLE001 - no solver means no mapping, not a crash
        return ("Something went wrong while computing the modes. The problem "
                "has been recorded; please try again, or contact support if it "
                "keeps happening.")
    return SOLVER.owner_message(exc)


def real_solve(job: dict, store_in, store_out, stage: StageReporter | None = None):
    """Run the production solver. Imported lazily so --help works off the Jetson.

    The import happens once per process, which is the point of the worker being
    warm: the first job pays it, every later job reuses the interpreter state.

    The element caps come from the CLAIM, not from constants here. The worker
    cannot know them: they are properties of the machines the app is configured
    for, and a second copy on the Jetson is a second source of truth that would
    drift -- which is precisely the defect (app cap 600k, solver default 80k)
    that made every large job fail. A claim that arrives without them is a
    protocol error, not a cue to guess.
    """
    if job.get("kind") == "pc":
        # A photonic-crystal job: no mesh, no GPU, no modes. Dispatched on the
        # claim's kind, which the app set from the row -- the worker does not
        # guess a kind from the payload's shape, because a cavity job with an
        # absent upload and a pc job would then be indistinguishable.
        from pc_solve import solve_pc  # noqa: PLC0415 - sibling module, late

        # The stage names the page shows. "solving" rather than "meshing": a pc
        # job's mesh is built inside the solve, and a stage that named a phase
        # the worker does not control would be a lie the progress bar repeats.
        if stage is not None:
            stage.stage = "solving"
        bands = solve_pc(job)
        # A summary shaped like the cavity one for the fields ``_done`` and the
        # log read, but carrying ``bands`` instead of ``modes``. ``_done``
        # branches on which key is present, NOT on job["kind"] a second time:
        # one decision, made here where the solve happened, so the two cannot
        # disagree about what was just produced.
        return {"job_id": str(job["job_id"]), "kind": "pc", "bands": bands,
                "modes": []}

    if job.get("kind") not in (None, "cavity"):
        # Every other kind is a NAMED failure, not a cavity solve. This used to
        # be a silent else: anything that was not `pc` fell through to the
        # cavity path, so a row carrying a kind the worker does not implement
        # would be solved as if it were a cavity -- the same shape of defect as
        # the cluster launching a pc row as a cavity Job, on the other machine.
        # `None` is allowed because it means the claim did not say, which is how
        # a cavity row created before `kind` existed arrives.
        raise WorkerError(
            f"the worker does not implement kind={job.get('kind')!r}; it solves "
            f"kind='cavity' and kind='pc'. Refusing rather than falling through "
            f"to the cavity solver, which would produce a result for a job the "
            f"worker did not understand."
        )

    if str(_SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(_SOLVER_DIR))
    import mse_maxwell_solver as SOLVER  # noqa: PLC0415 - deliberately late

    if job.get("max_tets") is None:
        raise WorkerError(
            "the claim carried no max_tets; the worker will not invent an "
            "element cap, because guessing one is how jobs the app accepted "
            "died in the solver"
        )
    # Set the solver's own cap for this job, so the mesh guard and the
    # post-mesh check agree with what the app approved.
    SOLVER.set_max_tets(int(job["max_tets"]))
    if job.get("cpu_max_tets") is not None:
        os.environ["MSE_CPU_MAX_TETS"] = str(int(job["cpu_max_tets"]))

    return SOLVER.solve(
        str(job["job_id"]), str(job["units"]), float(job["n_index"]),
        int(job["n_modes"]), store_in, store_out,
        max_tets=int(job["max_tets"]),
        grid=int(job.get("grid") or 64),
        verbose=True, backend="gpu", db=stage,
        min_feature_m=job.get("min_feature_m"),
        scale=float(job.get("scale") or 1.0),
        # Straight through, no parsing and no defaulting. The worker does not
        # interpret the table -- it has no geometry and no business deciding
        # which part is which -- and it must NOT turn a claim's absent table
        # into an empty one: the solver is told "vacuum" and told "a table with
        # no parts" by two different values, and only the app knows which it
        # meant.
        materials=job.get("materials"),
    )


def store_factory_for(root: Path):
    """A LocalStore factory. Kept as a function so tests can inject a fake."""
    if str(_SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(_SOLVER_DIR))
    import mse_storage as STORE  # noqa: PLC0415 - deliberately late

    def make(work_root: Path, job_id: str):
        return STORE.LocalStore(work_root)

    return make


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=None,
                        help=f"config file (default: {DEFAULT_ENV_FILE})")
    parser.add_argument("--url", help="override MSE_URL")
    parser.add_argument("--work-root", help="override the work directory")
    parser.add_argument("--lock-path", help="override the bench lock path")
    parser.add_argument("--poll-seconds", type=float,
                        help="how long to wait between empty claims")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="stop after N jobs (diagnostic runs; the service "
                             "runs until signalled)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.env_file, url=args.url,
                             work_root=args.work_root,
                             lock_path=args.lock_path,
                             poll_seconds=args.poll_seconds)
    except WorkerError as exc:
        print(f"[mse-worker] refusing to start: {exc}", file=sys.stderr,
              flush=True)
        return 2

    config.work_root.mkdir(parents=True, exist_ok=True)
    os.chmod(config.work_root, 0o700)
    # Sweep before the first claim, not only after a failure: a reboot is the
    # most likely moment for a stale failure directory to be the newest thing on
    # a full disk.
    for name in sweep_failed(config.work_root, config.keep_failed):
        print(f"[mse-worker] removed old failure directory {name}", flush=True)

    api = WorkerAPI(config.url, config.token)
    worker = Worker(config, api, BenchLock(config.lock_path),
                    solve=real_solve, store_factory=store_factory_for(_SOLVER_DIR))

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, worker.request_stop)

    print(f"[mse-worker] starting against {config.url}; work root "
          f"{config.work_root}; lock {config.lock_path}", flush=True)
    attempted = worker.claim_loop(max_jobs=args.max_jobs)
    return 0 if attempted >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
