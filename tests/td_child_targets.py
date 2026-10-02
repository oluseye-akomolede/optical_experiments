"""Functions a spawned td child runs in place of ``solve_td``.

A spawned child imports what it runs, so these live in a module of their own
rather than inside a test file.
"""
from __future__ import annotations

import os
import signal
import time

import td_solve


def killed(job, store_in, store_out, stage):
    """What the kernel's OOM killer does: SIGKILL, no exception, no outcome."""
    stage.stage = "solving"
    os.kill(os.getpid(), signal.SIGKILL)


def sigterm_then_finish(job, store_in, store_out, stage):
    """A systemd stop lands mid-run; the run must still finish."""
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.2)
    stage.stage = "writing results"
    return {"job_id": job["job_id"], "kind": "td", "modes": [],
            "result": {"traces": "large"}, "files": ["trace.bin"]}


def gpu_init_failure(job, store_in, store_out, stage):
    try:
        raise RuntimeError("cudaErrorInitializationError: initialization error")
    except RuntimeError as exc:
        raise ValueError("the solver could not start") from exc


class Odd(Exception):
    """Unpicklable: unpickling calls ``Odd(message)``, one argument short."""

    def __init__(self, a, b):
        super().__init__(f"{a} {b}")


def unpicklable(job, store_in, store_out, stage):
    raise Odd("half", "open")


def busy(job, store_in, store_out, stage):
    raise td_solve.JetsonBusy("The Jetson is busy.")


def _note_pid(job):
    with open(job["pid_file"], "w") as f:
        f.write(str(os.getpid()))


def exits_3(job, store_in, store_out, stage):
    """A crash that is not the OOM killer: a plain non-zero exit."""
    os._exit(3)


def aborts(job, store_in, store_out, stage):
    """A native crash (SIGABRT), which is not the OOM killer either."""
    os.abort()


def sleeps(job, store_in, store_out, stage):
    """Stuck inside one step: never reaches the run's own time check."""
    _note_pid(job)
    time.sleep(600)


def reports_then_hangs(job, store_in, store_out, stage):
    """Sends its outcome, then never exits (a non-daemon thread holds the
    process open, as a stuck driver teardown would)."""
    import threading
    _note_pid(job)
    threading.Thread(target=time.sleep, args=(600,), daemon=False).start()
    return {"job_id": job["job_id"], "kind": "td", "modes": [], "files": []}
