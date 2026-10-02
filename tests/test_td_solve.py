"""The ``td`` request contract: the keys the worker reads, and its refusals.

Tiny and pure: no store, no solver. The phase-5 form will be pinned against
``REQUEST_KEYS`` the way ``test_pc_solve`` pins ``pc_form``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import td_solve  # noqa: E402


def _complete():
    return {k: f"value-{k}" for k in td_solve.REQUEST_KEYS}


@pytest.mark.parametrize("missing", td_solve.REQUEST_KEYS)
def test_a_missing_key_is_refused_by_name(missing):
    tds = _complete()
    del tds[missing]
    with pytest.raises(td_solve.TDSolveError) as caught:
        td_solve.request_from({"tds": tds})
    assert missing in str(caught.value)


def test_every_missing_key_is_named_not_just_the_first():
    tds = _complete()
    gone = td_solve.REQUEST_KEYS[:2] + td_solve.REQUEST_KEYS[-1:]
    for key in gone:
        del tds[key]
    message = str(pytest.raises(td_solve.TDSolveError,
                                td_solve.request_from, {"tds": tds}).value)
    for key in gone:
        assert key in message


@pytest.mark.parametrize("tds", [None, [], "{}", 3])
def test_a_request_that_is_not_an_object_is_refused(tds):
    with pytest.raises(td_solve.TDSolveError, match="'tds' request object"):
        td_solve.request_from({"tds": tds})


def test_an_empty_object_is_refused_naming_every_key():
    message = str(pytest.raises(td_solve.TDSolveError,
                                td_solve.request_from, {"tds": {}}).value)
    for key in td_solve.REQUEST_KEYS:
        assert key in message


def test_null_values_are_kept_and_extra_keys_dropped():
    """Present-but-null is a real value (``direction`` for a point source);
    keys outside the contract (results written back) never reach the solve."""
    tds = _complete()
    tds["direction"] = None
    tds["bandwidth"] = 0
    tds["spectrum"] = {"written": "back"}
    request = td_solve.request_from({"tds": tds})
    assert request["direction"] is None
    assert request["bandwidth"] == 0
    assert "spectrum" not in request
    assert list(request) == list(td_solve.REQUEST_KEYS)


# ---------------------------------------------------------------------------
# solve_td (4.2): the free-memory check, the out-of-memory and 45-minute stops
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

FDTD = td_solve._solver()[0]


def _obj():
    import trimesh
    box = trimesh.creation.box(extents=(4, 3, 2))
    box.apply_translation((2, 1.5, 1))
    return ("o block\n" + trimesh.exchange.obj.export_obj(
        box, include_normals=False)).encode()


class _Store:
    def __init__(self):
        self.obj = _obj()

    def read(self, key):
        assert key == "jobs/j1/input.obj", key
        return self.obj


def _job(**changes):
    req = {"boundary": "metal", "source_kind": "point", "profile": "pulse",
           "frequency_hz": 6e10, "bandwidth": 3e10,
           "position": [1e-3, 1.5e-3, 1e-3], "direction": None,
           "polarisation": "z", "samples": None, "sample_duration_s": None,
           "duration_s": 2e-10, "probes": [[3e-3, 1.5e-3, 1e-3, "z"]],
           "dx_m": 1e-3}
    req.update(changes)
    return {"job_id": "j1", "kind": "td", "units": "mm", "scale": 1.0,
            "materials": None, "n_index": 1.5, "tds": req}


PLENTY = lambda: 10**12  # noqa: E731


def test_a_tiny_job_runs_and_rings():
    """Positive control for every stop below: the same job runs to the end."""
    out = td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY)
    traces = out["result"]["traces"]
    assert np.shape(traces) == (1, out["estimate"]["steps"])
    assert np.abs(traces).max() > 0, "the probe must see the pulse"


def _bytes_needed():
    return td_solve.solve_td(_job(), _Store(), xp=np,
                             free_bytes=PLENTY)["estimate"]["bytes"]


def test_too_little_free_memory_defers_before_any_step(monkeypatch):
    """Short by one byte: ``JetsonBusy`` (no verdict), and the solver never ran."""
    need = _bytes_needed()
    ran = []
    monkeypatch.setattr(FDTD, "simulate", lambda *a, **k: ran.append(1))
    with pytest.raises(td_solve.JetsonBusy, match="busy") as info:
        td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=lambda: need - 1)
    assert not isinstance(info.value, td_solve.TDSolveError), (
        "busy is no verdict on the job: it must not read as a refusal")
    assert ran == [], "the check comes before a single device array"


def test_exactly_enough_free_memory_runs():
    need = _bytes_needed()
    td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=lambda: need)


def test_a_run_past_45_minutes_is_stopped_with_advice():
    """The clock jumps past the limit after the run starts: a verdict naming
    the limit and the dx/duration advice, raised from inside the step loop."""
    ticks = iter([0.0] + [td_solve.RUN_LIMIT_S + 1.0] * 10**6)
    with pytest.raises(td_solve.TDSolveError,
                       match=r"stopped after 45 minutes.*at step 0 of .*dx"):
        td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY,
                          clock=lambda: next(ticks))


def test_a_run_just_inside_45_minutes_finishes():
    ticks = iter([0.0] + [float(td_solve.RUN_LIMIT_S)] * 10**6)
    td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY,
                      clock=lambda: next(ticks))


@pytest.mark.parametrize("exc", [
    MemoryError("out of memory allocating 1,234 bytes"),
    RuntimeError("cudaErrorMemoryAllocation: out of memory"),
])
def test_out_of_memory_during_the_run_is_a_verdict(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(FDTD, "simulate", boom)
    with pytest.raises(td_solve.TDSolveError, match="ran out of memory"):
        td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY)


def test_any_other_run_failure_passes_up_untranslated(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("an unrelated kernel fault")
    monkeypatch.setattr(FDTD, "simulate", boom)
    with pytest.raises(RuntimeError, match="unrelated") as info:
        td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY)
    assert not isinstance(info.value, td_solve.TDSolveError)


def test_a_layout_refusal_reaches_the_owner_as_a_verdict():
    """A probe outside the grid is refused by the layout; ``solve_td`` passes
    the layout's own sentence on as a ``TDSolveError``."""
    with pytest.raises(td_solve.TDSolveError, match="probe"):
        td_solve.solve_td(_job(probes=[[9e-3, 1.5e-3, 1e-3, "z"]]), _Store(),
                          xp=np, free_bytes=PLENTY)
