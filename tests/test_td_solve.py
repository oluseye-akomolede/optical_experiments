"""The ``td`` request contract: the keys the worker reads, and its refusals.

Tiny and pure: no store, no solver. The phase-5 form will be pinned against
``REQUEST_KEYS`` the way ``test_pc_solve`` pins ``pc_form``.
"""
from __future__ import annotations

import json
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


# --------------------------------------------------------------------------
# 4.3: the result files


class _OutStore:
    """Records every write, in order, keyed as the worker's ``_key`` keys."""

    def __init__(self):
        self.files = {}
        self.order = []

    def write(self, key, data):
        assert key.startswith("jobs/j1/") and key.count("/") == 2, key
        self.files[key[len("jobs/j1/"):]] = bytes(data)
        self.order.append(key[len("jobs/j1/"):])


def _strict_json(data):
    def refuse(token):
        raise AssertionError(f"{token} is not JSON a browser can parse")
    return json.loads(data, parse_constant=refuse)


#: Two probes on different components and cells, so a transposed trace or a
#: swapped probe row reads differently from the right one.
TWO_PROBES = [[3e-3, 1.5e-3, 1e-3, "z"], [2e-3, 2e-3, 1e-3, "x"]]


def _run_with_files(**changes):
    out = _OutStore()
    changes.setdefault("dx_m", 0.5e-3)  # a non-square grid (1 mm gives 5^3)
    res = td_solve.solve_td(_job(probes=TWO_PROBES, **changes), _Store(), out,
                            xp=np, free_bytes=PLENTY)
    return res, out


def test_the_trace_file_decodes_to_the_runs_traces_row_per_probe():
    res, out = _run_with_files()
    man = _strict_json(out.files["spectrum.json"])
    shape = tuple(man["trace"]["shape"])
    assert shape == (2, res["estimate"]["steps"]) and shape[0] != shape[1]
    got = np.frombuffer(out.files["trace.bin"], dtype="<f4").reshape(shape)
    want = np.asarray(res["result"]["traces"])
    assert np.array_equal(got, want)
    assert not np.array_equal(got[0], got[1]), "fixture: the probes must differ"
    assert [p["component"] for p in man["probes"]] == ["z", "x"]


def test_each_frame_file_is_that_frame_in_xyz_order():
    res, out = _run_with_files()
    man = _strict_json(out.files["spectrum.json"])
    frames = np.asarray(res["result"]["frames"])
    fr = man["frames"]
    assert fr["count"] == frames.shape[0] >= 2 and fr["axes"] == "xyz"
    assert tuple(fr["shape"]) == frames.shape[1:]
    assert len(set(fr["shape"])) == 3, "fixture: a non-square frame"
    assert len(fr["steps"]) == fr["count"] and fr["steps"][-1] == man["steps"] - 1
    for n in range(fr["count"]):
        got = np.frombuffer(out.files[f"frame_{n + 1:02d}.bin"], dtype="<f4")
        assert np.array_equal(got.reshape(fr["shape"]), frames[n]), n
    assert f"frame_{fr['count'] + 1:02d}.bin" not in out.files
    assert np.abs(frames[-1]).max() > 0


def test_the_manifest_is_written_last():
    res, out = _run_with_files()
    assert out.order[-1] == "spectrum.json"
    assert out.order[0] == "trace.bin"
    assert res["files"] == out.order


def test_each_probes_spectrum_spans_the_pulse_band_and_peaks_in_it():
    res, out = _run_with_files()
    man = _strict_json(out.files["spectrum.json"])
    lo, hi = man["band_hz"]
    assert (lo, hi) == (3e10, 9e10)
    for p, probe in enumerate(man["probes"]):
        f = np.asarray(probe["spectrum"]["frequency_hz"])
        a = np.asarray(probe["spectrum"]["amplitude"])
        assert f.size == a.size >= 3 and f.min() >= lo and f.max() <= hi
        df = 1.0 / (man["dt_s"] * man["steps"])
        assert f.min() - lo < df and hi - f.max() < df, "the band edges"
        want_f, want_a = FDTD.spectrum(np.asarray(res["result"]["traces"])[p],
                                       man["dt_s"])
        keep = (want_f >= lo) & (want_f <= hi)
        assert np.allclose(a, np.asarray(want_a)[keep], rtol=1e-6)


def test_probe_positions_are_the_snapped_nodes_in_metres():
    res, out = _run_with_files()
    man = _strict_json(out.files["spectrum.json"])
    for probe, asked in zip(man["probes"], TWO_PROBES):
        assert np.allclose(probe["position_m"], asked[:3], atol=0.25e-3)


def test_a_cw_spectrum_keeps_zero_to_twice_the_frequency():
    assert td_solve.spectrum_band({"profile": "cw", "frequency_hz": 5.0},
                                  5.0) == (0.0, 10.0)
    assert td_solve.spectrum_band({"profile": "pulse", "frequency_hz": 5.0,
                                   "bandwidth": 7.0}, 12.0) == (0.0, 12.0)


def test_a_non_finite_run_is_refused_before_any_file_is_written(monkeypatch):
    real = FDTD.simulate

    def unstable(*a, **k):
        r = real(*a, **k)
        r["frames"][-1, 0, 0, 0] = np.nan
        return r

    monkeypatch.setattr(FDTD, "simulate", unstable)
    out = _OutStore()
    with pytest.raises(td_solve.TDSolveError, match="not finite"):
        td_solve.solve_td(_job(), _Store(), out, xp=np, free_bytes=PLENTY)
    assert out.files == {}


def test_a_non_finite_spectrum_is_refused_before_any_file_is_written(monkeypatch):
    """Finite traces give finite spectra, so this forces one: the spectra are
    computed before the first write, so even this refusal leaves nothing."""
    real = FDTD.spectrum

    def broken(*a, **k):
        freqs, amp = real(*a, **k)
        amp = np.array(amp, dtype=float)
        amp[:] = np.nan
        return freqs, amp

    monkeypatch.setattr(FDTD, "spectrum", broken)
    out = _OutStore()
    with pytest.raises(td_solve.TDSolveError, match="not finite.*probe 1 spectrum"):
        td_solve.solve_td(_job(), _Store(), out, xp=np, free_bytes=PLENTY)
    assert out.files == {}


def test_the_memory_gate_counts_the_field_movie(monkeypatch):
    """The run records frames, so the bytes the free-memory gate and
    ``check_limits`` read must be the estimate WITH the movie -- and for this
    layout that is strictly more than without it, so the test can tell."""
    seen = []
    real = FDTD.estimate_run

    def spy(nodes, **kw):
        est = real(nodes, **kw)
        seen.append((nodes, {k: v for k, v in kw.items()
                             if k != "record_frames"}, est))
        return est

    monkeypatch.setattr(FDTD, "estimate_run", spy)
    out = td_solve.solve_td(_job(), _Store(), xp=np, free_bytes=PLENTY)
    assert len(seen) == 1
    nodes, kw, est = seen[0]
    with_movie = real(nodes, record_frames=True, **kw)["bytes"]
    without = real(nodes, record_frames=False, **kw)["bytes"]
    assert with_movie > without
    assert est["bytes"] == with_movie == out["estimate"]["bytes"]


class _Pool:
    def free_bytes(self):
        return 12_345


class _PoolXp:
    def get_default_memory_pool(self):
        return _Pool()


def test_free_memory_adds_the_blocks_cupys_pool_already_holds(tmp_path):
    """MemAvailable alone undercounts a warm worker: the pool's free blocks
    are reusable by this process but the kernel counts them as used."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemAvailable: 1000 kB\n")
    assert td_solve.available_bytes(_PoolXp(), meminfo) == 1000 * 1024 + 12_345
    assert td_solve.available_bytes(np, meminfo) == 1000 * 1024


# --- the run in its own process ------------------------------------------

import td_child_targets as TARGETS  # noqa: E402


class _Stages:
    def __init__(self):
        self.seen = []

    @property
    def stage(self):
        return self.seen[-1] if self.seen else None

    @stage.setter
    def stage(self, name):
        self.seen.append(name)


def test_a_tiny_job_runs_in_its_own_process_and_writes_its_files(tmp_path):
    """Positive control: the real solve in a spawned child, a real store."""
    import mse_storage
    store = mse_storage.LocalStore(tmp_path)
    store.write("jobs/j1/input.obj", _obj())
    stages = _Stages()
    out = td_solve.solve_td_isolated(_job(), store, store, stages)
    assert "result" not in out, "the arrays stay in the files"
    assert out["files"][0] == "trace.bin" and out["files"][-1] == "spectrum.json"
    for name in out["files"]:
        assert (tmp_path / "jobs" / "j1" / name).is_file(), name
    assert "writing results" in stages.seen, stages.seen


def test_a_kernel_kill_of_the_run_is_the_out_of_memory_verdict():
    stages = _Stages()
    with pytest.raises(td_solve.TDSolveError) as info:
        td_solve.solve_td_isolated(_job(), None, None, stages, run=TARGETS.killed)
    assert str(info.value) == td_solve.oom_message()
    assert "ran out of memory" in str(info.value)
    assert stages.seen == ["solving"]


def test_a_stop_signal_does_not_end_the_run():
    """systemd sends SIGTERM to the whole cgroup; the child finishes anyway."""
    out = td_solve.solve_td_isolated(_job(), None, None, None,
                                     run=TARGETS.sigterm_then_finish)
    assert out["files"] == ["trace.bin"] and "result" not in out


def test_a_childs_error_keeps_its_type_and_its_cause():
    """The worker's classify walks the cause chain: a GPU-init failure
    wrapped in another error is transient, and only the chain says so."""
    import mse_worker
    with pytest.raises(ValueError, match="could not start") as info:
        td_solve.solve_td_isolated(_job(), None, None, None,
                                   run=TARGETS.gpu_init_failure)
    assert "cudaErrorInitializationError" in str(info.value.__cause__)
    assert mse_worker.classify(info.value) == "transient"


def test_an_unpicklable_error_keeps_its_name_and_text():
    with pytest.raises(RuntimeError, match="half open") as info:
        td_solve.solve_td_isolated(_job(), None, None, None,
                                   run=TARGETS.unpicklable)
    assert type(info.value).__name__ == "Odd"


def test_busy_in_the_child_is_busy_in_the_parent():
    import mse_worker
    with pytest.raises(td_solve.JetsonBusy) as info:
        td_solve.solve_td_isolated(_job(), None, None, None, run=TARGETS.busy)
    assert mse_worker.classify(info.value) == "busy"


def test_the_worker_uploads_every_file_the_td_solve_writes(tmp_path):
    """Through the worker's real enumerator and allow-list, not a copy."""
    import mse_worker
    res, out = _run_with_files()

    class Store:
        root = tmp_path

        def _path(self, key):
            return tmp_path / key

    (tmp_path / "jobs" / "j1").mkdir(parents=True)
    for name, data in out.files.items():
        (tmp_path / "jobs" / "j1" / name).write_bytes(data)
    assert sorted(mse_worker._result_names(Store(), "j1")) == sorted(out.files)
