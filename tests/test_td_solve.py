"""The ``td`` request contract: the keys the worker reads, and its refusals.

Tiny and pure: no store, no solver. The phase-5 form will be pinned against
``REQUEST_KEYS`` the way ``test_pc_solve`` pins ``pc_form``.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import types
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


def test_an_upload_whose_parts_overlap_fails_naming_both_parts():
    """Two parts inside the block share volume (x in [2, 2.5] mm). The
    voxeliser's OverlapError is an FDTDError, so the owner reads its sentence
    as the job's verdict, with both part names in it."""
    import trimesh

    lines, base = [], 0
    for name, lo, hi in [("block", (0, 0, 0), (4, 3, 2)),
                         ("lens", (0.5, 0.5, 0.5), (2.5, 2.5, 1.5)),
                         ("prism", (2.0, 0.5, 0.5), (3.5, 2.5, 1.5))]:
        box = trimesh.creation.box(extents=np.subtract(hi, lo))
        box.apply_translation(np.add(lo, hi) / 2)
        lines.append(f"o {name}")
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in box.vertices.tolist()]
        lines += ["f " + " ".join(str(base + i + 1) for i in f)
                  for f in box.faces.tolist()]
        base += len(box.vertices)
    store = _Store()
    store.obj = ("\n".join(lines) + "\n").encode()
    job = _job()
    job["materials"] = {"block": 1.0, "lens": 1.5, "prism": 2.0}
    with pytest.raises(td_solve.TDSolveError,
                       match=r"parts 'lens' and 'prism' overlap"):
        td_solve.solve_td(job, store, xp=np, free_bytes=PLENTY)


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


def test_every_probe_node_is_a_movie_sample_position():
    """The small grid keeps every node (stride 1), so each probe's snapped
    position, computed by the probe path, must appear on every axis."""
    res, out = _run_with_files()
    fr = _strict_json(out.files["spectrum.json"])["frames"]
    assert fr["stride"] == [1, 1, 1], "fixture: every node kept"
    pos = fr["positions_m"]
    assert [len(p) for p in pos] == fr["shape"]
    for a in range(3):
        assert np.all(np.diff(pos[a]) > 0), "xyz"[a]
    for probe in _strict_json(out.files["spectrum.json"])["probes"]:
        for a in range(3):
            assert np.isclose(pos[a], probe["position_m"][a], rtol=0,
                              atol=1e-12).sum() == 1, (probe, "xyz"[a])


def test_movie_positions_follow_a_graded_grid_and_its_stride():
    lay = {"origin_m": [-1.0, 0.0, 2.0],
           "nodes": [np.array([0.0, 0.1, 0.3, 0.7, 1.5]),
                     np.array([0.0, 1.0, 2.0, 2.5]),
                     np.array([0.0, 0.25])]}
    got = td_solve.frame_positions(lay, (2, 3, 1), (3, 2, 2))
    assert got == [[-1.0, -0.7, 0.5], [0.0, 2.5], [2.0, 2.25]]


def test_a_movie_whose_shape_disagrees_with_its_stride_is_refused():
    lay = {"origin_m": [0.0] * 3, "nodes": [np.arange(5.0)] * 3}
    with pytest.raises(td_solve.TDSolveError, match="4 samples along y"):
        td_solve.frame_positions(lay, (2, 1, 2), (3, 4, 3))


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
        raw = [np.asarray(FDTD.spectrum(t, man["dt_s"])[1])[keep]
               for t in np.asarray(res["result"]["traces"])]
        peak = max(r.max() for r in raw)
        assert peak != pytest.approx(1.0), "fixture: the raw peak must not be 1"
        assert np.allclose(a, np.asarray(want_a)[keep] / peak, rtol=1e-6)
    assert man["spectrum_scale"] == "largest peak = 1"
    assert max(max(p["spectrum"]["amplitude"]) for p in man["probes"]) \
        == pytest.approx(1.0)


def _sine_spectrum(steps, dt=1e-12, f0=5e10):
    t = np.arange(steps) * dt
    freqs, amp = FDTD.spectrum(np.sin(2 * np.pi * f0 * t), dt)
    return np.asarray(freqs), np.asarray(amp)


def test_a_doubled_step_count_gives_the_same_normalised_spectrum():
    """The raw |rfft| of a steady wave doubles with the step count; the
    normalised spectrum keeps peak 1 at the same frequency, with the same
    curve shape around the peak, counted in frequency bins (both runs hold a
    whole number of periods: 50 and 100)."""
    f1, raw1 = _sine_spectrum(1000)
    f2, raw2 = _sine_spectrum(2000)
    assert raw2.max() / raw1.max() == pytest.approx(2.0, rel=1e-3)
    (n1,), _ = td_solve.normalise_spectra([raw1], ["z"])
    (n2,), _ = td_solve.normalise_spectra([raw2], ["z"])
    k1, k2 = int(n1.argmax()), int(n2.argmax())
    assert n1[k1] == n2[k2] == 1.0
    assert f1[k1] == pytest.approx(f2[k2])
    # The main lobe is [0.5, 1, 0.5] in both; the Hann window's finite-length
    # leakage differs by under 4e-4 (measured), well inside 1e-3.
    assert np.allclose(n1[k1 - 3:k1 + 4], n2[k2 - 3:k2 + 4], atol=1e-3)


def test_a_probe_with_no_signal_is_flat_zero_with_a_note_never_nan():
    live = np.array([0.0, 2.0, 4.0, 1.0])
    out, notes = td_solve.normalise_spectra([live, np.zeros(4)], ["z", "x"])
    assert np.array_equal(out[0], live / 4.0)
    assert np.array_equal(out[1], np.zeros(4))
    assert len(notes) == 1 and "probe 2 (x component)" in notes[0]
    assert "flat zero" in notes[0]

    out, notes = td_solve.normalise_spectra([np.zeros(3), np.zeros(3)], ["z", "y"])
    assert all(np.isfinite(a).all() and not a.any() for a in out)
    assert [n.split(" (")[0] for n in notes] == ["probe 1", "probe 2"]


def test_a_silent_probe_reaches_the_manifest_as_zeros_and_a_caveat(monkeypatch):
    real = FDTD.simulate

    def second_probe_silent(*a, **k):
        r = real(*a, **k)
        r["traces"] = np.array(r["traces"], dtype=float)
        r["traces"][1] = 0.0
        return r

    monkeypatch.setattr(FDTD, "simulate", second_probe_silent)
    _, out = _run_with_files()
    man = _strict_json(out.files["spectrum.json"])
    assert max(man["probes"][0]["spectrum"]["amplitude"]) == pytest.approx(1.0)
    assert not any(man["probes"][1]["spectrum"]["amplitude"])
    assert any(c.startswith("probe 2 (x component)") for c in man["caveats"])


def test_the_viewer_labels_the_spectrum_with_the_workers_scale():
    src = (Path(__file__).resolve().parent.parent / "web/static/tdplot.js").read_text()
    m = re.search(r"export const SPECTRUM_SCALE = '([^']*)';", src)
    assert m and m.group(1) == td_solve.SPECTRUM_SCALE
    assert "${SPECTRUM_SCALE}" in src and "arb. units)', series" not in src.split(
        "drawSpectra")[1]


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


@pytest.mark.parametrize("target, status", [("exits_3", "3"), ("aborts", "-6")])
def test_only_a_kill_signal_reads_as_out_of_memory(target, status):
    """SIGKILL is the OOM killer; any other death is the plain failure."""
    with pytest.raises(td_solve.TDSolveError) as info:
        td_solve.solve_td_isolated(_job(), None, None, None,
                                   run=getattr(TARGETS, target))
    assert str(info.value) != td_solve.oom_message()
    assert "ran out of memory" not in str(info.value)
    assert f"stopped unexpectedly (exit status {status})" in str(info.value)


def _reaped(pid_file) -> bool:
    """Dead AND reaped: a zombie still answers signal 0."""
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def pid_file(tmp_path):
    """Where a child notes its pid; SIGKILLed afterwards if still there, so a
    red run (the child ignores SIGTERM) ends now, not when its sleep does."""
    path = tmp_path / "pid"
    yield path
    if path.exists():
        try:
            os.kill(int(path.read_text()), 9)
        except ProcessLookupError:
            pass


def test_a_child_stuck_past_the_deadline_is_killed_reaped_and_timed_out(pid_file):
    started = time.monotonic()
    with pytest.raises(td_solve.TDSolveError, match="stopped after 45 minutes"):
        td_solve.solve_td_isolated(dict(_job(), pid_file=str(pid_file)), None,
                                   None, None, run=TARGETS.sleeps, poll_s=0.1,
                                   deadline_s=3.0, reap_s=30.0)
    assert time.monotonic() - started < 20, "the deadline did not kill it"
    assert _reaped(pid_file)


def test_a_child_stuck_in_setup_gets_the_setup_verdict(pid_file):
    started = time.monotonic()
    stage = types.SimpleNamespace(stage=None)
    with pytest.raises(td_solve.TDSolveError, match="setup .* took more than"):
        td_solve.solve_td_isolated(dict(_job(), pid_file=str(pid_file)), None,
                                   None, stage, run=TARGETS.stuck_in_setup,
                                   poll_s=0.1, setup_s=3.0, deadline_s=600.0,
                                   reap_s=30.0)
    assert time.monotonic() - started < 20, "the setup cap did not kill it"
    assert stage.stage == "laying out"
    assert _reaped(pid_file)


def test_the_run_clock_starts_at_solving_not_at_spawn():
    """Setup alone outlasts the run deadline; the run itself is well inside
    it. Timed from spawn, this child would be killed during setup."""
    out = td_solve.solve_td_isolated(
        dict(_job(), setup_sleep_s=3.0, run_sleep_s=0.3), None, None, None,
        run=TARGETS.long_setup_short_run, poll_s=0.1, deadline_s=2.0,
        setup_s=30.0)
    assert out["kind"] == "td"


def test_a_reported_outcome_wins_over_a_passed_deadline():
    """The clock is past every limit from its second reading on, but the
    child's outcome is already on its way: the outcome, not the time-out."""
    readings = iter([0.0])
    out = td_solve.solve_td_isolated(
        _job(), None, None, None, run=TARGETS.returns_at_once, poll_s=60.0,
        deadline_s=1.0, setup_s=1.0,
        clock=lambda: next(readings, 1e9))
    assert out["kind"] == "td"


def test_a_child_that_hangs_after_its_outcome_is_killed_and_reaped(pid_file):
    out = td_solve.solve_td_isolated(dict(_job(), pid_file=str(pid_file)), None,
                                     None, None, run=TARGETS.reports_then_hangs,
                                     poll_s=0.1, reap_s=1.0)
    assert out["kind"] == "td"
    assert _reaped(pid_file)


class _Link:
    """A scripted pipe, process and clock for ``solve_td_isolated``.

    Each blocking poll advances the clock by its timeout, then delivers the
    next batch of ``arrivals``; ``on_kill`` is a batch that lands as the
    child is killed (an outcome already in flight). Deterministic, and a
    loop that never ends fails after ``max_polls`` instead of hanging."""

    def __init__(self, arrivals, on_kill=(), max_polls=50):
        self.arrivals, self.on_kill = list(arrivals), list(on_kill)
        self.buffer, self.now, self.polls = [], 0.0, 0
        self.max_polls, self.alive, self.exitcode = max_polls, True, None
        self.killed_at = None

    # context
    def Pipe(self, duplex):
        return self, types.SimpleNamespace(close=lambda: None)

    def Process(self, target, args, daemon):
        return self

    # receive end
    def poll(self, timeout=0.0):
        if timeout and not self.buffer:
            self.polls += 1
            assert self.polls <= self.max_polls, "the wait loop never ended"
            self.now += timeout
            if self.arrivals:
                self.buffer.extend(self.arrivals.pop(0))
        return bool(self.buffer)

    def recv(self):
        message = self.buffer.pop(0)
        if message[0] != "stage":  # a real child exits once it has reported
            self.alive, self.exitcode = False, 0
        return message

    def close(self):
        pass

    # process
    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def kill(self):
        self.killed_at, self.alive, self.exitcode = self.now, False, -9
        self.buffer.extend(self.on_kill)

    def join(self, timeout=None):
        pass

    def clock(self):
        return self.now


def _isolated(link, **kw):
    return td_solve.solve_td_isolated(
        _job(), None, None, types.SimpleNamespace(stage=None), context=link,
        clock=link.clock, poll_s=1.0, **kw)


_OK = ("ok", {"kind": "td", "files": []})


def test_a_repeated_solving_report_does_not_restart_the_run_clock():
    link = _Link([[("stage", "solving")]] * 20)
    with pytest.raises(td_solve.TDSolveError, match="stopped after 45 minutes"):
        _isolated(link, deadline_s=3.0, setup_s=1000.0)
    assert link.killed_at is not None and link.killed_at <= 5.0, link.killed_at


def test_the_run_deadline_not_the_setup_cap_times_a_solving_child():
    link = _Link([[("stage", "solving")]])
    with pytest.raises(td_solve.TDSolveError, match="stopped after 45 minutes"):
        _isolated(link, deadline_s=3.0, setup_s=1000.0)
    assert link.killed_at is not None and link.killed_at <= 5.0, link.killed_at


def test_a_queued_outcome_is_read_before_the_clock_so_the_child_is_not_killed():
    """A stage report and the outcome arrive together, after the setup cap
    has passed: the outcome is read first, and nothing is killed."""
    link = _Link([[("stage", "laying out"), _OK]])
    assert _isolated(link, setup_s=0.5)["kind"] == "td"
    assert link.killed_at is None


def test_an_outcome_landing_as_the_child_is_killed_still_wins():
    link = _Link([[]], on_kill=[_OK])
    assert _isolated(link, setup_s=0.5)["kind"] == "td"
    assert link.killed_at is not None


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
