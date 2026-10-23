"""fdtd.estimate_run / check_limits: the 3.2 refusal (numpy, tiny).

The step count must use the dt simulate really takes, so the dt test runs
simulate itself on a graded gold grid, after checking that the gold lowers dt
below Courant (otherwise both sides would agree on plain Courant).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "dockerfiles", "maxwell_solver"))
import fdtd  # noqa: E402


def _graded_nodes():
    # non-square, graded on every axis, nanometre cells
    x = np.cumsum(np.r_[0.0, np.linspace(5e-9, 7e-9, 13)])
    y = np.cumsum(np.r_[0.0, np.linspace(6e-9, 4.5e-9, 11)])
    z = np.cumsum(np.r_[0.0, np.full(9, 5.5e-9)])
    return [x, y, z]


def _gold(nodes):
    region = np.zeros(tuple(len(n) for n in nodes), dtype=bool)
    region[4:9, 3:8, 3:7] = True
    return region


def _simulated_dt(nodes, metals):
    shape = tuple(len(n) for n in nodes)
    source = fdtd.Source((2, 2, 2), frequency=4e14, bandwidth=2e14)
    out = fdtd.simulate(np.ones(shape, dtype=np.float32), source, 1,
                        spacings=[np.diff(n) for n in nodes],
                        ade_terms=[fdtd.LorentzDrude(name, region)
                                   for name, region in metals.items()])
    return out["dt"]


def test_the_estimate_uses_the_dt_simulate_takes_with_and_without_a_metal():
    nodes = _graded_nodes()
    courant = fdtd.courant_dt(*(np.diff(n).min() for n in nodes))
    plain = fdtd.estimate_run(nodes, duration=1e-15)
    assert plain["dt"] == _simulated_dt(nodes, {}) == courant
    metals = {"Au": _gold(nodes)}
    gold = fdtd.estimate_run(nodes, duration=1e-15, metals=metals)
    assert gold["dt"] < 0.99 * courant       # the control: gold lowers dt
    assert gold["dt"] == _simulated_dt(nodes, metals)


def test_steps_round_the_duration_up_to_whole_steps():
    nodes = _graded_nodes()
    dt = fdtd.estimate_run(nodes, duration=1e-15)["dt"]
    assert fdtd.estimate_run(nodes, duration=100.5 * dt)["steps"] == 101
    assert fdtd.estimate_run(nodes, duration=100 * dt)["steps"] == 100


@pytest.mark.parametrize("boundary,metals,setup", [
    ("metal", {}, "metal"),
    ("open", {}, "open"),
    (("metal", "metal", "open"), {}, "open"),
    (("periodic", "periodic", "open"), {}, "open"),
    (("periodic", "periodic", "metal"), {}, "open"),
    ("metal", {"Au": None}, "dispersive"),
    ("open", {"Au": None}, "dispersive"),
])
def test_each_run_is_timed_by_its_own_or_a_slower_measured_rate(
        boundary, metals, setup):
    assert fdtd.rate_setup(boundary, metals) == setup
    # an unmeasured setup must never be timed faster than what was measured
    rates = fdtd.JETSON_RATES
    assert rates["dispersive"] < rates["open"] < rates["metal"]


def test_the_estimate_counts_cells_bytes_and_seconds_from_its_parts():
    nodes = _graded_nodes()
    metals = {"Au": _gold(nodes)}
    pec = np.zeros(tuple(len(n) for n in nodes), dtype=bool)
    e = fdtd.estimate_run(nodes, duration=3e-15, boundary="open",
                          metals=metals, pec=pec, record_peak=True)
    assert e["shape"] == (14, 12, 10)
    assert e["cells"] == 14 * 12 * 10
    assert e["bytes"] == fdtd.memory_estimate(
        e["shape"], boundary="open", metals=list(metals.items()), pec=True,
        record_peak=True)
    assert e["setup"] == "dispersive"
    assert e["seconds"] == pytest.approx(
        e["cells"] * e["steps"] / (0.9 * 45.30e6))


def test_each_rate_is_the_slowest_measured_row_over_both_jetson_runs():
    # the literal minimums of the two cited runs (row 4a5da177): a faster row,
    # one run dropped, or a typo in the table moves at least one of these
    assert fdtd.JETSON_RATES == pytest.approx(
        {"metal": 106.52e6, "open": 55.11e6, "dispersive": 45.30e6})
    assert set(fdtd.JETSON_RATE_RUNS) == {"6ffa8bd", "836c4b8"}


def test_the_time_estimate_errs_long_by_the_safety_factor():
    assert fdtd.RATE_SAFETY == 0.9
    nodes = _graded_nodes()
    e = fdtd.estimate_run(nodes, duration=3e-15, boundary="metal")
    assert e["seconds"] == pytest.approx(
        e["cells"] * e["steps"] / (0.9 * 106.52e6))


def _estimate(**over):
    e = {"cells": 1_000_000, "steps": 1_000, "bytes": 1e8, "seconds": 60.0,
         "dt": 9.6e-18, "finest": 5e-9}
    e.update(over)
    return e


def _budget(**over):
    """A device budget as ``fdtd.admission_budget`` returns one: the ceiling,
    the figure it came from and where that figure was read."""
    b = {"bytes": 6e9, "free": 7e9, "capacity": 8e9, "source": "gpu",
         "margin": 0.9, "where": "the GPU's free memory at admission (7.8 GB)"}
    b.update(over)
    return b


LIMITS = [("cells", 50_000_000, "Increase dx", "50,000,000", {"max_cells": 50_000_000}),
          ("steps", fdtd.MAX_STEPS, "Shorten the duration", "20,000", {}),
          ("bytes", 14e9, "metal walls", "14.0 GB", {"max_bytes": 14e9}),
          ("seconds", fdtd.MAX_SECONDS, "Shorten the duration", "30 minutes", {})]


@pytest.mark.parametrize("key,limit,lever,shown,ceilings", LIMITS)
def test_each_limit_admits_its_value_and_refuses_one_over(key, limit, lever,
                                                          shown, ceilings):
    fdtd.check_limits(_estimate(**{key: limit}), **ceilings)
    over = limit + 1 if isinstance(limit, int) else limit * 1.001
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(**{key: over}), **ceilings)
    (name, value, got_limit, text), = info.value.exceeded
    assert value == over and got_limit == limit
    assert lever in text and shown in text and str(info.value) == text


def test_the_two_ceilings_a_bench_may_raise_default_to_the_products_own():
    """The step and time ceilings are arguments, and their default is the cap.

    The paper-1 bench script passes them: the owner's 2026-10-16 decision let one
    flagged bench record run past both (39,650 steps and 47 minutes, priced on the
    Jetson). Every other caller -- ``worker/td_solve.py``, the web app -- passes
    nothing, which is what this pins: the default IS ``MAX_STEPS`` /
    ``MAX_SECONDS``, read at call time, so a caller that says nothing meets the
    product's ceiling and a test that moves the constant moves the limit with it.
    """
    over = _estimate(steps=fdtd.MAX_STEPS + 1, seconds=fdtd.MAX_SECONDS + 1.0)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(over)
    assert [name for name, *_ in info.value.exceeded] == ["steps", "time"]

    # The same estimate with the two ceilings this test raises, to exactly what it needs:
    # nothing is left to refuse.
    raised = {"max_steps": over["steps"], "max_seconds": over["seconds"]}
    fdtd.check_limits(over, **raised)

    # One step and one second over the raised pair, and the limit each refusal
    # reports is the raised one -- not the product's own, which the log line would
    # then name instead of the ceiling that actually applied.
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(steps=over["steps"] + 1,
                                    seconds=over["seconds"] + 1.0), **raised)
    reported = [(name, got) for name, _, got, _ in info.value.exceeded]
    assert reported == [("steps", over["steps"]), ("time", over["seconds"])]


def _fake_gpu(free, total, pooled=0):
    """cupy's surface as ``admission_budget`` reads it: ``Device().mem_info``
    and the default pool's free blocks."""
    class Device:
        def __new__(cls):
            return type("D", (), {"mem_info": (free, total)})()
    class Pool:
        def free_bytes(self):
            return pooled
    return type("XP", (), {"cuda": type("C", (), {"Device": Device}),
                           "get_default_memory_pool": staticmethod(lambda: Pool())})


def _meminfo(tmp_path, available_bytes):
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: 999999999 kB\n"
                    f"MemAvailable: {int(available_bytes) // 1024} kB\n")
    return path


def _cgroup(tmp_path, text):
    path = tmp_path / "memory.max"
    path.write_text(text)
    return path


def test_the_gpu_free_memory_is_the_figure_and_the_pool_counts(tmp_path):
    """The device the run holds is the GPU's, so its own free memory is the
    figure -- plus the blocks cupy's pool already holds, which the driver calls
    used and this process will not allocate again."""
    budget = fdtd.admission_budget(
        _fake_gpu(free=10e9, total=16e9, pooled=1.5e9),
        cgroup_paths=(str(_cgroup(tmp_path, "max")),),
        meminfo=str(_meminfo(tmp_path, 40e9)))
    assert budget["source"] == "gpu"
    assert budget["free"] == 11.5e9 and budget["capacity"] == 16e9
    assert budget["bytes"] == int(11.5e9 * 0.9)
    assert "GPU's memory free at admission" in budget["where"]


def test_the_a4000_that_refused_the_record_now_admits_it(tmp_path):
    """The measured case: 16 GB of card, 14.5 GB free, nothing pooled. The
    paper-1 dx-5 nm record's 11.55 GB has to pass, and 13.1 GB -- over the
    ceiling -- has to be refused with the ceiling, its source and the margin."""
    budget = fdtd.admission_budget(
        _fake_gpu(free=14.5e9, total=16e9),
        cgroup_paths=(str(_cgroup(tmp_path, "max")),),
        meminfo=str(_meminfo(tmp_path, 60e9)))
    assert budget["bytes"] == int(14.5e9 * 0.9)
    fdtd.check_limits(_estimate(bytes=11.55e9), budget=budget)   # the record
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(bytes=13.2e9), budget=budget)
    (name, value, got_limit, text), = info.value.exceeded
    assert (name, value, got_limit) == ("memory", 13.2e9, budget["bytes"])
    for part in ["13.20 GB", f"{budget['bytes'] / 1e9:.1f} GB",
                 "GPU's memory free at admission", "0.9 margin",
                 "metal walls"]:
        assert part in text, part


def test_the_container_limit_binds_over_the_hosts_own_memory(tmp_path):
    """The Jetson's host memory is not the container's: a 14 GiB cgroup limit
    is what the run may hold, so it is the capacity, and the available figure
    is what the ceiling is taken from."""
    budget = fdtd.admission_budget(
        None, cgroup_paths=(str(_cgroup(tmp_path, str(14 * 2 ** 30))),),
        meminfo=str(_meminfo(tmp_path, 25e9)))
    assert budget["source"] == "cgroup"
    assert budget["capacity"] == 14 * 2 ** 30
    assert budget["free"] == 24_999_999_488            # 25e9 floored to kB
    assert budget["bytes"] == int(min(14 * 2 ** 30, budget["free"]) * 0.9)
    assert "container's memory limit" in budget["where"]
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(bytes=14e9), budget=budget)
    text = str(info.value)
    for part in ["14.00 GB", "13.5 GB", "container's memory limit",
                 "0.9 margin"]:
        assert part in text, part


def test_no_limit_at_all_falls_back_to_the_systems_available_memory(tmp_path):
    for text in ("max", str(1 << 63)):              # v2 "unlimited", v1's huge
        budget = fdtd.admission_budget(
            None, cgroup_paths=(str(_cgroup(tmp_path, text)),),
            meminfo=str(_meminfo(tmp_path, 20e9)))
        assert budget["source"] == "system" and budget["capacity"] is None
        assert budget["bytes"] == int(20e9 * 0.9)


def test_the_capacity_budget_is_the_devices_size_not_its_free_memory(tmp_path):
    """``free=False`` is the worker's budget: a merely busy device must defer a
    run, not refuse one it could hold, so the ceiling there is the capacity."""
    budget = fdtd.admission_budget(
        _fake_gpu(free=2e9, total=16e9), free=False,
        cgroup_paths=(str(_cgroup(tmp_path, "max")),),
        meminfo=str(_meminfo(tmp_path, 60e9)))
    assert budget["bytes"] == int(16e9 * 0.9) and budget["free"] == 2e9
    assert "GPU's own memory" in budget["where"]


def test_a_full_container_names_its_free_memory_not_its_limit(tmp_path):
    """The limit is 15.0 GB and 4 GB is free: the ceiling comes from the 4 GB,
    so the refusal must say so -- naming the limit would name a figure the
    ceiling was never taken from."""
    budget = fdtd.admission_budget(
        None, cgroup_paths=(str(_cgroup(tmp_path, str(15_000_000_000))),),
        meminfo=str(_meminfo(tmp_path, 4e9)))
    assert budget["bytes"] == int(budget["free"] * 0.9) < 4e9
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(bytes=8e9), budget=budget)
    text = str(info.value)
    for part in ["8.00 GB", f"{budget['bytes'] / 1e9:.1f} GB",
                 f"{budget['free'] / 1e9:.1f} GB of a 15.0 GB limit"]:
        assert part in text, part


def test_the_cell_ceiling_follows_the_budget_and_fires_on_a_split_estimate():
    """With no explicit ``max_cells`` the ceiling is what the byte ceiling buys
    at the cheapest per-cell cost, so a run whose bytes fit always fits in
    cells. It still fires -- on an estimate that prices fewer bytes than its own
    cell count can hold, which is the estimator disagreement it exists to catch.
    """
    budget = _budget(bytes=6e9)
    cells = 6e9 // fdtd.bytes_per_cell()
    fdtd.check_limits(_estimate(cells=int(cells) - 1, bytes=0), budget=budget)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(cells=int(cells) + 1, bytes=0), budget=budget)
    (name, value, got_limit, text), = info.value.exceeded
    assert name == "cells" and value == int(cells) + 1
    assert got_limit == int(cells) and f"{int(cells):,}" in text
    assert "bytes a cell" in text


def test_the_budget_is_read_when_the_check_runs(monkeypatch):
    """``budget=None`` calls ``admission_budget()`` at check time, not an import
    time copy: a caller that says nothing is measured against the device as it
    is now, which is the only figure that can be right."""
    e = _estimate(bytes=11.55e9)
    monkeypatch.setattr(fdtd, "admission_budget", lambda *a, **k: _budget(bytes=20e9))
    fdtd.check_limits(e)                             # 11.55 < 20: admitted

    monkeypatch.setattr(fdtd, "admission_budget", lambda *a, **k: _budget(bytes=5e9))
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(e)                         # 11.55 > 5: refused
    (name, value, got_limit, _), = info.value.exceeded
    assert (name, value, got_limit) == ("memory", 11.55e9, 5e9)


def test_a_caller_off_the_device_checks_only_the_products_own_two():
    """The web app runs on the control plane; its memory is not the run's. With
    ``NO_DEVICE`` the memory and cell ceilings are not checked at all -- 11.55 GB
    of estimate passes on a machine that could never hold it -- while the two
    ceilings that describe the run itself still apply."""
    fdtd.check_limits(_estimate(bytes=11.55e9, cells=14_048_300),
                      budget=fdtd.NO_DEVICE)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(steps=fdtd.MAX_STEPS + 1),
                          budget=fdtd.NO_DEVICE)
    assert [name for name, *_ in info.value.exceeded] == ["steps"]


def test_a_memory_refusal_offers_the_frames_lever():
    """Past MAX_FRAMES the refusal adds the one lever that lowers the
    estimate without touching the grid: record fewer movie frames."""
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(bytes=4e9, frames=fdtd.MAX_FRAMES + 60),
                          max_bytes=3e9)
    assert f"fewer than the {fdtd.MAX_FRAMES + 60:,} movie frames" in str(info.value)


def test_the_refusal_names_the_value_that_broke_each_limit():
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(cells=9_261_000, steps=24_000,
                                    bytes=3.4e9, seconds=2400.0),
                          max_cells=8_000_000, max_bytes=3e9)
    names = [name for name, *_ in info.value.exceeded]
    assert names == ["cells", "steps", "memory", "time"]
    text = str(info.value)
    for part in ["9,261,000 cells", "24,000 time steps", "3.40 GB",
                 "40 minutes", "finest cell, 5 nm"]:
        assert part in text, part


def test_a_real_oversized_grid_is_refused_on_cells_alone():
    nodes = [np.linspace(0, 2e-6, 201)] * 3          # 201^3 = 8.12 M cells
    e = fdtd.estimate_run(nodes, duration=1e-15)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(e, max_cells=8_000_000)
    assert [name for name, *_ in info.value.exceeded] == ["cells"]
    assert "8,120,601 cells" in str(info.value)
