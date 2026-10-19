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


LIMITS = [("cells", fdtd.MAX_CELLS, "Increase dx", "8,000,000"),
          ("steps", fdtd.MAX_STEPS, "Shorten the duration", "20,000"),
          ("bytes", fdtd.MAX_BYTES, "metal walls", "3.0 GB"),
          ("seconds", fdtd.MAX_SECONDS, "Shorten the duration", "30 minutes")]


@pytest.mark.parametrize("key,limit,lever,shown", LIMITS)
def test_each_limit_admits_its_value_and_refuses_one_over(key, limit, lever,
                                                          shown):
    fdtd.check_limits(_estimate(**{key: limit}))
    over = limit + 1 if isinstance(limit, int) else limit * 1.001
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(**{key: over}))
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


def test_a_bench_may_raise_the_cell_ceiling_and_the_default_is_the_products_own():
    """``max_cells`` is the third ceiling, and its default is still 8 M.

    The record this change prices -- the paper's own scene at dx 10 nm with the
    open boundary held where r20's run held it -- is 14,048,300 cells, over the
    product's cap by 75.6%. A caller that says nothing -- ``worker/td_solve.py``,
    the web app -- still meets 8,000,000 exactly: the control here is the
    product's own default, and one cell over it is refused. The ceiling is read
    at call time, so a test that moves the constant moves the limit too.
    """
    e = _estimate(cells=14_048_300)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(e)
    (name, value, got_limit, text), = info.value.exceeded
    assert (name, value, got_limit) == ("cells", 14_048_300, fdtd.MAX_CELLS)
    assert "8,000,000" in text and "Increase dx" in text

    # At exactly what it needs, and above it: both admitted.
    fdtd.check_limits(e, max_cells=14_048_300)
    fdtd.check_limits(e, max_cells=15_000_000)

    # One cell over the raised ceiling, and the refusal names the raised ceiling
    # rather than the product's own -- which the log line would then call in force.
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(cells=14_048_301), max_cells=14_048_300)
    (name, value, got_limit, text), = info.value.exceeded
    assert (name, value, got_limit) == ("cells", 14_048_301, 14_048_300)
    assert "14,048,300" in text and "8,000,000" not in text


def test_a_bench_may_raise_the_memory_ceiling_and_the_default_is_the_products_own():
    """``max_bytes`` is the fourth ceiling, and its default is still 3.0 GB.

    The paper's own scene at dx 5 nm on the 16 GB A4000 estimates 11.55 GB of
    GPU memory, refused after about two hours of setup under the product's
    ceiling. A caller that says nothing -- ``worker/td_solve.py``, the web app --
    still meets 3.0 GB exactly: the control here is the product's own default,
    and one byte over it is refused. The ceiling is read at call time, so a test
    that moves the constant moves the limit too.
    """
    assert fdtd.MAX_BYTES == 3.0e9
    e = _estimate(bytes=11.55e9)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(e)
    (name, value, got_limit, text), = info.value.exceeded
    assert (name, value, got_limit) == ("memory", 11.55e9, fdtd.MAX_BYTES)
    assert "11.55 GB" in text and "3.0 GB" in text

    # At exactly what it needs, and above it: both admitted.
    fdtd.check_limits(e, max_bytes=11.55e9)
    fdtd.check_limits(e, max_bytes=14e9)

    # One byte over the raised ceiling, and the refusal names the raised ceiling
    # rather than the product's own.
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(bytes=14e9 + 1), max_bytes=14e9)
    (name, value, got_limit, text), = info.value.exceeded
    assert (name, value, got_limit) == ("memory", 14e9 + 1, 14e9)
    assert "over the limit of 14.0 GB" in text and "3.0 GB" not in text


def test_the_memory_default_is_the_constant_at_call_time(monkeypatch):
    """The default is ``MAX_BYTES`` itself, not a copy frozen at import.

    A caller that says nothing is measured against whatever the module's own
    ceiling is when the check runs. Moving the constant has to move the limit
    with it -- otherwise this test's own premise (and the docstring's) is a
    claim about a binding nobody exercised.
    """
    e = _estimate(bytes=11.55e9)
    monkeypatch.setattr(fdtd, "MAX_BYTES", 20e9)
    fdtd.check_limits(e)                             # 11.55 < 20: admitted

    monkeypatch.setattr(fdtd, "MAX_BYTES", 5e9)
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(e)                         # 11.55 > 5: refused
    (name, value, got_limit, _), = info.value.exceeded
    assert (name, value, got_limit) == ("memory", 11.55e9, 5e9)


def test_the_refusal_names_the_value_that_broke_each_limit():
    with pytest.raises(fdtd.EstimateError) as info:
        fdtd.check_limits(_estimate(cells=9_261_000, steps=24_000,
                                    bytes=3.4e9, seconds=2400.0))
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
        fdtd.check_limits(e)
    assert [name for name, *_ in info.value.exceeded] == ["cells"]
    assert "8,120,601 cells" in str(info.value)
