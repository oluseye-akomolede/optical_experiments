"""fdtd.memory_estimate against the solver's measured peak (numpy, tiny).

The estimate feeds the 3.2 refusal at 3.0 GB, so it must track what simulate
really allocates: each setup is measured with tracemalloc and compared, and the
metal term is checked as an INCREMENT (gold run minus the same run without it)
so its share cannot hide inside the larger open-wall total.
"""
import os
import sys
import tracemalloc

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "dockerfiles", "maxwell_solver"))
import fdtd  # noqa: E402

SHAPE = (22, 24, 26)                 # non-square; > 2 x the 10-cell CPML + 1


def _gold_region():
    region = np.zeros(SHAPE, dtype=bool)
    region[6:16, 7:17, 8:19] = True
    return region


def _measured(boundary, metals=(), **options):
    # float64, so simulate's asarray(eps, float32) copies it as cupy's
    # host-to-device asarray always does; a float32 array would be aliased
    # and tracemalloc would never count it (the gap the reviewer found)
    eps = np.ones(SHAPE, dtype=np.float64)
    source = fdtd.Source((11, 12, 11), frequency=4e14, bandwidth=2e14)
    terms = [fdtd.LorentzDrude(name, region) for name, region in metals]
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fdtd.simulate(eps, source, 6, [(11, 12, 13, "z")], dx=5e-9,
                      boundary=boundary, ade_terms=terms, **options)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("boundary", ["metal", "open",
                                      ("periodic", "periodic", "open")])
def test_the_estimate_tracks_the_measured_peak(boundary):
    measured = _measured(boundary)
    predicted = fdtd.memory_estimate(SHAPE, boundary=boundary)
    # numpy keeps no pool, so the host peak sits under the device estimate
    # by the pool share (host 0.93 metal, 0.85 open at this size); above
    # 1.0 the estimate would under-predict
    assert 0.8 <= measured / predicted <= 1.0, (
        f"{boundary}: measured {measured} B, formula {predicted} B")


def test_the_metal_increment_is_counted_on_its_own():
    metals = [("Au", _gold_region())]
    added = _measured("open", metals) - _measured("open")
    predicted = (fdtd.memory_estimate(SHAPE, boundary="open", metals=metals)
                 - fdtd.memory_estimate(SHAPE, boundary="open"))
    assert 0.9 <= added / predicted <= 1.1, (
        f"gold adds {added} B measured, {predicted} B by the formula")


def test_the_options_add_what_they_allocate():
    pec = np.zeros(SHAPE, dtype=bool)
    pec[2:5, 3:6, 4:8] = True
    # uint8, so simulate's asarray(pec, dtype=bool) copies it as cupy's
    # host-to-device asarray always does; a bool array would be reused.
    pec = pec.astype(np.uint8)
    base = _measured("metal")
    for options, flags in (({"pec": pec}, {"pec": True}),
                           ({"record_peak": True}, {"record_peak": True})):
        added = _measured("metal", **options) - base
        predicted = (fdtd.memory_estimate(SHAPE, **flags)
                     - fdtd.memory_estimate(SHAPE))
        assert 0.85 <= added / predicted <= 1.15, (
            f"{flags}: measured {added} B more, formula {predicted} B")
    with pytest.raises(fdtd.FDTDError):
        fdtd.memory_estimate(SHAPE, boundary="wavy")


#: The Jetson rate table (scripts/fdtd_rate.py at 6ffa8bd, cupy, 300 steps,
#: dx 5 nm, lead's run 2026-10-02 01:49Z): (setup, n, peak bytes). These are
#: the device's numbers, not the formula's, so a formula that under-predicts
#: the device (the old one did by 9-14%) fails here.
JETSON = (("metal", 100, 56.4e6), ("metal", 160, 232.0e6),
          ("open", 100, 110.6e6), ("open", 160, 456.7e6),
          ("gold", 100, 134.6e6), ("gold", 160, 554.2e6))


@pytest.mark.parametrize("kind,n,peak", JETSON)
def test_the_estimate_covers_the_jetson_peak(kind, n, peak):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fdtd_rate", os.path.join(os.path.dirname(__file__), "..",
                                  "scripts", "fdtd_rate.py"))
    rate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rate)
    shape, boundary, metals = rate.setup(kind, n)
    predicted = fdtd.memory_estimate(shape, boundary=boundary, metals=metals)
    # never under the device (the refusal must not admit a run that does not
    # fit), and not so far over that it refuses runs that do
    assert 0.9 <= peak / predicted <= 1.0, (
        f"{kind} n={n}: device {peak / 1e6:.1f} MB, formula "
        f"{predicted / 1e6:.1f} MB, ratio {peak / predicted:.3f}")


@pytest.mark.parametrize("shape", [SHAPE, (70, 24, 130)])
def test_the_field_movie_adds_what_it_allocates(shape):
    # the second shape is strided (70 -> 35, 130 -> 44), so the estimate's
    # sampled size is checked, not only the full-grid case
    eps = np.ones(shape, dtype=np.float64)
    source = fdtd.Source((11, 12, 11), frequency=4e14, bandwidth=2e14)

    def measured(**options):
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            fdtd.simulate(eps, source, 60, [(11, 12, 13, "z")], dx=5e-9,
                          **options)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    added = measured(frames=fdtd.frame_plan(shape, 60)) - measured()
    predicted = (fdtd.memory_estimate(shape, record_frames=True)
                 - fdtd.memory_estimate(shape))
    assert 0.85 <= added / predicted <= 1.0, (
        f"frames add {added} B measured, {predicted} B by the formula")


def test_the_estimate_grows_with_the_frame_count_it_is_given():
    """The frame count the request asks for is what the estimate costs, so a
    120-frame movie is refused where a 60-frame one fits. One frame is 4 bytes
    a sample: the difference between 120 and 60 frames is 60 frames' worth."""
    one = fdtd.memory_estimate(SHAPE, record_frames=True, max_frames=1)
    two = fdtd.memory_estimate(SHAPE, record_frames=True, max_frames=2)
    per_frame = two - one
    assert per_frame > 0, "a frame must cost something"
    sixty = fdtd.memory_estimate(SHAPE, record_frames=True, max_frames=60)
    many = fdtd.memory_estimate(SHAPE, record_frames=True, max_frames=120)
    assert many > sixty
    assert many - sixty == pytest.approx(60 * per_frame)
    # the default is 60, spelled out or not
    assert sixty == fdtd.memory_estimate(SHAPE, record_frames=True)
    # and with no movie at all the count changes nothing
    assert (fdtd.memory_estimate(SHAPE, max_frames=120)
            == fdtd.memory_estimate(SHAPE))


def test_a_120_frame_movie_allocates_what_the_formula_says_over_60():
    shape = (22, 24, 26)
    eps = np.ones(shape, dtype=np.float64)
    source = fdtd.Source((11, 12, 11), frequency=4e14, bandwidth=2e14)

    def measured(n):
        # a run longer than 60 steps, so the plan really is n frames long
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            fdtd.simulate(eps, source, 130, [(11, 12, 13, "z")], dx=5e-9,
                          frames=fdtd.frame_plan(shape, 130, max_frames=n))
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    frames_120, frames_60 = 120, 60
    assert len(fdtd.frame_plan(shape, 130, max_frames=frames_120)["steps"]) == 120
    assert len(fdtd.frame_plan(shape, 130, max_frames=frames_60)["steps"]) == 60
    added = measured(frames_120) - measured(frames_60)
    predicted = (fdtd.memory_estimate(shape, record_frames=True, max_frames=frames_120)
                 - fdtd.memory_estimate(shape, record_frames=True, max_frames=frames_60))
    # a few per cent over is the per-frame temporaries tracemalloc also sees;
    # the point is that the formula's increment is the frames' own bytes, not
    # that it is an exact account of every temporary
    assert 0.85 <= added / predicted <= 1.05, (
        f"60 more frames add {added} B measured, {predicted} B by the formula")
