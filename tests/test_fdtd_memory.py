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
    eps = np.ones(SHAPE, dtype=np.float32)
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
    assert 0.88 <= measured / predicted <= 1.05, (
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
