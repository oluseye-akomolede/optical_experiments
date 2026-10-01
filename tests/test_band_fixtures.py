"""Every JS band fixture must be a diagram the SERVER would actually send.

The two JS readers of ``pcs.bands`` (``web/static/bands.js``, ``web/static/
bandplot.js``) shipped transposed -- reading ``bands[band][k]`` while the wire
is ``bands[k][band]`` -- and their tests passed, because each fixture was
written in the reader's own orientation. A test whose input is authored by the
thing under test agrees with itself and can be wrong about the world.

So the fixtures live in one file, ``tests/js/fixtures/band_diagrams.json``, read
by both the JS tests and this one. Here they are run through the server's own
validator, ``web/worker_api.py::_band_payload``, which is the function that
decides what a pc ``/done`` body may store -- the same function that accepted
the real Jetson diagram. A fixture it would reject is a fixture the readers can
never be tested on in production, and this test says so at tier 1 instead.

The second property checked here is NON-SQUARE-ness: ``len(k_points)`` must
differ from the band count. A square diagram reads the same either way, which
is precisely how a transposition hides -- the pre-fix fixtures were 2x2 and
3x3. Every fixture is 3 k-points by 2 bands (or 1 band), so the band count and
the k-point count are different numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import worker_api as API  # noqa: E402

FIXTURE_PATH = ROOT / "tests" / "js" / "fixtures" / "band_diagrams.json"


def _fixtures():
    with FIXTURE_PATH.open() as handle:
        data = json.load(handle)
    return {name: value for name, value in data.items() if name != "_about"}


def test_the_fixture_file_has_the_diagrams_the_js_tests_name():
    """A guard on the file itself: an emptied fixture silently passes below.

    A parametrised test over an empty list collects zero cases, which pytest
    reports as success. The JS tests would then fail by name, but this file
    would look green while checking nothing, so the count is pinned.
    """
    names = set(_fixtures())
    assert len(names) >= 6, f"only {len(names)} band fixtures: {sorted(names)}"
    for required in ("jetson_tm_slice", "gap_between_1_and_2", "single_band"):
        assert required in names, f"the JS tests name {required}; it is gone"


@pytest.mark.parametrize("name", sorted(_fixtures()))
def test_every_band_fixture_is_a_diagram_the_server_would_store(name):
    """``_band_payload`` accepts it, so no reader is tested on an impossible shape.

    ``_band_payload`` is handed the same body the worker posts: ``{"bands":
    <diagram>}``. A ``None`` return is the server's refusal, and it is exactly
    what the pre-fix ``test_bandplot.mjs`` fixture earned (three k-points, two
    band rows) -- the renderer test was green on a body ``/done`` answers 400.
    """
    diagram = _fixtures()[name]
    got = API._band_payload({"bands": diagram})
    assert got is not None, (
        f"the fixture {name!r} is not a shape the server accepts; the JS reader "
        f"tests would be testing against a payload no job ever produces"
    )
    assert got["bands"] == diagram["bands"], (
        "the validator changed the fixture's numbers")


@pytest.mark.parametrize("name", sorted(_fixtures()))
def test_every_band_fixture_is_non_square(name):
    """The k-point count and the band count must differ.

    Square is the trap: with ``len(k_points) == len(bands[0])`` a reader that
    swaps the axes produces the same shapes, so both orientations pass and the
    transposition survives. This is asserted separately from the validation
    above because the server is content with a square diagram -- only the test
    suite has a reason to refuse one.
    """
    diagram = _fixtures()[name]
    n_k = len(diagram["k_points"])
    n_b = len(diagram["bands"][0])
    assert n_k != n_b, (
        f"{name!r} is {n_k}x{n_b}: a square fixture reads the same transposed, "
        f"so it cannot detect an orientation bug")
    assert len(diagram["labels"]) == n_k, (
        f"{name!r} has {len(diagram['labels'])} labels for {n_k} k-points; the "
        f"axis places one tick per k-point")


def test_the_jetson_slice_fixture_carries_the_measured_extremes():
    """The one fixture that is not invented: the real bench diagram's gap.

    Band 1 tops at 0.281 at M and band 2 bottoms at 0.421 at X on the real
    30-point TM diagram measured on the Jetson (2026-10-01). Pinned here so a
    later edit cannot soften it into a fixture that would pass transposed --
    the JS test asserts a gap is REPORTED, and this asserts the numbers that
    make that gap real.
    """
    d = _fixtures()["jetson_tm_slice"]
    bands = d["bands"]
    band1 = [row[0] for row in bands]
    band2 = [row[1] for row in bands]
    assert max(band1) == pytest.approx(0.281)
    assert band2[1] == pytest.approx(0.421)
    assert min(band2) == pytest.approx(0.421)
    assert max(band1) < min(band2), "the fixture must carry a real gap"
