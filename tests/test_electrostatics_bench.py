"""The bench fixtures and the bench's own independence from what it scores.

The bench is a script, but it is the only thing that pins eps_s against a
closed form, so it gets tests: a bench that silently stops discriminating is
worse than no bench, because it looks like coverage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SOLVER = Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"
sys.path.insert(0, str(SOLVER))

import electrostatics_bench as bench  # noqa: E402
import mse_bench_fixtures as F  # noqa: E402
import mse_electrostatics as E  # noqa: E402


def test_bench_passes_as_shipped():
    assert bench.main([]) == 0


def test_bench_fails_on_wrong_eps_reaching_the_solver(monkeypatch):
    """A 1% error in a dielectric must FAIL the bench.

    This is the mutation the row asked for. It is written as a test, not as a
    one-off manual mutation, because the first version of the bench could not
    detect it at all: the reference was derived from the same eps that went
    into the solve, so both sides scaled together and the relative error did
    not move. The guard in ``_fixture_with_trusted_eps`` is the tripwire; this
    test bypasses it to prove the SCORE also catches the error, so the two are
    not the same mechanism.
    """
    real = F.concentric_spheres

    def wrong_eps(**kw):
        problem, expected = real(**kw)
        problem["eps_static"]["shell_a"] *= 1.01
        return problem, expected

    monkeypatch.setattr(bench, "concentric_spheres", wrong_eps)
    monkeypatch.setattr(bench, "_fixture_with_trusted_eps",
                        lambda b, a, c, **kw: b(**kw))
    # main() RETURNS its exit status; it does not raise. An earlier version of
    # this test asserted SystemExit and so passed for the wrong reason -- the
    # failure it "detected" was its own bad assertion, not the bench's.
    assert bench.main([]) == 1


def test_trusted_eps_guard_refuses_a_fixture_that_rewrote_its_table(monkeypatch):
    real = F.concentric_spheres

    def wrong_eps(**kw):
        problem, expected = real(**kw)
        problem["eps_static"]["shell_a"] *= 1.01
        return problem, expected

    with pytest.raises(AssertionError, match="did not receive the trusted"):
        bench._fixture_with_trusted_eps(wrong_eps, 2.0, 5.0)
    assert real is F.concentric_spheres


def test_slab_matches_its_closed_form_exactly():
    problem, expected = F.two_layer_slab()
    C = bench._capacitance(problem)
    assert abs(C - expected["C"]) / expected["C"] < 1e-9


def test_shells_match_the_meshed_series():
    problem, expected = bench._fixture_with_trusted_eps(
        F.concentric_spheres, bench.EPS_A_BENCH, bench.EPS_B_BENCH,
        n_theta=bench.SPHERE_N_THETA, n_phi=bench.SPHERE_N_PHI,
    )
    C = bench._capacitance(problem)
    assert abs(C - expected["C_meshed"]) / expected["C_meshed"] < 2e-3


def test_sphere_fixture_is_a_valid_mesh():
    """The sphere has no closed form, so this checks the MESH, not the answer.

    It exists because three fixture defects showed up only here: a duplicated
    z-plane (which surfaced as a bare LinAlgError), a non-reciprocal C, and a
    solution outside the driven range.
    """
    p = F.sphere_between_plates()
    assert len(np.unique(p["points_m"].round(15), axis=0)) > 0.9 * len(p["points_m"])
    C = bench._capacitance_matrix_only(p)
    assert np.abs(C - C.T).max() < 1e-12 * np.abs(C).max()
    phi = bench._solution(p)["phi"]
    assert phi.min() >= -1e-9 and phi.max() <= 1 + 1e-9


def test_no_duplicate_node_planes_in_any_fixture():
    """A duplicated plane makes every tet there flat; tet_geometry inverts J."""
    for problem in (F.two_layer_slab()[0], F.sphere_between_plates()):
        pts = problem["points_m"]
        assert len(np.unique(pts.round(15), axis=0)) > 0.9 * len(pts)
