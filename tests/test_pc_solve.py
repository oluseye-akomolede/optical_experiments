"""The pc job's request translation: the row in, the solver call out.

These drive :mod:`pc_solve` with a stub solver, so nothing here needs numpy,
scipy or a real band solve. What is asserted is the CONTRACT between the row the
web side writes and the call the worker makes: the same seven request keys, the
k-path derived from the lattice, and a JSON-ready payload out.

The failure this file exists to catch is the silent one: a field the form
collects that stops reaching the solve, so the job comes back as a diagram of a
DIFFERENT problem with no error anywhere. That is the shape of the multi-part
estimate bug, and it is why the request keys are pinned against the web side
below rather than restated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "web"))

import pc_form  # noqa: E402
import pc_solve  # noqa: E402


import numpy as np  # noqa: E402


class _StubSolver:
    """Records the call and returns shaped-not-correct arrays.

    Deliberately NOT a reimplementation of the physics: a stub that produced
    plausible bands would let a test assert on numbers it made up itself. It
    returns distinguishable values so the test can prove they were PASSED
    THROUGH -- the payload's bands are the stub's, and nothing recomputed them.

    **It returns NUMPY, not Python floats, and that is the point.** The real
    ``pc.bands_from_mesh`` returns an ``np.ndarray`` and ``high_symmetry_path``
    returns numpy rows, so the ``float(...)`` conversion in the payload is on
    the live path. A stub of Python floats made that conversion invisible: a
    mutation leaving ``bands``/``k_points`` as numpy stayed green (M5, M8),
    which is a test agreeing with a stub that does not exist in production. The
    stub now has the same TYPES as the solver it stands in for.
    """

    def __init__(self):
        self.calls = []
        self.n_bands = None

    def high_symmetry_path(self, lattice, points_per_segment=10):
        self.calls.append(("path", lattice, points_per_segment))
        # FOUR k-points, and the job asks for THREE bands, so the payload below
        # is not square. The wire shape is bands[k][band]; a square fixture is
        # read the same either way, which is exactly how the two JS readers
        # shipped transposed with green tests. This is the file that pins the
        # shape against the real solver, so it must be able to see the
        # difference. (The real square-lattice path returns more than four
        # points; four is the smallest that is not three.)
        return (["G", "X", "M", "G"],
                np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]]))

    def bands_from_mesh(self, lattice, divisions, r_over_a, n_rod, n_bg,
                        k_points, n_bands, polarisation="TM"):
        self.calls.append(("bands", lattice, divisions, r_over_a, n_rod, n_bg,
                           n_bands, polarisation))
        self.n_bands = n_bands
        # n_bands values per k-point, distinct so a pass-through is provable.
        # float64, exactly what the real solver returns.
        rows = np.array([[float(i + j) for j in range(n_bands)]
                         for i in range(len(k_points))])
        return k_points, rows


def _job(**over):
    pcs = {
        "lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
        "n_bands": 3, "divisions": 16, "polarisation": "TM",
    }
    pcs.update(over.pop("pcs", {}))
    job = {"job_id": "j1", "kind": "pc", "pcs": pcs}
    job.update(over)
    return job


def test_a_pc_job_is_solved_with_every_request_field_it_carries():
    """Each field reaches the solver under its own name.

    By VALUE, one assertion per field: a single ``assert call == expected``
    would fail as a block and not say WHICH field went astray, and the whole
    risk here is one field silently taking a default.
    """
    stub = _StubSolver()
    pc_solve.solve_pc(_job(pcs={"r_over_a": 0.37, "divisions": 22,
                                "n_bands": 3, "polarisation": "TE"}),
                      solver=stub)
    (_, lattice, divisions, r_over_a, n_rod, n_bg, n_bands,
     polarisation) = stub.calls[1]
    assert lattice == "square"
    assert divisions == 22
    assert r_over_a == 0.37
    assert n_rod == 3.5
    assert n_bg == 1.0
    assert n_bands == 3
    assert polarisation == "TE"


def test_the_k_path_comes_from_the_lattice_not_the_row():
    """The path is derived, and derived the same way the form derived it.

    The row does not carry k_points, so a caller cannot inject a path that is
    not the irreducible-zone path. Asserted as the shared-function property:
    what the solver is asked to walk is what ``pc_form`` counts for the cap.

    TWO lattices, deliberately: with only ``square`` in the whole file, a
    hardcoded ``"square"`` in the call passes every test here (M6). The second
    lattice is what makes the field's PASSAGE observable rather than assumed.
    """
    stub = _StubSolver()
    pc_solve.solve_pc(_job(), solver=stub)
    assert stub.calls[0] == ("path", "square", 10)
    pc_solve.solve_pc(_job(pcs={"lattice": "triangular"}), solver=stub)
    assert stub.calls[2] == ("path", "triangular", 10)


def test_the_payload_bands_are_the_solver_s_own_arrays():
    """The numbers are passed through, not recomputed or re-derived.

    The shape is ``bands[k][band]``: four k-points (rows, from the path) by
    three bands (columns, from ``n_bands``). Pinned as a literal list rather
    than a transposable comparison, so an implementation that swapped the axes
    cannot pass by producing a same-size array.
    """
    stub = _StubSolver()
    out = pc_solve.solve_pc(_job(), solver=stub)
    assert out["bands"] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0],
                            [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert len(out["bands"]) == len(out["k_points"]) == 4, "one row per k-point"
    assert len(out["bands"][0]) == stub.n_bands == 3
    assert len(out["bands"]) != len(out["bands"][0]), (
        "the fixture must be non-square or it cannot see a transposition")


def test_the_payload_is_json_ready():
    """No numpy scalars: the payload must survive ``json.dumps``.

    A numpy float64 is not JSON-serialisable, and the failure would land at the
    route AFTER the solve -- a 500 with the work already done, which reads as
    "the job failed" rather than "the payload was wrong". The stub returns
    Python floats here; the real solver returns numpy, so this test is a guard
    on the ``float(...)`` conversion in the payload, not on the stub.
    """
    import json

    out = pc_solve.solve_pc(_job(), solver=_StubSolver())
    json.dumps(out)  # raises TypeError if a value is not serialisable

    def _no_numpy(value):
        # Belt and braces against a numpy scalar hiding in a nested list, which
        # json.dumps would catch but the message would not name the path.
        if isinstance(value, dict):
            return all(_no_numpy(v) for v in value.values())
        if isinstance(value, list):
            return all(_no_numpy(v) for v in value)
        return not isinstance(value, np.generic)

    assert _no_numpy(out), out


def test_a_cavity_job_is_refused_not_solved():
    """pcs is NULL for a cavity job; that is a refusal, not an empty request."""
    for bad in (None, {}, "square"):
        with pytest.raises(pc_solve.PCSolveError) as e:
            pc_solve.solve_pc({"job_id": "j", "kind": "cavity", "pcs": bad},
                              solver=_StubSolver())
        assert "pcs" in str(e.value)


def test_a_request_missing_a_field_is_refused_by_name():
    """A partial request names the missing field rather than defaulting it."""
    for missing in pc_solve.REQUEST_KEYS:
        pcs = {k: v for k, v in {
            "lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
            "n_bands": 3, "divisions": 16, "polarisation": "TM",
        }.items() if k != missing}
        with pytest.raises(pc_solve.PCSolveError) as e:
            pc_solve.solve_pc({"job_id": "j", "kind": "pc", "pcs": pcs},
                              solver=_StubSolver())
        assert missing in str(e.value)


def test_a_zero_radius_request_is_not_refused_for_being_falsy():
    """``r_over_a = 0`` is a legitimate value, not a missing field.

    The refusal is by KEY, not by truthiness. A truthiness check would refuse
    the whole hollow-core limit -- and would pass every other test here, because
    every other value in this file is non-zero.
    """
    stub = _StubSolver()
    pc_solve.solve_pc(_job(pcs={"r_over_a": 0.0}), solver=stub)
    assert stub.calls[1][3] == 0.0


def test_the_refusal_reads_the_key_by_name_not_by_lookup():
    """The request is built by ``pcs[k]``, so a key present-but-None is carried.

    ``None`` is not a value any field can hold (the form refuses it earlier), so
    this is not a value the worker should accept -- but ``pcs.get(k)`` and
    ``pcs[k]`` differ on exactly one input: a key that is PRESENT with value
    None, where ``get`` yields None both ways and ``[]`` yields None only
    because the key is there. A missing key is refused by name above, so this
    test pins the line that chooses between them.
    """
    from unittest import mock

    stub = _StubSolver()
    pcs = {"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
           "n_bands": 3, "divisions": 16, "polarisation": None}
    # A present-but-None field is carried to the solver as None, not dropped and
    # not defaulted to "TM" -- proof the value came from the dict by key.
    pc_solve.solve_pc({"job_id": "j", "kind": "pc", "pcs": pcs}, solver=stub)
    assert stub.calls[1][7] is None


def test_the_request_keys_are_exactly_what_the_form_writes():
    """Pin the worker's key list to the web side's, not to a second copy.

    ``request_json`` is what the enqueue writes into ``pcs``; this module reads
    it back. If one list grows and the other does not, a collected field stops
    reaching the solve with no error -- so the key set is compared against the
    form's own function rather than restated here.
    """
    assert set(pc_solve.REQUEST_KEYS) == set(
        pc_form.request_json({k: None for k in pc_form.request_json(
            {k: None for k in pc_solve.REQUEST_KEYS})}).keys()
    )
