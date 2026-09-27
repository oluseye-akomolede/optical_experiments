"""The cpu|gpu solver backend: which one is chosen, and what "gpu" costs.

Tier 1 throughout. The two things that make that possible without a GPU:

* the selection logic is pure environment, so it needs no CUDA at all, and
* the refinement loop is plain numpy once the per-step solve is a stub.

What these tests **cannot** show is that cuDSS gets the right answer. That needs
the Jetson; the note at the bottom of this file says so for whoever reads a green
run as more than it is.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_backend as B  # noqa: E402
import mse_eigen as EIG  # noqa: E402
import mse_fem as F  # noqa: E402

# the box builder the solver's own tests use -- no TetGen, so tier 1
from test_maxwell_solver import _box_matrices, structured_box_mesh  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_backend_env(monkeypatch):
    """Every test starts from the unset environment.

    Without this a stray ``MSE_SOLVER_BACKEND`` in the runner would change what
    the selection tests mean, and the failure would look like a code bug.
    """
    for name in (B.BACKEND_ENV, B.PRECISION_ENV, B.REFINE_STEPS_ENV,
                 B.RESIDUAL_TOL_ENV):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# 1. selection
# --------------------------------------------------------------------------

def test_the_backend_defaults_to_cpu_when_the_variable_is_unset():
    assert B.selected_backend() == B.CPU


@pytest.mark.parametrize("value", ["cpu", "gpu"])
def test_a_known_backend_name_is_returned_unchanged(monkeypatch, value):
    monkeypatch.setenv(B.BACKEND_ENV, value)
    assert B.selected_backend() == value


@pytest.mark.parametrize("value", ["GPU", "gpu2", "", " cpu", "cuda", "yes"])
def test_an_unknown_backend_raises_rather_than_defaulting(monkeypatch, value):
    """A typo must not become a silent CPU run.

    The failure mode this prevents is not a wrong answer, it is a *right* answer
    from the wrong machine: ``MSE_SOLVER_BACKEND=gpu`` misspelled would solve
    correctly on the CPU, take 40 minutes instead of 90 seconds, and report
    nothing wrong. Empty string counts as unknown, not as unset -- it can only
    come from ``MSE_SOLVER_BACKEND=``, which is a mistake.
    """
    monkeypatch.setenv(B.BACKEND_ENV, value)
    with pytest.raises(B.BackendError) as excinfo:
        B.selected_backend()
    message = str(excinfo.value)
    assert value in message, "the message must quote the offending value"
    assert "cpu" in message and "gpu" in message, "and name the legal ones"


# --------------------------------------------------------------------------
# 2. the gpu path without the CUDA stack -- the real path on this machine
# --------------------------------------------------------------------------

def test_gpu_without_the_cuda_stack_gives_a_clear_error_not_a_fallback():
    """On the VM (and in the cluster image) cupy/nvmath are genuinely absent.

    So this is not a contrived case: it is what ``MSE_SOLVER_BACKEND=gpu`` does
    today on every machine except the Jetson. The important half is the negative
    one -- it must **not** quietly solve on the CPU, because that would make "the
    GPU run was slow" a mystery with no error anywhere.
    """
    matrix = EIG.sparse.eye(4, format="csc")

    with pytest.raises(B.BackendError) as excinfo:
        B.factor(matrix, label="K", spd_lu={}, backend=B.GPU)

    message = str(excinfo.value)
    assert "gpu" in message, message
    assert "cpu" in message, "the message must say how to get a working run"
    # name the missing piece, whichever it is, so the operator knows what to install
    assert any(name in message for name in ("cupy", "nvmath")), message


def test_the_gpu_branch_is_never_reached_by_an_import_at_module_load():
    """The GPU packages must be imported lazily, inside the gpu branch.

    If ``mse_backend`` imported cupy at module level, this whole test module --
    and the solver -- would fail to import on the cluster. The assertion is that
    importing the module left the CUDA names out of it.
    """
    assert not hasattr(B, "cp"), "cupy must not be bound at module scope"
    assert not hasattr(B, "cupy"), "cupy must not be bound at module scope"
    assert not hasattr(B, "nvmath"), "nvmath must not be bound at module scope"


# --------------------------------------------------------------------------
# 3. both factorisations go through the factory
# --------------------------------------------------------------------------

def test_both_factorisations_are_requested_by_name():
    """``K`` and ``S``, each with its own label -- not one of them, and not by count.

    This is the assertion for the dispatch's actual requirement. The production
    ask is both matrices on the backend, because at 266k tets S's CPU
    factorisation is 19 s and the projection solves dominate the run; a backend
    that took only K would leave most of the time unimproved.

    The reference implementation decided which matrix was which by *counting*
    calls to ``splu``, which only works because it wrapped one fixed code path.
    Here the caller names it, so this test can assert the names.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    labels = []
    real_factor = EIG.BACKEND.factor

    def recording_factor(matrix, *, label, spd_lu, backend=None):
        labels.append((label, backend))
        return real_factor(matrix, label=label, spd_lu=spd_lu, backend=backend)

    EIG.BACKEND.factor = recording_factor
    try:
        EIG.solve_modes(
            A_free, B_free, 4, points,
            edges=edges, tets=tets, tet_edges=tet_edges,
            free_edges=free, constrained=constrained, verbose=False,
            backend=B.CPU,
        )
    finally:
        EIG.BACKEND.factor = real_factor

    assert [label for label, _ in labels] == ["K", "S"], (
        f"expected K then S through the factory, got {labels}"
    )
    assert all(backend == B.CPU for _, backend in labels), labels


def test_a_fake_gpu_factor_drives_the_projector_end_to_end(monkeypatch):
    """The solve plumbing works with any object that has ``.solve``.

    This is what makes the seam worth having: ``project``/``opinv`` and the
    Lanczos iteration touch only ``.solve``, so a stub proves the wiring without
    a GPU. If the factories ever returned something the closures could not use,
    this fails here rather than on the Jetson.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    class StubFactor:
        """Dense solve -- slow and wrong for a big matrix, exact for this one."""

        def __init__(self, matrix):
            self.dense = np.linalg.inv(matrix.toarray())

        def solve(self, x):
            return self.dense @ np.asarray(x, dtype=np.float64)

    calls = []

    real_factor = EIG.BACKEND.factor

    def stub_factor(matrix, *, label, spd_lu, backend=None):
        calls.append(label)
        return StubFactor(matrix)

    EIG.BACKEND.factor = stub_factor
    try:
        values, _, _ = EIG.solve_modes(
            A_free, B_free, 4, points,
            edges=edges, tets=tets, tet_edges=tet_edges,
            free_edges=free, constrained=constrained, verbose=False,
            backend=B.GPU,
        )
    finally:
        EIG.BACKEND.factor = real_factor

    assert EIG.LAST_BACKEND == B.GPU, "the stub backend must be what was recorded"
    assert calls == ["K", "S"], calls
    assert np.all(np.isfinite(values)), values
    assert np.all(np.asarray(values) > 0), (
        "the projector must remove the gradient null space, so no ~0 eigenvalue"
    )


def test_the_stub_backend_reproduces_the_cpu_answer():
    """The seam is a substitution, not a different solver.

    Same matrices, same code path, dense solves instead of sparse ones: the
    eigenvalues must agree to solver precision. If they do not, the plumbing is
    passing something other than what the CPU path passes.
    """
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    class StubFactor:
        def __init__(self, matrix):
            self.dense = np.linalg.inv(matrix.toarray())

        def solve(self, x):
            return self.dense @ np.asarray(x, dtype=np.float64)

    real_factor = EIG.BACKEND.factor
    EIG.BACKEND.factor = lambda m, *, label, spd_lu, backend=None: StubFactor(m)
    try:
        stub, _, _ = EIG.solve_modes(
            A_free, B_free, 4, points, edges=edges, tets=tets,
            tet_edges=tet_edges, free_edges=free, constrained=constrained,
            verbose=False, backend=B.GPU,
        )
        cpu, _, _ = EIG.solve_modes(
            A_free, B_free, 4, points, edges=edges, tets=tets,
            tet_edges=tet_edges, free_edges=free, constrained=constrained,
            verbose=False, backend=B.CPU,
        )
    finally:
        EIG.BACKEND.factor = real_factor

    assert np.allclose(np.sort(stub), np.sort(cpu), rtol=1e-8), (
        f"stub {np.round(np.sort(stub), 8)}\ncpu  {np.round(np.sort(cpu), 8)}"
    )


# --------------------------------------------------------------------------
# 4. the CPU path is unchanged
# --------------------------------------------------------------------------

def test_the_cpu_backend_is_still_superlu():
    """The default path must be the same factorisation object as before.

    A factory that rerouted cpu would still pass the eigenvalue tests -- the
    answer is the same either way -- so this pins the thing those tests cannot:
    that ``backend=cpu`` means SuperLU, using the SPD arguments it was given.
    """
    matrix = EIG.sparse.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    spd_lu = {"permc_spec": "MMD_AT_PLUS_A", "diag_pivot_thresh": 0.0,
              "options": {"SymmetricMode": True}}

    solved = B.factor(matrix, label="K", spd_lu=spd_lu, backend=B.CPU)

    assert type(solved).__name__ == "SuperLU", type(solved)
    dense = matrix.toarray()
    x = np.array([1.0, 2.0])
    assert np.allclose(dense @ solved.solve(x), x, rtol=1e-12)


def test_an_explicit_layout_is_passed_through_untouched():
    """``spd_ordering=False`` must still reach SuperLU as an empty dict.

    The tests' ability to run both orderings depends on the caller's dict
    arriving unmodified -- a factory that helpfully supplied defaults would
    silently make the two-ordering comparison compare a thing to itself.
    """
    seen = []
    matrix = EIG.sparse.eye(3, format="csc")
    real_splu = B.spla.splu
    B.spla.splu = lambda m, **kwargs: (seen.append(kwargs), real_splu(m, **kwargs))[1]
    try:
        B.factor(matrix, label="K", spd_lu={}, backend=B.CPU)
    finally:
        B.spla.splu = real_splu

    assert seen == [{}], f"the empty layout must arrive empty, got {seen}"


# --------------------------------------------------------------------------
# 5. the refinement loop, and the residual it must not exceed
# --------------------------------------------------------------------------

class _LossyStub:
    """The refinement loop with the GPU removed.

    ``_solve_once`` is deliberately inaccurate -- it scales the exact solution by
    ``(1 - error)`` -- so refinement has something to do and the residual is a
    real number rather than numerical noise. Only ``solve`` is exercised; the
    factorisation never runs.

    The convergence rate is exactly geometric, ``residual = error ** (steps+1)``
    (measured: 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7 for 0, 1, 3, 4, 5, 6 steps at
    ``error=0.1``), so clearing the 1e-6 tolerance takes 6 steps -- at 5 the
    residual is exactly 1e-6, which the check correctly rejects as not *under*
    the limit. ``CONVERGING`` is therefore 6: the arithmetic is the loop's, the
    step count is the stub's, and keeping them separate stops a stub that cannot
    converge from looking like a loop that does not.

    ``MEASURING`` opens the tolerance instead of bypassing the check, so the
    partial results in the monotonicity test can be inspected without the guard
    firing. The guard stays in the path -- a test that switched it off would
    prove nothing about it.
    """

    CONVERGING = 6
    MEASURING = 2.0        # above any residual this stub can produce

    def __init__(self, matrix, error=0.1, steps=3, tol=1e-6):
        self.A = matrix.tocsr()
        self.exact = np.linalg.inv(self.A.toarray())
        self.error = error
        self.single = True
        self.refine = steps
        self.tol = tol
        self.worst_residual = 0.0
        self.label = "K"
    def _solve_once(self, b):
        return (1.0 - self.error) * (self.exact @ np.asarray(b, dtype=np.float64))

    solve = B._CudssFactor.solve          # the production loop, unchanged

    # Bound explicitly rather than aliased from the class: reaching a plain
    # function through an instance binds `self`, so calling it with (A, x, b)
    # would shift every argument by one. This is a `staticmethod` on the real
    # class, so it unwraps to an ordinary three-argument function.
    def _relative_residual(self, A, x, b):
        return B._CudssFactor._relative_residual(A, x, b)


def test_refinement_reduces_the_residual_each_step():
    """A lossy factor plus refinement must beat the lossy factor alone.

    This is the whole claim of the single-precision mode, made checkable without
    a GPU: the correction loop is arithmetic, and arithmetic can be run here. The
    stub is 10% wrong per solve, which is far worse than float32 -- so a loop
    that converges on this cannot be passing by accident.
    """
    matrix = _spd_matrix(24)
    b = np.arange(1.0, 25.0)

    no_refine = _LossyStub(matrix, steps=0, tol=_LossyStub.MEASURING)
    one_step = _LossyStub(matrix, steps=1, tol=_LossyStub.MEASURING)
    three_steps = _LossyStub(matrix, steps=3, tol=_LossyStub.MEASURING)

    residual = lambda stub: stub._relative_residual(  # noqa: E731
        stub.A, stub.solve(b), b)

    r0, r1, r3 = residual(no_refine), residual(one_step), residual(three_steps)
    assert r1 < r0, f"one step must improve on none: {r1} !< {r0}"
    assert r3 < r1, f"three steps must improve on one: {r3} !< {r1}"
    assert r3 < 2e-4, f"three steps on a 10% error should reach ~1e-4, got {r3:.3e}"

    # and a step count past the geometric rate clears the tolerance outright
    converged = _LossyStub(matrix, steps=_LossyStub.CONVERGING)
    assert residual(converged) < converged.tol, residual(converged)


def test_the_refined_solve_is_float64():
    """The caller gets float64 in both precisions.

    The Lanczos iteration and the Rayleigh quotients run in float64; a float32
    return would silently downgrade them and defeat the refinement.
    """
    matrix = _spd_matrix(16)
    out = _LossyStub(matrix, steps=_LossyStub.CONVERGING).solve(np.ones(16))
    assert out.dtype == np.float64, out.dtype


def test_a_refinement_that_cannot_converge_raises():
    """The check that matters: never return an unconverged solve.

    A loop that runs its steps and returns whatever it has is the rig's
    signature failure -- a number that looks like an answer, is quietly less
    accurate than the double path it replaced, and reports nothing. This stub
    cannot converge (its ``_solve_once`` is wrong by 70% every time), so the
    solve must raise rather than return.
    """
    matrix = _spd_matrix(16)
    stub = _LossyStub(matrix, error=0.7, steps=2)

    with pytest.raises(B.BackendError) as excinfo:
        stub.solve(np.ones(16))

    message = str(excinfo.value)
    assert "did not converge" in message, message
    assert "K" in message, "the message must name which factorisation failed"
    assert B.REFINE_STEPS_ENV in message, "and how to raise the step count"
    assert B.PRECISION_ENV in message, "and the double-precision escape hatch"


def test_a_converging_solve_records_its_worst_residual():
    """The margin is reported, so a pass can be told from a near miss.

    "Residual under tolerance" is much weaker information than "residual was
    3e-12", and the lead asked for the latter in the log and in ``modes.json``.
    """
    matrix = _spd_matrix(20)
    stub = _LossyStub(matrix, steps=_LossyStub.CONVERGING)

    assert stub.worst_residual == 0.0, "nothing measured before the first solve"
    stub.solve(np.ones(20))
    assert 0.0 < stub.worst_residual < stub.tol, stub.worst_residual

    worse = _LossyStub(matrix, error=0.5, steps=_LossyStub.CONVERGING)
    with pytest.raises(B.BackendError):
        worse.solve(np.ones(20))
    assert worse.worst_residual > 0.0, (
        "a failed solve must still record what it measured"
    )


def test_the_residual_tolerance_is_configurable_and_validated(monkeypatch):
    monkeypatch.setenv(B.RESIDUAL_TOL_ENV, "1e-3")
    assert B.residual_tol() == 1e-3
    monkeypatch.setenv(B.RESIDUAL_TOL_ENV, "0")
    with pytest.raises(B.BackendError):
        B.residual_tol()
    monkeypatch.setenv(B.RESIDUAL_TOL_ENV, "loose")
    with pytest.raises(B.BackendError):
        B.residual_tol()


def test_the_default_precision_is_single():
    """Single is the production default; double is the override.

    Single is the only mode that fits the owner's 0.2 mm case (594k tets) --
    double fails outright above roughly 300-400k -- so a regression to double
    would take the largest uploads off the table rather than merely slow them.
    """
    assert B.gpu_precision() == "single"
    assert B.refine_steps() == B.DEFAULT_REFINE_STEPS


def test_double_precision_is_still_available(monkeypatch):
    monkeypatch.setenv(B.PRECISION_ENV, "double")
    assert B.gpu_precision() == "double"
    monkeypatch.setenv(B.PRECISION_ENV, "float32")
    with pytest.raises(B.BackendError):
        B.gpu_precision()


def test_double_precision_skips_the_refinement_and_the_check():
    """In float64 there is nothing to refine, so the loop is not run at all.

    Worth pinning because the residual check is skipped on this path: if double
    ever became the default again, a broken check would stop being exercised.
    """
    matrix = _spd_matrix(12)
    stub = _LossyStub(matrix, error=0.0, steps=3)
    stub.single = False
    stub.worst_residual = 0.0
    b = np.ones(12)

    out = stub.solve(b)

    assert stub.worst_residual == 0.0, "no refinement, so no residual measured"
    assert np.allclose(matrix.toarray() @ out, b, rtol=1e-12)


# --------------------------------------------------------------------------
# 6. the backend reaches the run's report
# --------------------------------------------------------------------------

def test_the_description_names_the_precision_on_the_gpu_backend():
    """``gpu`` alone is ambiguous once two precisions exist.

    A reader comparing two GPU runs needs to see which arithmetic produced them,
    because single+refine is a float32 factorisation with a correction loop, not
    a float64 one.
    """
    assert B.describe(B.CPU) == "cpu"
    assert B.describe(B.GPU) == "gpu (single+refine)"

    os.environ[B.PRECISION_ENV] = "double"
    try:
        assert B.describe(B.GPU) == "gpu"
    finally:
        del os.environ[B.PRECISION_ENV]


def test_solve_modes_records_the_backend_it_used():
    """What the caller reads back after the call, including on the CPU path."""
    points, tets, edges, tet_edges, free, constrained, A_free, B_free = _box_matrices()

    EIG.solve_modes(
        A_free, B_free, 4, points, edges=edges, tets=tets,
        tet_edges=tet_edges, free_edges=free, constrained=constrained,
        verbose=False, backend=B.CPU,
    )

    assert EIG.LAST_BACKEND == B.CPU
    assert EIG.LAST_BACKEND_DESCRIPTION == "cpu"
    assert EIG.LAST_WORST_RESIDUAL is None, (
        "the CPU path has no refinement, so it must report no residual rather "
        "than a zero that would read as a measurement"
    )


def test_modes_json_carries_the_backend_and_survives_the_real_reader():
    """Read the manifest back through the web app's parser, not ``json.loads``.

    ``web/fields.py`` validates the grid block and rejects a manifest that is
    missing it; adding a key must not disturb that. Using the real reader is the
    point -- a bare ``json.loads`` would prove the key was written, not that the
    consumer still accepts the file.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
    import fields as FLD  # noqa: E402

    manifest = {
        "units": "mm",
        "solver_backend": "gpu (single+refine)",
        "solver_worst_residual": 3.1e-12,
        "grid": {"nx": 2, "ny": 3, "nz": 4,
                 "origin": [0.0, 0.0, 0.0], "spacing": [1.0, 1.0, 1.0]},
        "modes": [],
    }
    parsed = FLD.load_modes_json(json.dumps(manifest).encode())

    assert parsed["solver_backend"] == "gpu (single+refine)"
    assert parsed["solver_worst_residual"] == 3.1e-12
    assert FLD.shape_of(parsed["grid"]) == (4, 3, 2)


def test_the_written_backend_is_the_runner_not_a_constant():
    """Guard against a hard-coded string in the report.

    The manifest has to say what *ran*. A literal ``"cpu"`` would pass every
    other test in this file and be wrong exactly when it matters -- on the
    Jetson, where the whole question is whether the GPU path was taken.
    """
    source = Path(EIG.__file__).read_text()
    assert "LAST_BACKEND_DESCRIPTION" in source

    report = Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver" / "mse_maxwell_solver.py"
    text = report.read_text()
    assert '"solver_backend": EIG.LAST_BACKEND_DESCRIPTION' in text, (
        "the manifest must read the backend the solver recorded"
    )
    assert '"solver_backend": "cpu"' not in text, "no hard-coded backend"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _spd_matrix(n: int, seed: int = 7) -> "EIG.sparse.spmatrix":
    """A small symmetric positive definite sparse matrix.

    Diagonally dominant with a random symmetric off-diagonal part, so it is SPD
    by construction and a dense inverse is exact enough to be the reference.
    """
    rng = np.random.default_rng(seed)
    off = rng.standard_normal((n, n)) * 0.1
    dense = 0.5 * (off + off.T) + n * np.eye(n)
    return EIG.sparse.csc_matrix(dense)


# --------------------------------------------------------------------------
# What a green run of this file does NOT prove
# --------------------------------------------------------------------------
#
# There is no GPU here, so nothing above shows that cuDSS factorises correctly,
# that its solves are accurate, or that single+refine really matches double on a
# 266k-tet mesh. Those are the lead's tier-2 Jetson run. What is established
# here: which backend is chosen, that an unknown one raises, that a missing CUDA
# stack raises instead of falling back, that both factorisations go through the
# factory, that the refinement loop converges and is checked, and that the
# backend and its worst residual reach the manifest.


# ---------------------------------------------------------------------------
# GPU memory: the warm worker must not accumulate device allocations
# ---------------------------------------------------------------------------
#
# On the Jetson the GPU has UNIFIED memory, so a per-job factorisation that is
# never released is system RAM -- and the MemoryMax cgroup does not reliably see
# cuDSS/nvmap allocations, so the first symptom is the OS killing the worker.
# These are tier 1: no GPU, no cupy. What they can prove is the SHAPE of the
# teardown -- that every created factor is closed, that closing is idempotent,
# and that the factors are closed BEFORE the pools are emptied.

def _fake_cupy(events: list):
    """A cupy module whose pools record when they were emptied.

    The real interface is ``get_default_memory_pool()`` and
    ``get_default_pinned_memory_pool()``, each returning an object with
    ``free_all_blocks()``. Modelling it rather than passing a bare Mock matters:
    a Mock would answer ANY call, so a release function that emptied the wrong
    pool -- or emptied a pool that does not exist -- would still pass.
    """
    import types

    class _Pool:
        def __init__(self, name):
            self.name = name

        def free_all_blocks(self):
            events.append(f"pool:{self.name}")

    cp = types.ModuleType("cupy")
    cp.get_default_memory_pool = lambda: _Pool("default")
    cp.get_default_pinned_memory_pool = lambda: _Pool("pinned")
    return cp


def test_every_created_gpu_factor_is_closed_and_the_pools_are_emptied_after(
        monkeypatch):
    """The order is the whole point: pockets before floors.

    Emptying a cupy pool while a live cuDSS factor still holds blocks reclaims
    only the scratch cupy allocated around it -- the bulk is in the factor, and
    it stays. So the assertion is not "both happened" but "close happened
    first", which a test that only counted calls could not tell apart.
    """
    events = []

    class _Solver:
        def free(self):
            events.append("solver.free")

    factors = []
    for _ in range(2):
        obj = B._CudssFactor.__new__(B._CudssFactor)
        obj.solver = _Solver()
        obj.cp = None
        factors.append(obj)

    B._GPU_FACTORS.update(factors)
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(events))

    closed = B.release_gpu_memory()

    assert closed == 2, f"expected both factors closed, got {closed}"
    assert events.count("solver.free") == 2, events
    # Both frees precede both pool empties.
    first_pool = min(i for i, e in enumerate(events) if e.startswith("pool:"))
    last_free = max(i for i, e in enumerate(events) if e == "solver.free")
    assert last_free < first_pool, (
        f"a pool was emptied while a factor was still alive: {events}")
    assert not B._GPU_FACTORS, "closed factors must leave the registry"


def test_closing_a_factor_twice_is_safe(monkeypatch):
    """Idempotent, because two owners may call it on the same unwind.

    The worker's cleanup and the registry's own clear both reach the same
    object, and this runs inside a ``finally``: a raise here would replace the
    job's real error with a teardown error and lose the reason the job failed.
    """
    calls = []

    class _Solver:
        def free(self):
            calls.append(1)

    obj = B._CudssFactor.__new__(B._CudssFactor)
    obj.solver = _Solver()
    obj.close()
    obj.close()

    assert calls == [1], f"free() ran {len(calls)} times, want exactly 1"


def test_a_factor_whose_device_is_gone_still_closes_quietly():
    """``free()`` on a lost device raises; that must not escape.

    A GPU reset, or an OOM kill already in flight, makes the teardown itself
    fail. There is nothing to recover -- the memory is gone either way -- and
    this runs on a path that may already be unwinding an exception, so the only
    correct behaviour is silence.
    """
    class _Solver:
        def free(self):
            raise RuntimeError("CUDA error: device not available")

    obj = B._CudssFactor.__new__(B._CudssFactor)
    obj.solver = _Solver()
    obj.close()                      # must not raise

    assert obj.solver is None, "the reference must still be dropped"


def test_release_gpu_memory_is_a_no_op_without_the_cuda_stack(monkeypatch):
    """On the CPU backend there is nothing to release and no import to fail on.

    The worker calls this on EVERY job, and most jobs run on the cluster. An
    ImportError escaping here would fail clean CPU jobs for a reason that has
    nothing to do with them.
    """
    monkeypatch.setitem(sys.modules, "cupy", None)
    B._GPU_FACTORS.clear()
    assert B.release_gpu_memory() == 0
