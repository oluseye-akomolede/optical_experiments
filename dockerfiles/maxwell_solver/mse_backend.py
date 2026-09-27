"""Which machine factorises the two SPD matrices, and how.

The shift-invert solve needs exactly two sparse factorisations, both symmetric
positive definite, both built once and solved many times:

* ``K = A + s B`` by the Lanczos iteration's ``OPinv``, and
* ``S = G^T B G`` by the divergence-free projector inside ``OPinv``.

Every ``OPinv`` application therefore does **two** solves, one against each. That
is why this module routes *both* rather than picking the bigger one: measured on
the Jetson, at 266k tets the CPU-side ``S`` factorisation alone is 19 s and the
projection solves dominate the run, so a GPU backend that took only ``K`` would
leave most of the time on the table.

The same solver package runs on the cluster (CPU) and on the Jetson (GPU). Which
one is chosen by:

    MSE_SOLVER_BACKEND=cpu   (default) -- SciPy/SuperLU, the cluster path
    MSE_SOLVER_BACKEND=gpu             -- cuDSS via nvmath, the Jetson path

Anything else raises. An unknown value is a configuration mistake and defaulting
it to ``cpu`` would hide it -- the run would succeed on the slow path and nobody
would know the setting had been ignored.

**The GPU import is lazy and there is no fallback.** A ``gpu`` run on a machine
without the CUDA stack must fail with a message naming what is missing, because
the alternative -- silently solving on the CPU -- would turn "the GPU run was
slow" into a mystery with no error in the log at all. The cluster image
(``python:3.12-slim`` + numpy/scipy) genuinely has no cupy or nvmath, so that is
the real path, not a hypothetical one.

The factorisation API is ``scratch/gpu-reference/profile_backend.py``'s
``CudssFactor``, verified working on the Jetson Orin (JetPack 6.2, CUDA 12.6).
What is deliberately *not* carried over is that harness's call-count heuristic --
it wrapped ``spla.splu`` and treated the first call as ``K``. That worked there
because it wrapped one fixed code path; here the caller says which matrix it is,
by name.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

#: the two backend names, in the order they are reported to the operator
CPU = "cpu"
GPU = "gpu"
BACKENDS = (CPU, GPU)

#: environment variable that selects the backend
BACKEND_ENV = "MSE_SOLVER_BACKEND"

#: factor in float32 and recover double precision by iterative refinement
PRECISION_ENV = "MSE_GPU_PRECISION"
REFINE_STEPS_ENV = "MSE_REFINE_STEPS"

#: how many refinement steps when ``MSE_GPU_PRECISION=single``
DEFAULT_REFINE_STEPS = 3

#: the most relative residual a refined solve may leave behind and still be used
RESIDUAL_TOL_ENV = "MSE_SOLVER_RESIDUAL_TOL"
DEFAULT_RESIDUAL_TOL = 1e-6

# what the gpu path needs, and what to tell the operator if it is absent
_GPU_PACKAGES = (
    ("cupy", "cupy"),
    ("cupyx.scipy.sparse", "cupyx.scipy.sparse"),
    ("nvmath", "nvmath"),
)


class BackendError(Exception):
    """The requested backend cannot be used, and falling back is not an option."""


def selected_backend() -> str:
    """The backend named by ``MSE_SOLVER_BACKEND``, defaulting to the CPU.

    Raises on any other value. An empty string is unknown, not "unset" -- it can
    only arrive from an explicit ``MSE_SOLVER_BACKEND=``, which is a mistake.
    """
    value = os.environ.get(BACKEND_ENV)
    if value is None:
        return CPU
    if value not in BACKENDS:
        raise BackendError(
            f"{BACKEND_ENV}={value!r} is not a known solver backend; "
            f"use one of {', '.join(BACKENDS)}"
        )
    return value


def describe(backend: str) -> str:
    """The backend name as it should appear in the log and in ``modes.json``.

    Reports the precision mode too, because ``single+refine`` changes what the
    numbers mean -- it is a float32 factorisation with a correction loop, not a
    float64 one -- and a reader comparing two runs needs to see that.
    """
    if backend == GPU and gpu_precision() == "single":
        return f"{GPU} (single+refine)"
    return backend


def gpu_precision() -> str:
    """``single`` (default) or ``double``, from ``MSE_GPU_PRECISION``.

    Single precision is the default because it is the only mode that fits the
    owner's largest case: measured on the Jetson at 266k tets it gives the same
    worst error as double (0.020%) with a ~3e-12 refined residual, a 2x faster
    factorisation and half the memory -- and ``double`` fails outright above
    roughly 300-400k tets, so a 594k-tet upload (the 0.2 mm case) has no double
    path at all. ``double`` remains available as an override.
    """
    value = os.environ.get(PRECISION_ENV, "single")
    if value not in ("double", "single"):
        raise BackendError(
            f"{PRECISION_ENV}={value!r} is not a known precision; use double or single"
        )
    return value


def residual_tol() -> float:
    """The most refined relative residual that may be returned without complaint.

    This is a check, not a statistic. Refinement that fails to converge means the
    float32 factorisation was not good enough for this matrix, and the solve it
    produced is quietly less accurate than the double path it replaced -- so it
    raises rather than being averaged into a log line.
    """
    raw = os.environ.get(RESIDUAL_TOL_ENV, str(DEFAULT_RESIDUAL_TOL))
    try:
        tol = float(raw)
    except ValueError as exc:
        raise BackendError(f"{RESIDUAL_TOL_ENV}={raw!r} is not a number") from exc
    if not tol > 0:
        raise BackendError(f"{RESIDUAL_TOL_ENV}={raw!r} must be positive")
    return tol


def refine_steps() -> int:
    raw = os.environ.get(REFINE_STEPS_ENV, str(DEFAULT_REFINE_STEPS))
    try:
        steps = int(raw)
    except ValueError as exc:
        raise BackendError(f"{REFINE_STEPS_ENV}={raw!r} is not an integer") from exc
    if steps < 0:
        raise BackendError(f"{REFINE_STEPS_ENV}={raw!r} must not be negative")
    return steps


def factor(A: sparse.spmatrix, *, label: str, spd_lu: dict,
           backend: str | None = None):
    """Factor ``A`` on the selected backend, returning an object with ``.solve``.

    ``label`` names the matrix (``"K"`` or ``"S"``) in diagnostics only -- it does
    not choose a backend, because both matrices go to the same one. It is what
    lets an error say *which* factorisation failed.

    ``spd_lu`` is the keyword dict for SuperLU's symmetric mode, built by the
    caller so that one flag governs both factorisations and the tests can turn it
    off. It is ignored by the GPU path, which is SPD by construction.

    ``solve(x)`` takes and returns **host float64 numpy arrays** on both
    backends, which is what makes this a drop-in seam.
    """
    backend = backend or selected_backend()
    if backend == CPU:
        # Through the module attribute, never a from-import: the test suite
        # monkey-patches `spla.splu` to assert both factorisations happen with
        # the symmetric arguments, and a from-imported name would escape it.
        return spla.splu(A, **spd_lu)
    if backend == GPU:
        return _gpu_factor(A, label=label)
    raise BackendError(
        f"{backend!r} is not a known solver backend; use one of {', '.join(BACKENDS)}"
    )


def _gpu_factor(A: sparse.spmatrix, *, label: str):
    """Factor on the GPU with cuDSS. Raises ``BackendError`` if the stack is absent."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import nvmath
    except ImportError as exc:
        missing = _missing_gpu_package()
        raise BackendError(
            f"{BACKEND_ENV}={GPU} needs the CUDA stack, but {missing} could not be "
            f"imported ({exc}). This image has no GPU support; run the GPU backend "
            f"on the Jetson, or set {BACKEND_ENV}={CPU}."
        ) from exc
    return _CudssFactor(A, cp=cp, csp=csp, nvmath=nvmath, label=label)


def _missing_gpu_package() -> str:
    """Name the first GPU package that cannot be imported, for the error message."""
    for module, name in _GPU_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            return name
    return "the CUDA stack"


class _CudssFactor:
    """A SuperLU-shaped object backed by cuDSS.

    factor once on the GPU, then ``.solve()`` many times. With
    ``MSE_GPU_PRECISION=single`` (the default) the factorisation is float32, half
    the memory, and each solve is corrected by iterative refinement against the
    **float64** matrix kept here on the host -- which is why ``self.A`` is
    retained rather than dropped after ``factorize()``.

    ``.solve`` returns float64 in both modes. That is the property that makes
    this substitutable for SuperLU, and the reason the precision mode is a
    runtime choice rather than a change to the caller.

    **Every refined solve is checked, not just corrected.** After the refinement
    steps the relative residual ``|b - A x| / |b|`` is measured and compared
    against ``residual_tol()``; if it is larger, the solve is not returned at all
    -- see ``solve``. The worst residual seen is kept in ``self.worst_residual``
    so the run can report how much margin it had.
    """

    def __init__(self, A, *, cp, csp, nvmath, label: str):
        self.cp = cp
        self.A = A.tocsr()                     # host float64; refinement needs it
        self.label = label
        self.single = gpu_precision() == "single"
        self.refine = refine_steps()
        self.tol = residual_tol()
        self.worst_residual = 0.0

        dtype = cp.float32 if self.single else cp.float64
        adv = nvmath.sparse.advanced
        opts = adv.DirectSolverOptions(
            sparse_system_type=adv.DirectSolverMatrixType.SPD,
            multithreading_lib=os.environ.get("CUDSS_MT_LIB"),
        )
        self.solver = adv.DirectSolver(
            csp.csr_matrix(self.A.astype(dtype)),
            cp.zeros(A.shape[0], dtype=dtype),
            options=opts,
        )
        self.solver.plan()
        self.solver.factorize()
        cp.cuda.Device().synchronize()

    def _solve_once(self, b: np.ndarray) -> np.ndarray:
        cp = self.cp
        dtype = cp.float32 if self.single else cp.float64
        self.solver.reset_operands(b=cp.asarray(b, dtype=dtype))
        return cp.asnumpy(self.solver.solve()).astype(np.float64)

    @staticmethod
    def _relative_residual(A, x: np.ndarray, b: np.ndarray) -> float:
        scale = float(np.linalg.norm(b))
        if scale == 0.0:
            # A zero right-hand side has an exactly zero solution; the norm of
            # the residual is then the only meaningful measure.
            return float(np.linalg.norm(A @ x))
        return float(np.linalg.norm(b - A @ x) / scale)

    def solve(self, x: np.ndarray) -> np.ndarray:
        out = self._solve_once(x)
        if self.single:
            # x += A^-1 (b - A x), with the residual against the float64 A. In
            # float64 this is exact enough that the iteration is the identity, so
            # there is nothing to gain from running it.
            for _ in range(self.refine):
                out = out + self._solve_once(x - self.A @ out)
            residual = self._relative_residual(self.A, out, x)
            self.worst_residual = max(self.worst_residual, residual)
            if residual > self.tol:
                # Returning this would be the quiet failure the rig keeps
                # meeting: a number that looks like a solve, is less accurate
                # than the double path it replaced, and says nothing. The
                # message carries what an operator needs to act -- which matrix,
                # how bad, and both knobs.
                raise BackendError(
                    f"the GPU single-precision solve of {self.label} did not "
                    f"converge: relative residual {residual:.3e} after "
                    f"{self.refine} refinement steps, over the {self.tol:.1e} "
                    f"limit. Increase {REFINE_STEPS_ENV} or set "
                    f"{PRECISION_ENV}=double."
                )
        return out
