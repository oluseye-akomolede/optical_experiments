"""Run a claimed photonic-crystal job: the request in, the bands out.

This is the ``kind = 'pc'`` counterpart to the 3-D ``real_solve``. It owns the
one thing that is easy to get wrong and expensive to get wrong quietly: the
translation from the row's stored request to the solver's call, and back.

**Why this is a separate module rather than a branch inside the worker.** The
band solve runs in the WORKER's process on the Jetson (CPU), not in the web pod
-- see the plan's §3: the web pod is `cpu: 500m` and a CPU-bound solve there
starves the 5 s scheduler tick, which is the failure, not the minutes. Keeping
the translation here means a test can drive it with a stub solver and assert the
call it makes, without a lease, a store or a container.

**The request is re-derived, never trusted.** ``pcs`` on the row is what the
owner asked for. The mesh is rebuilt from it by the solver's own
``unit_cell_mesh`` inside ``bands_from_mesh``, so the node count the web side
capped on and the mesh actually solved come from one definition. The row also
carries ``nodes``/``triangles`` from the web side's estimate; those are NOT read
here, deliberately -- a count is not a mesh, and trusting the count is how an
estimate and a solve drift apart (the multi-part estimate bug).

**The k-path is derived, not stored.** ``k_points`` is not a form field
(``web/pc_form.py`` derives it from the lattice), so it is not in the request
either. It is rebuilt from the lattice here, which is the only way the path the
owner saw in the diagram and the path solved can be guaranteed the same: they
are the same function of the same field.
"""
from __future__ import annotations

from typing import Any, Callable


class PCSolveError(Exception):
    """The claimed job does not carry a request this module can solve.

    A distinct type, not a bare ``ValueError``: the worker's failure path needs
    to tell a malformed row (deterministic -- fail the job, do not requeue it)
    from a transient solver error (release and retry). See the worker's
    ``/failed`` vs ``/release`` split.
    """


#: The request keys, in one place. ``web/pc_form.request_json`` writes exactly
#: these; this module reads exactly these. They are pinned against each other by
#: a test so a field added on the form cannot silently stop reaching the solve.
REQUEST_KEYS = ("lattice", "r_over_a", "n_rod", "n_bg", "n_bands",
                "divisions", "polarisation")


def _request_from(job: dict) -> dict:
    """The validated request dict out of a claimed job, or raise.

    ``pcs`` is NULL for a cavity job and ``{}`` would be a pc job with every
    field missing; both are refusals here. The check is by KEY rather than by
    truthiness, because ``r_over_a = 0`` and ``n_bg = 1.0`` are legitimate
    values and a truthiness test would refuse them.
    """
    pcs = job.get("pcs")
    if not isinstance(pcs, dict):
        raise PCSolveError(
            f"a pc job must carry a 'pcs' request object, got {type(pcs).__name__}"
        )
    # ``bands`` is written back into this same column when the job finishes, so
    # a second solve of a finished row would otherwise read its own output as
    # input. The request keys are what is read; anything else is ignored rather
    # than propagated into the solver call.
    missing = [k for k in REQUEST_KEYS if k not in pcs]
    if missing:
        raise PCSolveError(
            f"the 'pcs' request is missing {', '.join(missing)}"
        )
    return {k: pcs[k] for k in REQUEST_KEYS}


def solve_pc(job: dict, *, solver: Any = None) -> dict:
    """Solve a claimed pc job and return the payload to report.

    ``solver`` is the ``pc`` module (or a stand-in in a test). Imported lazily so
    this module can be imported where the solver is absent -- the web image
    ships ``pc.py`` for the form's node count, but the WORKER must not depend on
    the form's module to solve, and a test of the request translation must not
    need numpy.

    Returns a dict with the k-path labels, the k-points and the bands, all plain
    Python numbers, ready for JSON:

        {"lattice": ..., "polarisation": ...,
         "labels": [...], "k_points": [[kx, ky], ...], "bands": [[...], ...]}

    The bands are ``w a / (2 pi c)``, one vector of ``n_bands`` per k-point, as
    ``pc.solve_bands`` returns them (``sqrt(lam) / 2 pi``; see its docstring for
    the measured empty-lattice landmarks at Gamma, X and M).
    """
    request = _request_from(job)
    if solver is None:
        solver = _load_solver()

    labels, k_points = solver.high_symmetry_path(request["lattice"],
                                                 points_per_segment=10)
    kp, bands = solver.bands_from_mesh(
        request["lattice"],
        request["divisions"],
        request["r_over_a"],
        request["n_rod"],
        request["n_bg"],
        k_points,
        request["n_bands"],
        polarisation=request["polarisation"],
    )
    # ``tolist`` rather than leaving numpy arrays: this dict is JSON-encoded by
    # the caller, and a numpy float64 is not JSON-serialisable -- the failure
    # would land at the route as a 500 with the solve already done, the
    # "could-not-ask" shape. Converting here is the one place that knows the
    # payload is going on the wire.
    return {
        "lattice": request["lattice"],
        "polarisation": request["polarisation"],
        "labels": list(labels),
        "k_points": [[float(x), float(y)] for x, y in kp],
        "bands": [[float(v) for v in row] for row in bands],
    }


def _load_solver() -> Any:
    """The real ``pc`` module, imported at call time.

    Deferred rather than a module-level import for the reason in :func:`solve_pc`:
    the worker's own tests must be able to exercise the request translation
    with a stub, and the module must stay importable in an environment that
    lacks the solver's numpy/scipy.
    """
    import pc  # noqa: PLC0415 -- deferred deliberately; see the docstring.

    return pc
