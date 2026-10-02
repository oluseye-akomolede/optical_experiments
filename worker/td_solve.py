"""Run a claimed time-domain (FDTD) job: the request in, the results out.

This is the ``kind = 'td'`` counterpart to ``pc_solve``. It owns the translation
from the row's stored request (``tds``) to the solver's call, so a test can
drive it without a lease, a store or a GPU.

**The request is re-derived, never trusted.** As in ``pc_solve``, the grid and
the TF/SF box are rebuilt here from the request and the upload, never read back
from a stored estimate: a count is not a grid.

Phase 4.1 defines the request contract only. The solve itself (routing, the
free-memory check, results) lands with 4.2 and 4.3.
"""
from __future__ import annotations


class TDSolveError(Exception):
    """The claimed job does not carry a request this module can solve.

    A distinct type for the reason ``PCSolveError`` is one: a malformed row is
    deterministic (fail the job, do not requeue it), unlike a transient solver
    error (release and retry).
    """


#: The request keys, in one place. The phase-5 form's ``request_json`` will
#: write exactly these and a test will pin the two lists against each other, as
#: ``test_pc_solve`` does for ``pcs``. Every key is always present; a key that
#: does not apply to the chosen source holds JSON null (``direction`` for a
#: point source, ``position``/``polarisation`` axis rules for a plane wave,
#: ``samples``/``sample_duration_s`` unless the profile is ``drawn``,
#: ``bandwidth`` for ``cw``). Present-but-null and absent are different on
#: purpose: absent means the writer and this reader disagree about the
#: contract, which is the defect this list exists to catch.
#:
#: The TF/SF box is not a key: it is derived from the geometry's bounding box
#: and the clearance rule, for the reason ``k_points`` is not a pc key.
REQUEST_KEYS = ("boundary", "source_kind", "profile", "frequency_hz",
                "bandwidth", "position", "direction", "polarisation",
                "samples", "sample_duration_s", "duration_s", "probes", "dx_m")


def request_from(job: dict) -> dict:
    """The request dict out of a claimed job, or raise naming what is wrong.

    ``tds`` is NULL for every other kind and ``{}`` would be a td job with every
    field missing; both are refused. The check is by KEY, not truthiness, since
    ``null`` and ``0`` are legitimate values for some keys. Keys outside the
    contract are dropped rather than passed on, so results written back into
    the same column can never be read as input.
    """
    tds = job.get("tds")
    if not isinstance(tds, dict):
        raise TDSolveError(
            f"a td job must carry a 'tds' request object, got {type(tds).__name__}"
        )
    missing = [k for k in REQUEST_KEYS if k not in tds]
    if missing:
        raise TDSolveError(f"the 'tds' request is missing {', '.join(missing)}")
    return {k: tds[k] for k in REQUEST_KEYS}
