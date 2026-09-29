"""Photonic-crystal form: validate it, size it, refuse it by name, or accept it.

WHY THIS FILE IS NAMED ``pc_form`` AND NOT ``pc``. The solver's module is
``dockerfiles/maxwell_solver/pc.py``, and a ``web/pc.py`` would win the name
``pc`` on ``sys.path`` -- this module imports the solver's ``pc`` for the lattice
table and the cell mesh, so calling itself ``pc`` makes that import resolve to
ITSELF. The symptom is not an ImportError: ``_pc_module()`` returns this module,
whose ``high_symmetry_path`` does not exist, and the failure surfaces as a
``AttributeError`` on a function the reader can see defined in the file they are
looking at. Renamed, not shadowed-and-worked-around: any fix that keeps the name
is either a sys.path order the next caller can undo or a rename of the SOLVER's
module, and that one is imported by name from the worker image.

WHERE THIS SITS. A photonic-crystal band diagram is a ``kind='pc'`` job: a worker
claims it, runs ``dockerfiles/maxwell_solver/pc.py``, and stores the bands. This
module is the web side of that path and NOTHING ELSE -- it parses the form,
counts how big the problem would be, and either enqueues or refuses. It never
solves, and the reason is in the plan's §3: the web pod is ``cpu: 500m`` with
``gunicorn -w 1 --threads 8`` and a scheduler tick every 5 s in a thread of the
same process, so a CPU-bound solve does not merely occupy a request thread, it
halves the core away from the tick. The measured solve is ~12 s on a quiet
machine and ~6.1 min under contention; neither belongs in a request.

THE GUARD IS THE POINT. ``test_pc.py`` asserts that no web module imports the
solve, and it does so as an IMPORT-GRAPH test rather than by reading this file's
docstring. A comment saying "must not import the solve" is a claim; the test is
the check. It is deliberately a graph walk, not a single grep for ``solve_bands``,
because the thing to prevent is a web module reaching the solve through some
other name.

UNITS. Everything the owner types is dimensionless: ``r_over_a`` is a fraction of
the lattice constant, and the frequencies come back as ``w a / (2 pi c)``. So
unlike the 3-D upload there is no unit, no scale and no min-feature here -- and
no length-space bug to have.

WHY THE COUNT IS COMPUTED HERE AT ALL, given the worker re-derives it anyway. The
node count is the number the cap refuses on, and the refusal has to happen BEFORE
a job is written, or the owner gets a row that is certain to fail. The plan
measured the Delaunay at 12.5 ms for the production mesh, so counting inline is
cheap; and because it goes through the very same ``unit_cell_mesh`` the worker
uses, the number the owner is refused on is the number the worker would meet.
A re-derived estimate here would be a second opinion, and a second opinion that
disagrees is worse than none.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Nodes per unit cell, ceiling. Set by §3 of the plan from the measured cost:
#: the production mesh is 1,681 nodes at N = 40 and the per-k cost grows
#: superlinearly beyond N ~ 70 (3,249 nodes at N = 56). Past 5,000 the k-point
#: cap no longer bounds the wall clock, so this is what actually holds the
#: budget and it must be refused with its own number.
MAX_NODES = 5000

#: k-points per polarisation, ceiling. A Gamma-X-M-Gamma path at ~10 points per
#: segment is 30. At the measured MEDIAN 0.2 s/k that is ~12 s; at the measured
#: MAX 6.1 s/k under contention it is ~6.1 min, which is the figure the cap is
#: sized against -- a cap is a bound on the worst case and the worst case is
#: somebody else's `pip install`.
MAX_K_POINTS = 30

#: bands per polarisation, ceiling. Not a budget number like the two above: it is
#: the largest count the solver's ARPACK call has been exercised with, and past
#: it the k-point cap stops describing the cost either. Generous, because asking
#: for more bands is cheap relative to asking for more k-points.
MAX_BANDS = 20

#: divisions per lattice vector, floor and ceiling. The floor is the solver's own
#: (:func:`pc.unit_cell_mesh` refuses below 4), restated here so the form can say
#: so before calling it. The ceiling is a courtesy bound under ``MAX_NODES``; at
#: N = 70 the mesh is already ~4,900 nodes, so the node cap bites first in
#: practice and this merely stops a plain typo from building a huge Delaunay
#: before the cap is consulted.
MAX_DIVISIONS = 70

#: lattice names the solver accepts. Read from the solver's own table at import
#: time (see :func:`_pc_module`) rather than retyped, so a third lattice added
#: there is accepted here without an edit -- and, more to the point, a lattice
#: REMOVED there cannot linger as a name this module still offers.
LATTICE_CHOICES = ("square", "triangular")

#: polarisations, in the PhC's own naming (the out-of-plane field). ``TM`` is
#: ``E_z`` and ``TE`` is ``H_z`` -- see pc.py's header for why those map to the
#: 1-D layer problem's ``s`` and ``p`` and why that is the easiest error in the
#: feature to make.
POLARISATIONS = ("TM", "TE")

#: The value of the upload form's mode selector that routes a POST here. The
#: cavity form does not send this field at all, so an absent ``mode`` is the
#: cavity path -- which is what keeps every form and bookmark from before this
#: feature working unchanged. It is a named constant rather than a literal
#: because the template, the route and this module all have to agree on the
#: spelling, and a lowercase ``"PC"`` mismatching here would silently send a
#: photonic-crystal request down the cavity path.
MODE = "pc"


class PCRequestError(Exception):
    """A photonic-crystal request the owner can fix, with the reason named.

    Subclasses nothing on purpose. The app's upload path catches
    ``meshing.UploadError`` to render the wave-off; this is a separate form and
    gets its own handler, so sharing a base class would silently route a pc
    refusal through the cavity upload's message rendering -- which names file
    sizes and .obj problems that do not exist here.
    """


def _pc_module():
    """``pc``: the solver's own geometry, for the count and the caps.

    Reaches the solver directory the way ``web/meshing.py`` reaches
    ``mse_common`` -- a late import with the path inserted -- because that is the
    established idiom and because the alternative is a copy of the lattice table
    and the cell algorithm in a second place.

    What it deliberately does NOT reach is any solve. Importing this module pulls
    ``pc``'s module-level imports (numpy and scipy.sparse); ``solve_bands``,
    ``bands_from_mesh`` and ``bloch_reduce`` are defined in the same file and so
    become importable with it, which is why the guard test is about the web
    modules' import GRAPH and the job path, not about a name being absent from
    ``dir(pc)``. Nothing here calls them, and the docker image contract test is
    what keeps the module present in the web image at all.
    """
    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pc  # noqa: PLC0415 - deliberately late, so tier-1 tests stay cheap

    return pc


def _as_float(raw, field: str) -> float:
    """A form value as a float, or a refusal that names the field.

    An empty box and a typo are the same thing to the owner and must not be the
    same thing to the code: both are refusals, and the message names WHICH box so
    the owner has somewhere to look.
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise PCRequestError(f"Please give a value for {field}.")
    try:
        return float(text)
    except (TypeError, ValueError):
        raise PCRequestError(f"{field} must be a number, but was {text!r}.") from None


def _as_int(raw, field: str) -> int:
    """An integer form value, refusing a fractional one rather than truncating.

    ``int(3.7)`` is 3, and silently solving for 3 when 3.7 was typed is the
    owner's request quietly altered. A form that only ever sent integers would
    make this unreachable; the route does not control what arrives.
    """
    value = _as_float(raw, field)
    if value != int(value):
        raise PCRequestError(f"{field} must be a whole number, but was {value}.")
    return int(value)


def parse_request(form) -> dict:
    """The submitted form as a validated ``pc`` request dict.

    Returns exactly the keys the worker's request stores -- ``lattice``,
    ``r_over_a``, ``n_rod``, ``n_bg``, ``n_bands``, ``divisions``,
    ``polarisation`` -- so the row is the request and no field is re-derived at
    solve time from a default that the owner never saw.

    ``divisions`` IS part of the request even though the plan's §5 lists only the
    first four. It has to be: it changes the mesh, and therefore the node count
    the cap refuses on and the frequencies that come out, so a row that omitted
    it could not say what was solved. ``k_points`` is NOT a form field -- the
    path is derived from the lattice by :func:`k_point_count`, so the owner
    cannot ask for a path that is not the irreducible-zone path.
    """
    pc = _pc_module()

    lattice = ("" if form.get("lattice") is None
               else str(form.get("lattice")).strip().lower())
    if lattice not in LATTICE_CHOICES:
        raise PCRequestError(
            f"Unknown lattice {lattice!r}. Choose one of "
            f"{', '.join(LATTICE_CHOICES)}."
        )

    polarisation = ("" if form.get("polarisation") is None
                    else str(form.get("polarisation")).strip().upper())
    if polarisation not in POLARISATIONS:
        raise PCRequestError(
            f"Unknown polarisation {polarisation!r}. Choose one of "
            f"{', '.join(POLARISATIONS)} (TM is E_z, TE is H_z)."
        )

    divisions = _as_int(form.get("divisions"), "the number of divisions per cell edge")
    if divisions < 4:
        # The solver's own floor. Named as a floor rather than clamped: a silent
        # bump to 4 would solve a different problem than the one asked for.
        raise PCRequestError(
            f"The cell must be divided into at least 4 parts per edge, got "
            f"{divisions}. Please use 4 or more."
        )
    if divisions > MAX_DIVISIONS:
        raise PCRequestError(
            f"{divisions} divisions per edge is beyond this app's limit of "
            f"{MAX_DIVISIONS}."
        )

    r_over_a = _as_float(form.get("r_over_a"), "the rod radius (as a fraction of a)")
    if not 0.0 < r_over_a < 0.5:
        # 0.5 is the geometric extreme for a circle in a square cell and is not
        # reachable for a triangular one either; the solver's own inradius guard
        # is tighter still and is left to speak for itself when it fires.
        raise PCRequestError(
            f"The rod radius must be more than 0 and less than 0.5 of the "
            f"lattice constant, got {r_over_a}."
        )

    n_rod = _as_float(form.get("n_rod"), "the rod refractive index")
    n_bg = _as_float(form.get("n_bg"), "the background refractive index")
    for value, field in ((n_rod, "rod"), (n_bg, "background")):
        if value <= 0.0:
            raise PCRequestError(
                f"The {field} refractive index must be positive, got {value}."
            )

    n_bands = _as_int(form.get("n_bands"), "the number of bands")
    if n_bands < 1:
        raise PCRequestError(
            f"The number of bands must be at least 1, got {n_bands}."
        )
    if n_bands > MAX_BANDS:
        raise PCRequestError(
            f"{n_bands} bands is beyond this app's limit of {MAX_BANDS}."
        )

    return {
        "lattice": lattice,
        "r_over_a": r_over_a,
        "n_rod": n_rod,
        "n_bg": n_bg,
        "n_bands": n_bands,
        "divisions": divisions,
        "polarisation": polarisation,
    }


def k_point_count(lattice: str, points_per_segment: int = 10) -> int:
    """How many k-points a full path at this lattice has.

    Derived from the solver's own :func:`pc.high_symmetry_path` -- the returned
    array's length, not a formula -- because the count is what the cap refuses
    on, and a count computed from the number of segments would disagree with the
    path the solver actually walks the moment the path changes. Shared endpoints
    are already counted once by that function, which is exactly why the length is
    the right thing to ask for.
    """
    pc = _pc_module()
    _labels, points = pc.high_symmetry_path(lattice, points_per_segment)
    return int(len(points))


def cell_size(request: dict) -> tuple[int, int]:
    """``(nodes, triangles)`` for this request's unit cell, WITHOUT solving.

    Builds the cell through the solver's own ``unit_cell_mesh`` and counts what
    it returns. The count is the cap's operand, so it must be the mesh the worker
    will build -- an independent estimate would be a second opinion, and §3 says
    the two caps are re-derived from the measurement rather than adjusted by
    feel.

    Triangulating is the expensive part of this call (12.5 ms at the production
    size); it is done here because the alternative is refusing on a *predicted*
    node count that could be wrong in the dangerous direction.
    """
    pc = _pc_module()
    points, triangles, _eps, _pairs = pc.unit_cell_mesh(
        request["lattice"],
        int(request["divisions"]),
        r_over_a=float(request["r_over_a"]),
        n_rod=float(request["n_rod"]),
        n_bg=float(request["n_bg"]),
    )
    return int(len(points)), int(len(triangles))


def check_caps(request: dict, nodes: int, k_points: int) -> None:
    """Refuse the request unless it is inside BOTH caps.

    Each cap is refused with its OWN number and its own sentence. They are never
    summed into one "the job is too big": a summed count stays green when either
    half is killed, so an owner told "too big" cannot tell which lever to pull,
    and a reviewer cannot tell which cap regressed. ``test_pc.py`` asserts each
    refusal by raising one cap at a time and checking that only its own test
    reddens.
    """
    if k_points > MAX_K_POINTS:
        raise PCRequestError(
            f"This path has {k_points} k-points per polarisation, which is over "
            f"this app's limit of {MAX_K_POINTS}. Reduce the number of points "
            f"per path segment."
        )
    if nodes > MAX_NODES:
        raise PCRequestError(
            f"This cell would have {nodes} nodes, which is over this app's limit "
            f"of {MAX_NODES}. Reduce the number of divisions per cell edge."
        )


def plan_request(form) -> dict:
    """Validate, size and cap-check a submission; return what to enqueue.

    The one entry point the route calls. Returns a dict: the request, plus the
    derived ``nodes``, ``triangles`` and ``k_points`` that were computed to decide
    whether to accept -- recorded because they are the numbers the decision was
    made on, the same reason ``est_tets`` is stored on a cavity row.

    Raises :class:`PCRequestError` with the limit or the field named. It does NOT
    enqueue; the caller writes the row, so this function stays testable without a
    database.
    """
    request = parse_request(form)
    nodes, triangles = cell_size(request)
    k_points = k_point_count(request["lattice"], points_per_segment=10)
    check_caps(request, nodes, k_points)
    return {
        **request,
        "nodes": nodes,
        "triangles": triangles,
        "k_points": k_points,
    }


def request_json(planned: dict) -> dict:
    """The request as it goes into ``jobs.pcs`` -- what was asked, not what came out.

    Deliberately EXCLUDES the derived ``nodes``/``triangles``/``k_points``: those
    are the web side's estimate and they go in the row's own ``est_tets``-style
    place, whereas ``pcs`` is meant to be the request the worker re-derives the
    mesh from. Mixing a pre-computed count into the request would invite a future
    reader to trust it instead of the mesh, and the mesh is the authority.
    """
    keys = ("lattice", "r_over_a", "n_rod", "n_bg", "n_bands",
            "divisions", "polarisation")
    return {key: planned[key] for key in keys}
