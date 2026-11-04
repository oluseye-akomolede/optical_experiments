r"""Scalar P1 (linear tetrahedron) assembly for the electrostatics problem

    -div(eps_static * eps0 * grad phi) = rho

The weak form over a test function ``v`` that vanishes on the electrodes is

    \int eps_static eps0 grad(phi) . grad(v) dV  =  \int rho v dV
                                                   + (electrode terms)

so each tet contributes the 4x4 matrix

    K_ij = eps_static * eps0 * vol * (grad lambda_i . grad lambda_j)

with ``grad lambda_i`` the constant barycentric gradients. The material enters
only as a per-tet scalar, so a mesh whose tets carry a part index assembles with
one broadcast -- the same shape :func:`mse_fem.element_matrices` takes as its
``eps_by_tet`` array argument (an array there, a table-to-array helper here).

What this is exact for: a CONFORMING interface, i.e. one that lies on element
faces. There P1 with per-element ``eps`` reproduces the piecewise-linear layered
solution exactly -- measured at 2e-15 on a structured chain with a 1:2 contrast
and 3e-15 with 1:4 -- and refinement changes nothing, which is why the test pins
the exact values rather than a convergence rate. The mesher places part
boundaries on faces, so this is the case the product hits.

What it does NOT handle: a tet that SPANS two materials. Its single ``eps``
value is then wrong for part of its volume, and :func:`check_tet_regions` does
not detect such a tet -- it trusts the per-tet region index the mesher supplied.
A mesh whose interface cuts through elements needs either a mesher that snaps
the interface to faces, or an element that integrates ``eps`` over sub-regions.

This module is the SCALAR problem and does not touch :mod:`mse_fem`'s edge
elements; it shares only :func:`mse_fem.tet_geometry`, so the barycentric
gradients (and the ``inv(J).T`` trap they carry) exist in one place.

Units: ``points`` in metres, ``eps_static`` dimensionless (relative), so the
matrix is in ``eps0 * m``. The permittivity of free space is applied here so a
caller never has to remember it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

import mse_backend as BACKEND
import mse_fem
import fdtd

#: Re-exported from the time-domain solver so both paths answer "which device?"
#: the same way. ``budget=NO_DEVICE`` means the caller is not on the run's device.
NO_DEVICE = fdtd.NO_DEVICE

#: Re-exported: the admission ceiling of THIS process, read at call time.
admission_budget = fdtd.admission_budget

#: vacuum permittivity, F/m (CODATA), applied to the relative input
EPS0 = 8.8541878128e-12

#: SuperLU keywords for a symmetric positive-definite matrix. The reduced
#: system after a Dirichlet elimination is SPD for a well-posed device, so the
#: symmetric mode is both faster and a check in itself. Same dict as
#: :mod:`mse_eigen` builds, for the same reason.
SPD_LU = {
    "permc_spec": "MMD_AT_PLUS_A",
    "diag_pivot_thresh": 0.0,
    "options": {"SymmetricMode": True},
}

#: Node count above which :func:`solve` switches from the direct factorisation
#: to AMG-preconditioned CG. This is a CROSSOVER, not a memory limit, and it is
#: still a named estimate: it is where the direct solve inverts slower than the
#: iterative one, and task 1.4's agreement test is the only place it is
#: exercised. Memory does NOT set it -- a direct factorisation at this node
#: count stays far inside a 16 GiB container (see :data:`DIRECT_ESTIMATE`). Do
#: not derive it from the memory model: that would be a memory limit dressed up
#: as a speed choice.
AMG_NODE_CROSSOVER = 5 * 10**5

#: Measured peak-memory growth of the DIRECT (SuperLU) path, in bytes above the
#: empty-process baseline. Task 1.7 measured these on a bounded cluster Job
#: (sigma-worker2, x86_64, 2 cpu / 6 GiB, image mse-maxwell-solver-gpu:28997f1
#: with no GPU, code at b3bbdc1) using ``solve_memory_bench.py``; the job spec and
#: the raw output are in ``openspec/changes/electrical-simulation/memory-bench.md``.
#:
#: The first column is ``n_free``, the reduced unknown count the solve actually
#: factorises; the second is the peak RSS growth it took, IN BYTES. Fill in a
#: sparse direct factorisation is NOT linear in ``n`` -- over this table the bytes
#: per unknown climb from 6,148 B to 44,574 B, with a 2.2x jump between the last
#: two points as the ordering changes strategy. No closed law fits it (a power law
#: is 27 % off its own worst point, a linear fit 277 %), so the estimate
#: INTERPOLATES this table in log-log space and refuses beyond it rather than
#: extrapolate a curve it does not have. See :func:`_direct_bytes`.
#: The bench (``solve_memory_bench.py``) reports growth in MiB (``grow_kb / 1024``);
#: ``VmHWM`` is in KiB.  ``MIB`` converts those MiB figures to bytes.
MIB = 1_048_576

DIRECT_MEMORY_TABLE = (
    (1_859, 10.9 * MIB),
    (6_137, 32.2 * MIB),
    (14_375, 122 * MIB),
    (27_869, 339 * MIB),
    (47_915, 723 * MIB),
    (75_809, 1_444 * MIB),
    (112_847, 4_797 * MIB),
)

#: Measured peak-memory growth of the AMG path, which IS well described by a
#: power law over the range it was measured. ``bytes = coefficient * n ** exponent``,
#: fitted on the same cluster Job as :data:`DIRECT_MEMORY_TABLE`; the fit's worst
#: point is 2.0 % off over ``n_free`` 1,859..27,869. That range is the ONLY range
#: the law is measured for -- above it the estimate is scaled by
#: :data:`AMG_EXTRAPOLATION_MARGIN` and reported as extrapolated.
AMG_MEMORY_POWER_LAW = {"coefficient": 12_812.0, "exponent": 0.91}

#: The largest ``n_free`` the AMG power law was fitted to. Above this the law is
#: an extrapolation and the estimate is inflated to say so.
AMG_MEASURED_MAX_UNKNOWNS = 27_869

#: Safety factor applied to the AMG power law above its measured range. The law
#: under-predicts if the hierarchy's cost per level grows at larger sizes, and an
#: under-predicting memory estimate is the one that gets a pod OOM-killed. 1.5x is
#: a margin, not a measurement: the reported figure is labelled ``extrapolated``.
AMG_EXTRAPOLATION_MARGIN = 1.5

#: Relative residual AMG-CG must reach before its answer is accepted.
#:
#: 1e-12, not 1e-10: the direct path is exact to machine precision (~1e-15
#: measured on the contact chain), so the agreement test compares AMG against a
#: near-exact reference and every bit of the difference is AMG's. Measured at
#: rtol=1e-10 the AMG error against the exact answer reached 6.4e-11 -- too
#: close to the tolerance to assert against, and needlessly loose when the
#: alternative is exact.
AMG_RTOL = 1e-12

#: Factor by which the relative residual may exceed :data:`AMG_RTOL` before
#: the solve is reported as not converged. A bare ``>= rtol`` test would refuse
#: an answer that already meets the requested accuracy, so the check is against
#: the tolerance actually asked for.
AMG_RESIDUAL_SLACK = 10.0

#: The values :func:`solve` accepts for ``method``. A typo must be refused, not
#: silently treated as ``auto`` -- that would be the "the setting was ignored"
#: failure ``mse_backend.selected_backend`` refuses for the same reason.
METHODS = ("auto", "direct", "amg")

#: Relative size below which a barycentric coordinate counts as "on the face".
#: A point EXACTLY on a shared face is ambiguous -- both tets contain it -- and
#: :func:`field_at` must choose one. Face-coincident points are common rather
#: than pathological: an axis-aligned device puts electrode faces and sample
#: points on node planes. The tie-break is deterministic (the lowest tet index
#: whose weight is >= -this) so two calls on the same point agree to the bit.
BARY_EPS = 1e-9


def check_tet_regions(region_of_tet: np.ndarray, parts) -> np.ndarray:
    """Validate the mesh's own per-tet region ids, and return them as an array.

    The material of a linear tet is a property of the TET, not of its nodes:
    ``tetrahedralize`` returns ``region_of_tet`` from TetGen's region
    attributes, so every tet belongs wholly to one part even where the boundary
    runs between two tets sharing a face.

    Deriving the part from node membership instead is wrong and looks right:
    a boundary face's three nodes belong to both tets on either side, so every
    connected two-part mesh would be rejected as "straddling". This takes the
    mesh's answer.

    Refuses a region id with no matching part (a tet whose material nobody
    stated -- it would otherwise be assembled at some default) and a part that
    no tet uses (a part the caller believes is in the device and is not).
    """
    region_of_tet = np.asarray(region_of_tet, dtype=np.int64).ravel()

    used = sorted(int(r) for r in np.unique(region_of_tet).tolist())
    out_of_range = [r for r in used if r < 0 or r >= len(parts)]
    if out_of_range:
        raise ValueError(
            f"the mesh has {len(out_of_range)} region id(s) {out_of_range} with "
            f"no part; this mesh declares {len(parts)} part(s) ({', '.join(parts)})"
        )
    unused = [p for i, p in enumerate(parts) if i not in used]
    if unused:
        raise ValueError(
            f"part(s) {', '.join(unused)} have no tets in this mesh; refusing so "
            "a part the caller believes is present is not silently absent"
        )
    return region_of_tet


def eps_by_tet(
    parts: list[str],
    eps_static: dict[str, float],
    tet_part: np.ndarray,
    allow_missing=frozenset(),
) -> np.ndarray:
    """Per-tet relative permittivity from the caller's per-part table.

    Refuses, at the call and before any matrix is built: a part with no
    ``eps_static`` (a defaulted permittivity is a different device), an
    ``eps_static`` naming a part that does not exist (a typo would otherwise be
    dropped silently), and a non-positive or non-finite value (zero makes the
    matrix singular and a negative makes it indefinite -- neither is a device
    anyone can build).

    ``allow_missing`` names parts that may omit ``eps_static`` -- the held
    electrodes, whose tets contribute no stiffness row. Their entries are set
    to NaN so that using one where a permittivity is required is a loud
    failure rather than a silent zero.
    """
    # Unknown names FIRST: a typo like "gap " alongside a real "gap" would
    # otherwise be reported as a missing permittivity for "gap", pointing the
    # caller at the wrong line.
    unknown = [p for p in eps_static if p not in parts]
    if unknown:
        raise ValueError(
            f"eps_static names part(s) that do not exist: {', '.join(unknown)}; "
            f"this mesh has {', '.join(parts)}"
        )
    missing = [p for p in parts if p not in eps_static and p not in allow_missing]
    if missing:
        raise ValueError(
            f"no eps_static for part(s): {', '.join(missing)}; every part needs "
            "its own static permittivity"
        )
    values = np.empty(len(parts), dtype=np.float64)
    for i, p in enumerate(parts):
        if p not in eps_static:
            values[i] = np.nan      # a held electrode: never assembled
            continue
        v = float(eps_static[p])
        if not np.isfinite(v):
            raise ValueError(f"eps_static for part {p!r} is not finite: {eps_static[p]!r}")
        if v <= 0.0:
            raise ValueError(
                f"eps_static for part {p!r} must be positive, got {v!r}; a "
                "non-positive permittivity has no electrostatic solution"
            )
        values[i] = v
    return values[np.asarray(tet_part, dtype=np.int64)]


def element_stiffness(points: np.ndarray, tets: np.ndarray, eps_tet: np.ndarray):
    """Closed-form 4x4 P1 stiffness per tet, in ``eps0 * m``.

    ``K_ij = eps_static * eps0 * vol * (grad lambda_i . grad lambda_j)``, with
    ``vol`` the absolute tet volume and ``eps_tet`` the ``(n_tets,)`` relative
    permittivity from :func:`eps_by_tet`.

    Returns ``(K_elem, volumes)`` with shapes ``(n_tets, 4, 4)`` and
    ``(n_tets,)``.
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    det_J, grads = mse_fem.tet_geometry(points, tets)
    volumes = np.abs(det_J) / 6.0

    eps_tet = np.asarray(eps_tet, dtype=np.float64).reshape(-1)
    if eps_tet.shape[0] != tets.shape[0]:
        raise ValueError(
            f"eps_tet has {eps_tet.shape[0]} entries for {tets.shape[0]} tets"
        )

    # (n_tets, 4, 4): dot product of every pair of barycentric gradients
    # node indices a,b (axis 1) paired; k is the coordinate axis (2) and is
    # contracted. Writing "nai,naj->nij" reuses i for the node AND leaves j
    # unmatched-free, which einsum resolves to the 3-wide coordinate axis -- a
    # silent (n,3,3) for a 4-node element.
    dots = np.einsum("nak,nbk->nab", grads, grads)
    scale = (eps_tet * EPS0 * volumes).reshape(-1, 1, 1)
    return scale * dots, volumes


def assemble_stiffness(
    points: np.ndarray,
    tets: np.ndarray,
    parts: list[str],
    eps_static: dict[str, float],
    region_of_tet: np.ndarray,
    n_points: int | None = None,
    skip: set[str] | None = None,
):
    """Assemble the global sparse P1 stiffness matrix.

    ``region_of_tet`` is the per-tet part index the MESHER returned (TetGen
    region attributes), not anything derived from node membership -- see
    :func:`check_tet_regions`.

    ``skip`` names parts whose tets are NOT assembled -- the electrodes, whose
    nodes are held at a fixed potential and so contribute a boundary condition
    rather than a stiffness row. Those parts need no ``eps_static``, and their
    tets are left out of the matrix; they contribute ``(0, 0)`` entries, which
    :meth:`~scipy.sparse.csr_matrix.eliminate_zeros` drops.

    Returns ``(K, tet_part)`` with ``K`` a ``(n_points, n_points)`` CSR matrix.

    ``K`` is symmetric and every row sums to zero: a constant potential has no
    flux, which is the discrete statement of the divergence theorem and an
    exact property of this assembly (each ``K_elem`` has that property, and the
    scatter preserves it).
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    skip = set() if skip is None else set(skip)
    missing = [p for p in skip if p not in parts]
    if missing:
        raise ValueError(
            f"skip names part(s) {', '.join(missing)} that are not in this mesh"
        )

    tet_part = check_tet_regions(region_of_tet, parts)
    # Parts not skipped need their eps_static; the skipped ones never enter the
    # matrix, so requiring a permittivity for a held electrode would refuse a
    # device that is perfectly well posed.
    # Pass the caller's table through unchanged: filtering it to the parts
    # before eps_by_tet would silently drop a MISSPELLED key ("coret"), which
    # is exactly the typo that refusal exists to catch.
    eps_tet = eps_by_tet(parts, eps_static, tet_part, allow_missing=skip)
    K_elem, _ = element_stiffness(points, tets, eps_tet)
    if skip:
        # The held tets carry NaN eps and so NaN matrices; zeroing the whole
        # element (rather than only the NaN entries) keeps the sum exact and
        # leaves their nodes' rows to the other tets, which is the boundary
        # condition: an electrode node's value is imposed, not solved.
        held_tets = np.isin(tet_part, [i for i, p in enumerate(parts) if p in skip])
        K_elem[held_tets] = 0.0

    if n_points is None:
        n_points = int(points.shape[0])

    # K_elem.ravel() visits (i, j) with j fastest, so the row index repeats each
    # node 4 times within a tet (repeat along axis 1) and the column index tiles
    # the tet's four nodes (tile along axis 1). Getting these the other way round
    # scatters K to the wrong entries while still producing a symmetric-looking,
    # row-sum-zero matrix on a single-tet mesh -- only the multi-tet checks see it.
    rows = np.repeat(tets, 4, axis=1).ravel()        # (n_tets*16,)
    cols = np.tile(tets, (1, 4)).ravel()
    K = sparse.coo_matrix(
        (K_elem.ravel(), (rows, cols)), shape=(n_points, n_points)
    ).tocsr()
    K.sum_duplicates()
    K.eliminate_zeros()
    return K, tet_part


def electrode_nodes(
    tets: np.ndarray, parts: list[str], tet_part: np.ndarray, electrodes: dict[str, float]
) -> dict[str, np.ndarray]:
    """The node indices each electrode part occupies, validated.

    An electrode is a PART: every node of every tet in that part is held at its
    potential. A node on the boundary between an electrode and the material it
    drives therefore belongs to both, and is counted as electrode -- which is
    what makes the Dirichlet value apply at the contact rather than one element
    away from it.

    Refuses before any matrix work: an electrode naming a part this mesh does
    not have (a typo would otherwise clamp nothing and return a plausible
    field), a non-finite potential, and two electrodes that share a node at
    different potentials (no single value can be imposed there, and picking one
    would silently short them together).
    """
    parts = list(parts)
    unknown = [p for p in electrodes if p not in parts]
    if unknown:
        raise ValueError(
            f"electrode(s) {', '.join(unknown)} are not parts of this mesh; it "
            f"has {', '.join(parts)}"
        )
    for p, v in electrodes.items():
        if not np.isfinite(v):
            raise ValueError(f"electrode {p!r} has a non-finite potential: {v!r}")

    tets = np.asarray(tets, dtype=np.int64)
    tet_part = np.asarray(tet_part, dtype=np.int64)
    by_part: dict[str, np.ndarray] = {}
    for i, p in enumerate(parts):
        if p not in electrodes:
            continue
        nodes = np.unique(tets[tet_part == i])
        if nodes.size == 0:
            raise ValueError(
                f"electrode part {p!r} has no nodes; a contact with no nodes "
                "cannot be held at a potential"
            )
        by_part[p] = nodes

    names = list(by_part)
    for j, q in enumerate(names):
        for r in names[j + 1:]:
            shared = np.intersect1d(by_part[q], by_part[r], assume_unique=True)
            if shared.size and electrodes[q] != electrodes[r]:
                raise ValueError(
                    f"electrodes {q!r} ({electrodes[q]!r} V) and {r!r} "
                    f"({electrodes[r]!r} V) share {shared.size} node(s); two "
                    "potentials cannot be imposed on one node"
                )
    return by_part


def amg_solve(A, b, *, rtol: float = AMG_RTOL, maxiter: int = 500):
    """Solve ``A x = b`` by AMG-preconditioned CG, verifying the residual.

    ``A`` is the SPD reduced stiffness. The residual is recomputed against
    ``A`` and ``b`` AFTER pyamg returns, rather than read off pyamg's own
    ``residuals`` list: that list is relative to pyamg's internal initial
    residual, so it answers "how much did I improve" while this asks "is the
    answer right". They differ whenever the preconditioned system is badly
    scaled, which a permittivity jump of 1:12 makes likely.

    Raises :class:`RuntimeError` if the checked relative residual exceeds
    ``rtol`` -- a half-converged answer is never returned silently.
    """
    try:
        import pyamg
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "the AMG path needs pyamg; install it or stay under "
            f"{AMG_NODE_CROSSOVER} nodes for the direct solve"
        ) from exc

    A = sparse.csr_matrix(A)
    ml = pyamg.smoothed_aggregation_solver(A)
    x = ml.solve(b, tol=rtol, accel="cg", maxiter=maxiter)

    scale = np.linalg.norm(b)
    # A zero right-hand side has a zero exact answer, so there is nothing to
    # converge to; measure the residual against the operator scale instead of
    # dividing by zero.
    denom = scale if scale > 0 else np.linalg.norm(np.asarray(A.diagonal()))
    residual = np.linalg.norm(b - A @ x) / denom if denom > 0 else 0.0
    if not residual <= rtol * AMG_RESIDUAL_SLACK:
        raise RuntimeError(
            f"AMG-CG did not converge: relative residual {residual:.3e} against "
            f"rtol {rtol:.1e} (maxiter {maxiter}). Refusing to return a "
            "half-converged potential."
        )
    return x, float(residual)


class EstimateError(MemoryError):
    """A run was refused before allocating, because its estimate exceeds the budget.

    A ``MemoryError``, because that is what it is standing in for: the same
    outcome the kernel would have produced, raised early and with a number
    attached. It is its own class so a caller can distinguish a refusal it can
    act on (ask for a coarser mesh) from an allocation that genuinely failed.
    """


def _direct_bytes(n_free: int) -> float:
    """Interpolate :data:`DIRECT_MEMORY_TABLE` in log-log space; refuse beyond it.

    Log-log, because both axes span an order of magnitude and the cost between
    two measured points grows geometrically, so a straight line in log space is
    the least-wrong reading available. The estimate is taken on the UPPER-BOUND
    side (``exp`` of the interpolated log), which is why a point sitting exactly
    on a table entry reproduces that entry's measured bytes rather than drifting
    low.

    Below the table's smallest point the cost is the smallest measured rate
    scaled linearly. That is the ONE place this estimate can read low: the rate
    is not monotonic in size -- it falls from 6,148 to 5,502 B/unknown across the
    first two points before climbing to 44,574 B at the top -- so a tiny mesh may
    cost slightly more per unknown than the 6,148 B charged here. The error is
    bounded by that 6,148 vs 5,502 ratio (11 %) and applies only to meshes below
    1859 unknowns, where the absolute figure is tens of MB and the budget check is
    not what decides the outcome. Above the largest point it RAISES:
    extrapolating a curve that fits at 27 % error would produce a confidently
    wrong number, and the caller can act on "too big, use AMG" in a way it
    cannot act on "4.8 GiB".
    """
    table = DIRECT_MEMORY_TABLE
    smallest_n, smallest_b = table[0]
    largest_n, _ = table[-1]
    if n_free > largest_n:
        raise EstimateError(
            f"the direct solve has not been measured above {largest_n:,} free "
            f"unknowns, and this mesh has {n_free:,}. Direct-solve memory grows "
            "faster than any law fitted to the measurements (fill is not linear "
            "in the unknown count), so extrapolating it would be a guess. Use "
            "the AMG path (method='amg'), which has a measured model over a "
            "wider range, or a coarser mesh."
        )
    if n_free <= smallest_n:
        return float(n_free) * (smallest_b / smallest_n)
    # Find the bracketing pair and interpolate log(n) -> log(bytes).
    for (n0, b0), (n1, b1) in zip(table, table[1:]):
        if n_free <= n1:
            t = (math.log(n_free) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return float(math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0))))
    # Unreachable: n_free <= largest_n is guaranteed above.
    raise AssertionError(f"no bracket for n_free={n_free!r}")  # pragma: no cover


def _amg_bytes(n_free: int) -> tuple[float, bool]:
    """The AMG power law, with a stated margin above its measured range.

    Returns ``(bytes, extrapolated)``. ``extrapolated`` is ``True`` above
    :data:`AMG_MEASURED_MAX_UNKNOWNS`, where the margin has been applied and the
    caller must not present the figure as measured.
    """
    law = AMG_MEMORY_POWER_LAW
    raw = law["coefficient"] * float(n_free) ** law["exponent"]
    if n_free > AMG_MEASURED_MAX_UNKNOWNS:
        return raw * AMG_EXTRAPOLATION_MARGIN, True
    return raw, False


def _direct_is_viable(n_free: int) -> bool:
    """Whether the direct path is both chosen AND priceable at ``n_free``.

    Two limits govern direct, and they are not the same number:

    * ``AMG_NODE_CROSSOVER`` (5 x 10**5) is a SPEED choice -- above it AMG is
      expected to win, so ``auto`` switches.
    * ``DIRECT_MEMORY_TABLE``'s largest point (112,847) is a KNOWLEDGE limit --
      direct cost was never measured above it, so it cannot be priced.

    The crossover sits ABOVE the table, so between the two there is a band where
    ``auto`` would pick direct and then be unable to estimate it, refusing a mesh
    that AMG could have solved. Every product-scale mesh (E1 ~2.4e5, E2 ~2.4e5
    free unknowns) lands in that band, so the band is the common case, not an
    edge. ``auto`` must therefore treat "cannot be priced" as a reason to choose
    AMG, and the two limits are consulted together here so they cannot drift.
    """
    return n_free < AMG_NODE_CROSSOVER and n_free <= DIRECT_MEMORY_TABLE[-1][0]


def estimate_memory(n_free: int, *, method: str = "auto",
                    bytes_per_unknown: dict[str, float] | None = None) -> dict:
    """Estimate the peak memory of a solve, in bytes, before it allocates.

    ``n_free`` is the REDUCED unknown count -- the nodes not held at a fixed
    potential -- because those are the only ones factorised. Held electrodes
    cost nothing to solve, which is why a mesh's node count alone is the wrong
    input here.

    ``method`` follows :func:`solve`: ``"auto"`` estimates whichever path
    ``n_free`` would actually take, ``"direct"`` and ``"amg"`` force one.

    The model is MEASURED, on a bounded cluster Job, not fitted here: the direct
    path interpolates :data:`DIRECT_MEMORY_TABLE` (and refuses above it), AMG
    applies :data:`AMG_MEMORY_POWER_LAW` with a margin above its measured range.
    See ``openspec/changes/electrical-simulation/memory-bench.md``.

    Returns ``bytes``, the path estimated (``path``), the per-unknown rate, and
    two flags a caller must not flatten: ``measured`` (the figure derives from a
    measurement of this path) and ``extrapolated`` (the AMG law was applied
    beyond its measured range, with :data:`AMG_EXTRAPOLATION_MARGIN` applied).
    A refusal message should say which it is.

    ``bytes_per_unknown`` is a test and experiment seam that overrides the model
    with a flat rate, reproducing the pre-1.7 behaviour. It sets
    ``measured=False``, because a caller-supplied rate is not a measurement.
    """
    if n_free < 0:
        raise ValueError(f"n_free must not be negative, got {n_free!r}")
    if method not in METHODS:
        raise ValueError(
            f"method={method!r} is not a known solve method; use one of "
            f"{', '.join(METHODS)}"
        )
    path = "amg" if (method == "amg" or (method == "auto" and not _direct_is_viable(n_free))) else "direct"
    if bytes_per_unknown is not None:
        if path not in bytes_per_unknown:
            raise ValueError(
                f"no per-unknown rate for the {path!r} path; expected keys "
                f"{sorted(bytes_per_unknown)}"
            )
        rate = float(bytes_per_unknown[path])
        return {
            "bytes": float(n_free) * rate,
            "path": path,
            "bytes_per_unknown": rate,
            "measured": False,
            "extrapolated": False,
            "n_free": int(n_free),
        }
    if path == "amg":
        total, extrapolated = _amg_bytes(n_free)
    else:
        total, extrapolated = _direct_bytes(n_free), False
    rate = total / n_free if n_free else 0.0
    return {
        "bytes": float(total),
        "path": path,
        "bytes_per_unknown": rate,
        "measured": True,
        "extrapolated": extrapolated,
        "n_free": int(n_free),
    }


def check_budget(estimate: dict, *, budget=None) -> dict:
    """Refuse an ``estimate`` that exceeds ``budget``, naming both figures.

    ``budget`` is the memory this run may use, in bytes, and follows
    :func:`fdtd.check_limits` -- the time-domain path solved this same problem
    first and there is no reason for electrostatics to answer it differently:

    * ``None`` reads :func:`fdtd.admission_budget` of THIS process at call time,
      so the figure follows the cgroup limit or the device the caller is on.
    * :data:`NO_DEVICE` says the caller is not on the run's device, so nothing is
      read and the estimate is not checked. The web app asks on the control
      plane while the solve runs in a Job, and a figure read from the control
      plane would be the wrong machine's -- so that caller passes ``NO_DEVICE``
      rather than relying on a ``None`` that would silently read the wrong one.
    * A number checks against exactly that.

    A refusal on an unread device is impossible by construction here, which is
    why ``NO_DEVICE`` is the honest answer for a caller that cannot know: it
    says so, instead of guessing a ceiling from the wrong host.

    Raises :class:`EstimateError` with the estimate, the budget and the mesh
    setting that would bring the run under it -- a refusal a person can act on
    rather than a number they have to go and compare themselves.
    """
    if budget == NO_DEVICE:
        return estimate
    if budget is None:
        budget = admission_budget()["bytes"]
    budget = float(budget)
    if budget <= 0:
        raise ValueError(
            f"budget must be positive, got {budget!r}; pass NO_DEVICE to skip "
            "the check when the caller is not on the run's device"
        )
    if estimate["bytes"] <= budget:
        return estimate
    gb = 2**30
    if estimate.get("extrapolated"):
        kind = "extrapolated, with a 1.5x margin applied"
    elif estimate.get("measured"):
        kind = "measured"
    else:
        kind = "estimated"
    raise EstimateError(
        f"this solve needs about {estimate['bytes'] / gb:.2f} GiB "
        f"({kind}: {estimate['n_free']:,} free unknowns x "
        f"{estimate['bytes_per_unknown'] / 1e3:.2f} kB, {estimate['path']} path) "
        f"against a budget of {budget / gb:.2f} GiB. Refusing before allocating. "
        "Use a coarser mesh, a smaller domain, or a larger budget."
    )


def solve_memory(n_free: int, *, method: str = "auto", budget=None,
                 bytes_per_unknown: dict[str, float] | None = None) -> dict:
    """Estimate, then refuse if over ``budget``. What :func:`solve` calls."""
    estimate = estimate_memory(n_free, method=method,
                               bytes_per_unknown=bytes_per_unknown)
    return check_budget(estimate, budget=budget)


def solve(
    points_m: np.ndarray,
    tets: np.ndarray,
    parts: list[str],
    region_of_tet: np.ndarray,
    eps_static: dict[str, float],
    electrodes: dict[str, float],
    *,
    backend: str | None = None,
    method: str = "auto",
    budget=None,
):
    """Solve the electrostatic problem with the electrodes held at fixed volts.

    ``electrodes`` maps a part label to its potential in volts; every node of
    that part is a Dirichlet value. The remaining ("free") nodes are solved from
    the reduced system ``K_ff phi_f = -K_fe phi_e``, which is what imposing
    ``phi_e`` on the full system amounts to.

    A zero-normal-field (natural) condition holds on the outer boundary -- it is
    the boundary term's absence, so nothing is imposed there.

    ``method`` picks the solver for the reduced system:

    ``"auto"`` (default)  direct below :data:`AMG_NODE_CROSSOVER` nodes, AMG-CG above
    ``"direct"``          always the factorisation -- ignores the crossover
    ``"amg"``             always AMG-CG, whatever the mesh size

    ``"direct"`` exists because the crossover is a NAMED ESTIMATE and the only
    way to prove the two paths agree EITHER SIDE of it is to ask for each one on
    a mesh small enough for tier-1 tests. Without it the agreement test could
    only ever compare AMG against a direct answer on meshes above 5 x 10**5 nodes,
    which no tier-1 test can build.

    ``budget`` is the memory this run may use, in bytes; ``None`` does not check
    it. The estimate is made from the FREE node count and compared BEFORE any
    assembly, so a run that cannot fit is refused rather than dying part-way
    through allocating. It is not read from this process: see
    :func:`check_budget`.

    Returns a dict with ``phi`` (the ``(n_points,)`` nodal potential in volts),
    ``free`` (the boolean mask of solved nodes), ``electrodes`` (the part ->
    node indices mapping used), ``backend``, ``n_free``, ``method`` (the path
    actually taken) and ``residual`` (the AMG path's checked relative residual,
    or ``None`` for the direct one, which is exact).

    Refusals, all ``ValueError`` before factorising: the ones in
    :func:`electrode_nodes` and :func:`eps_by_tet`, plus an empty or unknown
    electrode set and an ``eps_static`` < 1 -- a relative permittivity below
    vacuum is not a material, and it is the shape of an index accidentally
    passed where a permittivity belongs.
    """
    points_m = np.asarray(points_m, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    n_points = int(points_m.shape[0])

    if method not in METHODS:
        raise ValueError(
            f"method={method!r} is not a known solve method; use one of "
            f"{', '.join(METHODS)}"
        )
    if not electrodes:
        raise ValueError(
            "no electrode: the problem needs at least one part held at a "
            "potential, otherwise the solution is arbitrary up to a constant "
            "and the reduced system is singular"
        )
    # Validate the contacts BEFORE any permittivity or matrix work, so every
    # electrode problem is reported as itself. Otherwise a NaN voltage is
    # reported as the gap's missing eps_static, which points at the wrong line.
    unknown_electrodes = [p for p in electrodes if p not in parts]
    if unknown_electrodes:
        raise ValueError(
            f"electrode(s) {', '.join(unknown_electrodes)} are not parts of this "
            f"mesh; it has {', '.join(parts)}"
        )
    for p, v in electrodes.items():
        if not np.isfinite(v):
            raise ValueError(f"electrode {p!r} has a non-finite potential: {v!r}")
    # An electrode is held, not assembled, so its own eps_static is not needed.
    # Every OTHER part is, which is what eps_by_tet below checks.
    assembled = {p: v for p, v in eps_static.items() if p not in electrodes}
    for p, v in assembled.items():
        if float(v) < 1.0:
            raise ValueError(
                f"eps_static for part {p!r} is {v!r}; a relative permittivity "
                "below 1 is not a material (is an optical index here?)"
            )

    tet_part = check_tet_regions(region_of_tet, parts)
    # The memory check comes BEFORE the assembly below, not before the
    # factorisation. Assembly is the first big allocation, so a check placed
    # later would spend the memory it is meant to protect.
    held = electrode_nodes(tets, parts, tet_part, electrodes)
    n_free = n_points - sum(len(nodes) for nodes in held.values())
    memory = solve_memory(n_free, method=method, budget=budget)

    K, _ = assemble_stiffness(points_m, tets, parts, eps_static, tet_part,
                              n_points=n_points, skip=set(electrodes))

    phi_e = np.zeros(n_points, dtype=np.float64)
    fixed = np.zeros(n_points, dtype=bool)
    for part, nodes in held.items():
        phi_e[nodes] = electrodes[part]
        fixed[nodes] = True
    free = ~fixed

    phi = phi_e.copy()
    residual = None
    solved_by = "direct"
    if free.any():
        # K_fe phi_e: the held values' pull on the free rows.
        rhs = -(K[free][:, fixed] @ phi_e[fixed])
        K_ff = K[free][:, free].tocsc()
        use_amg = method == "amg" or (method == "auto" and not _direct_is_viable(K_ff.shape[0]))
        if use_amg:
            phi[free], residual = amg_solve(K_ff, np.asarray(rhs).ravel())
            solved_by = "amg"
        else:
            lu = BACKEND.factor(K_ff, label="K_ff", spd_lu=SPD_LU, backend=backend)
            phi[free] = np.asarray(lu.solve(np.asarray(rhs).ravel()), dtype=np.float64)
            solved_by = "direct"


    return {
        "phi": phi,
        "free": free,
        "electrodes": held,
        "backend": backend or BACKEND.selected_backend(),
        "n_free": int(free.sum()),
        "method": solved_by,
        "residual": residual,
        # The estimate this run was checked against, so a caller can record what
        # a successful run cost rather than only the ones that were refused.
        "memory_estimate": memory,
        # The mesh and the inputs :func:`field_at` and :func:`_electrode_charge`
        # need. Carried here rather than passed alongside, because a solution
        # read with a DIFFERENT mesh is a silent wrong answer (the potential is
        # indexed by node, so a reordered mesh maps voltages to the wrong nodes).
        "points_m": points_m,
        "tets": tets,
        "parts": list(parts),
        "region_of_tet": tet_part,
        "eps_static": dict(eps_static),
        "tet_part_name": [parts[int(i)] for i in tet_part],
    }




def _barycentric(points: np.ndarray, tets: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of ``p`` in every tet, ``(n_tets, 4)``.

    ``lambda_a = p . grad(lambda_a) + c_a``, and ``grad(lambda_a)`` already
    satisfies ``sum_a lambda_a = 1``, so the affine constant ``c_a`` is
    ``delta_a0 - p0 . grad(lambda_a)`` for any node ``p0`` of the tet -- node 0
    is as good as any and costs one row.
    """
    _, grads = mse_fem.tet_geometry(points, tets)
    # grads is (n_tets, 4, 3), one gradient per SHAPE FUNCTION (lambda_a) per
    # tet. transposing to (n_tets, 3, 4) is what contracts the coordinate axis
    # with ``p`` and leaves the four lambda values: without it the einsum above
    # silently broadcast the tet axis into the node axis and returned a
    # (n_tets, n_tets, 4) array. ``c`` is derived the same way, then the two are
    # summed with ``p`` on the coordinate axis only.
    # grads is (n_tets, 4, 3): contract the COORDINATE axis with p0 and the
    # result is (n_tets, 4), one constant per shape function. It previously
    # contracted against ``grads.transpose(0, 2, 1)``, whose rows are indexed by
    # shape function and columns by coordinate, so p0 (3 long) broadcast against
    # the shape-function axis and produced (n_tets, n_tets, 4) -- a tet-by-tet
    # matrix that no point was ever tested against.
    c = np.array([1.0, 0.0, 0.0, 0.0]) - np.einsum(
        "nk,nak->na", points[tets][:, 0, :], grads)
    return np.einsum("nak,k->na", grads, p) + c


def field_at(solution: dict, point_m) -> np.ndarray:
    """E = -grad(phi) in V/m at ``point_m``, in the tet that contains it.

    ``phi`` is piecewise linear on the mesh, so ``grad(phi)`` is constant per
    element: ``sum_a phi_a grad(lambda_a)`` over the containing tet. No
    smoothing is done and none should be -- a caller that wants a field
    across a material interface is asking for two values, and gets the one on
    the side of the point it asked about.

    Refuses a point outside the mesh and a point inside an electrode part. The
    second is not pedantry: ``phi`` there is the imposed constant, so the
    piecewise-linear ``grad`` is 0 and the caller would receive E = 0 with no
    sign that the field was merely not computed. The electrode boundary layer
    is exactly where a device's field lives, so that silent zero is the most
    misleading answer this function could give.

    ``solution`` is a :func:`solve` result; ``tets`` and ``points_m`` are read
    from it, which is why ``solve`` carries them.
    """
    p = np.asarray(point_m, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"point_m must be a 3-vector, got shape {np.shape(point_m)}")

    points = solution["points_m"]
    tets = solution["tets"]
    lam = _barycentric(points, tets, p)
    inside = np.all(lam >= -BARY_EPS, axis=1)
    if not inside.any():
        raise ValueError(
            f"point {p.tolist()} is outside this mesh; "
            f"field_at only samples where the solution exists"
        )
    # A point ON an electrode boundary lies in several tets at once -- some
    # electrode, some not -- so taking the lowest-index containing tet made the
    # answer depend on the mesh's numbering: a point on the gap-side face of an
    # electrode was refused about half the time, and the two answers differed
    # in kind (a field vs an exception), not just in value. Prefer a
    # non-electrode tet; refuse only when EVERY containing tet is an electrode,
    # which is the honest reading of "strictly inside the electrode".
    containing = np.flatnonzero(inside)
    tet_part_name = solution["tet_part_name"]   # a LIST, not an array
    is_electrode = np.array([tet_part_name[int(t)] in solution["electrodes"]
                             for t in containing], dtype=bool)
    non_electrode = containing[~is_electrode]
    if non_electrode.size == 0:
        raise ValueError(
            f"point {p.tolist()} is inside electrode "
            f"{tet_part_name[int(containing[0])]!r}, where the potential is held constant; the field "
            f"there is not computed, so returning 0 would be a zero dressed up "
            f"as an answer"
        )
    tet = int(non_electrode[0])

    _, grads = mse_fem.tet_geometry(points, tets[tet:tet + 1])
    return -(solution["phi"][tets[tet]] @ grads[0])


def capacitance_matrix(
    points_m: np.ndarray,
    tets: np.ndarray,
    parts: list[str],
    region_of_tet: np.ndarray,
    eps_static: dict[str, float],
    electrodes: dict[str, float],
    *,
    backend: str | None = None,
    method: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    """The Maxwell capacitance matrix in farads, and the electrode order.

    ``C_ij = dQ_i / dV_j``: each electrode's charge is read from the assembled
    residual ``K phi`` restricted to that electrode's nodes, with electrode
    ``j`` at 1 V and every other at 0 V.

    **Read from the residual, not from the surface.** Summing ``K phi`` over a
    disjoint set of NODES gives exactly the flux a Gaussian surface around that
    electrode would, because every assembled row sums to zero. It is neither
    easier nor harder than an edge-based flux, but it is the definition the
    spec names and it keeps the answer tied to the matrix that was actually
    solved.

    The matrix is returned in the order of ``electrodes``. Rows are NOT
    normalised by sign: with ``K phi`` built for ``phi_j = 1``, a row's
    self-entry is positive and its couplings negative, which is the physical
    Maxwell matrix. Symmetry of the returned matrix is a consequence, not an
    assumption -- :func:`solve` runs ``K_ff`` SPD, and the same ``K`` is used
    for every driven electrode, so ``C`` is symmetric in exact arithmetic and
    the tests measure how far from it we are.
    """
    names = list(electrodes)
    if len(names) < 1:
        raise ValueError("capacitance_matrix needs at least one electrode")
    columns = []
    for driven in names:
        volts = {name: (1.0 if name == driven else 0.0) for name in names}
        result = solve(points_m, tets, parts, region_of_tet, eps_static, volts,
                       backend=backend, method=method)
        columns.append(_electrode_charge(result))
    return assemble_capacitance(columns, names), names


def assemble_capacitance(columns: list[dict[str, float]],
                         names: list[str]) -> np.ndarray:
    """Build ``C`` from one charge map per driven electrode.

    Split out from :func:`capacitance_matrix` because the ORIENTATION is not
    observable from any solve this module can run: every fixture these tests
    build is reciprocal, so ``C`` comes out exactly symmetric (measured max
    ``|C - C^T|`` of 0.0) and writing the drive into the row index instead of
    the column produces a numerically identical matrix. Transposing the write
    survived the whole test suite four times before this function existed.

    With the assembly separate, the convention is testable directly: a drive
    map that is deliberately NOT symmetric pins which index the drive occupies,
    and nothing about float symmetry is involved.
    """
    if len(columns) != len(names):
        raise ValueError(
            f"assemble_capacitance needs one charge map per electrode: "
            f"{len(columns)} maps for {len(names)} electrodes"
        )
    n = len(names)
    C = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        charges = columns[j]
        # ``charges`` is already Maxwell-signed (see _electrode_charge). The
        # drive, j, belongs in the COLUMN: C[i, j] is the charge on electrode i
        # when j is driven.
        for i, name in enumerate(names):
            C[i, j] = charges[name]
    return C


def _electrode_charge(solution: dict) -> dict[str, float]:
    """``(K phi)`` summed over each electrode's nodes, in coulombs.

    The sign, from the DERIVATION: ``K_ab = integral(eps0 * eps * grad(lam_a)
    . grad(lam_b))`` is the weak form of ``-div(eps0 * eps * grad(phi))``, and
    integrate-by-parts moves the divergence onto the test function, leaving a
    boundary flux with a MINUS sign. So ``(K phi)_a`` is minus the outward flux
    through that node's control volume, i.e. PLUS the charge stored on it. In
    the driven-at-1 V convention that makes the driven electrode's own entry
    positive and its neighbours' negative, which is a capacitor.

    Corroborated, not established, by measurement: on the three-plate stack
    with p0 driven at 1 V, ``p0`` sums to +2.83e-13 C and its neighbour to
    -2.83e-13 C, which is a positive electrode and its induced
    negative partner. An earlier version of this function negated that sum and
    its docstring asserted the negation was the charge; both were wrong by a
    global sign, which every symmetric fixture hides (``-C`` is as symmetric as
    ``C``) and only the asymmetric one exposed.

    The FULL assembled matrix is used, not ``K_ff``: the electrode rows are the
    ones carrying the charge, so the assembly here deliberately does not skip
    them. (``solve`` skips them because they are Dirichlet rows there; a
    residual read needs them, and re-assembling is cheap next to the solve.)
    """
    K, _ = assemble_stiffness(
        solution["points_m"], solution["tets"], solution["parts"],
        solution["eps_static"], solution["region_of_tet"],
        n_points=solution["phi"].shape[0],
    )
    residual = K @ solution["phi"]
    return {name: float(residual[nodes].sum())
            for name, nodes in solution["electrodes"].items()}
