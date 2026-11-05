"""Effective-mass quantum-confined states on a Cartesian box (task 1.8).

This is the half of the Stark calculation that turns a solved electrostatic
field into an energy shift: the confined-carrier eigenproblem, given the
particle's effective mass ``m*``, a confining box, and the potential energy
``V = -e * phi`` of the electrostatic solution.

The operator is the effective-mass Schr"odinger equation, discretised on a
uniform Cartesian grid with Dirichlet walls, ::

    H psi = [ -(hbar**2 / 2) * grad . ( (1/m*) grad ) + V ] psi = E psi

with Dirichlet walls because the box is the confining potential: an infinite
well is the model for a dot whose barrier is high enough that no carrier leaks.
``m*`` may be a single number (a uniform effective mass) or an array on the
grid, which is why it sits INSIDE the divergence rather than in front of the
Laplacian -- the two differ at a mass discontinuity and the Hermitian form is
the one that is physically right there. [Bastard, *Wave Mechanics Applied to
Semiconductor Heterostructures*, Les Editions de Physique 1988, ch. II-III, for
the effective-mass approximation and its boundary condition.]

Nothing here is specific to paper 1. ``m*``, the box and the permittivity are
plain inputs; there are no paper-1 defaults baked in. The dot's material values
are a separate, still-open question.

Discretisation
--------------
On a uniform grid with spacing ``h`` and index ``(i, j, k)``, the 7-point form
of ``-grad . (b grad)`` with ``b = 1/m*`` evaluated on the faces (the natural
choice, and the one that keeps ``H`` symmetric) is::

    (H psi)[i,j,k] = sum_over_6_faces  b_face * (psi[i,j,k] - psi[nbr]) / h**2

where ``b_face`` is the arithmetic mean of the two cell values of ``b``. With
uniform ``b`` this reduces to ``b * (6 psi - sum of 6 neighbours) / h**2``, the
familiar 7-point Laplacian, and the truncation error is ``O(h**2)``.

Units
-----
Inputs are in eV and nanometres because that is what a device is specified in,
and converted once, at the boundary, to SI. ``hbar**2 / (2 m_e)`` in eV nm**2 is
:data:`HBAR2_OVER_2ME_EV_NM2`; a particle of effective mass ``m* = r m_e`` has
``hbar**2 / (2 m*) = HBAR2_OVER_2ME_EV_NM2 / r``. Working in eV nm**2 keeps the
numbers near 1 instead of near 1e-20, which is where unit errors happen.

What this module does NOT do
----------------------------
- No exciton. The first-order Coulomb term, the binding energy and the p/beta
  fit are task 1.9.
- No field of its own. It consumes a potential from a tet solve, interpolated
  by :func:`interpolate_potential`. Task 1.8's requirement is that this
  interpolation be exact on a known-linear potential, which is what
  :mod:`tests.test_stark` checks: the tet solve's ``phi`` is piecewise linear,
  so interpolating it must reproduce the field to rounding.
- No tolerance-only test. Both analytic cases (harmonic oscillator and Airy)
  prove the convergence RATE, ``O(h**2)``, by measuring the order under
  refinement. A single tolerance would pass for a first-order scheme with a
  small constant, which is the failure this module exists to exclude.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mse_fem

#: Tolerance on the barycentric coordinates when deciding whether a point lies
#: inside a tet. A P1 mesh is conforming, so an interior node of a face has
#: lambda = 0 there and would be rejected by a strict ``>= 0``; this allowance
#: admits the boundary and nothing meaningfully outside it.
BARY_EPS = 1e-10

#: A tet is treated as degenerate (and skipped when locating a point) when its
#: signed volume is below this fraction of the mesh's MEDIAN |volume|. Relative,
#: so it means the same thing however the caller spells its unit. Slivers with
#: det near this share the "huge gradient, lambda plausible everywhere" failure
#: documented in :func:`interpolate_potential`; excluding them is what lets the
#: outside-the-mesh guard actually fire.
DEGENERATE_DET_FRACTION = 1e-9

#: ``hbar**2 / (2 m_e)`` in eV nm**2. CODATA 2018: hbar = 1.054571817e-34 J s,
#: m_e = 9.1093837015e-31 kg, eV = 1.602176634e-19 J. The combination is
#: ``(1.054571817e-34)**2 / (2 * 9.1093837015e-31) / 1.602176634e-19 * 1e18``
#: nm**2, = 0.0380998 eV nm**2. Computed here, not typed from memory, so it
#: carries its own derivation.
HBAR2_OVER_2ME_EV_NM2 = (
    (1.054571817e-34 ** 2) / (2.0 * 9.1093837015e-31) / 1.602176634e-19 * 1e18
)

#: Electron charge magnitude in coulombs, CODATA 2018.
E_CHARGE_C = 1.602176634e-19

#: First zero of the Airy function Ai(-x), i.e. the ground state of a particle
#: in a linear potential against a hard wall. The value every textbook quotes:
#: |a1| = 2.33811 (Abramowitz & Stegun 10.4.94, and DLMF 9.9.1). Checked
#: against the analytic series elsewhere rather than trusted alone.
AIRY_A1 = 2.338107410459767


class StarkError(ValueError):
    """A setup or input this module refuses rather than mis-solving."""


class Box(NamedTuple):
    """A Cartesian box in nanometres: the confining region.

    ``origin`` is the low corner, ``size`` the extent along x, y, z. The grid
    is ``n = (nx, ny, nz)`` CELLS, so there are ``n + 1`` nodes per axis and the
    boundary nodes are the Dirichlet walls.
    """

    origin: tuple[float, float, float]
    size: tuple[float, float, float]
    n: tuple[int, int, int]

    def axes(self):
        """The three node coordinate vectors, in nm."""
        out = []
        for o, s, m in zip(self.origin, self.size, self.n):
            if m < 1:
                raise StarkError(f"box needs at least 1 cell per axis, got {self.n}")
            out.append(np.linspace(o, o + s, m + 1))
        return out

    def h(self):
        """The grid spacing per axis, in nm. Uniform by construction."""
        return tuple(s / m for s, m in zip(self.size, self.n))

    def n_nodes(self):
        return (self.n[0] + 1) * (self.n[1] + 1) * (self.n[2] + 1)


def grid_index(shape: tuple[int, int, int]):
    """The node index arrays ``(I, J, K)`` for a box of shape ``shape``.

    ``shape`` is the number of NODES per axis. The flattened node id is the
    C-order ravel of ``(I, J, K)``, which is what :func:`build_hamiltonian`
    uses; any code that reshapes a solution must use the same convention.
    """
    return np.meshgrid(*[np.arange(m) for m in shape], indexing="ij")


def build_laplacian_eV_nm2(box: Box, mass_ratio):
    """Assemble ``-grad . ( (hbar**2 / 2 m*) grad )`` on the box, in eV nm**-2.

    ``mass_ratio`` is ``m* / m_e``: a positive scalar, or an array of node
    values (shape ``box.n_nodes()`` already flattened, or broadcastable to the
    box's node grid). It sits inside the divergence as a face-averaged
    coefficient, which is the Hermitian discretisation and the correct one at a
    mass step.

    Dirichlet walls: boundary nodes keep a diagonal of 1 and no off-diagonal
    coupling, so their row is ``psi = 0`` and the interior is solved with them
    eliminated. That is the infinite-well boundary condition, and it is applied
    here rather than by masking after the fact so the operator handed to the
    eigensolver is already the reduced one.

    Returns a ``(n_nodes, n_nodes)`` ``csr_matrix`` in eV nm**-2, i.e. the
    kinetic energy with lengths measured in nm.
    """
    nx, ny, nz = (m + 1 for m in box.n)
    hx, hy, hz = box.h()
    shape = (nx, ny, nz)

    I, J, K = grid_index(shape)
    flat = np.arange(nx * ny * nz).reshape(shape)

    b = np.asarray(mass_ratio, dtype=np.float64)
    if b.ndim == 0:
        b_grid = np.full(shape, float(b))
    else:
        b_grid = b.reshape(shape) if b.size == nx * ny * nz else np.broadcast_to(b, shape)
    if not np.all(np.isfinite(b_grid)):
        raise StarkError("mass_ratio contains non-finite values")
    if np.any(b_grid <= 0):
        raise StarkError(
            "mass_ratio must be positive everywhere; a zero or negative "
            "effective mass is not a particle"
        )

    # The prefactor: hbar^2/(2 m*) = HBAR2_OVER_2ME / (m*/m_e).
    coeff = HBAR2_OVER_2ME_EV_NM2 / b_grid  # eV nm**2 per node

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    # Interior mask: a node is interior if it is not on any wall.
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = True
    interior_idx = flat[interior]

    diag = np.zeros(shape, dtype=np.float64)

    # Each axis contributes two couplings per interior node: to the -1 and +1
    # neighbour along that axis. Slices are built once per axis with explicit
    # extent arithmetic rather than by juggling slice objects, because the
    # face array (length n-1) and the interior range (length n-2) differ and
    # that mismatch is what a broadcast error reports.
    for axis, (h, n) in enumerate(zip((hx, hy, hz), (nx, ny, nz))):
        h2 = h * h

        def _sl(lo: int, hi: int):
            s = [slice(None)] * 3
            s[axis] = slice(lo, hi)
            return tuple(s)

        # Face coefficient b_face[..., f, ...] between nodes f and f+1: the mean
        # of the two cell values of b, which is what keeps H symmetric.
        b_face = 0.5 * (b_grid[_sl(0, n - 1)] + b_grid[_sl(1, n)])
        # Hopping magnitude across that face, eV nm**-2.
        t_face = (HBAR2_OVER_2ME_EV_NM2 / b_face) / h2

        # dst is the interior range along this axis (nodes 1 .. n-1, n-1 of them).
        dst_sl = _sl(1, n - 1)
        dst_ids = flat[dst_sl]
        # Coupling to the -1 neighbour crosses face f = 0 .. n-2 (first n-1? no:
        # node i couples across the face between i-1 and i, which is face i-1).
        # For interior nodes i = 1 .. n-2: the -1 face is i-1 = 0 .. n-3, and
        # the +1 face is i = 1 .. n-2. Both have n-2 entries, matching dst.
        t_minus = t_face[_sl(0, n - 2)]
        t_plus = t_face[_sl(1, n - 1)]
        src_minus = flat[_sl(0, n - 2)]
        src_plus = flat[_sl(2, n)]

        rows.append(dst_ids.ravel())
        cols.append(src_minus.ravel())
        vals.append((-t_minus).ravel())

        rows.append(dst_ids.ravel())
        cols.append(src_plus.ravel())
        vals.append((-t_plus).ravel())

        # Diagonal accumulates the two hoppings. The sign is POSITIVE: the
        # 7-point form of -grad.(b grad) is diagonal +sum(t) and off-diagonal
        # -t, which on a chain gives eigenvalues 0 .. +4t. Getting this the
        # other way round yields a spectrum that is entirely NEGATIVE for a
        # positive-definite operator, which is how the sign error was caught.
        contrib = t_minus + t_plus
        diag_sl = dst_sl
        diag[diag_sl] = diag[diag_sl] + contrib

    # Assemble the interior block directly on the reduced index space. Boundary
    # nodes are eliminated at assembly rather than zeroed afterwards: with
    # Dirichlet walls psi = 0 there, the wall rows impose nothing on the
    # interior, and a hop that lands on a wall leaves only its diagonal
    # contribution (the -t already accumulated above), which is precisely the
    # elimination.
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    d = diag[interior]

    pos = np.full(nx * ny * nz, -1, dtype=np.int64)
    pos[interior_idx] = np.arange(interior_idx.size)
    keep = (pos[r] >= 0) & (pos[c] >= 0)
    H_ii = sp.coo_matrix(
        (v[keep], (pos[r[keep]], pos[c[keep]])), shape=(interior_idx.size,) * 2
    ).tocsr()
    H_ii = H_ii + sp.diags(d)
    return H_ii, interior_idx, flat


def build_hamiltonian(box: Box, mass_ratio, potential_eV=None):
    """The full confined-carrier Hamiltonian on the box's interior nodes.

    ``H = T + V`` with ``T`` from :func:`build_laplacian_eV_nm2` and ``V`` the
    potential energy on every node, restricted to the interior. ``potential_eV``
    is either ``None`` (zero potential) or an array of node values, shape
    ``box.n_nodes()`` or broadcastable to the node grid.

    Returns ``(H, interior_idx, flat)`` where ``H`` is ``(n_int, n_int)`` CSR,
    ``interior_idx`` are the flat indices of the interior nodes and ``flat`` is
    the full node index grid.
    """
    T, interior_idx, flat = build_laplacian_eV_nm2(box, mass_ratio)
    if potential_eV is None:
        return T, interior_idx, flat
    shape = (box.n[0] + 1, box.n[1] + 1, box.n[2] + 1)
    V = np.asarray(potential_eV, dtype=np.float64)
    if V.size != box.n_nodes():
        raise StarkError(
            f"potential_eV has {V.size} values, the box has {box.n_nodes()} nodes"
        )
    V_flat = V.reshape(-1)
    return T + sp.diags(V_flat[interior_idx]), interior_idx, flat


def lowest_states(box: Box, mass_ratio, *, n_states: int = 4,
                  potential_eV=None, tol: float = 1e-10, seed: int = 0,
                  return_converged: bool = False, maxiter: int = 2000,
                  residual_tol: float = 1e-6):
    """The lowest ``n_states`` eigenpairs of the confined-carrier Hamiltonian.

    Uses LOBPCG with a fixed seed, so a re-run reproduces the same eigenvalues
    to rounding. LOBPCG is a BLOCK method: it asks for a starting guess, and
    with a fixed seed the guess is deterministic, which is what makes a
    convergence test reproducible rather than flaky.

    The Hamiltonian is positive semi-definite (kinetic) plus whatever the
    potential does, so for a potential with a large negative floor a shift can
    help; none is applied by default because both analytic checks here have
    bounded-below spectra near zero.

    Returns ``(energies_eV, vectors)`` with ``vectors`` of shape
    ``(n_nodes, n_states)``, zero on the walls; that is the full-box shape,
    which is what a caller plotting a state wants.

    CONVERGENCE IS CHECKED, NOT ASSUMED. ``scipy.sparse.linalg.lobpcg`` does
    not report whether it met its tolerance in its return value: on failure it
    only emits a ``UserWarning`` and hands back whatever it has. A caller that
    ignores warnings therefore gets an unconverged eigenvalue with nothing to
    distinguish it from a converged one. Worse, LOBPCG's own stopping rule is
    an estimate, so "no warning" is not proof either. This function instead
    measures the true residual ``||H v - E v|| / (|E| ||v||)`` for every pair
    and compares it against ``residual_tol``.

    ``tol`` and ``residual_tol`` are different quantities and are not
    interchangeable: ``tol`` is LOBPCG's internal stopping estimate, whereas
    ``residual_tol`` bounds the residual that is actually reported here. The
    measured residual on these matrices floors around ``5e-10`` and does not
    improve with more iterations, so binding the check to ``tol`` would reject
    correct solves. ``residual_tol`` defaults to ``1e-6``: far tighter than
    any physics question here needs, and far looser than an unconverged solve
    (which lands orders of magnitude above it).

    A pair that fails that check raises :class:`StarkError` by default. Pass
    ``return_converged=True`` to receive ``(energies_eV, vectors, converged)``
    instead, where ``converged`` is a boolean array; that form is for callers
    that want to report the failure rather than die on it. The check is the
    authority in both cases: an unconverged pair is never returned as if it
    were fine.

    Pass a larger ``maxiter`` (via ``lowest_states(..., maxiter=...)``) when a
    residual failure is reported; the failure is a statement about the solve,
    not about the physics.
    """
    H, interior_idx, flat = build_hamiltonian(box, mass_ratio, potential_eV)
    m = interior_idx.size
    k = int(n_states)
    if k < 1:
        raise StarkError(f"n_states must be >= 1, got {n_states}")
    if k >= m:
        raise StarkError(
            f"asked for {k} states from an interior of {m} nodes; "
            f"need fewer states than unknown nodes"
        )

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, k))

    with warnings.catch_warnings(record=True) as caught:
        # lobpcg reports a missed tolerance only through a UserWarning, and
        # it has several differently-worded forms ("Exited at iteration",
        # "Exited postprocessing", ...). Recording rather than filtering is
        # deliberate: scipy manipulates the filter state itself, so an
        # `ignore` filter is not reliably honoured. The warnings are dropped
        # on purpose; the measured residual below is the authority, and it is
        # what is reported to the caller.
        warnings.simplefilter("always")
        evals, evecs = spla.lobpcg(H, X, largest=False, tol=tol,
                                   maxiter=maxiter)
    del caught

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # True residual per pair. LOBPCG's stopping estimate is not trustworthy on
    # its own (it can report "converged" to an estimate that is wrong), so
    # this measured quantity decides.
    converged = np.zeros(k, dtype=bool)
    for i in range(k):
        v = evecs[:, i]
        nrm = float(np.linalg.norm(v))
        if nrm == 0.0:
            continue
        r = H @ v - evals[i] * v
        scale = abs(float(evals[i])) * nrm
        # A near-zero eigenvalue makes the relative test meaningless; fall
        # back to the absolute residual against ||H|| ~ the row scale.
        if scale <= 0.0:
            scale = nrm * float(abs(H).sum(axis=1).max())
        converged[i] = (float(np.linalg.norm(r)) / scale) <= residual_tol
    if not converged.all() and not return_converged:
        bad = ", ".join(f"{evals[i]:.8g}" for i in np.flatnonzero(~converged))
        raise StarkError(
            f"lowest_states did not converge for {int((~converged).sum())} of "
            f"{k} eigenpairs (energies {bad}): the measured residual exceeds "
            f"{residual_tol:g} within maxiter={maxiter}; the returned values "
            f"would be unconverged. "
            f"Refusing to hand them back silently. Retry with a larger "
            f"maxiter, or pass return_converged=True to inspect them."
        )

    # Normalise: LOBPCG returns mass-orthonormal vectors for the identity
    # metric only up to its tolerance, so fix the sign and the norm explicitly.
    n_nodes = flat.size
    full = np.zeros((n_nodes, k), dtype=np.float64)
    full[interior_idx, :] = evecs
    for i in range(k):
        v = full[:, i]
        nrm = math.sqrt(float(np.dot(v, v)))
        if nrm == 0:
            raise StarkError(f"eigenvector {i} came back identically zero")
        if v[np.argmax(np.abs(v))] < 0:
            nrm = -nrm
        full[:, i] = v / nrm
    if return_converged:
        return evals, full, converged
    return evals, full


def laplacian_1d_eV_nm2(n_nodes: int, h_nm: float, mass_ratio: float = 1.0):
    """The 1-D ``-(hbar**2/2m*) d2/dx2`` with Dirichlet ends, on ``n_nodes`` nodes.

    ``(n_int, n_int)`` CSR over the interior nodes only, ``n_int = n_nodes - 2``.
    This exists so a 1-D analytic case can be tested without a transverse
    zero-point: a 3-D box with a scalar effective mass always carries kinetic
    energy on the other two axes, and however wide the box is made that
    zero-point converges to a small non-zero constant rather than to zero --
    which would put a floor under every "the error goes to zero" claim.
    """
    if n_nodes < 3:
        raise StarkError(f"need at least 3 nodes for a 1-D operator, got {n_nodes}")
    if h_nm <= 0:
        raise StarkError(f"h_nm must be positive, got {h_nm}")
    if mass_ratio <= 0:
        raise StarkError("mass_ratio must be positive")
    n_int = n_nodes - 2
    t = (HBAR2_OVER_2ME_EV_NM2 / mass_ratio) / (h_nm * h_nm)
    main = np.full(n_int, 2.0 * t)
    off = np.full(n_int - 1, -t)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def lowest_states_1d(n_nodes: int, h_nm: float, *, mass_ratio: float = 1.0,
                     potential_eV=None, n_states: int = 4,
                     tol: float = 1e-12, seed: int = 0):
    """The lowest ``n_states`` eigenpairs of the 1-D confined-carrier problem.

    ``potential_eV`` is the potential energy on all ``n_nodes`` nodes (the ends
    are the Dirichlet walls and their value is unused), or ``None`` for zero.
    Returns ``(energies_eV, vectors)`` with ``vectors`` ``(n_nodes, n_states)``
    and zero on the walls.
    """
    T = laplacian_1d_eV_nm2(n_nodes, h_nm, mass_ratio)
    n_int = n_nodes - 2
    if potential_eV is not None:
        V = np.asarray(potential_eV, dtype=np.float64).reshape(-1)
        if V.size != n_nodes:
            raise StarkError(
                f"potential_eV has {V.size} values, expected {n_nodes}")
        T = T + sp.diags(V[1:-1])
    k = int(n_states)
    if k < 1 or k >= n_int:
        raise StarkError(f"n_states={k} must satisfy 1 <= k < {n_int}")
    rng = np.random.default_rng(seed)
    evals, evecs = spla.lobpcg(T, rng.standard_normal((n_int, k)),
                               largest=False, tol=tol, maxiter=5000)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    full = np.zeros((n_nodes, k))
    full[1:-1, :] = evecs
    for i in range(k):
        v = full[:, i]
        nrm = math.sqrt(float(np.dot(v, v)))
        if nrm == 0:
            raise StarkError(f"eigenvector {i} came back identically zero")
        full[:, i] = v / nrm
    return evals, full


def hbar_omega_eV(omega_meV: float) -> float:
    """``hbar * omega`` in eV for ``omega`` quoted as a photon energy in meV.

    The convention throughout this module: ``omega_meV`` IS ``hbar omega``
    expressed in meV, so ``hbar omega = omega_meV * 1e-3`` eV. That keeps the
    analytic spectrum of :func:`harmonic` at ``omega_meV*(n + 1/2)`` in meV
    without the caller having to think in angular frequency.
    """
    return omega_meV * 1e-3


def harmonic(n_states: int = 4, *, half_width_nm: float = 10.0,
             omega_meV: float = 10.0, mass_ratio: float = 1.0,
             n_cells: int = 24):
    """The 1-D harmonic oscillator, as an analytic test case.

    Returns ``(n_nodes, h_nm, V, exact_eV)`` for :func:`lowest_states_1d`:
    ``V = (1/2) m* omega**2 x**2`` on the nodes, and ``exact_eV[n] =
    hbar*omega*(n + 1/2)``.

    The box must be several oscillator lengths wide or the walls push every
    level up: the ground state reaches ``exp(-(L/2 / l)**2 / 2)`` at the wall,
    with ``l = sqrt(hbar / (m* omega))`` the oscillator length. The default
    ``half_width_nm=10`` with ``omega_meV=10`` gives ``l ~ 2.76 nm``, so the
    wall sits at 3.6 ``l`` where the tail is ~2e-3 -- and that is the floor the
    convergence test will run into unless it is widened. Callers that want to
    measure an O(h^2) rate must widen the box as they refine.
    """
    if mass_ratio <= 0:
        raise StarkError("mass_ratio must be positive")
    if omega_meV <= 0:
        raise StarkError("omega_meV must be positive")
    if n_cells < 2:
        raise StarkError(f"n_cells must be >= 2, got {n_cells}")

    n_nodes = n_cells + 1
    h_nm = (2.0 * half_width_nm) / n_cells
    x_nm = -half_width_nm + np.arange(n_nodes) * h_nm

    hbar = 1.054571817e-34
    m_kg = mass_ratio * 9.1093837015e-31
    omega_si = (omega_meV * 1e-3) * E_CHARGE_C / hbar  # rad/s
    x_m = x_nm * 1e-9
    V = 0.5 * m_kg * omega_si ** 2 * x_m ** 2 / E_CHARGE_C  # J -> eV

    exact = (np.arange(n_states) + 0.5) * hbar_omega_eV(omega_meV)
    return n_nodes, h_nm, V, exact


def hard_box_ground_eV(sizes_nm, mass_ratio: float = 1.0) -> float:
    """Ground energy of a particle in a hard-walled box, the exact reference.

    ``E = (hbar^2 / 2 m*) * pi^2 * sum(1 / L_i^2)`` over the given side
    lengths. ``K = hbar^2/2m`` already carries the 1/2, so this is
    ``K pi^2 / L^2`` per axis, NOT ``pi^2 K / (2 L^2)`` (the 1.8 trap, a ratio of
    exactly 2 that no refinement moves).
    """
    if mass_ratio <= 0:
        raise StarkError("mass_ratio must be positive")
    sizes = [float(s) for s in sizes_nm]
    if not sizes or any(not (math.isfinite(s) and s > 0) for s in sizes):
        raise StarkError(f"box sides must be positive and finite, got {sizes_nm!r}")
    return (HBAR2_OVER_2ME_EV_NM2 / mass_ratio) * math.pi ** 2 * sum(
        1.0 / (s * s) for s in sizes)


def finite_well_in_box(depth_eV: float, width_nm: float, extent_nm: float,
                       mass_ratio: float = 1.0):
    """Exact even ground state of a 1-D finite square well INSIDE a hard box.

    The well ``|z| < a`` (``a = width/2``) has ``V = 0``; the barrier
    ``a < |z| < L/2`` has ``V = depth``; ``z = +-L/2`` are hard walls. Unlike
    the infinite-barrier root used in 1.9's test, this one carries the box, so
    it has no tail-truncation floor and a convergence order measured against it
    is the discretisation's alone.

    Matching the log-derivative at ``z = a`` with ``k = sqrt(m E / K)``::

        k tan(k a) = kappa coth(kappa d)        E < depth, kappa = sqrt(m (V0 - E) / K)
        k tan(k a) = q cot(q d)                 E > depth, q = sqrt(m (E - V0) / K)

    where ``d = L/2 - a`` is the barrier thickness. The root lies between the
    bare-box ground ``K pi^2 / (m L^2)`` and the hard-wall well ground
    ``K pi^2 / (m w^2)``, because ``0 <= V <= (hard well)`` pointwise.

    Returns ``(E_eV, z2_nm2)``: the energy and the exact ``<z^2>`` of the
    normalised state (integrated on a 4e5-point grid of the closed form, which
    is far below any discretisation error measured against it).
    """
    from scipy.optimize import brentq

    V0, w, L, m = float(depth_eV), float(width_nm), float(extent_nm), float(mass_ratio)
    if not (math.isfinite(V0) and V0 > 0):
        raise StarkError(f"depth must be positive, got {depth_eV!r}")
    if not (0 < w < L):
        raise StarkError(
            f"well width {w} nm must be strictly inside the box {L} nm; at equal "
            f"widths the barrier has no thickness and this is the infinite well")
    if m <= 0:
        raise StarkError("mass_ratio must be positive")
    K = HBAR2_OVER_2ME_EV_NM2
    a, d = 0.5 * w, 0.5 * (L - w)

    def mismatch(E):
        # The matching condition multiplied through so it has NO poles: tan
        # and cot both blow up inside the bracket, and a root next to a pole
        # (a very deep well sits right under ka = pi/2) is lost to any pole
        # filter. E < V0 is divided by kappa cosh(kappa d) > 0, E > V0 by q, so
        # the sign is kept and both branches meet at k d sin(ka) - cos(ka).
        k = math.sqrt(m * E / K)
        s, c = math.sin(k * a), math.cos(k * a)
        if E < V0:
            kap = math.sqrt(m * (V0 - E) / K)
            t = math.tanh(kap * d) / kap if kap * d > 1e-12 else d
            return k * s * t - c
        q = math.sqrt(m * (E - V0) / K)
        sq = math.sin(q * d) / q if q * d > 1e-12 else d
        return k * s * sq - c * math.cos(q * d)

    # Scan upward in k = sqrt(m E / K) from the bare-box ground, in steps of a
    # quarter of the box's level spacing pi/(4L), and stop at the FIRST sign
    # change. Even states of the whole box are pi/L apart in k, so a step of
    # pi/(4L) cannot straddle two roots; a scan uniform in E (the first
    # version) put the ground and several excited roots in one interval when
    # the well was thin, and returned an excited state.
    def E_of(k):
        return K * k * k / m

    k_lo = math.pi / L * (1.0 + 1e-12)
    k_hi = math.pi / w * (1.0 + 1e-9)
    step = math.pi / (4.0 * L)
    E = None
    k0, g0 = k_lo, mismatch(E_of(k_lo))
    while k0 < k_hi:
        k1 = min(k0 + step, k_hi)
        g1 = mismatch(E_of(k1))
        if g0 == 0.0:
            E = E_of(k0)
            break
        if np.sign(g0) != np.sign(g1):
            E = brentq(mismatch, E_of(k0), E_of(k1), xtol=1e-15, rtol=1e-14)
            break
        k0, g0 = k1, g1
    if E is None:
        raise StarkError("no ground-state root found between the two bounds")

    k = math.sqrt(m * E / K)
    z = np.linspace(0.0, 0.5 * L, 400_001)
    if E < V0:
        kap = math.sqrt(m * (V0 - E) / K)
        # sinh(kap (L/2 - z)) / sinh(kap d), written with decaying exponentials
        # only: a deep well has kap d in the thousands and sinh overflows.
        zz = np.clip(z, a, None)
        outside = (math.cos(k * a) * np.exp(-kap * (zz - a))
                   * -np.expm1(-2.0 * kap * (0.5 * L - zz)) / -math.expm1(-2.0 * kap * d))
    else:
        q = math.sqrt(m * (E - V0) / K)
        outside = math.cos(k * a) * np.sin(q * (0.5 * L - z)) / math.sin(q * d)
    psi = np.where(z <= a, np.cos(k * z), outside)
    p = psi * psi
    z2 = float(np.trapezoid(z * z * p, z) / np.trapezoid(p, z))
    return float(E), z2


def interpolate_potential(solution: dict, points_nm, *, chunk: int = 200_000):
    """The electrostatic potential ``phi`` (V) at arbitrary points, in nm.

    ``solution`` is a :func:`mse_electrostatics.solve` result. The P1 potential
    is piecewise linear on the mesh, so the interpolant is exact for a field
    that is linear within each element -- which is why the uniform-field test
    must reproduce it to rounding, and why the linear-potential case is a real
    check of the barycentric search rather than of the solver.

    This is a vectorised version of what :func:`mse_electrostatics.field_at`
    does for one point: barycentric coordinates from the affine shape functions,
    then the containing tet. It differs in two ways that matter here:

    - It takes MANY points (a box grid), so the containment search is batched
      in chunks rather than per point.
    - It does not refuse a point inside an electrode. ``field_at`` refuses
      because ``grad phi`` is zero there and returning E = 0 would be a lie;
      here we return the imposed CONSTANT potential, which is the true potential
      inside the electrode. A caller that wants the field must difference this
      interpolant, and should know the electrode interior has zero gradient by
      construction, not by physics.

    Raises :class:`StarkError` when a point lies outside the mesh: a silent
    nearest-tet answer would be a wrong number with no warning, and the whole
    point of the linear case is that the answer is checked exactly.
    """
    pts_m = solution["points_m"]
    tets = solution["tets"]
    phi = np.asarray(solution["phi"], dtype=np.float64)

    q = np.asarray(points_nm, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q.reshape(1, 3)
    q_m = q * 1e-9

    txyz = pts_m[tets]                               # (n_tets, 4, 3)
    # grad lambda_a for a = 0..3, shape (n_tets, 4, 3) per :func:`mse_fem.tet_geometry`.
    # Taking it from that helper rather than inverting J here keeps this in step
    # with the P1 assembly, including its inv(J).T convention.
    det_J, grads = mse_fem.tet_geometry(pts_m, tets)
    v0 = txyz[:, 0, :]                                # (n_tets, 3)

    # DEGENERATE TETS MUST BE EXCLUDED, not merely tolerated. A sliver tet has a
    # tiny det_J, hence huge barycentric gradients, and the map lambda(p) then
    # stays in [0, 1] over a region far larger than the tet. Measured on a real
    # 703-point box mesh: a point at the ORIGIN -- 130 nm outside the surface --
    # came back inside a tet of det 1.7e-22 with lambda = (0.688, 0.055, 0.025,
    # 0.231), summing to 1 and all positive, so the containment test passed and
    # the guard below never fired. The threshold is relative to the mesh's own
    # median |det| so it means the same thing at any scale.
    median_det = np.median(np.abs(det_J))
    plausible = np.abs(det_J) > (DEGENERATE_DET_FRACTION * median_det)

    out = np.empty(q_m.shape[0], dtype=np.float64)
    missing = np.zeros(q_m.shape[0], dtype=bool)

    for start in range(0, q_m.shape[0], chunk):
        stop = min(start + chunk, q_m.shape[0])
        p = q_m[start:stop]                          # (m, 3)
        # lambda_a(p) = grad lambda_a . (p - v0) + delta_{a0}: affine, so this is
        # exact for any point inside the tet and extrapolates the same linear
        # function outside it (which is what makes the >= 0 test meaningful).
        #
        # Shapes: grads (t, 4, 3), p (m, 3), v0 (t, 3). The einsum contracts the
        # spatial index k and broadcasts m against t, giving (m, t, 4). Written
        # this way round because ``p - v0[:, None, :]`` would broadcast to
        # (t, m, 3) first and land the axes in the wrong places.
        lam = np.einsum("tak,mk->mta", grads, p) - np.einsum(
            "tak,tk->ta", grads, v0)[None, :, :]
        # delta_{a0}: the value of lambda_0 at v0 is 1, so add 1 to the a = 0
        # entry and nothing to the others.
        lam[:, :, 0] += 1.0
        inside = np.all(lam >= -BARY_EPS, axis=2)    # (m, n_tets)
        inside &= plausible[None, :]
        has = inside.any(axis=1)
        missing[start:stop] = ~has
        if has.any():
            idx = np.argmax(inside, axis=1)          # first containing tet
            rows = np.flatnonzero(has)
            l = lam[rows, idx[rows], :]              # (n_has, 4)
            out[start + rows] = np.einsum("na,na->n", l, phi[tets[idx[rows]]])

    if missing.any():
        bad = q[missing][0]
        raise StarkError(
            f"{int(missing.sum())} point(s) lie outside the mesh; "
            f"first is {bad.tolist()} nm. Interpolation is only defined where "
            f"the solution exists."
        )
    return out[0] if single else out


def field_on_grid(solution: dict, points_nm) -> np.ndarray:
    """The electric field (V/m) at grid nodes, from the EXACT per-tet gradient.

    ``points_nm`` is ``(nx, ny, nz, 3)`` node coordinates in nanometres. Returns
    ``(nx, ny, nz, 3)``.

    The field is taken as the gradient of the P1 potential WITHIN the tet that
    contains each node -- ``E_k = -sum_a grad(lambda_a)_k phi_a`` -- which is
    the same quantity :func:`mse_electrostatics.field_at` returns, and is exact
    for the discrete solution rather than a difference of it.

    This replaces an earlier implementation that interpolated phi to the nodes
    and central-differenced it. That was measured 31% wrong on a parallel-plate
    case whose exact answer the per-tet gradient reproduces to 0.3% (E_z =
    -3.560 vs -3.571 MV/m): a central difference across a mesh that is only a
    few elements thick differences across kinks in a piecewise-linear function,
    which is a first-order-in-h error, not the O(h^2) the stencil promises on a
    smooth field. The per-tet gradient has no such error.

    A node shared by several tets takes the value from its containing tet,
    which is well defined because the P1 gradient agrees across a conforming
    face only in the normal component; the tangential jump is the physical
    artefact of a piecewise-linear field, and picking one tet is the same
    choice :func:`mse_electrostatics.field_at` makes.
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 4 or pts.shape[-1] != 3:
        raise StarkError(f"points_nm must be (nx,ny,nz,3), got {pts.shape}")

    pts_m = solution["points_m"]
    tets = solution["tets"]
    phi = np.asarray(solution["phi"], dtype=np.float64)
    _det, grads = mse_fem.tet_geometry(pts_m, tets)   # (t, 4, 3), grad lambda_a
    v0 = pts_m[tets][:, 0, :]

    # Per-tet constant field: E_k = -sum_a grad_a,k * phi_a.
    E_tet = -np.einsum("tak,ta->tk", grads, phi[tets])       # (t, 3) in V/m

    q = pts.reshape(-1, 3) * 1e-9                    # nm -> m
    lam = np.einsum("tak,mk->mta", grads, q) - np.einsum(
        "tak,tk->ta", grads, v0)[None, :, :]
    lam[:, :, 0] += 1.0
    inside = np.all(lam >= -BARY_EPS, axis=2)
    has = inside.any(axis=1)
    if not has.all():
        bad = pts.reshape(-1, 3)[~has][0]
        raise StarkError(
            f"{int((~has).sum())} grid node(s) lie outside the mesh; "
            f"first is {bad.tolist()} nm. The field is only defined where the "
            f"solution exists."
        )
    idx = np.argmax(inside, axis=1)
    return E_tet[idx].reshape(pts.shape[:3] + (3,))


def airy(*, field_eV_per_nm: float = 0.05, extent_nm: float = 120.0,
         mass_ratio: float = 1.0, n_cells: int = 240):
    """A particle in a linear potential against a hard wall: the Airy test case.

    ``V(x) = e * F * x`` for ``x >= 0``, with a Dirichlet wall at ``x = 0`` and
    the far end at ``x = extent_nm``. The exact spectrum is given by the zeros
    of the Airy function, ::

        E_n = ( (hbar**2 F**2) / (2 m*) )**(1/3) * |a_n|

    with ``|a_1| = 2.33811``. In eV nm units with ``F`` in eV/nm this is
    ``(K / r * F**2)**(1/3) * |a1|`` where ``K = hbar**2/(2 m_e)`` in eV nm**2
    and ``r = m*/m_e``. [This is the standard tilted-well/Airy problem; the
    zeros are DLMF 9.9.1, and the E_n relation follows from rescaling
    ``-K psi'' + F x psi = E psi`` with ``x = (K/F)^(1/3) z``.]

    The far wall must be far enough that the wavefunction has decayed: the
    turning point for the ground state is at ``x_t = E_1 / F``, which for
    ``F = 0.05 eV/nm`` is about ``x_t ~ 1.7 nm`` (``E_1 ~ 0.086 eV``), so a
    120 nm extent leaves the wall deep in the tail. That decay length grows like
    ``(K F^2)^(-1/3)``; the caller that refines must keep the extent fixed so the
    error measured is the discretisation's and not the wall's.

    Returns ``(n_nodes, h_nm, V, exact_eV)`` for :func:`lowest_states_1d`.
    """
    if field_eV_per_nm <= 0:
        raise StarkError(f"field_eV_per_nm must be positive, got {field_eV_per_nm}")
    if extent_nm <= 0:
        raise StarkError(f"extent_nm must be positive, got {extent_nm}")
    if mass_ratio <= 0:
        raise StarkError("mass_ratio must be positive")
    if n_cells < 2:
        raise StarkError(f"n_cells must be >= 2, got {n_cells}")

    n_nodes = n_cells + 1
    h_nm = extent_nm / n_cells
    x_nm = np.arange(n_nodes) * h_nm
    V = field_eV_per_nm * x_nm                     # eV, since e*F*x with F in eV/nm

    scale = (HBAR2_OVER_2ME_EV_NM2 / mass_ratio * field_eV_per_nm ** 2) ** (1.0 / 3.0)
    # The first few Airy zeros, ascending. Only the principal one is used by the
    # test, but returning the set makes the helper usable for the excited states
    # once a comparison needs them. Values: DLMF 9.9.1.
    zeros = np.array([2.338107410459767, 4.087949444130970,
                      5.520559828095551, 6.786708090071759])
    exact = scale * zeros
    return n_nodes, h_nm, V, exact
