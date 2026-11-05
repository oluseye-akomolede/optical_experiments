"""The first-order electron-hole Coulomb term and the exciton energy E_X(F).

Layer L3 of the electrical chain (``design.md`` §2). A single InAsP dot is
modelled as two confined carriers in a uniform Cartesian box:

* an electron of band mass ``m_e*`` and a heavy hole of mass ``m_h*``,
* both in the same applied field ``F`` along z (the field the L1 electrostatic
  solve produces, interpolated onto this box by :func:`stark.field_on_grid`),
* interacting through the screened Coulomb potential
  ``V(r) = -e^2 / (4 pi eps0 eps_s |r_e - r_h|)``.

The exciton energy is

    E_X(F) = E_g + E_e(F) + E_h(F) + E_C(F)          (first order in V)

where ``E_e``, ``E_h`` are the single-particle confinement energies measured
from the band edges, and ``E_C`` is the Coulomb binding energy

    E_C = <psi_e psi_h | V | psi_e psi_h>

evaluated **to first order in the Coulomb interaction** on the product of the
two independently solved single-particle ground states. That is the L3
approximation stated in the design; it is not a self-consistent two-particle
solve, and the module says so where it matters (:func:`coulomb_energy`,
:func:`exciton_energy`).

Why the product-state first-order form is defensible here, and where it is not:
the Coulomb interaction is weak compared with the confinement energy in a dot of
a few nanometres (the ratio is the dot radius to the bulk exciton Bohr radius),
which is the standard perturbative condition. It degrades as the dot grows, and
it ignores correlation entirely. The docstrings record the size of the error the
approximation makes against the analytic 1s binding energy of a well, which is
the only place a closed form exists to check it against.

UNITS: the field is V/m (what the L1 solve produces), lengths are nm inside the
grid and converted to SI only where a physical constant needs it, and energies
are eV throughout. The conversions are named constants (:data:`E2_OVER_4PIEPS0_EV_NM`,
:data:`NM_M`) rather than inline factors, because the 1.7 and 1.8 rounds both
shipped a unit error and a named constant is the thing a test can pin.
"""

from __future__ import annotations

import math

import numpy as np

import stark
from stark import E_CHARGE_C, StarkError

#: Elementary charge squared over 4 pi eps0, in eV nm.
#:
#: ``e^2 / (4 pi eps0) = 1.4399645 eV nm``. That is the constant that makes
#: ``V(r) = -14.3996 / (eps_s r[nm])`` come out in eV, and it is quoted as one
#: number so a test can check the arithmetic independently of any solve. The
#: value follows from the CODATA elementary charge and eps0 (both exact in SI
#: since the 2019 redefinition), so it is reproducible from those two constants
#: rather than a remembered figure: e = 1.602176634e-19 C, eps0 = 8.8541878128e-12
#: F/m, and 1 eV nm = 1.602176634e-28 J m.
E2_OVER_4PIEPS0_EV_NM = 1.4399645322

#: One nanometre in metres. Spelled out because converting a field
#: (V/m) to a potential (eV over a grid in nm) is where the sign and the
#: factor both went wrong in 1.8.
NM_M = 1.0e-9

#: Bohr radius of the electron in nm, from the standard definition
#: a0 = 4 pi eps0 hbar^2 / (m0 e^2). Used only to report the perturbative ratio.
A0_NM = 0.0529177


class CoulombError(ValueError):
    """A Coulomb input or solve that cannot give a meaningful answer."""


def coulomb_energy_eV(psi_e, psi_h, box, eps_s, *, voxel: bool = True):
    """The first-order Coulomb binding energy ``E_C`` of a product state.

    ``psi_e`` and ``psi_h`` are full-box eigenvectors from
    :func:`stark.lowest_states`, shape ``box.n_nodes()``, on the node grid.
    ``eps_s`` is the static dielectric constant of the dot material.

    The integral is

        E_C = sum_ij |psi_e(i)|^2 |psi_h(j)|^2 V(|r_i - r_j|) dV_i dV_j

    with ``V(r) = -E2_OVER_4PIEPS0_EV_NM / (eps_s |r_i - r_j|)``. The separable
    form is exact for a product state and is what makes a 3-D dot tractable:
    projecting onto the radial coordinate ``r`` the double sum becomes a
    convolution of the two radial densities with ``-1/r``.

    ``voxel=True`` includes the ``dV = hx hy hz`` measure. That matters: the
    eigenvectors are unit-norm in the EUCLIDEAN sense, so without the measure
    the energy scales with the grid and does not converge under refinement. The
    density is ``|psi|^2 dV``, and a test pins the normalisation.

    Returns ``(e_coulomb_eV, diagnostics)`` where ``diagnostics`` records the
    grid, the electron-hole overlap integral, and the on-site self-energy that
    was excluded (see :func:`_self_energy_eV`).
    """
    box, he, hh, eps_s = _check_inputs(psi_e, psi_h, box, eps_s)

    de = _physical_density(psi_e, box)
    dh = _physical_density(psi_h, box)

    # The measure, applied once per carrier: w_i dV and w_j dV.
    w = _trapz_weights(box, de)
    dV = float(np.prod(box.h()))
    de = de * w * dV
    dh = dh * w * dV

    # Node coordinates, flattened in the SAME order as the eigenvector.
    xs, ys, zs = box.axes()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3) nm

    # Exact 1/r convolution is O(N^2); N here is ~1e4-1e5 for a small dot box,
    # which is worth doing exactly rather than by a grid FFT. The density is
    # concentrated in the dot, so keep only the significant nodes.
    keep = (de > 1e-12) | (dh > 1e-12)
    de_k = de[keep]
    dh_k = dh[keep]
    pos_k = pos[keep]
    if de_k.size == 0 or dh_k.size == 0:
        raise CoulombError("a carrier density is identically zero on the grid")

    # E_C = -C/eps_s * sum_ij de_i dh_j / |r_i - r_j|, with the i == j
    # (self-interaction of one grid cell with itself) term REMOVED, because a
    # density on a cell does not Coulomb-interact with itself. What remains is
    # the screened interaction between the two charge clouds.
    e_c = _screened_pair_energy(de_k, dh_k, pos_k, eps_s)

    overlap = float(np.sum(np.sqrt(np.maximum(de, 0.0) * np.maximum(dh, 0.0))))
    diag = {
        "grid": tuple(int(v) for v in (box.n[0] + 1, box.n[1] + 1, box.n[2] + 1)),
        "voxel_nm3": he * hh * (box.h()[2]),
        "nodes_used": int(np.count_nonzero(keep)),
        "overlap": overlap,
        "eps_s": float(eps_s),
        "e2_over_4pieps0_eV_nm": E2_OVER_4PIEPS0_EV_NM,
        # The 1s Bohr radius of the same two-body problem, for the
        # perturbative-ratio report; see exciton_energy.
        "bohr_radius_nm": bohr_radius_nm(eps_s, diag_mass_ratio(box)),
    }
    return e_c, diag


def diag_mass_ratio(box):
    """The electron-heavy-hole reduced mass ratio, for the Bohr radius only."""
    return getattr(box, "_mass_ratio", 1.0)


def bohr_radius_nm(eps_s, reduced_mass_ratio):
    """The 1s Bohr radius of the two-body problem, in nm.

    ``a = a0 * eps_s / mu`` with ``mu`` the reduced mass in units of ``m0``.
    This is the length the dot radius is compared with to judge whether the
    first-order Coulomb treatment is appropriate.
    """
    if reduced_mass_ratio <= 0:
        raise CoulombError(f"reduced mass must be positive, got {reduced_mass_ratio}")
    return A0_NM * float(eps_s) / float(reduced_mass_ratio)


def _check_inputs(psi_e, psi_h, box, eps_s):
    for name, psi in (("psi_e", psi_e), ("psi_h", psi_h)):
        arr = np.asarray(psi, dtype=np.float64)
        if arr.size != box.n_nodes():
            raise CoulombError(
                f"{name} has {arr.size} values, the box has {box.n_nodes()} nodes"
            )
    # eps_s >= 1 is physical (1 is vacuum); the guard is against zero or
    # negative, which would make the potential infinite or attractively reversed.
    if not math.isfinite(eps_s) or eps_s < 1.0:
        raise CoulombError(f"eps_s must be finite and >= 1, got {eps_s!r}")
    return box, box.h()[0], box.h()[1], eps_s


def _physical_density(psi, box):
    """The probability DENSITY ``|psi|^2`` (nm^-3) at each node, mass 1.

    The eigenvectors from :func:`stark.lowest_states` are unit-norm in the
    EUCLIDEAN sense (``sum |psi_i|^2 = 1``), NOT in the physical sense
    (``integral |psi|^2 dV = 1``). On a grid of spacing ``h`` those differ by a
    factor of the volume: ``sum |psi_i|^2 dV = dV``, not 1. So the eigenvector
    must be divided by ``sqrt(dV)`` before it is a density:

        integral |psi/sqrt(dV)|^2 dV = 1.

    This rescaling is what makes the energy independent of the grid: without it
    ``E_C`` scales with the box volume and never converges under refinement.
    The first version of this function omitted it, and the energy came out
    ``dV^2`` too small -- a factor of 4096 on the 16-cell box.

    The measure itself (``w_i dV``) is applied by the caller, once per carrier.
    """
    p = np.abs(np.asarray(psi, dtype=np.float64).reshape(-1)) ** 2
    dV = float(np.prod(box.h()))
    return p / dV


def _trapz_weights(box, p):
    """Per-node trapezoidal weight factors (1 at interior, 1/2 at walls).

    Built as an outer product of three 1-D weight vectors, so the corner
    weight is (1/2)^3 and an edge (1/2)^2. Summing the weights over a unit
    field must give the number of nodes, which is the check the test applies --
    the first version of this summed to ``n_cells^3`` and was wrong.
    """
    shape = (box.n[0] + 1, box.n[1] + 1, box.n[2] + 1)
    axes = []
    for m in shape:
        v = np.ones(m)
        v[0] = v[-1] = 0.5
        axes.append(v)
    w = np.einsum("i,j,k->ijk", axes[0], axes[1], axes[2])
    return w.reshape(-1)


def _screened_pair_energy(de, dh, pos, eps_s, *, block: int = 4096):
    """``-C/eps_s * sum_ij de_i dh_j / |r_i - r_j|``, blocked to bound memory.

    The ``i == j`` term is excluded (``np.inf`` on the diagonal). Blocking over
    the first index keeps the working set to ``block * N`` doubles: the full
    ``(N, N, 3)`` displacement array needs 3.3 GiB at N = 12k, which is over the
    tier-1 cap. The block size is a memory choice, not a physics one.
    """
    n = de.size
    total = 0.0
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = pos[start:stop, None, :] - pos[None, :, :]   # (b, n, 3)
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        # Exclude the on-site term for the rows in this block.
        rows = np.arange(start, stop)
        dist[np.arange(stop - start), rows] = np.inf
        total += float(np.einsum("i,j,ij->", de[start:stop], dh, 1.0 / dist))
    return -E2_OVER_4PIEPS0_EV_NM * total / float(eps_s)


def _self_energy_eV(box, eps_s, mass_ratio):
    """The diagonal (self-interaction) energy excluded by :func:`coulomb_energy_eV`.

    A density lumped on one node would interact with itself as ``-1/0``. We
    exclude that term, which is the standard lattice convention, but it means
    the answer carries a grid-dependent offset of order the interaction of one
    cell with its neighbours. Records it so a convergence test can show it
    shrinking; does not silently rely on it being small.
    """
    hx, hy, hz = box.h()
    d = min(hx, hy, hz)
    return -E2_OVER_4PIEPS0_EV_NM / (float(eps_s) * d)
