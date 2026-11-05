"""The exciton Stark shift E_X(F) and the p/beta fit, for the paper-1 dot.

The model is the one the paper states, kept as thin as it can be:

    E_X(F) = E_e(F) + E_h(F) + E_C

where ``E_e`` and ``E_h`` are the ground-state energies of an electron and a
heavy hole in the L1 field, each solved separately as a single-band
effective-mass problem (``stark.lowest_states`` with the sawtooth potential),
and ``E_C`` is the first-order Coulomb term (``coulomb.coulomb_energy_eV``).

Why first order is enough here: the paper's dot is in the perturbative regime,
where the Coulomb energy is small against the confinement energy. This module
computes the ratio and reports it rather than asserting the regime; see
:func:`perturbative_ratio`.

The fit then recovers the dipole ``p`` and polarisability ``beta`` from

    dE(F) = E_X(F) - E_X(0) = -p F + beta F^2

by least squares over the computed sweep. That is a quadratic in F with two
unknowns, so two field points would in principle do; more are used so the
residual is a check rather than a decoration.

UNITS. Energies in eV, lengths in nm, fields in eV/nm internally, converted to
kV/cm only at the boundary for reporting. ``stark`` fixes the unit system and
this module does not introduce a second one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import coulomb as _coulomb
import stark as _stark

__all__ = [
    "EV_PER_NM_TO_KV_PER_CM",
    "ExcitonError",
    "DotMaterial",
    "ExcitonResult",
    "WELL_MASKS",
    "axis_inside_fraction",
    "dot_fill_fraction",
    "sawtooth_potential",
    "exciton_energy_eV",
    "fit_p_beta",
    "perturbative_ratio",
]

#: 1 eV/nm in kV/cm. 1 eV/nm = 1e7 V/cm = 1e4 kV/cm.
EV_PER_NM_TO_KV_PER_CM = 1.0e4


class ExcitonError(ValueError):
    """A material or field input is outside what this model can represent."""


@dataclass(frozen=True)
class DotMaterial:
    """The dot material parameters the exciton model needs.

    EVERY field here must come from a cited source. This class does not carry
    defaults on purpose: a silently defaulted permittivity is exactly the kind
    of invented number this project has had to retract before. Construct it
    explicitly at the call site and say where each value came from.
    """

    eps_s: float
    """Static (low-frequency) dielectric constant, dimensionless."""

    m_e_ratio: float
    """Electron effective mass over the free-electron mass."""

    m_h_ratio: float
    """Heavy-hole effective mass over the free-electron mass."""

    source: str = ""
    """Where the three values above came from: paper, table, page."""

    def __post_init__(self):
        for name in ("eps_s", "m_e_ratio", "m_h_ratio"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ExcitonError(f"{name} must be a positive finite number, got {v!r}")
        if self.eps_s < 1.0:
            raise ExcitonError(
                f"eps_s={self.eps_s} is below vacuum; that is not a material"
            )
        if not self.source:
            raise ExcitonError(
                "DotMaterial requires a source string naming where these values "
                "came from. An unsourced permittivity is not a parameter."
            )


@dataclass(frozen=True)
class ExcitonResult:
    """One field point: the two single-particle energies and the Coulomb term."""

    field_eV_per_nm: float
    e_electron_eV: float
    e_hole_eV: float
    e_coulomb_eV: float
    e_x_eV: float
    e_x_shift_eV: float
    """``E_X(F) - E_X(0)``; set by :func:`fit_p_beta`, zero here until then."""
    coulomb_over_confinement: float
    """``|E_C| / (E_e(F) + E_h(F))``: the perturbative-regime diagnostic."""
    coulomb_diagnostics: dict


def sawtooth_potential(
    box: _stark.Box,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
    well_lateral_nm=None,
    well_mask: str = "node",
) -> np.ndarray:
    """The L1 potential ``-e F z`` on the box's node grid, in eV.

    The field is along z. For a POSITIVE field the potential falls along +z, so
    a carrier of charge -e sees a ramp that pushes it toward +z.

    NO CONFINEMENT WELL IS ADDED. The returned array is ``-F z`` alone, so the
    "+ well" case is a free particle in the box and its energies and densities
    describe a delocalised carrier, not a quantum dot. A well depth can be
    passed separately via ``well_depth_eV``; with the default of None there is
    none. This is stated here because three docstrings in this module used to
    describe "the symmetric confinement well" as though this function created
    one, and it does not.
    """
    if not (isinstance(field_eV_per_nm, (int, float)) and math.isfinite(field_eV_per_nm)):
        raise ExcitonError(f"field must be a finite number, got {field_eV_per_nm!r}")
    if field_eV_per_nm < 0:
        raise ExcitonError(
            f"field must be non-negative; the spectrum is symmetric under "
            f"F -> -F and the fit uses |F|, got {field_eV_per_nm!r}"
        )
    return _potential_grid(box, field_eV_per_nm, well_depth_eV, well_height_nm,
                           well_lateral_nm, well_mask)


#: The two ways a dot's potential step is put on the node grid.
#:
#: ``"node"`` -- the 1.9 rule: a node is inside when it lies in the closed dot
#: region, and takes the full depth otherwise. A face that lands on a node
#: counts that node as inside, so the discrete well is up to one cell WIDER
#: than asked (half a cell each side) and every result carries an O(h) error.
#: Measured for the 1.9 bench well (4 nm, 0.8 eV, electron, h_z = 0.4 nm):
#: ``<z^2>`` = 1.391 nm^2 against the exact 1.2574, converging at first order
#: (1.322 / 1.289 / 1.273 at h_z = 0.2 / 0.1 / 0.05).
#:
#: ``"cell"`` -- each node takes ``depth * (1 - f)``, ``f`` the fraction of its
#: dual cell (``x +- h/2`` per axis) lying inside the dot. That preserves the
#: integral of the potential exactly on any grid, which is what removes the
#: O(h) width error: measured O(h^2) against exact answers when the faces lie
#: on nodes (see ``tests/test_lateral_dot.py``).
WELL_MASKS = ("node", "cell")


def axis_inside_fraction(nodes_nm, h_nm: float, lo_nm: float, hi_nm: float) -> np.ndarray:
    """Fraction of each node's dual cell ``[x - h/2, x + h/2]`` inside ``[lo, hi]``.

    1 deep inside, 0 far outside, 1/2 for a node sitting exactly on a face. The
    fractions times ``h`` sum to the overlap length of ``[lo, hi]`` with the
    covered span, so the cell-averaged step carries the right well WIDTH on any
    grid, which node sampling does not.
    """
    x = np.asarray(nodes_nm, dtype=np.float64)
    a = np.maximum(x - 0.5 * h_nm, lo_nm)
    b = np.minimum(x + 0.5 * h_nm, hi_nm)
    # Clipped to [0, 1]: b - a can exceed h by a rounding ulp.
    return np.clip((b - a) / h_nm, 0.0, 1.0)


def _check_well(box, well_height_nm, well_lateral_nm, well_mask):
    if well_mask not in WELL_MASKS:
        raise ExcitonError(f"well_mask must be one of {WELL_MASKS}, got {well_mask!r}")
    if not (math.isfinite(well_height_nm) and well_height_nm > 0):
        raise ExcitonError(f"well height must be positive, got {well_height_nm!r}")
    if well_lateral_nm is None:
        return None
    try:
        lat = tuple(float(v) for v in well_lateral_nm)
    except TypeError:
        lat = (float(well_lateral_nm),) * 2
    if len(lat) != 2:
        raise ExcitonError(f"well_lateral_nm is one side or (x, y), got {well_lateral_nm!r}")
    for side, extent, axis in zip(lat, box.size[:2], "xy"):
        if not (math.isfinite(side) and side > 0):
            raise ExcitonError(f"lateral dot side must be positive, got {side!r}")
        if side >= extent:
            # The same trap as a well as tall as its box: the barrier would sit
            # on (or past) the Dirichlet walls, where its value is unused, and
            # the lateral confinement silently becomes the hard box wall. Pass
            # None for "the well fills the box laterally" -- the 1.9 model.
            raise ExcitonError(
                f"lateral dot side {side} nm along {axis} must be strictly less "
                f"than the box extent {extent} nm; pass None for a well that "
                f"fills the box laterally")
    return lat


def dot_fill_fraction(box: _stark.Box, well_height_nm: float,
                      well_lateral_nm=None, well_mask: str = "cell") -> np.ndarray:
    """Per-node fraction of the dot occupying the node, on the box's node grid.

    The dot is a rectangular box centred on the grid box: ``well_height_nm``
    along z (the field axis) and ``well_lateral_nm`` along x and y (a single
    side, or an ``(x, y)`` pair). ``None`` laterally means the dot fills the box
    in x and y, which is the 1.9 model.

    ``well_mask`` picks how the faces are put on the grid (see
    :data:`WELL_MASKS`): ``"node"`` gives 0/1 by the closed-region test (1.9's
    rule), ``"cell"`` the dual-cell fraction, as a product of per-axis
    fractions because the dot is a product of three intervals.

    Returned in the ``meshgrid(..., indexing="ij")`` shape, which is the order
    :func:`stark.build_hamiltonian` flattens.
    """
    lat = _check_well(box, well_height_nm, well_lateral_nm, well_mask)
    axes = box.axes()
    hs = box.h()
    extents = [None, None, float(well_height_nm)]
    if lat is not None:
        extents[0], extents[1] = lat
    per_axis = []
    for nodes, h, ext in zip(axes, hs, extents):
        if ext is None:
            per_axis.append(np.ones(nodes.size))
            continue
        mid = 0.5 * (nodes[0] + nodes[-1])
        lo, hi = mid - 0.5 * ext, mid + 0.5 * ext
        if well_mask == "node":
            per_axis.append(((nodes >= lo) & (nodes <= hi)).astype(np.float64))
        else:
            per_axis.append(axis_inside_fraction(nodes, h, lo, hi))
    return np.einsum("i,j,k->ijk", *per_axis)


def _potential_grid(
    box: _stark.Box,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
    well_lateral_nm=None,
    well_mask: str = "node",
) -> np.ndarray:
    """``-e F z`` plus an optional symmetric well, on the box's node grid.

    The well is a finite box well of ``well_depth_eV``, ``well_height_nm`` along
    z and ``well_lateral_nm`` along x and y, centred on the box. Outside it the
    potential is raised by the depth: the dot's material is the well and the
    host (InP, both above/below AND around the dot in a nanowire) is the
    barrier, so the lateral barrier has the same depth as the axial one.

    ``well_lateral_nm=None`` (the default) fills the box in x and y: the 1.9
    model, where the only lateral confinement is the hard box wall.
    ``None`` depth means no well: the carrier is free in the box.

    ``well_mask="node"`` (the default) is the 1.9 sampling and reproduces its
    numbers exactly; ``"cell"`` is the cell-averaged step, which converges at
    second order. See :data:`WELL_MASKS` for why they differ.
    """
    xs, ys, zs = box.axes()
    # stark's potential is applied as a diagonal in the same flattened order as
    # the eigenvectors, which is meshgrid(..., indexing="ij").ravel().
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    V = -field_eV_per_nm * Z
    if well_depth_eV is not None:
        if well_height_nm is None:
            raise ExcitonError(
                "well_depth_eV needs well_height_nm: the well width is the dot "
                "height, which is not the box size"
            )
        if not (math.isfinite(well_depth_eV) and well_depth_eV > 0):
            raise ExcitonError(f"well depth must be positive, got {well_depth_eV!r}")
        fill = dot_fill_fraction(box, well_height_nm, well_lateral_nm, well_mask)
        V = V + well_depth_eV * (1.0 - fill)
    elif well_lateral_nm is not None:
        raise ExcitonError("well_lateral_nm needs a well depth: no depth means no dot")
    return V


def _ground_state(
    box: _stark.Box,
    mass_ratio: float,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
    well_lateral_nm=None,
    well_mask: str = "node",
):
    """Ground-state energy and vector of one species in the L1 potential."""
    V = _potential_grid(box, field_eV_per_nm, well_depth_eV, well_height_nm,
                        well_lateral_nm, well_mask)
    evals, full = _stark.lowest_states(box, mass_ratio, n_states=1, potential_eV=V)
    return float(evals[0]), full[:, 0], {"n_states": 1, "energy_eV": float(evals[0])}


def exciton_energy_eV(
    material: DotMaterial,
    box: _stark.Box,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
    well_lateral_nm=None,
    well_mask: str = "node",
) -> ExcitonResult:
    """``E_X`` at one field: electron + hole ground states + first-order Coulomb.

    The electron and hole see potentials of opposite sign in reality, but when
    the confinement well is symmetric the two single-particle problems are
    related by ``z -> -z``. Solving both with the SAME ``V = -eFz + V_well`` and
    the two masses gives the same pair of energies as the physical assignment,
    because a symmetric well has the same spectrum under reflection. That is a
    physics assumption, not a coding convenience, and it holds only while
    ``well_depth_eV`` describes a well centred on the box.

    With the default ``well_depth_eV=None`` there is NO well: the carrier is
    free in the box and ``E_X`` is not a quantum-dot energy. Pass a depth to get
    a confined result, and check ``<z^2>`` against the bare-box value
    ``L^2 (1/12 - 1/(2 pi^2))`` to see that it actually is confined.

    ``well_lateral_nm`` gives the dot its lateral size (a box dot, x and y),
    with the host as the barrier around it; ``None`` keeps the 1.9 well that
    fills the box laterally. The z-reflection argument above still holds,
    because the box dot is symmetric in z. ``well_mask`` is passed through to
    :func:`_potential_grid`.
    """
    # Route through the validated helper rather than a bare finite check, so
    # the sign rule cannot be bypassed by this one entry point. Earlier this
    # function checked only finiteness and a negative field silently solved.
    sawtooth_potential(box, field_eV_per_nm, well_depth_eV, well_height_nm,
                       well_lateral_nm, well_mask)

    well = (well_depth_eV, well_height_nm, well_lateral_nm, well_mask)
    e_e, psi_e, sol_e = _ground_state(box, material.m_e_ratio, field_eV_per_nm, *well)
    e_h, psi_h, sol_h = _ground_state(box, material.m_h_ratio, field_eV_per_nm, *well)

    e_c, diag = _coulomb.coulomb_energy_eV(psi_e, psi_h, box, material.eps_s)

    # ``e_c`` comes back NEGATIVE, and that is correct -- do not "fix" it to
    # positive. It is tempting: coulomb.py's potential is written
    # ``V = -e^2 / (4 pi eps0 eps_s |r|)``, which looks repulsive for two
    # same-sign densities, so a negative binding energy reads as a sign error.
    # The reason it is right is that ``e_e`` and ``e_h`` are energies measured
    # from a band edge sitting at ZERO, and the field pulls both carriers up
    # the ramp, so each can sit BELOW zero. ``e_x`` here is therefore a
    # SPACING between two such levels, not an absolute energy, and ``e_c`` is
    # the (attractive) correction to that spacing. Its sign is the sign of the
    # interaction, relative to the free-carrier reference, and the absolute
    # datum is arbitrary. Changing the reference shifts all three together.
    e_x = e_e + e_h + e_c
    confinement = e_e + e_h
    ratio = abs(e_c) / abs(confinement) if confinement != 0 else float("inf")

    return ExcitonResult(
        field_eV_per_nm=float(field_eV_per_nm),
        e_electron_eV=e_e,
        e_hole_eV=e_h,
        e_coulomb_eV=float(e_c),
        e_x_eV=float(e_x),
        e_x_shift_eV=0.0,
        coulomb_over_confinement=float(ratio),
        coulomb_diagnostics=diag,
    )


def fit_p_beta(results):
    """Recover ``p`` and ``beta`` from ``dE(F) = -p F + beta F^2``.

    ``results`` must be a sequence of :class:`ExcitonResult` covering at least
    two distinct non-zero fields, including a zero-field point to anchor the
    shift. Returns ``(p, beta, residuals, design)``: the fitted dipole in
    eV nm / (eV/nm) = nm, the polarisability in nm / (eV/nm)... in the units the
    model uses, the residual norm, and the least-squares design matrix so the
    caller can report the conditioning.

    The shift is measured against the FIELD-FREE result, which is located by
    its own ``field == 0`` entry rather than assumed to be ``results[0]``.

    WARNING -- read this before using the returned ``p``. For a SYMMETRIC
    confining well the potential ``-eFz`` is odd in z, so the spectrum is
    identical at ``+F`` and ``-F`` and the LINEAR Stark term is identically
    zero by symmetry. ``p`` is then not a measured dipole but the small
    asymmetric residue of the fit, and it comes out as a large number that is
    pure conditioning artefact: measured here as ``E(+F) - E(-F) = -4.7e-12 eV``
    at F = 0.01 eV/nm, i.e. zero to rounding.
    The fit is kept because the paper asks for the p/beta form and a real dot
    is not perfectly symmetric, but a caller reporting ``p`` for a symmetric
    dot is reporting noise. :func:`beta_low_field` is the stable estimator.

    That symmetry figure was measured with NO confining well present, where the
    box potential is symmetric trivially. It has NOT been re-measured with a
    well, and passing ``well_depth_eV`` is the correct way to make the premise
    true rather than assumed.
    """
    seq = list(results)
    zeros = [r for r in seq if r.field_eV_per_nm == 0.0]
    if len(zeros) != 1:
        raise ExcitonError(
            f"need exactly one zero-field point to anchor the shift, found {len(zeros)}"
        )
    if len(seq) < 3:
        raise ExcitonError(
            f"need two non-zero fields as well as the zero point, got {len(seq)} row(s)"
        )
    e0 = zeros[0].e_x_eV

    pts = [r for r in seq if r.field_eV_per_nm != 0.0]
    if len(pts) < 2:
        raise ExcitonError("need at least two non-zero fields to fit two parameters")
    fields = np.array([r.field_eV_per_nm for r in pts], dtype=float)
    dE = np.array([r.e_x_eV - e0 for r in pts], dtype=float)

    # dE = -p F + beta F^2  ->  design columns (F, F^2), coefficients (beta, -p)
    design = np.stack([fields, fields**2], axis=1)
    (beta, neg_p), *_ = np.linalg.lstsq(design, dE, rcond=None)
    p = -float(neg_p)
    beta = float(beta)

    pred = design @ np.array([beta, -p])
    resid = float(np.linalg.norm(dE - pred))
    return p, beta, resid, design


def beta_low_field(results):
    """``beta`` from the pure-quadratic model ``dE = beta F^2``, and its drift.

    This is the stable estimator for the paper-1 dot, whose well is symmetric
    so the linear term is zero by symmetry (see :func:`fit_p_beta`). It fits one
    parameter, so the residual is a genuine diagnostic rather than an artefact
    of an exactly-determined two-parameter system.

    Returns ``(beta, residual_norm, beta_drift)`` where ``beta_drift`` is the
    change in the per-point ``dE/F^2`` between the SMALLEST and LARGEST field.
    A drift that is a large fraction of ``beta`` means the quadratic model no
    longer holds at the top of the sweep and the fit is outside its regime;
    measured on the paper-1 geometry with eps_s=12.5, ``beta_drift/beta`` is
    about 0.10 up to F = 0.016 eV/nm and 0.36 by F = 0.04 eV/nm.

    Callers should quote ``beta`` as the LOW-FIELD value and report the drift
    next to it, not as a single number spanning the sweep.
    """
    seq = [r for r in results if r.field_eV_per_nm != 0.0]
    zeros = [r for r in results if r.field_eV_per_nm == 0.0]
    if len(zeros) != 1:
        raise ExcitonError(
            f"need exactly one zero-field reference point, found {len(zeros)}"
        )
    if len(seq) < 2:
        raise ExcitonError("need at least two non-zero fields")
    e0 = zeros[0].e_x_eV
    F = np.array([r.field_eV_per_nm for r in seq], dtype=float)
    dE = np.array([r.e_x_eV - e0 for r in seq], dtype=float)

    beta = float(np.sum(dE * F**2) / np.sum(F**4))
    resid = float(np.linalg.norm(dE - beta * F**2))

    # The drift is measured on the two ENDS of the sweep, so a caller passing
    # unsorted results still gets the extremes rather than the first two rows.
    order = np.argsort(F)
    per_point = dE / F**2
    drift = float(per_point[order[-1]] - per_point[order[0]])
    return beta, resid, drift


def perturbative_ratio(results) -> float:
    """The largest ``|E_C| / (E_e + E_h)`` over the sweep.

    The paper's treatment takes the Coulomb term to first order. That is only
    defensible while this ratio is well below 1; this function reports it so the
    claim is checked against the computed numbers rather than assumed.
    """
    ratios = [r.coulomb_over_confinement for r in results]
    if not ratios:
        raise ExcitonError("no results to summarise")
    return float(max(ratios))
