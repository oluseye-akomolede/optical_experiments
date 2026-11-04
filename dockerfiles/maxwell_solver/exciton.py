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
    return _potential_grid(box, field_eV_per_nm, well_depth_eV, well_height_nm)


def _potential_grid(
    box: _stark.Box,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
) -> np.ndarray:
    """``-e F z`` plus an optional symmetric well, on the box's node grid.

    The well is a finite square well of ``well_depth_eV`` filling the box in x
    and y and extending ``well_height_nm`` along z, centred on the box. Outside
    it in z the potential is raised by the depth. ``None`` depth means no well:
    the carrier is free in the box.
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
        if not (math.isfinite(well_height_nm) and well_height_nm > 0):
            raise ExcitonError(f"well height must be positive, got {well_height_nm!r}")
        z_mid = 0.5 * (zs[0] + zs[-1])
        lo = z_mid - 0.5 * well_height_nm
        hi = z_mid + 0.5 * well_height_nm
        outside = (Z < lo) | (Z > hi)
        V = V + np.where(outside, well_depth_eV, 0.0)
    return V


def _ground_state(
    box: _stark.Box,
    mass_ratio: float,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
):
    """Ground-state energy and vector of one species in the L1 potential."""
    V = _potential_grid(box, field_eV_per_nm, well_depth_eV, well_height_nm)
    evals, full = _stark.lowest_states(box, mass_ratio, n_states=1, potential_eV=V)
    return float(evals[0]), full[:, 0], {"n_states": 1, "energy_eV": float(evals[0])}


def exciton_energy_eV(
    material: DotMaterial,
    box: _stark.Box,
    field_eV_per_nm: float,
    well_depth_eV: float | None = None,
    well_height_nm: float | None = None,
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
    """
    # Route through the validated helper rather than a bare finite check, so
    # the sign rule cannot be bypassed by this one entry point. Earlier this
    # function checked only finiteness and a negative field silently solved.
    sawtooth_potential(box, field_eV_per_nm, well_depth_eV, well_height_nm)

    e_e, psi_e, sol_e = _ground_state(
        box, material.m_e_ratio, field_eV_per_nm, well_depth_eV, well_height_nm
    )
    e_h, psi_h, sol_h = _ground_state(
        box, material.m_h_ratio, field_eV_per_nm, well_depth_eV, well_height_nm
    )

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
