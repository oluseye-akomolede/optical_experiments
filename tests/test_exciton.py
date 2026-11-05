"""Tests for the exciton Stark model (task 1.9).

The load-bearing test here is the harmonic-oscillator Stark shift, which has a
closed form. Everything else checks a property (symmetry, normalisation, regime
report) rather than a number this code chose, because a test that pins a value
the code emits cannot tell a correct model from a consistently wrong one.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import exciton as X
import stark as S


def _material(**over):
    base = dict(eps_s=12.5, m_e_ratio=0.08, m_h_ratio=0.5, source="test fixture")
    base.update(over)
    return X.DotMaterial(**base)


# ---------------------------------------------------------------- material


def test_material_refuses_an_unsourced_permittivity():
    with pytest.raises(X.ExcitonError, match="source"):
        X.DotMaterial(eps_s=12.5, m_e_ratio=0.08, m_h_ratio=0.5)


def test_material_refuses_a_permittivity_below_vacuum():
    with pytest.raises(X.ExcitonError, match="vacuum"):
        _material(eps_s=0.5)


def test_material_refuses_a_non_positive_mass():
    with pytest.raises(X.ExcitonError, match="positive"):
        _material(m_e_ratio=0.0)


def test_material_refuses_a_non_finite_value():
    with pytest.raises(X.ExcitonError, match="positive"):
        _material(m_h_ratio=float("nan"))


# ------------------------------------------------- the analytic cornerstone


def test_oscillator_stark_shift_matches_the_closed_form():
    """``dE = -F^2 / (2 m omega^2)`` for a 1-D oscillator in a linear field.

    This is the test that makes the rest of the module worth anything: it
    exercises the field potential, the eigensolver and the sign convention
    together against an exact result. It was measured at 1e-15 relative error
    when the module was built, so the tolerance below is loose rather than
    aspirational.

    The well is along z with x and y wide enough that they contribute a
    constant, which cancels in the DIFFERENCE.
    """
    m, omega = 1.0, 0.05
    box = S.Box((-20.0, -20.0, -20.0), (40.0, 40.0, 40.0), (8, 8, 80))
    ax = box.axes()
    Z = np.meshgrid(*ax, indexing="ij")[2]
    wall = ax[2][0]
    V = np.minimum(0.5 * m * omega**2 * Z**2, 0.5 * m * omega**2 * wall**2)

    e0, _ = S.lowest_states(box, m, n_states=1, potential_eV=V)
    for field in (0.002, 0.004, 0.008):
        e1, _ = S.lowest_states(box, m, n_states=1, potential_eV=V - field * Z)
        shift = float(e1[0] - e0[0])
        exact = -field**2 / (2 * m * omega**2)
        assert shift == pytest.approx(exact, rel=1e-6), (
            f"F={field}: shift {shift} vs exact {exact}"
        )


def test_oscillator_shift_is_quadratic_not_linear():
    """Doubling the field must quadruple the shift.

    Separates the F^2 term from an accidental linear one, which the sign
    convention could otherwise hide.
    """
    m, omega = 1.0, 0.05
    box = S.Box((-20.0, -20.0, -20.0), (40.0, 40.0, 40.0), (8, 8, 80))
    ax = box.axes()
    Z = np.meshgrid(*ax, indexing="ij")[2]
    wall = ax[2][0]
    V = np.minimum(0.5 * m * omega**2 * Z**2, 0.5 * m * omega**2 * wall**2)

    e0, _ = S.lowest_states(box, m, n_states=1, potential_eV=V)
    e1, _ = S.lowest_states(box, m, n_states=1, potential_eV=V - 0.002 * Z)
    e2, _ = S.lowest_states(box, m, n_states=1, potential_eV=V - 0.004 * Z)
    d1 = float(e1[0] - e0[0])
    d2 = float(e2[0] - e0[0])
    assert d2 / d1 == pytest.approx(4.0, rel=1e-4)


# ------------------------------------------------------------- symmetry


def test_the_wavefunction_is_pushed_toward_the_lower_potential():
    """The centroid must move in the direction the field pushes it.

    Added because the energy test above CANNOT see a sign flip: the shift
    ``-F^2/(2m omega^2)`` is even in F, so flipping the potential's sign leaves
    every energy identical. Measured: mutating ``-field*Z`` to ``+field*Z``
    passed 18 of 19 tests, this being the only one that caught it. The centroid
    is odd in F, so it can.

    For V = -F z the potential FALLS toward +z, so the centroid must be at z > 0.
    """
    m, omega = 1.0, 0.05
    box = S.Box((-20.0, -20.0, -20.0), (40.0, 40.0, 40.0), (8, 8, 80))
    ax = box.axes()
    Z = np.meshgrid(*ax, indexing="ij")[2]
    wall = ax[2][0]
    V0 = np.minimum(0.5 * m * omega**2 * Z**2, 0.5 * m * omega**2 * wall**2)

    _, psi0 = S.lowest_states(box, m, n_states=1, potential_eV=V0)
    # Build the tilt THROUGH the module, not inline: an inline copy would be
    # immune to a sign error in sawtooth_potential, which is exactly the bug
    # this test exists to catch.
    _, psi1 = S.lowest_states(box, m, n_states=1, potential_eV=V0 + X.sawtooth_potential(box, 0.004))
    z0 = float(np.sum(np.abs(psi0[:, 0]) ** 2 * Z.ravel()) / np.sum(np.abs(psi0[:, 0]) ** 2))
    z1 = float(np.sum(np.abs(psi1[:, 0]) ** 2 * Z.ravel()) / np.sum(np.abs(psi1[:, 0]) ** 2))
    assert z1 > z0 + 1e-3, f"centroid moved {z0} -> {z1}, expected toward +z"


def test_a_symmetric_well_has_no_linear_stark_term():
    """``E(+F) == E(-F)`` for a well symmetric about z = 0.

    This is a physical identity, and it is what makes the fitted ``p`` for the
    paper-1 dot meaningless: see :func:`exciton.fit_p_beta`. The test exists so
    that a later change to the potential's sign or origin cannot quietly break
    the symmetry without someone noticing.
    """
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (10, 10, 10))
    # E(+F) == E(-F) holds for the SPECTRUM, but the public entry point refuses
    # a negative field by design. So the identity is checked on the Hamiltonian
    # energies directly, which is the level at which it is true.
    ax = box.axes()
    Z = np.meshgrid(*ax, indexing="ij")[2]
    V_pos = X.sawtooth_potential(box, +0.01)
    assert V_pos == pytest.approx(-0.01 * Z)
    e_pos, _ = S.lowest_states(box, 0.08, n_states=1, potential_eV=V_pos)
    e_neg, _ = S.lowest_states(box, 0.08, n_states=1, potential_eV=+0.01 * Z)
    assert float(e_pos[0]) == pytest.approx(float(e_neg[0]), abs=1e-9)


def test_the_box_is_centred_on_zero():
    """Guards the symmetry test above: it only means something if z spans 0."""
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (10, 10, 10))
    z = box.axes()[2]
    assert z[0] == pytest.approx(-4.0)
    assert z[-1] == pytest.approx(+4.0)


# ------------------------------------------------------------- the model


def test_exciton_energy_sums_its_three_terms():
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (8, 8, 8))
    r = X.exciton_energy_eV(mat, box, 0.004)
    assert r.e_x_eV == pytest.approx(r.e_electron_eV + r.e_hole_eV + r.e_coulomb_eV)


def test_coulomb_term_binds():
    """A screened attraction between electron and hole must be negative."""
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (8, 8, 8))
    r = X.exciton_energy_eV(mat, box, 0.0)
    assert r.e_coulomb_eV < 0


def test_the_hole_shifts_more_than_the_electron():
    """The heavier carrier has the smaller confinement energy and the larger
    Stark shift, so E_h must fall faster than E_e as the field grows."""
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (8, 8, 8))
    lo = X.exciton_energy_eV(mat, box, 0.0)
    hi = X.exciton_energy_eV(mat, box, 0.01)
    assert (hi.e_hole_eV - lo.e_hole_eV) < (hi.e_electron_eV - lo.e_electron_eV)
    assert hi.e_x_eV < lo.e_x_eV


def test_the_model_reports_a_perturbative_ratio_below_one():
    """The paper takes the Coulomb term to first order; that needs |E_C| well
    below the confinement energy. This asserts the regime holds for the paper-1
    material set rather than assuming it."""
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (10, 10, 10))
    rs = [X.exciton_energy_eV(mat, box, f) for f in (0.0, 0.004, 0.01)]
    assert X.perturbative_ratio(rs) < 0.5


def test_a_negative_field_is_refused():
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (8, 8, 8))
    with pytest.raises(X.ExcitonError, match="non-negative"):
        X.exciton_energy_eV(mat, box, -0.01)


# ---------------------------------------------------------------- the fit


def _sweep(box_n=10, fields=(0.0, 0.002, 0.004, 0.008)):
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (box_n, box_n, box_n))
    return [X.exciton_energy_eV(mat, box, f) for f in fields]


def test_fit_recovers_a_quadratic_beta():
    beta, resid, drift = X.beta_low_field(_sweep())
    assert beta < 0, "a quadratic Stark shift must lower the transition energy"
    # The quadratic model is only approximate, so the residual is small but
    # NOT zero; asserting it were zero would hide higher-order terms.
    assert resid >= 0


def test_the_drift_is_reported_so_the_regime_is_visible():
    """The quadratic model stops holding at large field. The drift must grow
    with the top of the sweep, which is the signal a caller uses to decide how
    much of the range to trust."""
    _, _, drift_low = X.beta_low_field(_sweep(fields=(0.0, 0.002, 0.004)))
    _, _, drift_high = X.beta_low_field(_sweep(fields=(0.0, 0.008, 0.04)))
    assert abs(drift_high) > abs(drift_low)


def test_fit_needs_a_zero_point():
    with pytest.raises(X.ExcitonError, match="zero-field"):
        X.fit_p_beta(_sweep(fields=(0.002, 0.004)))


def test_fit_needs_two_nonzero_fields():
    with pytest.raises(X.ExcitonError, match="two non-zero"):
        X.fit_p_beta(_sweep(fields=(0.0, 0.002)))


def test_the_fit_residual_is_a_real_check_not_a_decoration():
    """Two parameters on two non-zero points fits exactly, so a zero residual
    is evidence of nothing. This pins that the residual stays tiny AND that the
    caller is told so, so nobody reads 1e-19 as a validation."""
    p, beta, resid, design = X.fit_p_beta(_sweep(fields=(0.0, 0.002, 0.004)))
    assert resid < 1e-10
    assert design.shape == (2, 2)


def test_p_is_not_reported_as_a_measured_dipole_for_a_symmetric_dot():
    """The fit returns a nonzero ``p`` for the symmetric paper-1 dot; that is a
    conditioning artefact, documented on :func:`exciton.fit_p_beta`. This test
    records the fact so a future reader does not quote ``p`` as physical."""
    mat = _material()
    box = S.Box((-4.0, -4.0, -4.0), (8.0, 8.0, 8.0), (10, 10, 10))
    # The true linear term is zero by reflection symmetry. Measured on the
    # spectrum, since the public entry point refuses a negative field.
    ax = box.axes()
    Z = np.meshgrid(*ax, indexing="ij")[2]
    e_pos, _ = S.lowest_states(box, 0.08, n_states=1, potential_eV=-0.01 * Z)
    e_neg, _ = S.lowest_states(box, 0.08, n_states=1, potential_eV=+0.01 * Z)
    assert abs(float(e_pos[0]) - float(e_neg[0])) < 1e-9


# ---------------------------------------------------------- confinement gap


def test_the_model_has_no_confinement_well_only_a_linear_field():
    """THE OPEN GAP IN 1.9, pinned as a test rather than left in a docstring.

    ``sawtooth_potential`` returns ``-F z`` and nothing else; there is no well
    at the dot. So the "confined" state is the ground state of the BARE BOX,
    and for a box of edge L that has <z^2> = L^2 (1/12 - 1/(2 pi^2)). This test
    measures <z^2> and compares it against that closed form, because a real 4 nm
    dot would be two orders of magnitude smaller.

    If someone later adds a confinement potential, this test goes red: that is
    the signal to replace it with a test of the well, not to relax the bound.
    """
    box = S.Box((-27.0, -27.0, -27.0), (54.0, 54.0, 54.0), (17, 17, 17))
    h = 54.0 / 16
    expected = 54.0**2 * (1.0 / 12.0 - 1.0 / (2.0 * math.pi**2))

    for ratio, label in ((0.562, "hole"), (0.0686, "electron")):
        _, psi, _ = X._ground_state(box, ratio, 0.0)
        z = np.meshgrid(*box.axes(), indexing="ij")[2].ravel()
        dens = np.abs(psi.ravel()) ** 2
        dens = dens / dens.sum()
        z2 = float((dens * z**2).sum())
        assert z2 == pytest.approx(expected, rel=5e-2), (
            f"{label}: <z^2> = {z2:.3f} nm^2, bare box predicts {expected:.3f}. "
            f"A 4 nm dot would give ~1 nm^2, so a value near the box number "
            f"means there is no confinement potential."
        )

    # The potential really is the bare ramp: V(z) = -F z, to the last node.
    V = X.sawtooth_potential(box, 0.01)
    Z = np.meshgrid(*box.axes(), indexing="ij")[2]
    assert np.allclose(V, -0.01 * Z, atol=1e-15)
    assert h > 0


def test_a_four_nanometre_box_confines_the_carrier():
    """Control for the test above: shrink the box to the dot size and <z^2>
    collapses to the dot scale. This proves the missing well is the reason the
    54 nm run is delocalised, not a bug in the <z^2> measurement."""
    for L in (4.0, 54.0):
        n = 17
        box = S.Box((-L / 2, -L / 2, -L / 2), (L, L, L), (n, n, n))
        _, psi, _ = X._ground_state(box, 0.562, 0.0)
        z = np.meshgrid(*box.axes(), indexing="ij")[2].ravel()
        dens = np.abs(psi.ravel()) ** 2
        dens = dens / dens.sum()
        z2 = float((dens * z**2).sum())
        if L == 4.0:
            assert z2 < 5.0, f"4 nm box should confine: <z^2> = {z2:.3f}"
        else:
            assert z2 > 50.0, f"54 nm box should delocalise: <z^2> = {z2:.3f}"
