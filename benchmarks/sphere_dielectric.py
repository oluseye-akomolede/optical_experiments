"""Analytic TE modes of a dielectric sphere inside a PEC sphere -- the
reference the multi-material feature is measured against (tier 2, the LEAD
runs this on the Orin; scipy is a dev-only dependency and never enters an
image).

GEOMETRY. A core of radius ``a`` and refractive index ``n``, concentric inside
a PEC sphere of radius ``b``. The core is the dielectric; the shell a..b is
vacuum; the PEC wall is at b.

THE FIELD. For TE modes (the radial component of E is zero) with angular
number l, write the field as r*psi(r):
    core  (r < a):  psi = j_l(k r)
    shell (a<r<b):  psi = A j_l(n k r) + B y_l(n k r)
and require psi and psi' continuous at a, and psi(b) = 0. Note the wavenumber:
the SHELL carries the index n, so its argument is n*k*r -- including at r = b,
where the PEC condition lives. Getting that wrong is a real trap: it leaves the
answer independent of n in the limit that should be most sensitive to it, and
the finder still returns plausible-looking roots. See the homogeneous control
below, which exists precisely to catch it.

THE DETERMINANT is the 3x3 system in (1, A, B); a mode is a k where it vanishes.

THREE CONTROLS, all of which must pass before any tolerance means anything:
  1. n = 1 reduces to the bare PEC sphere: psi(b) = 0 with psi = r*j_l(kr),
     i.e. spherical_jn(1, k*b) = 0. Residuals are ~1e-16 (they are printed).
  2. HOMOGENEOUS LIMIT, a -> 0: the shell then fills the cavity, and every mode
     must be exactly the vacuum mode divided by n (4.493409/n for l=1). This
     control is what caught the shell-wavenumber bug above; control 1 cannot,
     because it only probes n = 1.
  3. Raising n must LOWER every mode (a longer optical path). Monotone per mode.

The old plan text for this benchmark quoted "2.995606, 5.150168, 7.269414" as
the bare-sphere TE modes. THEY ARE NOT: spherical_jn(1, 2.995606) = 0.346, not
0, so a residual check at those numbers is off by 0.35 rather than 1e-16. The
correct zeros are 4.493409, 7.725252, 10.904122 (and the cylindrical jn_zeros(1)
are a third, different set: 3.8317, 7.0156, ...). Any tolerance built on the
quoted numbers would have been measuring the wrong spectrum.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import spherical_jn, spherical_yn

#: l = 1 is the lowest TE family and the one the benchmark measures.
L = 1

#: How many angular numbers the reference carries by default. The SOLVER has no
#: l cutoff -- it returns every mode the mesh can resolve -- so a reference that
#: stops at l = 1 makes every l >= 2 mode the solver finds look spurious. That
#: is exactly what happened: the Orin's empty-sphere run reported "8 spurious"
#: modes, and they were the l = 2 quintet plus the TE l = 1 triplet. See
#: ``spectrum``.
L_MAX = 4

#: The benchmark geometry. a/b = 1/3 with n = 2 separates the lowest modes
#: (measured gaps 1.25, 1.37, 1.41) so a misidentified root is visible.
A_OVER_B = 1.0 / 3.0
N_INDEX = 2.0


def _j_deriv(l, z):
    """d/dz of the spherical Bessel j_l, by recurrence.

    ``spherical_jn(..., derivative=True)`` returns a tuple for ARRAY input and
    a bare float for scalar input, which made an earlier version of this file
    raise inside the bisection. The recurrence is unambiguous for both.
    """
    return spherical_jn(l - 1, z) - (l + 1) / z * spherical_jn(l, z)


def _y_deriv(l, z):
    return spherical_yn(l - 1, z) - (l + 1) / z * spherical_yn(l, z)


def _psi(which, l, z):
    """``(value, derivative)`` of r*psi for the core (``'j'``) or shell."""
    if which == "j":
        return z * spherical_jn(l, z), spherical_jn(l, z) + z * _j_deriv(l, z)
    return z * spherical_yn(l, z), spherical_yn(l, z) + z * _y_deriv(l, z)


def determinant(k, a, b, n, l=L):
    """The TE matching determinant at wavenumber ``k`` (may be an array).

    THE GEOMETRY, stated once here because this file had it backwards and every
    value it emitted was for the wrong problem (see the module docstring's
    WHICH REGION CARRIES eps). The CORE, ``r < a``, is the dielectric and
    carries index ``n``; the SHELL, ``a < r < b``, is vacuum; the PEC wall is at
    ``b``. So the core's argument is ``n*k*r`` and the shell's -- INCLUDING the
    PEC condition at ``b`` -- is ``k*r``.

    THE BUILD, stated because the previous version got the ELIMINATION wrong
    while every n=1 control stayed green. The unknowns are the shell amplitudes
    (A, B) and the core amplitude C, in ONE common basis -- ``j_l`` and ``y_l``
    as plain VALUES on both sides of the interface:
        R(a)  :  A j_l(ka)   + B y_l(ka)   - C j_l(nka)      = 0
        R'(a) :  A k j_l'(ka) + B k y_l'(ka) - C nk j_l'(nka) = 0
        R(b)  :  A j_l(kb)   + B y_l(kb)                       = 0
    The mode condition is ``det = 0``, and this is the whole function.

    WHY THE ONE-LINE FORM WAS WRONG. An earlier version collapsed the first two
    rows by Cramer's rule and wrote the result as a single expression. That
    collapse mixed the VALUE Wronskian with the DERIVATIVE Wronskian of the pair
    (r j_l(kr), r y_l(kr)) -- two different normalisations of the same 2x2 --
    which happens to be a pure rescaling, and therefore leaves the ROOTS at
    n = 1 untouched while moving them at every n > 1. Measured: the collapsed
    form gives TE l=1 at 3.8552 / 3.2039 for n = 2 / 3, the 3x3 above gives
    3.9215 / 2.9762, and the lead's independent oracle says 3.9215 / 2.9762.
    The collapse is exactly the kind of algebra that reads correct and is not.

    NO eps APPEARS IN TE, and that is physics rather than an omission. eps enters
    only through the derivative-continuity row, and the TE family (E tangential
    to the spheres) has no radial E component, so the eps-weighted combination
    never appears -- only the ``n`` that changes the core's WAVENUMBER. Since
    1/n = 1/n^2 = 1 at n = 1, no n=1 control can see a misplaced eps here; the
    ORACLE table below is the control that can.
    """
    k = np.asarray(k, dtype=float)
    nka, ka, kb = n * k * a, k * a, k * b

    # Build the (..., 3, 3) stack with the MATRIX axes LAST: np.stack puts a new
    # axis FIRST, so the rows go in last along axis 1 and the columns along 2.
    rows = np.stack([
        np.stack([spherical_jn(l, ka), spherical_yn(l, ka),
                  -spherical_jn(l, nka)], axis=1),
        np.stack([k * _j_deriv(l, ka), k * _y_deriv(l, ka),
                  -n * k * _j_deriv(l, nka)], axis=1),
        np.stack([spherical_jn(l, kb), spherical_yn(l, kb),
                  np.zeros_like(kb)], axis=1),
    ], axis=1)
    return np.linalg.det(rows)


def find_modes(a, b, n, l=L, k_max=20.0, n_points=40000, tol=1e-13):
    """Every TE_l mode with k in (0, k_max), ascending.

    Brackets are found on a grid and closed by bisection, so a mode is reported
    with the width of the final bracket as its error -- no root is taken from a
    scan minimum, which would report the grid spacing instead.
    """
    grid = np.linspace(0.01, k_max, n_points)
    values = determinant(grid, a, b, n, l)
    crossings = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)

    modes = []
    for i in crossings:
        lo, hi = float(grid[i]), float(grid[i + 1])
        f_lo = float(values[i])
        for _ in range(200):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            f_mid = float(determinant(np.array([mid]), a, b, n, l)[0])
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        modes.append(0.5 * (lo + hi))
    return np.array(modes)


def check_controls(verbose=True):
    """The TE controls, each returning a number rather than a yes/no.

    Returns ``(max_residual, max_empty_error, max_full_error, monotone)`` so a
    caller can assert on magnitudes instead of trusting a printed table.

    THE TWO LIMITS, and they were BOTH WRONG before this geometry fix. A
    dielectric CORE of radius ``a`` inside a vacuum shell a..b, PEC at b:

      a -> 0   the core vanishes: the cavity is entirely vacuum, so every mode
               must go to the EMPTY spectrum (bare PEC sphere). The old control
               asserted ``vacuum/n`` -- the a -> b answer -- which only made
               sense when the reference put n in the shell.
      a -> b   the core fills the cavity: every mode is the empty mode / n, by
               the exact scaling of the wave equation with eps = n^2.

    These two bracket the geometry and are checked separately, because a single
    limit cannot tell a swapped core from a swapped shell -- which is the defect
    they exist to catch.

    Each returned error is DIVIDED BY the approach distance: the convergence is
    first order (measured, TE and TM, a clean factor of 10 per decade), so the
    raw error is just the step in disguise and the constant is the thing worth
    bounding.
    """
    b = 1.0
    APPROACH = 1e-7

    # Control 1: n = 1 is the bare PEC sphere (core and shell identical).
    vacuum = find_modes(A_OVER_B, b, 1.0)
    residuals = np.abs(spherical_jn(L, vacuum * b))

    # Control 2a: a -> 0 is the EMPTY spectrum, in every TE_l family.
    empty_error = 0.0
    for n in (1.0, 2.0, 3.0):
        for l in (1, 2, 3):
            got = find_modes(APPROACH, b, n, l=l)
            want = find_modes(APPROACH, b, 1.0, l=l)
            m = min(len(got), len(want))
            if m:
                empty_error = max(empty_error,
                                  float(np.max(np.abs(got[:m] - want[:m])
                                               / want[:m])))

    # Control 2b: a -> b is the empty spectrum / n, in every TE_l family.
    full_error = 0.0
    for n in (2.0, 3.0):
        for l in (1, 2, 3):
            got = find_modes(b * (1 - APPROACH), b, n, l=l)
            want = find_modes(b * (1 - APPROACH), b, 1.0, l=l) / n
            m = min(len(got), len(want))
            if m:
                full_error = max(full_error,
                                 float(np.max(np.abs(got[:m] - want[:m])
                                              / want[:m])))

    # Control 3: raising n lowers EVERY mode.
    monotone = True
    series = [find_modes(A_OVER_B, b, n) for n in (1.0, 2.0, 3.0)]
    for lower, higher in zip(series, series[1:]):
        shared = min(len(lower), len(higher))
        monotone = monotone and bool(
            np.all(higher[:shared] < lower[:shared])
        )

    if verbose:
        print(f"control 1 (n=1 -> spherical_jn(1,kb)=0): "
              f"max |residual| = {residuals.max():.2e}")
        print(f"control 2a (a->0 -> EMPTY spectrum):      "
              f"error/approach = {empty_error:.2e}")
        print(f"control 2b (a->b -> EMPTY spectrum / n):  "
              f"error/approach = {full_error:.2e}")
        print(f"control 3 (raising n lowers every mode): "
              f"{'pass' if monotone else 'FAIL'}")
        print()
        print("modes at a/b = 1/3:")
        for n, modes in zip((1.0, 2.0, 3.0), series):
            print(f"  n={n}: {np.round(modes[:5], 6)}")

    # Normalised by the approach: the bound is on the CONSTANT, measured ~67
    # for TE and ~20 for TM. 200 leaves room for the mesh-side uses while still
    # failing loudly if a limit comes out 10x worse than it is now.
    return (residuals.max(), empty_error / APPROACH, full_error / APPROACH,
            monotone)


def bound_violations(a, b, n, l_max=L_MAX):
    """Reference roots that violate the min-max bound -- a code-free control.

    PHYSICS, no root finder involved. ``k^2`` is a Rayleigh quotient with an
    eps-weighted mass, so raising eps anywhere can only LOWER every ordered
    eigenvalue. Per (family, l), the sphere keeping them separate:

        k_i(a -> b, n)  <=  k_i(partial fill)  <=  k_i(a -> 0)

    and the fully-filled spectrum is exactly the empty spectrum / n. So every
    reference root must lie in ``[k_empty/n, k_empty]``, ordinal by ordinal.

    WHY THIS IS THE STRONGEST CONTROL HERE. It is independent of the root
    finder, of the determinant, and of the field convention: it cannot be fooled
    by an algebra error inside the thing it judges, which is the failure mode
    this module has already had three times. It would have caught the swapped
    geometry on its own -- measured on the WRONG-geometry TM roots, the first
    TWO ordinals of every l violated, e.g. n=3 TM l=1 gave 0.8585 against a
    lower bound of 2.7437/3 = 0.9146, which is impossible for a dielectric
    filling ANY fraction of the cavity (eps >= 1 can only pull a mode down
    toward k_empty/n, never below it).

    Returns a list of ``(family, l, ordinal, k, lo, hi)`` for the violations,
    so an empty list is the pass and the entries say which one broke.
    """
    out = []
    for family, finder in (("TE", find_modes), ("TM", find_modes_TM)):
        for l in range(1, l_max + 1):
            empty = finder(a, b, 1.0, l=l)
            part = finder(a, b, n, l=l)
            m = min(len(empty), len(part))
            for i in range(m):
                lo, hi = empty[i] / n, empty[i]
                if not (lo - 1e-12 <= part[i] <= hi + 1e-12):
                    out.append((family, l, i, float(part[i]),
                                float(lo), float(hi)))
    return out


def determinant_TM(k, a, b, n, l=L):
    """The TM matching determinant at wavenumber ``k`` (may be an array).

    TM is where eps does its work, and the place it enters is easy to get
    wrong -- this function has already been wrong once, in this file. It is NOT
    a 3x3. The unknown amplitudes (A, B) of the shell's j_l and y_l are fixed
    by TWO continuity conditions at r = a; the PEC condition at r = b is then a
    THIRD equation in those same two unknowns, i.e. the condition that makes
    the system overdetermined. That is the mode condition, and it is what is
    returned. Writing it as a 3x3 -- treating the core amplitude as a third
    unknown -- silently shifts the eps weight onto the wrong row and gives a
    finder whose first root is 2.081576 instead of the correct 2.743678.

    Matching at r = a (eps = n^2 in the CORE, 1 in the vacuum shell):
        r*psi  continuous       (tangential H)   A v + B w  = u
        (1/eps)(r*psi)' cont.   (tangential E)   A dv + B dw = du/n
    PEC at r = b (tangential E vanishes, evaluated in the VACUUM shell):
        A dvb + B dwb = 0

    Solving the first two for (A, B) and substituting gives the expression
    below, whose only eps factor is the 1/n on the core's ``du``.

    WHY 1/n AND NOT 1/n^2 -- the second defect in this function, and the one
    that survived the geometry fix. Both this file and the lead's oracle agree
    at n=1 whatever the weight is, because 1/n^2 = 1/n = 1 there; the weight is
    only observable for n != 1, which is why no n=1 control could see it.

    The derivation, by the chain rule. For TM the tangential E is proportional
    to ``(1/eps) d(psi)/dr`` with ``psi = r R``. Inside the core
    ``psi(r) = z j_l(z)`` at ``z = n k r``, so ``d/dr = n k d/dz`` -- ONE factor
    of n from the chain rule. The eps weight contributes ``1/eps = 1/n^2``, so
    the whole thing is ``n / n^2 = 1/n`` on the ``z``-derivative: the two powers
    of n cancel down to one. The old code applied ``1/eps`` directly to the
    ``z``-derivative and so dropped the chain-rule n, leaving ``1/n^2`` -- which
    is exactly one factor of n too small.

    TE has no eps at all for the same reason read the other way: its continuity
    rows are ``E_t ~ psi`` and ``H_t ~ d(psi)/dr`` with ``mu = 1``, so no eps
    weight appears in either, and only the n inside the core's WAVENUMBER
    survives.

    MEASURED against the lead's independent oracle (which shares no code with
    this file): with 1/n the first TM l=1 root is 2.74371 / 2.4873 / 2.3719 at
    n = 1 / 2 / 3 and l=2 gives n=2 3.7927, n=3 3.7522 -- every one exactly the
    oracle's value. With 1/n^2 it is 2.74371 / 2.42225 / 2.34835, i.e. correct
    only at n=1.

    THE ARGUMENTS, which were swapped here too: the core (``r < a``) is the
    dielectric, so its argument is ``n*k*r``; the shell (``a < r < b``) is
    vacuum, so its argument -- including the PEC radius ``b`` -- is ``k*r``.
    """
    k = np.asarray(k, dtype=float)
    nka, ka, kb = n * k * a, k * a, k * b

    u, du = _psi("j", l, nka)                # core (index n), at r = a
    v, dv = _psi("j", l, ka)                 # shell j-term (vacuum), at r = a
    w, dw = _psi("y", l, ka)                 # shell y-term (vacuum), at r = a
    _, dvb = _psi("j", l, kb)                # PEC: derivative at b, vacuum k
    _, dwb = _psi("y", l, kb)

    wronskian = v * dw - w * dv              # A, B as Cramer ratios
    a_amp = (u * dw - w * du / n) / wronskian
    b_amp = (v * du / n - u * dv) / wronskian
    return a_amp * dvb + b_amp * dwb


def _find(det, a, b, n, l, k_max, n_points, tol):
    """Brackets on a grid, closed by bisection. Shared by TE and TM."""
    grid = np.linspace(0.01, k_max, n_points)
    values = det(grid, a, b, n, l)
    crossings = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    modes = []
    for i in crossings:
        lo, hi = float(grid[i]), float(grid[i + 1])
        f_lo = float(values[i])
        for _ in range(200):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            f_mid = float(det(np.array([mid]), a, b, n, l)[0])
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        modes.append(0.5 * (lo + hi))
    return np.array(modes)


def find_modes_TM(a, b, n, l=L, k_max=20.0, n_points=40000, tol=1e-13):
    """Every TM_l mode with k in (0, k_max), ascending."""
    return _find(determinant_TM, a, b, n, l, k_max, n_points, tol)


#: The plan's quoted TM zero. It is correct -- see control 1.
PLAN_TM_ZERO = 2.743678


def _dielectric_first_roots(n, count=5):
    """First ``count`` roots of tan(z) = z/(1-z^2) at argument z = n*k*b.

    This is the TM_1 condition for a dielectric sphere of index n with NO PEC
    wall. It is the a -> 0 limit of this file's geometry (the shell swallows
    the core), and it is also exactly the n = 1 case at finite a. Both facts
    are controls below, so it is computed here rather than tabulated.
    """
    f = lambda z: spherical_jn(0, z) - spherical_jn(1, z) / z
    grid = np.linspace(1e-3, 20.0, 200000)
    values = f(grid)
    crossings = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    out = []
    for i in crossings:
        lo, hi = float(grid[i]), float(grid[i + 1])
        f_lo = float(values[i])
        for _ in range(200):
            if hi - lo < 1e-15:
                break
            mid = 0.5 * (lo + hi)
            f_mid = float(f(np.array([mid]))[0])
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        out.append(0.5 * (lo + hi))
        if len(out) >= count:
            break
    return np.array(out)


def tm_controls(verbose=True):
    """The TM controls, returning numbers rather than a yes/no.

    Returns ``(residual, monotone, independent)``.

    WHAT WENT WRONG HERE, because the shape recurs. This function has passed
    three times over three different (and wrong) TM finders in one session:

      1. A 3x3 with the eps weight on the wrong row. It returned roots
         (2.081576, 5.940370) and control 1 -- which asserted a near-zero
         residual of ``[z j_1]'`` AT ITS OWN ROOTS -- passed, because it was
         checking the finder against itself. A residual of the wrong equation.
      2. A ``d/dz[z f_l]`` that dropped the ``f_l`` term, which made the PEC
         condition look like it had the right zeros. Control 1 still passed.
      3. Me reporting, on the strength of (2), that the plan's number was a
         mischaracterization. It was not; I was.

    The fix is not a better residual, it is a DIFFERENT reference: the roots
    of ``tan z = z/(1-z^2)`` computed by a function that shares no code with
    the determinant. An independent reference cannot be fooled by an
    algebra error inside the thing it judges.

    ``independent`` is a THIRD-WAY check that matters here: that
    ``[z j_1]' = 0`` and ``tan z = z/(1-z^2)`` are the SAME equation. The plan
    calls the TM condition ``d/dr[r j_l] = 0`` and quotes 2.7437 for it. Both
    are right, and they are right together: expanding the derivative gives
    ``z j_0 - j_1 = 0``, which is exactly ``tan z = z/(1-z^2)`` after
    multiplying by ``z^2/sin z``. This is asserted rather than asserted-in-a-
    comment, because I twice convinced myself the two were different
    equations, and each time the arithmetic I did to convince myself was the
    error.

    CONTROL 1: n = 1, a/b = 1/3. Core and shell share an index, so the
    matching at r = a is between identical functions. The Wronskian
    ``j_1 y_1' - j_1' y_1 = 1/z^2`` never vanishes, which forces A = 1, B = 0:
    the shell is pure ``j_1`` and the geometry reduces to the bare PEC sphere.
    Its TM_1 condition is ``[z j_1]' = 0``. Measured first root: 2.743707.

    CONTROL 2 (a -> 0, the limit the most sensitive to the eps weight): the
    CORE vanishes, so the cavity is pure vacuum and every mode goes to the
    EMPTY spectrum -- NOT to the dielectric answer. (Before the geometry fix
    this control asserted ``_dielectric_first_roots(n)/n``, i.e. the a -> b
    answer, which is the opposite limit for a core-dielectric geometry; it
    reported a 200% error and the gate printed PASS anyway.) The a -> b limit
    is the one that gives ``_dielectric_first_roots(n)/n``, and it is checked
    separately as control 2b.

    THE EPS-WEIGHT SENSITIVITY, which is why the independent reference above
    is the check that matters. At n = 1 the eps weight is 1/1 = 1, so control
    1 is BLIND to whether eps is applied at all; and at a -> 0 the core
    derivative ``du`` is evaluated in the vanishing-core limit where the
    ``1/n^2`` prefactor cancels against the same-order asymptotics, so control
    2 is nearly blind too. Measured: dropping the eps weight entirely leaves
    controls 1 and 2 PASSING to 1e-14 while shifting every finite-a mode by
    8-12%. The ``independent`` check is what catches that class, because it
    compares the finder against an equation it shares no code with.

    A third control comparing against a solid-dielectric Mie denominator was
    written and DELETED -- it was broken (empty result, wrong equation) and
    its job is already done better by ``independent``. The measurements that
    justify the deletion are recorded at the deletion site below.
    """
    b = 1.0
    reference = _dielectric_first_roots(1.0, count=4)

    # Control 1: n = 1 is the bare PEC sphere.
    vacuum = find_modes_TM(A_OVER_B, b, 1.0)
    residual = float(np.max(np.abs(vacuum[:4] - reference[:4])))

    # The third-way check: the plan's name for the condition and the plan's
    # number for it describe the SAME equation.
    pec_ref = _pec_zero(1, count=4)
    independent = float(np.max(np.abs(pec_ref[:4] - reference[:4])))

    # The finite-a control REMOVED, deliberately, and this is the record of
    # why -- because a control that is deleted without a reason gets reinvented.
    #
    # It compared find_modes_TM at finite a against ``_solid_dielectric``, a
    # solid-dielectric-sphere Mie denominator written for the purpose. That
    # function was WRONG, and it was wrong in the rig's signature shape: it
    # returned an EMPTY array (no roots found) rather than raising, so a
    # caller reading its output saw "no modes here" instead of "I failed".
    # Measured before deleting it:
    #
    #   * at n = 1, where a solid dielectric sphere IS the bare PEC sphere,
    #     _solid_dielectric(1.0, a=1/3, 3) = [4.5316, 8.2311, 15.8705] while
    #     the independent _pec_zero(1, 3) = [2.7437, 6.1168, 9.3166]. Its first
    #     root 4.5316 is not a TM_1 root at all (they are 2.7437, 6.1168,
    #     9.3166) -- so it was not merely imprecise, it was the wrong equation.
    #   * in the a -> 0 limit it returned [] where the independent
    #     _dielectric_first_roots(n)/n reference gives [1.3719, 3.0584].
    #
    # The root-causing attempt (_corrected_ denominator, Riccati-Bessel
    # n*psi_1(nka)*psi_1'(ka) - psi_1(ka)*psi_1'(nka)) ALSO found no roots on
    # the same grid -- including at n = 1, where it must reduce to
    # [z j_1]' = 0. So the defect is in the reference's root-finding, not just
    # its formula, and fixing it properly means rebuilding a Mie solver.
    #
    # That rebuild is NOT worth doing, because the job this control existed to
    # do is already done by a BETTER reference. The stated purpose was "a
    # reference for the n != 1, finite-a regime, since control 1 is blind to
    # eps and control 2 is nearly blind". ``_pec_zero`` is that reference and
    # it is strictly stronger than _solid_dielectric would have been: it is
    # computed from a DIFFERENT equation ([z j_1]' = 0) than the determinant
    # it judges, so an algebra error inside the finder cannot fool it -- which
    # is the same argument the docstring above makes for preferring it to a
    # self-residual. Measured agreement at n = 1, a/b = 1/3: 1.3e-14.
    #
    # So the honest state is: the finder has ONE independent finite-a check
    # (that one), not two. Removing the broken second is better than shipping a
    # red control or a control that quietly checks nothing.

    # Control 3: raising n lowers every mode.
    monotone = True
    series = [find_modes_TM(A_OVER_B, b, n) for n in (1.0, 2.0, 3.0)]
    for lower, higher in zip(series, series[1:]):
        shared = min(len(lower), len(higher))
        monotone = monotone and bool(np.all(higher[:shared] < lower[:shared]))

    if verbose:
        print(f"TM control 1 (n=1 -> [z j_1]'=0, first {PLAN_TM_ZERO}): "
              f"max |err| = {residual:.2e}")
        print(f"TM independent check: |roots of [z j_1]'=0 - roots of "
              f"tan z=z/(1-z^2)| = {independent:.2e}  (the plan's name and "
              f"the plan's number are the same equation)")
        print(f"TM control 2a (a->0 -> EMPTY spectrum): error/approach "
              f"= {_homogeneous_error(b)[0]:.2e}")
        print(f"TM control 2b (a->b -> EMPTY spectrum / n): error/approach "
              f"= {_homogeneous_error(b)[1]:.2e}")
        print(f"TM control 3 (raising n lowers every mode): "
              f"{'pass' if monotone else 'FAIL'}")
        print()
        print("TM modes at a/b = 1/3:")
        for n, modes in zip((1.0, 2.0, 3.0), series):
            print(f"  n={n}: {np.round(modes[:5], 6)}")

    return residual, monotone, independent, *_homogeneous_error(b)


def _homogeneous_error(b):
    """The two limits of the CORE-dielectric geometry, as max relative errors.

    Both were wrong before the geometry fix, and this is what the wrong
    expectations cost: the old version compared ``find_modes_TM(1e-7, b, n)``
    against ``_dielectric_first_roots(n)/n``, i.e. it asserted that a vanishing
    core gives the DIELECTRIC answer. For a core-dielectric geometry a
    vanishing core removes the dielectric, so the answer is the EMPTY spectrum
    -- and the control reported a 200% error while the gate still printed PASS,
    which is the "a control that cannot fail" shape this file keeps meeting.

    Returns ``(empty_error, full_error)``, each already DIVIDED BY the approach
    distance, because the convergence is exactly first order and the raw error
    is meaningless without it. Measured (TM, a -> b):

        approach   raw worst error
        1e-5       2.061e-04
        1e-6       2.061e-05
        1e-7       2.061e-06
        1e-8       2.061e-07

    A clean factor of 10 per decade. So the limit is right and the residual is
    pure O(b - a); normalising makes the control a bound on the CONSTANT
    (measured ~20) rather than on the step, which is what a gate should test.
    An absolute threshold here would just be the approach distance in disguise,
    and would have to be loosened every time the step changed.
    """
    APPROACH = 1e-7
    empty_error = 0.0
    full_error = 0.0
    for n in (1.0, 2.0, 3.0):
        for l in (1, 2, 3):
            got = find_modes_TM(APPROACH, b, n, l=l)
            want = find_modes_TM(APPROACH, b, 1.0, l=l)
            m = min(len(got), len(want))
            if m:
                empty_error = max(empty_error, float(np.max(
                    np.abs(got[:m] - want[:m]) / want[:m])))
    for n in (2.0, 3.0):
        for l in (1, 2, 3):
            got = find_modes_TM(b * (1 - APPROACH), b, n, l=l)
            want = find_modes_TM(b * (1 - APPROACH), b, 1.0, l=l) / n
            m = min(len(got), len(want))
            if m:
                full_error = max(full_error, float(np.max(
                    np.abs(got[:m] - want[:m]) / want[:m])))
    return empty_error / APPROACH, full_error / APPROACH


def _pec_zero(l, count=4):
    """First ``count`` zeros of ``d/dz[z j_l(z)]`` -- the BARE PEC sphere."""
    f = lambda z: _pec_condition(l, z)
    grid = np.linspace(0.05, 20.0, 200000)
    values = f(grid)
    crossings = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    out = []
    for i in crossings:
        lo, hi = float(grid[i]), float(grid[i + 1])
        f_lo = float(values[i])
        for _ in range(200):
            if hi - lo < 1e-15:
                break
            mid = 0.5 * (lo + hi)
            f_mid = float(f(np.array([mid]))[0])
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        out.append(0.5 * (lo + hi))
        if len(out) >= count:
            break
    return np.array(out)

def _pec_condition(l, z):
    """d/dz [z*j_l(z)] at scalar z -- the BARE PEC-sphere TM condition.

    Note the shape. ``d/dz[z f_l] = f_l + z f_l'``, NOT ``z f_l'``: dropping
    the ``f_l`` term is a mistake I made twice while writing this file, and the
    second time it produced a "contradiction" I nearly reported as a defect in
    the plan. With the term dropped, ``[z j_1]'`` looks like its zeros are
    2.081576 -- they are not. With it, ``d/dz[z j_1] = z j_0 - j_1``, whose
    first zero is 2.743707, which IS the plan's number.

    Expanded once here so the identity is visible rather than asserted:
        z*j_0(z) - j_1(z) = z*spherical_jn(0,z) - spherical_jn(1,z)
    """
    return (spherical_jn(l, z)
            + z * (spherical_jn(l - 1, z) - (l + 1) / z * spherical_jn(l, z)))


def spectrum(a, b, n, l_max=L_MAX, count=8):
    """Every mode of the layered sphere up to ``l_max``, as multiplets.

    Returns a list of ``(kb, l, family, multiplicity)`` tuples sorted by ``kb``,
    where ``family`` is ``"TE"`` or ``"TM"`` and ``multiplicity`` is ``2*l+1``.

    WHY THIS EXISTS. ``find_modes``/``find_modes_TM`` evaluate ONE ``l``, and
    the module constant ``L = 1`` meant the reference only ever held the l = 1
    family. A cubic cavity's spectrum is a union over l with degeneracy 2l+1,
    so a reference at l = 1 alone is not an incomplete reference -- it is a
    different spectrum, and every higher-l mode the solver correctly finds reads
    as an error. Measured on the Orin: the empty sphere's "8 spurious" modes
    were 2.7501 (x3) and 3.878-3.882 (x5), i.e. the TM l = 2 and TE l = 2
    multiplets, which the l = 1-only reference simply did not contain.

    Each multiplet is emitted ONCE at its ``kb`` with its multiplicity, rather
    than expanded into 2l+1 copies. Expanding them would let a scorer match
    individual computed modes against individual reference modes and report a
    good score while a whole multiplet's worth of modes was missing -- the
    "a summed count hides a dead direction" shape. Matching is on the multiplet,
    so the count is part of the claim.
    """
    out = []
    for l in range(1, l_max + 1):
        for family, finder in (("TE", find_modes), ("TM", find_modes_TM)):
            kb = finder(a, b, n, l=l)
            for k in kb[:count]:
                out.append((float(k), l, family, 2 * l + 1))
    out.sort(key=lambda t: t[0])
    return out


def cluster_modes(computed, tol=0.02):
    """Group a sorted array of computed ``kb`` into degenerate multiplets.

    Two adjacent values belong to the same cluster when they agree within a
    RELATIVE ``tol``. The tolerance is not a fitting knob: a solver's
    degenerate modes agree to its residual (measured ~1e-12 at 2e-12 relative),
    while distinct multiplets are separated by much more than 2% at this
    geometry (measured gaps 1.25, 1.37, 1.41 on the lowest TE family). So 2%
    sits in a wide empty band and the clustering is not sensitive to it --
    which is asserted in the runner by perturbing the tolerance and checking the
    cluster count does not move.
    """
    computed = np.sort(np.asarray(computed, float))
    clusters = []
    for k in computed:
        if clusters and abs(k - clusters[-1][-1]) <= tol * abs(clusters[-1][-1]):
            clusters[-1].append(float(k))
        else:
            clusters.append([float(k)])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def score_multiplets(computed, reference, tol=0.02, n_requested=None):
    """Match computed clusters to reference multiplets, one-to-one.

    ``reference`` is ``spectrum``'s list of ``(kb, l, family, multiplicity)``.
    Returns a dict with, per anchored cluster: the centre, the mode count, the
    reference it matched, and the relative error -- plus the coverage and the
    list of clusters that matched nothing.

    A cluster counts as SPURIOUS only when no reference multiplet lies within
    ``tol`` of it. Reporting a cluster as spurious because it failed to match a
    reference that was never in the reference list is not a spurious mode, it is
    a truncated reference -- the distinction this whole function exists to
    enforce, because the first version of the scorer got it wrong.

    ``n_requested`` is the number of modes the solver was ASKED for. That is
    what separates a legitimate partial top multiplet from a genuinely missing
    mode, and position alone CANNOT do it -- measured: three modes at the lowest
    multiplet and three at the second is a real shortfall of the second, but the
    second is also the topmost cluster, so a position-only rule called it a
    cutoff and the check silently stopped being able to fail. Given
    ``n_requested``, a partial top cluster is a cutoff only when the solver
    returned everything it was asked for; otherwise the shortfall is real.
    When ``n_requested`` is None nothing is excused.
    """
    clusters = cluster_modes(computed, tol)
    ref = [(float(kb), int(l), fam, int(m)) for kb, l, fam, m in reference]
    used = [False] * len(ref)
    matched, unmatched = [], []
    for centre, count in clusters:
        best, best_d = None, np.inf
        for j, (kb, l, fam, m) in enumerate(ref):
            if used[j]:
                continue
            d = abs(kb - centre) / kb
            if d < best_d:
                best, best_d = j, d
        if best is not None and best_d <= tol:
            used[best] = True
            kb, l, fam, m = ref[best]
            matched.append({
                "centre": centre, "count": count, "reference_kb": kb,
                "l": l, "family": fam, "reference_multiplicity": m,
                "rel_error": best_d, "count_matches": count == m,
            })
        else:
            unmatched.append({"centre": centre, "count": count,
                              "nearest_rel": float(best_d)
                              if best is not None else None})

    # The mode-count cutoff is only excusable when the solver delivered the
    # count it was asked for AND the partial cluster is the topmost one.
    n_returned = int(np.sort(np.asarray(computed, float)).size)
    delivered_all = (n_requested is not None and n_returned >= n_requested)
    top = max((m["centre"] for m in matched), default=None)
    for m in matched:
        m["truncated_at_request"] = bool(
            not m["count_matches"] and delivered_all and top is not None
            and m["centre"] >= top - 1e-12
        )

    return {
        "matched": matched,
        "spurious": unmatched,
        "n_clusters": len(clusters),
        "n_matched": len(matched),
        "n_returned": n_returned,
        "n_requested": n_requested,
        "worst_rel_error": (max(m["rel_error"] for m in matched)
                            if matched else None),
        # Real count mismatches: a multiplet short of modes with no cutoff to
        # excuse it. This is the field that must be EMPTY for a good run, and
        # it must be able to be non-empty -- see the docstring on n_requested.
        "count_mismatches": [m for m in matched
                             if not m["count_matches"]
                             and not m["truncated_at_request"]],
        "truncated_at_request": [m for m in matched
                                 if m["truncated_at_request"]],
        "unmatched_reference": [
            {"kb": kb, "l": l, "family": fam, "multiplicity": m}
            for j, (kb, l, fam, m) in enumerate(ref) if not used[j]
        ],
    }


# The lead's independent oracle table, hard-coded here as a THIRD-OPINION
# control. Every value was produced by ``~/mse-bench-lead-oracle.py``, which
# shares no code, no helper and no convention with this file -- it builds the
# TE/TM determinants from the shell-field products directly. The n=1 column is
# the bare PEC sphere (closed form). This is what catches an error that every
# n=1 control is blind to BY CONSTRUCTION: at n=1 every plausible eps weight
# equals 1, so the whole family of eps-placement mistakes is invisible there.
# That is exactly how the 1/n-vs-1/n^2 defect survived the previous fix.
ORACLE = {
    # (family, l, n): [first root, second root]
    ("TE", 1, 1.0): [4.4934], ("TM", 1, 1.0): [2.7437],
    ("TE", 2, 1.0): [5.7635], ("TM", 2, 1.0): [3.8702],
    ("TE", 1, 2.0): [3.9215, 5.6759], ("TM", 1, 2.0): [2.4873, 5.0123],
    ("TE", 2, 2.0): [5.4795], ("TM", 2, 2.0): [3.7927],
    ("TE", 1, 3.0): [2.9762, 5.0204], ("TM", 1, 3.0): [2.3719, 4.1294],
    ("TE", 2, 3.0): [4.2886], ("TM", 2, 3.0): [3.7522, 5.3405],
}
ORACLE_TOL = 1e-4


def oracle_errors(verbose=True):
    """Worst |root - oracle| over the whole table, and the count checked.

    Returns ``(worst_error, n_checked, details)``. A table this size is worth
    checking in full: a single wrong cell is a wrong physics claim, and the
    tolerance is 1e-4 because the oracle's own roots are quoted to 4 decimals.
    """
    worst, checked, details = 0.0, 0, []
    for (family, l, n), expected in sorted(ORACLE.items()):
        finder = find_modes if family == "TE" else find_modes_TM
        got = finder(A_OVER_B, 1.0, n, l=l, k_max=6.0, n_points=20001)
        for i, want in enumerate(expected):
            checked += 1
            if i >= len(got):
                details.append((family, l, n, i, None, want, float("inf")))
                worst = float("inf")
                continue
            err = abs(float(got[i]) - want)
            worst = max(worst, err)
            if err > ORACLE_TOL:
                details.append((family, l, n, i, float(got[i]), want, err))
    if verbose:
        for family, l, n, i, got, want, err in details:
            print(f"  oracle MISMATCH {family} l={l} n={n} ord={i}: "
                  f"got {got} want {want} ({err:.2e})")
    return worst, checked, details


def main(argv):
    max_residual, empty_error, full_error, mono_te = check_controls()
    tm_res, tm_mono, tm_disc, tm_empty, tm_full = tm_controls()
    oracle_worst, oracle_checked, _ = oracle_errors()

    # THE BOUND, as a gate rather than a printout. Independent of every finder
    # in this file: eps >= 1 can only lower an ordered eigenvalue, so no
    # reference root may fall below k_empty/n. This is the control that would
    # have caught the swapped geometry on its own -- and it is checked here at
    # the same a/b the benchmark uses, for both families and l = 1..3.
    violations = []
    for n in (2.0, 3.0):
        violations += bound_violations(A_OVER_B, 1.0, n, l_max=3)

    # ``tm_disc`` is a DISTANCE, compared against a tolerance -- not a flag.
    # It was being used as a bare truthiness gate (``and tm_disc``), and
    # because the two references agree EXACTLY it is 0.0, which is falsy: the
    # best possible score made the runner report FAIL, while the line above it
    # printed "0.00e+00" and read as perfection. A value used as a boolean when
    # it is really a magnitude is the same shape as the rest of this file's
    # traps, one level up in the harness.
    # The limit errors are normalised by the approach distance, so these bounds
    # are on the convergence CONSTANT (~67 TE, ~20 TM measured), not on a step.
    ok = (max_residual < 1e-12 and empty_error < 200 and full_error < 200
          and mono_te and tm_res < 1e-6 and tm_mono and tm_disc < 1e-9
          and tm_empty < 200 and tm_full < 200
          and oracle_worst < ORACLE_TOL
          and not violations)
    print()
    print(f"ORACLE ({oracle_checked} roots vs the lead's independent table): "
          f"worst |err| = {oracle_worst:.2e} (tol {ORACLE_TOL:.0e})")
    if violations:
        print(f"BOUND: {len(violations)} violation(s) -- a reference root below "
              f"k_empty/n is impossible for ANY dielectric fill:")
        for fam, l, i, k, lo, hi in violations[:8]:
            print(f"  {fam} l={l} ord={i}: k={k:.4f} < bound {lo:.4f}")
    else:
        print("BOUND: pass (every reference root in [k_empty/n, k_empty])")
    print("CONTROLS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
