"""Self-check for the 1-D layered transfer-matrix oracle used by feature 2.

The plan (docs/photonic-crystal-plan.md, Control B) claims the closed-form
transfer matrix agrees with an independent ODE shooting solve to 1e-10 at
eps2 = 4, and that the 1/eps^2 admittance form fails that check. This script
is that claim, runnable, so the numbers in the plan can be reproduced rather
than trusted.

Geometry: period a = 1, layer A (eps = 1) for x in [0, f*a), layer B (eps2)
for x in [f*a, a). The stripe normal is x; ky is the k-component PARALLEL to
the layers and is conserved. s and p polarisations differ only when ky != 0
AND eps2 != 1 -- both conditions are required, and the script checks that too.

  s (PhC TM, E_z): (u, u') continuous,  eta_s = kx
  p (PhC TE, H_z): (v, (1/eps)v') continuous,  eta_p = kx/eps

Dev-only. scipy is a benchmark dependency and never enters an image.
"""
import numpy as np
from scipy.integrate import solve_ivp

EPS1, F, A = 1.0, 0.5, 1.0


def _m_s(k, d):
    if abs(k) < 1e-14:
        return np.array([[1.0, d], [0.0, 1.0]])
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, s / k], [-k * s, c]])


def _m_p(k, eps, d):
    """Monodromy of the CONTINUOUS state (v, w), w = (1/eps) dv/dx."""
    if abs(k) < 1e-14:
        return np.array([[1.0, eps * d], [0.0, 1.0]])
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, eps * s / k], [-k * s / eps, c]])


def closed_form_trace(lam, ky, eps2, pol):
    k1 = np.sqrt(lam * EPS1 - ky ** 2 + 0j)
    k2 = np.sqrt(lam * eps2 - ky ** 2 + 0j)
    if pol == "s":
        return np.trace(_m_s(k2, (1 - F) * A) @ _m_s(k1, F * A))
    return np.trace(_m_p(k2, eps2, (1 - F) * A) @ _m_p(k1, EPS1, F * A))


def _m_p_wrong(k, eps, d):
    """The 1/eps^2 admittance -- the error this check exists to catch."""
    if abs(k) < 1e-14:
        return np.array([[1.0, eps * d], [0.0, 1.0]])
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, eps * eps * s / k], [-k * s / eps, c]])


def closed_form_trace_wrong(lam, ky, eps2):
    k1 = np.sqrt(lam * EPS1 - ky ** 2 + 0j)
    k2 = np.sqrt(lam * eps2 - ky ** 2 + 0j)
    return np.trace(_m_p_wrong(k2, eps2, (1 - F) * A) @ _m_p_wrong(k1, EPS1, F * A))


def shooting_trace(lam, ky, eps2, pol, rtol=1e-13):
    """Independent solve: integrate the state across the cell, no closed form."""
    if pol == "s":
        def rhs(x, y):
            e = EPS1 if x < F * A else eps2
            return [y[1], (ky ** 2 - lam * e) * y[0]]
    else:
        def rhs(x, y):
            e = EPS1 if x < F * A else eps2
            return [e * y[1], (ky ** 2 / e - lam) * y[0]]
    cols = []
    for ic in ([1, 0], [0, 1]):
        sol = solve_ivp(rhs, [0, A], ic, rtol=rtol, atol=rtol * 1e-2)
        cols.append(sol.y[:, -1])
    return np.trace(np.column_stack(cols))


def main():
    kys = np.linspace(0.05, 1.2, 7)
    lams = (1.5, 4.0, 9.0, 15.0)

    worst13 = max(
        abs(closed_form_trace(l, ky, 4.0, p) - shooting_trace(l, ky, 4.0, p, 1e-13))
        for ky in kys for l in lams for p in ("s", "p")
    )
    print(f"oracle self-check (eps2=4, rtol=1e-13, {len(kys)*len(lams)*2} points): "
          f"{worst13:.3e}  required 1e-10 -> {'PASS' if worst13 < 1e-10 else 'FAIL'}")

    for rtol in (1e-10, 1e-12):
        w = max(
            abs(closed_form_trace(l, ky, 4.0, p) - shooting_trace(l, ky, 4.0, p, rtol))
            for ky in kys for l in lams for p in ("s", "p")
        )
        print(f"  (rtol={rtol:.0e} -> {w:.3e}; rtol is the binding constraint)")

    wrong = max(
        abs(closed_form_trace_wrong(l, ky, 4.0) - shooting_trace(l, ky, 4.0, "p", 1e-13))
        for ky in (0.3, 0.9) for l in (2.0, 6.5)
    )
    print(f"control: the 1/eps^2 admittance vs the same oracle: {wrong:.3e} -> "
          f"{'correctly REJECTED' if wrong > 1e-10 else 'WRONGLY ACCEPTED'}")

    for ky in (0.0, 0.3, 0.9):
        ds = abs(closed_form_trace(2.0, ky, 4.0, "s") - closed_form_trace(2.0, ky, 4.0, "p"))
        print(f"  eps2=4, ky={ky}: |s-p| = {ds:.3e}"
              f"{'  <- parallel component required' if ds < 1e-12 else ''}")
    for ky in (0.3,):
        ds = abs(closed_form_trace(2.0, ky, 1.0, "s") - closed_form_trace(2.0, ky, 1.0, "p"))
        print(f"  eps2=1, ky={ky}: |s-p| = {ds:.3e}  <- uniform: cannot fail")

    # --- the BAND finder, which is what Control B actually compares against ---
    # pc.layered_band_at_k is a SECOND consumer of the same closed form, and it
    # is the one the mesh is compared to. Check the whole finder (root scan +
    # bisection) against a root of the INDEPENDENT shooting trace, not against
    # the closed form it was built from -- otherwise the check only restates it.
    #
    # UNITS. pc.layered_band_at_k takes (kx, ky) in 2 pi/a units (the mesh's
    # convention) and returns the band w a / (2 pi c). The shooting trace here is
    # in RAW wavenumbers, so ky_raw = 2 pi ky and the returned band is
    # sqrt(lam)/2 pi. Both sides therefore carry the same 2 pi and the check
    # compares like with like.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "dockerfiles", "maxwell_solver"))
    import pc  # noqa: E402

    from scipy.optimize import brentq

    def band_shooting(ky, eps2, pol):
        """The band from the INDEPENDENT shooting trace, in the same units.

        The root is at lam ~ (2 pi ky)^2 / eps2, which grows with ky, so the
        bracket is found by scanning for the sign change rather than pinned --
        a hard-coded [1e-9, 0.5] silently misses the higher-ky cases.
        """
        ky_raw = 2.0 * np.pi * ky
        g = lambda l: float(np.real(shooting_trace(l, ky_raw, eps2, pol, 1e-13))) - 2.0
        lam_max = 2.0 * (ky_raw ** 2) / min(eps2, 1.0) + 5.0
        # A COARSE scan finds the bracket; solve_ivp at rtol=1e-13 is the cost, and
        # a fine scan spends it thousands of times. 120 points bracket the low root
        # (the band nearest the light line) and brentq then refines with ~40 extra
        # solves, not thousands.
        grid = np.linspace(1e-12, lam_max, 120)
        prev_x, prev_f = None, None
        for x in grid:
            fx = g(x)
            if prev_f is not None and (prev_f <= 0.0 < fx or prev_f >= 0.0 > fx):
                lam = brentq(g, prev_x, x, xtol=1e-14)
                return float(np.sqrt(lam) / (2.0 * np.pi))
            prev_x, prev_f = x, fx
        raise RuntimeError(f"no shooting root below lam={lam_max} at ky={ky}")

    worst = 0.0
    for eps2 in (4.0, 12.0):
        for ky in (0.3, 0.6):
            for pol in ("s", "p"):
                mine = pc.layered_band_at_k(0.0, ky, eps2, 0.5, pol)
                theirs = band_shooting(ky, eps2, pol)
                d = abs(mine - theirs)
                worst = max(worst, d)
                print(f"  band kx=0 ky={ky} eps2={eps2} {pol}: "
                      f"mine={mine:.12f} shooting={theirs:.12f} |d|={d:.2e}")
    print(f"  band-finder self-check: worst |d| = {worst:.3e}  required 1e-9 -> "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    # the could-not-ask guard: the finder must reproduce the EXACT uniform band,
    # which an edge-scan oracle cannot (it lands on a band edge, 5.7% away).
    unif = max(
        abs(pc.layered_band_at_k(0.0, ky, 1.0, 0.5, p) - ky)
        for ky in (0.2, 0.3, 0.9) for p in ("s", "p")
    )
    print(f"  uniform band (eps2=1) vs exact |k|=ky: worst = {unif:.3e}  required "
          f"1e-10 -> {'PASS' if unif < 1e-10 else 'FAIL'}")

    # The unit pin at eps != 1: a uniform cell at eps gives exactly |k|/sqrt(eps).
    # This is the relation that a dropped 2 pi on either side breaks, and it is
    # the one the eps = 1 check above cannot see (w = |k| there).
    unit = max(
        abs(pc.layered_band_at_k(kx, ky, eps, 0.5, p, eps1=eps)
            - np.hypot(kx, ky) / np.sqrt(eps))
        for eps in (2.25, 4.0) for kx, ky in ((0.13, 0.37), (0.2, 0.0))
        for p in ("s", "p")
    )
    print(f"  unit pin, uniform eps!=1 vs |k|/sqrt(eps): worst = {unit:.3e}  "
          f"required 1e-12 -> {'PASS' if unit < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
