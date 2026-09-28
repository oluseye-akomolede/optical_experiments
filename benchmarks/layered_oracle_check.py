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


if __name__ == "__main__":
    main()
