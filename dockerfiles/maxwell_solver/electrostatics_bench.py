"""Score the electrostatics solver against meshes with known answers.

Runs the two fixtures whose answers are CLOSED FORM -- the two-layer slab and
the concentric shells -- and prints the measured values beside the reference.
Exit code 1 if any scored quantity is outside its tolerance, so this can gate a
release; ``--report-only`` prints without judging.

Three fixtures exist (see ``mse_bench_fixtures``); the third, a dielectric
sphere between plates, has NO closed form and is NOT scored here. Its numbers
are printed under --report-only and labelled UNVALIDATED, because a stable
value is evidence of convergence, not of correctness.

Why the references are computed twice for the shells
----------------------------------------------------
The capacitor fixtures model their conductors as thin SHELLS a few nodes thick
(a thin conductor is what keeps the physical and numerical boundary in the
same place). Those shells are held at a fixed potential, so the dielectric
starts at the shell's inner face, not at ``r_in``. An FEM answer therefore
converges to the closed form evaluated at the MESHED radii, and differs from
the idealised one by a few percent that no refinement removes. Both are
reported; the meshed reference is the one that carries a tolerance, because
scoring against the idealised number would report a fixture property as a
solver error. Measured on this bench: the shells sit 0.04% from the meshed
reference and 7.1% from the idealised one, and the gap is pure geometry.

Usage
-----
    python electrostatics_bench.py                 # score, exit 1 on failure
    python electrostatics_bench.py --report-only    # print, never fail
    python electrostatics_bench.py --method amg     # exercise the AMG path
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import mse_electrostatics as E
from mse_bench_fixtures import (
    concentric_spheres,
    sphere_between_plates,
    two_layer_slab,
)

#: Relative tolerances. The slab is a structured box with node planes ON every
#: interface, so its answer is exact to round-off and the bound is tight.
#:
#: The shells need a tolerance chosen from MEASUREMENT, not from "a few percent
#: feels safe". Two numbers decide it, both measured on this fixture:
#:
#:   mesh floor    the solver sits 4.36e-04 from the meshed reference and the
#:                 error is FLAT under refinement (4.0e-4 at 8x16, 4.6e-4 at
#:                 24x48). It is the theta staircase discretising a 1/r
#:                 profile, not a convergence gap, so no resolution removes it.
#:   sensitivity   1% too much eps in shell_a moves the score to 8.4e-03; 1%
#:                 in shell_b to 2.5e-03.
#:
#: A 2e-2 bound -- the first version here -- cleared the floor by 46x and was
#: too slack to notice a 10% eps error (it scored 1.7e-02 and PASSED). 2e-3
#: sits 4.6x above the floor and below BOTH 1% sensitivities, so the smallest
#: perturbation worth calling a bug fails, and the mesh's own error does not.
#: The slack between floor and bound (4.6x) is the margin a refactor has to
#: work with; if a future change lands nearer the floor, tighten the mesh, do
#: not widen the bound.
TOL_SLAB = 1e-9
TOL_SHELL = 2e-3

#: Resolution for the scored runs: the coarsest that converges. Higher costs
#: time without changing a verdict.
SPHERE_N_THETA = 12
SPHERE_N_PHI = 24


class Result:
    """One scored quantity: measured, reference, tolerance and a name."""

    def __init__(self, name, measured, reference, tol, unit=""):
        self.name = name
        self.measured = float(measured)
        self.reference = float(reference)
        self.tol = tol
        self.unit = unit

    @property
    def rel_error(self) -> float:
        if self.reference == 0.0:
            return abs(self.measured)
        return abs(self.measured - self.reference) / abs(self.reference)

    @property
    def ok(self) -> bool:
        return self.rel_error <= self.tol

    def line(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        u = f" {self.unit}" if self.unit else ""
        return (f"  [{verdict}] {self.name:<34} measured {self.measured:>12.6e}{u}  "
                f"reference {self.reference:>12.6e}{u}  "
                f"rel {self.rel_error:>9.2e}  tol {self.tol:.0e}")


def _capacitance(problem) -> float:
    C, _ = E.capacitance_matrix(
        problem["points_m"], problem["tets"], problem["parts"],
        problem["region_of_tet"], problem["eps_static"],
        problem["electrodes"], method=_METHOD,
    )
    # The driving electrode is the diagonal entry for the first electrode; the
    # matrix is Maxwell-signed, so a positive capacitance is its absolute value
    # on that entry.
    return float(abs(C[0, 0]))


def _capacitance_matrix_only(problem):
    """The full Maxwell capacitance matrix, for inspecting its structure."""
    C, _ = E.capacitance_matrix(
        problem["points_m"], problem["tets"], problem["parts"],
        problem["region_of_tet"], problem["eps_static"],
        problem["electrodes"], method=_METHOD,
    )
    return C


def _solution(problem):
    return E.solve(
        problem["points_m"], problem["tets"], problem["parts"],
        problem["region_of_tet"], problem["eps_static"],
        problem["electrodes"], method=_METHOD,
    )


def _potential_at(problem, radius: float) -> float:
    """phi at a node lying on the sphere of the given radius, or the nearest one.

    The fixtures put node planes exactly ON their interfaces, so ``radius`` is a
    node radius and this is an exact lookup rather than an interpolation. The
    nearest-node fallback is there for float rounding only.
    """
    sol = E.solve(
        problem["points_m"], problem["tets"], problem["parts"],
        problem["region_of_tet"], problem["eps_static"],
        problem["electrodes"], method=_METHOD,
    )
    r = np.linalg.norm(problem["points_m"], axis=1)
    on = np.abs(r - radius) <= 1e-12 + 1e-6 * radius
    if on.any():
        return float(sol["phi"][on].mean())
    return float(sol["phi"][int(np.argmin(np.abs(r - radius)))])


#: The permittivities the reference is built from. These are the TRUSTED
#: values: they are passed INTO the fixture explicitly and the fixture is then
#: checked to have received exactly them.
#:
#: Why this is not just restating the fixture's defaults. An earlier version
#: took the reference from whatever eps the fixture happened to use, so
#: changing one dielectric scaled the solver's measured value and the reference
#: by the SAME factor -- the relative error did not move (7.10e-04 before and
#: after) and a wrong eps_s scored a clean PASS. A reference derived from the
#: input it is meant to police cannot detect a bad input. This is the whole
#: reason the bench exists, so the independence is asserted below rather than
#: assumed.
EPS_A_BENCH = 2.0
EPS_B_BENCH = 5.0


def _fixture_with_trusted_eps(builder, eps_a, eps_b, **kw):
    """Build a fixture on the trusted eps, refusing if it used anything else.

    The refusal is the mutation-detector: a changed EPS_A_BENCH changes the
    reference AND the solve, which proves nothing; a changed value reaching the
    fixture some other way (a default, an override) means the two sides of the
    comparison are no longer independent, and the bench must not score it.
    """
    problem, expected = builder(eps_a=eps_a, eps_b=eps_b, **kw)
    table = problem["eps_static"]
    # The two fixtures name their dielectric parts differently (slab: a/b,
    # shells: shell_a/shell_b), so find the parts that actually carry eps_a and
    # eps_b by value rather than hardcoding a key.
    got = sorted(v for k, v in table.items() if k not in ("inner", "outer",
                                                          "bottom", "top"))
    if got != sorted([eps_a, eps_b]):
        raise AssertionError(
            f"fixture did not receive the trusted permittivities: asked for "
            f"eps_a={eps_a}, eps_b={eps_b}; its eps_static holds {got}. The "
            f"reference would share its input with the value it scores, so no "
            f"wrong eps_s could ever fail."
        )
    return problem, expected


def run_slab() -> list[Result]:
    problem, expected = _fixture_with_trusted_eps(
        two_layer_slab, EPS_A_BENCH, EPS_B_BENCH)
    out = [
        Result("slab capacitance", _capacitance(problem), expected["C"], TOL_SLAB, "F"),
    ]
    return out


def run_shells() -> list[Result]:
    problem, expected = _fixture_with_trusted_eps(
        concentric_spheres, EPS_A_BENCH, EPS_B_BENCH,
        n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI,
    )
    out = [
        # Scored against the MESHED reference, for the reason in the module
        # docstring. The idealised value is printed for context only.
        Result("shells capacitance (meshed ref)",
               _capacitance(problem), expected["C_meshed"], TOL_SHELL, "F"),
        Result("shells interface potential",
               _potential_at(problem, expected["r_mid"]),
               expected["phi_rmid"], 5e-2),
    ]
    return out


def context_lines() -> list[str]:
    """Unscored numbers that explain the scored ones."""
    problem, expected = _fixture_with_trusted_eps(
        concentric_spheres, EPS_A_BENCH, EPS_B_BENCH,
        n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI,
    )
    measured = _capacitance(problem)
    ideal = expected["C"]
    rel = (measured - ideal) / ideal

    # The sphere has NO closed form (see sphere_between_plates). It is built and
    # solved anyway, because three fixture defects that the scored cases could
    # not show up here: a duplicated z-plane (a bare LinAlgError from
    # tet_geometry, nowhere near the cause), a non-reciprocal C, and a solution
    # that left the driven range. A fixture that only exists to be looked at
    # still has to be checked for being a mesh at all.
    sphere = sphere_between_plates()
    C_s = _capacitance_matrix_only(sphere)
    phi = _solution(sphere)["phi"]
    asym = float(np.abs(C_s - C_s.T).max() / max(float(np.abs(C_s).max()), 1e-300))
    return [
        f"  shells vs IDEALISED conductor radii: rel {rel:+.3e} "
        f"(reference {ideal:.6e} F) -- NOT scored: the fixture's conductors are",
        f"  shells ({expected['r_inner_shell']:.4e} and "
        f"{expected['r_outer_shell']:.4e} m), so this gap is geometry.",
        f"  sphere in air (no closed form): C = {abs(C_s[0, 0]):.6e} F, "
        f"phi in [{phi.min():.4f}, {phi.max():.4f}] V, "
        f"max|C - C^T|/max|C| = {asym:.1e}",
    ]


_METHOD = "direct"


def main(argv=None) -> int:
    global _METHOD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true",
                    help="print the numbers, never exit non-zero")
    ap.add_argument("--method", default="direct", choices=("direct", "amg"),
                    help="linear solver behind the Dirichlet reduction")
    args = ap.parse_args(argv)
    _METHOD = args.method

    results = run_slab() + run_shells()
    print(f"electrostatics bench -- linear method: {_METHOD}")
    for group in ("slab", "shells"):
        print(f"\n{group}:")
        for r in results:
            if group == "slab" and not r.name.startswith("slab"):
                continue
            if group == "shells" and not r.name.startswith("shells"):
                continue
            print(r.line())
    print("\ncontext (not scored):")
    for line in context_lines():
        print(line)

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"FAILED: {len(failed)} of {len(results)} scored quantities out of "
              f"tolerance: " + ", ".join(r.name for r in failed))
        return 0 if args.report_only else 1
    print(f"all {len(results)} scored quantities within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
