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
#: The shells need tolerances chosen from MEASUREMENT, not from "a few percent
#: feels safe". Every number below is a row of the convergence table, and the
#: table is reproduced by ``--sweep`` so a future change can re-derive them.
#:
#: SHELL CAPACITANCE converges. Against the FIXED meshed reference (the
#: conductor shell faces are at r_in + t_shell and r_out - t_shell, and t_shell
#: is a constant, so C_meshed does NOT move with n_r) the error falls from
#: 2.9e-03 to 4.2e-07 as n_r goes 8 -> 128, always from ABOVE, at an observed
#: order of 2.0 -- which is the O(h^2) an energy-based P1 capacitance is
#: expected to show. An earlier version of this comment claimed the error was
#: FLAT and that no resolution removed it. That was measured with the ANGULAR
#: resolution alone: with the radial layers fixed the error is indeed flat
#: (7.29e-02 at 6/12, 7.29e-02 at 24/48), because the radial error is then the
#: floor. Refining the radial layers is what makes it converge.
#:
#: At the scored mesh (n_r = 48, theta/phi 16/32) the floor is 7.05e-05.
#: A 1% error in eps_a scores 8.2e-03 or 8.3e-03 (either direction); 1% in
#: eps_b scores 1.66e-03 or 1.78e-03. TOL_SHELL = 5e-04 sits 7.1x above the
#: floor and 3.3x below the WEAKEST of those four, so a 1% eps error in either
#: shell, in either direction, fails -- and the mesh's own error does not.
TOL_SLAB = 1e-9
TOL_SHELL = 5e-4

#: The interface potential is a SANITY CHECK, NOT A DETECTOR, and this
#: tolerance is set to say so rather than to pretend otherwise. Measured on
#: the four perturbations (``--perturb``): a 1% eps error moves the capacitance
#: by 16.5x to 3.3x the shell tolerance, but moves this potential by only
#: 0.1x to 0.2x of a 1e-2 bound -- it does not clear it in ANY of the four
#: cases. Giving it a bound tight enough to catch eps would mean a bound below
#: the mesh's own floor (6.5e-04 at n_r=48), i.e. a band around the solver's
#: output, which is precisely what a tolerance must not be.
#:
#: So it is scored at a loose bound that catches gross breakage only -- a
#: wrong voltage, a swapped electrode, a solve that returns a level rather
#: than a field -- and the eps mutation is proved by the CAPACITANCE, which
#: does discriminate. It earns its place because it is the only direct check
#: on the potential: capacitance cannot see a solve that gets the field right
#: and the absolute level wrong.
TOL_PHI_SHELL = 5e-3

#: Resolution for the scored runs. THETA/PHI must be high enough that the
#: angular error is below the radial floor at n_r = 48 (measured: at n_r 48 the
#: error is 6.5e-4 at 6/12 and 7.8e-5 at 24/48, so 16/32 is past the knee);
#: n_r = 48 sets the floor the tolerances above are built on.
SPHERE_N_THETA = 16
SPHERE_N_PHI = 32
SPHERE_N_R = 48


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
        n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI, n_r=SPHERE_N_R,
    )
    out = [
        # Scored against the MESHED reference, for the reason in the module
        # docstring. The idealised value is printed for context only.
        Result("shells capacitance (meshed ref)",
               _capacitance(problem), expected["C_meshed"], TOL_SHELL, "F"),
        Result("shells interface potential",
               _potential_at(problem, expected["r_mid"]),
               expected["phi_rmid"], TOL_PHI_SHELL),
    ]
    return out


def context_lines() -> list[str]:
    """Unscored numbers that explain the scored ones."""
    problem, expected = _fixture_with_trusted_eps(
        concentric_spheres, EPS_A_BENCH, EPS_B_BENCH,
        n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI, n_r=SPHERE_N_R,
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


def _shells_measured(n_theta, n_phi, n_r):
    """(capacitance, interface potential) for one resolution, against references.

    Returns the signed errors against BOTH references -- the meshed one the
    bench scores and the idealised one it prints -- because the difference
    between them is what caught an earlier wrong comment: the meshed reference
    moves with the mesh only through the shells' radii, which are fixed, so a
    converging solver converges on it and does NOT converge on the idealised
    one (the staircase shell faces are a fixed offset).
    """
    problem, expected = _fixture_with_trusted_eps(
        concentric_spheres, EPS_A_BENCH, EPS_B_BENCH,
        n_theta=n_theta, n_phi=n_phi, n_r=n_r,
    )
    cap = _capacitance(problem)
    phi = _potential_at(problem, expected["r_mid"])
    return {
        "cap_meshed": (cap - expected["C_meshed"]) / expected["C_meshed"],
        "cap_ideal": (cap - expected["C"]) / expected["C"],
        "phi_meshed": phi - expected["phi_rmid"],
    }


def convergence_table() -> str:
    """The table TOL_SHELL and TOL_PHI_SHELL are built from. ``--sweep``.

    Three sweeps, because the ORDER only shows up in one of them:

      * radial alone   -- theta/phi held at the scored value, n_r refined. This
        is the one that converges: an earlier comment read the angular sweep,
        saw a flat error, and concluded nothing converged.
      * angular alone  -- n_r held at the scored value. Flat, because at a
        fixed radial floor the angular error is already below it.
      * both           -- the observed order, ~2.0 as expected for energy-based
        P1, so the errors above are discretisation and not bookkeeping.

    The last row is the scored resolution and its numbers are the floor the
    tolerances clear.
    """
    lines = [
        "radial refinement only (theta/phi fixed at the scored resolution)",
        f"  {'n_r':>5}{'cap err(meshed)':>18}{'cap err(ideal)':>17}{'phi err':>12}",
    ]
    for n_r in (8, 16, 24, 32, 48, 64, 96, 128):
        e = _shells_measured(SPHERE_N_THETA, SPHERE_N_PHI, n_r)
        lines.append(f"  {n_r:>5}{e['cap_meshed']:>18.2e}"
                     f"{e['cap_ideal']:>17.2e}{e['phi_meshed']:>12.2e}")

    lines.append("")
    lines.append("angular refinement only (n_r fixed at the scored resolution)")
    lines.append(f"  {'th/phi':>8}{'cap err(meshed)':>18}{'phi err':>12}")
    for nt, nph in ((6, 12), (12, 24), (16, 32), (24, 48)):
        e = _shells_measured(nt, nph, SPHERE_N_R)
        lines.append(f"  {f'{nt}/{nph}':>8}{e['cap_meshed']:>18.2e}"
                     f"{e['phi_meshed']:>12.2e}")

    lines.append("")
    lines.append("both refined together (observed order)")
    lines.append(f"  {'n_r':>5}{'th/phi':>8}{'cap err(meshed)':>18}{'order':>8}")
    prev = None
    for (nt, nph), n_r in (((8, 16), 16), ((12, 24), 32), ((16, 32), 48),
                           ((24, 48), 96)):
        e = _shells_measured(nt, nph, n_r)
        err = abs(e["cap_meshed"])
        order = "" if prev in (None, 0.0) else f"{__import__('math').log2(prev / err):.2f}"
        lines.append(f"  {n_r:>5}{f'{nt}/{nph}':>8}{e['cap_meshed']:>18.2e}"
                     f"{order:>8}")
        prev = err
    return "\n".join(lines)


def perturbation_table() -> str:
    """What a 1% eps error scores, in both shells and both directions. ``--perturb``.

    This is the sensitivity half of TOL_SHELL. It is a TABLE, not a prose claim,
    because the first version of this bench passed a 10% eps error: the
    reference was derived from the same fixture invocation as the solve, so
    mutating eps scaled both sides and the score did not move at all. Every row
    here must clear TOL_SHELL for the tolerance to mean anything.
    """
    base = _shells_measured(SPHERE_N_THETA, SPHERE_N_PHI, SPHERE_N_R)
    # The unperturbed reference, built once. Every perturbed row is scored
    # against THESE numbers, never against its own fixture's reference.
    base_problem, base_expected = _fixture_with_trusted_eps(
        concentric_spheres, EPS_A_BENCH, EPS_B_BENCH,
        n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI, n_r=SPHERE_N_R,
    )
    base_c = base_expected["C_meshed"]
    base_phi = base_expected["phi_rmid"]
    base_r_mid = base_expected["r_mid"]
    lines = [
        f"scored resolution n_r={SPHERE_N_R}, theta/phi={SPHERE_N_THETA}/{SPHERE_N_PHI}",
        f"  TOL_SHELL={TOL_SHELL:.1e}  TOL_PHI_SHELL={TOL_PHI_SHELL:.1e}",
        f"  {'case':>14}{'cap err':>12}{'vs TOL':>10}{'phi err':>12}{'vs TOL':>10}",
    ]

    def row(name, cap, phi):
        return (f"  {name:>14}{cap:>12.2e}{cap / TOL_SHELL:>10.1f}x"
                f"{phi:>12.2e}{abs(phi) / TOL_PHI_SHELL:>10.1f}x")

    lines.append(row("baseline", base["cap_meshed"], base["phi_meshed"]))
    for shell in ("a", "b"):
        for sign in (1.01, 0.99):
            # THE SOLVE IS PERTURBED; THE REFERENCE IS NOT. The fixture is
            # rebuilt with one eps moved by 1%, but only its PROBLEM is used --
            # the reference dict is discarded and the score is judged against
            # the unperturbed `base` fixture's closed form. Scoring against the
            # perturbed fixture's own reference would rescale both sides and
            # report ~0, which is exactly how an earlier version of this bench
            # passed a 10% eps error.
            eps_a, eps_b = EPS_A_BENCH, EPS_B_BENCH
            if shell == "a":
                eps_a *= sign
            else:
                eps_b *= sign
            problem, _ = _fixture_with_trusted_eps(
                concentric_spheres, eps_a, eps_b,
                n_theta=SPHERE_N_THETA, n_phi=SPHERE_N_PHI, n_r=SPHERE_N_R,
            )
            cap = (_capacitance(problem) - base_c) / base_c
            phi = _potential_at(problem, base_r_mid) - base_phi
            lines.append(row(f"eps_{shell} x{sign}", cap, phi))
    return "\n".join(lines)


def main(argv=None) -> int:
    global _METHOD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true",
                    help="print the numbers, never exit non-zero")
    ap.add_argument("--method", default="direct", choices=("direct", "amg"),
                    help="linear solver behind the Dirichlet reduction")
    ap.add_argument("--sweep", action="store_true",
                    help="print the convergence table the tolerances are built "
                         "from, then exit; re-derive them before changing one")
    ap.add_argument("--perturb", action="store_true",
                    help="print what a 1%% eps error in each shell scores, then "
                         "exit; this is the sensitivity half of the tolerances")
    args = ap.parse_args(argv)
    _METHOD = args.method

    if args.sweep:
        print(convergence_table())
        return 0
    if args.perturb:
        print(perturbation_table())
        return 0

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
