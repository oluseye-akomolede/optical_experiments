# Purcell reference: a dipole above air / SiO2 / gold

`dockerfiles/maxwell_solver/purcell_reference.py` gives the exact normalised
decay rate `F = P / P0` of an electric dipole above a planar stack, from the
Sommerfeld-integral form of the reflected field, to validate the FDTD
solver's Purcell measurement. Tests: `tests/test_purcell_reference.py`
(21 tests, 2.5 s, `OMP_NUM_THREADS=2`). OpenSpec change:
`openspec/changes/purcell-layered-reference/`.

## Sources

- L. Novotny and B. Hecht, *Principles of Nano-Optics* (Cambridge, 2006),
  ch. 10 "Dipole emission near planar interfaces" (pp. 335-360): the
  decay-rate integrals (`F_perp`, `F_par` below) and the far-field
  radiation patterns; ch. 8 "Light emission and optical interactions in
  nanoscale environments", sec. "Classical lifetimes and decay rates", for
  `P / P0 = 1 + (6 pi eps0 / |p|^2 k^3) Im(p* . E_s)` and the dipole
  field. Equation numbers differ between editions and were not checked
  against the book's text from this sandbox, so sections are cited.
- R. R. Chance, A. Prock and R. Silbey, "Molecular fluorescence and energy
  transfer near interfaces", *Adv. Chem. Phys.* **37**, 1 (1978), sec. II:
  the same integrals, and the mirror limits.
- W. Lukosz and R. E. Kunz, *J. Opt. Soc. Am.* **67**, 1607 and 1615 (1977):
  emission into a lossless dielectric, including the forbidden-light band.
- Gold: `fdtd.permittivity("Au", f)`, the Rakić 1998 Lorentz-Drude fit
  already in the repo (`fdtd.RAKIC_1998`); `-27.61 + 2.45i` at 900 nm.

No paper was fetched: the formulas are the textbook ones, restated in the
module docstring, and every closed form used is derived below. A web search
confirmed the chapter titles and page range only.

## Formulation

`s = k_rho / k0`, `s_z = sqrt(1 - s^2)` with `Im s_z >= 0`, `r_p`, `r_s`
the stack's Fresnel coefficients from the air side
(`r_p = (eps2 kz1 - eps1 kz2) / (eps2 kz1 + eps1 kz2)`, so a perfect
conductor has `r_p = +1`, `r_s = -1`), `e = exp(2 i k0 d s_z)`:

```
F_perp = 1 + (3/2) Re  int_0^inf (s^3 / s_z) r_p e ds
F_par  = 1 + (3/4) Re  int_0^inf (s / s_z) [r_s - s_z^2 r_p] e ds
```

Radiated into the air (far field, `s < 1`):

```
P_up,perp / P0 = (3/4) int_0^1 (s^3 / s_z) |1 + r_p e|^2 ds
P_up,par  / P0 = (3/8) int_0^1 (s / s_z) [ |1 + r_s e|^2 + s_z^2 |1 - r_p e|^2 ] ds
```

Transmitted into a lossless substrate of index `n` (flux of each plane wave
of the dipole's angular spectrum through the stack; `q3 = kz3 / k0`, `t_p`
the H-field and `t_s` the E-field stack transmission coefficients,
`E = exp(-2 k0 d Im s_z)`; the `1 / |s_z|^2` is the Weyl `1 / kz` weight of
the dipole's own spectrum, and `|t_p|^2` vanishes like `|s_z|^2` at `s = 1`):

```
P_down,perp / P0   = (3/4) int_0^n s^3 Re(q3) |t_p|^2 / (eps3 |s_z|^2) E ds
P_down,par,TE / P0 = (3/8) int_0^n s   Re(q3) |t_s|^2 / |s_z|^2         E ds
P_down,par,TM / P0 = (3/8) int_0^n s   Re(q3) |t_p|^2 / eps3           E ds
```

The three transmitted integrals were derived here (angular spectrum of a
dipole, interface flux per plane wave, continued through the `1 < s < n`
band). They are not copied from a reference, which is why the lossless
conservation test matters: the Sommerfeld total along a complex path and
these real-axis fluxes agree to 1e-15 at every height tried, with and
without a film, for both orientations. The chance of two wrong formulas
agreeing that well is nil.

### The integration path

The surface-plasmon pole of a lossy metal sits just above the real `s`
axis (`Im s > 0` for `Im eps > 0`); `1 / s_z` has a branch point at
`s = 1` on it. The module integrates along

```
s(u) = u - i h sin(pi u / s_b),   0 <= u <= s_b = 1 + max(1, n_film, Re n_sub)
```

then along the real axis from `s_b` to where `exp(-2 k0 d sqrt(s^2 - 1))`
is `exp(-50)`. Passing below every singularity is the vanishing-loss limit,
so a grounded dielectric slab (a lossless film on a perfect conductor,
whose TM0 mode has a real pole for `1 < s < n_film`) is handled by the same
path: `F` is the same for `h = 0.25, 0.5, 1.0` to 1e-15, with the guided
power reported as `nonradiative`. The detour also damps the
`exp(2 i k0 d s_z)` oscillation, so large heights cost nothing extra. The
principal square root is continuous along the whole path and matches the
`Im >= 0` determination on the axis at both ends (shown in the module).

### The image dipole over a perfect conductor (check 1)

The reflected field at the dipole is the field of the image at distance
`R = 2d` (same sign for a perpendicular dipole, reversed for a parallel
one). With the dipole field

```
E = (1 / 4 pi eps0) { k^2 (n x p) x n e^{ikR}/R + [3 n (n.p) - p] (1/R^3 - ik/R^2) e^{ikR} }
```

and `F = 1 + (6 pi eps0 / |p|^2 k^3) Im(p* . E_image)`, `x = 2kd`:

```
F_perp = 1 + 3     [ sin x / x^3 - cos x / x^2 ]            (-> 2 at the mirror)
F_par  = 1 + (3/2) [ sin x / x^3 - cos x / x^2 - sin x / x ] (-> 0 at the mirror)
```

### The glass interface at `d -> 0` (check 4)

For a dipole in air at a lossless half-space of index `n`, `F(d -> 0)` is
elementary. For `s > n` the integrand is imaginary and drops out; for
`1 < s < n` only `Im r` survives the real part (`s_z = i a`):
`Im r_p = 2 n^2 a b / (n^4 a^2 + b^2)`, `Im r_s = 2 a b / (a^2 + b^2)`,
`a = sqrt(s^2 - 1)`, `b = sqrt(n^2 - s^2)`. Substitutions: on `0 < s < 1`,
`v = s_z` and the Euler substitution `sqrt(v^2 + n^2 - 1) = v + z` make the
integrand rational in `z`; on `1 < s < n`, `w = b` does the same. For
`n = 3/2` (sympy, exact; the script is below, and the test's
`glass_interface_closed_form` evaluates the result):

```
L = 2 ln[ (1 + sqrt13)(sqrt13 - sqrt5) / ((sqrt13 + sqrt5)(sqrt13 - 1)) ]
    + ln[ (9 sqrt13 + 13 sqrt5) / (9 sqrt13 - 13 sqrt5) ]
F_perp(0) = 2511/1690 + (2187 sqrt13 / 10985) L = 2.30357446575872
F_par(0)  = 2653/1690 - ( 486 sqrt13 / 10985) L = 1.38809456316473
```

A 30-digit mpmath quadrature of the real-axis integrals gives the same
digits. The module at `d = lambda / 10^8` gives 2.3035743 and 1.3880945
(the difference, 1e-7, is the term linear in `d`).

```python
import sympy as sp
n = sp.Rational(3, 2); c2 = n**2 - 1
z, w = sp.symbols('z w', positive=True)
v = (c2 - z**2) / (2*z); R = (c2 + z**2) / (2*z); dv = sp.diff(v, z)
rp = (n**2*v - R) / (n**2*v + R); rs = (v - R) / (v + R)
I1 = sp.integrate(sp.simplify((1 - v**2) * rp * dv), (z, sp.sqrt(c2), n - 1))
I2 = sp.integrate(2*n**2*w**2*(n**2 - w**2) / (n**4*(n**2 - 1) - (n**4 - 1)*w**2), (w, 0, sp.sqrt(c2)))
J1 = sp.integrate(sp.simplify((rs - v**2*rp) * dv), (z, sp.sqrt(c2), n - 1))
u = n**2 - w**2
J2 = sp.integrate(sp.simplify((w/(n**2 - 1) + n**2*(u - 1)*w/(n**4*(u - 1) + n**2 - u)) * (-2*w)), (w, sp.sqrt(c2), 0))
print(sp.N(1 + sp.Rational(3, 2)*(I1 + I2), 15), sp.N(1 + sp.Rational(3, 4)*(J1 + J2), 15))
```

### Quenching (check 5)

For `k d << 1` the `s >> 1` tail dominates, `r_p -> (eps - 1)/(eps + 1)`,
`s^3 / s_z -> -i s^2`, and `int s^2 e^{-2kds} ds = 2 / (2kd)^3`:

```
F_perp,nonrad ~ (3/8)  Im[(eps - 1)/(eps + 1)] / (kd)^3
F_par,nonrad  ~ (3/16) Im[(eps - 1)/(eps + 1)] / (kd)^3
```

Bare gold at 900 nm (`Im[(eps-1)/(eps+1)] = 0.006876`):

| d (nm) | nonrad perp | quasi-static | nonrad par | quasi-static |
| --- | --- | --- | --- | --- |
| 2 | 950.28 | 947.23 | 474.12 | 473.61 |
| 3 | 283.37 | 280.66 | 140.72 | 140.33 |
| 4 | 120.94 | 118.40 | 59.53 | 59.20 |
| 6 | 37.43 | 35.08 | 17.80 | 17.54 |
| 8 | 17.04 | 14.80 | 7.62 | 7.40 |

Log-log slope over 2-5 nm: -2.962 (perp), -2.991 (par); the next term in
the expansion is `O(1/d)`, which is what pulls the slope off -3.

## The CLI table (the job's stack)

`OMP_NUM_THREADS=2 python3 dockerfiles/maxwell_solver/purcell_reference.py`
(6 nm above 12 nm of n = 1.45 on Au):

```
      nm       F_perp   rad_perp        F_par    rad_par
   850.0       6.2234     1.0168       1.5550     0.1417
   860.0       6.1540     1.0279       1.5291     0.1382
   870.0       6.0905     1.0387       1.5059     0.1348
   880.0       6.0327     1.0493       1.4853     0.1316
   890.0       5.9800     1.0595       1.4669     0.1285
   900.0       5.9321     1.0695       1.4507     0.1255
   910.0       5.8886     1.0793       1.4365     0.1226
   920.0       5.8492     1.0888       1.4240     0.1198
   930.0       5.8137     1.0980       1.4134     0.1172
   940.0       5.7817     1.1071       1.4043     0.1146
   950.0       5.7532     1.1159       1.3967     0.1121
```

`rad_*` is the share of `P0` that reaches the air far field; the rest
(`F - rad`) is absorbed in the gold. For the solver this is the number to
compare a Purcell measurement against, and the radiated share is the number
a flux box in the air above the stack should see.

## Mutation sweep

Each mutation was applied to the module alone (string replacement), the 21
tests run, and the module restored (`21 passed` after the sweep). "Caught"
means the test(s) its docstring names went red; the full list of reds is in
the third column.

| mutation | result | tests that went red |
| --- | --- | --- |
| M1 round-trip phase `exp(i k d s_z)` in the total | caught, 12 failed | pec_matches_the_image_dipole, pec_radiates_everything, large_height (both), unit_index_film_is_no_film, lossless_dielectric_conserves_power, gold_quenching |
| M2 3/2 and 3/4 prefactors swapped | caught, 12 failed | pec_matches_the_image_dipole, pec_radiates_everything, large_height (both), lossless_dielectric (both), gold_quenching |
| M3 sign of the `s_z^2 r_p` term (parallel) | caught, 6 failed | pec_matches_the_image_dipole[par], pec_radiates_everything[par], large_height_parallel, lossless_dielectric (par, both), gold_quenching[par] |
| M4 far field: `s_z^2` weight dropped on the p part | caught, 3 failed | pec_radiates_everything[par], lossless_dielectric (par, both) |
| M5 film round trip written as a single pass | caught, 4 failed | unit_index_film_is_no_film, lossless_dielectric_conserves_power |
| M6 Airy denominator `1 - r12 r23 E` | caught, 7 failed | pec_matches_the_image_dipole, unit_index_film, zero_thickness_film, lossless_dielectric_conserves_power |
| M7 transmission without the single-pass phase | caught, 4 failed | unit_index_film_is_no_film, lossless_dielectric_conserves_power |
| M8 substrate `r_p` with its eps factors swapped | caught, 11 failed | gold_quenching, lossless_dielectric (both), unit_index_film, zero_thickness_film, large_height_parallel, api_shape |
| M9 real-axis tail cut short (`TAIL_DECADES = 2`) | caught, 4 failed | gold_quenching, unit_index_film_is_no_film |
| M10 conjugated metal permittivity | caught, 3 failed | gold_quenching, api_shape |
| M11 path integral without its Jacobian | caught, 16 failed | quadrature_and_path_are_converged and 14 others |
| M12 detour returns to the axis at the branch point (`s_b = 1`) | caught, 16 failed | quadrature_and_path_are_converged and 14 others |
| M13 forbidden-light band dropped from the transmitted power | caught, 2 failed | lossless_dielectric_conserves_power (both) |
| M14 Weyl weight `1/s_z^2` instead of `1/|s_z|^2` | caught, 4 failed | lossless_dielectric (both tests, both orientations) |
| M15 TE/TM transmission weights swapped (parallel) | caught, 1 failed | lossless_dielectric_conserves_power[par] |
| M16 image-dipole closed form with the wrong sign | caught, 2 failed | image_dipole_limits_at_the_mirror, pec_matches_the_image_dipole[perp] |
| M17 transmitted perpendicular weight `s` instead of `s^3` | caught, 2 failed | lossless_dielectric (perp, both) |
| M18 detour into the upper half-plane (crosses the plasmon pole) | caught, 17 failed | gold_quenching[par], grounded_slab, quadrature_and_path and 13 others |
| M19 far field `|1 - r_p e|^2` for the perpendicular dipole | caught, 3 failed | pec_radiates_everything[perp], lossless_dielectric_conserves_power[perp], grounded_slab |
| M20 path returns to the axis before the substrate branch point (`s_b = 1.5`) | caught, 4 failed | lossless_dielectric (both tests, both orientations) |

## Numbers and the commands that reproduce them

```
OMP_NUM_THREADS=2 python3 -m pytest tests/test_purcell_reference.py -q      # 21 passed, ~2.5 s
OMP_NUM_THREADS=2 python3 dockerfiles/maxwell_solver/purcell_reference.py   # the table above
```

```python
# from the repo root
import sys, math, numpy as np; sys.path.insert(0, "dockerfiles/maxwell_solver")
import purcell_reference as pr
lam = 900e-9; k0 = 2 * math.pi / lam
max(abs(pr.purcell_layered(lam, 0.5 * x / k0, 0, 1.0, "PEC", o).F - pr.image_dipole_pec(0.5 * x, o))
    for o in pr.ORIENTATIONS for x in np.geomspace(0.1, 20, 40))             # 2.4e-14
pr.purcell_layered(lam, 1e-8 * lam, 0, 1.0, 2.25, "perp").F                   # 2.3035743
pr.purcell_layered(lam, 1e-8 * lam, 0, 1.0, 2.25, "par").F                    # 1.3880945
pr.purcell_layered(lam, 0.1 * lam, 0, 1.0, 2.25, "perp").nonradiative         # -4e-16
h = np.array([2e-9, 3e-9, 4e-9, 5e-9])
np.polyfit(np.log(h), np.log([pr.purcell_layered(lam, d, 0, 1.0, "Au", "perp").nonradiative for d in h]), 1)[0]  # -2.962
b = pr.purcell_layered(lam, 6e-9, 12e-9, 1.45, "Au", "perp").F               # 5.932082651691691
pr.purcell_layered(lam, 6e-9, 12e-9, 1.45, "Au", "perp", points=48).F - b     # 2.7e-15
pr.purcell_layered(lam, 6e-9, 12e-9, 1.45, "Au", "perp", detour=1.0).F - b    # -8.9e-16
```

## What is not claimed

- The dipole sits in air. A dipole inside the film, or in a medium of index
  other than 1, is not covered (the normalisation `P0` would change).
- A lossy film is refused (`film_index` must be real): the transmitted-flux
  accounting assumes the film absorbs nothing.
- `height_m = 0` is refused: on a lossy substrate the non-radiative rate is
  infinite there. Use a small height (the glass `d -> 0` check uses
  `lambda / 10^8`).
- `transmitted` is 0 for any substrate with loss, however small; a lossy
  dielectric's far-field transmission is not separated from its absorption.
