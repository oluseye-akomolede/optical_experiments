# source-cell-permittivity

## Why

`fdtd.delivered_power`, `delivered_power_spectrum`, `purcell_spectrum` and
`dipole_moment` take a scalar `eps_r`: the `eps` in `J = -eps0 eps S`, the
current the source's `dt * S` stands for, and the `eps` in the work
`W_n = eps0 eps V dt S_n (post_n + post_{n-1}) / 2`. That is right only when
it is the permittivity the source's own E sample is updated with.
`scripts/paper1_nanowire.py` passes `lay["eps"][cell]`, the layout's node
value. This is the deferred item "The dielectric as a function, not a
constant" of `paper1-nanowire-purcell`.

How `simulate` builds the coefficient (line numbers on `main` at 9b627d6,
`dockerfiles/maxwell_solver/fdtd.py`): it casts the layout's array to float32
(`eps = xp.asarray(eps_r, dtype=f32)`, 2509), forms
`inv_eps = 1 / (EPS0 * eps)` once (2658), and advances each E component with
the node value at its own array index -- `Ex[:-1, sy, sz] += dt *
inv_eps[:-1, sy, sz] * ...` (2918), `Ey` (2925), `Ez` (2932) -- with no
averaging of its own (any sub-cell averaging happened when the layout built
the array, `fdtd_grid.average_eps`); the only override is a Lorentz-Drude
term, which marks every sample whose segment has a metal node (684-706) and
rescales its curl increment by `eps_r - 1` (718, applied in `advance`, 748) so
that sample is updated with `eps_inf = 1`.

So in a dielectric, at an interface or not, the node value the script passes
IS the applied one, and the scalar reading is not wrong there. It is wrong on
a metal's sample: measured on a 37^3 grid (dx 10 nm, CPML 6, CW Ez dipole at
16 cells per wavelength, phasor box of half-extent 8 read over 8 periods), a
source in eps 4 with a lossless Lorentz node above it along its axis reads
the surface power +0.0008 off with the applied 1 and +3.0032 off with the
cell's 4. The same grid with the source on an eps-4 block's face (no metal)
reads +0.0023 with the cell value (vacuum: +0.0016).

## What Changes

- `fdtd.source_permittivity(eps_r, cell, component, *, ade_terms=(),
  boundary="metal")`: the permittivity the update applies at a point source's
  sample -- `float(eps_r[cell])`, with every term that claims the sample
  adding its rescale `eps / eps_inf - 1` as `advance` does.
- `LorentzDrude.samples(component, shape, periodic)`: the mask of samples the
  metal owns, now the one place that rule lives; `bind` builds its storage
  from it and `LorentzDrude.applied_permittivity` reads it. `EPS_INF = 1.0`
  names the value the rescale targets (bit-identical arithmetic).
- `simulate` returns `"source_eps"` for a point source (`None` otherwise),
  computed by `source_permittivity` from the run's own `eps_r`, `ade_terms`
  and `boundary`. Every other record is unchanged.
- The readers keep their scalar `eps_r`; their docstrings name
  `"source_eps"` as the value to pass.
- `scripts/paper1_nanowire.py` reads with the applied value and prints both:
  `source    node (...), eps_r <cell> (cell), <applied> (applied at E<axis>)`.
  The host index stays `sqrt` of the node value (the material's index), and
  `_run` refuses a run whose `"source_eps"` differs from the printed value.

## Impact

- `dockerfiles/maxwell_solver/fdtd.py`, `scripts/paper1_nanowire.py`;
  `tests/test_fdtd_source_eps.py` (new, 14 tests, about 3.5 s at
  `OMP_NUM_THREADS=2`), one test in `tests/test_paper1_nanowire.py`, five
  cases in `tests/check_discriminating.py`.
- No caller in a dielectric sees a different number: the applied value there
  is `eps_r[cell]` exactly (the caller's float64, not the float32 grid copy).
- The float32 rounding of the grid's coefficient (under 6e-8 relative) is not
  part of the value.
