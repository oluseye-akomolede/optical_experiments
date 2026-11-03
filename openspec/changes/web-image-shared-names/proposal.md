# Give the web image the backend NAMES without the backend module

## Why

The A4000 deploy (mse-web `072ed7a`) crashlooped and was rolled back on
2026-11-03 04:55Z:

    /opt/app/jobs.py:28: import mse_backend -> ModuleNotFoundError

The site was down about two minutes, and the Recreate strategy is why it was
down at all rather than degraded. The cause is an import I introduced in
`a5fdf5d`, while fixing the reviewer's blocker that `MSE_SOLVER_BACKEND` was the
literal `"cuda"`.

That fix was correct about the VALUE and wrong about the SOURCE. `web/jobs.py`
needed the backend name, and `mse_backend` is the module that defines it -- but
`mse_backend` imports scipy at module level (for the SuperLU factorisations),
and the web image has no scipy. So importing it fixed the backend value and
broke the image.

`tests/test_web_imports.py` catches this. It FAILS on main. It was not run:
not by my suites, not by review, not by the lead's. That is the part worth
recording -- the guard existed, worked, and was simply not in anyone's run
list, which is the same "check that could not fail" shape the file's own
docstring warns about, one level up.

## What changes

1. `mse_common` defines `CPU`, `GPU` and `BACKENDS`. It is the one module both
   images already COPY, and it imports only numpy.
2. `mse_backend` **re-exports** them (`from mse_common import ...`). Every
   existing `mse_backend.GPU` / `mse_backend.BACKENDS` caller is unchanged, and
   there remains exactly one definition.
3. `web/jobs.py` reads `COMMON.GPU`. It does not import `mse_backend`.
4. `mse_backend.py` is deliberately NOT copied into the web image. Copying it
   would drag in scipy, which is the dependency this whole split exists to avoid.

Separately, on the same branch: `requirements-gpu.txt` pins
`cuda-bindings==12.9.9`. The A4000 smoke Job warned that cuda-bindings was built
for CUDA 13 against a CUDA 12 driver. The solve was still exact (`rel_residual`
1.0e-16 at n=20000), but an unpinned cuda-bindings floats onto 13.x wheels and
the next rebuild would differ from this one silently.

## Impact

- `dockerfiles/maxwell_solver/mse_common.py`, `mse_backend.py`
- `web/jobs.py`
- `dockerfiles/maxwell_solver/requirements-gpu.txt`
- No live quota, deployment or cluster change: the lead deploys.
- No behaviour change on the solver side: the backend names are identical and
  bound, not copied.

## What this does NOT fix

`cuda-bindings==12.9.9` is pinned from the Jetson's freeze and verified to
resolve alongside the rest of `requirements-gpu.txt`, but the GPU image must be
rebuilt and its probe re-read to confirm the warning is gone. The pin makes the
result reproducible; it does not by itself prove the A4000 driver accepts it.
