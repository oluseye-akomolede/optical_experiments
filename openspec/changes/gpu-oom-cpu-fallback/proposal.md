# GPU out-of-memory on the shared Jetson: retry on the CPU, and say why

## Why

A production job (a small two-part upload, 10,661 tets, 10 modes) failed on the
Jetson with:

    EigenError: could not factor A + sB: cudaErrorMemoryAllocation: out of memory

The same shape solves fine with `MSE_SOLVER_BACKEND=cpu`. The cause was not the
object: another rig's TensorRT demo was holding Jetson device memory, leaving
~370 MB free of 7.6 GB at the moment the factorisation ran. The Jetson is a
shared machine, so this will happen again.

Two things are wrong with how that failure is handled today:

1. **Nothing retries.** `mse_backend` documents "there is no fallback" as a
   *deliberate* rule, and for its original purpose it is right: a `gpu` run on
   a machine with no CUDA stack must not silently solve on the CPU, because that
   turns "the GPU run was slow" into a mystery with no error in the log. But
   "the CUDA stack is absent" is a *configuration* fact, known before any work
   is done. "The device was out of memory when this factorisation asked for
   it" is a *transient contention* fact, arriving after the mesh is built and
   assembly has already been paid for. Re-running that problem on the CPU is not
   hiding a mistake; it is the only way to finish work that is about to be
   thrown away.
2. **The owner's message is the only defence.** `owner_message` already handles
   this correctly -- since 2026-09-27 an OOM is described as the GPU's memory
   being full, never as the object being too complex -- because that branch was
   written when the same shared-Jetson event first appeared. It exists to be
   read; it does not make the job succeed. A 10,661-element object that the same
   machine solves in seconds on the CPU was reported to the owner as a failure.

The starting state, honestly: no retry existed, and a generic EigenError text
is what a retry-less OOM degrades to if the message branch is ever bypassed.
The blame-the-object wording for an OOM was NOT this change's to fix -- it was
already fixed on 2026-09-27.

That was not the end of it. A second production job (e9423834) failed on the
SAME board and the owner was still told to "try a simpler object". The chain
there was a different shape neither the memory words nor the GPU-init words
recognised: `NvMapMemAllocInternalTagged: ... error 12` (the driver's
carve-out was exhausted) followed by `DynamicLibNotFoundError` for
`libcudss.so.0`. So this change also has to recognise what the real failure
looked like, and say something true about it.

## What changes

In the GPU path (`mse_backend`, reached from `mse_eigen.solve_modes`), when a
factorisation or solve fails with an out-of-memory error, retry the **same
problem once** on the CPU backend, provided the CPU estimate fits this worker's
limits. Log one line saying the fallback happened and why.

The owner-facing message for a failure that is still out of memory after that
SHALL say the solver machine was out of memory or busy, and SHALL NOT say the
object is too complex.

The distinction this rests on, and what it does NOT relax:

- Absent CUDA stack -> **still raises**, unchanged. That is a configuration
  error and the CPU path must not mask it.
- Out of memory -> fallback once. A second OOM, or an OOM the CPU cannot fit,
  raises, and the message names the machine.

## Impact

- `dockerfiles/maxwell_solver/mse_backend.py` — the OOM classifier and the
  one-shot CPU retry.
- `dockerfiles/maxwell_solver/mse_eigen.py` — `solve_modes` retries through the
  same path so both factorisations and the eigensolve are covered.
- `dockerfiles/maxwell_solver/mse_maxwell_solver.py` — `owner_message` stops
  advising "a simpler object" for a memory failure; the run's reported backend
  says when it fell back.
- `openspec/specs/maxwell-solver/spec.md` — the backend requirement gains the
  fallback and its boundary. The "no fallback" sentence is narrowed, not
  deleted: its configuration case stays.
- Tests: `tests/test_backend.py`, `tests/test_maxwell_solver.py` (tier 1, fakes).

No deploy. Numbers come from fakes; the real path is verified by the reviewer
against the spec.
