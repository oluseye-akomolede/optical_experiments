# Tasks: give the web image the backend names without the backend module

## 1. The shared constant

- [x] 1.1 `mse_common`: define `CPU`, `GPU`, `BACKENDS`, with the docstring
      stating why they are here and not in `mse_backend`.
- [x] 1.2 `mse_backend`: re-export them from `mse_common` instead of defining
      them, so every existing caller keeps working and there is one definition.
- [x] 1.3 `web/jobs.py`: take `COMMON.GPU`; drop `import mse_backend`.
- [x] 1.4 Do NOT copy `mse_backend.py` into the web image: it imports scipy.

## 2. The GPU pin

- [x] 2.1 `requirements-gpu.txt`: `cuda-bindings==12.9.9`, read from the
      Jetson's `gpu-venv` and `harness-venv` (both 12.9.9, 2026-11-03).
- [x] 2.2 Verify the pin resolves WITH the rest of the file, not alone: a
      `pip install --dry-run` over these plus `requirements.txt` resolves
      cuda-bindings 12.9.9, cuda-core 0.7.0, cuda-pathfinder 1.8.5,
      cuda-toolkit 12.9.2.0, nvmath-python 0.9.0, cupy-cuda12x 14.2.0, no conflict.

## 3. Proof

- [x] 3.1 Reproduce the red on main's image, not just in a test:
      build `072ed7a`'s web image, then
      `docker run --entrypoint python <img> -c "import app"`
      -> `ModuleNotFoundError: No module named 'mse_backend'` at jobs.py:28.
- [x] 3.2 Same command on the fixed image -> `APP IMPORT OK`.
- [x] 3.3 `tests/test_web_imports.py`: 1 failed on main's `jobs.py`, 13 passed
      on the fix. The `[jobs]` case is the specific red.
- [x] 3.4 `test_web_app.py`, `test_backend.py`, `test_worker.py`,
      `test_web_imports.py`: 160 passed.
- [ ] 3.5 Rebuild the GPU image and confirm the probe still prints
      `GPU STACK IMPORTS OK` + `cuDSS LOADS OK` and `pip show cuda-bindings`
      reports 12.x.

## Notes

**Two ways this went wrong while fixing it, both worth keeping.**

The first rebuild of the fixed image shipped a tree where one of my four edits
had been silently reverted. I had copied main's version of three files over mine
to prove the routing failures were pre-existing, then restored them with
`git checkout --`, and `git checkout --` restored the COMMITTED version, which
did not contain my uncommitted `mse_common` edit. The suite and the container
both went green afterwards, so nothing in the output would have told me. It was
caught only by grepping for the new constant before trusting the rebuild. A
restore step that overwrites uncommitted work cannot be verified by the tests
that follow it.

**The six `test_routing.py` failures are NOT from this change.** They are
`ModuleNotFoundError: No module named 'tetgen'` in this seat's interpreter, and
the identical six fail with main's versions of every file this change touches.
Stated so a later reader does not attribute them to this branch.

**The guard that missed this already existed.** `test_web_imports.py` was
written after the dispatch crashloop for exactly this class of failure. It
worked. It was not in the run list, so the second occurrence of the same defect
got past three seats. The check was never the gap; running it was.
