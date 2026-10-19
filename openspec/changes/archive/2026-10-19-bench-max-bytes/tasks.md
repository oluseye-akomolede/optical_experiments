# bench-max-bytes — tasks

- [x] 1. `fdtd.check_limits`: add `max_bytes=None` as the fourth ceiling, defaulting to
      `MAX_BYTES` read at call time; the memory refusal and its `exceeded` tuple name the
      ceiling that applied. Docstring updated to four ceilings and the measured 11.55 GB case.
- [x] 2. `scripts/paper1_nanowire.py`: `--max-bytes GB` built with `_raised_cap` (with a
      GB-formatted `show=` so the floor refusal reads "2.0 GB", not bytes); `run_limits`
      carries `max_bytes` with `max()`; both `raised` computations (report, ladder) read it;
      the `caps` line names the memory ceiling in its parenthetical.
- [x] 3. Spec delta: `openspec/changes/bench-max-bytes/specs/time-domain/spec.md` adds the
      memory-ceiling requirement and its four scenarios.
- [x] 4. Tests in `tests/test_fdtd_estimate.py`: default refuses 11.55e9 naming 3.0 GB;
      `max_bytes=14e9` admits; 14e9+1 refuses naming 14.0 GB; default is read at call time
      (moved `MAX_BYTES` moves it). Tests in `tests/test_paper1_nanowire.py`: the four
      `run_limits` expectations gain `max_bytes`; the parser reads GB and clamps; the lowering
      tests cover `--max-bytes 2` and a hand-built namespace; a memory-only raise prints the
      parenthetical and the plain call prints none.
- [x] 5. Run `tests/test_fdtd_estimate.py` and `tests/test_paper1_nanowire.py` (background,
      `nice -n 19`, `ulimit -v 4194304`); both green recorded in the row.
      22 passed in 0.39 s; 37 passed in 1324.56 s (22:04).
- [x] 6. Mutation check: flip `e["bytes"] > max_bytes` back to `MAX_BYTES` and confirm a test
      goes red; restore and verify by diff. Also froze the default (`max_bytes = 3.0e9`).
      First mutant killed `test_a_bench_may_raise_the_memory_ceiling_and_the_default_is_the_products_own`
      only; second killed `test_the_memory_default_is_the_constant_at_call_time` only.
      Restored: sha256 `dcaf7273...381d68` matches the pre-mutation file, 22 passed.
- [x] 7. `openspec validate bench-max-bytes --strict` (exit 0) and
      `openspec validate --specs --strict` (12 passed, 0 failed).
