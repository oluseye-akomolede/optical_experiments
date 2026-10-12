# Tasks — td-bloch-v6-benchmark

- [ ] 1. Benchmark V6 against the existing pc band solver: an extruded 2-D
  crystal, TM and TE, kz = 0, frequencies at chosen k-points within 1%, every
  band present. (Was `time-domain-fdtd` task 6.2.)
  Bench `scripts/fdtd_bench_v6.py` written, with the homogeneous |k + G| / n
  check at six k (tier-1 `--tiny`: four k, 26 s, in tests/test_fdtd_bench_v6.py).
  Crystal lines also pass a two-resolution convergence gate (48 and 64
  cells, eps sampled 9 per cell edge, 150 periods recorded).
  Ticked when the lead's six Jetson commands (its docstring) all exit 0.
  Not started without the owner's direction.
