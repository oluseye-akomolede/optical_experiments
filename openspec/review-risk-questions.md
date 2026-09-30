# Review-risk questions (mse)

Yes/no questions asked of each builder change, answered from the diff alone.
Any "yes" means the change is risky and gets a full review (reviewer checks it against the
spec, with a test that reddens on the bug); all "no" means it gets a simple review.

1. **Does it change the solver's physics: FEM assembly, boundary conditions, the eigen solve, or how per-part refractive index enters?**
   A wrong result here is a plausible-looking wrong frequency; nothing crashes.
2. **Does it change units, scale or length conversion anywhere (the size x scale x unit chain, mm/m, the mesh edge)?**
   Unit and scale mix-ups are this codebase's most repeated bug; they give objects 1000x off, and estimates wrong by up to 27x.
3. **Does it change the mesh-size or tet-count estimate, decimation, or the limits that refuse an upload?**
   These gates are what keep an oversized job from exhausting memory on the Jetson or the cluster.
4. **Does it change job states, claim/lease/release, or any reaper or timeout that ends a job?**
   A reaper once killed every Jetson job; a release that doesn't reset status leaves a job stuck forever.
5. **Does it delete or overwrite stored data (DB rows, MinIO objects, result files), or change the DB schema?**
   Owner uploads and solved modes can't be regenerated for free.
6. **Does it change resource bounds: k8s limits/quota, thread counts, Jetson MemoryMax, or where a job runs?**
   Unbounded work has crashed the shared VM and hits the power-capped namespace.
7. **Does it add or widen an exception handler, a default/fallback value, or an `available`/status flag?**
   Our signature defect is "could not ask" reported as "nothing there": a swallowed error shown as an empty result.
8. **Does it change a route, a JSON payload shape, or a file name that the viewer or the worker consumes?**
   The web app, the worker and the viewer are deployed separately; a one-sided change breaks the other side silently.
9. **Does it change how the viewer maps field data to the screen: array vs grid axes, the raymarch frame, or the shared colour scale?**
   These bugs draw a confident but wrong picture (x/z swapped, glow in the wrong place, a faint mode auto-brightened).
10. **Does it change a Dockerfile, requirements, entrypoint, module-level import, or k8s manifest?**
    Green tests have shipped an image that crashlooped on import; the built image must be smoke-tested.
11. **Does it touch credentials, tokens or DSNs, or add a call to an external service (Postgres, MinIO, Harbor, Slack)?**
    Secrets must come from the environment and never reach argv, logs or the DeepSeek seats.
12. **Does it change the GPU/CPU solver backend choice or the precision check (the post-refinement residual bound)?**
    Single precision is only safe because a residual above the bound raises; weakening that hides wrong modes.
