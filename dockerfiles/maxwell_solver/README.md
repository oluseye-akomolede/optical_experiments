# MSE Maxwell cavity-mode solver

Computes the lowest **optical** modes of a closed perfectly-conducting (PEC)
cavity: the owner uploads a watertight `.obj`, the solver treats its interior as
a cavity filled with a uniform medium of refractive index `n`, and solves the
full-vector electromagnetic eigenproblem

    curl(curl E) = k0^2 n^2 E,      n x E = 0 on the boundary

for the lowest `n_modes` modes, writing the natural frequency, the in-medium
wavenumber `k`, the vacuum wavelength, and a sampled `E` field per mode.

It is a plain Python program (no Spark, no Java). In the cluster it runs as a
`batch/v1` Job in namespace `mode-search-engine`, launched by the web app; the
FEM work runs on the worker nodes, never on the control plane.

## Running

    python mse_maxwell_solver.py \
        --job-id <uuid> --units mm --n-index 1.0 --n-modes 20

`--local-in DIR` / `--local-out DIR` run the same pipeline against a directory
with no MinIO and no database, which is how the tests and the bench exercise it.

| flag | meaning |
| --- | --- |
| `--units` | `mm` (default), `cm`, `m`, `in` |
| `--n-index` | refractive index of the filling medium, default `1.0` |
| `--n-modes` | how many modes to compute, default `20` |
| `--max-tets` | element cap (`MSE_MAX_TETS`, default 80000) |
| `--grid` | samples along the longest axis of the output field grid |
| `--epw` | edges per wavelength for the resolution rule (`MSE_EDGES_PER_WAVELENGTH`, default 7) |

Credentials come from the environment only (`MINIO_*`, `DB_*`, `MSE_BUCKET`).

## Method

Lowest-order **Nédélec (Whitney) edge elements** on linear tetrahedra. The weak
form is `∫ curl E · curl v = k^2 ∫ ε_r E · v` with `ε_r = n^2`; PEC is enforced
by eliminating every edge lying on the boundary surface. Element matrices have
closed forms and are assembled vectorized with numpy, then the eigenproblem is
solved with a projected shift-invert Lanczos iteration (`mse_eigen.py`).

Mesh resolution is set by **Weyl's law**, `N(k) ~ V k^3 / (3 pi^2)`, so the
target element edge comes from the *enclosed volume* and the cube root of the
mode count — not linearly from the mode number, and not from the bounding box.

## Known limitations

**Planar-heavy shapes over-count elements.** TetGen cannot make interior
elements much larger than the surface facets, so an over-fine surface floors the
whole mesh. `decimate_to_edge_length` coarsens such a surface with quadric
decimation before tetrahedralizing, which resolves the common case (a scan or
CAD export). But a shape whose facets are **coplanar** decimates to a planar
triangulation with long, thin diagonal slivers, which TetGen must respect: an
over-refined 10x7x5 mm box meshes at about 45,700 elements where the same box
from a 12-face surface needs 10,375, a 4.4x over-count. The result is still
under the element cap and still accurate (the box bench passes at 0.29%), so it
is a cost, not a correctness problem. Curved shapes (a sphere) are unaffected:
they decimate cleanly to within ~15% of the ideal count.

Note the tet count is **not monotonic** in the decimated face count, so
decimating harder is not a general fix; the solver targets the
isotropically-correct face count for the resolution edge instead.

**Heavy runs belong on the worker nodes.** Because the meshing and eigen solve
are memory-hungry, do not run them on the openrig VM or on `sigma-control1`.
Locally only tiny unit tests (a few dozen tets, pure numpy) are appropriate; the
bench (`tests/bench_box.py`) is a tier-2 tool for the Jetson or a bounded
Kubernetes Job, not a laptop command.
