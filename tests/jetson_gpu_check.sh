#!/usr/bin/env bash
# Tier-2 GPU check for the solver backend -- run BY THE LEAD on the Jetson.
#
# What this does NOT do: prove anything about cuDSS that the VM tests could
# have proved. The VM proves the routing, the error paths, the plumbing and the
# refinement arithmetic. This run is the only thing that shows the cuDSS
# factorisation is correct and that single+refine really matches double.
#
# Usage (on the Jetson, from a checkout on the branch under test):
#   bash tests/jetson_gpu_check.sh <path-to-cudss-0.7.1>
#
# Copy this to the Jetson or run it from the shared checkout; it takes the cuDSS
# prefix as its only argument because that path is machine-specific.

set -euo pipefail

CUDSS="${1:?usage: $0 <cudss-0.7.1-prefix>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Jetson environment. All four are load-bearing; the README in
# scratch/gpu-reference/ records what breaks without them.
export LD_LIBRARY_PATH="$CUDSS/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDSS_MT_LIB="$CUDSS/lib/libcudss_mtlayer_gomp.so.0"
export CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1
export PYTHONPATH="$REPO/dockerfiles/maxwell_solver:$REPO/tests:${PYTHONPATH:-}"

# nvmath-python must be 0.9.0 (1.0's CUDA-13 bindings abort on the CUDA 12
# driver) and cuDSS must be the linux-aarch64 redist, not the pip wheel.
python3 - <<'PY'
import nvmath, sys
print(f"[check] nvmath-python {nvmath.__version__}", flush=True)
if nvmath.__version__.split(".")[0] != "0":
    sys.exit("nvmath-python must be 0.9.x on this board")
PY

# Memory cap by cgroup, NOT ulimit -v: CUDA cannot initialise under an
# address-space limit. 5G is the bench figure; raise it only if a real case
# needs it, and remember the board shares RAM between CPU and GPU.
run() {
  local backend="$1" precision="$2" label="$3"
  echo "=== $label ==="
  systemd-run --user --scope -p MemoryMax=5G --quiet \
    env MSE_SOLVER_BACKEND="$backend" \
        MSE_GPU_PRECISION="$precision" \
        MSE_THREADS=4 \
    bash -c "cd '$REPO' && python3 tests/bench_box.py --n-modes 20"
}

# 1. the CPU reference
run cpu double "cpu baseline (what the cluster produces)"

# 2. the GPU default: single precision with refinement
run gpu single "gpu, single+refine (the production default)"

# 3. the GPU with the double override
run gpu double "gpu, double (the override)"

cat <<'EOF'

What to compare across the three runs:
  * [mse] solver backend: -- must read "cpu", then "gpu (single+refine)", then
    "gpu", and the gpu lines must carry "worst refined residual".
  * the eigenvalue table -- all three must agree to solver precision; this is
    the accuracy check, and it is the point of the run.
  * wall time -- confirms the GPU is actually being used (the 266k-tet figure
    was ~19 s of CPU-side S factorisation alone; single was ~2x the double
    factor speed).
  * the worst refined residual -- the lead's measured figure at 266k was ~3e-12.
    Anything approaching 1e-6 means the refinement is barely converging and the
    check that now guards it would raise on a slightly harder matrix.

If MSE_GPU_PRECISION=single raises "did not converge", that is the new guard
working: raise MSE_REFINE_STEPS (default 3) or accept the doubled memory of
MSE_GPU_PRECISION=double, and report which case hit it.
EOF
