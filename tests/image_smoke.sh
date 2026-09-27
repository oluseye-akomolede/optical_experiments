#!/usr/bin/env bash
#
# Build both images and prove they can IMPORT -- the check that was missing.
#
# The dispatch deploy crashlooped because no test ever started the web image.
# Every unit test ran with the solver directory on sys.path, so `import app`
# worked in the suite and failed in the container. This script builds the two
# images and runs the smallest thing that asks the container's first question:
# can the entrypoint be imported with no database, no bucket and no network?
#
# It is deliberately NOT a test in the pytest suite: a docker build takes
# minutes and needs a daemon, and a suite that is slow gets skipped. Run it
# before handing off anything that touches web/ or dockerfiles/, and before a
# deploy -- see worker/README.md.
#
# Usage:  tests/image_smoke.sh [web|solver|both]     (default: both)
#
# DOCKER_CONFIG: this rig keeps its docker credentials in ~/.docker-mse and
# never touches the shared ~/.docker/config.json (see the rig memory). The
# variable is defaulted, not exported over, so an explicit value wins.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-both}"
TAG_BASE="${IMAGE_SMOKE_TAG_BASE:-mse-imagesmoke}"

export DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker-mse}"

log() { printf '\n=== %s\n' "$*" >&2; }
fail() { printf '\n!!! %s\n' "$*" >&2; exit 1; }

# A build that cannot reach the daemon should say so in one line, not in a
# traceback that reads like a Dockerfile fault.
docker info >/dev/null 2>&1 \
    || fail "no docker daemon reachable (DOCKER_CONFIG=$DOCKER_CONFIG)"

build_and_import() {
    local name="$1" dockerfile="$2" tag="$3" import_expr="$4"
    log "building $name ($dockerfile)"
    docker build -f "$ROOT/$dockerfile" -t "$tag" "$ROOT"
    log "importing inside $name"
    # --network none and no env: if the import needs the DB or MinIO it must
    # fail HERE, where we are looking at it, rather than in the cluster.
    docker run --rm --network none "$tag" python -c "$import_expr"
    log "$name: import OK"
}

web() {
    build_and_import "web" "dockerfiles/web/Dockerfile" "$TAG_BASE-web:local" \
        'import app; app.build_app; import config, meshing, jobs, worker_api; print("web import ok")'
}

solver() {
    build_and_import "solver" "dockerfiles/maxwell_solver/Dockerfile" \
        "$TAG_BASE-solver:local" \
        'import mse_maxwell_solver as S; S.max_tets; import mse_common, mse_meshing; print("solver import ok")'
}

case "$WHICH" in
    web)    web ;;
    solver) solver ;;
    both)   web; solver ;;
    *)      fail "unknown target $WHICH (expected web, solver or both)" ;;
esac

log "image smoke passed: $WHICH"
