#!/bin/sh
# Run the time-domain result-card browser check (dockerfiles/rendercheck/
# tdcheck.mjs) against a real tiny solve, bounded.
#
#   scripts/run_tdcheck.sh            # the page scripts in this tree
#   scripts/run_tdcheck.sh a6127d3    # web/static/viewer.js from that revision
#
# Exit 0 = ALL PASS. Run from the repo root. The solve is tiny and memory-capped;
# the browser runs in the rendercheck image with --cpus=2 --memory=2g, and the
# whole check is cut at 10 minutes.
set -eu
IMAGE=${RENDERCHECK_IMAGE:-harbor.arcadia-ecs.local/mode-search-engine/mse-rendercheck:34cba6e}
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT

(ulimit -v 4194304; PYTHONDONTWRITEBYTECODE=1 nice -n 19 .venv-mse/bin/python -B \
  scripts/tdcheck_fixture.py "$DIR") >"$DIR/fixture.log" 2>&1 \
  || { tail -20 "$DIR/fixture.log"; exit 2; }
tail -1 "$DIR/fixture.log"
cp dockerfiles/rendercheck/tdcheck.mjs "$DIR/"
if [ $# -ge 1 ]; then
  git show "$1:web/static/viewer.js" >"$DIR/static/viewer.js"
  echo "viewer.js from $1"
fi
chmod -R a+rX "$DIR"

timeout 600 docker --config "${DOCKER_CONFIG:-$HOME/.docker-mse}" run --rm \
  --cpus=2 --memory=2g -v "$DIR:/work:ro" --entrypoint node \
  -w /opt/rendercheck "$IMAGE" /work/tdcheck.mjs
