#!/bin/sh
# Run the time-domain result-card browser check (dockerfiles/rendercheck/
# tdcheck.mjs) against a real tiny solve, bounded.
#
#   scripts/run_tdcheck.sh            # the page scripts in this tree
#   scripts/run_tdcheck.sh a6127d3    # web/static/viewer.js from that revision
#
# Exit 0 = ALL PASS. Run from the repo root. The solve is tiny and memory-capped;
# the browser runs in the rendercheck image with --cpus=2 --memory=2g, and the
# whole check is cut at 10 minutes. The container is named, and removed on any
# exit, so a killed run leaves none behind.
set -eu
IMAGE=${RENDERCHECK_IMAGE:-harbor.arcadia-ecs.local/mode-search-engine/mse-rendercheck:34cba6e}
DIR=$(mktemp -d)
NAME=mse-tdcheck-$$
DOCKER="docker --config ${DOCKER_CONFIG:-$HOME/.docker-mse}"
trap '$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$DIR"' EXIT
trap 'exit 130' INT TERM HUP

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

timeout 600 $DOCKER run --rm --name "$NAME" \
  --cpus=2 --memory=2g -v "$DIR:/work:ro" --entrypoint node \
  -w /opt/rendercheck "$IMAGE" /work/tdcheck.mjs &
# waited on, not run in the foreground: sh runs a trap only after a foreground
# command ends, so a TERM to this script would otherwise wait out the browser
wait $!
