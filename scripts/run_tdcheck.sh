#!/bin/sh
# Run the time-domain result-card browser check (dockerfiles/rendercheck/
# tdcheck.mjs), bounded.
#
#   scripts/run_tdcheck.sh                      # a tiny real solve, built here
#   scripts/run_tdcheck.sh --rev a6127d3        # ...with viewer.js from that revision
#   scripts/run_tdcheck.sh --dir /tmp/tdmirror  # a real job mirrored by
#                                               # scripts/tdcheck_fetch.py
#
# Exit 0 = ALL PASS. Run from the repo root. The solve is tiny and memory-capped;
# the browser runs in the rendercheck image with --cpus=2 --memory=2g, and the
# whole check is cut at 10 minutes. The container is named and removed when the
# script exits or is stopped by INT, TERM or HUP. A SIGKILL of this script runs
# no trap and leaves the container up: remove it with docker rm -f mse-tdcheck-<pid>.
#
# A --dir mirror is never written to: --rev on one works on a copy.
set -eu
IMAGE=${RENDERCHECK_IMAGE:-harbor.arcadia-ecs.local/mode-search-engine/mse-rendercheck:34cba6e}
DIR=; REV=; MADE=
NAME=mse-tdcheck-$$
DOCKER="docker --config ${DOCKER_CONFIG:-$HOME/.docker-mse}"
USAGE="usage: scripts/run_tdcheck.sh [--dir DIR] [--rev REV]"
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 2; }; DIR=$2; shift 2 ;;
    --dir=*) DIR=${1#--dir=}; shift ;;
    --rev) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 2; }; REV=$2; shift 2 ;;
    --rev=*) REV=${1#--rev=}; shift ;;
    --*) echo "$USAGE" >&2; exit 2 ;;
    *) REV=$1; shift ;;                       # a bare revision, as before
  esac
done
cleanup() { $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true; [ -z "$MADE" ] || rm -rf "$MADE"; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
CHECK=dockerfiles/rendercheck/tdcheck.mjs
[ -f "$CHECK" ] || { echo "$CHECK not found: run from the repo root" >&2; exit 2; }

MIRROR=$DIR
if [ -n "$MIRROR" ]; then
  # Everything the check reads, checked before the browser starts: a mirror
  # with a hole in it must fail here, naming the hole, not as a page defect.
  # The list is every file tdcheck.mjs reads by name -- tests/test_run_tdcheck.py
  # derives it from that file's own source and fails if the check grows a read
  # this misses -- plus the page's entry module and the movie module it imports.
  for f in job.html status.json outline.glb files/spectrum.json files/trace.bin \
           files static/tdplot.js static/viewer.js static/tdmovie.js; do
    [ -r "$MIRROR/$f" ] || { echo "not a job mirror: $MIRROR/$f is not readable" >&2; exit 2; }
  done
  echo "mirror: $MIRROR"
else
  DIR=$(mktemp -d); MADE=$DIR
  (ulimit -v 4194304; PYTHONDONTWRITEBYTECODE=1 nice -n 19 .venv-mse/bin/python -B \
    scripts/tdcheck_fixture.py "$DIR") >"$DIR/fixture.log" 2>&1 \
    || { tail -20 "$DIR/fixture.log"; exit 2; }
  tail -1 "$DIR/fixture.log"
  chmod -R a+rX "$DIR"
fi
if [ -n "$REV" ]; then
  # The fixture is disposable and writable in place; a mirror is copied first.
  if [ -z "$MADE" ]; then MADE=$(mktemp -d); DIR=$MADE; cp -a "$MIRROR/." "$DIR/"; chmod -R a+rX "$DIR"; fi
  git show "$REV:web/static/viewer.js" >"$DIR/static/viewer.js"
  echo "viewer.js from $REV"
fi

timeout 600 $DOCKER run --rm --name "$NAME" \
  --cpus=2 --memory=2g -v "$DIR:/work:ro" \
  -v "$(pwd)/$CHECK:/opt/rendercheck/tdcheck.mjs:ro" \
  --entrypoint node "$IMAGE" /opt/rendercheck/tdcheck.mjs &
# waited on, not run in the foreground: sh runs a trap only after a foreground
# command ends, so a TERM to this script would otherwise wait out the browser
wait $!
