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
# whole check is cut at a cap derived from the movie's frame count (see CAP
# below; TDCHECK_TIMEOUT overrides it and the cap is printed). The container is
# named and removed when the
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

# The cap. The check draws every movie frame through software WebGL in this
# 2-CPU container, so its cost follows the movie's length, and a fixed 600 s is
# not enough: the real production job 180cb65f (60 frames, two-digit names) was
# killed at rc=124 after 45 PASS, and the 120-frame open-boundary fixture
# mirror here needed ~740 s on this loaded VM. 12 s a frame is ~2x the slowest
# per-frame cost measured on that fixture (740 s / 120 = 6.2 s, against 2.2 s
# on an idle VM), and the floor keeps a short movie bounded. A mirror that is
# not a movie at all (no manifest to read) is given the solver's own maximum.
FRAMES=120
if [ -r "$DIR/files/spectrum.json" ]; then
  FRAMES=$(.venv-mse/bin/python -c 'import json,sys
frames = json.load(open(sys.argv[1])).get("frames") or {}
print(int(frames.get("count") or 120))' "$DIR/files/spectrum.json" 2>/dev/null || echo 120)
fi
CAP=${TDCHECK_TIMEOUT:-$(( 180 + 12 * FRAMES ))}
echo "cap: ${CAP}s for ${FRAMES} frames (TDCHECK_TIMEOUT overrides)"

timeout "$CAP" $DOCKER run --rm --name "$NAME" \
  --cpus=2 --memory=2g -v "$DIR:/work:ro" \
  -v "$(pwd)/$CHECK:/opt/rendercheck/tdcheck.mjs:ro" \
  --entrypoint node "$IMAGE" /opt/rendercheck/tdcheck.mjs &
# waited on, not run in the foreground: sh runs a trap only after a foreground
# command ends, so a TERM to this script would otherwise wait out the browser
rc=0
wait $! || rc=$?
if [ "$rc" = 124 ]; then
  # 124 is timeout(1)'s own status, not a verdict from the page: name the cap
  # and the frame count, so a run cut short is never read as a failed page.
  echo "harness timed out after ${CAP}s (${FRAMES} frames; TDCHECK_TIMEOUT=${TDCHECK_TIMEOUT:-unset}): the check was cut while still running" >&2
  exit 124
fi
exit "$rc"
