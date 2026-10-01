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
    #
    # --entrypoint python OVERRIDES THE IMAGE'S ENTRYPOINT, and that override is
    # the whole reason this leg works. Docker APPENDS a `docker run` command to
    # the image's ENTRYPOINT rather than replacing it, and the solver image's
    # entrypoint is `python /opt/app/mse_maxwell_solver.py`. So the natural form
    #
    #     docker run --rm --network none "$tag" python -c "$import_expr"
    #
    # does not import anything: it runs the solver AS A SCRIPT with a bogus
    # --job-id, and the container exits 2 printing argparse's usage line. The
    # `usage: mse_maxwell_solver ... --job-id` that comes out of that reads
    # exactly like an import-time failure inside the module, and it sent this
    # leg's diagnosis chasing a stale `__pycache__` that was never at fault
    # (the baked .pyc's top level is `if __name__ == "__main__": sys.exit(main())`,
    # identical to the source). The web image has no ENTRYPOINT, so the same
    # form works there -- which is why only the solver leg ever failed.
    docker run --rm --network none --entrypoint python "$tag" -c "$import_expr"
    log "$name: import OK"
}

web() {
    build_and_import "web" "dockerfiles/web/Dockerfile" "$TAG_BASE-web:local" \
        'import app; app.build_app; import config, meshing, jobs, worker_api; print("web import ok")'
    web_multipart "$TAG_BASE-web:local"
    web_pc_plan "$TAG_BASE-web:local"
}

# The multi-part refusal leg: the SPLIT must actually run in the WEB image.
#
# This leg exists because the venv is a MORE capable environment than this
# container, and the difference hid a production defect. `trimesh`'s split()
# needs a "graph engine" -- networkx, or scipy as a fallback. networkx is absent
# everywhere; scipy is present in the dev venv and in the SOLVER image but
# ABSENT from the web image. So the upload's "which part is open?" refusal threw
# `no graph engines available!` in exactly the deployed environment that runs
# it, while the venv reported all-clear. `web/meshing.py` now splits with a
# numpy union-find instead, and this leg is what keeps it that way.
#
# Same transcript mechanism as the solver's geometry leg, and for the same
# measured reason: stdout from an attached `docker run` does not reach the host
# here, so a crashed probe and a passing probe both print nothing. The probe
# writes a file, this function copies it out, and it requires BOTH a first and a
# last line so a truncated run cannot read as a clean one. `-i` is required:
# without it `python -` reads an empty script and exits 0 having done nothing.
web_multipart() {
    local tag="$1" out rc
    log "multi-part refusals inside web"

    set +e
    docker run --rm -i --network none --entrypoint python "$tag" - \
        > /tmp/web_multipart_leg.txt 2>&1 <<'PROBE'
import sys

import numpy as np
import trimesh

import meshing

failures = []


def check(name, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + name
          + (("  " + detail) if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def obj_of(scene_dict):
    return trimesh.Scene(scene_dict).export(file_type="obj").encode("utf-8")


def open_box(extents=(2.0, 2.0, 2.0)):
    """A box with one face dropped, so it is not watertight."""
    box = trimesh.creation.box(extents=extents)
    box.update_faces(np.arange(len(box.faces) - 1))
    return box


print("PROBE-START", flush=True)

# --- the graph-engine absence is REAL here, and is the point of this leg ----
# Asserted rather than commented: if a future web image gains scipy or networkx,
# this leg's premise changes and that should be visible, not silent.
import importlib

engines = [m for m in ("networkx", "scipy") if importlib.util.find_spec(m)]
print("graph engines present in the web image: "
      + (", ".join(engines) or "none"), flush=True)
check("the web image really has no graph engine (the reason this leg exists)",
      engines == [])

# --- a closed two-part file splits, names in file order --------------------
closed = obj_of({
    "wall": trimesh.creation.box(extents=(6.0, 6.0, 6.0)),
    "core": trimesh.creation.box(extents=(2.0, 2.0, 2.0)),
})
names = meshing.parse_parts(closed, "mm")
check("a closed two-part file splits in the web image",
      names == ["wall", "core"], "names=" + str(names))

# --- an OPEN part is refused BY NAME ---------------------------------------
# This is the case that was broken: with the old split this raised
# "no graph engines available!" before the refusal could name anything.
open_core = obj_of({
    "wall": trimesh.creation.box(extents=(6.0, 6.0, 6.0)),
    "core": open_box(),
})
try:
    meshing.validate_upload("open.obj", open_core, 50 * 1024 * 1024)
    check("an open part in a two-part file is refused", False, "no refusal")
except meshing.UploadError as exc:
    message = str(exc)
    check("an open part in a two-part file is refused", True)
    check("and the refusal NAMES the open part", "'core'" in message,
          message)

# --- touching parts are refused, naming both ------------------------------
# measured: TetGen's docheck catches OVERLAP and MISSES touching, so touching
# needs its own instrument. Two boxes sharing a face plane exactly.
touch = obj_of({
    "a": trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
    "b": trimesh.creation.box(extents=(1.0, 1.0, 1.0))
         .apply_translation([1.0, 0.0, 0.0]),
})
try:
    meshing.parse_parts(touch, "mm")
    check("face-to-face touching parts are refused", False, "no refusal")
except meshing.UploadError as exc:
    message = str(exc)
    check("face-to-face touching parts are refused", True)
    check("and the refusal names both parts",
          "'a'" in message and "'b'" in message, message)

# --- duplicate group names are refused ------------------------------------
# Two parts the owner cannot tell apart are two parts a material table cannot
# address. Built as raw text because trimesh would merge same-named groups --
# and a bare name scan is exactly the thing that cannot see this.
duplicate = (
    "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\n"
    "o part\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    "v 5 0 0\nv 6 0 0\nv 5 1 0\nv 5 0 1\n"
    "o part\nf 5 6 7\nf 5 6 8\nf 5 7 8\nf 6 7 8\n"
).encode("utf-8")
try:
    meshing.parse_parts(duplicate, "mm")
    check("duplicate group names are refused", False, "no refusal")
except meshing.UploadError as exc:
    check("duplicate group names are refused", True)
    check("and the refusal names the duplicate", "'part'" in str(exc), str(exc))

# --- a single open part still gets the file-level message -----------------
# The control on the NAMING: a one-part file has no part to name, so the
# message must fall back rather than inventing a name. Without this, a change
# that always printed a part name would look correct on the two-part case.
solo = open_box().export(file_type="obj").encode("utf-8")
try:
    meshing.parse_parts(solo, "mm")
    check("a single open part is refused", False, "no refusal")
except meshing.UploadError:
    check("a single open part is refused", True)

print("PROBE-END", flush=True)
if failures:
    print("WEB MULTIPART LEG FAILED: %d case(s): %s" % (len(failures), failures),
          flush=True)
sys.exit(1 if failures else 0)
PROBE
    rc=$?
    set -e

    cat /tmp/web_multipart_leg.txt >&2
    if ! grep -q "^PROBE-START" /tmp/web_multipart_leg.txt; then
        fail "the multi-part probe did not reach its first line in the web image"
    fi
    if ! grep -q "^PROBE-END" /tmp/web_multipart_leg.txt; then
        fail "the multi-part probe stopped before its last line; its output is above"
    fi
    if [ "$rc" -ne 0 ]; then
        fail "the multi-part probe failed in the web image (exit $rc)"
    fi
    log "web: multi-part refusals OK"
}

# The photonic-crystal form leg: the form's own plan must RUN in the web image.
#
# This is the leg that was missing, and its absence shipped a 500. The image
# built, pushed, deployed, and died on the FIRST pc submission:
#
#     /opt/app/pc.py:37 ModuleNotFoundError: No module named 'scipy'
#
# web/pc_form.py imported the SOLVER to get the unit cell's node count, and the
# solver imports scipy -- which the leg above asserts this image does not have.
# Every test was green because the checkout installs scipy for the solver, so
# the form's import resolved here and nowhere else. Same shape as the multi-part
# leg above, one layer up: the venv is a more capable environment than the
# container, and the difference is exactly where the defect lived.
#
# run PC.plan_request on a REAL small form, in the image, and report the numbers
# it decides on -- not an import probe. `import pc_form` alone would have passed
# on the broken image because pc_form itself imports fine; the import of `pc`
# happens inside plan_request.
web_pc_plan() {
    local tag="$1" out rc
    log "the pc form's plan inside web"

    set +e
    docker run --rm -i --network none --cpus=2 --memory=2g \
        --entrypoint python "$tag" - \
        > /tmp/web_pc_plan_leg.txt 2>&1 <<'PROBE'
import importlib.util
import sys

import pc_form

failures = []


def check(name, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + name
          + (("  " + detail) if detail else ""), flush=True)
    if not ok:
        failures.append(name)


print("PROBE-START", flush=True)

# The form the route actually hands in: every value a string, as request.form
# gives it. div 8 is deliberately tiny -- this leg asks whether the form RUNS
# here, not how big a cell the box can take.
form = {
    "lattice": "square",
    "r_over_a": "0.2",
    "n_rod": "3.5",
    "n_bg": "1.0",
    "n_bands": "6",
    "divisions": "8",
    "polarisation": "TM",
}
try:
    planned = pc_form.plan_request(form)
except Exception as exc:  # noqa: BLE001 - the failure IS the result here
    check("PC.plan_request runs in the web image", False,
          "%s: %s" % (type(exc).__name__, exc))
else:
    check("PC.plan_request runs in the web image", True)
    check("and it derives a node count", planned["nodes"] > 0,
          "nodes=%s" % planned["nodes"])
    check("and it derives the k-point count", planned["k_points"] == 30,
          "k_points=%s" % planned["k_points"])
    check("and it carries no triangle count (the web side does not triangulate)",
          "triangles" not in planned, str(sorted(planned)))

# The premise of the fix, asserted rather than commented: the web image has no
# scipy and no solver module, which is WHY the form must reach only the numpy
# geometry. If a future image gains scipy this leg stops proving anything, and
# that should be visible here rather than silent.
check("the web image still has no scipy",
      importlib.util.find_spec("scipy") is None)
check("the solver module pc is not in the image at all",
      importlib.util.find_spec("pc") is None)

print("PROBE-END", flush=True)
if failures:
    print("PC PLAN LEG FAILED: %d case(s): %s" % (len(failures), failures),
          flush=True)
sys.exit(1 if failures else 0)
PROBE
    rc=$?
    set -e

    cat /tmp/web_pc_plan_leg.txt >&2
    if ! grep -q "^PROBE-START" /tmp/web_pc_plan_leg.txt; then
        fail "the pc form probe did not reach its first line in the web image"
    fi
    if ! grep -q "^PROBE-END" /tmp/web_pc_plan_leg.txt; then
        fail "the pc form probe stopped before its last line; its output is above"
    fi
    if [ "$rc" -ne 0 ]; then
        fail "the pc form probe failed in the web image (exit $rc)"
    fi
    log "web: the pc form's plan OK"
}

solver() {
    build_and_import "solver" "dockerfiles/maxwell_solver/Dockerfile" \
        "$TAG_BASE-solver:local" \
        'import mse_maxwell_solver as S; S.max_tets; import mse_common, mse_meshing; print("solver import ok")'
    solver_geometry "$TAG_BASE-solver:local"
}

# The geometry leg: the dependency-free primitives must WORK in the image, not
# merely import. `mse_geometry` and `mse_parts` were written against numpy
# precisely because `rtree`, `manifold3d` and `embree` are absent here and
# `Trimesh.contains` / `intersection` / `ray.contains_points` all RAISE in this
# container -- while all three work fine in the dev venv. So a leg that runs in
# the venv proves nothing about production; this is the same crashloop shape
# `tests/test_web_imports.py` exists for, one layer down.
#
# Every case carries a control, and the exit code is what decides: the probe
# exits non-zero on a wrong answer rather than printing one. The probe is passed
# with `python -c` rather than as a file ON PURPOSE: `python /path/to/script.py`
# puts the SCRIPT'S directory on sys.path, not the working directory, so a probe
# run as a file in /tmp cannot import /opt/app at all. With `-c` the first
# sys.path entry is the empty string, which resolves to the cwd, which is /opt/app.
#
# PROVE IT RAN. A probe that dies early in this image writes to the container's
# stdout, and that stream is not propagated to the host for an attached `docker
# run` -- so a crashed probe and a passing probe both print nothing here, and the
# exit code is the only honest signal. The probe therefore writes its transcript
# to a file and this function reads it back out with `docker cp`, which does
# propagate, and it asserts a first and a last line so a truncated run cannot
# read as a clean one.
solver_geometry() {
    local tag="$1" out rc
    log "geometry primitives inside solver"

    # `python -` reads the script from STDIN, so the probe needs no quoting at
    # all -- and it must not, because this probe contains double quotes and a
    # nested `sh -c` string that a shell-quoted form mangles silently.
    #
    # The probe cannot print its result to the terminal: stdout from an attached
    # `docker run` against this image does not reach the host here (measured: the
    # first version of this leg printed nothing at all, passing and failing
    # identically). So it writes a transcript to a file and this function copies
    # that file out, which DOES propagate -- and then asserts the transcript has
    # both its first and its last line, because a probe that died early writes no
    # file and would otherwise read exactly like a clean pass.
    # `-i` IS REQUIRED and is not decoration. Without it docker attaches no
    # stdin, `python -` reads an EMPTY script and exits 0 having done nothing:
    # measured, the whole probe ran as a no-op and the leg then failed on its
    # missing first line, which reads exactly like a broken probe and was not.
    set +e
    docker run --rm -i --network none --entrypoint python "$tag" - \
        > /tmp/geometry_leg.txt 2>&1 <<'PROBE'
import sys

import numpy as np
import trimesh

import mse_geometry as GEO
import mse_parts as PARTS

failures = []


def check(name, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + name
          + (("  " + detail) if detail else ""), flush=True)
    if not ok:
        failures.append(name)


print("PROBE-START", flush=True)

# The reason this leg exists: prove the absence is real in THIS container, so a
# future image that quietly gains rtree is a visible change and not a silent
# switch to a code path the tests never exercise.
import importlib

havemods = [m for m in ("rtree", "manifold3d", "embreex", "pyembree", "shapely")
            if importlib.util.find_spec(m)]
print("geometry backends present in the image: "
      + (", ".join(havemods) or "none"), flush=True)
for m in ("rtree", "manifold3d", "embreex"):
    check("the image really lacks " + m, m not in havemods)

# --- inside / on / outside on a polyhedron -------------------------------
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
# the inradius, measured -- comparing against a point at |p| < 1 would be
# testing the IDEAL sphere while the mesh is a polyhedron with a smaller
# inscribed radius, which is the reference error an earlier session made.
face_dist = np.abs(np.einsum(
    "ij,ij->i", sphere.face_normals,
    sphere.triangles[:, 0] - sphere.centroid.reshape(1, 3)))
inradius = float(face_dist.min())
top = float(sphere.vertices[:, 2].max())

inside, on = GEO.point_in_mesh(sphere, np.array([
    [0.0, 0.0, 0.0],        # centre: inside
    [0.0, 0.0, top],        # a vertex: ON the surface
    [0.0, 0.0, 2.0],        # outside
]))
check("centre is inside", bool(inside[0]))
check("an exact vertex reads as ON, not inside", bool(on[1]))
check("a far point is outside", not inside[2] and not on[2])
check("a point inside the inradius is inside",
      bool(GEO.point_in_mesh(sphere, np.array([[0.0, 0.0, 0.5 * inradius]]))[0][0]),
      "inradius=%.6f" % inradius)
check("the inradius is below the circumradius", inradius < 1.0)
# the control that makes the ON case mean something: the same point, nudged
# 2 percent of the inradius inward, must read as strictly inside
in2, on2 = GEO.point_in_mesh(
    sphere, np.array([[0.0, 0.0, top - 0.02 * inradius]]))
check("and that vertex nudged 2% inward reads as inside, not on",
      bool(in2[0]) and not bool(on2[0]))

# --- the torus seed: the centroid is in the HOLE --------------------------
torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
t_in, _ = GEO.point_in_mesh(torus, np.asarray(torus.centroid).reshape(1, 3))
t_tube, _ = GEO.point_in_mesh(torus, np.array([[1.0, 0.0, 0.0]]))
check("the torus centroid is OUTSIDE (it is in the hole)", not bool(t_in[0]))
check("a point on the tube is inside", bool(t_tube[0]))
seed = np.asarray(PARTS.find_region_seed(torus), dtype=float).reshape(1, 3)
s_in, s_on = GEO.point_in_mesh(torus, seed)
check("the found torus seed is strictly inside",
      bool(s_in[0]) and not bool(s_on[0]),
      "seed=" + str(seed.ravel().round(4).tolist()))

# --- the wall/core pair, and the seed that must dodge the core ------------
wall = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
core = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
c_in, _ = GEO.point_in_mesh(core, np.asarray(wall.centroid).reshape(1, 3))
check("the wall centroid really is inside the core (the fixture is honest)",
      bool(c_in[0]))
check("part_contains(wall, core)", bool(GEO.part_contains(wall, core)))
check("not part_contains(core, wall)", not bool(GEO.part_contains(core, wall)))
w_seed = np.asarray(PARTS.find_region_seed(wall, children=[core]),
                    dtype=float).reshape(1, 3)
w_in, w_on = GEO.point_in_mesh(wall, w_seed)
wc_in, wc_on = GEO.point_in_mesh(core, w_seed)
check("the wall seed is inside the wall and outside the core",
      bool(w_in[0]) and not bool(w_on[0]) and not bool(wc_in[0])
      and not bool(wc_on[0]),
      "seed=" + str(w_seed.ravel().round(4).tolist()))

# --- the split: names and parts must line up, in file order ---------------
text = trimesh.Scene(dict(wall=wall, core=core)).export(file_type="obj")
names, parts = PARTS.split_obj_parts(text)
check("split_obj_parts returns the two names in file order",
      names == ["wall", "core"], "names=" + str(names))
check("and two watertight parts",
      len(parts) == 2 and all(bool(p.is_watertight) for p in parts),
      "volumes=" + str([round(float(p.volume), 4) for p in parts]))

# --- touching: the half of the interface check TetGen does NOT catch ------
# The three meshes get three DISTINCT names, and the translation is applied to a
# box built at the origin rather than to one of the loop's own variables. An
# earlier version of this leg reused the name for the separated box, so the last
# -- separated -- case silently tested the touching pair, and the leg reported a
# FAIL that looked like a geometry bug and was a fixture alias. Distinct names,
# and each box built where it is used.
far_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
far_box.apply_translation([2.0, 0.0, 0.0])     # a clear 1.5 gap
touch_a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
touch_b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
touch_b.apply_translation([1.0, 0.0, 0.0])     # face-to-face, exactly touching
# EACH DIRECTION IS ASSERTED SEPARATELY, and that is the whole point of these
# two lines. Summing the two index arrays looked equivalent and is not: a
# mutation that forces EITHER direction empty still leaves the sum non-zero, so
# a combined count is a check that cannot fail against exactly the bug it exists
# to catch. Measured with `b_on_a` forced empty: the sum stayed 4 and the leg
# stayed green. One assertion per direction, so killing either one reddens.
touch_a_on_b, touch_b_on_a = GEO.touching_vertices(touch_a, touch_b)
check("face-to-face: a vertex of the first lies on the second",
      len(touch_a_on_b) > 0, str(len(touch_a_on_b)) + " vertices")
check("face-to-face: and a vertex of the second lies on the first",
      len(touch_b_on_a) > 0, str(len(touch_b_on_a)) + " vertices")
sep_a_on_b, sep_b_on_a = GEO.touching_vertices(touch_a, far_box)
check("separated: no vertex of the first lies on the second",
      len(sep_a_on_b) == 0, str(len(sep_a_on_b)) + " vertices")
check("separated: no vertex of the second lies on the first",
      len(sep_b_on_a) == 0, str(len(sep_b_on_a)) + " vertices")

print("PROBE-END", flush=True)
if failures:
    print("GEOMETRY LEG FAILED: %d case(s): %s" % (len(failures), failures),
          flush=True)
sys.exit(1 if failures else 0)
PROBE
    rc=$?
    set -e

    cat /tmp/geometry_leg.txt >&2
    if ! grep -q "^PROBE-START" /tmp/geometry_leg.txt; then
        fail "the geometry probe did not reach its first line in the solver image"
    fi
    if ! grep -q "^PROBE-END" /tmp/geometry_leg.txt; then
        fail "the geometry probe stopped before its last line; its output is above"
    fi
    if [ "$rc" -ne 0 ]; then
        fail "the geometry probe failed in the solver image (exit $rc)"
    fi
    log "solver: geometry OK"
}

case "$WHICH" in
    web)    web ;;
    solver) solver ;;
    both)   web; solver ;;
    *)      fail "unknown target $WHICH (expected web, solver or both)" ;;
esac

log "image smoke passed: $WHICH"
