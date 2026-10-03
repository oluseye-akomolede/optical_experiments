"""Write the inputs for dockerfiles/rendercheck/tdcheck.mjs into DIR.

A real tiny time-domain solve (the tests' own fixture) writes its result
files; the Flask app renders the done job page and status, and serves every
result file through its real routes. web/static is copied beside them.
The solve is a 120-frame window, so the three-digit frame names and a caption
about a window (not the whole run) are exercised.
The scene's boundary is TDCHECK_BOUNDARY (default "metal"). It is a knob
because the two boundaries put the manifest's sampled positions in *different*
relations to the object -- metal samples the object's own box, open samples
OPEN_PAD_CELLS * dx of padding on every side of it -- so a check of where the
outline is can only be trusted if it has been run against both. Metal is the
default because the repo's own job fixture is metal (tests/test_td_solve._job).
Run from the repo root, memory-capped:
  (ulimit -v 4194304; .venv-mse/bin/python scripts/tdcheck_fixture.py DIR)
  TDCHECK_BOUNDARY=open (ulimit -v 4194304; .venv-mse/bin/python ... DIR)
"""
import io, json, os, pathlib, shutil, sys
import numpy as np
import trimesh
sys.path[:0] = ["tests"]
import test_td_solve as T
import test_web_routes as W

# A 120-frame, three-digit window, so the name width is exercised: the old
# two-digit rule and the old page both read it wrong. What tdcheck.mjs checks
# is derived from whatever manifest it is served, so the window is asserted
# here, where it is asked for, rather than in the check.
WINDOW = {"movie_start_s": 0.2e-10, "movie_end_s": 2e-10, "frame_count": 120}


def _fixture_obj() -> bytes:
    """The solve's upload: the tests' own 4x3x2 mm block, plus a part beside it
    at NEGATIVE x, so the parts' combined minimum is not the upload's origin.

    That is the point of the extra part, and it is the row's named trap. A job
    whose object starts at its own origin cannot tell an outline drawn in the
    object's own coordinates from one drawn in the grid's: both are the same
    numbers, so the check that compares them is an identity and cannot fail
    ("a mesh-frame overlay looked right on a centred cube"). With the partner,
    the manifest's positions run from -2 mm and the page has to draw the
    outline there too; an outline in the grid's frame (and a unit or scale slip)
    lands outside the one-cell tolerance and FAILs naming the axis.

    The original block stays exactly where it was, so the probes and the source
    the job asks for (1..3 mm) are still inside it.
    """
    block = trimesh.creation.box(extents=(4, 3, 2))
    block.apply_translation((2, 1.5, 1))          # 0..4 x 0..3 x 0..2 mm
    pin = trimesh.creation.box(extents=(1, 1, 1))
    pin.apply_translation((-1.5, 1.0, 0.5))       # -2..-1 mm: moves the minimum
    return trimesh.Scene({"block": block, "pin": pin}).export(
        file_type="obj").encode()


class _FixtureStore(T._Store):
    """The tests' store, holding the two-part upload above instead of a box."""

    def __init__(self):
        super().__init__()
        self.obj = _fixture_obj()


out_dir = pathlib.Path(sys.argv[1])
solve_store = _FixtureStore()
# The request's own positions are in the layout's frame, whose origin is the
# parts' minimum corner -- ``_locate(nodes, request["position"])`` with no
# conversion, and the form's typed value is used as it stands -- not in the
# upload's coordinates. With the pin at negative x the block sits at 2..6 mm
# there, so the source and probes are the single-part fixture's own points
# 2 mm further along: (1, 1.5, 1) mm and (3, 1.5, 1) / (2, 2, 1) mm move to
# 3, 5 and 4 mm. Left where they were, the source lands on the pin's corner and
# the layout refuses it -- "the source position ... sits inside the metal
# boundary's conductor" -- which is how this fixture's frame was measured.
#
# Two parts need a materials table, and the solver says so rather than guessing
# ("2 parts need a materials table; got none"). Both are the same glass block
# the single-part fixture was, so the scene's physics is unchanged and only the
# domain's far corner moved. The solver fills the rest of the domain with
# vacuum, which is what a cavity is.
PLACED = {"position": [3e-3, 1.5e-3, 1e-3],
          "probes": [[5e-3, 1.5e-3, 1e-3, "z"], [4e-3, 2e-3, 1e-3, "z"]]}
#: The scene's boundary, for the two mirrors the check is run against. The
#: positions above are in the parts' frame either way (the frame's origin is the
#: parts' minimum corner on both), so the same scene solves under both.
BOUNDARY = os.environ.get("TDCHECK_BOUNDARY", "metal")
_res, out = T._run_with_files(solve_store, job_changes={"materials": {"block": 1.5, "pin": 1.5}},
                              **WINDOW, **PLACED, boundary=BOUNDARY)
app, store, db, _l = W.make_app(out_dir / "_store")
# The row's own units and scale must be the ones the solve above was given
# (test_td_solve._job: units "mm", scale 1.0), or the outline the page fetches
# would be a different size from the field in this same mirror. And the stored
# upload must be the very bytes that solve read, so the outline is the object
# the field was computed on.
job_id = W._done_td_job(db, units="mm")
store.write(W.STORE.job_key(job_id, "input.obj"), solve_store.obj)
for name, data in out.files.items():
    store.write(W.STORE.job_key(job_id, name), data)
c = app.test_client()
(out_dir / "job.html").write_bytes(c.get(f"/jobs/{job_id}").data)
(out_dir / "status.json").write_bytes(c.get(f"/jobs/{job_id}/status").data)
files = out_dir / "files"; files.mkdir(exist_ok=True)
man0 = json.loads(out.files["spectrum.json"])
frames = man0["frames"] or {"count": 0, "digits": 3}
assert (frames["count"], frames["digits"]) == (WINDOW["frame_count"], 3), frames
# the width comes from the manifest, so this fixture cannot guess it wrong
frame_names = [f"frame_{n:0{frames['digits']}d}.bin"
               for n in range(1, frames["count"] + 1)]
for name in ("spectrum.json", "trace.bin", *frame_names):
    r = c.get(f"/jobs/{job_id}/files/{name}"); assert r.status_code == 200, name
    (files / name).write_bytes(r.data)
# The object outline, through its real route, beside the page that fetches it.
# A fixture that cannot produce its own outline would make tdcheck's outline
# check fail for the fixture's reason and look like a page defect.
r = c.get(f"/jobs/{job_id}/outline.glb")
assert r.status_code == 200, (r.status_code, r.data[:200])
(out_dir / "outline.glb").write_bytes(r.data)
shutil.copytree("web/static", out_dir / "static", dirs_exist_ok=True)
man = json.loads((files / "spectrum.json").read_text())
scene = trimesh.load(io.BytesIO(r.data), file_type="glb")
box = np.asarray(scene.bounds, dtype=float)
print("job", job_id, "boundary", BOUNDARY, "probes", len(man["probes"]), "steps", man["steps"], "caveats", len(man["caveats"]), "frames", man["frames"]["count"], man["frames"]["shape"], "digits", man["frames"]["digits"], "first", frame_names[0], "last", frame_names[-1], "outline", len(scene.geometry), "parts", box[0].tolist(), box[1].tolist(), "manifest_positions_x", [min(man["frames"]["positions_m"][0]), max(man["frames"]["positions_m"][0])])
