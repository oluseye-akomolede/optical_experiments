"""Write the inputs for dockerfiles/rendercheck/tdcheck.mjs into DIR.

A real tiny time-domain solve (the tests' own fixture) writes its result
files; the Flask app renders the done job page and status, and serves every
result file through its real routes. web/static is copied beside them.
The solve is the 120-frame window tdcheck checks: three-digit frame names and
a caption about a window, not the whole run.
Run from the repo root, memory-capped:
  (ulimit -v 4194304; .venv-mse/bin/python scripts/tdcheck_fixture.py DIR)
"""
import json, pathlib, shutil, sys
sys.path[:0] = ["tests"]
import test_td_solve as T
import test_web_routes as W

# The window and count tdcheck.mjs asserts (a 120-frame, three-digit movie, so
# the old two-digit name rule and the old page both read it wrong).
WINDOW = {"movie_start_s": 0.2e-10, "movie_end_s": 2e-10, "frame_count": 120}

out_dir = pathlib.Path(sys.argv[1])
_res, out = T._run_with_files(**WINDOW)
app, store, db, _l = W.make_app(out_dir / "_store")
job_id = W._done_td_job(db)
for name, data in out.files.items():
    store.write(W.STORE.job_key(job_id, name), data)
c = app.test_client()
(out_dir / "job.html").write_bytes(c.get(f"/jobs/{job_id}").data)
(out_dir / "status.json").write_bytes(c.get(f"/jobs/{job_id}/status").data)
files = out_dir / "files"; files.mkdir(exist_ok=True)
man0 = json.loads(out.files["spectrum.json"])
frames = man0["frames"] or {"count": 0, "digits": 3}
# the width comes from the manifest, so this fixture cannot guess it wrong
frame_names = [f"frame_{n:0{frames['digits']}d}.bin"
               for n in range(1, frames["count"] + 1)]
for name in ("spectrum.json", "trace.bin", *frame_names):
    r = c.get(f"/jobs/{job_id}/files/{name}"); assert r.status_code == 200, name
    (files / name).write_bytes(r.data)
shutil.copytree("web/static", out_dir / "static", dirs_exist_ok=True)
man = json.loads((files / "spectrum.json").read_text())
print("job", job_id, "probes", len(man["probes"]), "steps", man["steps"], "caveats", len(man["caveats"]), "frames", man["frames"]["count"], man["frames"]["shape"], "digits", man["frames"]["digits"], "first", frame_names[0], "last", frame_names[-1])
