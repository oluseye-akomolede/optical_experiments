"""Write the inputs for dockerfiles/rendercheck/tdcheck.mjs into DIR.

A real tiny time-domain solve (the tests' own fixture) writes its result
files; the Flask app renders the done job page and status, and serves every
result file through its real routes. web/static is copied beside them.
Run from the repo root, memory-capped:
  (ulimit -v 4194304; .venv-mse/bin/python scripts/tdcheck_fixture.py DIR)
"""
import json, pathlib, shutil, sys
sys.path[:0] = ["tests"]
import test_td_solve as T
import test_web_routes as W

out_dir = pathlib.Path(sys.argv[1])
_res, out = T._run_with_files()
app, store, db, _l = W.make_app(out_dir / "_store")
job_id = W._done_td_job(db)
for name, data in out.files.items():
    store.write(W.STORE.job_key(job_id, name), data)
c = app.test_client()
(out_dir / "job.html").write_bytes(c.get(f"/jobs/{job_id}").data)
(out_dir / "status.json").write_bytes(c.get(f"/jobs/{job_id}/status").data)
files = out_dir / "files"; files.mkdir(exist_ok=True)
man0 = json.loads(out.files["spectrum.json"])
frame_names = [f"frame_{n:02d}.bin" for n in range(1, (man0["frames"] or {"count": 0})["count"] + 1)]
for name in ("spectrum.json", "trace.bin", *frame_names):
    r = c.get(f"/jobs/{job_id}/files/{name}"); assert r.status_code == 200, name
    (files / name).write_bytes(r.data)
shutil.copytree("web/static", out_dir / "static", dirs_exist_ok=True)
man = json.loads((files / "spectrum.json").read_text())
print("job", job_id, "probes", len(man["probes"]), "steps", man["steps"], "caveats", len(man["caveats"]), "frames", man["frames"]["count"], man["frames"]["shape"])
