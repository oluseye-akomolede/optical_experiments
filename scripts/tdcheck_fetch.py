"""Mirror one real job's page and result files into DIR for tdcheck.mjs.

The openrig VM cannot resolve the tailnet, so run this where the app answers
-- the Jetson -- and copy the directory back:

  ssh jetson "python3 ~/mse-bench/tdcheck_fetch.py \\
      https://mse.taildfdd3b.ts.net aa6f083b ~/mse-bench/tdmirror"
  ssh jetson "tar -C ~/mse-bench -cf - tdmirror" > /tmp/tdmirror.tar
  tar -C /tmp -xf /tmp/tdmirror.tar
  scripts/run_tdcheck.sh --dir /tmp/tdmirror

A JOB may be a full job id or any prefix of one that matches exactly one job
on the index page; an ambiguous or unknown prefix is an error, never a guess
about which run was mirrored.

What it writes, exactly what dockerfiles/rendercheck/tdcheck.mjs reads from
TD_DIR:

  job.html      /jobs/<id>, the real page
  status.json   /jobs/<id>/status, the payload the page runs on
  outline.glb   the object outline, at the URL status.json itself names -- so
                the mirror holds whatever that page will actually fetch
  files/        every result file the manifest names: spectrum.json, trace.bin
                and the frames (names taken from the manifest's own count and
                digit width, not from a constant here)
  static/       the /static assets that page loads, following its module
                imports, so the page is exercised with the code it shipped

Stdlib only: the Jetson's python3 has no requests. Every fetch that does not
answer 200 aborts the run with the URL and the status -- a mirror with a hole
in it must not look like a job that has no such file.
"""
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 120
# Only path-like references count. A JS string can look like one of these --
# GLTFLoader.js holds ``"srgb-linear"`` and three.module.min.js a concatenated
# ``"+t.width+"`` -- and a colour-space name is not a request.
PATH = r"""["']((?:\.{1,2})?/[^"']+)["']"""
LINK = re.compile(r"(?:src|href)\s*=\s*" + PATH)
CSS_URL = re.compile(r"""url\(\s*["']?((?:\.{1,2})?/[^"')]+)["']?\s*\)""")
IMPORT = re.compile(r"(?:from|import)\s*\(?\s*" + PATH)


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"tdcheck_fetch: {url} -> HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"tdcheck_fetch: {url} -> {e.reason}")


FULL_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def pick(job, ids):
    """The one job id ``job`` names out of ``ids``: a full id, or a prefix that
    matches exactly one of them. Zero matches and several matches are both
    errors: the job that gets mirrored must not be a guess. A full id is taken
    as given -- the index lists one page, and an older job is still fetchable.
    """
    if FULL_ID.fullmatch(job):
        return job
    hits = sorted({i for i in ids if i.startswith(job)})
    if len(hits) != 1:
        sys.exit(f"tdcheck_fetch: {job!r} matches {len(hits)} jobs: {hits}")
    return hits[0]


def resolve(base, job):
    """The full job id for ``job``, from the index page's own list of them."""
    return pick(job, re.findall(r"/jobs/([0-9a-zA-Z-]+)",
                                get(base + "/").decode("utf8", "replace")))


def frame_names(man):
    """The frame file names, from the manifest's own count and digit width."""
    frames = man.get("frames")
    if not frames:
        return []
    return [f"frame_{n:0{frames['digits']}d}.bin"
            for n in range(1, frames["count"] + 1)]


def static_assets(page_url, html):
    """Every /static asset reachable from ``html``, following module imports."""
    seen, queue = {}, []          # url -> bytes
    def refs(url, text, pattern):
        for m in pattern.finditer(text):
            ref = (m.group(1) or '').strip()
            if not ref or ref.startswith(("data:", "#", "//")):
                continue
            u = urllib.parse.urljoin(url, ref)
            if urllib.parse.urlparse(u).path.startswith("/static/") and u not in seen:
                queue.append(u)
    # A pattern per kind of file: three.module.min.js contains the GLSL name
    # ``url(srgb-linear)``, which is not a request, so only real CSS is scanned
    # for url(...).
    SCAN = {".js": IMPORT, ".mjs": IMPORT, ".css": CSS_URL}
    refs(page_url, html.decode("utf8", "replace"), LINK)
    while queue:
        u = queue.pop()
        body = get(u)
        seen[u] = body
        scan = SCAN.get(pathlib.Path(urllib.parse.urlparse(u).path).suffix)
        if scan:
            refs(u, body.decode("utf8", "replace"), scan)
    return seen


def main(base, job, out_dir):
    base = base.rstrip("/")
    job_id = resolve(base, job)
    out = pathlib.Path(out_dir)
    (out / "files").mkdir(parents=True, exist_ok=True)
    job_url = f"{base}/jobs/{job_id}"
    page = get(job_url)
    (out / "job.html").write_bytes(page)
    status = get(f"{job_url}/status")
    (out / "status.json").write_bytes(status)
    # The outline's URL comes from the payload the page itself runs on, so the
    # mirror holds the file that page will ask for. An older app names none:
    # say so here, where the app's age is the fact, rather than mirroring a
    # page whose outline check is then guaranteed to fail.
    outline_url = json.loads(status).get("outline_url")
    if not outline_url:
        sys.exit(f"tdcheck_fetch: {job_url}/status names no outline_url: the "
                 "deployed app predates the object outline")
    outline = get(urllib.parse.urljoin(job_url, outline_url))
    (out / "outline.glb").write_bytes(outline)
    spectrum = get(f"{job_url}/files/spectrum.json")
    (out / "files/spectrum.json").write_bytes(spectrum)
    man = json.loads(spectrum)
    names = ["trace.bin", *frame_names(man)]
    total = 0
    for name in names:
        body = get(f"{job_url}/files/{name}")
        (out / "files" / name).write_bytes(body)
        total += len(body)
    assets = static_assets(job_url, page)
    for u, body in assets.items():
        f = out / urllib.parse.urlparse(u).path.lstrip("/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(body)
    print(f"job {job_id} probes {len(man['probes'])} steps {man['steps']} "
          f"dt_s {man['dt_s']!r} band_hz {man['band_hz']} "
          f"frames {man['frames'] and (man['frames']['count'], man['frames']['digits'])} "
          f"files {len(names)} ({total / 1e6:.1f} MB) outline {len(outline) / 1e6:.3f} MB "
          f"static {len(assets)} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
