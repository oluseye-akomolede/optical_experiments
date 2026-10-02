"""scripts/tdcheck_fetch.py -- which job is mirrored, and what it downloads.

The script exists so the time-domain render check can run against a real
production job instead of a fixture. That gives it three jobs, and each is a
place where a wrong answer looks like a right one:

* **Which job.** A job may be named by a prefix. A prefix that matches two runs
  must stop, not pick one: a mirror of the wrong job is a check of the wrong
  page, and it would pass.
* **What is fetched.** The frame file names come from the manifest's own count
  and digit width, so an old two-digit run and a 120-frame run both name files
  that exist.
* **What is a request.** The static bundle is crawled for the page's own code.
  A JavaScript string that merely *looks* like a path is not a request --
  ``vendor/three.module.min.js`` holds ``"+t.width+"`` and a GLSL colour-space
  name -- and chasing one into a 404 makes a complete mirror look like a page
  that failed. This happened twice while writing it, which is why it is pinned
  here.

Tier 1: no network, no browser. The fetch is faked through the module's own
``get``, and everything else is pure.

What these tests cannot show is that the page then checks correctly; that is
what ``scripts/run_tdcheck.sh`` and the mirror itself are for.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    """The script as a module. It has a __main__ guard, so this runs nothing."""
    spec = importlib.util.spec_from_file_location(
        "tdcheck_fetch", ROOT / "scripts" / "tdcheck_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Two runs sharing a prefix, and one that shares nothing.
IDS = ["aa6f083b-633c-437d-bbdc-0e0a0f5d76bc",
       "aa6f083b-1111-4444-8888-cccccccccccc",
       "01234567-89ab-cdef-0123-456789abcdef"]


def test_pick_takes_a_full_id_or_a_unique_prefix():
    mod = load()
    assert mod.pick(IDS[0], IDS) == IDS[0]
    assert mod.pick(IDS[0], []) == IDS[0]          # an older job, off the index page
    assert mod.pick("0123", IDS) == IDS[2]
    assert mod.pick("aa6f083b-1111", IDS) == IDS[1]


def test_pick_refuses_a_prefix_that_names_no_one_or_two():
    mod = load()
    with pytest.raises(SystemExit) as e:
        mod.pick("aa6f083b", IDS)                  # both of the first two
    assert "matches 2 jobs" in str(e.value)
    with pytest.raises(SystemExit) as e:
        mod.pick("deadbeef", IDS)
    assert "matches 0 jobs" in str(e.value)


def test_frame_names_follow_the_manifests_own_count_and_width():
    mod = load()
    three = {"frames": {"count": 120, "digits": 3}}
    assert mod.frame_names(three)[0] == "frame_001.bin"
    assert mod.frame_names(three)[1] == "frame_002.bin"
    assert mod.frame_names(three)[-1] == "frame_120.bin"
    assert len(mod.frame_names(three)) == 120
    # An older run's two-digit names, and the manifest that says so.
    two = {"frames": {"count": 60, "digits": 2}}
    assert mod.frame_names(two)[0] == "frame_01.bin"
    assert mod.frame_names(two)[-1] == "frame_60.bin"
    # A job with no movie has no frames to fetch, and that is not an error.
    assert mod.frame_names({"frames": None}) == []
    assert mod.frame_names({}) == []


def test_static_assets_follow_the_page_and_only_path_like_strings(monkeypatch):
    mod = load()
    pages = {
        "http://mse/static/viewer.js":
            b'import { si } from "./tdplot.js";\nconst c = "srgb-linear";\n',
        "http://mse/static/tdplot.js": b"export function si() {}\n",
        "http://mse/static/style.css": b'body { background: url("/static/bg.png"); }\n',
        "http://mse/static/bg.png": b"\x89PNG",
    }
    asked = []

    def fake_get(url):
        asked.append(url)
        return pages[url]                          # a URL not in pages is a KeyError: a fetch nothing asked for

    monkeypatch.setattr(mod, "get", fake_get)
    html = (b'<script type="module" src="/static/viewer.js"></script>'
            b'<link rel="stylesheet" href="/static/style.css">')
    got = mod.static_assets("http://mse/jobs/x", html)
    assert sorted(got) == sorted(pages)
    # The colour-space name is a string in the JS, not a path, and no request
    # was made for it: that is the whole rule this pins.
    assert "http://mse/static/srgb-linear" not in asked


def test_the_patterns_read_a_reference_and_ignore_a_lookalike():
    """The four patterns directly: a path is fetched, a bare string is not."""
    mod = load()
    assert mod.LINK.findall('<script src="/static/viewer.js"></script>') == ["/static/viewer.js"]
    assert mod.IMPORT.findall('import { si } from "./tdplot.js";') == ["./tdplot.js"]
    assert mod.IMPORT.findall("await import('/static/movie.js')") == ["/static/movie.js"]
    assert mod.CSS_URL.findall('background: url(/static/bg.png)') == ["/static/bg.png"]
    for lookalike in ('const c = "srgb-linear";',       # a GLSL colour space
                      '"x=" + "+t.width+"',             # a concatenated size
                      "import { Color } from 'three';"):  # a package, not a path here
        assert mod.IMPORT.findall(lookalike) == []
        assert mod.LINK.findall(lookalike) == []
