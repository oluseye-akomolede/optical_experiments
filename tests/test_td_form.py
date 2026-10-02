"""The time-domain form, end to end through the upload route (tasks 5.1, 5.1a).

Every malformed input on the 5.1a list is POSTed to ``/`` with ``mode=td`` and
must come back as the route's refusal (a 302 with a flash naming the problem,
and no row), never a 500. A happy-path row is then read back by the WORKER's
own ``request_from`` and laid out by its own ``layout``, so the form cannot
store a request the worker refuses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))

trimesh = pytest.importorskip("trimesh")

import db as DB  # noqa: E402
import td_form as TDF  # noqa: E402
from test_web_routes import io_bytes, make_app  # noqa: E402


def _close(expected):
    """Relative only: pytest.approx's default 1e-12 absolute tolerance would
    pass ANY femtosecond or nanometre value."""
    return pytest.approx(expected, rel=1e-9, abs=0)


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    return box


def _obj(*named):
    """A multi-part .obj: one ``o <name>`` group per (name, mesh)."""
    lines, base = [], 0
    for name, mesh in named:
        lines.append(f"o {name}")
        lines += [f"v {float(x)!r} {float(y)!r} {float(z)!r}"
                  for x, y, z in mesh.vertices]
        lines += ["f " + " ".join(str(base + 1 + int(i)) for i in face)
                  for face in mesh.faces]
        base += len(mesh.vertices)
    return ("\n".join(lines) + "\n").encode()


def _scene():
    """A 1.0 x 0.8 x 0.6 um glass block holding a 0.2 um gold cube."""
    return _obj(("block", _box([0, 0, 0], [1.0, 0.8, 0.6])),
                ("cube", _box([0.6, 0.5, 0.3], [0.8, 0.7, 0.5])))


def _fields(data=None, **over):
    fields = {
        "mode": "td", "units": "um", "scale": "",
        "td_unit": "nm", "td_boundary": "metal", "td_source_kind": "point",
        "td_profile": "pulse", "td_frequency_thz": "450",
        "td_bandwidth_thz": "100", "td_polarisation": "z",
        "td_position": "300 300 300", "td_duration_fs": "20",
        "td_dx": "50", "td_probes": "250 250 250 z\n400 300 200 x",
        "part_index.block": "1.5", "part_index.cube": "Au",
    }
    fields.update(over)
    fields = {k: v for k, v in fields.items() if v is not None}
    fields["meshfile"] = (io_bytes(_scene() if data is None else data),
                          "scene.obj")
    return fields


def _post(tmp_path, data=None, **over):
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    resp = client.post("/", data=_fields(data, **over),
                       content_type="multipart/form-data")
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    return resp, db, store, flashes


def _refused(tmp_path, needle, data=None, **over):
    resp, db, _store, flashes = _post(tmp_path, data, **over)
    assert resp.status_code == 302, (resp.status_code, resp.data[:400])
    assert db.jobs == {}, "a refused td request was queued anyway"
    assert any(needle.lower() in m.lower() for m in flashes), (needle, flashes)
    return flashes


def test_the_form_keys_are_the_workers_request_keys():
    import td_solve

    assert TDF.REQUEST_KEYS == td_solve.REQUEST_KEYS


def test_a_td_form_queues_a_jetson_td_row_the_worker_can_lay_out(tmp_path):
    resp, db, store, flashes = _post(tmp_path)
    assert resp.status_code == 302, (flashes, resp.data[:400])
    (job_id, job), = db.jobs.items()
    assert job["kind"] == DB.TD and job["target"] == DB.JETSON
    assert job["materials"] == {"block": 1.5, "cube": "Au"}
    tds = job["tds"]
    assert tds["frequency_hz"] == _close(450e12)
    assert tds["bandwidth"] == _close(100e12)
    assert tds["dx_m"] == _close(50e-9)
    assert tds["duration_s"] == _close(20e-15)
    assert tds["position"] == _close([300e-9] * 3)
    assert tds["probes"][1][:3] == _close([400e-9, 300e-9, 200e-9])
    assert tds["probes"][1][3] == "x"
    assert tds["direction"] is None and tds["samples"] is None
    assert tds["coatings"] is None

    # The worker's own reader and layout accept the stored row.
    import td_solve

    fdtd, fdtd_layout, mse_parts = TDF._solver()
    request = td_solve.request_from(job)
    names, parts = mse_parts.split_obj_parts(
        store.read(f"jobs/{job_id}/input.obj"))
    lay = fdtd_layout.layout(names, parts, request, length_scale=1e-6,
                             materials=job["materials"], n_index=job["n_index"])
    # The graded grid is not uniform, so check the node, not an index guess.
    for axis, i in enumerate(lay["cell"]):
        nodes = lay["nodes"][axis]
        assert abs(nodes[i] - 300e-9) <= np.max(np.diff(nodes)) / 2 + 1e-15
    assert set(lay["metals"]) == {"Au"}
    assert job["est_tets"] == int(np.prod(lay["shape"]))


def test_a_coated_part_reaches_the_worker_as_a_shell(tmp_path):
    """The page's coating fields for the gold cube (thickness in the form's
    unit, nm) are stored in metres, and the worker's own reader and layout
    paint a conductor shell just outside the cube's face."""
    resp, db, store, flashes = _post(
        tmp_path, **{"part_coating.cube": "pec",
                     "part_coating_thickness.cube": "10"})
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    (job_id, job), = db.jobs.items()
    (part, (material, thickness)), = job["tds"]["coatings"].items()
    assert (part, material) == ("cube", "pec") and thickness == _close(10e-9)

    import td_solve

    _fdtd, fdtd_layout, mse_parts = TDF._solver()
    names, parts = mse_parts.split_obj_parts(
        store.read(f"jobs/{job_id}/input.obj"))
    lay = fdtd_layout.layout(names, parts, td_solve.request_from(job),
                             length_scale=1e-6, materials=job["materials"],
                             n_index=job["n_index"])
    x, y, z = lay["nodes"]
    j, k = int(np.argmin(np.abs(y - 0.6e-6))), int(np.argmin(np.abs(z - 0.4e-6)))
    shell = (x > 0.59e-6 + 1e-15) & (x < 0.6e-6 - 1e-15)
    beyond = (x > 0.5e-6) & (x < 0.58e-6)
    assert shell.any() and np.asarray(lay["pec"])[shell, j, k].all()
    assert not np.asarray(lay["pec"])[beyond, j, k].any()
    assert job["est_tets"] == int(np.prod(lay["shape"]))


@pytest.mark.parametrize("over, needle", [
    ({"part_coating.lid": "Au", "part_coating_thickness.lid": "10"},
     "coating names 'lid', which is not a part of this upload"),
    ({"part_coating.cube": "Au"}, "coating of part 'cube' needs a thickness (nm)"),
    ({"part_coating.cube": "Xx", "part_coating_thickness.cube": "10"},
     "coating material for part 'cube' must be a refractive index"),
    ({"part_coating.cube": "0.5", "part_coating_thickness.cube": "10"},
     "must be 1 or more"),
    ({"part_coating.cube": "Au", "part_coating_thickness.cube": "0"},
     "coating thickness of part 'cube' (nm) must be greater than zero"),
    ({"part_coating.cube": "Au", "part_coating_thickness.cube": "thin"},
     "coating thickness of part 'cube' (nm) must be a number"),
    ({"part_coating_thickness.cube": "10"}, "without a coating material"),
    ({"part_coating.block": "Au", "part_coating_thickness.block": "10"},
     "enclose the cavity"),
])
def test_a_bad_coating_is_refused_naming_the_part(tmp_path, over, needle):
    _refused(tmp_path, needle, **over)


def test_no_coating_posts_nothing_and_queues_none(tmp_path):
    resp, db, _store, flashes = _post(tmp_path, **{"part_coating.cube": ""})
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    assert next(iter(db.jobs.values()))["tds"]["coatings"] is None


def test_the_page_posts_the_coating_fields_the_form_reads(tmp_path):
    app, _store, _db, _launcher = make_app(tmp_path)
    html = app.test_client().get("/").get_data(as_text=True)
    for prefix in (TDF.COATING_PREFIX, TDF.COATING_THICKNESS_PREFIX):
        assert html.count(f"'{prefix}' + name") == 1, prefix


def test_a_plane_wave_in_open_space_is_queued(tmp_path):
    resp, db, _store, flashes = _post(
        tmp_path, td_boundary="open", td_source_kind="plane",
        td_direction="+x", td_position=None, td_dx="100",
        td_probes="500 400 300 z")
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    tds = next(iter(db.jobs.values()))["tds"]
    assert tds["direction"] == "+x" and tds["position"] is None


def test_a_plane_wave_in_a_metal_box_is_refused_naming_the_conflict(tmp_path):
    _refused(tmp_path, "plane wave needs 'open'", td_source_kind="plane",
             td_direction="+x")


# --- the 5.1a list, each malformed input end to end -------------------------
@pytest.mark.parametrize("raw, needle", [
    ("", "material for part 'cube'"),             # a null material
    ("null", "material for part 'cube'"),
    ("[1.5, 2]", "material for part 'cube'"),     # a list material
    ("0.5", "must be 1 or more"),
    ("nan", "must be 1 or more"),
])
def test_a_bad_material_is_refused_by_part(tmp_path, raw, needle):
    _refused(tmp_path, needle, **{"part_index.cube": raw})


def test_a_material_for_a_part_not_in_the_upload_is_refused(tmp_path):
    _refused(tmp_path, "not in the upload: 'lid'", **{"part_index.lid": "2"})


def test_a_part_with_no_material_is_refused(tmp_path):
    _refused(tmp_path, "no material for 'cube'", **{"part_index.cube": None})


def _plates(gap=0.1):
    """Two gold plates with a film between them, NOT nested: the cavity
    parser refuses this (no part encloses the others); a td run takes it."""
    t, f = 0.2, 0.2
    return _obj(("plate_bottom", _box([0, 0, 0], [1.0, 0.8, t])),
                ("film", _box([0, 0, t + gap], [1.0, 0.8, t + gap + f])),
                ("plate_top", _box([0, 0, t + 2 * gap + f],
                                   [1.0, 0.8, 2 * t + 2 * gap + f])))


def _ask_parts(tmp_path, data, **form):
    app, _store, _db, _launcher = make_app(tmp_path)
    resp = app.test_client().post(
        "/parts", data={"meshfile": (io_bytes(data), "plates.obj"),
                        "units": "um", **form},
        content_type="multipart/form-data")
    return resp.status_code, resp.get_json()


def test_parts_names_a_non_nested_td_scene_the_way_the_td_post_does(tmp_path):
    # The real /parts, no stub: the page's pickers come from this answer.
    status, body = _ask_parts(tmp_path, _plates(), mode="td")
    assert status == 200, body
    assert body["parts"] == ["plate_bottom", "film", "plate_top"]
    # ...and the td POST takes exactly those names, so the pickers it shows
    # are pickers it can submit.
    resp, db, _store, flashes = _post(
        tmp_path, _plates(), td_boundary="open",
        td_position="500 400 400", td_probes="500 400 400 z",
        **{"part_index.block": None, "part_index.cube": None,
           "part_index.plate_bottom": "Au", "part_index.film": "1.5",
           "part_index.plate_top": "Au"})
    assert resp.status_code == 302 and db.jobs, flashes
    (job,) = db.jobs.values()
    assert sorted(job["materials"]) == sorted(body["parts"])


def test_a_film_flush_between_two_plates_is_taken_at_upload(tmp_path):
    """Live defect (2026-10-02): flush faces leave the merged mesh open, so the
    upload's closed check fell back to the cavity parser, which refused it."""
    status, body = _ask_parts(tmp_path, _plates(gap=0.0), mode="td")
    assert status == 200, body
    assert body["parts"] == ["plate_bottom", "film", "plate_top"]
    resp, db, _store, flashes = _post(
        tmp_path, _plates(gap=0.0), td_boundary="open",
        td_position="500 400 300", td_probes="500 400 300 z",
        **{"part_index.block": None, "part_index.cube": None,
           "part_index.plate_bottom": "Au", "part_index.film": "1.5",
           "part_index.plate_top": "Au"})
    assert resp.status_code == 302 and db.jobs, flashes


def test_an_open_td_part_is_refused_by_name(tmp_path):
    # The closed check td now skips: the td split must refuse the hole itself.
    box = _box([0, 0, 0], [1, 1, 1])
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-2], process=False)
    data = _obj(("lid", _box([0, 0, 2], [1, 1, 3])), ("cup", open_box))
    status, body = _ask_parts(tmp_path, data, mode="td")
    assert status == 400 and "'cup' is not a closed surface" in body["error"], body
    _refused(tmp_path, "'cup' is not a closed surface", data=data,
             td_boundary="open",
             **{"part_index.block": None, "part_index.cube": None,
                "part_index.lid": "1.5", "part_index.cup": "1.5"})


def test_parts_that_share_volume_are_refused_at_upload_naming_both(tmp_path):
    """5.4e: refused before queueing, not failed later in the worker."""
    lens = _box([0, 0, 0], [0.6, 0.8, 0.6])
    prism = _box([0.4, 0, 0], [1.0, 0.8, 0.6])
    flashes = _refused(
        tmp_path, "'lens' and 'prism' overlap", _obj(("lens", lens), ("prism", prism)),
        td_boundary="open", td_position="200 400 300", td_probes="800 400 300 z",
        **{"part_index.block": None, "part_index.cube": None,
           "part_index.lens": "1.5", "part_index.prism": "2"})
    assert any("may touch, but not share volume" in m for m in flashes), flashes


def test_parts_still_runs_the_cavity_rules_for_a_cavity_upload(tmp_path):
    # the control: the same file without mode=td meets the cavity parser
    status, body = _ask_parts(tmp_path, _plates())
    assert status == 400 and body["error"], body


def test_the_page_asks_parts_for_the_td_split():
    page = (Path(TDF.__file__).parent / "templates" / "index.html").read_text()
    assert page.count("body.append('mode', 'td');") == 1


def test_a_multi_part_upload_without_materials_is_refused_in_plain_words(tmp_path):
    """Never one material for every part (lead, 08:15Z): a multi-part td
    upload that posts no per-part materials is refused, naming its parts and
    pointing at the page's pickers."""
    flashes = _refused(tmp_path, "has 2 parts ('block', 'cube')",
                       **{"part_index.cube": None, "part_index.block": None})
    assert any("Choose one for every part" in m for m in flashes), flashes
    assert not any("materials table" in m for m in flashes), flashes


def test_an_empty_mesh_is_refused(tmp_path):
    _refused(tmp_path, "empty", data=b"")


def test_an_open_surface_is_refused(tmp_path):
    box = _box([0, 0, 0], [1, 1, 1])
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-2], process=False)
    _refused(tmp_path, "closed", data=_obj(("block", open_box)),
             **{"part_index.cube": None})


def test_a_zero_size_part_is_refused(tmp_path):
    # Refused by the upload reader (its faces collapse), before any layout.
    flat = _box([0, 0, 0], [1, 1, 1])
    flat.vertices[:, 2] = 0.0
    _refused(tmp_path, "does not read consistently",
             data=_obj(("block", _box([0, 0, 0], [1.0, 0.8, 0.6])),
                       ("cube", flat)))


def test_nan_coordinates_are_refused(tmp_path):
    data = _scene().replace(b"v 1.0 ", b"v nan ", 1)
    assert b"nan" in data
    _refused(tmp_path, "does not read consistently", data=data)


def test_a_zero_scale_is_refused(tmp_path):
    _refused(tmp_path, "scale must be between", scale="0")


def test_a_scene_flat_on_one_axis_is_refused(tmp_path):
    """The only way to a one-node axis: every graded axis keeps both of its
    ends, so a coarse dx cannot make one; a scene with no extent on an axis
    can, and it is refused readably (its faces collapse on reading)."""
    flat = _box([0, 0, 0], [1.0, 0.8, 0.6])
    flat.vertices[:, 2] = 0.0
    _refused(tmp_path, "does not read consistently", data=_obj(("block", flat)),
             **{"part_index.cube": None})


def test_a_coarse_dx_still_lays_out_at_least_two_nodes_per_axis(tmp_path):
    resp, db, _store, flashes = _post(tmp_path, td_dx="5000")
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes


@pytest.mark.parametrize("field, raw", [
    ("td_duration_fs", "nan"), ("td_duration_fs", "-1"),
    ("td_frequency_thz", "inf"), ("td_dx", "zero"),
    ("td_bandwidth_thz", ""),
])
def test_a_bad_number_is_refused_by_field(tmp_path, field, raw):
    _refused(tmp_path, "must be", **{field: raw})


def test_a_mesh_library_failure_is_refused_not_a_500(tmp_path, monkeypatch):
    _fdtd, _layout, mse_parts = TDF._solver()

    def boom(_data):
        raise RuntimeError("the triangle library fell over")

    monkeypatch.setattr(mse_parts, "split_obj_parts", boom)
    flashes = _refused(tmp_path, "the triangle library fell over")
    assert any("could not be laid out" in m for m in flashes), flashes


def test_a_type_error_from_the_geometry_is_refused_not_a_500(tmp_path,
                                                             monkeypatch):
    _fdtd, fdtd_layout, _parts = TDF._solver()

    def boom(*_a, **_k):
        return None + 1

    monkeypatch.setattr(fdtd_layout, "plan", boom)
    _refused(tmp_path, "could not be laid out")


def test_a_source_in_the_gold_cube_is_refused_naming_it(tmp_path):
    _refused(tmp_path, "part cube", td_position="700 600 400")


def test_a_probe_in_the_gold_cube_is_refused_naming_it(tmp_path):
    _refused(tmp_path, "part cube", td_probes="700 600 400 z")


def test_a_source_outside_the_grid_is_refused(tmp_path):
    _refused(tmp_path, "outside the grid", td_position="300 300 900")


def test_a_probe_inside_the_absorbing_layer_is_refused(tmp_path):
    _refused(tmp_path, "absorbing layer", td_boundary="open",
             td_probes="-400 300 300 z")


def test_a_malformed_probe_line_is_refused(tmp_path):
    _refused(tmp_path, "probe line", td_probes="1 2 3 w")


# --- each limit, readably ---------------------------------------------------
def test_too_many_cells_is_refused_naming_the_count(tmp_path):
    _refused(tmp_path, "cells, over the limit of 8,000,000", td_dx="3",
             td_duration_fs="1")


def test_too_many_steps_is_refused_naming_the_count(tmp_path):
    _refused(tmp_path, "time steps, over the limit of 20,000",
             td_duration_fs="100000")


@pytest.mark.parametrize("limit, needle", [
    ("MAX_BYTES", "GB of GPU memory"), ("MAX_SECONDS", "minutes, over"),
])
def test_memory_and_time_limits_reach_the_page(tmp_path, monkeypatch, limit,
                                               needle):
    fdtd, _layout, _parts = TDF._solver()
    monkeypatch.setattr(fdtd, limit, 0)
    _refused(tmp_path, needle)


# --- the drawn waveform -----------------------------------------------------
def _drawn(samples, duration_fs="10"):
    return {"td_profile": "drawn", "td_bandwidth_thz": None,
            "td_samples": " ".join(map(str, samples)),
            "td_sample_duration_fs": duration_fs}


def test_a_smooth_drawing_is_queued(tmp_path):
    t = np.linspace(0, 1, 200)
    # Zero-mean and zero at both ends (the check removes the mean, so a bump
    # would gain a step at each end): a 50 THz burst under a Gaussian.
    smooth = np.sin(2 * np.pi * 2 * t) * np.exp(-((t - 0.5) / 0.15) ** 2)
    resp, db, _store, flashes = _post(tmp_path,
                                      **_drawn(np.round(smooth, 5), "40"))
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    tds = next(iter(db.jobs.values()))["tds"]
    assert len(tds["samples"]) == 200 and tds["bandwidth"] is None
    assert tds["sample_duration_s"] == _close(40e-15)


def test_a_sharp_drawing_is_refused_naming_f_res(tmp_path):
    square = [0, 0, 1, 1, 0, 0, -1, -1] * 25
    _refused(tmp_path, "the highest frequency this grid resolves",
             **_drawn(square))


def test_a_constant_drawing_is_refused(tmp_path):
    _refused(tmp_path, "flat", **_drawn([1, 1, 1, 1]))


def test_a_drawing_under_two_steps_is_refused(tmp_path):
    _refused(tmp_path, "under two time steps", **_drawn([0, 1, 0], "0.01"))


def test_a_plane_wave_polarised_along_its_travel_is_refused(tmp_path):
    _refused(tmp_path, "transverse", td_boundary="open",
             td_source_kind="plane", td_direction="+x", td_polarisation="x",
             td_position=None, td_dx="100", td_probes="500 400 300 z")


def test_the_drawn_limit_counts_the_gold_cube_index(tmp_path):
    """A 275 THz burst: under the grid's f_res for the glass alone (n = 1.5,
    about 400 THz at dx = 50 nm), over it once gold's |n| at 450 THz counts.
    A check that knew only the dielectric would queue it."""
    t = np.linspace(0, 1, 400)
    burst = np.sin(2 * np.pi * 11 * t) * np.exp(-((t - 0.5) / 0.15) ** 2)
    _refused(tmp_path, "the highest frequency this grid resolves",
             **_drawn(np.round(burst, 5), "40"))
    # The positive control: the same burst with the cube made of glass too.
    resp, db, _store, flashes = _post(tmp_path / "glass",
                                      **_drawn(np.round(burst, 5), "40"),
                                      **{"part_index.cube": "1.5"})
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes


# --- the upload page's td arm ------------------------------------------------
#
# The page and web/td_form.py meet only through field names and option values.
# These read both sides from their sources -- the names td_form actually reads,
# the tuples it actually checks against -- so a renamed field or a new choice
# on one side alone goes red here instead of as a refusal the owner cannot fix.

class _Controls(__import__("html.parser").parser.HTMLParser):
    """Every named form control on a page: name -> (tag, attrs, [option values]),
    and the ``data-modes`` of the fieldset each sits in."""

    def __init__(self):
        super().__init__()
        self.controls, self.modes, self.selected = {}, {}, {}
        self._fieldsets, self._select = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "fieldset":
            self._fieldsets.append(attrs.get("data-modes"))
        elif tag in ("input", "select", "textarea") and attrs.get("name"):
            self.controls[attrs["name"]] = (tag, attrs, [])
            self.modes[attrs["name"]] = next(
                (m for m in reversed(self._fieldsets) if m), None)
            if tag == "select":
                self._select = attrs["name"]
        elif tag == "option" and self._select:
            self.controls[self._select][2].append(attrs.get("value"))
            if "selected" in attrs:
                self.selected[self._select] = attrs.get("value")

    def handle_endtag(self, tag):
        if tag == "fieldset":
            self._fieldsets.pop()
        elif tag == "select":
            self._select = None


def _page(tmp_path):
    app, _store, _db, _launcher = make_app(tmp_path)
    resp = app.test_client().get("/")
    assert resp.status_code == 200
    parser = _Controls()
    parser.feed(resp.get_data(as_text=True))
    return parser


def test_the_page_offers_exactly_the_td_fields_the_form_reads(tmp_path):
    import re  # noqa: PLC0415
    source = (Path(TDF.__file__)).read_text()
    read = set(re.findall(r'(?:_field|_choice)\(form, "(td_\w+)"', source))
    read |= set(re.findall(r'form\.get\("(td_\w+)"', source))
    assert len(read) >= 14, read          # the parser itself must have found them
    page = _page(tmp_path)
    offered = {n for n in page.controls if n.startswith("td_")}
    assert offered == read, (sorted(offered - read), sorted(read - offered))
    assert all(page.modes[n] == "td" for n in offered), page.modes


def test_the_td_choices_on_the_page_are_the_ones_the_form_accepts(tmp_path):
    page = _page(tmp_path)
    expected = {"td_unit": tuple(TDF.LENGTH_UNITS), "td_boundary": TDF.BOUNDARIES,
                "td_source_kind": TDF.SOURCE_KINDS, "td_profile": TDF.PROFILES,
                "td_polarisation": TDF.POLARISATIONS,
                "td_direction": TDF.DIRECTIONS}
    for name, choices in expected.items():
        assert sorted(page.controls[name][2]) == sorted(choices), name
    assert "td" in page.controls["mode"][2]


def test_the_part_pickers_offer_exactly_the_materials_the_form_accepts(tmp_path):
    import json, re  # noqa: E401, PLC0415
    app, _store, _db, _launcher = make_app(tmp_path)
    html = app.test_client().get("/").get_data(as_text=True)
    found = re.findall(r"var named = (\[[^\]]*\]);", html)
    assert len(found) == 1, found
    offered = json.loads(found[0])
    fdtd = TDF._solver()[0]
    assert offered == list(TDF.named_materials(fdtd))
    assert "pec" in offered and len(offered) > 1
    for name in offered:          # every offered choice parses as itself
        assert TDF.parse_materials({"part_index.a": name}, fdtd) == {"a": name}


def test_the_td_arm_shows_the_upload_it_reads(tmp_path):
    # The td route reads meshfile, units and scale. They live in the shared
    # upload fieldset, which must be shown for td too: a td arm without them
    # posts a request with no file the owner could have chosen.
    page = _page(tmp_path)
    for name in ("meshfile", "units", "scale"):
        assert "td" in (page.modes[name] or "").split(), (name, page.modes[name])


def test_the_pages_own_td_defaults_are_accepted(tmp_path):
    # Submit what the browser would post for the td arm as first shown: every
    # default value, the shown (point, pulse) fields only, plus the two fields
    # that start empty because only the owner can know them.
    page = _page(tmp_path)
    fields = {"mode": "td", "units": "um", "scale": ""}
    for name, (tag, attrs, options) in page.controls.items():
        if not name.startswith("td_"):
            continue
        if tag == "select":   # the browser posts the first option unless one is selected
            fields[name] = page.selected.get(name, options[0])
        else:
            fields[name] = attrs.get("value", "")
    assert fields["td_source_kind"] == "point" and fields["td_profile"] == "pulse"
    for hidden in ("td_direction", "td_samples", "td_sample_duration_fs"):
        fields.pop(hidden)
    fields["td_position"] = "0.5 0.4 0.3"
    fields["td_probes"] = "0.5 0.4 0.5 z"
    # One part: a multi-part upload posts per-part pickers, which only the
    # page's script fills in (formcheck.mjs covers that path).
    block = _obj(("block", _box([0, 0, 0], [1.0, 0.8, 0.6])))
    fields["meshfile"] = (io_bytes(block), "block.obj")
    app, _store, db, _launcher = make_app(tmp_path / "post")
    client = app.test_client()
    resp = client.post("/", data=fields, content_type="multipart/form-data")
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    assert resp.status_code == 302 and len(db.jobs) == 1, (fields, flashes)


# --- the movie window (change td-movie-window) -------------------------------

def _estimate(tmp_path, **over):
    """The cap-check estimate the form computes for these fields, through the
    form's own reader and layout (so the test compares against the number the
    refusal itself uses, not a guess)."""
    fdtd, fdtd_layout, _parts = TDF._solver()
    fields = _fields(**over)
    blob = fields.pop("meshfile")[0]
    data = blob.read() if hasattr(blob, "read") else blob
    request = TDF.parse_request(fields)
    names, parts = TDF.split_parts(data)
    planned = fdtd_layout.plan(names, parts, request, length_scale=1e-6,
                               materials=TDF.parse_materials(fields, fdtd),
                               n_index=1.0)
    return fdtd_layout.estimate(planned, request)


def test_the_movie_fields_reach_the_worker_in_seconds_and_a_count(tmp_path):
    over = {"td_movie_start_fs": "5", "td_movie_end_fs": "15",
            "td_frame_count": "12"}
    resp, db, _store, flashes = _post(tmp_path, **over)
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    job = next(iter(db.jobs.values()))
    tds = job["tds"]
    assert tds["movie_start_s"] == _close(5e-15)
    assert tds["movie_end_s"] == _close(15e-15)
    assert tds["frame_count"] == 12
    # the worker's own reader and plan take the stored row, and the frames are
    # the window's: inside 5..15 fs, the last at the window's own end. The step
    # bounds come from the run's dt, not from an assumed one.
    import td_solve
    est = _estimate(tmp_path, **over)
    dt = float(est["dt"])
    plan = TDF._solver()[1].movie_window(td_solve.request_from(job), est)
    times = [s * dt for s in plan["steps"]]
    assert len(times) == 12, plan
    assert all(5e-15 <= t <= 15e-15 for t in times), times
    assert 0 < 15e-15 - times[-1] < dt, (
        "the last frame is the window's own end", times[-1])
    gap = times[1] - times[0]
    assert 0 < times[0] - 5e-15 <= gap + dt, (
        "the first frame is one bucket inside the window, as in today's plan",
        times[0])
    assert plan["steps"][-1] < est["steps"] - 1, "the window is shorter than the run"


def test_a_form_without_the_movie_fields_queues_todays_request(tmp_path):
    """The compatibility promise at the form: the fields the page has always
    had (an older page, or the page's own blanks) still queue a job, and the
    worker plans today's whole-run movie, step for step."""
    import fdtd
    import td_solve
    resp, db, _store, flashes = _post(tmp_path)
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    job = next(iter(db.jobs.values()))
    tds = job["tds"]
    assert (tds["movie_start_s"], tds["movie_end_s"], tds["frame_count"]) == (
        None, None, None)
    est = _estimate(tmp_path)
    plan = TDF._solver()[1].movie_window(td_solve.request_from(job), est)
    want = fdtd.frame_plan(tuple(est["shape"]), est["steps"],
                           max_frames=fdtd.MAX_FRAMES)
    assert plan == want


def test_the_window_spread_and_end_are_the_workers_own(tmp_path):
    """The form does not spread the frames itself: it stores the window and the
    worker plans it, so both read the same rule (``fdtd.frame_plan``'s), and
    the stored request carries no plan of its own."""
    over = {"td_movie_start_fs": "5", "td_movie_end_fs": "15",
            "td_frame_count": "7"}
    resp, db, _store, flashes = _post(tmp_path, **over)
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    tds = next(iter(db.jobs.values()))["tds"]
    assert "steps" not in tds and "stride" not in tds, sorted(tds)
    request = TDF.parse_request(_fields(**over))
    est = _estimate(tmp_path, **over)
    dt = float(est["dt"])
    plan = TDF._solver()[1].movie_window(request, est)
    times = [s * dt for s in plan["steps"]]
    assert len(times) == 7, plan
    assert all(5e-15 <= t <= 15e-15 for t in times)
    assert 0 < 15e-15 - times[-1] < dt, "the last frame is the window's end"
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert max(gaps) - min(gaps) < 2 * dt, "evenly spread over the window"


@pytest.mark.parametrize("over, needle", [
    ({"td_movie_start_fs": "15", "td_movie_end_fs": "5"}, "at or before its start"),
    ({"td_movie_start_fs": "15", "td_movie_end_fs": "15"}, "at or before its start"),
    ({"td_movie_end_fs": "999"}, "outside the run"),
    ({"td_movie_start_fs": "-3"}, "outside the run"),
    ({"td_frame_count": "121"}, "at most 120"),
    ({"td_frame_count": "0"}, "at least 1"),
    ({"td_frame_count": "-4"}, "at least 1"),
    ({"td_frame_count": "2.5"}, "whole number"),
    ({"td_frame_count": "lots"}, "whole number"),
])
def test_a_movie_the_run_cannot_honour_is_refused_before_queueing(tmp_path, over,
                                                                 needle):
    _refused(tmp_path, needle, **over)


def test_a_window_that_holds_enough_steps_is_not_refused(tmp_path):
    """The guard on the guard: the same shape of request that is refused above
    is accepted when the window is wide enough for its frames."""
    resp, db, _store, flashes = _post(tmp_path, td_movie_start_fs="1",
                                      td_movie_end_fs="2", td_frame_count="2")
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes


def test_the_relation_between_a_window_and_its_frames_is_refused(tmp_path):
    """The two refusals that are about steps, not about the window's position:
    a window narrower than one time step, and one that holds steps but fewer of
    them than the frames asked for. Both use the run's own dt (asserted here, so
    a fixture whose dt changed fails loudly rather than passing vacuously)."""
    dt = float(_estimate(tmp_path)["dt"])
    # a window of a ten-thousandth of a femtosecond is under the run's own step
    assert dt > 2e-19, f"fixture: dt is {dt} s, not coarser than the window"
    _refused(tmp_path, "holds no time steps",
             td_movie_start_fs="5", td_movie_end_fs="5.0001")
    # a 1 fs window of a 20 fs run: more than 60 steps, fewer than 120
    resp, db, _store, flashes = _post(tmp_path, td_movie_start_fs="5",
                                      td_movie_end_fs="6", td_frame_count="60")
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
    _refused(tmp_path, "holds only", td_movie_start_fs="5",
             td_movie_end_fs="6", td_frame_count="120")


def test_a_120_frame_movie_is_refused_where_the_same_run_at_60_fits(
        tmp_path, monkeypatch):
    """The estimate is the requested count's, so a machine with only this much
    memory refuses the 120-frame movie and still takes the 60-frame one."""
    fdtd, _layout, _parts = TDF._solver()
    sixty = _estimate(tmp_path)
    many = _estimate(tmp_path, td_frame_count="120")
    assert many["bytes"] > sixty["bytes"], "fixture: more frames cost more"
    assert many["frames"] == 120 and sixty["frames"] == 60
    monkeypatch.setattr(fdtd, "MAX_BYTES", (sixty["bytes"] + many["bytes"]) // 2)
    refused = _refused(tmp_path, "GB of GPU memory", td_frame_count="120")
    assert any("fewer than the 120 movie frames" in m for m in refused), refused
    resp, db, _store, flashes = _post(tmp_path)
    assert resp.status_code == 302 and len(db.jobs) == 1, flashes
