"""The roundness check for the axisymmetric (r, z) form (task 2.6).

Every fixture is a small generated prism: an N-gon (circle, ellipse or hexagon)
extruded along an axis, caps fanned from a centre vertex. The figure each one
must measure is written next to it, derived from the polygon, not from the code:
a regular N-gon of circumradius R has inradius R cos(pi/N), and a 2N-point
ellipse polygon of semi-axes a > b has circumradius a and inradius just under b.

The 2 % limit is ``mse_geometry.ROUNDNESS_TOLERANCE``. The two elliptic cases
straddle it (1.5 % and 3 %), so moving the limit past either one turns a test
red; tasks.md 2.6 records that mutation run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))
sys.path.insert(0, str(ROOT / "web"))

import mse_geometry as GEO  # noqa: E402


def _prism(a=1.0, b=None, n=128, height=3.0, phase=0.0):
    """A closed prism over an n-gon with semi-axes (a, b), axis +z from z=0."""
    b = a if b is None else b
    ang = phase + 2.0 * np.pi * np.arange(n) / n
    ring = np.c_[a * np.cos(ang), b * np.sin(ang)]
    verts = np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, height)],
                       [[0, 0, 0], [0, 0, height]]])
    bot, top = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [bot, j, i], [top, n + i, n + j]]
    return verts, np.array(faces)


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# --- the measurement ------------------------------------------------------

def test_cylinder_passes_with_its_own_radius():
    verts, faces = _prism(a=2.0, n=256)
    m = GEO.require_round(verts, faces, "nm")
    assert m["circumradius"] == pytest.approx(2.0, rel=1e-9)
    assert m["inradius"] == pytest.approx(2.0 * np.cos(np.pi / 256), rel=1e-9)
    assert abs(m["axis_direction"][2]) == pytest.approx(1.0, abs=1e-9)


def test_faceted_cylinder_passes_at_its_facet_ratio():
    # 24 facets: 1/cos(7.5 deg) = 1.00863, inside 2 %.
    verts, faces = _prism(a=1.0, n=24)
    m = GEO.require_round(verts, faces, "nm")
    assert m["ratio"] == pytest.approx(1.0 / np.cos(np.pi / 24), rel=1e-9)


def test_hexagonal_prism_is_refused_naming_both_figures_and_the_limit():
    # The paper-1 wire: circumradius 239.4 nm, inradius 239.4 cos 30 = 207.3 nm.
    verts, faces = _prism(a=239.4, n=6, height=1500.0)
    with pytest.raises(GEO.NotRoundError) as err:
        GEO.require_round(verts, faces, "nm")
    msg = str(err.value)
    assert "circumradius of 239.4 nm" in msg
    assert "inradius of 207.3 nm" in msg
    assert "15.47%" in msg  # 1/cos(30 deg) - 1
    assert "at most 2%" in msg
    assert "3-D form" in msg


def test_elliptic_cylinder_at_3_percent_is_refused():
    verts, faces = _prism(a=1.03, b=1.0, n=256)
    m = GEO.measure_roundness(verts, faces)
    assert m["ratio"] == pytest.approx(1.03, abs=1e-3)
    with pytest.raises(GEO.NotRoundError, match="at most 2%"):
        GEO.require_round(verts, faces, "um")


def test_elliptic_cylinder_at_1_5_percent_passes():
    verts, faces = _prism(a=1.015, b=1.0, n=256)
    m = GEO.require_round(verts, faces, "um")
    assert m["ratio"] == pytest.approx(1.015, abs=1e-3)


def test_tilted_and_offset_cylinder_passes_about_its_detected_axis():
    verts, faces = _prism(a=1.0, n=256, height=4.0)
    rot = _rotation([1.0, 2.0, 0.5], 0.7)
    moved = verts @ rot.T + np.array([5.0, -3.0, 2.0])
    m = GEO.require_round(moved, faces, "mm")
    assert m["ratio"] == pytest.approx(1.0 / np.cos(np.pi / 256), rel=1e-9)
    assert abs(m["axis_direction"] @ (rot @ [0, 0, 1])) == pytest.approx(1.0, abs=1e-9)
    # Control: the same body measured about the coordinate z axis through its
    # centroid is far from round, so the pass above is the axis detection's.
    about_z = GEO.roundness_about(moved, faces, m["axis_point"], [0, 0, 1])
    assert about_z["ratio"] > 1.5


def test_squat_cylinder_where_the_normal_tensor_is_degenerate():
    # L = 2R makes the normal tensor isotropic; the second moment still finds z.
    verts, faces = _prism(a=1.0, n=256, height=2.0)
    m = GEO.require_round(verts, faces, "nm")
    assert abs(m["axis_direction"][2]) == pytest.approx(1.0, abs=1e-6)


# --- the upload refusal ---------------------------------------------------

def _obj(*parts):
    """OBJ text with one ``g`` group per (name, verts, faces)."""
    lines, base = [], 1
    for name, verts, faces in parts:
        lines.append(f"g {name}")
        lines += [f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in verts]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in faces]
        base += len(verts)
    return ("\n".join(lines) + "\n").encode()


def test_upload_refuses_a_hexagonal_wire_with_upload_error():
    pytest.importorskip("trimesh")
    import meshing

    data = _obj(("wire", *_prism(a=239.4, n=6, height=1500.0)))
    meshing.validate_upload("wire.obj", data, 10 * 1024 * 1024)
    with pytest.raises(meshing.UploadError, match="circumradius of 239.4 nm"):
        meshing.refuse_unless_round(data, "nm")


def test_upload_judges_every_part_not_only_the_biggest():
    pytest.importorskip("trimesh")
    import meshing

    wire = _prism(a=1.0, n=128, height=3.0)
    assert meshing.refuse_unless_round(_obj(("wire", *wire)), "um")["ratio"] < 1.02
    # A hexagonal pad on the same axis, above the wire and not touching it.
    hv, hf = _prism(a=0.5, n=6, height=0.5)
    pad = (hv + [0, 0, 3.5], hf)
    with pytest.raises(meshing.UploadError, match="at most 2%"):
        meshing.refuse_unless_round(_obj(("wire", *wire), ("pad", *pad)), "um")


def _tube(r_in, r_out, n=128, height=3.0):
    """A closed annular prism: outer wall out, inner wall in, ring caps."""
    ang = 2.0 * np.pi * np.arange(n) / n
    c, s = np.cos(ang), np.sin(ang)
    rings = [np.c_[r * c, r * s, np.full(n, z)]
             for r, z in ((r_out, 0), (r_out, height), (r_in, 0), (r_in, height))]
    ob, ot, ib, it = (k * n for k in range(4))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i],      # outer
                  [ib + i, it + j, ib + j], [ib + i, it + i, it + j],      # inner
                  [ob + i, ib + j, ob + j], [ob + i, ib + i, ib + j],      # bottom
                  [ot + i, ot + j, it + j], [ot + i, it + j, it + i]]      # top
    return np.vstack(rings), np.array(faces)


def test_core_inside_a_tube_is_judged_loop_by_loop():
    # Three round loops per slice (radii 1, 1.2, 2). Comparing the biggest
    # with the smallest would read 2.0; each loop alone reads 1/cos(pi/128).
    pytest.importorskip("trimesh")
    import meshing

    data = _obj(("tube", *_tube(1.2, 2.0)), ("core", *_prism(a=1.0, n=128)))
    m = meshing.refuse_unless_round(data, "um")
    assert m["ratio"] == pytest.approx(1.0 / np.cos(np.pi / 128), rel=1e-9)
