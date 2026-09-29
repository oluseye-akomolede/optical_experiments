"""Write tests/js/fixtures/two_part_surface.glb -- the traverse/add fixture.

Two disjoint closed boxes in one glb *scene* with two mesh nodes, which is the
shape the viewer's surface loader has to handle and the shape a single-part glb
cannot test: `gltf.scene.traverse` is the thing under test, and a bug that drops
every mesh but the last is invisible with one mesh.

Written with numpy and struct directly rather than with trimesh: trimesh is a
solver-side dependency and is not installed everywhere this test runs (the JS
tests run under plain node, and the fixture is checked in so they do not need
to regenerate it). The format emits exactly the subset of glTF 2.0 the browser's
GLTFLoader needs -- one binary buffer, float32 POSITION, uint32 indices.

    python tests/js/fixtures/make_two_part_surface.py
"""
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Two boxes, disjoint in x. The gap is deliberate: a traversal that skips a
# sibling leaves an obvious hole, and no renderer can merge them into looking
# like one part.
PART_A = dict(name="part_a", lo=(-5.0, -2.0, -2.0), hi=(-1.0, 2.0, 2.0))
PART_B = dict(name="part_b", lo=(1.0, -1.5, -2.0), hi=(5.0, 1.5, 2.0))


def box_triangles(lo, hi):
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], dtype=np.float32)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return v, np.asarray(idx, dtype=np.uint32)


def main():
    parts = [box_triangles(p["lo"], p["hi"]) for p in (PART_A, PART_B)]

    # One buffer view per part per attribute, so the accessors are self-
    # contained and an offset bug shows up as a wrong box rather than as a
    # silently shifted mesh.
    blob = bytearray()
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []
    for i, (verts, idx) in enumerate(parts):
        voff = len(blob)
        blob += verts.tobytes()
        ioff = len(blob)
        blob += idx.tobytes()
        while len(blob) % 4:
            blob += b"\x00"

        buffer_views.append({"buffer": 0, "byteOffset": voff,
                             "byteLength": int(verts.nbytes), "target": 34962})
        buffer_views.append({"buffer": 0, "byteOffset": ioff,
                             "byteLength": int(idx.nbytes), "target": 34963})
        vmin = verts.min(axis=0).astype(float).tolist()
        vmax = verts.max(axis=0).astype(float).tolist()
        accessors.append({"bufferView": len(buffer_views) - 2,
                          "componentType": 5126, "count": int(len(verts)),
                          "type": "VEC3", "min": vmin, "max": vmax})
        accessors.append({"bufferView": len(buffer_views) - 1,
                          "componentType": 5125, "count": int(len(idx)),
                          "type": "SCALAR"})
        meshes.append({"name": [PART_A, PART_B][i]["name"],
                       "primitives": [{"attributes": {"POSITION": len(accessors) - 2},
                                       "indices": len(accessors) - 1, "mode": 4}]})
        nodes.append({"name": [PART_A, PART_B][i]["name"], "mesh": i})

    doc = {
        "asset": {"version": "2.0", "generator": "make_two_part_surface.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = bytes(blob)
    while len(bin_bytes) % 4:
        bin_bytes += b"\x00"

    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, 0)
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    out += struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes

    path = HERE / "two_part_surface.glb"
    path.write_bytes(bytes(out))
    print(f"wrote {path} ({len(out)} bytes), "
          f"{len(meshes)} meshes, {len(nodes)} nodes")


if __name__ == "__main__":
    main()
