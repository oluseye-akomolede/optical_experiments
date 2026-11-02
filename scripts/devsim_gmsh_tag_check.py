"""Check which element-line spelling devsim 2.11.2 reads as the physical group.

The electrical design's 0.2 note used to tell the build to write element lines as
`id type ntags element_index physical_number nodes...` with three tags. That
parses without error, but the reader takes `ilist[3]` as the physical number
(src/meshing/GmshParser.y: `physical_number = ilist[3]`,
`element_node_begin = 3 + number_tags`). With a separate element index in front,
the element index becomes the physical group, so every triangle lands in a group
named after its index and `add_gmsh_region(gmsh_name="device")` builds a region
with no nodes in it -- a wrong answer with no error.

This script decides it against the installed devsim rather than by reading the
grammar: it loads the same two-triangle mesh written both ways, maps BOTH
candidate group names to regions, and reports which region ends up holding the
nodes.

    python scripts/devsim_gmsh_tag_check.py

Needs devsim 2.11.2. The x86-64 wheel does not bundle a BLAS, so it also needs
DEVSIM_MATH_LIBS="libblas.so.3:liblapack.so.3" (the aarch64 wheel bundles its own).
"""

import os
import tempfile

import devsim

_HEAD = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 7 device
2 1 outer
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
"""

# Two triangles over the unit square. Gmsh 2.2: `id type ntags physical geom nodes...`
_CORRECT = (
    _HEAD
    + """$Elements
2
1 2 2 7 1 1 2 3
2 2 2 7 1 1 3 4
$EndElements
"""
)

# The spelling the note used to require: ntags=3, with a separate element index
# (8) before the physical number (7), so ilist[3] is the index rather than the
# physical group.
_DOCUMENTED = (
    _HEAD
    + """$Elements
2
1 2 3 8 7 1 1 2 3
2 2 3 8 7 1 1 3 4
$EndElements
"""
)


def _nodes_per_region(text, name):
    """Map candidate names to regions and report each region's node count."""
    path = os.path.join(tempfile.mkdtemp(prefix=f"devsim-{name}-"), "m.msh")
    with open(path, "w") as f:
        f.write(text)
    mesh = f"{name}_mesh"
    devsim.create_gmsh_mesh(file=path, mesh=mesh)
    candidates = ("device", "outer", "8")
    for gmsh_name in candidates:
        devsim.add_gmsh_region(
            mesh=mesh, gmsh_name=gmsh_name, region=f"{name}_{gmsh_name}", material="m"
        )
    devsim.finalize_mesh(mesh=mesh)
    device = f"{name}_device"
    devsim.create_device(mesh=mesh, device=device)
    counts = {}
    for gmsh_name in candidates:
        region = f"{name}_{gmsh_name}"
        try:
            counts[gmsh_name] = len(
                devsim.get_node_model_values(device=device, region=region, name="x")
            )
        except Exception:  # a region with no nodes has no models
            counts[gmsh_name] = 0
    return counts


def main():
    correct = _nodes_per_region(_CORRECT, "correct")
    documented = _nodes_per_region(_DOCUMENTED, "documented")

    print("Gmsh 2.2 element line (2 tags, physical group 7 named 'device'):")
    print(f"  nodes per region: {correct}")
    print()
    print("Three-tag line from the old note (element index 8 before physical 7):")
    print(f"  nodes per region: {documented}")
    print()

    if correct.get("device") != 4:
        raise SystemExit(
            f"FAIL: Gmsh 2.2 should put all 4 nodes in 'device', got {correct}"
        )
    if documented.get("device") != 0:
        raise SystemExit(
            "FAIL: the three-tag spelling also filled 'device' -- this check "
            f"cannot tell the two spellings apart, so it proves nothing: {documented}"
        )
    if documented.get("8") != 4:
        raise SystemExit(
            f"FAIL: expected the element index 8 to become the physical group, got {documented}"
        )
    print("OK: Gmsh 2.2 -> 'device' holds all 4 nodes (correct).")
    print("    Three tags -> group '8' holds all 4 nodes and 'device' is empty")
    print("    (the element index read as the physical number).")
    print("The build must use ordinary Gmsh 2.2 element lines.")


if __name__ == "__main__":
    main()
