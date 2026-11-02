"""Check how devsim 2.11.2 weights the weak form for a 2-D cylindrical mesh.

Run with DEVSIM_MATH_LIBS set if the wheel cannot find a system BLAS
(e.g. DEVSIM_MATH_LIBS="libblas.so.3:liblapack.so.3"). Prints EdgeCouple before and
after cylindrical_edge_couple, so the claim in openspec/changes/electrical-simulation/
design.md section 2 can be re-checked rather than believed.

Expected: EdgeCouple is identical before and after; CylindricalEdgeCouple rises with r.
"""
import devsim as d
d.set_parameter(name="raxis_zero", value=0.0); d.set_parameter(name="raxis_variable", value="x")
d.create_2d_mesh(mesh="m")
d.add_2d_mesh_line(mesh="m", dir="x", pos=0.5, ps=0.25)
d.add_2d_mesh_line(mesh="m", dir="x", pos=1.0, ps=0.25)
for j in range(3): d.add_2d_mesh_line(mesh="m", dir="y", pos=j*0.5, ps=0.5)
d.add_2d_region(mesh="m", material="Silicon", region="si", xl=0.5, xh=1.0, yl=0.0, yh=1.0)
d.finalize_mesh(mesh="m"); d.create_device(mesh="m", device="dv")
X=d.get_node_model_values(device="dv", region="si", name="x")
print("BEFORE cylindrical_edge_couple:")
print("  EdgeCouple:", [round(v,4) for v in d.get_edge_model_values(device="dv", region="si", name="EdgeCouple")])
d.cylindrical_edge_couple(device="dv", region="si")
print("AFTER:")
print("  EdgeCouple            :", [round(v,4) for v in d.get_edge_model_values(device="dv", region="si", name="EdgeCouple")])
print("  CylindricalEdgeCouple :", [round(v,4) for v in d.get_edge_model_values(device="dv", region="si", name="CylindricalEdgeCouple")])
