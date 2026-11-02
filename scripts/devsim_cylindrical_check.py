"""Check how devsim 2.11.2 weights the weak form for a 2-D cylindrical mesh.

Run with DEVSIM_MATH_LIBS set if the wheel cannot find a system BLAS
(e.g. DEVSIM_MATH_LIBS="libblas.so.3:liblapack.so.3"). Prints EdgeCouple before and
after cylindrical_edge_couple, so the claim in openspec/changes/electrical-simulation/
design.md section 2 can be re-checked rather than believed.

Expected: EdgeCouple is identical before and after; CylindricalEdgeCouple rises with r.

Also prints the cylindrical node-volume sum against the exact annulus volume
pi*(r_out^2 - r_in^2)*h at N_r = 10, 20, 40, which is the 1 + 1/(9*N_r) figure
design.md section 2 quotes. If the numbers below disagree with that text, the
text is wrong, not the script.
"""
import math
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

# --- Cylindrical node-volume convergence -------------------------------------
# A uniform annular grid over [r_in, r_out] x [0, h], N_r cells radially and one
# axially. The exact volume is pi*(r_out^2 - r_in^2)*h; the check is whether the
# sum of devsim's cylindrical node volumes converges to it, and how fast.
def _annulus_volume_error(nr, r_in=1.0, r_out=2.0, h=1.0):
    m = f"annulus_{nr}"
    d.create_2d_mesh(mesh=m)
    for i in range(nr + 1):
        d.add_2d_mesh_line(mesh=m, dir="x", pos=r_in + (r_out - r_in) * i / nr, ps=1.0)
    d.add_2d_mesh_line(mesh=m, dir="y", pos=0.0, ps=1.0)
    d.add_2d_mesh_line(mesh=m, dir="y", pos=h, ps=1.0)
    d.add_2d_region(mesh=m, material="Silicon", region="si",
                    xl=r_in, xh=r_out, yl=0.0, yh=h)
    d.finalize_mesh(mesh=m)
    dev = f"dev_{nr}"
    d.create_device(mesh=m, device=dev)
    d.cylindrical_node_volume(device=dev, region="si")
    total = sum(d.get_node_model_values(device=dev, region="si", name="CylindricalNodeVolume"))
    exact = math.pi * (r_out**2 - r_in**2) * h
    return total, exact


print()
print("Cylindrical node-volume sum vs exact annulus pi*(r_out^2-r_in^2)*h:")
print("  N_r   devsim          exact          ratio      N_r*(ratio-1)")
for nr in (10, 20, 40):
    total, exact = _annulus_volume_error(nr)
    ratio = total / exact
    print(f"  {nr:<5d} {total:<15.9f} {exact:<14.9f} {ratio:.9f}  {nr * (ratio - 1):.6f}")
