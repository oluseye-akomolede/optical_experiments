"""Meshes with KNOWN electrostatics answers, for the bench to score the solver against.

Three fixtures with answers, in increasing order of what each can prove:

1. ``two_layer_slab``      -- series capacitors, CLOSED FORM for C and for the
                              interface potential. Catches any eps_s handling
                              error, including the "ignore eps_s" mutation.
2. ``concentric_spheres``  -- the only fixture with a curved, radially varying
                              field, in two dielectric shells. Closed-form C and
                              CLOSED-FORM phi(r), so it tests the field's shape
                              and not just a total charge.
3. ``sphere_between_plates`` -- no closed form and no cheap converged
                              reference; the bench reports its numbers and
                              flags it as unvalidated. Kept because it is the
                              first fixture whose field is genuinely 3-D.

**Where the expected values come from.** Every closed form below is the
textbook statement of its geometry, typed independently of anything the solver
computes. That independence is the point: an expectation derived from the
solver's formulas would agree with the solver whatever it did. Fixture 3 has no
closed form, and is reported as raw numbers, never as "correct".

Units: metres, volts, farads.
"""

from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12   # vacuum permittivity, F/m (CODATA 2018)

_CUBE_TETS = np.array(
    [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
)


def _box_tets(xs, ys, zs):
    """A structured box of cubes, each split into six tets about one diagonal.

    Node planes are placed ON the material interfaces, so no element straddles
    two parts. That matters for the bench specifically: a straddling element
    would make the measured answer depend on how the straddle was assigned,
    which is a mesh artefact standing in for a solver property.
    """
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    points = np.array(
        [[xs[i], ys[j], zs[k]] for i in range(nx + 1) for j in range(ny + 1)
         for k in range(nz + 1)]
    )
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corner = {(a, b, c): node(i + a, j + b, k + c)
                          for a in (0, 1) for b in (0, 1) for c in (0, 1)}
                for t in _CUBE_TETS:
                    bits = [tuple(int(v) for v in np.unravel_index(u, (2, 2, 2))) for u in t]
                    tets.append([corner[b] for b in bits])
    return points, np.array(tets, dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. Two-layer slab: two series capacitors, closed form
# ---------------------------------------------------------------------------

def two_layer_slab(eps_a=2.0, eps_b=5.0, d_a=100e-6, d_b=40e-6,
                   side=400e-6, n_side=8, n_a=4, n_b=4):
    """Two dielectric layers between two plates, with a held plane between them.

    z = 0 .. d_a          : part ``a``, eps_a
    z = d_a .. d_a+d_b    : part ``b``, eps_b
    Every node at z = 0 is held at 0 V and every node at the top face at 1 V.
    The lateral box is exactly the plate size, and the sides are natural
    (zero normal field) -- which is precisely the assumption that makes the
    1-D closed form correct, so the fixture's error is pure discretisation.

    The two layers are in SERIES: the same displacement D crosses both, so

        C   = (1/C_a + 1/C_b)^-1,   C_i = eps0 eps_i A / d_i
        D   = C V / A
        E_i = D / (eps0 eps_i)

    and the potential at the interface is the capacitive divider
    ``phi(z=d_a) = C_b / (C_a + C_b) * V``.

    Returns ``(problem, expected)``. ``problem`` carries ``electrodes`` with the
    plate part labels ``"bottom"`` and ``"top"``.
    """
    # An electrode in this solver is a whole PART, so the plates must be
    # separate one-element-thick layers -- there is no way to hold a single
    # FACE at a potential. That is fine for the closed form: a conductor layer
    # adds no capacitance in series (its own field is zero), and its eps is
    # never assembled because an electrode's part is skipped.
    t_plate = min(d_a, d_b) / 8.0
    widths = np.array(
        [t_plate] + [d_a / n_a] * n_a + [d_b / n_b] * n_b + [t_plate]
    )
    zs = np.concatenate([[0.0], np.cumsum(widths)])
    xs = np.linspace(0.0, side, n_side + 1)
    points, tets = _box_tets(xs, xs, zs)
    cz = points[tets][:, :, 2].mean(axis=1)
    z_a_mid = t_plate + d_a
    region_of_tet = np.where(
        cz < t_plate, 0,
        np.where(cz < z_a_mid, 1,
                 np.where(cz < z_a_mid + d_b, 2, 3)),
    ).astype(np.int64)
    parts = ["bottom", "a", "b", "top"]
    region_of_tet = region_of_tet  # 0 bottom plate, 1 a, 2 b, 3 top plate

    area = side * side
    c_a = EPS0 * eps_a * area / d_a
    c_b = EPS0 * eps_b * area / d_b
    c_series = 1.0 / (1.0 / c_a + 1.0 / c_b)
    displacement = c_series * 1.0 / area

    expected = {
        "C": c_series,
        # The potential at the a/b interface, with bottom 0 V and top 1 V.
        # In series the charge is shared, so the voltage divides in proportion
        # to 1/C: the drop across a is (1/C_a)/((1/C_a)+(1/C_b)), which is
        # c_b/(c_a+c_b) = 0.862 here. Checked against the solver's profile,
        # which measures exactly 0.86206897 at the interface node plane.
        "phi_interface": c_b / (c_a + c_b),
        "z_interface": t_plate + d_a,
        "E_a": displacement / (EPS0 * eps_a),      # V/m inside layer a
        "E_b": displacement / (EPS0 * eps_b),
    }
    problem = {
        "points_m": points, "tets": tets, "parts": parts,
        "region_of_tet": region_of_tet,
        # A conductor has no meaningful eps, but eps_by_tet requires a value for
        # every part before its skip logic runs. 1.0 is honest: an electrode's
        # part is never assembled, so the number is unused, and a real
        # dielectric constant here would suggest it mattered.
        "eps_static": {"a": eps_a, "b": eps_b, "bottom": 1.0, "top": 1.0},
        "electrodes": {"bottom": 0.0, "top": 1.0},
        "d_a": d_a,
        "z_plate": t_plate,
        "side": side,
    }
    return problem, expected


# ---------------------------------------------------------------------------
# 2. Concentric spheres: closed-form C and phi(r)
# ---------------------------------------------------------------------------

def concentric_spheres(r_in=50e-6, r_mid=75e-6, r_out=100e-6,
                       eps_a=2.0, eps_b=5.0, n_r=16, n_theta=6, n_phi=12):
    """Two dielectric shells between a solid inner sphere and a hollow outer one.

    inner conductor | shell a (eps_a) | shell b (eps_b) | outer conductor

    The closed form is the spherical capacitance of shells in series:

        C_i = 4 pi eps0 eps_i / (1/r_inner_i - 1/r_outer_i)

    and the potential in each shell follows ``phi = A + B/r`` with A, B fixed by
    the two boundary potentials. For the innermost conductor held at V and the
    outermost at 0:

        phi(r) = V * (1/r_out - 1/r) / (1/r_out - 1/r_in)     (single shell)

    extended per shell by continuity of phi and of D = eps dphi/dr.

    MESH. A structured grid in spherical coordinates. Two choices that make it
    usable rather than approximate:

    * The theta nodes INCLUDE the poles (0 and pi). A grid that stops short of
      them leaves two conical holes, and with a natural boundary condition that
      is an open surface the flux escapes through -- measured as a capacitance
      that falls short by a few percent and does not close as the mesh refines.
    * Each conductor is a THIN SHELL of its own part, one radial cell thick,
      because an electrode is a whole part and there is no way to hold a face.
      The shells add no series capacitance (a conductor's field is zero), so the
      closed form above is unchanged.

    The tets are anisotropic near the poles; the bench's job is to show the
    measured answer still lands on the closed form. A failure here means the
    radial field is wrong, not that the mesh is pretty.

    Returns ``(problem, expected)`` with ``expected["phi_of_r"]`` a callable.
    """
    if not (r_in < r_mid < r_out):
        raise ValueError("need r_in < r_mid < r_out")

    t_shell = (r_out - r_in) / 40.0
    rs = np.concatenate([
        np.linspace(r_in, r_in + t_shell, 2),
        np.linspace(r_in + t_shell, r_mid, n_r // 2 + 1)[1:-1],
        np.linspace(r_mid, r_out - t_shell, n_r - n_r // 2 + 1),
        np.linspace(r_out - t_shell, r_out, 2)[1:],
    ])
    # Poles HELD OFF the axis by a small but finite amount. A true pole makes
    # every phi node coincide, which collapses the first and last theta bands
    # into 6-node wedges instead of 8-node hexahedra. Splitting a wedge with
    # the cube tets keeps only HALF its volume (measured: 93.7% of the shell),
    # and no 6-node tiling I tried reproduced the convex hull -- the wedge is
    # not convex. Holding the poles at a small theta sidesteps the whole class:
    # every cell stays a proper hexahedron, the surface still closes because
    # the first/last ring wraps in phi, and the volume error that remains is
    # the ordinary O(h^2) staircase, which converges under refinement instead
    # of sitting at a fixed 6.6%.
    t_pole = 1.0e-3                      # rad; ~0.06% of the radial extent
    thetas = np.concatenate([[t_pole],
                             np.linspace(t_pole, np.pi - t_pole, n_theta + 1)[1:-1],
                             [np.pi - t_pole]])
    phis = np.linspace(0.0, 2 * np.pi, n_phi + 1)[:-1]

    def node(i, j, k):
        return (i * (n_theta + 1) + j) * n_phi + k

    points = np.array([
        [rs[i] * np.sin(thetas[j]) * np.cos(phis[k]),
         rs[i] * np.sin(thetas[j]) * np.sin(phis[k]),
         rs[i] * np.cos(thetas[j])]
        for i in range(rs.size) for j in range(n_theta + 1) for k in range(n_phi)
    ])

    tets = []
    tet_layer = []
    for i in range(rs.size - 1):
        for j in range(n_theta):
            for k in range(n_phi):
                k2 = (k + 1) % n_phi
                # Every cell is a proper hexahedron: the polar rings are held
                # off the axis (see t_pole above), so no corner collapses.
                corner = {}
                for a in (0, 1):
                    for b in (0, 1):
                        for c in (0, 1):
                            corner[(a, b, c)] = node(i + a, j + b, k2 if c else k)
                for t in _CUBE_TETS:
                    bits = [tuple(int(v) for v in np.unravel_index(u, (2, 2, 2))) for u in t]
                    tets.append([corner[b] for b in bits])
                    tet_layer.append(i)
    tets = np.array(tets, dtype=np.int64)
    tet_layer = np.array(tet_layer, dtype=np.int64)

    # Drop the nodes the pole collapse left unreferenced. Allocating a full
    # n_phi ring at each pole and then pointing every cell at one of them
    # leaves the rest as isolated vertices, and an isolated free node has an
    # all-zero row in K_ff -- which is what made SuperLU report "Factor is
    # exactly singular" rather than any kind of geometric complaint.
    used = np.unique(tets)
    renumber = np.full(points.shape[0], -1, dtype=np.int64)
    renumber[used] = np.arange(used.size)
    points = points[used]
    tets = renumber[tets]

    # Assign by radial INDEX, not by radius: a tet's centroid sits inside its
    # own cell but not at its centre, so a radius test on a thin conductor
    # shell selects nothing at all (measured: region "outer" got 0 tets). The
    # index is recorded as each tet is built, so the polar degeneracy cannot
    # shift it.
    n_radial = rs.size - 1
    # 0 inner conductor, 1 shell a, 2 shell b, 3 outer conductor
    region_of_tet = np.where(
        tet_layer == 0, 0,
        np.where(tet_layer == n_radial - 1, 3,
                 np.where(rs[tet_layer] < r_mid, 1, 2)),
    ).astype(np.int64)
    parts = ["inner", "shell_a", "shell_b", "outer"]

    # The radii the mesh holds potentials at. Every reference below is built on
    # these, never on r_in / r_out: the conductors are thin SHELLS, so the
    # dielectric spans shell-face to shell-face. Using the idealised radii for
    # one reference and the meshed ones for another is what produced a 7.1%
    # "capacitance error" and a 3% "interface error" that were both pure book-
    # keeping -- each was flat under refinement, which is the tell.
    r_inner_shell = float(rs[1])          # outer face of the inner conductor
    r_outer_shell = float(rs[-2])         # inner face of the outer conductor

    # Closed form on the IDEALISED radii: what the fixture was specified to be,
    # reported but never scored (see electrostatics_bench).
    c_a = 4 * np.pi * EPS0 * eps_a / (1.0 / r_in - 1.0 / r_mid)
    c_b = 4 * np.pi * EPS0 * eps_b / (1.0 / r_mid - 1.0 / r_out)
    c_series = 1.0 / (1.0 / c_a + 1.0 / c_b)

    # Closed form on the MESHED radii: the quantity an FEM answer converges to,
    # and the one the bench scores against.
    c_a_m = 4 * np.pi * EPS0 * eps_a / (1.0 / r_inner_shell - 1.0 / r_mid)
    c_b_m = 4 * np.pi * EPS0 * eps_b / (1.0 / r_mid - 1.0 / r_outer_shell)
    c_series_meshed = 1.0 / (1.0 / c_a_m + 1.0 / c_b_m)

    # phi(r) with phi(r_in) = 1 and phi(r_out) = 0, piecewise A + B/r.
    def _coeffs(phi_lo, phi_hi, ra, rb):
        # phi = A + B/r, phi(ra) = phi_lo, phi(rb) = phi_hi
        b = (phi_hi - phi_lo) / (1.0 / rb - 1.0 / ra)
        a = phi_lo - b / ra
        return a, b

    # Voltage divides in proportion to 1/C, exactly as in the slab: the drop
    # ACROSS shell a is (1/C_a) / (1/C_a + 1/C_b), which equals c_b/(c_a+c_b).
    # That is the DROP, not the potential at the interface. With phi(r_in)=1
    # the interface sits one drop below it. An earlier version of this fixture
    # used the drop as the potential and so reported phi(r_mid)=0.794 where the
    # solver -- correctly -- gave 0.206; the sphere was blamed for a sign error
    # in the reference. Cross-checked against the uniform-eps case, where the
    # 1/r profile is exact, and the solver matched to 1e-4.
    # On the MESHED series, for the same reason the capacitance reference is.
    phi_rmid = 1.0 - c_b_m / (c_a_m + c_b_m)
    a_a, b_a = _coeffs(1.0, phi_rmid, r_inner_shell, r_mid)
    a_b, b_b = _coeffs(phi_rmid, 0.0, r_mid, r_outer_shell)

    def phi_of_r(r):
        r = np.asarray(r, dtype=np.float64)
        return np.where(r < r_mid, a_a + b_a / r, a_b + b_b / r)

    # The mesh's conductors are thin SHELLS, not the points r_in and r_out, so
    # the staircase answer differs from the idealised closed form by ~7% no
    # matter how fine the angular grid -- it does not converge away, because
    # the conductor surfaces are offset by t_shell by construction. Scoring a
    # solver against the idealised number would then report that 7% as solver
    # error. This second reference evaluates the same closed form at the radii
    # the mesh ACTUALLY holds potentials at, which is the right thing to
    # compare an FEM answer against; the gap between the two references is a
    # property of the fixture, not of the solver. Measured: the solver sits
    # 0.04% from this value and 7.1% from the idealised one.

    expected = {"C": c_series, "phi_of_r": phi_of_r, "r_mid": r_mid,
                "phi_rmid": phi_rmid,
                "C_meshed": c_series_meshed,
                "r_inner_shell": r_inner_shell,
                "r_outer_shell": r_outer_shell}
    problem = {
        "points_m": points, "tets": tets, "parts": parts,
        "region_of_tet": region_of_tet,
        "eps_static": {"inner": 1.0, "shell_a": eps_a, "shell_b": eps_b,
                       "outer": 1.0},
        "electrodes": {"inner": 1.0, "outer": 0.0},
        "r_in": r_in, "r_out": r_out,
    }
    return problem, expected


# ---------------------------------------------------------------------------
# 3. Sphere between parallel plates: no closed form
# ---------------------------------------------------------------------------

def sphere_between_plates(plate_gap=200e-6, plate_side=400e-6, radius=40e-6,
                          eps_sphere=4.0, eps_air=1.0, n_side=10, n_z=10):
    """A dielectric sphere suspended between two plates.

    NO CLOSED FORM. Laplace's equation with this geometry has no elementary
    solution, and the series solution is a Legendre expansion that converges
    slowly at the plate contact. The bench therefore reports the numbers and
    the mesh-to-mesh change, and labels them UNVALIDATED: a stable value is
    evidence of convergence, not of correctness.

    The two plates are the top and bottom faces, held 0 V and 1 V; the sphere is
    mid-gap. Because an electrode is a whole PART, each plate is its own thin
    conductor layer rather than a held face.

    The mesh is a structured box with the sphere approximated by whichever tets
    have their centroid inside the radius -- a staircased sphere, which is
    honest about what it is. The staircase is why this fixture cannot have a
    tight tolerance.
    """
    t_plate = plate_gap / 40.0
    # Each block supplies its SHARED endpoint only in the first block: the
    # middle linspace already ends at plate_gap - t_plate, so keeping both ends
    # of the last block emitted that plane twice. A duplicated z-plane makes
    # every tet in that layer exactly flat, and tet_geometry inverts J, so the
    # symptom is a bare LinAlgError("Singular matrix") far from the cause.
    zs = np.concatenate([
        np.linspace(0.0, t_plate, 2)[:-1],
        np.linspace(t_plate, plate_gap - t_plate, n_z + 1),
        np.linspace(plate_gap - t_plate, plate_gap, 2)[1:],
    ])
    xs = np.linspace(-plate_side / 2, plate_side / 2, n_side + 1)
    points, tets = _box_tets(xs, xs, zs)
    centres = points[tets].mean(axis=1)
    r = np.linalg.norm(centres - np.array([0.0, 0.0, plate_gap / 2]), axis=1)
    cz = centres[:, 2]
    region_of_tet = np.where(
        cz < t_plate, 0,
        np.where(cz > plate_gap - t_plate, 3, np.where(r < radius, 2, 1)),
    ).astype(np.int64)
    parts = ["bottom", "air", "sphere", "top"]
    problem = {
        "points_m": points, "tets": tets, "parts": parts,
        "region_of_tet": region_of_tet,
        # The staircase only approximates the sphere's volume, so reporting a
        # "known" answer here would be false. There is none.
        "eps_static": {"bottom": 1.0, "air": eps_air, "sphere": eps_sphere, "top": 1.0},
        "electrodes": {"bottom": 0.0, "top": 1.0},
        "n_side": n_side, "n_z": n_z, "plate_gap": plate_gap, "radius": radius,
    }
    return problem


# ---------------------------------------------------------------------------
# 4. Thin dielectric between thick ones, meshed by TetGen (task 1.6c)
# ---------------------------------------------------------------------------

#: The thin-layer capacitor's stack, bottom to top: (name, thickness_m, eps).
#: Plates are conductor layers (an electrode is a whole part, see fixture 1).
#: The thin layer has the LOW permittivity, so it carries a large share of the
#: voltage (about 40 % here) and a mesh that mis-assigns it cannot hide. The
#: values are bench contrasts, not materials.
THIN_CAP_STACK = (
    ("bottom", 20e-9, 1.0),
    ("thick_a", 200e-9, 10.0),
    ("thin", 10e-9, 2.0),
    ("thick_b", 200e-9, 10.0),
    ("top", 20e-9, 1.0),
)
#: Side of the square plates, metres. Small, so the thin layer's sub-slab
#: facets (elements about as wide as 10 nm / 3) stay a tier-1 mesh.
THIN_CAP_SIDE = 60e-9


def thin_layer_capacitor(edge_m: float, *, conforming: bool = True,
                         n_across: int | None = None):
    """A thin dielectric between two thick ones, meshed by TetGen, with its answer.

    The exact answer is the 1-D series capacitor, ``C = eps0 A / sum(d_i/eps_i)``
    over the dielectric layers, with ``E_i = V / (eps_i sum(d_j/eps_j))``. The
    sides are natural (zero normal field) and the plates cover the whole
    cross-section, which is exactly the assumption that makes the 1-D form exact.

    ``conforming=True`` meshes it with :func:`mse_meshing.tetrahedralize_layers`.
    ``conforming=False`` is the CONTROL: the SAME PLC with every internal facet
    removed (one closed solid, the 1.6b shape), meshed with the same switches,
    each tet labelled by the layer its centroid falls in. It exists so the
    conforming result is measured against what it replaces, not in a vacuum.
    """
    import mse_meshing as MESH

    if n_across is None:
        n_across = MESH.LAYER_ELEMENTS_ACROSS
    layers = [(name, th) for name, th, _eps in THIN_CAP_STACK]
    eps = {name: e for name, _th, e in THIN_CAP_STACK}
    side = THIN_CAP_SIDE
    outline = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    planes = MESH.layer_planes(layers)
    if conforming:
        points, tets, region = MESH.tetrahedralize_layers(
            outline, layers, edge_m, n_across=n_across)
    else:
        points, tets, region = _one_solid_centroid_labelled(
            MESH, outline, layers, edge_m, planes)

    dielectric = [(n, th) for n, th, _e in THIN_CAP_STACK if n not in ("bottom", "top")]
    s = sum(th / eps[n] for n, th in dielectric)
    area = side * side
    expected = {
        "C": EPS0 * area / s,
        "E": {n: 1.0 / (eps[n] * s) for n, _th in dielectric},
    }
    problem = {
        "points_m": points, "tets": tets, "parts": [n for n, _t in layers],
        "region_of_tet": np.asarray(region, dtype=np.int64),
        "eps_static": eps,
        "electrodes": {"bottom": 0.0, "top": 1.0},
        "planes_m": planes, "layers": layers, "side": side, "edge_m": edge_m,
    }
    return problem, expected


def _one_solid_centroid_labelled(MESH, outline, layers, edge_m, planes):
    """The 1.6b shape: the layer PLC without its internal facets, labelled by centroid."""
    import tetgen

    vertices, faces, _seeds = MESH.layer_stack_plc(outline, layers, edge_m, n_across=1)
    z = vertices[faces][:, :, 2]
    flat = np.all(np.abs(z - z[:, :1]) < 1e-12 * planes[-1], axis=1)
    interior = flat & (z[:, 0] > planes[0]) & (z[:, 0] < planes[-1])
    faces = faces[~interior]
    used = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    vertices, faces = vertices[used], remap[faces].astype(np.int32)
    offset, scale = MESH._canonical_frame(vertices)
    tet = tetgen.TetGen((vertices - offset) * scale, faces)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=2.0,
                       maxvolume=MESH.target_maxvolume(None, edge_m * scale),
                       fixedvolume=True)
    points = np.asarray(tet.node, dtype=float)[:, :3] / scale + offset
    tets = np.asarray(tet.elem, dtype=np.int64)
    cz = points[tets][:, :, 2].mean(axis=1)
    region = np.clip(np.searchsorted(planes, cz, side="right") - 1, 0, len(layers) - 1)
    return points, tets, region
