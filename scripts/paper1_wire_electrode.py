#!/usr/bin/env python3
"""Bench: paper 1's nanowire with a TOP ELECTRODE -- the electrostatic wire fixture.

**UNVALIDATED.** This fixture has no measured answer to be checked against. It
exists so that the field the gate applies to the paper-1 wire is a number we
computed on the paper's own geometry rather than a number we assumed. Nothing
downstream may quote its output as validated; see "Status" below.

What it builds
--------------
The same three parts as :mod:`paper1_nanowire` -- a gold mirror, a 12 nm SiO2
buffer and a hexagonal InP prism -- plus a top electrode that contacts the
wire's top facet. The geometry is imported from ``paper1_nanowire`` rather than
re-typed, so the two benches cannot drift apart: a change to the wire's
dimensions moves both.

Geometry, with its source in paper 1 (arXiv:2601.04440), as task 0.2 recorded it
---------------------------------------------------------------------------------
- **Wire:** wurtzite InP hexagonal prism, height H = 1375 nm, diameter
  D = 420 nm. "the optimal cavity dimensions (height of 1375 nm and width of
  420 nm) that we found" (p. 3, Results); the Fig. 1 caption agrees. We take
  these from p. 3 and NOT from the Fig. 5a caption, which swaps them
  ("D=1375 nm and H=420 nm", p. 8) and contradicts the rest of the paper.
- **The 1.14 factor** is the paper's own hex-to-cylinder relation: "A scaling
  factor of 1.14 was applied to the diameter as the nanowire has a hexagonal
  cross-section" (p. 10, S2). Applied as ``2s = 1.14 D`` -- s the hexagon's side
  -- it gives a corner-to-corner width of 478.8 nm, a circumradius of 239.4 nm
  and a flat-to-flat of 414.6 nm.
- **Oxide:** 12 nm SiO2 -- "a 12 nm thick oxide layer optimizes the Purcell
  enhancement and extraction efficiency" (p. 4), over a gold mirror.
- **Probe:** on-axis, 30 nm below the top -- "a QD emitter placed on-axis at
  30 nm below the top" (Fig. 2 caption, p. 5), at "the second antinode from the
  top" (p. 3).

Permittivities -- static, and NOT n^2
-------------------------------------
The paper states refractive indices and never a permittivity: "an index of
refraction n=3.44 ... to simulate the nanowire material (Wurtzite InP)" (p. 9,
S1). n^2 = 11.83 is a value at the optical band, not a static permittivity, so
it is used for NEITHER material here.

- **InP eps_s = 12.5** at 300 K (Ioffe, NSM archive, basic-parameters table;
  task 0.2, recorded in design.md).
- **SiO2 eps_s = 3.9** -- paper 1 gives no SiO2 value anywhere, so this is taken
  from a standard reference for thermal oxide. See :data:`EPS_SIO2` for the
  citation and its caveat.
- Gold is a conductor and is treated as an equipotential (it is an electrode,
  not a dielectric region).

Status: UNVALIDATED; the ONE-PART mesh does not resolve the oxide (the layered one does, 1.6c)
---------------------------------------------------------------------------------------------
There is no measurement of the field in this structure to compare against. The
paper reports a Purcell factor from an FDTD run, not a potential or a field.

Worse, and measured: **the field this fixture computes is not converged in the
mesh**, so it cannot be a gate for anything. Driving the only lever the one-part
mesher exposes (``n_modes``) from 20 to 120 changes ``|E|`` at the probe by
**40.1 %** and NOT monotonically (557, 780, 690 kV/m per volt). The cause is a
12 nm oxide inside a 1567 nm stack: every element must span both a 1375 nm wire
and a 12 nm layer, and the tet z-span only falls as roughly ``n^-1/3``. The
z-span spread is wide, so both ends of it matter: at ``n_modes=200`` (153k tets)
the MEDIAN element spans 20.4 nm in z while the WORST spans 51.7 nm (at
``n_modes=20`` the worst is 110.4 nm). Even the median is 1.7x the oxide and the
worst 4.3x, so resolution alone does not reach 12 nm; doing so would need on the
order of 10^6 tets, past tier 1 and past this rig's memory cap.

A single closed solid is used because it is the ONLY shape that reaches the
mesher: :class:`mse_parts.PartSet` refuses touching parts and requires one part
to enclose all others, and ``sizing=`` is refused on a one-part upload. So the
graded layered mesh this device needs has no path in the current mesher. Every
layer is labelled by the z of the tet's centroid, and at these resolutions tets
STRADDLE the layer boundaries (measured: at ``n_modes=20``, 1064 of 9775 tets,
10.9 %, cross z=120 nm or z=132 nm; a worst-case element has a 175 nm diagonal
spanning a 12 nm oxide), so the labels are applied to elements whose permittivity
is not uniform.
That, and not the physics, is what moves with resolution.

What this fixture therefore establishes, and what it does not:

- ESTABLISHED: the solver path works end to end on the paper's geometry, the
  result is exactly linear in the applied volts (max |dE| = 0 at 2V/2 vs 1V and
  at -1V vs 1V), and the prism's volume matches the closed form exactly.
- NOT ESTABLISHED: any field value. None may be quoted downstream.

**Task 1.6c (2026-11-05): the layered path.** :func:`mesh_fixture_layered`
meshes the stack with :func:`mse_meshing.tetrahedralize_layers`, which puts an
internal facet at every layer plane (and sub-slab facets in the oxide), so no
tet straddles an interface. On that mesh the field at the probe is the same to
8e-15 across 9.7k, 38k and 107k tets and equals the exact 1-D value
(:func:`field_per_volt_1d`, 707.483 kV/m per V) -- run ``--layered-convergence``.
That agreement is exact, not approximate, because this fixture IS a 1-D stack:
every layer has the wire's outline and both electrodes cover it, so the
potential is piecewise linear in z and lies in the P1 space. The 40 % spread
above was therefore entirely the centroid labels. The fixture is still
UNVALIDATED against the paper's device: the paper's pad is wider than the wire
(:func:`footprint_error`), and that 3-D fringing field is not modelled here.
:func:`mesh_fixture` (the one-part path) is kept, with its negative test, as
the record of what the one-part mesher does.

Usage
-----
    python scripts/paper1_wire_electrode.py            # layered mesh, prints E/V
    python scripts/paper1_wire_electrode.py --coarse   # a mesh small enough for tests
    python scripts/paper1_wire_electrode.py --layered-convergence   # task 1.6c check
    python scripts/paper1_wire_electrode.py --one-part --convergence  # the 1.6b negative

The script is also importable: :func:`mesh_fixture` and
:func:`wire_field_per_volt` are what ``tests/test_paper1_wire_electrode.py``
drives.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import trimesh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "dockerfiles", "maxwell_solver"))

import paper1_nanowire as P1  # noqa: E402

import dot_material as DOT  # noqa: E402
import mse_electrostatics as E  # noqa: E402
import mse_meshing as MESH  # noqa: E402
import mse_parts as PARTS  # noqa: E402

NM = 1e-9

#: The paper's own hex-to-cylinder relation, re-exported so a caller reads the
#: same constant the wire was built from.
HEX_NW_RATIO = P1.HEX_NW_RATIO

#: Wire dimensions, from paper 1 p. 3 (see the module docstring for why p. 3
#: and not the Fig. 5a caption).
HEIGHT_NM = 1375.0
DIAMETER_NM = 420.0

#: The oxide's thickness, from p. 4.
OXIDE_NM = 12.0

#: The probe's depth below the wire's top facet, from the Fig. 2 caption (p. 5).
PROBE_BELOW_TOP_NM = 30.0

#: Static permittivity of InP at 300 K, from Ioffe's basic-parameters table
#: (task 0.2). NOT n^2 = 11.83, which is the optical-band value. Read from
#: :mod:`dot_material`, which owns the cited value, rather than re-typed here.
EPS_INP = DOT.INP_EPS_S.value

#: Static permittivity of SiO2. **Paper 1 states no SiO2 value** ("no SiO2 index
#: or eps is stated anywhere", design.md §1), so this is a standard-reference
#: value for thermally grown oxide, eps_r ~ 3.9 (the widely quoted figure for
#: thermal SiO2, e.g. Sze, "Physics of Semiconductor Devices", 3rd ed., ch. 4;
#: also the ITRS-era standard value). It is NOT read from paper 1 and it is NOT
#: n^2 (~2.1 for fused silica at the optical band, which is the wrong quantity
#: anyway). A 3.9 vs 2.1 difference in the oxide barely moves the wire's field
#: -- the wire is 114x taller than the oxide -- but the value is a reference
#: number, not a measurement of this stack.
EPS_SIO2 = 3.9

#: How far the top electrode extends beyond the wire, in nm. It is a planar
#: contact over the wire's top facet, so it is as wide as the wire plus a pad;
#: the pad exists so the electrode is a real volume in the mesh rather than a
#: zero-thickness sheet, which TetGen cannot make a region of.
ELECTRODE_PAD_NM = 60.0

#: The electrode's own thickness, in nm.
ELECTRODE_THICKNESS_NM = 60.0

#: How far the gold mirror extends below the wire, in nm. It is the back
#: electrode and must cover the oxide's footprint.
MIRROR_THICKNESS_NM = 120.0

#: Parts that are electrodes: their nodes are held at the applied potential.
ELECTRODE_PARTS = ("mirror", "top")


def circumradius_nm() -> float:
    """The wire's circumradius from its diameter, through the paper's 1.14 factor.

    The paper writes ``2s = 1.14 D`` (p. 10, S2) with s the hexagon's side, so
    the corner-to-corner width is ``1.14 D`` and the circumradius is half that.
    ``paper1_nanowire.scene`` applies exactly this relation for ``--hex paper``;
    this function is that one line, named so the fixture can print it.
    """
    return DIAMETER_NM * HEX_NW_RATIO / 2.0


def materials() -> dict:
    """Part -> static relative permittivity, for the dielectric parts only.

    The electrodes are held at fixed volts, so their permittivity is never
    used; they carry none here and :func:`mse_electrostatics.solve` is given
    ``eps_static`` for every part anyway because the assembler asks for it.
    A conductor's value is 1.0 by convention in that call and is inert.
    """
    return {"mirror": 1.0, "oxide": EPS_SIO2, "wire": EPS_INP, "top": 1.0}


def probe_point() -> np.ndarray:
    """The probe, in metres: on the axis, 30 nm below the wire's top facet."""
    z_top = (MIRROR_THICKNESS_NM + OXIDE_NM + HEIGHT_NM) * NM
    return np.array([0.0, 0.0, z_top - PROBE_BELOW_TOP_NM * NM])


def stacked_solid_with_labels():
    """ONE closed solid for the whole dielectric stack, plus its layer labels.

    Why one solid and not four parts: :class:`mse_parts.PartSet` is the ONLY
    supported multi-part input, and :func:`mse_parts.load_parts` guarantees as a
    CONTRACT that no two parts intersect or touch and that one part ENCLOSES
    every other (the PEC wall). This stack violates both by construction: the
    layers share faces, and a 12 nm oxide cannot enclose a 1375 nm wire.
    :func:`mse_electrostatics.solve` does not need that contract -- it takes
    points, tets and a per-tet part array -- but a multi-part object can only
    reach the mesher through a ``PartSet``, so a stacked device cannot be
    expressed as parts here at all. A one-part solid can, so that is what this
    builds.

    Geometry: a hexagonal prism of circumradius :func:`circumradius_nm`, cut
    into four z-slabs by the layer boundaries. Every slab has the SAME hexagonal
    outline, so the solid is a prism and the mesh is exact: there is no
    cross-section step to triangulate and nothing to approximate.

    The electrode pad's margin (:data:`ELECTRODE_PAD_NM`) therefore does NOT
    widen the electrode -- it is folded into the outline of the whole stack
    instead, and :func:`footprint_error` reports how much area that adds around
    the wire. This is the one place the fixture departs from the paper's
    geometry, and it is reported rather than hidden. It changes the fringing
    field at the pad edge; it does not change which tets are "wire", because
    layers are separated by z, not by outline.

    Each tet is labelled by the layer its centroid falls in. The boundary
    planes are at known z, but **the mesh is not constrained to them**: this is
    a single solid with no internal facets, so TetGen is free to span an element
    across a boundary and the centroid then assigns the whole element to one
    layer. It does happen, and it is measured, not assumed -- at ``n_modes=20``
    about 10.9 % of tets straddle (see :func:`straddle_counts` and
    ``test_tets_straddle_layer_boundaries``). The label is therefore applied to
    elements whose permittivity is not uniform, which is why the field this
    fixture reports is not converged.
    """
    radius = circumradius_nm() * NM
    zs = [
        0.0,
        MIRROR_THICKNESS_NM * NM,
        (MIRROR_THICKNESS_NM + OXIDE_NM) * NM,
        (MIRROR_THICKNESS_NM + OXIDE_NM + HEIGHT_NM) * NM,
        (MIRROR_THICKNESS_NM + OXIDE_NM + HEIGHT_NM + ELECTRODE_THICKNESS_NM) * NM,
    ]
    labels = ["mirror", "oxide", "wire", "top"]
    assert len(zs) == len(labels) + 1

    hex_xy = [(radius * math.cos(math.pi / 6.0 + k * math.pi / 3.0),
               radius * math.sin(math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)]

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    rings = []
    for z in zs:
        ring = []
        for (x, y) in hex_xy:
            ring.append(len(verts))
            verts.append((x, y, z))
        rings.append(ring)

    # End caps: fan from vertex 0 of each ring. The bottom is wound to face -z
    # and the top to face +z.
    for k in range(1, len(rings[0]) - 1):
        faces.append((rings[0][0], rings[0][k + 1], rings[0][k]))
    for k in range(1, len(rings[-1]) - 1):
        faces.append((rings[-1][0], rings[-1][k], rings[-1][k + 1]))

    # Side walls: six quads per slab, each split into two triangles.
    for r in range(len(rings) - 1):
        lower, upper = rings[r], rings[r + 1]
        n = len(lower)
        for k in range(n):
            a, b = lower[k], lower[(k + 1) % n]
            c, d = upper[(k + 1) % n], upper[k]
            faces.append((a, b, c))
            faces.append((a, c, d))

    solid = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                            faces=np.asarray(faces, dtype=np.int64), process=False)

    def label_fn(points, tets) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        idx = np.asarray(tets)
        cent_z = pts[idx][:, :, 2].mean(axis=1)
        out = np.empty(cent_z.shape[0], dtype=object)
        for name, lo, hi in zip(labels, zs[:-1], zs[1:]):
            out[(cent_z >= lo) & (cent_z < hi)] = name
        # The top slab is closed at z_el_top, and a centroid can land exactly on
        # the last boundary only in a degenerate mesh; catch it rather than
        # leave it unlabelled.
        out[cent_z >= zs[-1]] = labels[-1]
        return out

    return solid, label_fn, zs, labels


def straddle_counts(mesh: dict) -> dict:
    """Measure how many tets cross a layer boundary, and by how much.

    The labelling in :func:`stacked_solid_with_labels` is per centroid, which
    is only exact if no element spans a boundary. This is the measurement that
    shows it does: an element "straddles" when its vertex z-values fall on both
    sides of a boundary plane.

    Returns the total, the straddling count and fraction, and per-boundary
    crossings, plus the widest z-span and the widest diagonal span overall.
    """
    if "planes_m" in mesh:
        zs = np.asarray(mesh["planes_m"], dtype=np.float64)
    else:
        # :func:`mesh_fixture` (the 1.6b lumpy fixture) predates planes_m and
        # carries only one outline, so its planes are the flat stack's.
        _s, _lf, zs, _l = stacked_solid_with_labels()
    pts = np.asarray(mesh["points_m"], dtype=np.float64)
    tets = np.asarray(mesh["tets"])
    z = pts[tets][:, :, 2]
    zlo, zhi = z.min(axis=1), z.max(axis=1)
    # Interior boundary planes only; the outer faces are not boundaries.
    inner = np.asarray(zs[1:-1], dtype=np.float64)
    straddles = (zlo[:, None] < inner[None, :]) & (zhi[:, None] > inner[None, :])
    n = int(tets.shape[0])
    n_straddle = int(np.any(straddles, axis=1).sum())
    span_z = zhi - zlo
    # Full diagonal extent of each element's bounding box.
    ext = pts[tets].max(axis=1) - pts[tets].min(axis=1)
    diag = np.linalg.norm(ext, axis=1)
    return {
        "n_tets": n,
        "n_straddle": n_straddle,
        "fraction": (n_straddle / n) if n else 0.0,
        "per_boundary_nm": {float(b / NM): int(straddles[:, i].sum())
                            for i, b in enumerate(inner)},
        "span_z_median_nm": float(np.median(span_z) / NM),
        "span_z_max_nm": float(span_z.max() / NM),
        "diag_max_nm": float(diag.max() / NM),
    }


def footprint_error():
    """Report the pad margin's effect on the outline area.

    The fixture gives every layer the wire's hexagonal outline, so the electrode
    is a hexagon of the wire's size rather than the paper's square pad with
    :data:`ELECTRODE_PAD_NM` of margin. This returns ``(hexagon_area,
    padded_square_area, ratio)`` -- the ratio is how much LARGER a square pad of
    that margin would have been. Reported so the departure is a number, not a
    silence.
    """
    radius = circumradius_nm() * NM
    pad = radius + ELECTRODE_PAD_NM * NM
    hexagon = 3 * math.sqrt(3) / 2 * radius ** 2
    square = (2 * pad) ** 2
    return hexagon, square, square / hexagon

def layers() -> list[tuple[str, float]]:
    """The stack as ``(name, thickness_m)`` bottom to top, for the layered mesher."""
    _solid, _label, zs, labels = stacked_solid_with_labels()
    return [(name, hi - lo) for name, lo, hi in zip(labels, zs[:-1], zs[1:])]


def hex_outline_m() -> list[tuple[float, float]]:
    """The wire's hexagon in metres, the same corners :func:`stacked_solid_with_labels` uses."""
    radius = circumradius_nm() * NM
    return [(radius * math.cos(math.pi / 6.0 + k * math.pi / 3.0),
             radius * math.sin(math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)]


def field_per_volt_1d() -> float:
    """The exact |E| in the wire per applied volt, for THIS fixture's geometry.

    Every layer has the same outline and both electrodes cover it, with natural
    (zero normal field) side walls, so the fixture is a 1-D series stack: the
    same displacement crosses oxide and wire and ``E_wire = V / (eps_wire *
    (t_ox/eps_ox + H/eps_wire))``. That is a property of the fixture's
    simplification (see :func:`footprint_error`), not of the paper's device,
    which has a pad wider than the wire and a 3-D fringing field.
    """
    s = OXIDE_NM * NM / EPS_SIO2 + HEIGHT_NM * NM / EPS_INP
    return 1.0 / (EPS_INP * s)


def mesh_fixture_layered(*, edge_nm: float = 200.0, n_across: int | None = None,
                         max_tets: int = 400_000, verbose: bool = False):
    """The stack meshed as CONFORMING layers (task 1.6c), same dict as :func:`mesh_fixture`.

    Every layer plane is an internal facet, so no tet straddles an interface
    and ``region_of_tet`` is TetGen's own region attribute, not a centroid
    guess; the oxide is split into ``n_across`` sub-slabs so it has at least
    that many elements across its 12 nm.
    """
    kw = {} if n_across is None else {"n_across": n_across}
    stack = layers()
    points_m, tets, region_of_tet = MESH.tetrahedralize_layers(
        hex_outline_m(), stack, edge_nm * NM, max_tets=max_tets,
        verbose=verbose, **kw)
    return {
        "points_m": points_m,
        "tets": tets,
        "parts": [name for name, _t in stack],
        "region_of_tet": region_of_tet,
        "eps_static": materials(),
        "probe": probe_point(),
        "planes_m": MESH.layer_planes(stack),
        "layers": stack,
    }


#: The layered convergence ladder: (bulk edge nm, oxide sub-slabs), coarse to
#: fine. Both levers refine together, the wire's edge and the oxide's.
LAYERED_LADDER = ((400.0, 1), (200.0, 2), (100.0, 3))


def pad_outline_m() -> list[tuple[float, float]]:
    """The ELECTRODE outline in metres: the wire's hexagon grown by the pad margin.

    Same 6-fold shape as :func:`hex_outline_m`, so the two rings nest exactly
    (same angles, larger radius) and the interface between them is an annulus.
    """
    radius = (circumradius_nm() + ELECTRODE_PAD_NM) * NM
    return [(radius * math.cos(math.pi / 6.0 + k * math.pi / 3.0),
             radius * math.sin(math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)]


def pad_outlines_m() -> list[list[tuple[float, float]]]:
    """One outline per layer for the pad-WIDENED stack (task 1.10c).

    Paper 1's device is a sandwich: gold mirror below, wire, top electrode
    above, with BOTH electrodes wider than the wire. That is a
    wide-narrow-wide stack, which 1.10a's monotonic-narrowing rule refused
    outright -- so the paper's own geometry could not be meshed at all until
    1.10c generalised the interface code.

    The electrodes get :func:`pad_outline_m`; the oxide and the wire keep the
    wire's own hexagon, because the oxide is grown ON the wire.
    """
    wire = hex_outline_m()
    pad = pad_outline_m()
    return {"mirror": pad, "oxide": wire, "wire": wire, "top": pad}


def mesh_fixture_padded(*, edge_nm: float = 200.0, n_across: int | None = None,
                        max_tets: int = 400_000, verbose: bool = False):
    """The stack meshed with the paper's REAL footprint (task 1.10c).

    Same contract as :func:`mesh_fixture_layered`, but each layer carries its
    own outline, so the electrodes are wider than the wire.

    The geometry is now the paper's real footprint, but the solve is NOT yet a
    3-D fringing field, and this docstring claimed it was. Measured (task
    1.10c, 2026-11-06): the free nodes stop at the wire's own rim, r = 239.4 nm,
    because no ambient medium is meshed -- the pad overhang is empty space with
    nothing to carry a field. The solved potential therefore varies by ~1e-16
    across x within a z-level and the padded solve reproduces
    :func:`field_per_volt_1d`'s series value. Pinned by
    ``test_the_padded_solve_is_still_one_dimensional``, which fails the moment
    the ambient is meshed. Until then this fixture buys the correct GEOMETRY,
    not a fringing field.
    """
    kw = {} if n_across is None else {"n_across": n_across}
    stack = layers()
    by_name = pad_outlines_m()
    outlines = [by_name[name] for name, _t in stack]
    points_m, tets, region_of_tet = MESH.tetrahedralize_per_region(
        outlines, stack, edge_nm * NM, max_tets=max_tets, verbose=verbose, **kw)
    return {
        "points_m": points_m,
        "tets": tets,
        "parts": [name for name, _t in stack],
        "region_of_tet": region_of_tet,
        "eps_static": materials(),
        "probe": probe_point(),
        "planes_m": MESH.layer_planes(stack),
        "layers": stack,
        "outlines": outlines,
    }


def mesh_fixture(*, max_tets: int = 60_000, n_modes: int = 4, verbose: bool = False):
    """Tetrahedralise the stack and return the solver's own arguments.

    Returns a dict of exactly the keyword arguments :func:`solve_wire` passes to
    :func:`mse_electrostatics.solve`, plus ``probe``.

    ``n_index=1.0`` because the electrical solver has no refractive index and
    the optical edge rule must not drive this mesh.
    """
    solid, label_fn, _zs, labels = stacked_solid_with_labels()
    points_m, tets, _ = MESH.tetrahedralize(
        solid, "m", n_modes=n_modes, n_index=1.0,
        max_tets=max_tets, verbose=verbose,
    )
    # ``solve`` wants a per-tet integer region id, not a name: part names live in
    # ``parts`` and ``region_of_tet`` indexes into it.
    named = label_fn(points_m, tets)
    index_of = {name: i for i, name in enumerate(labels)}
    region_of_tet = np.asarray([index_of[n] for n in named], dtype=np.int64)
    return {
        "points_m": points_m,
        "tets": tets,
        "parts": list(labels),
        "region_of_tet": region_of_tet,
        "eps_static": materials(),
        "probe": probe_point(),
    }


def solve_wire(mesh: dict, volts: float, *, method: str = "auto", budget=None) -> dict:
    """Solve the fixture with ``top`` at ``volts`` and the mirror at 0 V.

    The mirror is the back electrode and is held at 0; the top electrode takes
    the applied potential. That is the gate configuration design.md describes
    ("the gold mirror is the back electrode ... plus a top electrode. The field
    at the dot is set by V_G").
    """
    electrodes = {name: 0.0 for name in ELECTRODE_PARTS if name in mesh["parts"]}
    if "top" in electrodes:
        electrodes["top"] = volts
    return E.solve(
        mesh["points_m"], mesh["tets"], mesh["parts"], mesh["region_of_tet"],
        mesh["eps_static"], electrodes, method=method, budget=budget,
    )


def wire_field_per_volt(mesh: dict, *, method: str = "auto", budget=None) -> dict:
    """The field at the probe, per applied volt, in V/m per V.

    Solved once at 1 V and divided by 1 V: the problem is linear, and solving at
    1 V keeps the number a pure geometry-and-permittivity quantity that does not
    depend on the voltage chosen. :func:`field_report` checks the linearity
    separately rather than assuming it.

    Returns ``field_per_volt`` (the vector, V/m per V), ``magnitude_per_volt``,
    ``component_z_per_volt`` and the solve's own diagnostics.
    """
    sol = solve_wire(mesh, 1.0, method=method, budget=budget)
    field = E.field_at(sol, mesh["probe"])
    return {
        "field_per_volt": field,
        "magnitude_per_volt": float(np.linalg.norm(field)),
        "component_z_per_volt": float(field[2]),
        "method": sol["method"],
        "n_free": int(sol["n_free"]),
        "residual": sol["residual"],
    }


def field_report(mesh: dict, *, method: str = "auto", budget=None) -> dict:
    """The per-volt field plus the two linearity checks that make it trustworthy.

    A single solve at 1 V is only meaningful if the operator really is linear in
    the boundary values. This solves at 2 V and at -1 V and asserts the ratios,
    so the report carries its own evidence rather than asking the reader to
    assume linearity.
    """
    base = wire_field_per_volt(mesh, method=method, budget=budget)
    two = E.field_at(solve_wire(mesh, 2.0, method=method, budget=budget),
                     mesh["probe"]) / 2.0
    neg = E.field_at(solve_wire(mesh, -1.0, method=method, budget=budget),
                     mesh["probe"]) / -1.0
    out = dict(base)
    out["linearity"] = {
        "two_v_over_two_vs_one_v": float(np.max(np.abs(two - base["field_per_volt"]))),
        "minus_one_v_vs_one_v": float(np.max(np.abs(neg - base["field_per_volt"]))),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coarse", action="store_true",
                    help="a mesh small enough for a tier-1 test")
    ap.add_argument("--max-tets", type=int, default=None)
    ap.add_argument("--method", default="auto", choices=sorted(E.METHODS))
    ap.add_argument("--convergence", action="store_true",
                    help="run at n_modes 20/60/120 and report the spread (the "
                         "number that says whether any value here is usable)")
    ap.add_argument("--one-part", action="store_true",
                    help="mesh with the old one-part, centroid-labelled path (1.6b)")
    ap.add_argument("--layered-convergence", action="store_true",
                    help="the same check on the conforming layered mesh (task 1.6c)")
    ap.add_argument("--budget", type=float, default=None,
                    help="memory budget in GiB; refused before assembly if over")
    args = ap.parse_args(argv)

    max_tets = args.max_tets if args.max_tets is not None else (8_000 if args.coarse else 60_000)
    budget = None if args.budget is None else int(args.budget * 2**30)

    print(f"UNVALIDATED fixture -- paper 1's wire with a top electrode")
    print(f"  wire    H = {HEIGHT_NM:.0f} nm, D = {DIAMETER_NM:.0f} nm "
          f"(paper 1 p. 3); circumradius {circumradius_nm():.1f} nm via 2s = 1.14 D")
    print(f"  oxide   {OXIDE_NM:.0f} nm SiO2, eps_s = {EPS_SIO2}")
    print(f"  wire    eps_s = {EPS_INP} (Ioffe, 300 K) -- NOT n^2 = 11.83")
    print(f"  probe   on axis, {PROBE_BELOW_TOP_NM:.0f} nm below the top (Fig. 2 caption)")

    if args.layered_convergence:
        # Task 1.6c: the same check on the CONFORMING layered mesh.
        print()
        print("  convergence on the LAYERED mesh (internal facets at every plane):")
        exact = field_per_volt_1d()
        values = []
        for edge_nm, k in LAYERED_LADDER:
            m = mesh_fixture_layered(edge_nm=edge_nm, n_across=k)
            st = MESH.layer_straddles(m["points_m"], m["tets"], m["region_of_tet"],
                                      m["planes_m"])
            r = wire_field_per_volt(m, method=args.method, budget=budget)
            values.append(r["magnitude_per_volt"])
            print(f"    edge {edge_nm:5.0f} nm, oxide x{k}: {len(m['tets']):7d} tets, "
                  f"straddling {st['crossing']}  |E| = "
                  f"{r['magnitude_per_volt']:.9g} V/m per V  "
                  f"(rel to 1-D {r['magnitude_per_volt'] / exact - 1:+.2e})")
        lo, hi = min(values), max(values)
        print(f"    spread {(hi - lo) / lo:.3e}; exact 1-D value for this fixture's "
              f"geometry {exact:.9g} V/m per V")
        return 0

    if args.convergence:
        # The measured headline of this fixture: the answer moves with the mesh,
        # and not monotonically, because a 12 nm oxide inside a 1567 nm stack
        # cannot be resolved by a one-part mesh. Printed as a RANGE so nobody
        # can lift a single figure out of it.
        print()
        print("  convergence in n_modes (the only lever a one-part mesh has):")
        values = []
        for nm in (20, 60, 120):
            m = mesh_fixture(max_tets=400_000, n_modes=nm)
            r = wire_field_per_volt(m, method=args.method, budget=budget)
            values.append(r["magnitude_per_volt"])
            print(f"    n_modes {nm:4d}: {len(np.asarray(m['tets'])):7d} tets  "
                  f"|E| = {r['magnitude_per_volt']:.6g} V/m per V")
        lo, hi = min(values), max(values)
        print(f"    spread {(hi - lo) / lo:.1%}  -- NOT converged, so no single "
              f"value above is quotable")
        return 0

    if args.one_part:
        mesh = mesh_fixture(max_tets=max_tets)
    else:
        mesh = mesh_fixture_layered(edge_nm=400.0 if args.coarse else 200.0)
    n_tets = int(np.asarray(mesh["tets"]).shape[0])
    print(f"  mesh    {int(np.asarray(mesh['points_m']).shape[0])} nodes, {n_tets} tets")

    rep = field_report(mesh, method=args.method, budget=budget)
    print()
    print(f"  E at the probe, per volt:  |E| = {rep['magnitude_per_volt']:.6g} V/m per V")
    print(f"                             Ez  = {rep['component_z_per_volt']:.6g} V/m per V")
    print(f"  solver: {rep['method']}, {rep['n_free']} free nodes")
    lin = rep["linearity"]
    print(f"  linearity: 2V/2 vs 1V max|dE| = {lin['two_v_over_two_vs_one_v']:.3g}; "
          f"-1V vs 1V = {lin['minus_one_v_vs_one_v']:.3g}")
    print()
    if args.one_part:
        print("  NOT CONVERGED: this one-part mesh cannot resolve the 12 nm oxide, and the")
        print("  value above moves ~40% across mesh sizes (run --convergence). Do not quote it.")
    else:
        print(f"  layered mesh: converged (run --layered-convergence); exact 1-D value "
              f"{field_per_volt_1d():.9g} V/m per V")
        print("  UNVALIDATED against the paper's device: every layer has the wire's")
        print("  outline (footprint_error), so this is a 1-D stack, not the 3-D pad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
