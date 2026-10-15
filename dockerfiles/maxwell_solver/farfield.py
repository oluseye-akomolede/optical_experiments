"""Near-field to far-field transform and collection efficiency at one frequency.

WHAT THIS DOES. Given the tangential E and H on a box around an emitter,
recorded as complex phasors at one frequency, it returns

* the far-field radiation intensity U(theta, phi) in the upper half-space
  (W/sr), and the pattern U / max U;
* the fraction of the upward power inside a cone of numerical aperture NA,
  with NA = sin(theta_max) in air;
* the total power leaving the box, from the surface Poynting flux.

It is post-processing: nothing here runs a solver. ``fdtd.simulate`` records
no phasor box yet; when a field monitor does, it feeds the layout below.

THE TRANSFORM. Love's surface equivalence (Balanis, *Antenna Theory*, 3rd ed.,
sections 12.2-12.3; Harrington, *Time-Harmonic Electromagnetic Fields*,
section 3-5). With the outward normal n and the fields just outside the
surface S, the equivalent currents

    J_s = n x H          M_s = -n x E

radiating in FREE SPACE reproduce the true field everywhere outside S and
zero inside. In the far zone, with the ``exp(-i w t)`` convention this
solver uses (``fdtd.Metal``), an outgoing wave is ``exp(+ikr)/r`` and

    N(r^) = int_S J_s exp(-i k r^.r') dS'      L(r^) = int_S M_s exp(-i k r^.r')
    E_theta = i k/(4 pi) (eta N_theta + L_phi) exp(ikr)/r
    E_phi   = i k/(4 pi) (eta N_phi - L_theta) exp(ikr)/r
    U = r^2 |E|^2 / (2 eta)

These are Balanis eqs. 12-10 with every ``j`` conjugated. The surface is
integrated by the midpoint rule: each sample is the field at the centre of a
rectangular cell and carries that cell's area, so the error is O(h^2) in the
sample spacing h. The medium outside the box is vacuum (air): k = 2 pi f / c.

INPUT ARRAY LAYOUT (the contract a field monitor has to meet). The surface
is a list of ``Face`` objects, one per box face, each with:

    normal    one of fdtd.DIRECTIONS: "+x", "-x", "+y", "-y", "+z", "-z",
              the OUTWARD normal of this face of the box
    position  the face's coordinate along its normal axis, metres, in the
              same global frame as the sample positions
    u_edges   (n_u + 1,) and  v_edges  (n_v + 1,): the cell edges along the
              two in-plane axes, metres, global frame, strictly increasing.
              (u, v) are the two non-normal axes IN INCREASING AXIS ORDER:
              "+-x" -> (y, z),  "+-y" -> (x, z),  "+-z" -> (x, y),
              which is what slicing a (nx, ny, nz) field array at one index
              of the normal axis leaves behind.
    E, H      (3, n_u, n_v) complex: the GLOBAL Cartesian components
              (Ex, Ey, Ez) and (Hx, Hy, Hz), V/m and A/m, at the centre of
              each cell, i.e. at (u_edges[i] + u_edges[i+1]) / 2 and the
              same in v. E and H are colocated: a Yee-grid monitor must
              interpolate its staggered components to the cell-face centres
              before building a Face. The normal components are accepted
              and ignored (only n x E and n x H enter).

``sample_box`` builds this layout from any callable ``fields(points)`` and
is how the tests feed analytic dipoles in; a monitor can build it the same
way from its recorded arrays. ``surface_power`` needs the same layout.

A SURFACE OPEN AT THE BOTTOM (the mirror case). A box with no "-z" face,
for an emitter on a mirror: only the top and the four sides are integrated,
still with the free-space Green's function. What that approximates: the
equivalence theorem reproduces outside S the field of the sources INSIDE S
only (the extinction theorem removes every exterior source). The mirror's
induced current is a source; the part of it inside the box (the box's
bottom must then be inside the metal, below a few skin depths, where the
field is negligible: 27 nm for gold at 900 nm from
``fdtd.permittivity("Au", f)``) is accounted for, and the part of the mirror
OUTSIDE the box is not. The transform therefore sees the direct wave plus
the reflection from the patch of mirror under the box, and nothing of the
reflection from the rest of the mirror. That is a large error unless the
box is laterally much wider than the Fresnel zones of the reflection,
measured (tests/test_farfield.py, a vertical dipole 0.3 wavelengths above a
perfect mirror): in a box 1 wavelength wide the upward far-field power is
43% below the flux through the faces and the pattern is 0.67 of its peak
off; at 8 wavelengths wide the power is still 15% low. The dropped "-z"
face itself contributes nothing only where the field on it is negligible;
a bottom face in the dielectric above the mirror is simply missing part
of S. Prefer ``pec_mirror_z`` for a metal mirror.

``pec_mirror_z`` is exact for a PERFECT conductor: the box's bottom is
placed on the mirror plane z = pec_mirror_z, and the included faces radiate
together with their images (Balanis section 4.7.2: a horizontal electric
current images with the opposite sign, a vertical one with the same;
magnetic currents the other way round). Image theory makes the half-space
field of the box currents equal to the free-space field of the currents
plus their images, so the remaining approximation for gold at 900 nm is
treating its reflection as that of a perfect conductor (|r| = 0.98 at
normal incidence rather than 1, and a reflection phase that is not exactly
pi), plus any thin layer on the mirror outside the box (the paper's 12 nm
SiO2) being ignored there. The "-z" face on the plane itself is then
redundant: its tangential E is zero there and its J cancels its own image,
which is why it may be omitted (measured: including it changes the far
field by 1e-15).

REFERENCE FIELDS. ``hertzian_dipole`` is the exact field of an electric
dipole in vacuum (Jackson, *Classical Electrodynamics*, 3rd ed., eq. 9.18,
SI, ``exp(-i w t)``) and ``dipole_power`` its radiated power (eq. 9.24):
the closed forms the tests check the transform against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fdtd import C0, DIRECTIONS, EPS0, ETA0

#: Directions per chunk of the phase matrix: 256 directions x 10,000 samples
#: x 16 bytes is 41 MB, so the transform never holds a large matrix.
DIRECTION_CHUNK = 256

#: Gauss-Legendre nodes in cos(theta) and uniform nodes in phi for the
#: hemisphere integrals. The far field of a box of half-size a is band-limited
#: to about k a harmonics in angle, so 32 x 64 is exact for a box up to several
#: wavelengths across (test_farfield.py: the full-sphere integral matches the
#: Poynting flux to 0.1%, which is the surface sampling's error, not the
#: quadrature's).
DEFAULT_N_THETA = 32
DEFAULT_N_PHI = 64

#: The two in-plane axes of a face, in increasing axis order, by normal axis.
_IN_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


class FarFieldError(ValueError):
    """A surface that does not meet the layout above, or a bad parameter."""


def _parse_normal(normal: str) -> tuple[int, int]:
    """``"+x"`` -> (axis 0, sign +1)."""
    if normal not in DIRECTIONS:
        raise FarFieldError(f"a face normal must be one of {DIRECTIONS}, got {normal!r}")
    return "xyz".index(normal[1]), (1 if normal[0] == "+" else -1)


def _edges(values, name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise FarFieldError(f"{name} must be a 1-D array of at least 2 edges, got shape {edges.shape}")
    if not np.all(np.diff(edges) > 0):
        raise FarFieldError(f"{name} must be strictly increasing")
    return edges


@dataclass(frozen=True)
class Face:
    """One face of the box, in the layout the module docstring defines."""

    normal: str
    position: float
    u_edges: np.ndarray
    v_edges: np.ndarray
    E: np.ndarray
    H: np.ndarray

    def __post_init__(self):
        _parse_normal(self.normal)
        u = _edges(self.u_edges, "u_edges")
        v = _edges(self.v_edges, "v_edges")
        shape = (3, u.size - 1, v.size - 1)
        e = np.asarray(self.E, dtype=np.complex128)
        h = np.asarray(self.H, dtype=np.complex128)
        for name, arr in (("E", e), ("H", h)):
            if arr.shape != shape:
                raise FarFieldError(
                    f"face {self.normal!r}: {name} must have shape {shape} "
                    f"(3 components, one sample per cell), got {arr.shape}")
        object.__setattr__(self, "u_edges", u)
        object.__setattr__(self, "v_edges", v)
        object.__setattr__(self, "E", e)
        object.__setattr__(self, "H", h)
        object.__setattr__(self, "position", float(self.position))

    @property
    def axis(self) -> int:
        return _parse_normal(self.normal)[0]

    @property
    def unit_normal(self) -> np.ndarray:
        axis, sign = _parse_normal(self.normal)
        n = np.zeros(3)
        n[axis] = sign
        return n

    def points(self) -> np.ndarray:
        """(n_u, n_v, 3) global positions of the cell centres."""
        return face_points(self.normal, self.position, self.u_edges, self.v_edges)

    def areas(self) -> np.ndarray:
        """(n_u, n_v) cell areas."""
        return np.diff(self.u_edges)[:, None] * np.diff(self.v_edges)[None, :]


def face_points(normal: str, position: float, u_edges, v_edges) -> np.ndarray:
    """(n_u, n_v, 3) cell-centre positions of a face in the global frame."""
    axis, _sign = _parse_normal(normal)
    u = _edges(u_edges, "u_edges")
    v = _edges(v_edges, "v_edges")
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:-1] + v[1:])
    ua, va = _IN_PLANE[axis]
    pts = np.empty((uc.size, vc.size, 3))
    pts[..., axis] = position
    pts[..., ua] = uc[:, None]
    pts[..., va] = vc[None, :]
    return pts


def sample_box(low, high, cells, fields, *, faces=DIRECTIONS) -> list:
    """Build the surface of the box ``low..high`` by sampling ``fields``.

    ``cells`` is ``(nx, ny, nz)``, the number of uniform cells per axis;
    ``fields(points)`` takes ``(..., 3)`` positions and returns ``(E, H)``
    of the same leading shape plus a trailing Cartesian axis. ``faces`` picks
    which of the six faces to build (drop ``"-z"`` for the mirror case).
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    cells = tuple(int(c) for c in cells)
    if low.shape != (3,) or high.shape != (3,) or len(cells) != 3:
        raise FarFieldError("sample_box needs low (3,), high (3,) and cells (3,)")
    if not np.all(high > low) or min(cells) < 1:
        raise FarFieldError(f"the box {low}..{high} with {cells} cells is empty on some axis")
    edges = [np.linspace(low[a], high[a], cells[a] + 1) for a in range(3)]
    out = []
    for normal in faces:
        axis, sign = _parse_normal(normal)
        ua, va = _IN_PLANE[axis]
        position = high[axis] if sign > 0 else low[axis]
        pts = face_points(normal, position, edges[ua], edges[va])
        e, h = fields(pts)
        out.append(Face(normal, position, edges[ua], edges[va],
                        np.moveaxis(np.asarray(e), -1, 0), np.moveaxis(np.asarray(h), -1, 0)))
    return out


def _check_faces(faces) -> list:
    faces = list(faces)
    if not faces:
        raise FarFieldError("the surface has no faces")
    for face in faces:
        if not isinstance(face, Face):
            raise FarFieldError(f"every face must be a Face, got {type(face).__name__}")
    return faces


# --------------------------------------------------------------------------
# (c) total power through the surface
# --------------------------------------------------------------------------

def surface_power(faces) -> float:
    """Time-averaged power leaving through the faces: 1/2 Re int (E x H*) . n dS, in W.

    For a closed box in a lossless medium this is the total radiated power.
    For a surface open at the bottom it is the power through the included
    faces only (on a perfect mirror, all the power that reaches the far zone).
    """
    total = 0.0
    for face in _check_faces(faces):
        poynting = 0.5 * np.real(np.cross(face.E, np.conj(face.H), axis=0))  # (3, n_u, n_v)
        flux = np.tensordot(face.unit_normal, poynting, axes=(0, 0))          # (n_u, n_v)
        total += float(np.sum(flux * face.areas()))
    return total


# --------------------------------------------------------------------------
# (a) far field
# --------------------------------------------------------------------------

def _equivalent_currents(faces, pec_mirror_z):
    """Flattened ``(points (n, 3), J (n, 3), M (n, 3), dS (n,))`` over all faces.

    With ``pec_mirror_z`` set, every sample is followed by its image in the
    plane z = pec_mirror_z, so the free-space transform of the result is the
    half-space field above a perfect conductor.
    """
    pts, js, ms, ds = [], [], [], []
    for face in faces:
        n = face.unit_normal
        e = np.moveaxis(face.E, 0, -1).reshape(-1, 3)
        h = np.moveaxis(face.H, 0, -1).reshape(-1, 3)
        pts.append(face.points().reshape(-1, 3))
        js.append(np.cross(n, h))           # J_s = n x H
        ms.append(-np.cross(n, e))          # M_s = -n x E
        ds.append(face.areas().reshape(-1))
    points = np.concatenate(pts)
    J = np.concatenate(js)
    M = np.concatenate(ms)
    dS = np.concatenate(ds)
    if pec_mirror_z is not None:
        z_m = float(pec_mirror_z)
        tolerance = 1e-9 * (float(np.max(np.abs(points))) + abs(z_m))
        if np.any(points[:, 2] < z_m - tolerance):
            raise FarFieldError(
                f"pec_mirror_z={z_m}: every sample must lie on or above the mirror plane")
        image = points.copy()
        image[:, 2] = 2.0 * z_m - points[:, 2]
        J_img = J * np.array([-1.0, -1.0, 1.0])
        M_img = M * np.array([1.0, 1.0, -1.0])
        points = np.concatenate([points, image])
        J = np.concatenate([J, J_img])
        M = np.concatenate([M, M_img])
        dS = np.concatenate([dS, dS])
    return points, J, M, dS


def _direction_frame(theta, phi):
    """``(r^, theta^, phi^)`` each ``(n_theta, n_phi, 3)`` on the angle grid."""
    t, p = np.meshgrid(np.asarray(theta, dtype=np.float64),
                       np.asarray(phi, dtype=np.float64), indexing="ij")
    st, ct = np.sin(t), np.cos(t)
    sp, cp = np.sin(p), np.cos(p)
    r_hat = np.stack([st * cp, st * sp, ct], axis=-1)
    t_hat = np.stack([ct * cp, ct * sp, -st], axis=-1)
    p_hat = np.stack([-sp, cp, np.zeros_like(t)], axis=-1)
    return r_hat, t_hat, p_hat


def _radiation_vectors(points, J, M, dS, k, r_hat):
    """``N`` and ``L`` (n_dir, 3) for the unit directions ``r_hat`` (n_dir, 3)."""
    jw = J * dS[:, None]
    mw = M * dS[:, None]
    N = np.empty((r_hat.shape[0], 3), dtype=np.complex128)
    L = np.empty_like(N)
    for start in range(0, r_hat.shape[0], DIRECTION_CHUNK):
        rh = r_hat[start:start + DIRECTION_CHUNK]
        phase = np.exp(-1j * k * (rh @ points.T))         # (chunk, n)
        N[start:start + DIRECTION_CHUNK] = phase @ jw
        L[start:start + DIRECTION_CHUNK] = phase @ mw
    return N, L


@dataclass(frozen=True)
class FarField:
    """The far field on an angle grid.

    ``E_theta`` and ``E_phi`` are the complex amplitudes with ``exp(ikr)/r``
    stripped (units V): the field at distance r is ``E_theta exp(ikr)/r``.
    ``intensity`` is the radiation intensity U in W/sr; ``pattern`` is
    U / max U.
    """

    frequency: float
    theta: np.ndarray
    phi: np.ndarray
    E_theta: np.ndarray
    E_phi: np.ndarray

    @property
    def intensity(self) -> np.ndarray:
        return (np.abs(self.E_theta) ** 2 + np.abs(self.E_phi) ** 2) / (2.0 * ETA0)

    @property
    def pattern(self) -> np.ndarray:
        u = self.intensity
        peak = float(u.max())
        if peak <= 0.0:
            raise FarFieldError("the far field is identically zero; nothing to normalise")
        return u / peak


def far_field(faces, frequency: float, theta, phi, *, pec_mirror_z=None) -> FarField:
    """The far field of the surface currents on ``faces`` at ``frequency`` (Hz).

    ``theta`` (n_theta,) and ``phi`` (n_phi,) in radians define the grid;
    ``theta`` is measured from +z, ``phi`` from +x. Only the upper half-space
    ``theta <= pi/2`` is meaningful for a surface open at the bottom or with
    ``pec_mirror_z`` set; a closed box in free space is valid on the full sphere.
    """
    faces = _check_faces(faces)
    frequency = float(frequency)
    if not frequency > 0.0:
        raise FarFieldError(f"frequency must be positive, got {frequency}")
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    if theta.ndim != 1 or phi.ndim != 1:
        raise FarFieldError("theta and phi must be 1-D arrays")
    k = 2.0 * math.pi * frequency / C0
    points, J, M, dS = _equivalent_currents(faces, pec_mirror_z)
    r_hat, t_hat, p_hat = _direction_frame(theta, phi)
    grid = r_hat.shape[:2]
    N, L = _radiation_vectors(points, J, M, dS, k, r_hat.reshape(-1, 3))
    N = N.reshape(grid + (3,))
    L = L.reshape(grid + (3,))
    N_t = np.sum(N * t_hat, axis=-1)
    N_p = np.sum(N * p_hat, axis=-1)
    L_t = np.sum(L * t_hat, axis=-1)
    L_p = np.sum(L * p_hat, axis=-1)
    scale = 1j * k / (4.0 * math.pi)
    return FarField(frequency, theta, phi,
                    scale * (ETA0 * N_t + L_p),
                    scale * (ETA0 * N_p - L_t))


def radiation_pattern(faces, frequency: float, theta, phi, *, pec_mirror_z=None) -> np.ndarray:
    """``U(theta, phi) / max U`` on the grid, (n_theta, n_phi)."""
    return far_field(faces, frequency, theta, phi, pec_mirror_z=pec_mirror_z).pattern


# --------------------------------------------------------------------------
# (b) power in a cone, collection fraction
# --------------------------------------------------------------------------

def cone_quadrature(theta_max: float, n_theta: int, n_phi: int):
    """Nodes and weights integrating ``int U dOmega`` over ``theta <= theta_max``.

    Gauss-Legendre in ``mu = cos(theta)`` on ``[cos theta_max, 1]`` (the
    intensity is a polynomial in the direction cosines for any finite source,
    so this converges spectrally) and the periodic trapezoid rule in ``phi``.
    Returns ``(theta (n_theta,), phi (n_phi,), weights (n_theta, n_phi))``.
    """
    if not 0.0 < theta_max <= math.pi:
        raise FarFieldError(f"theta_max must be in (0, pi], got {theta_max}")
    if n_theta < 1 or n_phi < 1:
        raise FarFieldError("n_theta and n_phi must be at least 1")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_theta))
    lo, hi = math.cos(theta_max), 1.0
    mu = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    w_mu = 0.5 * (hi - lo) * weights
    theta = np.arccos(mu)
    phi = np.arange(int(n_phi)) * (2.0 * math.pi / n_phi)
    w = w_mu[:, None] * np.full((1, int(n_phi)), 2.0 * math.pi / n_phi)
    return theta, phi, w


def cone_power(faces, frequency: float, theta_max: float, *, n_theta=DEFAULT_N_THETA,
               n_phi=DEFAULT_N_PHI, pec_mirror_z=None) -> float:
    """``int U dOmega`` over the cone ``theta <= theta_max`` about +z, in W."""
    theta, phi, w = cone_quadrature(theta_max, n_theta, n_phi)
    ff = far_field(faces, frequency, theta, phi, pec_mirror_z=pec_mirror_z)
    return float(np.sum(ff.intensity * w))


def hemisphere_power(faces, frequency: float, **kwargs) -> float:
    """``int U dOmega`` over the upper hemisphere ``z > 0``, in W."""
    return cone_power(faces, frequency, math.pi / 2.0, **kwargs)


def collection_fraction(faces, frequency: float, numerical_aperture: float, **kwargs) -> float:
    """Fraction of the upward far-field power inside ``theta <= asin(NA)``, NA in air.

    Keyword arguments are those of ``cone_power``.
    """
    na = float(numerical_aperture)
    if not 0.0 < na <= 1.0:
        raise FarFieldError(f"the numerical aperture must be in (0, 1] in air, got {na}")
    inside = cone_power(faces, frequency, math.asin(na), **kwargs)
    upward = hemisphere_power(faces, frequency, **kwargs)
    if upward <= 0.0:
        raise FarFieldError("no upward far-field power; the collection fraction is undefined")
    return inside / upward


# --------------------------------------------------------------------------
# reference fields: the Hertzian dipole
# --------------------------------------------------------------------------

def hertzian_dipole(points, moment, position, frequency: float):
    """Exact fields of an electric dipole ``p`` (C m) at ``position`` in vacuum.

    Jackson, *Classical Electrodynamics*, 3rd ed., eq. 9.18 in SI with the
    ``exp(-i w t)`` convention (``n`` the unit vector from the dipole):

        H = (c k^2 / 4 pi) (n x p) exp(ikr)/r (1 - 1/(ikr))
        E = (1 / 4 pi eps0) { k^2 (n x p) x n exp(ikr)/r
                              + [3 n (n.p) - p] (1/r^3 - ik/r^2) exp(ikr) }

    ``points`` is ``(..., 3)`` in metres; returns ``(E, H)`` of shape
    ``(..., 3)`` in V/m and A/m. Valid at every point except the dipole.
    """
    p = np.asarray(moment, dtype=np.complex128)
    r0 = np.asarray(position, dtype=np.float64)
    if p.shape != (3,) or r0.shape != (3,):
        raise FarFieldError("moment and position must be 3-vectors")
    k = 2.0 * math.pi * float(frequency) / C0
    d = np.asarray(points, dtype=np.float64) - r0
    r = np.linalg.norm(d, axis=-1)
    if np.any(r == 0.0):
        raise FarFieldError("a sample point coincides with the dipole")
    n = d / r[..., None]
    rr = r[..., None]
    phase = np.exp(1j * k * r)[..., None]
    n_cross_p = np.cross(n, p[None, :] + 0 * n)
    n_dot_p = np.sum(n * p, axis=-1)[..., None]
    H = (C0 * k ** 2 / (4.0 * math.pi)) * n_cross_p * phase / rr * (1.0 - 1.0 / (1j * k * rr))
    E = (1.0 / (4.0 * math.pi * EPS0)) * (
        k ** 2 * np.cross(n_cross_p, n) * phase / rr
        + (3.0 * n * n_dot_p - p) * (1.0 / rr ** 3 - 1j * k / rr ** 2) * phase)
    return E, H


def dipole_power(moment, frequency: float) -> float:
    """Total power radiated by the dipole in vacuum: ``c^2 eta0 k^4 |p|^2 / (12 pi)`` (Jackson eq. 9.24)."""
    p = np.asarray(moment, dtype=np.complex128)
    k = 2.0 * math.pi * float(frequency) / C0
    return float(C0 ** 2 * ETA0 * k ** 4 * np.sum(np.abs(p) ** 2) / (12.0 * math.pi))
