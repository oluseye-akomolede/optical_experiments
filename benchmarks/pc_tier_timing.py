"""Time one k-point of the 2-D PhC solve at the production mesh.

The plan (docs/photonic-crystal-plan.md sec.3) sets its caps -- 30 k-points per
polarisation, 5000 nodes -- from a MEASURED per-k cost, not an estimate. This
script is that measurement, runnable, so the caps can be re-derived instead of
recalled.

It assembles the P1 stiffness and eps-weighted mass on the production mesh
(40 divisions per lattice vector -> 1681 nodes / 3200 triangles, the counts the
plan quotes), then times scipy.sparse.linalg.eigsh with shift-invert at a small
negative sigma, repeated, and reports the median and the MAX (the max is what a
cap must be set from -- the spread here is large and is not warm-up).

Dev-only. scipy is already in the solver image; nothing here ships in an image
or a venv.
"""
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay

A = 1.0
R_OVER_A = 0.2
N_ROD_SQ = 3.4 ** 2   # a silicon-like rod; the exact value does not change the cost
N_BG_SQ = 1.0


def production_mesh(n_div):
    xs = np.linspace(0, A, n_div + 1)
    ys = np.linspace(0, A, n_div + 1)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    T = Delaunay(P).simplices
    return P, T


def assemble(P, T):
    """P1 stiffness K and eps-weighted mass M, in the cell's own index space."""
    n = len(P)
    centre = np.array([A / 2, A / 2])
    rod = np.linalg.norm(P[T].mean(axis=1) - centre, axis=1) < R_OVER_A
    eps = np.where(rod, N_ROD_SQ, N_BG_SQ)

    rows, cols, kv = [], [], []
    mrows, mcols, mv = [], [], []
    for t, (i, j, k) in enumerate(T):
        p0, p1, p2 = P[i], P[j], P[k]
        det = ((p1[0] - p0[0]) * (p2[1] - p0[1])
               - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        area = abs(det) / 2
        if area < 1e-14:
            continue
        s = np.sign(det)
        b = np.array([P[j, 1] - P[k, 1], P[k, 1] - P[i, 1],
                      P[i, 1] - P[j, 1]]) / (2 * area * s)
        c = np.array([P[k, 0] - P[j, 0], P[i, 0] - P[k, 0],
                      P[j, 0] - P[i, 0]]) / (2 * area * s)
        Ke = area * (np.column_stack([b, c]) @ np.column_stack([b, c]).T)
        Me = area / 12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        idx = [i, j, k]
        for x in range(3):
            for y in range(3):
                rows.append(idx[x]); cols.append(idx[y]); kv.append(Ke[x, y])
                mrows.append(idx[x]); mcols.append(idx[y]); mv.append(Me[x, y] * eps[t])
    K = coo_matrix((kv, (rows, cols)), shape=(n, n)).tocsr().astype(np.complex128)
    M = coo_matrix((mv, (mrows, mcols)), shape=(n, n)).tocsr().astype(np.complex128)
    return K, M


def main(n_div=40, n_bands=12, repeats=12, sigma=-0.5):
    P, T = production_mesh(n_div)
    print(f"production mesh: {len(P)} nodes, {len(T)} triangles "
          f"(N={n_div} divisions per lattice vector)")
    K, M = assemble(P, T)
    times = []
    for _ in range(repeats):
        t0 = time.time()
        eigsh(K, k=n_bands, M=M, sigma=sigma, which="LM", maxiter=3000)
        times.append(time.time() - t0)
    t = np.array(times)
    print(f"sparse shift-invert, complex Hermitian, {n_bands} bands, "
          f"sigma={sigma}, {repeats} independent k-points:")
    print(f"  per-k: median {np.median(t):.1f} s   MAX {t.max():.1f} s   "
          f"min {t.min():.1f} s")
    print(f"  30 k-points x 2 polarisations at the MAX: {t.max() * 60 / 60:.1f} min")
    print("  (the max is what the plan's k-point cap is set from)")


if __name__ == "__main__":
    main()
