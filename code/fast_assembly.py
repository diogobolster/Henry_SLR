"""Vectorized sparse-matrix assembly for the Henry solver."""
import numpy as np
import scipy.sparse as sp


def transport_matrix_fast(g, U, V, Pe, dirichlet_right=True):
    """Steady advection-diffusion operator; identical stencil to
    henry_solver.build_transport_matrix but vectorized."""
    nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
    N = nx * ny
    iPe = 1.0 / Pe
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    K = (I * ny + J).ravel()
    rows, cols, vals = [], [], []

    def add(mask, coff, v):
        k = K[mask.ravel()]
        rows.append(k); cols.append(k + coff)
        vals.append(np.broadcast_to(v, k.shape).astype(float).ravel()
                    if np.isscalar(v) else v)

    left = (I == 0); right = (I == nx - 1)
    inter = ~left & ~right
    bot = inter & (J == 0); top = inter & (J == ny - 1)
    mid = inter & ~bot & ~top

    # Dirichlet rows
    for m in (left, right):
        k = K[m.ravel()]
        rows.append(k); cols.append(k); vals.append(np.ones(k.size))

    u = U.ravel(); v = V.ravel()
    # x-direction terms (all interior rows)
    for m in (mid, bot, top):
        mk = m.ravel()
        k = K[mk]
        rows.append(k); cols.append(k - ny)
        vals.append(-u[mk] / (2 * dx) - iPe / dx**2)
        rows.append(k); cols.append(k + ny)
        vals.append(u[mk] / (2 * dx) - iPe / dx**2)
    # y-direction: middle rows
    mk = mid.ravel(); k = K[mk]
    rows.append(k); cols.append(k - 1)
    vals.append(-v[mk] / (2 * dy) - iPe / dy**2)
    rows.append(k); cols.append(k + 1)
    vals.append(v[mk] / (2 * dy) - iPe / dy**2)
    # wall rows (ghost closure, v=0 at wall)
    mk = bot.ravel(); k = K[mk]
    rows.append(k); cols.append(k + 1)
    vals.append(np.full(k.size, -2 * iPe / dy**2))
    mk = top.ravel(); k = K[mk]
    rows.append(k); cols.append(k - 1)
    vals.append(np.full(k.size, -2 * iPe / dy**2))
    # diagonals for all interior rows
    for m in (mid, bot, top):
        mk = m.ravel(); k = K[mk]
        rows.append(k); cols.append(k)
        vals.append(np.full(k.size, 2 * iPe / dx**2 + 2 * iPe / dy**2))

    A = sp.csc_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    b = np.zeros(N)
    if dirichlet_right:
        b[K[right.ravel()]] = 1.0
    return A, b


def solve_steady_fast(g, lu_flow, Pe, alpha, tol=1e-9, maxit=300, relax=0.7,
                      chi0=None):
    import scipy.sparse.linalg as spla
    from henry_solver import solve_flow
    if chi0 is None:
        chi = (np.exp(Pe * (g.X - 1.0)) - np.exp(-Pe)) / (1.0 - np.exp(-Pe))
        chi = np.ascontiguousarray(np.broadcast_to(chi, (g.nx, g.ny)))
    else:
        chi = chi0.copy()
    it, dchi = 0, np.inf
    for it in range(maxit):
        H, U, V = solve_flow(g, lu_flow, chi, alpha)
        A, b = transport_matrix_fast(g, U, V, Pe)
        chi_new = spla.spsolve(A, b).reshape(g.nx, g.ny)
        chi_new = np.clip(chi_new, 0.0, 1.0)
        dchi = np.max(np.abs(chi_new - chi))
        chi = relax * chi_new + (1 - relax) * chi
        if dchi < tol:
            break
    H, U, V = solve_flow(g, lu_flow, chi, alpha)
    return H, U, V, chi, it, dchi
