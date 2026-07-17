"""
henry_solver.py
===============
Full nonlinear variable-density Henry-problem solver in the dimensionless
variables of Bolster, Tartakovsky & Dentz (2007) / the JCH manuscript:

    xi  = x/L in [0,1],   eta = y/L in [0,zeta],  zeta = d/L
    U   = u/ubar,         tau = t*ubar/L,         Pe = ubar*L/D
    alpha = (rho_s-rho_f)/rho_f * L/Dh0   (saltwater coupling)

Governing equations (equivalent-freshwater-head form):
    Flow:  div(grad H) = -alpha * d(chi)/d(eta)
           U = -dH/dxi,  V = -dH/deta - alpha*chi
    Salt (steady or transient):
           dchi/dtau + U dchi/dxi + V dchi/deta = (1/Pe) Lap(chi)
    Contaminant (transient, conservative):
           dG/dtau + U dG/dxi + V dG/deta = (1/Pe) Lap(G)

Boundary conditions:
    Flow:  H(0,eta) = 1                     (prescribed inland head)
           H(1,eta) = -alpha*eta            (hydrostatic seawater head,
                                             datum: sea level at aquifer bottom)
           top/bottom: V = 0  ->  dH/deta = -alpha*chi   (Robin)
    Salt:  chi(0)=0, chi(1)=1, no-flux top/bottom
    Cont.: G(0)=0, G(1)=0,  no-flux top/bottom;  G(tau=0)=1

Numerics: uniform grid, 2nd-order central FD (cell Peclet << 2 verified),
sparse direct solves (factorized once; matrices are alpha-independent),
Picard iteration for the steady coupled problem, explicit time stepping
for transient transport.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ZETA = 0.5


class HenryGrid:
    def __init__(self, nx=161, ny=81, zeta=ZETA):
        self.nx, self.ny, self.zeta = nx, ny, zeta
        self.x = np.linspace(0.0, 1.0, nx)
        self.y = np.linspace(0.0, zeta, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')  # (nx,ny)

    def idx(self, i, j):
        return i * self.ny + j


def build_flow_matrix(g: HenryGrid):
    """Laplacian for H with Dirichlet at xi=0,1 and Robin (dH/deta given)
    at eta=0,zeta. Matrix independent of alpha and chi (they enter the RHS)."""
    nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
    N = nx * ny
    A = sp.lil_matrix((N, N))
    for i in range(nx):
        for j in range(ny):
            k = g.idx(i, j)
            if i == 0 or i == nx - 1:          # Dirichlet in xi
                A[k, k] = 1.0
                continue
            # interior x-part
            A[k, g.idx(i - 1, j)] += 1.0 / dx**2
            A[k, g.idx(i + 1, j)] += 1.0 / dx**2
            A[k, k] += -2.0 / dx**2
            # y-part with ghost-node Robin closure at j=0, ny-1
            if j == 0:
                # ghost: (H[1]-H[-1])/(2dy) = bflux -> H[-1] = H[1] - 2dy*bflux
                A[k, g.idx(i, 1)] += 2.0 / dy**2
                A[k, k] += -2.0 / dy**2
            elif j == ny - 1:
                A[k, g.idx(i, ny - 2)] += 2.0 / dy**2
                A[k, k] += -2.0 / dy**2
            else:
                A[k, g.idx(i, j - 1)] += 1.0 / dy**2
                A[k, g.idx(i, j + 1)] += 1.0 / dy**2
                A[k, k] += -2.0 / dy**2
    return spla.splu(A.tocsc())


def solve_flow(g, lu_flow, chi, alpha):
    """Solve div grad H = -alpha * dchi/deta with BCs; return H, U, V."""
    nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
    rhs = np.zeros((nx, ny))
    # interior source: -alpha * dchi/deta (central)
    dchidy = np.zeros_like(chi)
    dchidy[:, 1:-1] = (chi[:, 2:] - chi[:, :-2]) / (2 * dy)
    dchidy[:, 0] = 0.0   # no-flux salt -> dchi/deta = 0 at walls
    dchidy[:, -1] = 0.0
    rhs[1:-1, :] = -alpha * dchidy[1:-1, :]
    # Robin closure contributions: dH/deta = -alpha*chi at walls
    # bottom j=0: ghost H[-1] = H[1] - 2dy*bflux, bflux = -alpha*chi
    # contributes  -2/dy * bflux ... derive: y-part = (H[1]-2H[0]+H[-1])/dy^2
    #   = (2H[1]-2H[0])/dy^2 - 2*bflux/dy   -> move known part to RHS
    bflux_bot = -alpha * chi[1:-1, 0]
    bflux_top = -alpha * chi[1:-1, -1]
    rhs[1:-1, 0] += 2.0 * bflux_bot / dy
    rhs[1:-1, -1] += -2.0 * bflux_top / dy
    # Dirichlet BCs
    rhs[0, :] = 1.0                      # inland head
    rhs[-1, :] = -alpha * g.y            # hydrostatic seawater
    H = lu_flow.solve(rhs.ravel()).reshape(nx, ny)

    U = np.zeros_like(H)
    V = np.zeros_like(H)
    U[1:-1, :] = -(H[2:, :] - H[:-2, :]) / (2 * dx)
    U[0, :] = -(H[1, :] - H[0, :]) / dx * 1.0  # one-sided (2nd-order optional)
    U[-1, :] = -(H[-1, :] - H[-2, :]) / dx
    V[:, 1:-1] = -(H[:, 2:] - H[:, :-2]) / (2 * dy) - alpha * chi[:, 1:-1]
    V[:, 0] = 0.0
    V[:, -1] = 0.0
    return H, U, V


def build_transport_matrix(g, U, V, Pe, dirichlet_right=True):
    """Steady advection-diffusion operator L[c] = U dc/dxi + V dc/deta
    - (1/Pe) Lap(c), central differences, no-flux top/bottom via ghosts.
    Dirichlet at xi=0 (c=0) and xi=1 (c=1 for salt)."""
    nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
    N = nx * ny
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    iPe = 1.0 / Pe
    for i in range(nx):
        for j in range(ny):
            k = g.idx(i, j)
            if i == 0:
                A[k, k] = 1.0; b[k] = 0.0; continue
            if i == nx - 1:
                A[k, k] = 1.0; b[k] = 1.0 if dirichlet_right else 0.0; continue
            u, v = U[i, j], V[i, j]
            # x advection (central) + diffusion
            A[k, g.idx(i - 1, j)] += -u / (2 * dx) - iPe / dx**2
            A[k, g.idx(i + 1, j)] += u / (2 * dx) - iPe / dx**2
            A[k, k] += 2 * iPe / dx**2
            # y with no-flux ghosts (v=0 at walls, ghost c[-1]=c[1])
            if j == 0:
                A[k, g.idx(i, 1)] += -2 * iPe / dy**2      # c[1]+c[-1]=2c[1]
                A[k, k] += 2 * iPe / dy**2
                # v ~ 0 at wall; advective y-term vanishes
            elif j == ny - 1:
                A[k, g.idx(i, ny - 2)] += -2 * iPe / dy**2
                A[k, k] += 2 * iPe / dy**2
            else:
                A[k, g.idx(i, j - 1)] += -v / (2 * dy) - iPe / dy**2
                A[k, g.idx(i, j + 1)] += v / (2 * dy) - iPe / dy**2
                A[k, k] += 2 * iPe / dy**2
    return A.tocsc(), b


def solve_steady_state(g, lu_flow, Pe, alpha, tol=1e-9, maxit=200, relax=0.7,
                       chi0=None, verbose=False):
    """Picard iteration for the coupled steady flow + salt problem."""
    if chi0 is None:
        # 1D advection-diffusion profile as initial guess
        xi = g.X
        chi = (np.exp(Pe * (xi - 1.0)) - np.exp(-Pe)) / (1.0 - np.exp(-Pe))
    else:
        chi = chi0.copy()
    H = U = V = None
    for it in range(maxit):
        H, U, V = solve_flow(g, lu_flow, chi, alpha)
        A, b = build_transport_matrix(g, U, V, Pe, dirichlet_right=True)
        chi_new = spla.spsolve(A, b).reshape(g.nx, g.ny)
        chi_new = np.clip(chi_new, 0.0, 1.0)
        dchi = np.max(np.abs(chi_new - chi))
        chi = relax * chi_new + (1 - relax) * chi
        if verbose:
            print(f"    Picard {it:3d}: max|dchi| = {dchi:.3e}")
        if dchi < tol:
            break
    H, U, V = solve_flow(g, lu_flow, chi, alpha)
    return H, U, V, chi, it, dchi


# ---------------------------------------------------------------------------
# Transient contaminant transport (explicit, 2nd-order central in space)
# ---------------------------------------------------------------------------

def contaminant_rhs(G, U, V, Pe, dx, dy):
    """dG/dtau for interior points; BCs handled by caller."""
    iPe = 1.0 / Pe
    R = np.zeros_like(G)
    # x fluxes (interior in x; boundaries Dirichlet handled outside)
    Gx = (G[2:, 1:-1] - G[:-2, 1:-1]) / (2 * dx)
    Gxx = (G[2:, 1:-1] - 2 * G[1:-1, 1:-1] + G[:-2, 1:-1]) / dx**2
    Gy = (G[1:-1, 2:] - G[1:-1, :-2]) / (2 * dy)
    Gyy = (G[1:-1, 2:] - 2 * G[1:-1, 1:-1] + G[1:-1, :-2]) / dy**2
    R[1:-1, 1:-1] = (-U[1:-1, 1:-1] * Gx - V[1:-1, 1:-1] * Gy
                     + iPe * (Gxx + Gyy))
    # no-flux walls: ghost G[:,-1] = G[:,1]; v=0 there
    Gx_b = (G[2:, 0] - G[:-2, 0]) / (2 * dx)
    Gxx_b = (G[2:, 0] - 2 * G[1:-1, 0] + G[:-2, 0]) / dx**2
    Gyy_b = 2 * (G[1:-1, 1] - G[1:-1, 0]) / dy**2
    R[1:-1, 0] = -U[1:-1, 0] * Gx_b + iPe * (Gxx_b + Gyy_b)
    Gx_t = (G[2:, -1] - G[:-2, -1]) / (2 * dx)
    Gxx_t = (G[2:, -1] - 2 * G[1:-1, -1] + G[:-2, -1]) / dx**2
    Gyy_t = 2 * (G[1:-1, -2] - G[1:-1, -1]) / dy**2
    R[1:-1, -1] = -U[1:-1, -1] * Gx_t + iPe * (Gxx_t + Gyy_t)
    return R


def run_contaminant(g, Pe, vel_of_tau, tau_end, n_out=60, dt=None,
                    record_pts=None):
    """March G from G=1 with G(0)=G(1)=0. vel_of_tau(tau)->(U,V).
    Returns tau_out, mean_hist, local_hist(dict), snapshots(dict)."""
    dx, dy = g.dx, g.dy
    if dt is None:
        # stability: diffusion + CFL with margin
        Umax = 2.5
        dt = min(0.20 * Pe * dx**2 / (1 + (dx / dy)**2),
                 0.4 * dx / Umax)
    nsteps = int(np.ceil(tau_end / dt))
    dt = tau_end / nsteps
    G = np.ones((g.nx, g.ny))
    G[0, :] = 0.0
    G[-1, :] = 0.0
    tau_out = np.linspace(0, tau_end, n_out + 1)
    out_steps = np.round(tau_out / dt).astype(int)
    mean_hist, snaps = [], {}
    local_hist = {p: [] for p in (record_pts or [])}

    def record(G):
        mean_hist.append(np.trapezoid(
            np.trapezoid(G, g.y, axis=1), g.x) / g.zeta)
        for p in local_hist:
            i = np.argmin(abs(g.x - p[0])); j = np.argmin(abs(g.y - p[1]))
            local_hist[p].append(G[i, j])

    record(G)
    ko = 1
    for n in range(1, nsteps + 1):
        tau = n * dt
        U, V = vel_of_tau(tau - 0.5 * dt)
        # RK2 (midpoint)
        k1 = contaminant_rhs(G, U, V, Pe, dx, dy)
        Gm = G + 0.5 * dt * k1
        Gm[0, :] = 0.0; Gm[-1, :] = 0.0
        k2 = contaminant_rhs(Gm, U, V, Pe, dx, dy)
        G = G + dt * k2
        G[0, :] = 0.0; G[-1, :] = 0.0
        if ko <= n_out and n == out_steps[ko]:
            record(G)
            snaps[tau_out[ko]] = G.copy()
            ko += 1
    for p in local_hist:
        local_hist[p] = np.array(local_hist[p])
    return tau_out, np.array(mean_hist), local_hist, snaps
