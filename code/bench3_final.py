"""Final benchmark: correct first-order theory (numerically integrated O(alpha)
equations) vs full nonlinear Henry solver.

First-order system (exact evaluation, no series/Green's truncation):
   dG0/dtau + dG0/dxi           = (1/Pe) Lap G0            (base flushing)
   dG1/dtau + dG1/dxi - (1/Pe)LapG1 = -U1 dG0/dxi          (unit-alpha corr.)
   quasi-static ansatz:  G = G0 + alpha(tau) * G1
   exact first-order (SLR): dG1x/dtau + ... = -alpha(t) U1 dG0/dxi
Outputs: results/bench3.npz
"""
import numpy as np, os, time, sys
sys.path.insert(0, '.')
from henry_solver import HenryGrid, build_flow_matrix, contaminant_rhs
from fast_assembly import solve_steady_fast
from figures import lambda1, U1_IP, ZETA
from test3_whichU1 import U1_exact_bvp

os.makedirs('results', exist_ok=True)
Pe = 10.0
T0 = 1.0 / lambda1(Pe)
g = HenryGrid(nx=161, ny=81)
lu = build_flow_matrix(g)
out = {'x': g.x, 'y': g.y, 'Pe': Pe, 'T0': T0}

# ---------- velocity fields ----------
Hs, Us, Vs, chis, _, _ = solve_steady_fast(g, lu, Pe, 0.02)
U1_true = (Us - 1.0) / 0.02
U1_ms = U1_IP(g.X, g.Y, ZETA, N=40)
out['U1_true'] = U1_true; out['U1_ms'] = U1_ms
out['U1_bd'] = U1_exact_bvp(g.X, g.Y, ZETA, N=400)

alphas_lib = np.round(np.arange(0.0, 2.51, 0.125), 3)
lib_U, lib_V, lib_chi = [], [], []
chi_w = None
t0 = time.time()
for al in alphas_lib:
    H, U, V, chi_w, it, d = solve_steady_fast(g, lu, Pe, al, chi0=chi_w)
    lib_U.append(U); lib_V.append(V); lib_chi.append(chi_w)
lib_U = np.array(lib_U); lib_V = np.array(lib_V)
print(f"steady library: {time.time()-t0:.0f}s")
out['alphas_lib'] = alphas_lib
out['chi_05'] = lib_chi[4]; out['chi_10'] = lib_chi[8]; out['chi_20'] = lib_chi[16]

def vel_static(al):
    i = np.argmin(abs(alphas_lib - al))
    return lambda tau: (lib_U[i], lib_V[i])

def vel_slr(eps):
    def f(tau):
        a = min(eps * tau, alphas_lib[-1])
        i = min(max(np.searchsorted(alphas_lib, a), 1), len(alphas_lib) - 1)
        w = (a - alphas_lib[i - 1]) / (alphas_lib[i] - alphas_lib[i - 1])
        return ((1 - w) * lib_U[i - 1] + w * lib_U[i],
                (1 - w) * lib_V[i - 1] + w * lib_V[i])
    return f

# ---------- joint march of G0, G1 (and G1x for SLR) ----------
def march_theory(tau_end, n_out, eps_list=(), U1f=None):
    """March G0 and unit-alpha G1 (+ exact first-order G1x per eps).
    Returns tau_out and dict of field snapshots + histories."""
    dx, dy = g.dx, g.dy
    dt = 0.20 * Pe * dx**2 / (1 + (dx / dy)**2)
    nsteps = int(np.ceil(tau_end / dt)); dt = tau_end / nsteps
    U0 = np.ones((g.nx, g.ny)); V0 = np.zeros((g.nx, g.ny))
    G0 = np.ones((g.nx, g.ny)); G0[0] = 0; G0[-1] = 0
    G1 = np.zeros((g.nx, g.ny))
    G1x = {e: np.zeros((g.nx, g.ny)) for e in eps_list}
    tau_out = np.linspace(0, tau_end, n_out + 1)
    steps = np.round(tau_out / dt).astype(int)
    hist = {'mean0': [], 'G0': {}, 'G1': {}, 'G1x': {e: {} for e in eps_list}}

    def src(G0f):
        dG0 = np.zeros_like(G0f)
        dG0[1:-1] = (G0f[2:] - G0f[:-2]) / (2 * dx)
        return -(U1f * dG0)

    def bc(A):
        A[0] = 0; A[-1] = 0

    ko = 1
    snap0 = {0.0: G0.copy()}
    for n in range(1, nsteps + 1):
        t_mid = (n - 0.5) * dt
        # RK2 for the coupled linear system (G0 autonomous)
        k1_0 = contaminant_rhs(G0, U0, V0, Pe, dx, dy)
        s1 = src(G0)
        k1_1 = contaminant_rhs(G1, U0, V0, Pe, dx, dy) + s1
        G0m = G0 + 0.5 * dt * k1_0; bc(G0m)
        G1m = G1 + 0.5 * dt * k1_1; bc(G1m)
        k2_0 = contaminant_rhs(G0m, U0, V0, Pe, dx, dy)
        s2 = src(G0m)
        k2_1 = contaminant_rhs(G1m, U0, V0, Pe, dx, dy) + s2
        for e in eps_list:
            k1x = contaminant_rhs(G1x[e], U0, V0, Pe, dx, dy) + e * (t_mid - 0.5 * dt) * s1
            Gxm = G1x[e] + 0.5 * dt * k1x; bc(Gxm)
            k2x = contaminant_rhs(Gxm, U0, V0, Pe, dx, dy) + e * t_mid * s2
            G1x[e] = G1x[e] + dt * k2x; bc(G1x[e])
        G0 = G0 + dt * k2_0; bc(G0)
        G1 = G1 + dt * k2_1; bc(G1)
        if ko <= n_out and n == steps[ko]:
            hist['G0'][tau_out[ko]] = G0.copy()
            hist['G1'][tau_out[ko]] = G1.copy()
            for e in eps_list:
                hist['G1x'][e][tau_out[ko]] = G1x[e].copy()
            ko += 1
    return tau_out, hist

def dom_mean(F):
    return np.trapezoid(np.trapezoid(F, g.y, axis=1), g.x) / g.zeta

def at(F, p):
    return F[np.argmin(abs(g.x - p[0])), np.argmin(abs(g.y - p[1]))]

tau_end = 2.4 * T0
t0 = time.time()
tau_th, TH = march_theory(tau_end, 48, eps_list=(1.0, 3.0),
                          U1f=U1_true)
print(f"theory march (true U1): {time.time()-t0:.0f}s")
t0 = time.time()
_, TH_ms = march_theory(tau_end, 48, eps_list=(1.0, 3.0), U1f=U1_ms)
print(f"theory march (ms U1): {time.time()-t0:.0f}s")

# trapping zone from correct G1 at 0.5 T0
tt = tau_th[np.argmin(abs(tau_th - 0.5 * T0))]
G1ref = TH['G1'][tt]
k = np.unravel_index(np.argmax(G1ref), G1ref.shape)
trap_true = (g.x[k[0]], g.y[k[1]])
print(f"trapping zone (max G1, true U1, tau=0.5T0): {trap_true}")
G1ref_ms = TH_ms['G1'][tt]
k2 = np.unravel_index(np.argmax(G1ref_ms), G1ref_ms.shape)
print(f"trapping zone (ms U1): ({g.x[k2[0]]:.3f},{g.y[k2[1]]:.3f})")
out['G1_field_true'] = G1ref; out['G1_field_ms'] = G1ref_ms
out['trap_true'] = np.array(trap_true)
PT = (0.94, 0.48)
out['PT'] = np.array(PT)

taus_keys = tau_th[1:]  # skip 0
out['tau_th'] = tau_th

# theory histories
for tag, TT in [('true', TH), ('ms', TH_ms)]:
    for al in [0.5, 1.0]:
        out[f'th_mean_a{al}_{tag}'] = np.array(
            [1.0] + [dom_mean(np.clip(TT['G0'][t] + al * TT['G1'][t], 0, None))
                     for t in taus_keys])
        out[f'th_loc_a{al}_{tag}'] = np.array(
            [1.0] + [at(np.clip(TT['G0'][t] + al * TT['G1'][t], 0, None), trap_true)
                     for t in taus_keys])
    for e in [1.0, 3.0]:
        out[f'th_mean_e{e}_{tag}'] = np.array(
            [1.0] + [dom_mean(np.clip(TT['G0'][t] + e * t * TT['G1'][t], 0, None))
                     for t in taus_keys])
        out[f'th_loc_e{e}_{tag}'] = np.array(
            [1.0] + [at(np.clip(TT['G0'][t] + e * t * TT['G1'][t], 0, None), trap_true)
                     for t in taus_keys])
# exact first-order convolution (true U1 only)
for e in [1.0, 3.0]:
    out[f'th_mean_e{e}_exact'] = np.array(
        [1.0] + [dom_mean(np.clip(TH['G0'][t] + TH['G1x'][e][t], 0, None))
                 for t in taus_keys])
    out[f'th_loc_e{e}_exact'] = np.array(
        [1.0] + [at(np.clip(TH['G0'][t] + TH['G1x'][e][t], 0, None), trap_true)
                 for t in taus_keys])
out['th_mean_a0'] = np.array([1.0] + [dom_mean(TH['G0'][t]) for t in taus_keys])
out['th_loc_a0'] = np.array([1.0] + [at(TH['G0'][t], trap_true) for t in taus_keys])

# ---------- nonlinear runs ----------
from henry_solver import run_contaminant
runs = {}
for al in [0.0, 0.5, 1.0]:
    t0 = time.time()
    taus, mh, lh, snaps = run_contaminant(g, Pe, vel_static(al), tau_end,
                                          n_out=48,
                                          record_pts=[PT, trap_true])
    runs[('a', al)] = (taus, mh, lh)
    out[f'num_mean_a{al}'] = mh
    out[f'num_loc_a{al}'] = lh[trap_true]
    out[f'num_locPT_a{al}'] = lh[PT]
    print(f"nonlinear static a={al}: {time.time()-t0:.0f}s")
for e in [1.0, 3.0]:
    t0 = time.time()
    taus, mh, lh, snaps = run_contaminant(g, Pe, vel_slr(e), tau_end,
                                          n_out=48,
                                          record_pts=[PT, trap_true])
    out[f'num_mean_e{e}'] = mh
    out[f'num_loc_e{e}'] = lh[trap_true]
    print(f"nonlinear SLR eps={e}: {time.time()-t0:.0f}s")
out['taus_num'] = taus

# ---------- long runs for decay-rate (mu1) test ----------
slopes = []
for al in [0.0, 0.5, 1.0]:
    t0 = time.time()
    taus_l, mh_l, _, _ = run_contaminant(g, Pe, vel_static(al), 5.0 * T0,
                                         n_out=50)
    m = (taus_l >= 3.5 * T0) & (mh_l > 0)
    p = np.polyfit(taus_l[m], np.log(mh_l[m]), 1)
    slopes.append(-p[0])
    print(f"decay-rate a={al}: lambda_eff={-p[0]:.4f} ({time.time()-t0:.0f}s)")
out['slopes_alpha'] = np.array([0.0, 0.5, 1.0])
out['slopes'] = np.array(slopes)
out['lambda1'] = lambda1(Pe)

# ---------- error map ----------
alphas_map = np.arange(0.25, 2.51, 0.25)
em_t, el_t, em_m, el_m = [], [], [], []
iT0 = np.argmin(abs(tau_th - T0)); tK = tau_th[iT0]
for al in alphas_map:
    taus_m, mh, lh, _ = run_contaminant(g, Pe, vel_static(al), 1.02 * T0,
                                        n_out=6, record_pts=[trap_true])
    nm = np.interp(tK, taus_m, mh); nl = np.interp(tK, taus_m, lh[trap_true])
    for tag, TT, em, el in [('true', TH, em_t, el_t), ('ms', TH_ms, em_m, el_m)]:
        Gp = np.clip(TT['G0'][tK] + al * TT['G1'][tK], 0, None)
        em.append(abs(dom_mean(Gp) - nm) / nm)
        el.append(abs(at(Gp, trap_true) - nl) / nl)
    print(f"map a={al}: true=({em_t[-1]:.3f},{el_t[-1]:.3f}) "
          f"ms=({em_m[-1]:.3f},{el_m[-1]:.3f})")
out['alphas_map'] = alphas_map
out['err_mean_true'] = np.array(em_t); out['err_loc_true'] = np.array(el_t)
out['err_mean_ms'] = np.array(em_m); out['err_loc_ms'] = np.array(el_m)

np.savez('results/bench3.npz', **out)
print("saved results/bench3.npz")
