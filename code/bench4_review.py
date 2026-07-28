"""Review-response runs:
 A. Monitoring-point sensitivity (R2 #8): local G0,G1 histories at a cluster
    of points in the seaward trapping zone; local concentration curves under
    coupling forcing eps_alpha = 3.
 B. Outflow-BC sensitivity: absorbing Dirichlet vs zero-gradient outflow at
    xi=1; global decay rate and local trapping history comparison.
 C. Monte Carlo uncertainty propagation for the two redesigned worked
    examples under the corrected head-controlled formulation:
       eps_f  = sdot*omega*L^2/(K*dh^2)     (fractional rate)
       alpha0 = delta*L/dh
       eps_a  = alpha0*eps_f                (absolute coupling rate)
       eps*   = 2*(lambda1(Pe) + alpha0*zeta*Pe/4)/zeta
Saves results/bench4.npz and prints a summary.
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from henry_solver import HenryGrid, contaminant_rhs
from figures import lambda1

ZETA = 0.5
Pe = 10.0
T0 = 1.0 / lambda1(Pe)
out = {}

# ─────────────────────────────────────────────────────────────
# A. march G0, G1 recording a cluster of monitoring points
# ─────────────────────────────────────────────────────────────
from test3_whichU1 import U1_exact_bvp
g = HenryGrid(nx=161, ny=81)
U1f = U1_exact_bvp(g.X, g.Y, ZETA, N=400)
PTS = [(0.90, 0.40), (0.94, 0.40), (0.90, 0.50), (0.94, 0.50), (0.98, 0.45)]
idx = [(np.argmin(abs(g.x - p[0])), np.argmin(abs(g.y - p[1]))) for p in PTS]

def march_points(tau_end, n_out):
    dx, dy = g.dx, g.dy
    dt = 0.20 * Pe * dx**2 / 2
    nsteps = int(np.ceil(tau_end / dt)); dt = tau_end / nsteps
    U0 = np.ones((g.nx, g.ny)); V0 = np.zeros((g.nx, g.ny))
    G0 = np.ones((g.nx, g.ny)); G0[0] = 0; G0[-1] = 0
    G1 = np.zeros((g.nx, g.ny))
    tau_out = np.linspace(0, tau_end, n_out + 1)
    steps = np.round(tau_out / dt).astype(int)
    h0 = [[1.0] for _ in PTS]; h1 = [[0.0] for _ in PTS]
    ko = 1
    for n in range(1, nsteps + 1):
        k1_0 = contaminant_rhs(G0, U0, V0, Pe, dx, dy)
        dG0 = np.zeros_like(G0); dG0[1:-1] = (G0[2:] - G0[:-2]) / (2 * dx)
        k1_1 = contaminant_rhs(G1, U0, V0, Pe, dx, dy) - U1f * dG0
        G0m = G0 + 0.5 * dt * k1_0; G0m[0] = 0; G0m[-1] = 0
        G1m = G1 + 0.5 * dt * k1_1; G1m[0] = 0; G1m[-1] = 0
        k2_0 = contaminant_rhs(G0m, U0, V0, Pe, dx, dy)
        dG0m = np.zeros_like(G0m); dG0m[1:-1] = (G0m[2:] - G0m[:-2]) / (2 * dx)
        k2_1 = contaminant_rhs(G1m, U0, V0, Pe, dx, dy) - U1f * dG0m
        G0 = G0 + dt * k2_0; G0[0] = 0; G0[-1] = 0
        G1 = G1 + dt * k2_1; G1[0] = 0; G1[-1] = 0
        if ko <= n_out and n == steps[ko]:
            for k, (i, j) in enumerate(idx):
                h0[k].append(G0[i, j]); h1[k].append(G1[i, j])
            ko += 1
    return tau_out, np.array(h0), np.array(h1)

t0 = time.time()
tau_m, H0, H1 = march_points(1.6 * T0, 40)
print(f"A. monitoring cluster march: {time.time()-t0:.0f}s")
out['tau_m'] = tau_m; out['H0'] = H0; out['H1'] = H1
out['PTS'] = np.array(PTS)
eps_a = 3.0
for k, p in enumerate(PTS):
    loc = np.clip(H0[k] + eps_a * tau_m * H1[k], 0, None)
    ref = np.clip(H0[k], 0, None)
    i1 = np.argmin(abs(tau_m - T0))
    print(f"   ({p[0]:.2f},{p[1]:.2f}): Gamma(T0) SLR={loc[i1]:.3f} "
          f"static={ref[i1]:.3f}  retardation={(loc[i1]-ref[i1])/ref[i1]*100:+.0f}%")

# ─────────────────────────────────────────────────────────────
# B. outflow BC sensitivity (alpha=0 and alpha=1 nonlinear velocity)
# ─────────────────────────────────────────────────────────────
from henry_solver import build_flow_matrix, run_contaminant
from fast_assembly import solve_steady_fast

def run_bc(U, V, tau_end, outflow='dirichlet'):
    dx, dy = g.dx, g.dy
    dt = 0.20 * Pe * dx**2 / 2
    nsteps = int(np.ceil(tau_end / dt)); dt = tau_end / nsteps
    G = np.ones((g.nx, g.ny)); G[0] = 0
    if outflow == 'dirichlet': G[-1] = 0
    else: G[-1] = G[-2]
    taus = np.linspace(0, tau_end, 41)
    steps = np.round(taus / dt).astype(int)
    mh = [np.trapezoid(np.trapezoid(G, g.y, axis=1), g.x) / ZETA]
    it, jt = np.argmin(abs(g.x - 0.94)), np.argmin(abs(g.y - 0.50))
    lh = [G[it, jt]]
    ko = 1
    def bc(A):
        A[0] = 0
        if outflow == 'dirichlet': A[-1] = 0
        else: A[-1] = A[-2]
    for n in range(1, nsteps + 1):
        k1 = contaminant_rhs(G, U, V, Pe, dx, dy)
        Gm = G + 0.5 * dt * k1; bc(Gm)
        k2 = contaminant_rhs(Gm, U, V, Pe, dx, dy)
        G = G + dt * k2; bc(G)
        if ko <= 40 and n == steps[ko]:
            mh.append(np.trapezoid(np.trapezoid(G, g.y, axis=1), g.x) / ZETA)
            lh.append(G[it, jt]); ko += 1
    return taus, np.array(mh), np.array(lh)

lu = build_flow_matrix(g)
H, U1n, V1n, chi, _, _ = solve_steady_fast(g, lu, Pe, 1.0)
U0f = np.ones((g.nx, g.ny)); V0f = np.zeros((g.nx, g.ny))
res_bc = {}
t0 = time.time()
for al, (Uv, Vv) in [('a0', (U0f, V0f)), ('a1', (U1n, V1n))]:
    for bctype in ['dirichlet', 'outflow']:
        taus_b, mh, lh = run_bc(Uv, Vv, 2.0 * T0, bctype)
        res_bc[(al, bctype)] = (mh, lh)
        m = (taus_b >= 1.0 * T0)
        p = np.polyfit(taus_b[m], np.log(np.clip(mh[m], 1e-12, None)), 1)
        print(f"B. {al} {bctype:9s}: mean(T0)={np.interp(T0,taus_b,mh):.4f} "
              f"loc(T0)={np.interp(T0,taus_b,lh):.4f} late-slope={-p[0]:.3f}")
print(f"B. outflow sensitivity: {time.time()-t0:.0f}s")
out['taus_bc'] = taus_b
for k, v in res_bc.items():
    out[f'bc_{k[0]}_{k[1]}_mean'] = v[0]; out[f'bc_{k[0]}_{k[1]}_loc'] = v[1]

# ─────────────────────────────────────────────────────────────
# C. Monte Carlo uncertainty for redesigned examples
# ─────────────────────────────────────────────────────────────
YR = 365.25 * 86400
DELTA = 0.025
rng = np.random.default_rng(7)
N = 20000

EX = {
 'Ex1': dict(K=1e-4, L=200.0, dh=5.0, om=0.30, ell=20.0),
 'Ex2': dict(K=1e-7, L=300.0, dh=5.0, om=0.35, ell=60.0),
}
def ratio_stats(p):
    K = p['K'] * 3.0**rng.uniform(-1, 1, N)          # x/÷3
    dh = p['dh'] * rng.uniform(0.7, 1.3, N)
    L = p['L'] * rng.uniform(0.8, 1.2, N)
    ell = p['ell'] * 2.0**rng.uniform(-1, 1, N)       # x/÷2
    sd = rng.uniform(3.0, 10.0, N) / 1000 / YR
    om = p['om']
    Pe_s = L / ell
    lam = np.pi**2 / Pe_s + Pe_s / 4
    a0 = DELTA * L / dh
    epsf = sd * om * L**2 / (K * dh**2)
    epsa = a0 * epsf
    estar = 2 * (lam + a0 * ZETA * Pe_s / 4) / ZETA
    ratio = epsa / estar
    ubar = K * dh / (om * L)
    T0s = 1 / lam
    qs_ok = epsf * T0s <= 0.5
    pert_ok = (a0 + epsa * T0s) <= 1.8
    return dict(ratio=ratio, a0=a0, epsf=epsf, epsa=epsa, estar=estar,
                Pe=Pe_s, valid=qs_ok & pert_ok)

for name, p in EX.items():
    s = ratio_stats(p)
    q = np.percentile(s['ratio'], [5, 25, 50, 75, 95])
    print(f"C. {name}: ratio median={q[2]:.3g}  5-95%=[{q[0]:.3g},{q[4]:.3g}] "
          f" alpha0 med={np.median(s['a0']):.2f} "
          f" P(ratio>0.3)={np.mean(s['ratio']>0.3):.2f} "
          f" P(ratio>1)={np.mean(s['ratio']>1):.3f} "
          f" frac within validity={np.mean(s['valid']):.2f}")
    out[f'mc_{name}_ratio'] = s['ratio']; out[f'mc_{name}_valid'] = s['valid']
    out[f'mc_{name}_a0'] = s['a0']

# nominal values for the manuscript table
print("\nNominal example values (3 and 10 mm/yr):")
for name, p in EX.items():
    for sd_mm in [3.0, 10.0]:
        sd = sd_mm / 1000 / YR
        Pe_s = p['L'] / p['ell']
        lam = np.pi**2 / Pe_s + Pe_s / 4
        a0 = DELTA * p['L'] / p['dh']
        epsf = sd * p['om'] * p['L']**2 / (p['K'] * p['dh']**2)
        epsa = a0 * epsf
        estar = 2 * (lam + a0 * ZETA * Pe_s / 4) / ZETA
        ub = p['K'] * p['dh'] / (p['om'] * p['L'])
        print(f"  {name} sdot={sd_mm}mm/yr: Pe={Pe_s:.0f} a0={a0:.2f} "
              f"epsf={epsf:.3g} epsa={epsa:.3g} eps*={estar:.1f} "
              f"ratio={epsa/estar:.3g} qs={epsf/lam:.2f} "
              f"Tflush={1/lam*p['L']/ub/YR:.1f}yr")

np.savez('results/bench4.npz', **out)
print("saved results/bench4.npz")
