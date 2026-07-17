"""Benchmark part 1: velocity fields, wedge structure, Gamma1 comparison.
Saves results to results/bench1.npz"""
import numpy as np, os, time, sys
sys.path.insert(0, '.')
from henry_solver import HenryGrid, build_flow_matrix, solve_flow
from fast_assembly import solve_steady_fast
from figures import U1_IP, lambda1, ZETA
from test3_whichU1 import U1_exact_bvp
from pert_theory import gamma1_field
from scipy.interpolate import RegularGridInterpolator

os.makedirs('results', exist_ok=True)
Pe = 10.0
g = HenryGrid(nx=161, ny=81)
lu = build_flow_matrix(g)

# ---- linearized (small-alpha) full velocity field ----
a_small = 0.02
t0 = time.time()
H, U, V, chi, it, d = solve_steady_fast(g, lu, Pe, a_small)
U1_num = (U - 1.0) / a_small
V1_num = V / a_small
print(f"linearized field: {time.time()-t0:.1f}s its={it}")
print(f"  <U1_num>={U1_num[1:-1,:].mean():.5f} (zeta/2={ZETA/2})")
print(f"  U1_num range [{U1_num[2:-2,:].min():.3f},{U1_num[2:-2,:].max():.3f}]")

U1_ms = U1_IP(g.X, g.Y, ZETA, N=40)          # manuscript formula
U1_bd = U1_exact_bvp(g.X, g.Y, ZETA, N=400)   # corrected boundary-only

# ---- steady wedges for context / eta-integral test ----
wedges = {}
chi_ws = None
for al in [0.5, 1.0, 1.5, 2.0]:
    H, Uw, Vw, chi_w, it, d = solve_steady_fast(g, lu, Pe, al, chi0=chi_ws)
    chi_ws = chi_w
    wedges[al] = (Uw, Vw, chi_w)
    # eta-integral of (U-1)/alpha: constancy in xi underpins mu1=0
    etaint = np.trapezoid((Uw - 1.0) / al, g.y, axis=1)
    print(f"  alpha={al}: eta-int of U1 over xi in [0.1,0.9]: "
          f"min={etaint[16:-16].min():.4f} max={etaint[16:-16].max():.4f} "
          f"(zeta^2/2={ZETA**2/2})")

# ---- Gamma1 fields: manuscript U1 vs true linearized U1 ----
T0 = 1.0 / lambda1(Pe)
tau_ref = 0.5 * T0
xe = np.linspace(0, 1, 81); ye = np.linspace(0, ZETA, 41)
Xe, Ye = np.meshgrid(xe, ye, indexing='ij')

interp = RegularGridInterpolator((g.x, g.y), U1_num, bounds_error=False,
                                 fill_value=None)
U1n_at = lambda X, Y: interp(np.stack([X, Y], axis=-1))
U1m_at = lambda X, Y: U1_IP(X, Y, ZETA, N=40)

t0 = time.time()
G1_true = gamma1_field(Xe, Ye, tau_ref, Pe, U1n_at)
G1_ms = gamma1_field(Xe, Ye, tau_ref, Pe, U1m_at)
print(f"Gamma1 fields: {time.time()-t0:.1f}s")

for name, G1 in [("true-U1", G1_true), ("manuscript-U1", G1_ms)]:
    k = np.unravel_index(np.argmax(G1), G1.shape)
    print(f"  Gamma1[{name}]: max={G1.max():.4f} at "
          f"(xi,eta)=({xe[k[0]]:.3f},{ye[k[1]]:.3f}); min={G1.min():.4f}")

np.savez('results/bench1.npz',
         x=g.x, y=g.y, U1_num=U1_num, V1_num=V1_num, U1_ms=U1_ms,
         U1_bd=U1_bd, xe=xe, ye=ye, G1_true=G1_true, G1_ms=G1_ms,
         chi_05=wedges[0.5][2], chi_10=wedges[1.0][2],
         chi_20=wedges[2.0][2], Pe=Pe, tau_ref=tau_ref)
print("saved results/bench1.npz")
