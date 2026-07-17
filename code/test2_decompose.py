"""Test 2: decompose first-order velocity into boundary-driven and
interior-buoyancy parts; identify what the paper's U1_IP represents."""
import numpy as np, sys
sys.path.insert(0, '.')
from henry_solver import HenryGrid, build_flow_matrix, solve_flow
from figures import U1_IP, ZETA

g = HenryGrid(nx=121, ny=61)
lu = build_flow_matrix(g)
Pe = 10.0
alpha = 1e-3   # linear regime; chi -> chi0 (uncoupled 1D profile)

chi0 = (np.exp(Pe * (g.X - 1.0)) - np.exp(-Pe)) / (1.0 - np.exp(-Pe))
chi0 = np.broadcast_to(chi0, (g.nx, g.ny)).copy()

# (a) boundary forcing only: interior source & Robin use chi=0
Hb, Ub, Vb = solve_flow(g, lu, np.zeros_like(chi0), alpha)
# (b) full: boundary + interior buoyancy with chi0
Hf, Uf, Vf = solve_flow(g, lu, chi0, alpha)

U1b = (Ub - 1.0) / alpha
U1f = (Uf - 1.0) / alpha
U1a = U1_IP(g.X, g.Y, ZETA, N=60)

for name, U1n in [("boundary-only", U1b), ("full(chi0)", U1f)]:
    e = np.abs(U1n - U1a)[2:-2, :]
    print(f"{name:14s}: mean={U1n[1:-1,:].mean():+.4f} "
          f"range=[{U1n[2:-2,:].min():+.3f},{U1n[2:-2,:].max():+.3f}] "
          f"max|err| vs U1_IP={e.max():.4f} rms={np.sqrt((e**2).mean()):.4f}")
print(f"analytic U1_IP: mean={U1a.mean():+.4f} "
      f"range=[{U1a.min():+.3f},{U1a.max():+.3f}]")
# where is the residual of the full field largest?
r = U1f - U1a
imax = np.unravel_index(np.argmax(np.abs(r[2:-2, :])), r[2:-2, :].shape)
print(f"full-field residual peak at xi={g.x[imax[0]+2]:.3f}, "
      f"eta={g.y[imax[1]]:.3f}, value={r[imax[0]+2, imax[1]]:+.3f}")
# vertical velocity structure check
print(f"V1 full range: [{(Vf/alpha)[2:-2,1:-1].min():+.3f},"
      f"{(Vf/alpha)[2:-2,1:-1].max():+.3f}]")
