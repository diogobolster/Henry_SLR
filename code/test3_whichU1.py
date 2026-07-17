"""Test 3: which analytic first-order field does the numerics match?
Paper's U1_IP:  zeta/2 - sum l*pi*a_l cos(l pi eta/zeta) cosh(l pi xi/zeta)/sinh(l pi/zeta)
Exact BVP sol:  zeta/2 - sum (l*pi/zeta)*a_l cos(...) cosh(...)/sinh(...)
(differ by factor 1/zeta = 2 on the mode amplitudes for zeta=0.5)
"""
import numpy as np, sys
sys.path.insert(0, '.')
from henry_solver import HenryGrid, build_flow_matrix, solve_flow
from figures import U1_IP, ZETA

def U1_exact_bvp(xi, eta, zeta=ZETA, N=400):
    res = np.full_like(xi, zeta / 2.0, dtype=float)
    for l in range(1, N + 1):
        al = 2 * zeta * (1 - (-1)**l) / (l**2 * np.pi**2)
        if al == 0.0:
            continue
        arg_num = l * np.pi * xi / zeta
        arg_den = l * np.pi / zeta
        # stable cosh/sinh ratio: exp(arg_num-arg_den) * (1+e^-2num)/(1-e^-2den)
        ratio = np.exp(arg_num - arg_den) * (1 + np.exp(-2 * arg_num)) / (1 - np.exp(-2 * arg_den))
        res -= al * (l * np.pi / zeta) * np.cos(l * np.pi * eta / zeta) * ratio
    return res

g = HenryGrid(nx=241, ny=121)   # finer grid for sharper test
lu = build_flow_matrix(g)
alpha = 1e-3
Hb, Ub, Vb = solve_flow(g, lu, np.zeros((g.nx, g.ny)), alpha)
U1n = (Ub - 1.0) / alpha

mask = (g.X >= 0.05) & (g.X <= 0.95)   # away from end boundaries
U1p = U1_IP(g.X, g.Y, ZETA, N=200)
U1e = U1_exact_bvp(g.X, g.Y)

for name, U1a in [("paper U1_IP", U1p), ("exact BVP (1/zeta)", U1e)]:
    e = np.abs(U1n - U1a)[mask]
    scale = np.abs(U1a[mask]).max()
    print(f"{name:20s}: max|err|={e.max():.4f} rms={np.sqrt((e**2).mean()):.5f} "
          f"(field scale {scale:.2f})")

# pointwise probes
for (xq, yq) in [(0.8, 0.05), (0.9, 0.05), (0.9, 0.45), (0.94, 0.48),
                 (0.7, 0.25), (0.94, 0.02)]:
    i = np.argmin(abs(g.x - xq)); j = np.argmin(abs(g.y - yq))
    print(f"  ({xq:.2f},{yq:.2f}): num={U1n[i,j]:+.4f} "
          f"paper={U1p[i,j]:+.4f} exact={U1e[i,j]:+.4f}")
