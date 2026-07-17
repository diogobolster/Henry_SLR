"""Test 1: steady solver sanity + first-order velocity comparison."""
import numpy as np, time, sys
sys.path.insert(0, '.')
from henry_solver import HenryGrid, build_flow_matrix, solve_steady_state
from figures import U1_IP, ZETA

g = HenryGrid(nx=121, ny=61)
t0 = time.time()
lu = build_flow_matrix(g)
print(f"flow matrix factorized in {time.time()-t0:.1f}s")

Pe = 10.0
for alpha in [0.0, 0.1, 0.2]:
    t0 = time.time()
    H, U, V, chi, it, dchi = solve_steady_state(g, lu, Pe, alpha, verbose=False)
    Unum1 = (U - 1.0) / alpha if alpha > 0 else U - 1.0
    # analytical U1 on same grid
    Xa, Ya = g.X, g.Y
    U1a = U1_IP(Xa, Ya, ZETA, N=40)
    if alpha > 0:
        err = np.abs(Unum1 - U1a)
        # interior comparison (avoid boundary one-sided diffs)
        e_int = err[2:-2, :]
        print(f"alpha={alpha}: picard_its={it} dchi={dchi:.1e} "
              f"t={time.time()-t0:.1f}s")
        print(f"  U range num: [{U.min():.3f},{U.max():.3f}]  "
              f"mean U-1)/a = {Unum1[1:-1,:].mean():.4f} vs zeta/2={ZETA/2}")
        print(f"  (U-1)/alpha vs U1_IP: max|err| interior={e_int.max():.3f}, "
              f"rms={np.sqrt((e_int**2).mean()):.3f}, "
              f"U1a range=[{U1a.min():.3f},{U1a.max():.3f}]")
        # profiles at a few xi
        for xq in [0.5, 0.8, 0.94]:
            i = np.argmin(abs(g.x - xq))
            print(f"   xi={xq}: num1 mid-eta={Unum1[i, 30]:.4f}  "
                  f"analyt={U1a[i, 30]:.4f}")
    else:
        print(f"alpha=0: max|U-1|={np.abs(U-1).max():.2e} "
              f"max|V|={np.abs(V).max():.2e} its={it} "
              f"t={time.time()-t0:.1f}s")
        print(f"  chi mid-depth profile at xi=0.5..1: "
              f"{chi[np.argmin(abs(g.x-0.9)), 30]:.4f} (expect ~exp(-Pe*0.1)~{np.exp(-1):.3f})")
