"""Regenerate the manuscript's figures with the CORRECTED implementation:
 - Gamma0: correct series / marched base solution (no decay double-count,
   correct e^{-Pe/2} coefficients)
 - Gamma1: numerically integrated O(alpha) equation with the corrected
   closed-form boundary-driven U1 (factor l*pi/zeta)
 - trapping point taken as argmax Gamma1 (Pe=10, tau=0.5 T0)
Outputs fig1_zones_v2, fig1b_mechanism_v2, fig2_flushing_v2, fig5_contrast_v2,
fig3_regime_v2, fig4_validity_v2 (png+pdf) into results_v2/.
"""
import numpy as np, os, sys, time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, '.')
from henry_solver import HenryGrid, contaminant_rhs

os.makedirs('results_v2', exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 11, 'legend.fontsize': 10, 'figure.dpi': 150,
})
ZETA = 0.5

def lambda1(Pe):
    return np.pi**2 / Pe + Pe / 4.0

def U1_corrected(xi, eta, zeta=ZETA, N=400):
    """Corrected closed-form boundary-driven first-order velocity:
       U1 = zeta/2 - sum (l pi / zeta) a_l cos(l pi eta/zeta)
                     cosh(l pi xi/zeta)/sinh(l pi/zeta),
       a_l = 2 zeta (1-(-1)^l)/(l^2 pi^2).  (stable cosh/sinh ratio)"""
    res = np.full_like(xi, zeta / 2.0, dtype=float)
    for l in range(1, N + 1):
        al = 2 * zeta * (1 - (-1)**l) / (l**2 * np.pi**2)
        if al == 0.0:
            continue
        an, ad = l * np.pi * xi / zeta, l * np.pi / zeta
        ratio = np.exp(an - ad) * (1 + np.exp(-2 * an)) / (1 - np.exp(-2 * ad))
        res -= al * (l * np.pi / zeta) * np.cos(l * np.pi * eta / zeta) * ratio
    return res

# ---------------- theory march: G0 and unit-alpha G1 ----------------
def march(Pe, tau_end, n_out, nx=161, ny=81, snap_taus=()):
    g = HenryGrid(nx=nx, ny=ny)
    dx, dy = g.dx, g.dy
    U1f = U1_corrected(g.X, g.Y)
    dt = 0.20 * Pe * dx**2 / (1 + (dx / dy)**2)
    dt = min(dt, 0.4 * dx / 1.5)
    nsteps = int(np.ceil(tau_end / dt)); dt = tau_end / nsteps
    U0 = np.ones((nx, ny)); V0 = np.zeros((nx, ny))
    G0 = np.ones((nx, ny)); G0[0] = 0; G0[-1] = 0
    G1 = np.zeros((nx, ny))
    tau_out = np.linspace(0, tau_end, n_out + 1)
    steps = np.round(tau_out / dt).astype(int)
    snap_steps = {int(round(t / dt)): t for t in snap_taus}
    H = dict(tau=tau_out, mean0=[1.0], mean1=[0.0], snaps={})
    loc_series = {}  # filled after trap point known; store fields sparsely
    def dm(F):
        return np.trapezoid(np.trapezoid(F, g.y, axis=1), g.x) / ZETA
    # also store G0,G1 at output times at ALL points of the top row & interior
    H['G0_top'] = [np.ones(nx)]; H['G1_top'] = [np.zeros(nx)]
    ko = 1
    for n in range(1, nsteps + 1):
        k1_0 = contaminant_rhs(G0, U0, V0, Pe, dx, dy)
        dG0 = np.zeros_like(G0); dG0[1:-1] = (G0[2:] - G0[:-2]) / (2 * dx)
        s1 = -(U1f * dG0)
        k1_1 = contaminant_rhs(G1, U0, V0, Pe, dx, dy) + s1
        G0m = G0 + 0.5 * dt * k1_0; G0m[0] = 0; G0m[-1] = 0
        G1m = G1 + 0.5 * dt * k1_1; G1m[0] = 0; G1m[-1] = 0
        k2_0 = contaminant_rhs(G0m, U0, V0, Pe, dx, dy)
        dG0m = np.zeros_like(G0m); dG0m[1:-1] = (G0m[2:] - G0m[:-2]) / (2 * dx)
        s2 = -(U1f * dG0m)
        k2_1 = contaminant_rhs(G1m, U0, V0, Pe, dx, dy) + s2
        G0 = G0 + dt * k2_0; G0[0] = 0; G0[-1] = 0
        G1 = G1 + dt * k2_1; G1[0] = 0; G1[-1] = 0
        if n in snap_steps:
            H['snaps'][snap_steps[n]] = (G0.copy(), G1.copy())
        if ko <= n_out and n == steps[ko]:
            H['mean0'].append(dm(G0)); H['mean1'].append(dm(G1))
            H['G0_top'].append(G0[:, -1].copy())
            H['G1_top'].append(G1[:, -1].copy())
            ko += 1
    H['mean0'] = np.array(H['mean0']); H['mean1'] = np.array(H['mean1'])
    H['G0_top'] = np.array(H['G0_top']); H['G1_top'] = np.array(H['G1_top'])
    H['grid'] = g
    return H

# ------------- runs -------------
T0_10 = 1.0 / lambda1(10.0)
t0 = time.time()
H10 = march(10.0, 2.4 * T0_10, 48,
            snap_taus=(0.3 * T0_10, 0.5 * T0_10, 0.6 * T0_10, 1.0 * T0_10))
print(f"Pe=10 march: {time.time()-t0:.0f}s", flush=True)
T0_100 = 1.0 / lambda1(100.0)
t0 = time.time()
H100 = march(100.0, 3.0 * T0_100, 40, nx=241, ny=121)
print(f"Pe=100 march: {time.time()-t0:.0f}s", flush=True)
T0_1 = 1.0 / lambda1(1.0)
t0 = time.time()
H1 = march(1.0, 3.0 * T0_1, 40)
print(f"Pe=1 march: {time.time()-t0:.0f}s", flush=True)

# trapping point from corrected G1 at 0.5 T0 (Pe=10)
g10 = H10['grid']
G0h, G1h = H10['snaps'][0.5 * T0_10]
k = np.unravel_index(np.argmax(G1h), G1h.shape)
XI_T, ETA_T = g10.x[k[0]], g10.y[k[1]]
print(f"corrected trapping point: ({XI_T:.3f}, {ETA_T:.3f})", flush=True)
ix, ie = k[0], k[1]

taus10 = H10['tau']
loc0 = np.array([1.0] + [H10['snaps'].get(t, (None,))[0][ix, ie]
                          if t in H10['snaps'] else np.nan for t in taus10[1:]])
# use G0_top/G1_top series at trap xi (trap is at top wall eta=zeta)
loc0 = H10['G0_top'][:, ix]
loc1 = H10['G1_top'][:, ix]

# ═════════ FIG 2 v2: flushing histories ═════════
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.0))
cols = ['#1f77b4', '#ff7f0e', '#2ca02c']
for (Pe, HH, T0), col in zip([(1.0, H1, T0_1), (10.0, H10, T0_10),
                              (100.0, H100, T0_100)], cols):
    t = HH['tau'] / T0
    stat = HH['mean0']
    slr = np.clip(HH['mean0'] + 1.0 * HH['tau'] * HH['mean1'], 1e-12, None)
    ax_a.semilogy(t, stat, color=col, ls='--', lw=1.5, alpha=0.6)
    ax_a.semilogy(t, slr, color=col, ls='-', lw=2.2,
                  label=r'$\mathrm{Pe}=%g$' % Pe)
ax_a.axhline(np.exp(-1), color='grey', lw=0.8, ls=':')
ax_a.text(0.06, np.exp(-1) * 1.12, r'$e^{-1}$', fontsize=9, color='grey')
ax_a.set_xlim(0, 3.0); ax_a.set_ylim(2e-2, 1.4)
ax_a.set_xlabel(r'$\tau/T_0$ [-]')
ax_a.set_ylabel(r'$\langle\Gamma\rangle/\langle\Gamma\rangle_0$ [-]')
ax_a.set_title('(a) Domain-averaged flushing: SLR $\\varepsilon=1$ (solid)\n'
               'vs static (dashed); SLR accelerates decay ($\\mu_1>0$)')
ax_a.legend(loc='lower left', framealpha=0.9)

eps_v = [0, 1, 3, 6]
eps_c = ['#2166ac', '#74add1', '#f46d43', '#d73027']
eps_ls = ['-', '--', '-.', ':']
for e, col, ls in zip(eps_v, eps_c, eps_ls):
    loc = np.clip(loc0 + e * taus10 * loc1, 0, None)
    ax_b.plot(taus10 / T0_10, loc, color=col, ls=ls, lw=2.2,
              label=r'$\varepsilon=%d$' % e)
    glob = np.clip(H10['mean0'] + e * taus10 * H10['mean1'], 0, None)
    ax_b.plot(taus10 / T0_10, glob, color=col, ls=ls, lw=0.9, alpha=0.45)
ax_b.axhline(np.exp(-1), color='grey', lw=0.8, ls=':')
ax_b.axhline(1.0, color='grey', lw=0.5, ls=':')
ax_b.set_xlim(0, 1.5)
ax_b.set_xlabel(r'$\tau/T_0$ [-]')
ax_b.set_ylabel(r'$\Gamma/\Gamma_0$ [-]')
ax_b.set_title('(b) Seaward trapping zone $(\\xi_t,\\eta_t)=(%.2f,%.2f)$ '
               '(thick)\nvs global average (thin), $\\mathrm{Pe}=10$'
               % (XI_T, ETA_T))
hs = [Line2D([0], [0], color=c, ls=l, lw=2.2, label=r'$\varepsilon=%d$' % e)
      for e, c, l in zip(eps_v, eps_c, eps_ls)]
hs += [Line2D([0], [0], color='k', lw=2.2, label='trapping zone'),
       Line2D([0], [0], color='k', lw=0.9, alpha=0.5, label='global avg')]
ax_b.legend(handles=hs, loc='upper right', framealpha=0.9, fontsize=8.5,
            ncol=2)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig2_flushing_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig2 v2 saved', flush=True)

# ═════════ FIG 1 v2: field snapshots under SLR ═════════
alpha0, eps = 0.5, 0.8
fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
levels = np.linspace(0, 1, 13)
for kk, (ax, tq) in enumerate(zip(axes, [0.3 * T0_10, 0.6 * T0_10, T0_10])):
    G0s, G1s = H10['snaps'][tq]
    a_slr = alpha0 + eps * tq
    C_slr = np.clip(G0s + a_slr * G1s, 0, None)
    C_sta = np.clip(G0s + alpha0 * G1s, 0, None)
    Cmax = max(C_slr.max(), C_sta.max())
    C_slr /= Cmax; C_sta /= Cmax
    cf = ax.contourf(g10.x, g10.y, C_slr.T, levels=levels, cmap='RdYlBu_r',
                     extend='both')
    ax.contour(g10.x, g10.y, C_sta.T, levels=[0.5], colors='k',
               linewidths=1.5, linestyles='--')
    ax.contour(g10.x, g10.y, C_slr.T, levels=[0.5], colors='k',
               linewidths=2.0)
    ax.set_xlabel(r'$\xi$ [-]')
    ax.set_title(r'$\tau=%.1f\,T_0$,  $\alpha=%.2f$' % (tq / T0_10, a_slr))
    ax.set_aspect('equal')
    if kk == 0:
        ax.set_ylabel(r'$\eta$ [-]')
        ax.legend(handles=[
            Line2D([0], [0], color='k', lw=2.0, label='SLR'),
            Line2D([0], [0], color='k', lw=1.5, ls='--', label='static')],
            loc='upper left', fontsize=8.5, framealpha=0.9)
plt.colorbar(cf, ax=axes.tolist(), label=r'$\Gamma/\Gamma_{\max}$ [-]',
             shrink=0.88, pad=0.02)
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig1_zones_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig1 v2 saved', flush=True)

# ═════════ FIG 1b v2: redistribution field ═════════
G0s, G1s = H10['snaps'][0.5 * T0_10]
vmax = G0s.max()
dC = (np.clip(G0s + 1.0 * G1s, 0, None) - G0s) / vmax
fig, ax = plt.subplots(figsize=(6.5, 3.5))
clim = np.abs(dC).max()
cf = ax.contourf(g10.x, g10.y, dC.T, levels=np.linspace(-clim, clim, 21),
                 cmap='RdBu_r', extend='both')
ax.contour(g10.x, g10.y, dC.T, levels=[0], colors='k', linewidths=1.4)
ax.contour(g10.x, g10.y, (G0s / vmax).T, levels=[0.25, 0.5, 0.75],
           colors='k', linewidths=0.6, linestyles='--', alpha=0.4)
ax.plot(XI_T, ETA_T, 'k*', ms=13,
        label=r'trapping zone $(\xi_t,\eta_t)=(%.2f,\,%.2f)$' % (XI_T, ETA_T))
plt.colorbar(cf, ax=ax, label=r'$\Delta\Gamma/\Gamma_{0,\max}$ [-]',
             pad=0.02, shrink=0.95)
ax.set_xlabel(r'$\xi$ [-]  (land $\leftarrow\;\rightarrow$ sea)')
ax.set_ylabel(r'$\eta$ [-]')
ax.set_aspect('equal'); ax.legend(loc='upper left', fontsize=9,
                                  framealpha=0.9)
ax.set_title(r'Wedge-driven redistribution '
             r'$\Delta\Gamma=\Gamma(\alpha=1)-\Gamma(\alpha=0)$, '
             r'$\mathrm{Pe}=10$, $\tau=0.5T_0$', fontsize=10)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig1b_mechanism_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig1b v2 saved', flush=True)

# ═════════ FIG 5 v2: relative deviations ═════════
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.4))
for (Pe, HH, T0), col in zip([(1.0, H1, T0_1), (10.0, H10, T0_10),
                              (100.0, H100, T0_100)], cols):
    dev = 1.0 * HH['tau'] * HH['mean1'] / np.clip(HH['mean0'], 1e-12, None)
    ax_a.plot(HH['tau'] / T0, 100 * dev, color=col, lw=2.0,
              label=r'$\mathrm{Pe}=%g$' % Pe)
ax_a.axhline(0, color='grey', lw=0.6)
ax_a.set_xlabel(r'$\tau/T_0$ [-]')
ax_a.set_ylabel(r'global deviation $\varepsilon\tau\langle\Gamma_1\rangle/'
                r'\langle\Gamma_0\rangle$ [%]')
ax_a.set_title('(a) Domain-averaged SLR effect, $\\varepsilon=1$\n'
               '(negative: SLR accelerates global flushing)')
ax_a.legend(framealpha=0.9)
for e, col in zip([1, 3, 6], ['#74add1', '#f46d43', '#d73027']):
    dev = e * taus10 * loc1 / np.clip(loc0, 1e-12, None)
    ax_b.plot(taus10 / T0_10, 100 * dev, color=col, lw=2.0,
              label=r'$\varepsilon=%d$' % e)
ax_b.axhline(0, color='grey', lw=0.6)
ax_b.set_xlabel(r'$\tau/T_0$ [-]')
ax_b.set_ylabel(r'local deviation $\varepsilon\tau\,\Gamma_1/\Gamma_0$ [%]')
ax_b.set_title('(b) Seaward trapping zone, $\\mathrm{Pe}=10$\n'
               '(positive: SLR retards local cleanup)')
ax_b.legend(framealpha=0.9)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig5_contrast_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig5 v2 saved', flush=True)

# ═════════ FIG 3 v2: regime diagram ═════════
def eps_star(Pe, alpha0=0.0, zeta=ZETA, g_corr=0.12):
    return lambda1(Pe) / (zeta / 2.0 * (1.0 + alpha0 * g_corr))

Pe_arr = np.logspace(-0.3, 2.2, 200)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for a0, col, ls in zip([0.0, 0.5, 1.0], ['#2166ac', '#f4a582', '#d6604d'],
                       ['-', '--', '-.']):
    ax.loglog(Pe_arr, [eps_star(p, a0) for p in Pe_arr], color=col, ls=ls,
              lw=2.4, label=r'$\alpha_0=%.1f$' % a0)
e0 = np.array([eps_star(p) for p in Pe_arr])
ax.fill_between(Pe_arr, 1e-2, e0, alpha=0.10, color='#2166ac')
ax.fill_between(Pe_arr, e0, 500, alpha=0.10, color='#d6604d')
ax.text(0.72, 0.20, 'FLUSHING', transform=ax.transAxes, fontsize=13,
        color='#2166ac', alpha=0.85, ha='center', style='italic',
        fontweight='bold')
ax.text(0.20, 0.78, 'TRAPPING', transform=ax.transAxes, fontsize=13,
        color='#d6604d', alpha=0.85, ha='center', style='italic',
        fontweight='bold')
ax.axvspan(10, 100, alpha=0.07, color='grey')
ax.plot(2 * np.pi, 4 * np.pi, 'ko', ms=7)
ax.annotate(r'$\mathrm{Pe}^*=2\pi$', xy=(2 * np.pi, 4 * np.pi),
            xytext=(20, 5), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1.0))
ax.plot(10.0, 0.57, 's', color='#1a9641', ms=10, zorder=6,
        label='Ex. 1 (high-$K$, flushing)')
ax.plot(5.0, 60.0, '^', color='#d73027', ms=10, zorder=6,
        label='Ex. 2 (low-$K$, trapping)')
ax.set_xlabel(r'P\'eclet number $\mathrm{Pe}$ [-]')
ax.set_ylabel(r'Dimensionless SLR rate $\varepsilon$ [-]')
ax.set_xlim(Pe_arr[0], Pe_arr[-1]); ax.set_ylim(0.01, 500)
ax.legend(loc='upper left', framealpha=0.92, fontsize=9)
ax.set_title(r'Regime diagram: conservative boundary '
             r'$\varepsilon^*(\mathrm{Pe},\alpha_0)$')
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig3_regime_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig3 v2 saved', flush=True)

# ═════════ FIG 4 v2: validity map with measured ceilings ═════════
fig, ax = plt.subplots(figsize=(6.2, 4.6))
tau_r = np.linspace(0.05, 3.0, 200)
# curves eps = alpha_ceil/tau  with tau in units of T0
for aceil, col, lab in [(1.0, '#6da7ec', 'measured 5% (local) ceiling'),
                        (1.8, '#2a78d6', 'measured 15% (local) ceiling'),
                        (2.5, '#104281', 'measured 10% (global) ceiling')]:
    ax.plot(tau_r, aceil / (tau_r * T0_10), color=col, lw=2.0, label=lab)
for e, m in [(1, 'o'), (3, 's'), (6, '^')]:
    ax.plot(1.0, e, m, color='#d73027', ms=7)
    ax.text(1.04, e * 1.05, r'$\varepsilon=%d$' % e, fontsize=8,
            color='#d73027')
ax.set_yscale('log')
ax.set_xlabel(r'$\tau/T_0$ [-]')
ax.set_ylabel(r'$\varepsilon$ [-]')
ax.set_title('Domain of validity: first-order predictions reliable below\n'
             'each curve (ceilings measured against nonlinear simulation)')
ax.legend(framealpha=0.9, fontsize=8.5)
ax.set_xlim(0.05, 3.0); ax.set_ylim(0.1, 60)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results_v2/fig4_validity_v2.{ext}', bbox_inches='tight',
                dpi=150)
print('fig4 v2 saved', flush=True)
print('ALL V2 FIGURES DONE', flush=True)
