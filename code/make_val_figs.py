"""Publication-quality validation figures from bench1/bench3 results."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10.5, 'axes.labelsize': 11.5,
    'axes.titlesize': 10.5, 'legend.fontsize': 9, 'figure.dpi': 150,
})
ZETA = 0.5
D = np.load('results/bench3.npz')
x, y = D['x'], D['y']
Pe, T0 = float(D['Pe']), float(D['T0'])
lam1 = float(D['lambda1'])
trap = D['trap_true']
X, Y = np.meshgrid(x, y, indexing='ij')

C_NUM = '#1a1a1a'
C_TH = '#2a78d6'
C_MS = '#e34948'
C_A = ['#2166ac', '#f46d43', '#d73027']

# ═════════ Figure V1: flow-field validation ═════════
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4),
                         gridspec_kw={'width_ratios': [1.15, 1, 1]})
ax = axes[0]
xi0 = 1.0 + np.log(0.5) / Pe          # alpha=0 diffusive chi=0.5 position
ax.axvline(xi0, color='grey', lw=1.2, ls=':')
ax.text(xi0 - 0.006, 0.02, r'$\alpha=0$', fontsize=8.5, color='grey',
        rotation=90, va='bottom', ha='right')
for al, key, col in [(0.5, 'chi_05', '#86b6ef'),
                     (1.0, 'chi_10', '#2a78d6'),
                     (2.0, 'chi_20', '#104281')]:
    ax.contour(x, y, D[key].T, levels=[0.5], colors=[col], linewidths=2.0)
ax.set_xlabel(r'$\xi$ [-]'); ax.set_ylabel(r'$\eta$ [-]')
ax.set_title('(a) Steady wedge: $\\chi=0.5$ isochlor (nonlinear)\n'
             'interface rotates about mid-depth as $\\alpha$ grows')
ax.legend(handles=[Line2D([0], [0], color=c, lw=2.0,
                          label=r'$\alpha=%.1f$' % a)
                   for a, c in [(0.5, '#86b6ef'), (1.0, '#2a78d6'),
                                (2.0, '#104281')]],
          loc='upper left', framealpha=0.9)
ax.set_xlim(0.82, 1.0); ax.set_ylim(0, ZETA)

ax = axes[1]
U1n, U1b, U1m = D['U1_true'], D['U1_bd'], D['U1_ms']
for eta_q, lw in [(0.46, 2.0), (0.06, 1.2)]:
    j = np.argmin(abs(y - eta_q))
    ax.plot(x, U1n[:, j], '-', color=C_NUM, lw=lw)
    ax.plot(x, U1b[:, j], '--', color=C_TH, lw=lw)
    ax.plot(x, U1m[:, j], ':', color=C_MS, lw=lw + 0.2)
ax.axhline(ZETA / 2, color='grey', lw=0.6, ls=':')
ax.text(0.03, ZETA / 2 + 0.04, r'$\zeta/2$', fontsize=8.5, color='grey')
ax.set_xlabel(r'$\xi$ [-]'); ax.set_ylabel(r'$U_1$ [-]')
ax.set_title('(b) First-order velocity $U_1$\n'
             r'thick: $\eta=0.46$; thin: $\eta=0.06$')
ax.legend(handles=[
    Line2D([0], [0], color=C_NUM, lw=1.8, label='numerical (full physics)'),
    Line2D([0], [0], color=C_TH, lw=1.8, ls='--',
           label='boundary-driven, corrected'),
    Line2D([0], [0], color=C_MS, lw=1.8, ls=':', label='manuscript formula')],
    loc='upper left', framealpha=0.9)
ax.set_xlim(0, 1)

ax = axes[2]
al_cols = ['#86b6ef', '#2a78d6', '#104281']
for (al, key), col in zip([(0.5, 'chi_05'), (1.0, 'chi_10'),
                           (2.0, 'chi_20')], al_cols):
    pass
# eta-integral constancy: recompute from library states isn't saved; use U1_true
etaint = np.trapezoid(U1n, y, axis=1)
ax.plot(x, etaint, color=C_NUM, lw=1.8,
        label=r'$\int_0^\zeta U_1\,d\eta$ (numerical)')
ax.axhline(ZETA**2 / 2, color=C_TH, lw=1.4, ls='--',
           label=r'$\zeta^2/2$ (theory, App. B)')
ax.set_ylim(0.0, 0.25)
ax.set_xlabel(r'$\xi$ [-]')
ax.set_ylabel(r'$\int_0^\zeta U_1\, d\eta$  [-]')
ax.set_title('(c) Key ingredient of $\\mu_1=0$:\n'
             r'$\eta$-integral of $U_1$ is constant in $\xi$')
ax.legend(loc='lower center', framealpha=0.9)
ax.set_xlim(0, 1)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig_val1_flow.{ext}', bbox_inches='tight', dpi=150)
print('fig_val1 saved')

# ═════════ Figure V2: transport validation (2x2) ═════════
tn = D['taus_num']; tt = D['tau_th']
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

ax = axes[0, 0]
for al, col in zip([0.0, 0.5, 1.0], C_A):
    ax.semilogy(tn / T0, D[f'num_mean_a{al}'], '-', color=col, lw=2.0,
                label=r'$\alpha=%.1f$' % al)
    key = 'th_mean_a0' if al == 0.0 else f'th_mean_a{al}_true'
    ax.semilogy(tt / T0, D[key], '--', color=col, lw=1.2, alpha=0.85)
ax.axhline(np.exp(-1), color='grey', lw=0.6, ls=':')
sl = D['slopes']
ax.text(0.03, 0.05, 'late-time decay rates (fit 3.5–5$T_0$):\n'
        r'$\lambda_{\rm eff}$ = %.2f, %.2f, %.2f' % tuple(sl) + '\n'
        r'adjoint theory $\lambda_{\rm eff}\!=\!\lambda_1\!+\!\alpha\zeta\mathrm{Pe}/4$:'
        + '\n%.2f, %.2f, %.2f  (rate increases with $\\alpha$)'
        % (sl[0], sl[0] + 1.25 * 0.5, sl[0] + 1.25),
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax.set_xlabel(r'$\tau/T_0$ [-]')
ax.set_ylabel(r'$\langle\Gamma\rangle$ [-]')
ax.set_title('(a) Domain-averaged flushing, static wedge\n'
             'solid: nonlinear numerics; dashed: first-order theory')
ax.legend(loc='upper right', framealpha=0.9)

ax = axes[0, 1]
for al, col in zip([0.0, 0.5, 1.0], C_A):
    ax.plot(tn / T0, D[f'num_loc_a{al}'], '-', color=col, lw=2.0,
            label=r'$\alpha=%.1f$' % al)
    key = 'th_loc_a0' if al == 0.0 else f'th_loc_a{al}_true'
    ax.plot(tt / T0, D[key], '--', color=col, lw=1.2, alpha=0.85)
ax.axhline(np.exp(-1), color='grey', lw=0.6, ls=':')
ax.set_xlabel(r'$\tau/T_0$ [-]')
ax.set_ylabel(r'$\Gamma(\xi_t,\eta_t)$ [-]')
ax.set_title('(b) Local concentration, seaward trapping zone\n'
             r'$(\xi_t,\eta_t)=(%.2f,%.2f)$, static wedge'
             % (trap[0], trap[1]))
ax.legend(loc='upper right', framealpha=0.9)

ax = axes[1, 0]
for e, col in zip([1.0, 3.0], ['#2166ac', '#d73027']):
    ax.plot(tn / T0, D[f'num_loc_e{e}'], '-', color=col, lw=2.0,
            label=r'$\varepsilon=%.0f$' % e)
    ax.plot(tt / T0, D[f'th_loc_e{e}_true'], '--', color=col, lw=1.2)
    ax.plot(tt / T0, D[f'th_loc_e{e}_exact'], ':', color=col, lw=1.6)
ax.plot(tn / T0, D['num_loc_a0.0'], '-', color='grey', lw=1.0, alpha=0.7,
        label=r'$\varepsilon=0$ ref.')
ax.set_xlabel(r'$\tau/T_0$ [-]')
ax.set_ylabel(r'$\Gamma(\xi_t,\eta_t)$ [-]')
ax.set_title('(c) SLR runs, local trapping zone\n'
             'solid: nonlinear; dashed: quasi-static ansatz; '
             'dotted: exact 1st-order')
ax.legend(loc='upper right', framealpha=0.9)

ax = axes[1, 1]
am = D['alphas_map']
ax.plot(am, 100 * D['err_mean_true'], 'o-', color=C_TH, lw=1.8,
        label='global mean (corrected theory)')
ax.plot(am, 100 * D['err_loc_true'], 's-', color='#104281', lw=1.8,
        label='local, trapping zone (corrected)')
ax.plot(am, 100 * D['err_loc_ms'], 's--', color=C_MS, lw=1.4, alpha=0.8,
        label='local (manuscript $U_1$)')
ax.axhline(5, color='grey', lw=0.7, ls=':')
ax.axhline(15, color='grey', lw=0.7, ls='--')
ax.text(2.52, 5, '5%', fontsize=8, color='grey', va='bottom', ha='right')
ax.text(2.52, 15, '15%', fontsize=8, color='grey', va='bottom', ha='right')
ax.set_xlabel(r'$\alpha$ [-]')
ax.set_ylabel(r'relative error at $\tau=T_0$  [%]')
ax.set_title('(d) First-order accuracy vs wedge strength\n'
             'measured validity ceiling for the perturbation theory')
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig_val2_transport.{ext}',
                bbox_inches='tight', dpi=150)
print('fig_val2 saved')

# ═════════ Figure V3: corrected Gamma1 redistribution field ═════════
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, key, ttl in [(axes[0], 'G1_field_true',
                      '(a) $\\Gamma_1$ — corrected first-order theory'),
                     (axes[1], 'G1_field_ms',
                      '(b) $\\Gamma_1$ — manuscript $U_1$ formula')]:
    G1 = D[key]
    clim = np.abs(G1).max()
    cf = ax.contourf(x, y, G1.T, levels=np.linspace(-clim, clim, 21),
                     cmap='RdBu_r')
    ax.contour(x, y, G1.T, levels=[0], colors='k', linewidths=1.0)
    k = np.unravel_index(np.argmax(G1), G1.shape)
    ax.plot(x[k[0]], y[k[1]], 'k*', ms=13,
            label=r'max $\Gamma_1$: (%.2f, %.2f)' % (x[k[0]], y[k[1]]))
    ax.plot(0.94, 0.48, 'wo', mec='k', ms=7,
            label='manuscript point (0.94, 0.48)')
    ax.set_xlabel(r'$\xi$ [-]'); ax.set_title(ttl, fontsize=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
axes[0].set_ylabel(r'$\eta$ [-]')
plt.colorbar(cf, ax=axes.tolist(), label=r'$\Gamma_1$ [-]', shrink=0.9,
             pad=0.015)
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig_val3_gamma1.{ext}', bbox_inches='tight',
                dpi=150)
print('fig_val3 saved')
