"""Figures for the reformulated revision: monitoring cluster, Monte Carlo
uncertainty, and regime diagram v3 (absolute rate eps = alpha0*eps_f,
analytic alpha0-dependence of eps*)."""
import numpy as np, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, '.')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10.5, 'axes.labelsize': 11.5,
    'axes.titlesize': 10.5, 'legend.fontsize': 9, 'figure.dpi': 150,
})
ZETA = 0.5
def lambda1(Pe): return np.pi**2 / Pe + Pe / 4

D = np.load('results/bench4.npz')
T0 = 1.0 / lambda1(10.0)

# ═════════ fig_cluster ═════════
tau, H0, H1 = D['tau_m'], D['H0'], D['H1']
PTS = D['PTS']
cols = ['#2166ac', '#74add1', '#1baf7a', '#d73027', '#f4a582']
eps = 3.0
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for k, (p, c) in enumerate(zip(PTS, cols)):
    loc = np.clip(H0[k] + eps * tau * H1[k], 0, None)
    ref = np.clip(H0[k], 0, None)
    i1 = np.argmin(abs(tau - T0))
    ret = (loc[i1] - ref[i1]) / ref[i1] * 100
    lw = 2.6 if (abs(p[0] - 0.94) < 1e-6 and abs(p[1] - 0.50) < 1e-6) else 1.8
    ax.plot(tau / T0, loc, color=c, lw=lw,
            label=rf'$({p[0]:.2f},{p[1]:.2f})$: $+{ret:.0f}\%$')
    ax.plot(tau / T0, ref, color=c, lw=1.0, ls='--', alpha=0.55)
ax.axvline(1.0, color='grey', lw=0.7, ls=':')
ax.set_xlabel(r'$\tau/T_0$ [-]')
ax.set_ylabel(r'$\Gamma(\xi,\eta)$ [-]')
ax.set_title('Local histories across the seaward trapping zone, '
             r'$\mathrm{Pe}=10$' '\n'
             r'solid: coupling forcing $\varepsilon=3$; dashed: static '
             r'reference; labels: coupling-induced retardation at $\tau=T_0$')
ax.legend(loc='upper right', framealpha=0.92, title='monitoring point')
ax.set_xlim(0, 1.6)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig_cluster.{ext}', bbox_inches='tight', dpi=150)
print('fig_cluster saved')

# ═════════ fig_mc ═════════
fig, ax = plt.subplots(figsize=(6.8, 4.2))
data = [np.log10(D['mc_Ex1_ratio']), np.log10(D['mc_Ex2_ratio'])]
valid = [D['mc_Ex1_valid'].mean(), D['mc_Ex2_valid'].mean()]
bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False,
                widths=0.5, patch_artist=True, tick_labels=[
                    'Example 1\n(steep sandy)', 'Example 2\n(slow silty)'])
for patch, col in zip(bp['boxes'], ['#86b6ef', '#f4a582']):
    patch.set_facecolor(col); patch.set_alpha(0.8)
for med in bp['medians']:
    med.set_color('#0b0b0b'); med.set_linewidth(2)
ax.axvline(0, color='#d73027', lw=1.6, ls='--')
ax.text(0.05, 1.72, r'threshold $\varepsilon/\varepsilon^*=1$', fontsize=9,
        color='#b02a2a')
for y, (v, d2) in enumerate(zip(valid, data), start=1):
    ax.text(np.percentile(d2, 95) + 0.25, y,
            f'{v*100:.0f}% of draws\nwithin validity', fontsize=8.5,
            va='center', color='#52514e')
ax.set_xlabel(r'$\log_{10}(\varepsilon/\varepsilon^*)$ [-]')
ax.set_title('Monte Carlo uncertainty propagation of the screening ratio\n'
             r'($2{\times}10^4$ draws; $K\times/\div3$, $\Delta h_0\pm30\%$,'
             r' $L\pm20\%$, $\ell\times/\div2$, $\dot{s}\in[3,10]$ mm/yr)')
ax.set_xlim(-5.5, 1.5)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig_mc.{ext}', bbox_inches='tight', dpi=150)
print('fig_mc saved')

# ═════════ fig3_regime_v3 ═════════
def eps_star(Pe, a0):
    return 2 * (lambda1(Pe) + a0 * ZETA * Pe / 4) / ZETA

Pe_arr = np.logspace(-0.3, 2.2, 200)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for a0, col, ls in zip([0.0, 1.0, 1.8], ['#2166ac', '#f4a582', '#d6604d'],
                       ['-', '--', '-.']):
    ax.loglog(Pe_arr, [eps_star(p, a0) for p in Pe_arr], color=col, ls=ls,
              lw=2.4, label=rf'$\alpha_0={a0:.1f}$')
e0 = np.array([eps_star(p, 0.0) for p in Pe_arr])
ax.fill_between(Pe_arr, 1e-4, e0, alpha=0.10, color='#2166ac')
ax.fill_between(Pe_arr, e0, 1e3, alpha=0.10, color='#d6604d')
ax.text(0.70, 0.16, 'FLUSHING-DOMINATED', transform=ax.transAxes, fontsize=11.5,
        color='#2166ac', alpha=0.85, ha='center', style='italic',
        fontweight='bold')
ax.text(0.24, 0.83, 'TRAPPING CONCERN', transform=ax.transAxes, fontsize=11.5,
        color='#d6604d', alpha=0.85, ha='center', style='italic',
        fontweight='bold')
ax.axvspan(10, 100, alpha=0.07, color='grey')
ax.plot(2 * np.pi, 4 * np.pi, 'ko', ms=7)
ax.annotate(r'$\mathrm{Pe}^*=2\pi$', xy=(2 * np.pi, 4 * np.pi),
            xytext=(18, 60), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1.0))
# examples at 6 mm/yr nominal
ax.plot(10.0, 9.1e-4, 's', color='#1a9641', ms=10, zorder=6,
        label='Ex. 1 (flushing)')
ax.plot(5.0, 3.6, '^', color='#d73027', ms=10, zorder=6,
        label='Ex. 2 (elevated)')
ax.set_xlabel(r"P\'eclet number $\mathrm{Pe}$ [-]")
ax.set_ylabel(r'Absolute SLR coupling rate $\varepsilon=\alpha_0\varepsilon_f$ [-]')
ax.set_xlim(Pe_arr[0], Pe_arr[-1]); ax.set_ylim(1e-4, 1e3)
ax.legend(loc='lower right', framealpha=0.92, fontsize=9)
ax.set_title(r'Screening regime diagram: '
             r'$\varepsilon^*(\mathrm{Pe},\alpha_0)$')
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/fig3_regime_v3.{ext}', bbox_inches='tight',
                dpi=150)
print('fig3_regime_v3 saved')
