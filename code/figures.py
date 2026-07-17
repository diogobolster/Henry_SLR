"""
LEGACY IMPLEMENTATION -- RETAINED FOR COMPARISON ONLY
=====================================================
This is the figure-generation script from the ORIGINAL submission of the
paper. It contains two documented defects in the evaluation of Gamma0
(a double-counted advective decay factor and a mirrored series
coefficient; see Appendix D of the revised paper) and understates the
U1 mode amplitudes by a factor 1/zeta (see Appendix A).

It is kept in this repository because the benchmark and diagnostic
scripts import its analytical helper functions (lambda1, eps_star, U1_IP)
and compare its output against the corrected implementation.

DO NOT use this script to generate figures. Use figures_corrected.py.
"""

"""
figures.py
==========
Generates all figures for:

  "Contaminant fate in coastal aquifers under sea level rise:
   analytical solutions for a quasi-static saltwater wedge"

  Bolster et al. (in prep.)

This script implements the intermediate-Pe perturbation solutions of
Bolster, Tartakovsky & Dentz (2007), Adv. Water Resour. 30, 1962-1972,
extended to a quasi-statically evolving coupling parameter alpha(tau).

Figures produced
----------------
fig1_zones.pdf / .png
    Time evolution of the contaminant concentration field under SLR
    (Pe=10, alpha0=0.5, eps=0.8). Three snapshots at tau = 0.3, 0.6,
    1.0 * T0. Solid C=0.5 contour = SLR case; dashed = static reference.

fig1b_mechanism.pdf / .png
    Single-panel spatial redistribution figure. Shows DeltaC = C(alpha=1)
    - C(alpha=0) at Pe=10, tau=0.5*T0. Red = accumulation at seaward
    boundary; blue = depletion in interior. Star marks monitoring point.

fig2_flushing.pdf / .png
    Two-panel concentration history figure.
    (a) Global domain-averaged decay on semilogy for Pe=1,10,100;
        eps=1 (solid) vs eps=0 (dashed) -- nearly indistinguishable.
    (b) Local concentration at seaward trapping zone (0.94, 0.48) vs
        global average for Pe=10, eps=0,1,3,6. Shows two-speed behavior.

fig3_regime.pdf / .png
    Log-log regime diagram in (Pe, eps) space showing the flushing-trapping
    boundary eps*(Pe, alpha0) for alpha0=0, 0.5, 1.0. Includes typical
    coastal Pe band, Pe* annotation, and two example aquifer points.

Dependencies
------------
numpy, scipy, matplotlib  (all standard; pip install if needed)

Usage
-----
    python figures.py

Outputs are written to OUTPUT_DIR (default: "./" = current directory).
Change OUTPUT_DIR below to write elsewhere.

Notes on numerical implementation
----------------------------------
- Gamma0_norm: normalized by spatial maximum to avoid exp(Pe/2*xi)
  overflow at large Pe. Raw Gamma0_raw is also provided for panel (a)
  of Figure 2 which needs correct temporal decay.
- Gamma1_norm: computed using normalized Gamma0 as source, so the
  returned field is on the same scale as Gamma0_norm.
- Perturbation validity: alpha << ~1.84 at Pe=10. In Figure 2b, eps
  values 0,1,3,6 give alpha(T0) = 0,0.29,0.86,1.72 -- all within range.
  eps* ~ 14 at Pe=10 is well outside the perturbation regime.
- All time quadratures use Gauss-Legendre (n_t points); spatial
  integrals use trapezoidal rule on uniform grids.
- N=40 modes for Gamma0, N=15 for Gamma1 (convergence verified).

Key analytical results encoded here
-------------------------------------
- lambda1(Pe) = pi^2/Pe + Pe/4  (leading decay eigenvalue)
- eps_star(Pe, alpha0) = lambda1(Pe) / (zeta/2 * (1 + alpha0*g))
  with g=0.12 and zeta=0.5  (critical SLR rate, Appendix C)
- Pe* = 2*pi, eps*_min = 4*pi ~ 12.57  (analytical minimum at zeta=0.5)
- <U1> = zeta/2 = 0.25  (domain-averaged velocity correction, Appendix A)
- mu1 = 0  (eigenvalue unchanged at first order, Appendix B)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "./"   # change to write figures elsewhere

# ── figure style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'legend.fontsize':10,
    'figure.dpi':     150,
})

# ── domain constants ──────────────────────────────────────────────────────────
ZETA  = 0.5    # dimensionless aquifer depth  zeta = d/L
N_G0  = 40     # Fourier modes for Gamma0
N_G1  = 15     # Fourier modes for Gamma1
N_U1  = 20     # Fourier modes for U1

# Seaward trapping zone: point of maximum Gamma1 > 0 at Pe=10
# (verified numerically; used as local monitoring point in Figure 2b)
XI_TRAP  = 0.94
ETA_TRAP = 0.48


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def lambda1(Pe):
    """
    Leading eigenvalue of the uncoupled transport operator.
        lambda1(Pe) = pi^2/Pe + Pe/4
    Minimum at Pe* = 2*pi where lambda1(2*pi) = pi.
    """
    return np.pi**2 / Pe + Pe / 4.0


def eps_star(Pe, alpha0=0.0, zeta=ZETA, g_corr=0.12):
    """
    Critical dimensionless SLR rate (Appendix C):
        eps* = lambda1(Pe) / (<U1> * (1 + alpha0*g_corr))
    where <U1> = zeta/2 is the domain-averaged velocity correction.

    g_corr ~ 0.12 captures weak alpha0-dependence estimated numerically.
    Setting alpha0=0 gives the leading-order result eps* = lambda1/(zeta/2).

    Analytical minimum: Pe* = 2*pi, eps*_min = 2*pi/zeta (= 4*pi for zeta=0.5).
    """
    return lambda1(Pe) / (zeta / 2.0 * (1.0 + alpha0 * g_corr))


def U1_IP(xi, eta, zeta=ZETA, N=N_U1):
    """
    First-order x-velocity correction, Bolster (2007) eq. (21):
        U1^IP = zeta/2
                - sum_{l=1}^N l*pi*a_l * cos(l*pi*eta/zeta)
                              * cosh(l*pi*xi/zeta) / sinh(l*pi/zeta)
    with a_l = 2*zeta*(1 - (-1)^l) / (l^2*pi^2).

    Domain average: <U1> = zeta/2 exactly (sum integrates to zero over eta).
    Spatial variation: min ~ -0.92, max ~ 1.42 for zeta=0.5 (Pe=10).
    The spatially-varying part drives local accumulation at xi~0.94.
    """
    result = np.full_like(xi, zeta / 2.0, dtype=float)
    for l in range(1, N + 1):
        al = 2 * zeta * (1 - (-1)**l) / (l**2 * np.pi**2)
        result -= (l * np.pi * al
                   * np.cos(l * np.pi * eta / zeta)
                   * np.cosh(l * np.pi * xi / zeta)
                   / np.sinh(l * np.pi / zeta))
    return result


def Gamma0_norm(xi, eta, tau, Pe, zeta=ZETA, N=N_G0):
    """
    Leading-order natural attenuation concentration, Bolster (2007) eq. (25),
    normalized by spatial maximum C_max(tau) to prevent overflow at large Pe.

    Raw solution:
        Gamma0 = 8*pi * exp(Pe/2*(xi - tau/2))
                 * sum_n a_n * sin(n*pi*xi) * exp(-lambda_n*tau)
    where lambda_n = Pe/4 + pi^2*n^2/Pe,
          a_n = n*(1 - (-1)^n*exp(Pe/2)) / (Pe^2/4 + n^2*pi^2).

    Returns C/C_max in [0, 1]. Negative values clipped to 0.
    """
    if tau <= 0:
        tau = 1e-6
    result = np.zeros_like(xi, dtype=float)
    for n in range(1, N + 1):
        an = (n * (1 - (-1)**n * np.exp(min(Pe / 2, 500)))
              / (Pe**2 / 4 + n**2 * np.pi**2))
        decay = np.exp(-(Pe / 4 + np.pi**2 * n**2 / Pe) * tau)
        result += an * np.sin(n * np.pi * xi) * decay
    raw = np.clip(8 * np.pi * np.exp(Pe / 2 * (xi - tau / 2)) * result,
                  0, None)
    vmax = raw.max()
    return raw / vmax if vmax > 0 else raw


def Gamma0_raw(xi, eta, tau, Pe, zeta=ZETA, N=N_G0):
    """
    Raw (unnormalized) Gamma0. Used in Figure 2a for correct temporal
    decay in domain average. Overflows at Pe >> 20 -- only use for Pe<=20
    or when domain-averaging immediately after computation.
    """
    if tau <= 0:
        tau = 1e-6
    result = np.zeros_like(xi, dtype=float)
    for n in range(1, N + 1):
        an = (n * (1 - (-1)**n * np.exp(min(Pe / 2, 500)))
              / (Pe**2 / 4 + n**2 * np.pi**2))
        decay = np.exp(-(Pe / 4 + np.pi**2 * n**2 / Pe) * tau)
        result += an * np.sin(n * np.pi * xi) * decay
    return np.clip(8 * np.pi * np.exp(Pe / 2 * (xi - tau / 2)) * result,
                   0, None)


def Gamma1_norm(xi_grid, eta_grid, tau, Pe,
                zeta=ZETA, N=N_G1, n_sp=22, n_t=16):
    """
    First-order concentration correction, Bolster (2007) eq. (31).

    Satisfies (eq. 26):
        dC1/dtau + dC1/dxi = (1/Pe)*Laplacian(C1) - U1*dC0/dxi

    Solution via Green's function convolution:
        C1(xi,eta,tau) = -int_0^tau int_Omega
                          G(xi,eta; xi1,eta1; tau-t)
                          * U1(xi1,eta1) * dC0/dxi1 d_xi1 d_eta1 dt

    where G is the Green's function (Bolster 2007, eq. 29):
        G = 4/zeta * sum_{p>=1} sum_{m>=0}
              sin(p*pi*xi)*sin(p*pi*xi1)
              * cos(m*pi*eta/zeta)*cos(m*pi*eta1/zeta)
              * exp(-pi^2*p^2/Pe * dt) * exp(-pi^2*m^2/(Pe*zeta^2) * dt)

    Implementation:
    - Gauss-Legendre quadrature in time (n_t points over [0, tau])
    - Trapezoidal rule in space (n_sp x n_sp/2 source grid)
    - Modal projection to evaluate convolution without full matrix

    Sign: G1 > 0 where wedge accumulates contaminant (xi~0.94, eta~0.48
    at Pe=10). G1 < 0 in the interior where wedge clears contaminant.
    """
    if tau <= 0:
        return np.zeros_like(xi_grid)

    t_nodes, t_wts = np.polynomial.legendre.leggauss(n_t)
    t_nodes = 0.5 * (t_nodes + 1) * tau
    t_wts   = 0.5 * t_wts * tau

    xi1_v  = np.linspace(0.01, 0.99, n_sp)
    eta1_v = np.linspace(0.01, zeta - 0.01, max(n_sp // 2, 8))
    xi1g, eta1g = np.meshgrid(xi1_v, eta1_v)
    dxi1  = xi1_v[1] - xi1_v[0]
    deta1 = eta1_v[1] - eta1_v[0]

    result = np.zeros_like(xi_grid, dtype=float)
    u1_src = U1_IP(xi1g, eta1g, zeta, N)

    for tk, wk in zip(t_nodes, t_wts):
        dt = tau - tk
        if dt < 1e-12:
            continue
        # Source: -U1 * dGamma0/dxi  (V1 term zero since dGamma0/deta=0)
        G0_src  = Gamma0_norm(xi1g, eta1g, tk, Pe, zeta, N)
        dG0dxi  = np.gradient(G0_src, xi1_v, axis=1)
        phi_src = -(u1_src * dG0dxi)

        for p in range(1, N + 1):
            decay_p  = np.exp(-np.pi**2 * p**2 / Pe * dt)
            sinp_out = np.sin(p * np.pi * xi_grid)
            sinp_src = np.sin(p * np.pi * xi1g)
            for m in range(0, 5):
                norm     = (4.0 / zeta) if m > 0 else (2.0 / zeta)
                decay_m  = np.exp(-np.pi**2 * m**2 / (Pe * zeta**2) * dt)
                cosm_out = np.cos(m * np.pi * eta_grid / zeta)
                cosm_src = np.cos(m * np.pi * eta1g / zeta)
                proj = np.sum(sinp_src * cosm_src * phi_src) * dxi1 * deta1
                result += (wk * norm * decay_p * decay_m
                           * sinp_out * cosm_out * proj)
    return result


def domain_avg(C, xi_v, eta_v, zeta=ZETA):
    """Domain-averaged concentration (trapezoid rule)."""
    return np.trapezoid(np.trapezoid(C, eta_v, axis=0), xi_v) / zeta


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SPATIAL GRID
# ─────────────────────────────────────────────────────────────────────────────
N_XI, N_ETA = 55, 28
xi_v  = np.linspace(0.02, 0.98, N_XI)
eta_v = np.linspace(0.02, ZETA - 0.02, N_ETA)
xig, etag = np.meshgrid(xi_v, eta_v)

# Index of seaward trapping zone on shared grid
ix_trap = np.argmin(abs(xi_v - XI_TRAP))
ie_trap = np.argmin(abs(eta_v - ETA_TRAP))

# Pe values and plot styles used across figures
PE_LIST   = [1, 10, 100]
COLORS_PE = ['#1f77b4', '#ff7f0e', '#2ca02c']
LS_PE     = ['-', '--', '-.']


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: TIME EVOLUTION UNDER SLR
# ─────────────────────────────────────────────────────────────────────────────
def make_figure1(Pe=10, alpha0=0.5, eps=0.8, n_sp=20, n_t=14):
    """
    Three-panel figure: full concentration field at tau = 0.3, 0.6, 1.0 * T0.

    SLR case (alpha = alpha0 + eps*tau): solid C=0.5 contour.
    Static wedge (alpha = alpha0 fixed):  dashed C=0.5 contour.

    The progressive inland shift of solid relative to dashed illustrates
    the zone migration predicted by eq. (zone_shift) in the paper.
    Color fill shows the full normalized concentration field for the SLR case.

    Pe=10 is used because it sits at the minimum of lambda1 (slowest flushing,
    most interesting spatial structure) and is central to the realistic range.
    """
    T0       = 1.0 / lambda1(Pe)
    tau_vals = [0.3 * T0, 0.6 * T0, T0]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    cmap   = plt.cm.RdYlBu_r
    levels = np.linspace(0, 1, 13)

    print(f"  Fig 1: Pe={Pe}, T0={T0:.3f}")

    for k, (ax, tau) in enumerate(zip(axes, tau_vals)):
        alpha_slr    = alpha0 + eps * tau
        alpha_static = alpha0

        G0 = Gamma0_norm(xig, etag, tau, Pe)
        G1 = Gamma1_norm(xig, etag, tau, Pe, n_sp=n_sp, n_t=n_t)

        C_slr    = np.clip(G0 + alpha_slr    * G1, 0, None)
        C_static = np.clip(G0 + alpha_static * G1, 0, None)
        Cmax = max(C_slr.max(), C_static.max())
        if Cmax > 0:
            C_slr /= Cmax; C_static /= Cmax

        cf = ax.contourf(xi_v, eta_v, C_slr,
                         levels=levels, cmap=cmap, extend='both')
        ax.contour(xi_v, eta_v, C_static, levels=[0.5],
                   colors='k', linewidths=1.5, linestyles='--')
        ax.contour(xi_v, eta_v, C_slr,    levels=[0.5],
                   colors='k', linewidths=2.0, linestyles='-')

        ax.set_xlabel(r'$\xi$', fontsize=11)
        ax.set_title(r'$\varepsilon\tau={:.2f}$'.format(eps * tau), fontsize=11)
        ax.set_aspect('equal')
        ax.text(0.97, 0.95, r'$\alpha={:.2f}$'.format(alpha_slr),
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

        if k == 0:
            ax.set_ylabel(r'$\eta$', fontsize=11)
            leg = [Line2D([0],[0], color='k', lw=2.0, ls='-',  label='SLR'),
                   Line2D([0],[0], color='k', lw=1.5, ls='--', label='Static')]
            ax.legend(handles=leg, loc='upper left', fontsize=8.5,
                      framealpha=0.9, handlelength=1.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, ZETA)

    plt.colorbar(cf, ax=axes.tolist(), label=r'$\Gamma/\Gamma_{\max}$',
                 shrink=0.88, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    fig.suptitle(
        r'Natural attenuation: SLR (solid) vs.\ static wedge (dashed, $C=0.5$)'
        '\n'
        r'$\mathrm{Pe}=%.0f,\;\alpha_0=%.1f,\;\varepsilon=%.1f$'
        % (Pe, alpha0, eps),
        y=1.04, fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR + f'fig1_zones.{ext}',
                    bbox_inches='tight', dpi=150)
    print("  Figure 1 saved.")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1b: SPATIAL REDISTRIBUTION (single panel)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure1b(Pe=10, alpha_ref=1.0, n_sp=24, n_t=16):
    """
    Single-panel figure: DeltaC = C(alpha=alpha_ref) - C(alpha=0).
    Computed at Pe=10, tau=0.5*T0.

    Answers the question: where does the wedge trap contaminant?
    Red = accumulation (DeltaC > 0) at far seaward boundary (~xi=0.94).
    Blue = depletion  (DeltaC < 0) in the aquifer interior (~xi=0.73).
    Solid black: DeltaC=0 boundary between the two zones.
    Dashed: reference C(alpha=0) contours at 25%, 50%, 75% for context.
    Star: seaward trapping zone monitoring point, used in Figure 2b.

    The trapping zone location (XI_TRAP, ETA_TRAP) = (0.94, 0.48) is
    verified as the point of maximum G1 > 0 at this Pe and tau.
    """
    T0  = 1.0 / lambda1(Pe)
    tau = 0.5 * T0

    # Finer grid for smoother contours in this single-panel figure
    n_xi_f, n_eta_f = 70, 35
    xi_f  = np.linspace(0.01, 0.99, n_xi_f)
    eta_f = np.linspace(0.01, ZETA - 0.01, n_eta_f)
    xif, etaf = np.meshgrid(xi_f, eta_f)

    print(f"  Fig 1b: Pe={Pe}, tau={tau:.4f} (= 0.5*T0)...")
    G0 = Gamma0_norm(xif, etaf, tau, Pe)
    G1 = Gamma1_norm(xif, etaf, tau, Pe, n_sp=n_sp, n_t=n_t)

    vmax = G0.max() if G0.max() > 0 else 1.0
    C0   = G0 / vmax
    C1   = np.clip(G0 + alpha_ref * G1, 0, None) / vmax
    dC   = C1 - C0
    print(f"    dC range: [{dC.min():.4f}, {dC.max():.4f}]")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    clim   = max(abs(dC.min()), abs(dC.max()))
    levels = np.linspace(-clim, clim, 21)
    cf = ax.contourf(xi_f, eta_f, dC, levels=levels,
                     cmap='RdBu_r', extend='both')

    # Zero contour
    ax.contour(xi_f, eta_f, dC, levels=[0.0],
               colors='k', linewidths=1.5, linestyles='-')

    # Reference concentration contours
    ax.contour(xi_f, eta_f, C0, levels=[0.25, 0.50, 0.75],
               colors='k', linewidths=0.6, linestyles='--', alpha=0.4)

    # Monitoring point star
    ax.plot(XI_TRAP, ETA_TRAP, 'k*', ms=12, zorder=5,
            label=r'Monitoring point $(\xi,\eta)=(%.2f,\,%.2f)$'
                  % (XI_TRAP, ETA_TRAP))

    plt.colorbar(cf, ax=ax,
                 label=r'$\Delta C = C(\alpha=%.0f) - C(\alpha=0)$' % alpha_ref,
                 pad=0.02, shrink=0.95)

    ax.set_xlabel(r'$\xi$  (land $\leftarrow\quad\rightarrow$ sea)',
                  fontsize=12)
    ax.set_ylabel(r'$\eta$', fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, ZETA)
    ax.set_aspect('equal')
    ax.text(0.02, 0.08, 'land', transform=ax.transAxes,
            fontsize=9, color='grey')
    ax.text(0.93, 0.08, 'sea',  transform=ax.transAxes,
            fontsize=9, color='grey')

    ax.text(0.80, 0.42, r'$\Delta C > 0$' + '\n(accumulation)',
            fontsize=8.5, ha='center', color='#b2182b',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))
    ax.text(0.52, 0.42, r'$\Delta C < 0$' + '\n(depletion)',
            fontsize=8.5, ha='center', color='#2166ac',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title(
        r'Wedge-driven redistribution: $\Delta C = C(\alpha=%.0f)-C(\alpha=0)$'
        '\n'
        r'$\mathrm{Pe}=%d$, $\tau=0.5\,T_0$.  '
        r'Solid: $\Delta C=0$.  Dashed: $C(\alpha=0)$ contours.'
        % (alpha_ref, Pe),
        fontsize=10)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR + f'fig1b_mechanism.{ext}',
                    bbox_inches='tight', dpi=150)
    print("  Figure 1b saved.")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: CONCENTRATION HISTORY (two panels)
# ─────────────────────────────────────────────────────────────────────────────
def make_figure2(n_tau=30, n_sp=18, n_t=12):
    """
    Two-panel figure demonstrating the two-speed flushing behavior.
    alpha0 = 0 throughout (start from unperturbed state).

    Panel (a) -- Global average, semilogy, Pe=1,10,100:
        eps=1 (solid) vs eps=0 (dashed) for each Pe.
        Curves nearly indistinguishable -> mu1=0 confirmed visually.
        Pe=10 decays slowest (minimum of lambda1 at Pe*=2*pi).
        Uses Gamma0_raw for domain averaging to get correct temporal decay.

    Panel (b) -- Local vs global, Pe=10, eps=0,1,3,6:
        LOCAL  = concentration at (XI_TRAP, ETA_TRAP) where G1 > 0.
                 Thick lines. Decays more slowly as eps increases.
                 At eps=6, rises above initial value -> trapping signature.
        GLOBAL = domain-averaged concentration. Thin lines, same color.
                 Barely changes with eps -> global flushing unaffected.
        All eps values within perturbation validity (alpha(T0) <= 1.72).
        eps* ~ 14 is outside the perturbation regime (noted on figure).
    """
    alpha0      = 0.0
    eps_a       = 1.0
    eps_b_vals  = [0, 1, 3, 6]
    eps_b_cols  = ['#2166ac', '#74add1', '#f46d43', '#d73027']
    eps_b_ls    = ['-', '--', '-.', ':']
    Pe_b        = 10.0
    T0_b        = 1.0 / lambda1(Pe_b)

    # ── Panel (b): Pe=10, varying eps ────────────────────────────────────
    tau_b = np.linspace(0.01 * T0_b, 2.2 * T0_b, n_tau)
    print("  Fig 2 panel (b)...")
    acc_b = {e: [] for e in eps_b_vals}
    avg_b = {e: [] for e in eps_b_vals}
    for tau in tau_b:
        G0 = Gamma0_norm(xig, etag, tau, Pe_b)
        G1 = Gamma1_norm(xig, etag, tau, Pe_b, n_sp=n_sp, n_t=n_t)
        for eps in eps_b_vals:
            C = np.clip(G0 + (alpha0 + eps * tau) * G1, 0, None)
            acc_b[eps].append(C[ie_trap, ix_trap])
            avg_b[eps].append(domain_avg(C, xi_v, eta_v))
    for eps in eps_b_vals:
        a0 = acc_b[eps][0]; v0 = avg_b[eps][0]
        acc_b[eps] = np.array(acc_b[eps]) / a0
        avg_b[eps] = np.array(avg_b[eps]) / v0

    # ── Panel (a): global decay, three Pe values ──────────────────────────
    print("  Fig 2 panel (a)...")
    global_curves = {}
    for Pe in PE_LIST:
        T0_a  = 1.0 / lambda1(Pe)
        tau_a = np.linspace(0.005 * T0_a, 3.0 * T0_a, n_tau)
        for eps in [0, eps_a]:
            avg_v = []
            for tau in tau_a:
                # Raw G0 for correct domain-average temporal decay
                G0r  = Gamma0_raw(xig, etag, tau, Pe)
                G1   = Gamma1_norm(xig, etag, tau, Pe, n_sp=n_sp, n_t=n_t)
                vmax = G0r.max() if G0r.max() > 0 else 1.0
                C    = np.clip(G0r + (alpha0 + eps * tau) * G1 * vmax,
                               0, None)
                avg_v.append(domain_avg(C, xi_v, eta_v))
            avg_v = np.array(avg_v) / avg_v[0]
            global_curves[(Pe, eps)] = (tau_a / T0_a, avg_v)
        print(f"    Pe={Pe} done")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.0))

    # Panel (a)
    all_a  = np.concatenate([v for (Pe,eps),(t,v) in global_curves.items()])
    ymin_a = max(all_a[all_a > 1e-4].min() * 0.4, 1e-4)

    for Pe, col, lst in zip(PE_LIST, COLORS_PE, LS_PE):
        t0, c0 = global_curves[(Pe, 0)]
        t1, c1 = global_curves[(Pe, eps_a)]
        ax_a.semilogy(t0, c0, color=col, ls='--', lw=1.5, alpha=0.6)
        ax_a.semilogy(t1, c1, color=col, ls=lst,  lw=2.2,
                      label=r'$\mathrm{Pe}=%d$' % Pe)

    ax_a.axhline(np.exp(-1), color='grey', lw=0.8, ls=':')
    ax_a.text(0.05, np.exp(-1) * 0.72, r'$e^{-1}$', fontsize=9, color='grey')
    ax_a.set_ylim(ymin_a, 1.5); ax_a.set_xlim(0, 3.0)
    ax_a.set_xlabel(r'$\tau / T_0$')
    ax_a.set_ylabel(r'$\langle\Gamma\rangle / \langle\Gamma\rangle_0$')
    ax_a.set_title('(a) Global average: SLR (solid) vs static (dashed)\n'
                   r'$\varepsilon=1$, $\alpha_0=0$ -- nearly indistinguishable')
    ax_a.legend(loc='upper right', framealpha=0.9)

    # Panel (b)
    all_b  = np.concatenate([list(acc_b[e]) + list(avg_b[e])
                              for e in eps_b_vals])
    ymin_b = all_b.min() * 0.88
    ymax_b = all_b.max() * 1.08

    for eps, col, lst in zip(eps_b_vals, eps_b_cols, eps_b_ls):
        alpha_at_T0 = alpha0 + eps * T0_b
        lbl = r'$\varepsilon=%d$  ($\alpha(T_0)=%.1f$)' % (eps, alpha_at_T0)
        ax_b.plot(tau_b / T0_b, acc_b[eps],
                  color=col, ls=lst, lw=2.2, label=lbl)      # local: thick
        ax_b.plot(tau_b / T0_b, avg_b[eps],
                  color=col, ls=lst, lw=0.9, alpha=0.45)      # global: thin

    ax_b.axhline(np.exp(-1), color='grey', lw=0.8, ls=':')
    ax_b.text(0.05, np.exp(-1) * 0.91, r'$e^{-1}$', fontsize=9, color='grey')
    ax_b.axhline(1.0, color='grey', lw=0.5, ls=':')
    ax_b.set_ylim(ymin_b, ymax_b); ax_b.set_xlim(0, 2.2)

    ax_b.text(0.98, 0.97,
              r'$\varepsilon^*\approx14$ outside' + '\nperturbation range',
              transform=ax_b.transAxes, fontsize=7.5, ha='right', va='top',
              color='#555555',
              bbox=dict(boxstyle='round,pad=0.25', fc='#f5f5f5', alpha=0.9))

    eps_handles   = [Line2D([0],[0], color=c, ls=l, lw=2.2,
                            label=r'$\varepsilon=%d$' % e)
                     for e, c, l in zip(eps_b_vals, eps_b_cols, eps_b_ls)]
    style_handles = [
        Line2D([0],[0], color='k', lw=2.2, label='seaward trap. zone'),
        Line2D([0],[0], color='k', lw=0.9, alpha=0.5, label='global avg')]
    ax_b.legend(handles=eps_handles + style_handles,
                loc='upper left', framealpha=0.9, fontsize=8.5, ncol=2)

    ax_b.set_xlabel(r'$\tau / T_0$')
    ax_b.set_ylabel(r'$\Gamma / \Gamma_0$')
    ax_b.set_title('(b) Seaward trapping zone (thick) vs global (thin)\n'
                   r'$\mathrm{Pe}=10$, $\alpha_0=0$: '
                   r'higher $\varepsilon$ retards local cleanup')

    fig.suptitle('Contaminant concentration history under sea level rise',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR + f'fig2_flushing.{ext}',
                    bbox_inches='tight', dpi=150)
    print("  Figure 2 saved.")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: REGIME DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
def make_figure3():
    """
    Log-log regime diagram in (Pe, eps) space.

    Flushing-trapping boundary:
        eps*(Pe, alpha0) = lambda1(Pe) / (zeta/2 * (1 + alpha0*g))
    Analytical minimum: Pe*=2*pi, eps*_min=4*pi~12.57 (zeta=0.5).
    Three alpha0 curves nearly parallel (ratio ~0.89, independent of Pe),
    confirming weak sensitivity of the threshold to initial intrusion state.

    Example aquifer systems (Section 4 of paper):
        Ex 1: K=1e-4 m/s, L=500m, Dh0=0.5m -> Pe~10, eps~0.5 (FLUSHING)
        Ex 2: K=1e-5 m/s, L=300m, Dh0=0.1m -> Pe~5,  eps~50 (TRAPPING)
    """
    Pe_arr      = np.logspace(-0.3, 2.2, 100)
    alpha0_list = [0.0, 0.5, 1.0]
    colors3     = ['#2166ac', '#f4a582', '#d6604d']
    ls3         = ['-', '--', '-.']
    ylim_top    = 500.0

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    eps_curves = {}
    for alpha0_v, col, lst in zip(alpha0_list, colors3, ls3):
        eps_v = np.array([eps_star(Pe, alpha0_v) for Pe in Pe_arr])
        eps_curves[alpha0_v] = eps_v
        ax.loglog(Pe_arr, eps_v, color=col, ls=lst, lw=2.5,
                  label=r'$\alpha_0=%.1f$' % alpha0_v)

    # Shade flushing / trapping regions
    e0 = eps_curves[0.0]
    ax.fill_between(Pe_arr, 1e-2, e0,       alpha=0.10, color='#2166ac')
    ax.fill_between(Pe_arr, e0,  ylim_top,  alpha=0.10, color='#d6604d')

    ax.text(0.72, 0.20, 'FLUSHING', transform=ax.transAxes,
            fontsize=13, color='#2166ac', alpha=0.85,
            ha='center', style='italic', fontweight='bold')
    ax.text(0.20, 0.78, 'TRAPPING', transform=ax.transAxes,
            fontsize=13, color='#d6604d', alpha=0.85,
            ha='center', style='italic', fontweight='bold')

    # Typical coastal Pe range
    ax.axvspan(10, 100, alpha=0.07, color='grey',
               label=r'Typical coastal Pe')

    # Analytical minimum Pe* = 2*pi
    idx_min = np.argmin(eps_curves[0.0])
    Pe_min  = Pe_arr[idx_min]
    eps_min = eps_curves[0.0][idx_min]
    ax.plot(Pe_min, eps_min, 'ko', ms=7, zorder=5)
    ax.annotate(r'$\mathrm{{Pe}}^*=2\pi\approx{:.1f}$'.format(Pe_min),
                xy=(Pe_min, eps_min),
                xytext=(Pe_min * 3.5, eps_min * 0.4),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='k', lw=1.0))
    ax.text(Pe_min * 1.1, eps_min * 1.7,
            r'$\varepsilon^*_{\min}=4\pi\approx12.6$',
            fontsize=8, color='k')

    # Vertical lines at Pe=1,10,100
    for Pe_mark, col in zip(PE_LIST, COLORS_PE):
        ax.axvline(Pe_mark, color=col, lw=1.0, ls=':', alpha=0.7)
        ax.text(Pe_mark * 1.12, 0.02,
                r'$\mathrm{Pe}=%d$' % Pe_mark,
                fontsize=8, color=col, rotation=90, va='bottom')

    # Example aquifer systems
    ax.plot(10.0, 0.5,  's', color='#1a9641', ms=10, zorder=6,
            label=r'Ex. 1 (high-$K$, flushing)')
    ax.plot(5.0,  50.0, '^', color='#d73027', ms=10, zorder=6,
            label=r'Ex. 2 (low-$K$, trapping)')
    ax.annotate('Ex. 1', xy=(10.0, 0.5),  xytext=(3.5, 0.12),
                fontsize=8, color='#1a9641',
                arrowprops=dict(arrowstyle='->', color='#1a9641', lw=0.8))
    ax.annotate('Ex. 2', xy=(5.0, 50.0),  xytext=(1.5, 120.0),
                fontsize=8, color='#d73027',
                arrowprops=dict(arrowstyle='->', color='#d73027', lw=0.8))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"P\'{e}clet number $\mathrm{Pe}$")
    ax.set_ylabel(r'Dimensionless SLR rate $\varepsilon$')
    ax.set_xlim(Pe_arr[0], Pe_arr[-1])
    ax.set_ylim(0.01, ylim_top)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=9)
    ax.set_title(r'Regime diagram: $\varepsilon^*(\mathrm{Pe},\alpha_0)$')

    print(f"\n  Regime diagram key values:")
    print(f"    Pe* = {2*np.pi:.4f}  (= 2*pi)")
    print(f"    eps*_min = {4*np.pi:.4f}  (= 4*pi, zeta=0.5)")
    for Pe in [1, 5, 10, 20, 50, 100]:
        print(f"    Pe={Pe:3d}: eps*(0)={eps_star(Pe,0):.2f}  "
              f"eps*(0.5)={eps_star(Pe,0.5):.2f}  "
              f"eps*(1.0)={eps_star(Pe,1.0):.2f}")

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR + f'fig3_regime.{ext}',
                    bbox_inches='tight', dpi=150)
    print("  Figure 3 saved.")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 60)
    print("Generating figures for Bolster et al. (in prep.)")
    print("=" * 60)

    print("\nFigure 1: time evolution under SLR (Pe=10)...")
    make_figure1()

    print("\nFigure 1b: spatial redistribution (Pe=10)...")
    make_figure1b()

    print("\nFigure 2: concentration history (Pe=1,10,100)...")
    make_figure2()

    print("\nFigure 3: regime diagram...")
    make_figure3()

    print("\n" + "=" * 60)
    print("All figures saved to:", OUTPUT_DIR)
    print("=" * 60)
