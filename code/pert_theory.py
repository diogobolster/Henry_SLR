"""First-order perturbation transport predictions with an arbitrary
first-order velocity field U1 (analytic formula or numerical linearized
field). Green's-function convolution as in Bolster (2007) eq. 31, but with
raw (unnormalized) Gamma0 source for a clean quantitative comparison.

Supports the quasi-static ansatz  Gamma = Gamma0 + alpha(tau)*Gamma1(tau)
and the exact first-order response for time-varying alpha:
   Gamma1_exact(tau) = -int_0^tau int G(.,tau-t) alpha(t) U1 dGamma0/dxi dt
(the ansatz pulls alpha(tau) outside the convolution).
"""
import numpy as np
from figures import Gamma0_raw, ZETA


def gamma1_field(eval_X, eval_Y, tau, Pe, U1_at, zeta=ZETA,
                 Np=20, Nm=6, n_sp=26, n_t=16, alpha_of_t=None):
    """Gamma1 (raw scale) on eval grid at time tau.
    U1_at(xi_grid, eta_grid) -> U1 values.
    If alpha_of_t is given, returns the *exact first-order field*
    (alpha inside the convolution); else the unit-alpha Gamma1."""
    if tau <= 0:
        return np.zeros_like(eval_X)
    t_nodes, t_wts = np.polynomial.legendre.leggauss(n_t)
    t_nodes = 0.5 * (t_nodes + 1) * tau
    t_wts = 0.5 * t_wts * tau

    xi1 = np.linspace(0.005, 0.995, n_sp)
    eta1 = np.linspace(0.005, zeta - 0.005, max(n_sp // 2, 10))
    X1, Y1 = np.meshgrid(xi1, eta1, indexing='ij')
    dxi1 = xi1[1] - xi1[0]
    deta1 = eta1[1] - eta1[0]
    u1 = U1_at(X1, Y1)

    # precompute mode shapes
    sinp_src = [np.sin(p * np.pi * X1) for p in range(Np + 1)]
    cosm_src = [np.cos(m * np.pi * Y1 / zeta) for m in range(Nm + 1)]
    sinp_out = [np.sin(p * np.pi * eval_X) for p in range(Np + 1)]
    cosm_out = [np.cos(m * np.pi * eval_Y / zeta) for m in range(Nm + 1)]

    result = np.zeros_like(eval_X, dtype=float)
    for tk, wk in zip(t_nodes, t_wts):
        dt = tau - tk
        if dt < 1e-12:
            continue
        G0 = Gamma0_raw(X1, Y1, tk, Pe, zeta)
        dG0 = np.gradient(G0, xi1, axis=0)
        src = -(u1 * dG0)
        if alpha_of_t is not None:
            src = src * alpha_of_t(tk)
        for p in range(1, Np + 1):
            dp = np.exp(-np.pi**2 * p**2 / Pe * dt)
            if dp < 1e-14:
                continue
            for m in range(0, Nm + 1):
                norm = (4.0 / zeta) if m > 0 else (2.0 / zeta)
                dm = np.exp(-np.pi**2 * m**2 / (Pe * zeta**2) * dt)
                proj = np.sum(sinp_src[p] * cosm_src[m] * src) * dxi1 * deta1
                result += (wk * norm * dp * dm * proj
                           * sinp_out[p] * cosm_out[m])
    return result


def predict_histories(taus, Pe, U1_at, eps=0.0, alpha0=0.0, pts=(),
                      zeta=ZETA, ngrid=(61, 31), exact_conv=False):
    """Return dict with domain-mean and local histories of the
    perturbation prediction Gamma0 + alpha*Gamma1 at the given taus."""
    xv = np.linspace(0.0, 1.0, ngrid[0])
    yv = np.linspace(0.0, zeta, ngrid[1])
    X, Y = np.meshgrid(xv, yv, indexing='ij')
    mean_h, loc_h = [], {p: [] for p in pts}
    for tau in taus:
        G0 = Gamma0_raw(X, Y, max(tau, 1e-8), Pe, zeta)
        if eps == 0.0 and alpha0 == 0.0:
            G = G0
        elif exact_conv and eps != 0.0:
            a_of_t = (lambda t: alpha0 + eps * t)
            G1x = gamma1_field(X, Y, tau, Pe, U1_at, zeta,
                               alpha_of_t=a_of_t)
            G = G0 + G1x
        else:
            G1 = gamma1_field(X, Y, tau, Pe, U1_at, zeta)
            G = G0 + (alpha0 + eps * tau) * G1
        G = np.clip(G, 0.0, None)
        mean_h.append(np.trapezoid(np.trapezoid(G, yv, axis=1), xv) / zeta)
        for p in pts:
            i = np.argmin(abs(xv - p[0])); j = np.argmin(abs(yv - p[1]))
            loc_h[p].append(G[i, j])
    return (np.array(mean_h),
            {p: np.array(v) for p, v in loc_h.items()})
