"""Mechanistic components of VECTRA: thermal suitability, temperature-dependent
extrinsic incubation and generation interval, and intervention effects on the
reproduction number.

Anchors taken from the literature (not fitted here):
  * Ae. aegypti dengue R0 thermal limits/optimum: 17.8, 29.1, 34.6 degC (Mordecai et al., 2017).
  * Mean EIP 15 d at 25 degC and 6.5 d at 30 degC (Chan & Johansson, 2012) -> log-linear EIP(T).
  * Mean intrinsic incubation 5.9 d (Chan & Johansson, 2012).
Assumptions (varied in the sensitivity analysis): host-to-vector and vector-to-host contact delay
C0 = 4 d, baseline adult mortality MU0 = 0.10 / d, intervention efficacies E_L, E_A, E_S.
"""
import numpy as np, torch

T0, TOPT, TM = 17.8, 29.1, 34.6
R_SHAPE = (TM - TOPT) / (TOPT - T0)
EIP25, EIP30 = 15.0, 6.5
KAPPA = np.log(EIP25 / EIP30) / 5.0
IIP, C0 = 5.9, 4.0
MU0 = 0.10
GI_CV = 0.5
K_GI = 8                       # renewal kernel support (weeks)
ACTIONS = ("larviciding", "adulticiding", "source_reduction")
EFF = dict(larviciding=0.30, adulticiding=0.20, source_reduction=0.20)
COST = dict(larviciding=1.0, adulticiding=1.5, source_reduction=0.8)


def log_suitability(T):
    """log of the Yan-Hunt beta-shaped thermal suitability with exponent 1 (peak value 0 at TOPT)."""
    lib = torch if torch.is_tensor(T) else np
    T = lib.clip(T, T0 + 0.3, TM - 0.3)
    return lib.log((T - T0) / (TOPT - T0)) + R_SHAPE * lib.log((TM - T) / (TM - TOPT))


def eip_days(T):
    lib = torch if torch.is_tensor(T) else np
    return lib.clip(EIP25 * lib.exp(-KAPPA * (T - 25.0)), 4.0, 40.0)


def gi_mean_weeks(T):
    return (IIP + eip_days(T) + C0) / 7.0


def gi_kernel(m):
    """Discretised gamma generation-interval kernel with mean m (weeks) and CV GI_CV,
    evaluated on lags 1..K_GI. Works for numpy arrays or torch tensors of any shape."""
    is_t = torch.is_tensor(m)
    lib = torch if is_t else np
    shape = 1.0 / GI_CV ** 2
    k = lib.arange(1, K_GI + 1, dtype=m.dtype if is_t else float)
    if is_t:
        k = k.to(m.device)
    scale = m[..., None] / shape
    # log-density of gamma at bin centres, normalised over the support
    if is_t:
        logp = (shape - 1) * torch.log(k) - k / scale - shape * torch.log(scale) - torch.lgamma(torch.tensor(shape))
        return torch.softmax(logp, dim=-1)
    from scipy.special import gammaln
    logp = (shape - 1) * np.log(k) - k / scale - shape * np.log(scale) - gammaln(shape)
    p = np.exp(logp - logp.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def phi(action, T, eff=None, mu0=MU0):
    """Reduction of log R under full coverage of an action at mean temperature T.
    Larviciding / source reduction scale adult density; adulticiding raises mortality, which
    lowers equilibrium density (1/mu) and survival through the EIP (exp(-mu*EIP))."""
    eff = EFF if eff is None else eff
    lib = torch if torch.is_tensor(T) else np
    e = eff[action]
    if action == "adulticiding":
        return 2.0 * np.log1p(e) + mu0 * e * eip_days(T)
    return -np.log1p(-e) + 0.0 * T
