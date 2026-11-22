"""Vector-control allocation: mean LP, CVaR LP (Rockafellar-Uryasev) and SPO+ decision-focused loss."""
import numpy as np, torch
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
import vectra_mech as M

ACTS = M.ACTIONS
COSTS = np.array([M.COST[a] for a in ACTS])
BUDGET = 6.0
H0 = 3          # interventions act on transmission from week H0 onward


def efficacy(tbar, eff=None, mu0=M.MU0):
    """1 - exp(-phi_a(Tbar)) for each action: array [..., N, H, A]."""
    return np.stack([1 - np.exp(-M.phi(a, tbar, eff, mu0)) for a in ACTS], -1)


CHANNEL = None   # per-region endemic channel; surge control targets the epidemic excess above it


def excess(Y, channel=None):
    ch = CHANNEL if channel is None else channel
    return Y if ch is None else np.maximum(Y - ch[:, None], 0.0)


def averted_matrix(Y, tbar, eff=None, mu0=M.MU0, channel=None):
    """Delta_{i,a} = sum_{h>=H0} (1 - exp(-phi_a(Tbar_{i,h}))) [Y_{i,h} - c_i]^+ .  Y [...,N,H] -> [...,N,A]."""
    E = efficacy(tbar, eff, mu0)
    Z = excess(Y, channel)
    return (E[..., H0 - 1:, :] * Z[..., H0 - 1:, None]).sum(-2)


def _base_constraints(N, budget):
    nA = len(ACTS)
    Aub = lil_matrix((N + 1, N * nA))
    for i in range(N):
        Aub[i, i * nA:(i + 1) * nA] = 1.0
    Aub[N, :] = np.tile(COSTS, N)
    b = np.concatenate([np.ones(N), [budget]])
    return Aub, b


_CACHE = {}


def solve_mean(Delta, budget=BUDGET):
    """max sum x*Delta  s.t. sum_a x_ia <= 1, sum c_a x_ia <= B, 0<=x<=1."""
    N, nA = Delta.shape
    key = (N, budget)
    if key not in _CACHE:
        A_, b_ = _base_constraints(N, budget); _CACHE[key] = (csr_matrix(A_), b_)
    A_, b_ = _CACHE[key]
    r = linprog(-Delta.reshape(-1), A_ub=A_, b_ub=b_, bounds=(0, 1), method="highs")
    return r.x.reshape(N, nA)


def solve_cvar(Ys, Deltas, alpha=0.9, budget=BUDGET):
    """min CVaR_alpha of residual burden  L_s(x) = sum_{i,h>=H0} Y_s - sum x Delta_s."""
    S, N, nA = Deltas.shape
    nx = N * nA
    const = Ys[..., H0 - 1:].sum((-1, -2))                      # [S]
    c = np.concatenate([np.zeros(nx), [1.0], np.full(S, 1.0 / ((1 - alpha) * S))])
    A0, b0 = _base_constraints(N, budget)
    A = lil_matrix((N + 1 + S, nx + 1 + S))
    A[: N + 1, :nx] = A0
    A[N + 1:, :nx] = -Deltas.reshape(S, -1)
    A[N + 1:, nx] = -1.0
    for s in range(S):
        A[N + 1 + s, nx + 1 + s] = -1.0
    b = np.concatenate([b0, -const])
    bounds = [(0, 1)] * nx + [(None, None)] + [(0, None)] * S
    r = linprog(c, A_ub=csr_matrix(A), b_ub=b, bounds=bounds, method="highs")
    return r.x[:nx].reshape(N, nA)


def evaluate(x, Delta_true, budget=BUDGET):
    x_star = solve_mean(Delta_true, budget)
    a, a_star = float((x * Delta_true).sum()), float((x_star * Delta_true).sum())
    return dict(averted=a, oracle=a_star, regret=1 - a / max(a_star, 1e-9))


# ----------------------------------------------------------------------------- SPO+ loss
class SPOPlus:
    """SPO+ surrogate (Elmachtoub & Grigas) for the allocation LP with cost vector c = -Delta / scale.
    Gradient w.r.t. predicted cost is 2 (x*(c) - x*(2c_hat - c))."""

    def __init__(self, W, lam=1.0, budget=BUDGET):
        self.W, self.lam, self.budget = W, lam, budget

    def __call__(self, model, mu, origins, tb, tg, eval_only=False):
        tbf = tb[..., 3:]
        eff = torch.tensor(efficacy(tbf.numpy()), dtype=mu.dtype)                     # [B,N,H,A]
        ch = torch.zeros(mu.shape[1]) if CHANNEL is None else torch.tensor(CHANNEL, dtype=mu.dtype)
        z_hat = torch.relu(mu - ch[None, :, None]); z_true = torch.relu(tg - ch[None, :, None])
        d_hat = (eff[..., H0 - 1:, :] * z_hat[..., H0 - 1:, None]).sum(-2)           # [B,N,A]
        d_true = (eff[..., H0 - 1:, :] * z_true[..., H0 - 1:, None]).sum(-2)
        vals, surr = [], []
        for b in range(mu.shape[0]):
            if (tg[b] == 0).all() or float(d_true[b].sum()) <= 0:
                continue
            scale = float(d_true[b].max(-1).values.sum()) + 1.0
            c = -d_true[b].numpy() / scale
            ch = -d_hat[b] / scale
            x1 = solve_mean(-c, self.budget)
            x2 = solve_mean(-(2 * ch.detach().numpy() - c), self.budget)
            v = -((2 * ch.detach().numpy() - c) * x2).sum() + 2 * (ch.detach().numpy() * x1).sum() - (c * x1).sum()
            vals.append(v)
            surr.append((ch * torch.tensor(2 * (x1 - x2), dtype=mu.dtype)).sum())
        if not vals:
            return 0.0 if eval_only else torch.tensor(0.0)
        if eval_only:
            return self.lam * float(np.mean(vals))
        s = torch.stack(surr).mean()
        return self.lam * (s - s.detach() + float(np.mean(vals)))
