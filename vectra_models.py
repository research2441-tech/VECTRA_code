"""Forecasting models: VECTRA (mechanistic-neural renewal) and neural baselines."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import vectra_mech as M

L_HIST, H = 12, 8
QS = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])


# ----------------------------------------------------------------------------- tensors
class Window:
    """Builds per-origin inputs. Origin t = last observed week; targets t+1..t+H."""

    def __init__(self, D, A, feat_stats=None):
        self.D, self.A = D, A
        y = D["y"]; self.y = y
        Ay = y @ A.T if A.shape[0] > 1 else np.zeros_like(y)
        cum52 = np.array([y[max(0, t - 51): t + 1].sum(0) for t in range(len(y))])
        self.cum52 = cum52
        raw = np.stack([np.log1p(y), np.log1p(Ay), D["temp"], D["hum"], np.log1p(D["rain"]),
                        np.log1p(D["delay"]), np.log1p(cum52)], -1)        # [T,N,7]
        te = D["split"]["train_end"]
        if feat_stats is None:
            mu = raw[: te + 1].reshape(-1, raw.shape[-1]).mean(0)
            sd = raw[: te + 1].reshape(-1, raw.shape[-1]).std(0) + 1e-6
            feat_stats = (mu, sd)
        self.feat_stats = feat_stats
        z = (raw - feat_stats[0]) / feat_stats[1]
        ang = 2 * np.pi * D["woy"] / 52.18
        seas = np.stack([np.sin(ang), np.cos(ang)], -1)[:, None, :].repeat(y.shape[1], 1)
        self.X = np.concatenate([z, seas], -1).astype(np.float32)            # [T,N,9]
        # 4-week trailing mean temperature (observed) and the same quantity with climatology for unseen weeks
        self.temp = D["temp"]; self.clim = D["temp_clim"]

    def pool_init(self):
        te = self.D["split"]["train_end"]
        return 3.0 * self.cum52[: te + 1].max(0) + 100.0

    def tbar_obs(self, t, temp=None, n=3):
        temp = self.temp if temp is None else temp
        return np.stack([temp[s - 3: s + 1].mean(0) for s in range(t - n + 1, t + 1)], 1)

    def tbar_future(self, t, temp=None):
        """Trailing 4-week mean temperature for target weeks t+1..t+H; weeks after t use climatology."""
        temp = self.temp if temp is None else temp
        T, N = temp.shape
        out = np.zeros((N, H))
        for h in range(1, H + 1):
            ws = range(t + h - 4, t + h)
            vals = []
            for w in ws:
                if w <= t:
                    vals.append(temp[w])
                else:
                    woy = self.D["woy"][min(w, T - 1)] if w < T else ((self.D["woy"][t] + (w - t) - 1) % 52) + 1
                    vals.append(self.clim[woy])
            out[:, h - 1] = np.mean(vals, 0)
        return out

    def batch(self, origins, end, X=None, y=None, temp=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        xs, yh, tb, tgt, msk, cum = [], [], [], [], [], []
        for t in origins:
            cum.append(y[max(0, t - 51): t + 1].sum(0))
            xs.append(X[t - L_HIST + 1: t + 1].transpose(1, 0, 2))
            yh.append(y[t - L_HIST + 1: t + 1].T)
            tb.append(np.concatenate([self.tbar_obs(t, temp), self.tbar_future(t, temp)], 1))
            tg = np.zeros((y.shape[1], H)); mk = np.zeros((y.shape[1], H))
            for h in range(1, H + 1):
                if t + h <= end:
                    tg[:, h - 1] = self.y[t + h]; mk[:, h - 1] = 1
            tgt.append(tg); msk.append(mk)
        f = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
        yh_t = f(yh)
        self.last_cum = f(cum)
        return f(xs), torch.cat([yh_t, self.last_cum[..., None]], -1), f(tb), f(tgt), f(msk)


def nb_nll(y, mu, r):
    mu = mu.clamp_min(1e-6)
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
             + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))


def nb_sample(mu, r, n):
    p = (mu / (mu + r)).clamp(1e-6, 1 - 1e-6)
    return torch.distributions.NegativeBinomial(total_count=r, probs=p).sample((n,))


class Encoder(nn.Module):
    def __init__(self, N, f_in=9, hid=32, emb=4):
        super().__init__()
        self.gru = nn.GRU(f_in, hid, batch_first=True)
        self.emb = nn.Embedding(N, emb)
        self.out_dim = hid + emb

    def forward(self, x):                      # x [B,N,L,F]
        B, N, L, Fd = x.shape
        _, h = self.gru(x.reshape(B * N, L, Fd))
        h = h[-1].reshape(B, N, -1)
        e = self.emb.weight[None].expand(B, -1, -1)
        return torch.cat([h, e], -1)


# ----------------------------------------------------------------------------- VECTRA
class VECTRA(nn.Module):
    """log R_{i,t+h} = b_i + gamma * log S(Tbar) + delta_{i,t,h};  lambda = R * sum_k w_k(Tbar) ymix_{t+h-k} + eta_i
    ymix = (1-omega) y_i + omega (A y)_i ; delta from a bounded neural closure."""

    def __init__(self, N, A, thermal=True, gi_temp=True, graph=True, closure=True, bound=2.0, deplete=False,
                 pool_init=None, obs_closure=True, obs_bound=1.0, n_rhat=3):
        super().__init__()
        self.n_rhat = n_rhat
        self.obs_closure, self.obs_bound = obs_closure and closure, obs_bound
        self._nu = None
        self.deplete = deplete
        self.p = nn.Parameter(torch.tensor(np.log(pool_init) if pool_init is not None else np.full(N, 12.0),
                                           dtype=torch.float32))
        self.N = N; self.thermal, self.gi_temp, self.graph, self.closure = thermal, gi_temp, graph and N > 1, closure
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(N))
        self.g_raw = nn.Parameter(torch.tensor(0.0))
        self.w_raw = nn.Parameter(torch.tensor(-2.0))
        self.eta = nn.Parameter(torch.zeros(N))
        self.r_raw = nn.Parameter(torch.full((N,), 1.5))
        self.bound = bound
        self.rho_raw = nn.Parameter(torch.tensor(2.0))
        if closure:
            self.enc = Encoder(N)
            self.head = nn.Sequential(nn.Linear(self.enc.out_dim, 32), nn.GELU(), nn.Linear(32, 3 * H))
        self.dh = nn.Parameter(torch.zeros(H))

    @property
    def gamma(self):
        return F.softplus(self.g_raw) if self.thermal else torch.tensor(0.0)

    @property
    def omega(self):
        return torch.sigmoid(self.w_raw) if self.graph else torch.tensor(0.0)

    def disp(self):
        return self._r_full

    def log_r(self, x, yh, tbar):
        """Origin-anchored reproduction number propagated by temperature:
        log R_{t+h} = rho^h log Rhat_t + (1-rho^h) b_i + gamma [log S(Tbar_{t+h}) - log S(Tbar_t)] + delta_h."""
        y12 = yh[..., :-1]
        tb_obs, tb_fut = tbar[..., :3], tbar[..., 3:]
        w_obs = self.kernel(tb_obs)                                             # [B,N,3,K]
        mix = self.mix(y12)
        lam = []
        for j in range(3):                                                      # s = t-2, t-1, t
            s_idx = y12.shape[-1] - 3 + j
            hist = mix[..., s_idx - M.K_GI: s_idx]                              # lags K..1
            lam.append((hist.flip(-1) * w_obs[..., j, :]).sum(-1))
        lam_list = lam
        nr = self.n_rhat
        lam_n = torch.stack(lam_list[-nr:], -1).sum(-1)
        log_rhat = torch.log((y12[..., -nr:].sum(-1) + 1.0) / (lam_n + 1.0))
        hs = torch.arange(1, H + 1, dtype=tbar.dtype)
        rho = torch.sigmoid(self.rho_raw)
        decay = rho ** hs
        lr = decay * log_rhat[..., None] + (1 - decay) * self.b[None, :, None]
        lr = lr + self.gamma * (M.log_suitability(tb_fut) - M.log_suitability(tb_obs[..., -1:]))
        delta = torch.zeros_like(lr)
        kap = torch.zeros_like(lr)
        self._nu = torch.zeros_like(lr)
        if self.closure:
            o = self.head(self.enc(x))
            delta = self.bound * torch.tanh(o[..., :H] / self.bound)
            kap = 2.0 * torch.tanh(o[..., H:2 * H] / 2.0)
            self._nu = self.obs_bound * torch.tanh(o[..., 2 * H:] / self.obs_bound) if self.obs_closure else torch.zeros_like(lr)
        self._r_full = F.softplus(self.r_raw[None, :, None] + self.dh + kap) + 1e-3
        return lr + delta, delta

    def mix(self, buf):
        if not self.graph:
            return buf
        om = self.omega
        return (1 - om) * buf + om * torch.einsum("ij,...jk->...ik", self.A, buf)

    def kernel(self, tbar):
        m = M.gi_mean_weeks(tbar) if self.gi_temp else torch.full_like(tbar, 2.5)
        return M.gi_kernel(m)                                                  # [B,N,H,K]

    def roll(self, yh, logr, w, sample=0, shift=None):
        """Renewal recursion. yh [B,N,K+1]: last K counts (oldest..newest) and the trailing 52-week total.
        Returns means (sample=0) or samples [S,B,N,H]."""
        yh, cum = yh[..., -1 - M.K_GI:-1], yh[..., -1]
        if sample:
            cum = cum[None].expand(sample, -1, -1).clone()
        pool = torch.exp(self.p)
        r_all = self.disp()
        if sample:
            buf = yh[None].expand(sample, -1, -1, -1).clone()
        else:
            buf = yh.clone()
        outs = []
        om = self.omega
        for h in range(H):
            mix = self.mix(buf)
            lam_in = (mix.flip(-1) * w[..., h, :]).sum(-1)                     # lag 1..K
            lr = logr[..., h] if shift is None else logr[..., h] - shift[..., h]
            if self.deplete:
                lr = lr + torch.log((1 - cum / pool).clamp_min(0.05))
            mu = torch.exp(lr) * lam_in + torch.exp(self.eta)
            r = r_all[..., h]
            nu = self._nu[..., h] if (self.closure and self._nu is not None) else 0.0
            mu_obs = mu * torch.exp(nu) if torch.is_tensor(nu) else mu
            if sample:
                rr = r.expand_as(mu)
                ratio = torch.exp(nu) if torch.is_tensor(nu) else 1.0
                obs = torch.distributions.NegativeBinomial(
                    total_count=rr, probs=(mu_obs / (mu_obs + rr)).clamp(1e-6, 1 - 1e-6)).sample()
                nxt = obs / ratio                     # latent incidence fed back into the renewal
                outs.append(obs)
            else:
                nxt = mu
                outs.append(mu_obs)
            cum = cum + nxt
            buf = torch.cat([buf[..., 1:], nxt[..., None]], -1)
        return torch.stack(outs, -1)

    def forward(self, x, yh, tbar):
        logr, delta = self.log_r(x, yh, tbar)
        w = self.kernel(tbar[..., 3:])
        return self.roll(yh, logr, w), delta

    @torch.no_grad()
    def sample(self, x, yh, tbar, n=200, shift=None, gen=None):
        logr, _ = self.log_r(x, yh, tbar)
        w = self.kernel(tbar[..., 3:])
        return self.roll(yh, logr, w, sample=n, shift=shift)


# ----------------------------------------------------------------------------- neural baselines
class GRUNB(nn.Module):
    """Global recurrent model with negative-binomial likelihood (DeepAR-style likelihood,
    direct multi-horizon head). graph=True mixes hidden states over the coupling graph."""

    def __init__(self, N, A, graph=False):
        super().__init__()
        self.enc = Encoder(N); d = self.enc.out_dim
        self.graph = graph and N > 1
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        if self.graph:
            self.gw = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 2 * H))

    def forward(self, x, yh, tbar):
        z = self.enc(x)
        if self.graph:
            z = z + F.gelu(self.gw(torch.einsum("ij,bjd->bid", self.A, z)))
        o = self.head(z)
        mu = torch.exp(o[..., :H].clamp(-10, 10) + torch.log1p(yh[..., -2:-1]))
        self._r = F.softplus(o[..., H:]) + 1e-3
        return mu, torch.zeros_like(mu)

    def disp(self):
        return self._r

    @torch.no_grad()
    def sample(self, x, yh, tbar, n=200, shift=None, gen=None):
        mu, _ = self.forward(x, yh, tbar)
        if shift is not None:
            mu = mu * torch.exp(-shift)
        return nb_sample(mu, self._r, n)


class LSTMEnc(nn.Module):
    """LSTM encoder used by the LSTM-GBM hybrid baseline (trained on log1p targets with MSE)."""

    def __init__(self, f_in=9, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(f_in, hid, batch_first=True)
        self.head = nn.Linear(hid, H)

    def forward(self, x):
        B, N, L, Fd = x.shape
        o, (h, _) = self.lstm(x.reshape(B * N, L, Fd))
        h = h[-1]
        return self.head(h).reshape(B, N, H), h.reshape(B, N, -1)
