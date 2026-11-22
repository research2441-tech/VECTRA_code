"""Training loops and baseline forecasters. Every forecaster returns, for a list of origins,
quantiles Q [O,N,H,len(QS)], means [O,N,H] and (optionally) sample paths [S,O,N,H]."""
import time, numpy as np, torch
from scipy.stats import nbinom
import lightgbm as lgb
from sklearn.linear_model import PoissonRegressor
import vectra_models as VM, vectra_mech as M

H, QS = VM.H, VM.QS
T_START = max(VM.L_HIST, M.K_GI) - 1


def origin_sets(D):
    s = D["split"]
    return (list(range(T_START, s["train_end"])), list(range(s["train_end"], s["val_end"])),
            list(range(s["val_end"], s["end"])))


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def _disp(model, mu):
    r = model.disp()
    return r[None, :, None].expand_as(mu) if r.dim() == 1 else r


def train_torch(model, W, D, seed=0, epochs=250, lr=3e-3, lam_g=0.05, patience=30, bs=16, verbose=False,
                dfl=None, refit=False, fixed_epochs=None):
    """Early stopping on the validation year. With refit=True the caller re-trains a fresh model on
    train+validation origins for the selected number of epochs (fixed_epochs) before testing."""
    set_seed(seed)
    tr, va, _ = origin_sets(D)
    te, ve = D["split"]["train_end"], D["split"]["val_end"]
    if fixed_epochs is not None:
        tr = tr + va; te = ve
        epochs, patience = fixed_epochs, 10 ** 9
    btr = W.batch(tr, te); bva = W.batch(va, ve)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        for k in range(0, len(tr), bs):
            idx = perm[k:k + bs]
            x, yh, tb, tg, mk = [a[idx] for a in btr]
            mu, delta = model(x, yh, tb)
            nll = (VM.nb_nll(tg, mu, _disp(model, mu)) * mk).sum() / mk.sum()
            loss = nll + lam_g * (delta ** 2).mean()
            if dfl is not None:
                loss = loss + dfl(model, mu, [tr[i] for i in idx], tb, tg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            x, yh, tb, tg, mk = bva
            mu, delta = model(x, yh, tb)
            v = ((VM.nb_nll(tg, mu, _disp(model, mu)) * mk).sum() / mk.sum()).item()
            if dfl is not None:
                v = v + dfl(model, mu, va, tb, tg, eval_only=True)
        if fixed_epochs is not None:
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}; best = v
            continue
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(ep, round(v, 4))
    model.load_state_dict(best_state)
    return dict(val_loss=best, epochs=ep + 1, best_epoch=ep + 1 - (bad if fixed_epochs is None else 0),
                train_time=time.time() - t0)


def predict_torch(model, W, origins, end, n=200, seed=0, X=None, y=None, temp=None, shift=None):
    set_seed(seed)
    model.eval()
    x, yh, tb, _, _ = W.batch(origins, end, X=X, y=y, temp=temp)
    with torch.no_grad():
        mu, _ = model(x, yh, tb)
        S = model.sample(x, yh, tb, n=n, shift=shift).numpy()        # [S,O,N,H]
    Q = np.moveaxis(np.quantile(S, QS, axis=0), 0, -1)
    return dict(Q=Q, mean=S.mean(0), S=S.astype(np.float32), tbar=tb.numpy()[..., 3:])


# ----------------------------------------------------------------------------- tabular baselines
def tab_features(W, t, h=None):
    X = W.X
    f = [X[t], X[t - 1][:, :2], X[t - 2][:, :2], X[t - 3][:, :2], X[max(t - 7, 0)][:, :1]]
    return np.concatenate(f, -1)


def _design(W, origins, end, region_onehot=True):
    rows, tg, mk, tb = [], [], [], []
    N = W.y.shape[1]
    for t in origins:
        F_ = tab_features(W, t)
        tbar = W.tbar_future(t)
        if region_onehot:
            F_ = np.concatenate([F_, np.eye(N)], -1)
        rows.append(F_); tb.append(tbar)
        g = np.full((N, H), np.nan)
        for h in range(1, H + 1):
            if t + h <= end:
                g[:, h - 1] = W.y[t + h]
        tg.append(g)
    return np.array(rows), np.array(tg), np.array(tb)


def fit_set(D, refit):
    tr, va, _ = origin_sets(D)
    return (tr + va, D["split"]["val_end"]) if refit else (tr, D["split"]["train_end"])


def persistence(W, D, origins, end, refit=True):
    tr, te = fit_set(D, refit)
    res = {h: [] for h in range(1, H + 1)}
    for t in tr:
        for h in range(1, H + 1):
            if t + h <= te:
                res[h].append(np.log1p(W.y[t + h]) - np.log1p(W.y[t]))
    q = {h: np.quantile(np.concatenate(res[h]), QS) for h in res}
    Q = np.zeros((len(origins), W.y.shape[1], H, len(QS)))
    for o, t in enumerate(origins):
        for h in range(1, H + 1):
            Q[o, :, h - 1] = np.expm1(np.log1p(W.y[t])[:, None] + q[h][None])
    Q = np.clip(Q, 0, None)
    return dict(Q=Q, mean=Q[..., 4])


def glm_nb(W, D, origins, end, refit=True):
    tr, te = fit_set(D, refit)
    Xtr, Ytr, Ttr = _design(W, tr, te)
    Xte, _, Tte = _design(W, origins, end)
    O, N = len(origins), W.y.shape[1]
    Q = np.zeros((O, N, H, len(QS))); MU = np.zeros((O, N, H))
    for h in range(H):
        a = np.concatenate([Xtr, M.log_suitability(Ttr[..., h:h + 1])], -1).reshape(-1, Xtr.shape[-1] + 1)
        yv = Ytr[..., h].reshape(-1); ok = ~np.isnan(yv)
        m = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(a[ok], yv[ok])
        mu_tr = m.predict(a[ok])
        r = np.clip(np.mean(mu_tr ** 2) / max(np.mean((yv[ok] - mu_tr) ** 2 - mu_tr), 1e-6), 0.05, 100)
        b = np.concatenate([Xte, M.log_suitability(Tte[..., h:h + 1])], -1).reshape(-1, Xte.shape[-1] + 1)
        mu = m.predict(b).reshape(O, N)
        MU[:, :, h] = mu
        p = r / (r + mu)
        Q[:, :, h] = nbinom.ppf(QS[None, None], r, p[..., None])
    return dict(Q=Q, mean=MU)


def _lgb_quantiles(Xtr, Ytr, Xte, seed=0):
    O, N = Xte.shape[:2]
    Q = np.zeros((O, N, H, len(QS)))
    for h in range(H):
        a = Xtr[..., h, :] if Xtr.ndim == 4 else Xtr
        yv = np.log1p(Ytr[..., h]).reshape(-1); ok = ~np.isnan(yv)
        A_ = a.reshape(-1, a.shape[-1])[ok]
        B_ = (Xte[..., h, :] if Xte.ndim == 4 else Xte).reshape(-1, A_.shape[-1])
        for k, q in enumerate(QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=seed, verbose=-1)
            m.fit(A_, yv[ok])
            Q[:, :, h, k] = np.expm1(m.predict(B_)).reshape(O, N)
    Q = np.sort(np.clip(Q, 0, None), -1)
    return Q


def _with_tbar(X, T):
    s = M.log_suitability(T)                      # [O,N,H]
    return np.concatenate([np.repeat(X[:, :, None, :], H, 2), s[..., None]], -1)


def lightgbm_q(W, D, origins, end, seed=0, refit=True):
    tr, te = fit_set(D, refit)
    Xtr, Ytr, Ttr = _design(W, tr, te)
    Xte, _, Tte = _design(W, origins, end)
    Q = _lgb_quantiles(_with_tbar(Xtr, Ttr), Ytr, _with_tbar(Xte, Tte), seed)
    return dict(Q=Q, mean=Q[..., 4])


def lstm_gbm(W, D, origins, end, seed=0, epochs=200, patience=25, refit=True):
    set_seed(seed)
    tr, va, _ = origin_sets(D)
    te, ve = D["split"]["train_end"], D["split"]["val_end"]
    enc = VM.LSTMEnc()
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    btr, bva = W.batch(tr, te), W.batch(va, ve)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        for k in range(0, len(tr), 16):
            idx = perm[k:k + 16]
            x, _, _, tg, mk = [a[idx] for a in btr]
            p, _ = enc(x)
            loss = (((p - torch.log1p(tg)) ** 2) * mk).sum() / mk.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x, _, _, tg, mk = bva
            p, _ = enc(x)
            v = ((((p - torch.log1p(tg)) ** 2) * mk).sum() / mk.sum()).item()
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    enc.load_state_dict(state)

    def hid(orig, e):
        with torch.no_grad():
            x = W.batch(orig, e)[0]
            return enc(x)[1].numpy()
    trf, tef = fit_set(D, refit)
    Xtr, Ytr, Ttr = _design(W, trf, tef); Xte, _, Tte = _design(W, origins, end)
    Xtr = np.concatenate([Xtr, hid(trf, tef)], -1); Xte = np.concatenate([Xte, hid(origins, end)], -1)
    Q = _lgb_quantiles(_with_tbar(Xtr, Ttr), Ytr, _with_tbar(Xte, Tte), seed)
    return dict(Q=Q, mean=Q[..., 4])
