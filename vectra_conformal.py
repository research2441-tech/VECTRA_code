"""Interval calibration: model-native, static split conformal, and adaptive conformal inference
(ACI, Gibbs & Candes 2021) with scores normalised by the model's own predictive spread."""
import numpy as np
from vectra_models import QS

QI = {q: i for i, q in enumerate(QS)}
MAX_HALF = 3.0


def _scale(Q):
    return (np.log1p(Q[..., QI[0.9]]) - np.log1p(Q[..., QI[0.1]])) / 2.5631 + 0.05


def _q(scores, level):
    n = len(scores)
    if n == 0:
        return np.inf
    if level >= 1:
        return np.max(scores) * 2.0
    k = min(n - 1, int(np.ceil((n + 1) * level)) - 1)
    return np.sort(scores)[max(k, 0)]


def calibrate(Q, Y, origins, n_cal, alpha=0.2, mode="aci_region_h", eta=0.05):
    """Q, Y over the concatenated calibration+test origins (chronological). The first n_cal origins
    only provide scores; intervals are returned for the remaining origins.
    mode: raw | split | aci_h | aci_region_h."""
    O, N, H = Y.shape
    m = np.log1p(Q[..., QI[0.5]]); s = _scale(Q)
    score = np.abs(np.log1p(np.nan_to_num(Y, nan=0.0)) - m) / s
    lo = np.zeros((O, N, H)); hi = np.zeros((O, N, H))
    if mode == "raw":
        a = alpha / 2
        ql = min(QS, key=lambda q: abs(q - a)); qh = min(QS, key=lambda q: abs(q - (1 - a)))
        return Q[n_cal:, ..., QI[ql]], Q[n_cal:, ..., QI[qh]]
    keyf = (lambda i, h: (i, h)) if mode == "aci_region_h" else (lambda i, h: h)
    S = {}; A = {}
    # static split: scores from calibration block only, pooled across regions per horizon
    if mode == "split":
        for h in range(H):
            sc = score[:n_cal, :, h][~np.isnan(Y[:n_cal, :, h])]
            qv = _q(sc, 1 - alpha)
            half = np.minimum(qv * s[:, :, h], MAX_HALF)
            lo[:, :, h] = np.expm1(m[:, :, h] - half); hi[:, :, h] = np.expm1(m[:, :, h] + half)
        return np.clip(lo[n_cal:], 0, None), hi[n_cal:]
    t_of = {t: o for o, t in enumerate(origins)}
    for o, t in enumerate(origins):
        # reveal every target whose week is now observed (origin o' with o'+h == t)
        for h in range(1, H + 1):
            op = t_of.get(t - h)
            if op is None:
                continue
            for i in range(N):
                if np.isnan(Y[op, i, h - 1]):
                    continue
                k = keyf(i, h - 1)
                sc = S.setdefault(k, [])
                at = A.setdefault(k, alpha)
                q_used = _q(np.array(sc), 1 - at) if sc else np.inf
                err = float(score[op, i, h - 1] > q_used)
                if o >= n_cal or True:
                    A[k] = at + eta * (alpha - err)
                sc.append(score[op, i, h - 1])
        for i in range(N):
            for h in range(H):
                k = keyf(i, h)
                sc = np.array(S.get(k, []))
                qv = _q(sc, 1 - A.get(k, alpha)) if len(sc) else 3.0
                half = min(qv * s[o, i, h], MAX_HALF)          # cap: at most a factor of e^3 around the median
                lo[o, i, h] = np.expm1(m[o, i, h] - half); hi[o, i, h] = np.expm1(m[o, i, h] + half)
    return np.clip(lo[n_cal:], 0, None), hi[n_cal:]


def interval_metrics(lo, hi, Y, alpha):
    ok = ~np.isnan(Y)
    cov = ((Y >= lo) & (Y <= hi))[ok].mean()
    width = np.log1p(hi)[ok] - np.log1p(lo)[ok]
    IS = (hi - lo) + (2 / alpha) * (lo - Y) * (Y < lo) + (2 / alpha) * (Y - hi) * (Y > hi)
    return dict(coverage=float(cov), log_width=float(np.mean(width)), interval_score=float(np.nanmean(IS[ok])))
