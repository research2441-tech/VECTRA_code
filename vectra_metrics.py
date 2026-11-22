"""Forecast and alarm metrics."""
import numpy as np
from vectra_models import QS

ALPHAS = [(0.025, 0.975), (0.05, 0.95), (0.1, 0.9), (0.25, 0.75)]


def targets(y, origins, end, N, H):
    Y = np.full((len(origins), N, H), np.nan)
    for o, t in enumerate(origins):
        for h in range(1, H + 1):
            if t + h <= end:
                Y[o, :, h - 1] = y[t + h]
    return Y


def wis(Q, Y):
    """Weighted interval score (Bracher et al., 2021) with K=4 central intervals and the median."""
    qi = {q: i for i, q in enumerate(QS)}
    m = Q[..., qi[0.5]]
    s = 0.5 * np.abs(Y - m)
    for lo, hi in ALPHAS:
        a = 2 * lo
        l, u = Q[..., qi[lo]], Q[..., qi[hi]]
        IS = (u - l) + (2 / a) * (l - Y) * (Y < l) + (2 / a) * (Y - u) * (Y > u)
        s = s + (a / 2) * IS
    return s / (len(ALPHAS) + 0.5)


def coverage(lo, hi, Y):
    ok = ~np.isnan(Y)
    return float(((Y >= lo) & (Y <= hi))[ok].mean())


def summary(Q, Y, thr=None, lead=4):
    ok = ~np.isnan(Y)
    qi = {q: i for i, q in enumerate(QS)}
    med = Q[..., qi[0.5]]
    W = wis(Q, Y)
    out = dict(WIS=float(np.nanmean(W)), MAE=float(np.nanmean(np.abs(med - Y))),
               RMSE=float(np.sqrt(np.nanmean((med - Y) ** 2))),
               cov80=coverage(Q[..., qi[0.1]], Q[..., qi[0.9]], Y),
               cov95=coverage(Q[..., qi[0.025]], Q[..., qi[0.975]], Y),
               logMAE=float(np.nanmean(np.abs(np.log1p(med) - np.log1p(Y)))))
    out["WIS_h"] = [float(np.nanmean(W[:, :, h])) for h in range(W.shape[2])]
    out["WIS_region"] = [float(np.nanmean(W[:, i])) for i in range(W.shape[1])]
    if thr is not None:
        a = med[:, :, lead - 1] > thr[None]; tru = Y[:, :, lead - 1] > thr[None]
        v = ~np.isnan(Y[:, :, lead - 1])
        tp = (a & tru & v).sum(); fn = (~a & tru & v).sum(); fp = (a & ~tru & v).sum()
        out.update(alarm_TP=int(tp), alarm_FN=int(fn), alarm_FP=int(fp),
                   alarm_sens=float(tp / max(tp + fn, 1)), alarm_prec=float(tp / max(tp + fp, 1)),
                   alarm_F1=float(2 * tp / max(2 * tp + fp + fn, 1)))
    return out
