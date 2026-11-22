"""Data access and preprocessing for VECTRA.

Primary data: weekly probable dengue cases (SINAN) and station climate (INMET) for 43
Brazilian mesoregions, 2019-2023, harmonised and hosted on the Hugging Face Hub
(dataset thiagorfreitas/dengue-tcc2-data, folder data/model_ready/mesorregiao).
External data: InfoDengue weekly series for Belem (PA), hosted at JeanDias/InfoDengue_Belem.
"""
import os, urllib.request
import numpy as np, pandas as pd

HF = "https://huggingface.co/datasets"
MESO_FILES = {s: f"{HF}/thiagorfreitas/dengue-tcc2-data/resolve/main/data/model_ready/mesorregiao/{s}.parquet"
              for s in ("train", "val", "test")}
BELEM_FILE = f"{HF}/JeanDias/InfoDengue_Belem/resolve/main/dengue_34-1.csv"
STATE_OF = {"31": "MG", "32": "ES", "33": "RJ", "35": "SP", "52": "GO", "53": "DF"}


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _fill_climate(X, woy, train_mask, groups):
    """X: [T, N] climate matrix with NaN. Fill short gaps by interpolation (<=3 weeks),
    then by same-state mean of the week, then by training-period week-of-year climatology."""
    df = pd.DataFrame(X)
    df = df.interpolate(limit=3, limit_area="inside", axis=0)
    X = df.values.copy()
    for g in np.unique(groups):
        cols = np.where(groups == g)[0]
        m = np.nanmean(X[:, cols], axis=1) if len(cols) > 1 else np.full(X.shape[0], np.nan)
        for c in cols:
            miss = np.isnan(X[:, c])
            X[miss, c] = m[miss]
    clim = np.full((54, X.shape[1]), np.nan)
    for w in range(1, 54):
        sel = (woy == w) & train_mask
        if sel.any():
            clim[w] = np.nanmean(X[sel], axis=0)
    clim = pd.DataFrame(clim).interpolate(axis=0, limit_direction="both").values
    glob = np.nanmean(clim, axis=1, keepdims=True)
    clim = np.where(np.isnan(clim), glob, clim)
    for t in range(X.shape[0]):
        miss = np.isnan(X[t])
        X[t, miss] = clim[woy[t], miss]
    return X, clim


def load_meso(cache="."):
    parts = [pd.read_parquet(_get(u, f"{cache}/data_meso_{s}.parquet")) for s, u in MESO_FILES.items()]
    d = pd.concat(parts, ignore_index=True)
    d["unit"] = d["ibge_municipio"].astype(str)
    d = d.sort_values(["ano", "semana_epidemiologica", "unit"])
    # the final epidemiological week of 2023 is a partial stub (incomplete reporting) and is excluded
    d = d[~((d.ano == 2023) & (d.semana_epidemiologica == 53))]
    units = sorted(d.unit.unique())
    base_idx = d.pivot_table(index=["ano", "semana_epidemiologica"], columns="unit",
                             values="qt_confirmados_provaveis", aggfunc="first").index
    piv = lambda c: d.pivot_table(index=["ano", "semana_epidemiologica"], columns="unit", values=c,
                                  aggfunc="first", dropna=False).reindex(index=base_idx, columns=units)
    y = piv("qt_confirmados_provaveis").values.astype(np.float64)
    idx = piv("qt_confirmados_provaveis").index
    year = np.array([a for a, _ in idx]); woy = np.array([w for _, w in idx])
    train_mask = year <= 2021
    groups = np.array([STATE_OF[u[:2]] for u in units])
    clim = {}
    for name, col in [("temp", "temp_mean_c"), ("hum", "humidity_mean_pct"), ("rain", "rain_sum_mm")]:
        raw = piv(col).values.astype(np.float64)
        miss = np.isnan(raw).mean()
        X, cl = _fill_climate(raw, woy, train_mask, groups)
        clim[name] = X; clim[name + "_clim"] = cl; clim[name + "_missing"] = float(miss)
    delay = np.clip(np.nan_to_num(piv("atraso_notificacao_medio_dias").values.astype(np.float64)), 0, 60)
    return dict(name="SE-Brazil", y=y, year=year, woy=woy, units=units, groups=groups,
                temp=clim["temp"], hum=clim["hum"], rain=clim["rain"], delay=delay,
                temp_clim=clim["temp_clim"], hum_clim=clim["hum_clim"], rain_clim=clim["rain_clim"],
                missing={k: clim[k + "_missing"] for k in ("temp", "hum", "rain")},
                split=dict(train_end=int(np.where(year <= 2021)[0].max()),
                           val_end=int(np.where(year <= 2022)[0].max()), end=len(year) - 1))


def load_belem(cache="."):
    b = pd.read_csv(_get(BELEM_FILE, f"{cache}/data_belem.csv")).sort_values("SE")
    b = b[(b.SE >= 201001) & (b.SE <= 202452)].reset_index(drop=True)
    year = (b.SE // 100).values; woy = (b.SE % 100).values
    y = b["casos"].values.astype(np.float64)[:, None]
    train_mask = year <= 2020
    groups = np.array(["PA"])
    temp, tc = _fill_climate(b["tempmed"].values[:, None].astype(float), woy, train_mask, groups)
    hum, hc = _fill_climate(b["umidmed"].values[:, None].astype(float), woy, train_mask, groups)
    rain = np.zeros_like(temp); rc = np.zeros((54, 1))
    return dict(name="Belem", y=y, year=year, woy=woy, units=["Belem"], groups=groups,
                temp=temp, hum=hum, rain=rain, delay=np.zeros_like(temp),
                temp_clim=tc, hum_clim=hc, rain_clim=rc,
                missing={"temp": float(b["tempmed"].isna().mean()), "hum": float(b["umidmed"].isna().mean()), "rain": 1.0},
                split=dict(train_end=int(np.where(year <= 2020)[0].max()),
                           val_end=int(np.where(year <= 2021)[0].max()), end=len(year) - 1))


def coupling_graph(D, k=5):
    """Inter-regional coupling estimated from training-period co-movement of weekly log-growth.
    Returns a row-normalised kNN matrix with zero diagonal (no mobility data are assumed)."""
    y = D["y"][: D["split"]["train_end"] + 1]
    N = y.shape[1]
    if N == 1:
        return np.zeros((1, 1))
    g = np.diff(np.log1p(y), axis=0)
    C = np.corrcoef(g.T); np.fill_diagonal(C, -np.inf)
    A = np.zeros((N, N))
    for i in range(N):
        nb = np.argsort(C[i])[::-1][:k]
        w = np.clip(C[i, nb], 1e-3, None)
        A[i, nb] = w / w.sum()
    return A
