"""Statistical analysis of the saved results (results_*.json) -> results_analysis.json."""
import json, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

R = lambda n: json.load(open(f"results_{n}.json"))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci(v, B=5000, seed=0, block=4):
    """Moving-block bootstrap of the mean over weekly origins."""
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v); out = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, int(np.ceil(n / block)))])[:n]
        out.append(v[idx].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    F = R("forecast_main"); out = {}
    models = [m for m in F if not m.startswith("w/o")]
    summ = {}
    for m, runs in F.items():
        summ[m] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                   for k in ["WIS", "MAE", "RMSE", "logMAE", "cov80", "cov95", "alarm_sens", "alarm_prec", "alarm_F1"]}
        summ[m]["WIS_h"] = np.mean([r["WIS_h"] for r in runs], 0).tolist()
        summ[m]["WIS_h_sd"] = np.std([r["WIS_h"] for r in runs], 0).tolist()
        for k in ["params", "train_time", "predict_time", "fit_time", "gamma", "omega", "rho", "best_epoch"]:
            if k in runs[0]:
                summ[m][k] = [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
    out["forecast_summary"] = summ
    reg = {m: np.mean([r["WIS_region"] for r in F[m]], 0) for m in F}
    tests = {}
    base = reg["VECTRA"]; ps = []; names = []
    for m in F:
        if m == "VECTRA": continue
        st = wilcoxon(base, reg[m]); ps.append(st.pvalue); names.append(m)
        tests[m] = dict(median_diff=float(np.median(base - reg[m])), p=float(st.pvalue),
                        rank_biserial=rank_biserial(base, reg[m]), vectra_better_regions=int((base < reg[m]).sum()))
    for m, a in zip(names, holm(ps)):
        tests[m]["p_holm"] = float(a)
    out["wilcoxon_vs_vectra_region_WIS"] = tests
    core = ["Persistence", "NB-GLM", "LightGBM-Q", "LSTM-GBM", "GRU-NB", "Graph-GRU-NB", "Mechanistic-only", "VECTRA"]
    M = np.stack([reg[m] for m in core], 1)
    fr = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.mean([rankdata(row) for row in M], 0)
    k, n = M.shape[1], M.shape[0]
    cd = 3.031 * np.sqrt(k * (k + 1) / (6 * n))           # Nemenyi critical difference, q_0.05 for k=8
    out["friedman"] = dict(stat=float(fr.statistic), p=float(fr.pvalue), mean_ranks=dict(zip(core, ranks.tolist())), nemenyi_cd=float(cd))

    Dd = R("decision"); dsum = {}
    for m, runs in Dd.items():
        if m.startswith("_"): continue
        dsum[m] = dict(regret=[float(np.mean([r["regret"] for r in runs])), float(np.std([r["regret"] for r in runs]))],
                       regret_p90=float(np.mean([r["regret_p90"] for r in runs])),
                       averted_frac=float(np.mean([r["averted"] / r["oracle"] for r in runs])))
    out["decision_summary"] = dsum
    per = {m: np.mean([r["per_origin"] for r in Dd[m]], 0) for m in Dd if not m.startswith("_")}
    comp = {}
    for a, b in [("VECTRA", "Reactive (current-week excess)"), ("GRU-NB", "Reactive (current-week excess)"),
                 ("VECTRA", "GRU-NB"), ("VECTRA", "LightGBM-Q"), ("VECTRA", "Mechanistic-only")]:
        d = per[a] - per[b]
        comp[f"{a} - {b}"] = dict(mean=float(d.mean()), ci95=boot_ci(d), wilcoxon_p=float(wilcoxon(per[a], per[b]).pvalue))
    out["decision_paired"] = comp
    sp = Dd["_spo_finetune"]; lk = Dd["_likelihood_finetune"]
    out["spo_vs_likelihood"] = dict(spo_regret=[float(np.mean([r["regret"] for r in sp])), float(np.std([r["regret"] for r in sp]))],
                                    lik_regret=[float(np.mean([r["regret"] for r in lk])), float(np.std([r["regret"] for r in lk]))],
                                    spo_WIS=float(np.mean([r["WIS"] for r in sp])), lik_WIS=float(np.mean([r["WIS"] for r in lk])),
                                    paired_p=float(wilcoxon([r["regret"] for r in sp], [r["regret"] for r in lk]).pvalue))
    out["cvar"] = {a: dict(regret=float(np.mean([r["regret"] for r in v])), regret_p90=float(np.mean([r["regret_p90"] for r in v])))
                   for a, v in Dd["_cvar"].items()}
    C = R("calibration"); cs = {}
    for m, modes in C.items():
        if m.startswith("_"): continue
        cs[m] = {k: {q: float(np.mean([r[q] for r in v])) for q in ["coverage", "log_width", "interval_score", "region_cov_sd", "region_cov_min", "cov_growth_h4", "cov_decline_h4"]}
                 for k, v in modes.items()}
        for k, v in modes.items():
            cs[m][k]["cov_h"] = np.mean([r["cov_h"] for r in v], 0).tolist()
    cs["_eta"] = {e: {q: float(np.mean([r[q] for r in v])) for q in ["coverage", "log_width", "interval_score"]} for e, v in C["_eta"].items()}
    out["calibration_summary"] = cs
    T = R("transfer"); out["transfer_summary"] = {k: ([float(np.mean(v)), float(np.std(v))] if isinstance(v, list) else v) for k, v in T.items()}
    Rb = R("robustness"); out["robustness_summary"] = {m: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for m, d in Rb.items()}
    B = R("belem"); bs = {}
    for m, runs in B.items():
        if m.startswith("_"): continue
        bs[m] = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in ["WIS", "MAE", "cov80", "cov95", "alarm_F1"]}
    bs["_calibration"] = {m: {k: float(np.mean([r[k]["coverage"] for r in v])) for k in ["raw", "aci"]} for m, v in B["_calibration"].items()}
    bs["_calibration_IS"] = {m: {k: float(np.mean([r[k]["interval_score"] for r in v])) for k in ["raw", "aci"]} for m, v in B["_calibration"].items()}
    bs["_info"] = B["_info"]
    out["belem_summary"] = bs
    json.dump(out, open("results_analysis.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main(); print("analysis written")
