"""Runs every experiment reported in the VECTRA manuscript and writes results_*.json.
Usage:  python run_all.py [--quick]   (quick: 2 seeds, fewer scenario paths; for Colab smoke tests)"""
import os, sys, json, time, copy, platform, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
import vectra_data as vd, vectra_models as VM, vectra_train as TR, vectra_metrics as MT
import vectra_decision as DC, vectra_conformal as CF, vectra_mech as M, vectra_offload as OF

QUICK = "--quick" in sys.argv
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
NS = 100 if QUICK else 200
OUT = "."
H = VM.H
CFG = dict(vectra=dict(lam_g=0.01, lr=3e-3), grunb=dict(lr=1e-2), graphgru=dict(lr=1e-2), mech=dict(lr=3e-3))


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(f"results_{name}.json", "w"), default=conv, indent=1)


def builders(N, A):
    return {
        "VECTRA": lambda: VM.VECTRA(N, A, n_rhat=1),
        "Mechanistic-only": lambda: VM.VECTRA(N, A, n_rhat=1, closure=False),
        "GRU-NB": lambda: VM.GRUNB(N, A),
        "Graph-GRU-NB": lambda: VM.GRUNB(N, A, graph=True),
        # ablations
        "w/o thermal suitability & T-dependent GI": lambda: VM.VECTRA(N, A, n_rhat=1, thermal=False, gi_temp=False),
        "w/o T-dependent GI": lambda: VM.VECTRA(N, A, n_rhat=1, gi_temp=False),
        "w/o coupling graph": lambda: VM.VECTRA(N, A, n_rhat=1, graph=False),
        "w/o observation closure": lambda: VM.VECTRA(N, A, n_rhat=1, obs_closure=False),
    }


def kw_for(name):
    if name.startswith("GRU") or name.startswith("Graph"):
        return CFG["grunb"]
    if name == "Mechanistic-only":
        return CFG["mech"]
    return CFG["vectra"]


CACHE = "."


def fit_pair(mk, W, D, seed, kw, tag=None):
    """Train-only model (early stopped on the validation year) and the refit model (train+validation,
    same number of epochs). Trained weights are cached so that later stages can be re-run."""
    key = None if tag is None else f"cache_{tag.replace('/', '_').replace(' ', '_').replace('&', 'and')}_s{seed}"
    if key and os.path.exists(key + ".json"):
        m0, m1 = mk(), mk()
        m0.load_state_dict(torch.load(key + "_m0.pt")); m1.load_state_dict(torch.load(key + "_m1.pt"))
        return m0, m1, json.load(open(key + ".json"))
    m0 = mk(); info = TR.train_torch(m0, W, D, seed=seed, **kw)          # train-only model (val scores)
    m1 = mk(); info1 = TR.train_torch(m1, W, D, seed=seed, fixed_epochs=info["best_epoch"], **kw)
    inf = dict(best_epoch=info["best_epoch"], train_time=info["train_time"] + info1["train_time"], val_loss=info["val_loss"])
    if key:
        torch.save(m0.state_dict(), key + "_m0.pt"); torch.save(m1.state_dict(), key + "_m1.pt")
        json.dump(inf, open(key + ".json", "w"))
    return m0, m1, inf


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ============================================================================ main dataset
def main_experiments():
    D = vd.load_meso(); A = vd.coupling_graph(D); W = VM.Window(D, A)
    N = D["y"].shape[1]
    tr, va, te = TR.origin_sets(D); end, ve = D["split"]["end"], D["split"]["val_end"]
    Y = MT.targets(D["y"], te, end, N, H); Yv = MT.targets(D["y"], va, ve, N, H)
    thr = np.percentile(D["y"][: D["split"]["train_end"] + 1], 90, axis=0)
    DC.CHANNEL = thr.copy(); DC.H0 = 3
    dec_or = [t for t in te if t + H <= end]
    Ytrue = MT.targets(D["y"], dec_or, end, N, H)
    Tobs = np.array([np.stack([D["temp"][t + h - 3: t + h + 1].mean(0) for h in range(1, H + 1)], 1) for t in dec_or])
    Dtrue = DC.averted_matrix(Ytrue, Tobs)
    dsel = [te.index(t) for t in dec_or]
    info = dict(dataset=D["name"], N=N, weeks=len(D["y"]), units=D["units"], groups=list(D["groups"]),
                train_end=int(D["split"]["train_end"]), val_end=int(ve), end=int(end),
                climate_missing=D["missing"], n_test_origins=len(te), n_decision_origins=len(dec_or),
                total_cases={int(yr): float(D["y"][D["year"] == yr].sum()) for yr in np.unique(D["year"])},
                threshold_p90=thr, mean_temp=D["temp"][: D["split"]["train_end"] + 1].mean(0),
                omega_graph_degree=int((A > 0).sum(1).mean()))
    save("dataset_info", info)

    def dec_eval(Pmean, tbar, budget=DC.BUDGET, eff=None, mu0=M.MU0, Dt=None):
        Dh = DC.averted_matrix(Pmean, tbar, eff, mu0)
        Dt = Dtrue if Dt is None else Dt
        R = [DC.evaluate(DC.solve_mean(Dh[o], budget), Dt[o], budget) for o in range(len(dec_or))]
        return dict(regret=float(np.mean([r["regret"] for r in R])),
                    regret_p90=float(np.quantile([r["regret"] for r in R], 0.9)),
                    averted=float(sum(r["averted"] for r in R)), oracle=float(sum(r["oracle"] for r in R)),
                    per_origin=[r["regret"] for r in R])

    res = {}; store = {}
    # ---- tabular / statistical baselines
    t0 = time.time(); r = TR.persistence(W, D, te, end); res["Persistence"] = [MT.summary(r["Q"], Y, thr)]
    store["Persistence"] = [r]; res["Persistence"][0]["fit_time"] = time.time() - t0
    t0 = time.time(); r = TR.glm_nb(W, D, te, end); res["NB-GLM"] = [MT.summary(r["Q"], Y, thr)]
    store["NB-GLM"] = [r]; res["NB-GLM"][0]["fit_time"] = time.time() - t0
    res["LightGBM-Q"], store["LightGBM-Q"] = [], []
    res["LSTM-GBM"], store["LSTM-GBM"] = [], []
    for s in SEEDS:
        t0 = time.time(); r = TR.lightgbm_q(W, D, te, end, seed=s); x = MT.summary(r["Q"], Y, thr); x["fit_time"] = time.time() - t0
        res["LightGBM-Q"].append(x); store["LightGBM-Q"].append(r)
        t0 = time.time(); r = TR.lstm_gbm(W, D, te, end, seed=s); x = MT.summary(r["Q"], Y, thr); x["fit_time"] = time.time() - t0
        res["LSTM-GBM"].append(x); store["LSTM-GBM"].append(r)
        print("tabular seed", s, flush=True)
    # also val-period predictions for conformal warm-up of LightGBM (train-only fit)
    lgb_val = TR.lightgbm_q(W, D, va, ve, seed=0, refit=False)

    # ---- neural models and ablations
    B = builders(N, A); val_store = {}; models = {}
    for name, mk in B.items():
        res[name], store[name], val_store[name], models[name] = [], [], [], []
        for s in SEEDS:
            m0, m1, inf = fit_pair(mk, W, D, s, kw_for(name), tag=f"{D['name']}_{name}")
            rv = TR.predict_torch(m0, W, va, ve, n=NS, seed=s)
            t0 = time.time(); r = TR.predict_torch(m1, W, te, end, n=NS, seed=s); inf["predict_time"] = time.time() - t0
            x = MT.summary(r["Q"], Y, thr); x.update(inf); x["params"] = n_params(m1)
            if hasattr(m1, "rho_raw"):
                x.update(gamma=float(m1.gamma), omega=float(m1.omega), rho=float(torch.sigmoid(m1.rho_raw)))
            res[name].append(x); store[name].append(r); val_store[name].append(rv); models[name].append(m1)
            print(name, s, round(x["WIS"], 2), flush=True)
    save("forecast_main", res)

    # ---- decision experiments
    dres = {}
    yt = np.stack([np.repeat(D["y"][t][:, None], H, 1) for t in dec_or])
    tb_dec = store["VECTRA"][0]["tbar"][dsel]
    dres["Reactive (current-week excess)"] = [dec_eval(yt, tb_dec)]
    dres["Persistence"] = [dec_eval(store["Persistence"][0]["Q"][dsel, ..., 4], tb_dec)]
    dres["NB-GLM"] = [dec_eval(store["NB-GLM"][0]["mean"][dsel], tb_dec)]
    for name in ["LightGBM-Q", "LSTM-GBM", "GRU-NB", "Graph-GRU-NB", "Mechanistic-only", "VECTRA"]:
        dres[name] = [dec_eval(r["mean"][dsel], r["tbar"][dsel] if "tbar" in r else tb_dec) for r in store[name]]
    # CVaR variants on VECTRA scenarios
    cv = {}
    for alpha in [0.5, 0.8, 0.9]:
        cv[alpha] = []
        for r in store["VECTRA"]:
            S = r["S"][:, dsel]; Ds = DC.averted_matrix(S, r["tbar"][dsel][None])
            R = [DC.evaluate(DC.solve_cvar(S[:, o], Ds[:, o], alpha), Dtrue[o]) for o in range(len(dec_or))]
            cv[alpha].append(dict(regret=float(np.mean([q["regret"] for q in R])),
                                  regret_p90=float(np.quantile([q["regret"] for q in R], 0.9))))
    dres["_cvar"] = {str(k): v for k, v in cv.items()}
    # regret-trained (SPO+) fine-tuning vs an equal-length likelihood-only fine-tuning
    spo, ctl = [], []
    for s, m in zip(SEEDS, models["VECTRA"]):
        for lst, dfl in [(spo, DC.SPOPlus(W, lam=5.0)), (ctl, None)]:
            m2 = copy.deepcopy(m)
            TR.train_torch(m2, W, D, seed=s, dfl=dfl, fixed_epochs=15, **dict(CFG["vectra"], lr=1e-3))
            r = TR.predict_torch(m2, W, dec_or, end, n=NS, seed=s)
            q = dec_eval(r["mean"], r["tbar"]); rt = TR.predict_torch(m2, W, te, end, n=NS, seed=s)
            q["WIS"] = MT.summary(rt["Q"], Y)["WIS"]; lst.append(q)
    dres["_spo_finetune"] = spo; dres["_likelihood_finetune"] = ctl
    # sensitivity of the allocation advantage to the intervention assumptions
    sens = {}
    base_eff = dict(M.EFF)
    for key, settings in [("budget", [3.0, 6.0, 12.0]), ("efficacy_scale", [0.5, 1.0, 1.5]),
                          ("mu0", [0.07, 0.10, 0.14]), ("channel_pct", [75, 90, 95])]:
        sens[key] = {}
        for v in settings:
            eff = {a: min(0.9, base_eff[a] * (v if key == "efficacy_scale" else 1.0)) for a in base_eff}
            mu0 = v if key == "mu0" else M.MU0; bud = v if key == "budget" else DC.BUDGET
            old = DC.CHANNEL
            if key == "channel_pct":
                DC.CHANNEL = np.percentile(D["y"][: D["split"]["train_end"] + 1], v, axis=0)
            Dt = DC.averted_matrix(Ytrue, Tobs, eff, mu0)
            row = {"Reactive": dec_eval(yt, tb_dec, bud, eff, mu0, Dt)["regret"]}
            for name in ["GRU-NB", "VECTRA", "LightGBM-Q"]:
                row[name] = float(np.mean([dec_eval(r["mean"][dsel], tb_dec, bud, eff, mu0, Dt)["regret"] for r in store[name]]))
            sens[key][str(v)] = row; DC.CHANNEL = old
    dres["_sensitivity"] = sens
    # chosen actions versus temperature (VECTRA seed 0)
    r = store["VECTRA"][0]; Dh = DC.averted_matrix(r["mean"][dsel], r["tbar"][dsel])
    acts = []
    for o in range(len(dec_or)):
        x = DC.solve_mean(Dh[o])
        for i in range(N):
            if x[i].sum() > 1e-6:
                acts.append(dict(temp=float(r["tbar"][dsel][o, i, 2:].mean()), action=int(np.argmax(x[i])), coverage=float(x[i].sum())))
    dres["_actions"] = acts
    # plan quality of each deployment mode (all seeds): global LP, stale plan reuse, fog state LPs with
    # equal shares or with the state shares of the last coordinated plan, and the reactive rule
    states = sorted(set(D["groups"]))
    gidx = {s: np.where(D["groups"] == s)[0] for s in states}
    modes = {k: [] for k in ["global", "stale", "state_equal", "state_share"]}
    for r in store["VECTRA"]:
        Dh = DC.averted_matrix(r["mean"][dsel], r["tbar"][dsel])
        plans = [DC.solve_mean(Dh[o]) for o in range(len(dec_or))]
        g_, st_, se_, ss_ = [], [], [], []
        for o in range(len(dec_or)):
            g_.append(DC.evaluate(plans[o], Dtrue[o])["regret"])
            X = np.zeros_like(Dh[o])
            for s_ in states:
                X[gidx[s_]] = DC.solve_mean(Dh[o][gidx[s_]], DC.BUDGET * len(gidx[s_]) / N)
            se_.append(DC.evaluate(X, Dtrue[o])["regret"])
            if o > 0:
                st_.append(DC.evaluate(plans[o - 1], Dtrue[o])["regret"])
                prev = plans[o - 1]; spent = {s_: float((prev[gidx[s_]] * DC.COSTS).sum()) for s_ in states}
                X = np.zeros_like(Dh[o])
                for s_ in states:
                    if spent[s_] > 1e-9:
                        X[gidx[s_]] = DC.solve_mean(Dh[o][gidx[s_]], spent[s_])
                ss_.append(DC.evaluate(X, Dtrue[o])["regret"])
        modes["global"].append(np.mean(g_)); modes["stale"].append(np.mean(st_))
        modes["state_equal"].append(np.mean(se_)); modes["state_share"].append(np.mean(ss_))
    dres["_deploy_modes"] = {k: float(np.mean(v)) for k, v in modes.items()}
    dres["_deploy_modes"]["reactive"] = dres["Reactive (current-week excess)"][0]["regret"]
    dres["_deploy_modes_sd"] = {k: float(np.std(v)) for k, v in modes.items()}
    save("decision", dres)

    # ---- calibration (val year warms up the online calibrators; train-only models score 2022)
    cres = {}
    for name in ["VECTRA", "GRU-NB", "LightGBM-Q"]:
        cres[name] = {}
        for alpha in [0.2, 0.05]:
            for mode in ["raw", "split", "aci_h", "aci_region_h"]:
                rows = []
                pairs = (zip(val_store[name], store[name]) if name != "LightGBM-Q" else [(lgb_val, store[name][0])])
                for rv, rt in pairs:
                    Qc = np.concatenate([rv["Q"], rt["Q"]]); Yc = np.concatenate([Yv, Y])
                    lo, hi = CF.calibrate(Qc, Yc, va + te, len(va), alpha, mode)
                    mtr = CF.interval_metrics(lo, hi, Y, alpha)
                    cov_reg = [((Y[:, i] >= lo[:, i]) & (Y[:, i] <= hi[:, i]))[~np.isnan(Y[:, i])].mean() for i in range(N)]
                    mtr["region_cov_sd"] = float(np.std(cov_reg)); mtr["region_cov_min"] = float(np.min(cov_reg))
                    # coverage in growth vs decline weeks (sign of the realised 4-week change)
                    g = (Y[:, :, 3] > np.array([D["y"][t] for t in te]))
                    okg = ~np.isnan(Y[:, :, 3])
                    inside = (Y >= lo) & (Y <= hi)
                    mtr["cov_growth_h4"] = float(inside[:, :, 3][g & okg].mean()); mtr["cov_decline_h4"] = float(inside[:, :, 3][~g & okg].mean())
                    mtr["cov_h"] = [float(inside[:, :, h][~np.isnan(Y[:, :, h])].mean()) for h in range(H)]
                    rows.append(mtr)
                cres[name][f"{mode}@{alpha}"] = rows
    eta_s = {}
    for eta in [0.005, 0.01, 0.02, 0.05, 0.1]:
        rows = []
        for rv, rt in zip(val_store["VECTRA"], store["VECTRA"]):
            lo, hi = CF.calibrate(np.concatenate([rv["Q"], rt["Q"]]), np.concatenate([Yv, Y]), va + te, len(va), 0.2, "aci_region_h", eta)
            rows.append(CF.interval_metrics(lo, hi, Y, 0.2))
        eta_s[str(eta)] = rows
    cres["_eta"] = eta_s
    save("calibration", cres)

    # ---- robustness: right truncation of the two most recent weeks, and temperature noise
    rob = {}
    for name in ["VECTRA", "GRU-NB", "Graph-GRU-NB"]:
        rob[name] = {}
        for c in [1.0, 0.8, 0.6]:
            w = []
            for s, m in zip(SEEDS[:3], models[name][:3]):
                Qs = []
                for t in te:
                    Dt_ = dict(D); yy = D["y"].copy(); yy[t] = yy[t] * c; yy[t - 1] = yy[t - 1] * (1 + c) / 2
                    Dt_["y"] = yy; Wt = VM.Window(Dt_, A, feat_stats=W.feat_stats)
                    Qs.append(TR.predict_torch(m, Wt, [t], end, n=NS, seed=s)["Q"])
                w.append(MT.summary(np.concatenate(Qs), Y)["WIS"])
            rob[name][f"trunc_{c}"] = w
        for sd in [1.0, 2.0]:
            w = []
            rng = np.random.default_rng(7)
            Dn = dict(D); tn = D["temp"].copy(); tn[ve - 12:] += rng.normal(0, sd, tn[ve - 12:].shape); Dn["temp"] = tn
            Wn = VM.Window(Dn, A, feat_stats=W.feat_stats)
            for s, m in zip(SEEDS[:3], models[name][:3]):
                w.append(MT.summary(TR.predict_torch(m, Wn, te, end, n=NS, seed=s)["Q"], Y)["WIS"])
            rob[name][f"tnoise_{sd}"] = w
    save("robustness", rob)

    # ---- counterfactual illustration (VECTRA seed 0): intervention-conditioned forecast fans
    m = models["VECTRA"][0]
    cf = {}
    big = np.argsort(-D["y"][te[0]: te[0] + 12].sum(0))[:4]
    t_or = te[5]
    x, yh, tb, _, _ = W.batch([t_or], end)
    for a in ["none"] + list(M.ACTIONS):
        if a == "none":
            shift = None
        else:
            ph = M.phi(a, tb[..., 3:])                                        # torch tensor
            ph[..., : DC.H0 - 1] = 0.0
            shift = ph
        torch.manual_seed(0)
        S = m.sample(x, yh, tb, n=NS, shift=shift).numpy()[:, 0]
        cf[a] = dict(q=np.quantile(S[:, big], [0.1, 0.5, 0.9], axis=0))
    cf["regions"] = [D["units"][i] for i in big]; cf["origin"] = int(t_or)
    cf["history"] = D["y"][t_or - 11: t_or + 1, big].T
    cf["truth"] = D["y"][t_or + 1: t_or + 1 + H, big].T
    save("counterfactual", cf)

    # ---- measured workloads for the deployment simulation
    x, yh, tb, _, _ = W.batch(dec_or[:10], end)
    t0 = time.time()
    for _ in range(3):
        m.sample(x, yh, tb, n=NS)
    t_inf = (time.time() - t0) / (3 * 10 * N)
    r0 = store["VECTRA"][0]; Dh0 = DC.averted_matrix(r0["mean"][dsel], r0["tbar"][dsel])
    t0 = time.time()
    for o in range(len(dec_or)):
        DC.solve_mean(Dh0[o])
    t_plan = (time.time() - t0) / len(dec_or)
    idx = np.where(D["groups"] == "SP")[0]
    t0 = time.time()
    for o in range(len(dec_or)):
        DC.solve_mean(Dh0[o][idx], 2.0)
    t_state = (time.time() - t0) / len(dec_or)
    Qtmp = np.concatenate([val_store["VECTRA"][0]["Q"], store["VECTRA"][0]["Q"]])
    t0 = time.time(); CF.calibrate(Qtmp, np.concatenate([Yv, Y]), va + te, len(va), 0.2, "aci_region_h")
    t_cal = (time.time() - t0) / ((len(va) + len(te)) * N)
    try:
        mhz = float([l for l in open("/proc/cpuinfo") if "cpu MHz" in l][0].split(":")[1])
    except Exception:
        mhz = 2500.0
    f_ref = mhz * 1e6
    wl = dict(infer=t_inf * f_ref, calib=t_cal * f_ref, plan_global=t_plan * f_ref, plan_state=t_state * f_ref,
              in_bits=VM.L_HIST * 9 * 32 + 12 * 32, out_bits=NS * H * 32 + H * 9 * 32, plan_bits=N * 3 * 32,
              f_ref=f_ref, t_infer_s=t_inf, t_plan_s=t_plan, t_state_s=t_state, t_cal_s=t_cal)
    sim = OF.Simulator(wl, N, D["groups"])
    dep = dict(workload=wl, modes_regret=dres["_deploy_modes"], policies={})
    dm = dres["_deploy_modes"]
    mode_reg = {"global": dm["global"], "partial": dm["global"], "state_share": dm["state_share"],
                "state_equal": dm["state_equal"], "stale": dm["stale"], "reactive": dm["reactive"]}
    for pol in ["cloud", "fog", "edge", "greedy", "vectra"]:
        dep["policies"][pol] = {}
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            out = sim.simulate(pol, p, reps=50 if QUICK else 200)
            en = np.array([o["energy"] for o in out], float); la = np.array([o["latency"] for o in out], float)
            mm = [o["mode"] for o in out]
            dep["policies"][pol][str(p)] = dict(
                new_plan=float(np.mean([x not in ("stale",) for x in mm])),
                coordinated=float(np.mean([x in ("global", "partial") for x in mm])),
                mode_share={k: float(np.mean([x == k for x in mm])) for k in set(mm)},
                energy_J=float(np.nanmean(en)), latency_p95=float(np.nanpercentile(la, 95)) if np.isfinite(la).any() else None,
                exp_regret=float(np.mean([mode_reg[x] for x in mm])))
    save("deployment", dep)
    return store, models


# ============================================================================ thermal transfer
def transfer_experiment():
    D = vd.load_meso()
    Tm = D["temp"][: D["split"]["train_end"] + 1].mean(0)
    hot = np.where(Tm >= np.median(Tm))[0]; cool = np.where(Tm < np.median(Tm))[0]

    def sub(idx):
        E = dict(D)
        for k in ["y", "temp", "hum", "rain", "delay", "temp_clim", "hum_clim", "rain_clim"]:
            E[k] = D[k][:, idx]
        E["units"] = [D["units"][i] for i in idx]; E["groups"] = D["groups"][idx]; return E
    Dc, Dh = sub(cool), sub(hot)
    Ac, Ah = vd.coupling_graph(Dc), vd.coupling_graph(Dh)
    Wc = VM.Window(Dc, Ac); Wh = VM.Window(Dh, Ah, feat_stats=Wc.feat_stats)
    _, _, te = TR.origin_sets(D); end = D["split"]["end"]
    Yh = MT.targets(Dh["y"], te, end, len(hot), H)

    def move(m, N2, A2):
        m2 = copy.deepcopy(m); sd = m2.state_dict()
        for k, v in list(sd.items()):
            if k != "A" and v.dim() >= 1 and v.shape[0] == len(cool):
                sd[k] = v.mean(0, keepdim=True).expand(N2, *v.shape[1:]).clone()
        if hasattr(m2, "enc"):
            m2.enc.emb = torch.nn.Embedding(N2, sd["enc.emb.weight"].shape[1])
        for nm in ["b", "eta", "r_raw", "p"]:
            if hasattr(m2, nm):
                setattr(m2, nm, torch.nn.Parameter(sd[nm]))
        m2.A = torch.tensor(A2, dtype=torch.float32); sd["A"] = m2.A
        m2.load_state_dict(sd); m2.N = N2
        return m2
    B = builders(len(cool), Ac)
    out = dict(cool_mean_temp=float(Tm[cool].mean()), hot_mean_temp=float(Tm[hot].mean()), n_cool=len(cool), n_hot=len(hot))
    for name in ["VECTRA", "w/o thermal suitability & T-dependent GI", "Mechanistic-only", "GRU-NB", "Graph-GRU-NB"]:
        w = []
        for s in SEEDS:
            _, m1, _ = fit_pair(B[name], Wc, Dc, s, kw_for(name), tag=f"transfer_{name}")
            w.append(MT.summary(TR.predict_torch(move(m1, len(hot), Ah), Wh, te, end, n=NS, seed=s)["Q"], Yh)["WIS"])
        out[name] = w
        print("transfer", name, np.round(w, 2), flush=True)
    save("transfer", out)


# ============================================================================ external dataset
def belem_experiment():
    D = vd.load_belem(); A = np.zeros((1, 1)); W = VM.Window(D, A)
    _, va, te = TR.origin_sets(D); end, ve = D["split"]["end"], D["split"]["val_end"]
    Y = MT.targets(D["y"], te, end, 1, H); Yv = MT.targets(D["y"], va, ve, 1, H)
    thr = np.percentile(D["y"][: D["split"]["train_end"] + 1], 90, axis=0)
    res = {"Persistence": [MT.summary(TR.persistence(W, D, te, end)["Q"], Y, thr)],
           "NB-GLM": [MT.summary(TR.glm_nb(W, D, te, end)["Q"], Y, thr)]}
    res["LightGBM-Q"] = [MT.summary(TR.lightgbm_q(W, D, te, end, seed=s)["Q"], Y, thr) for s in SEEDS]
    B = builders(1, A); cal = {}
    for name in ["VECTRA", "Mechanistic-only", "GRU-NB", "w/o thermal suitability & T-dependent GI"]:
        res[name] = []; cal[name] = []
        for s in SEEDS:
            m0, m1, inf = fit_pair(B[name], W, D, s, kw_for(name), tag=f"{D['name']}_{name}")
            rt = TR.predict_torch(m1, W, te, end, n=NS, seed=s); rv = TR.predict_torch(m0, W, va, ve, n=NS, seed=s)
            x = MT.summary(rt["Q"], Y, thr); x.update(inf); res[name].append(x)
            lo, hi = CF.calibrate(np.concatenate([rv["Q"], rt["Q"]]), np.concatenate([Yv, Y]), va + te, len(va), 0.2, "aci_region_h")
            raw = CF.calibrate(np.concatenate([rv["Q"], rt["Q"]]), np.concatenate([Yv, Y]), va + te, len(va), 0.2, "raw")
            cal[name].append(dict(aci=CF.interval_metrics(lo, hi, Y, 0.2), raw=CF.interval_metrics(*raw, Y, 0.2)))
        print("belem", name, np.round([r["WIS"] for r in res[name]], 2), flush=True)
    res["_calibration"] = cal
    res["_info"] = dict(weeks=len(D["y"]), test_weeks=len(te), train_end=int(D["split"]["train_end"]),
                        years=[int(D["year"][0]), int(D["year"][-1])], mean_temp=float(D["temp"].mean()),
                        test_cases=float(D["y"][D["split"]["val_end"] + 1:].sum()))
    save("belem", res)


if __name__ == "__main__":
    t0 = time.time()
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
               platform=platform.platform(), processor=platform.processor(), cpus=os.cpu_count(), quick=QUICK, seeds=SEEDS, n_paths=NS)
    stages = [a for a in sys.argv[1:] if not a.startswith("--")] or ["main", "transfer", "belem"]
    if "main" in stages: main_experiments()
    if "transfer" in stages: transfer_experiment()
    if "belem" in stages: belem_experiment()
    env["runtime_s"] = time.time() - t0
    save("environment_" + "_".join(stages), env)
    print("done in", round(time.time() - t0), "s")
