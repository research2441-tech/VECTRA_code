"""Result figures (Fig. 3 - Fig. 9) from results_*.json and cached models. Grayscale only."""
import json, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vectra_mech as M

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "mathtext.fontset": "dejavuserif", "savefig.dpi": 600})
OUT = "."
R = lambda n: json.load(open(f"results_{n}.json"))
GR = ["black", "0.35", "0.55", "0.75"]
LS = ["-", "--", "-.", ":"]
MK = ["o", "s", "^", "D", "v", "x", "P", "*"]


def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def fig3():
    T = np.linspace(14, 34.3, 400)
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.85))
    ax[0].plot(T, np.exp(M.log_suitability(T)), color="black")
    for t in (M.T0, M.TOPT, M.TM):
        ax[0].axvline(t, color="0.6", lw=0.6, ls=":")
    ax[0].set_ylabel("$S(T)$"); ax[0].set_title("Thermal suitability", fontsize=7.5)
    ax[1].plot(T, M.eip_days(T), color="black"); ax[1].scatter([25, 30], [M.EIP25, M.EIP30], s=12, color="black", zorder=3)
    ax[1].set_ylabel("EIP (days)"); ax[1].set_title("Extrinsic incubation", fontsize=7.5)
    ax[2].plot(T, M.gi_mean_weeks(T), color="black"); ax[2].set_ylabel("Mean GI (weeks)"); ax[2].set_title("Generation interval", fontsize=7.5)
    for k, a in enumerate(M.ACTIONS):
        ax[3].plot(T, 1 - np.exp(-M.phi(a, T)), color=GR[k], ls=LS[k], label=a.replace("_", " "))
    ax[3].plot(T, (1 - np.exp(-M.phi("adulticiding", T))) / M.COST["adulticiding"], color="black", lw=0.6, ls=":",
               label="adulticiding per team-week")
    ax[3].set_ylabel("$1-e^{-\\phi_a(T)}$"); ax[3].set_title("Action efficacy", fontsize=7.5)
    ax[3].legend(fontsize=5.4, loc="upper right", handlelength=2.0, borderaxespad=0.1, labelspacing=0.25)
    ax[3].set_ylim(0, 1.32); ax[3].set_yticks([0, 0.25, 0.5, 0.75])
    for i, a in enumerate(ax):
        a.set_xlabel("Temperature (°C)"); panel_label(a, "(" + "abcd"[i] + ")")
    fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/fig3_mechanistic.png"); plt.close(fig)


def fig4():
    A = R("analysis")["forecast_summary"]; F = R("analysis")["friedman"]
    models = ["VECTRA", "GRU-NB", "Graph-GRU-NB", "NB-GLM", "LightGBM-Q", "LSTM-GBM", "Mechanistic-only", "Persistence"]
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    h = np.arange(1, 9)
    for k, m in enumerate(models):
        y = np.array(A[m]["WIS_h"]); e = np.array(A[m]["WIS_h_sd"])
        ax[0].errorbar(h + (k - 3.5) * 0.04, y, yerr=e, color=GR[k % 4], ls=LS[k // 2 % 4], marker=MK[k], ms=3,
                       lw=0.9 if m != "VECTRA" else 1.6, capsize=1.5, label=m)
    ax[0].set_xlabel("Forecast horizon $h$ (weeks)"); ax[0].set_ylabel("WIS (cases)")
    ax[0].legend(fontsize=5.8, ncol=2, loc="upper left", handlelength=2.6, columnspacing=1.0)
    ax[0].set_ylim(0, max(np.array(A["Persistence"]["WIS_h"])) * 1.35)
    mr = F["mean_ranks"]; order = sorted(mr, key=mr.get)
    y = np.arange(len(order))
    ax[1].barh(y, [mr[m] for m in order], color=["black" if m == "VECTRA" else "0.7" for m in order], height=0.6)
    ax[1].set_yticks(y); ax[1].set_yticklabels(order, fontsize=6.4); ax[1].invert_yaxis()
    best = mr[order[0]]
    ax[1].axvline(best + F["nemenyi_cd"], color="black", ls="--", lw=0.7)
    ax[1].text(best + F["nemenyi_cd"] + 0.08, len(order) - 0.6, "best + CD", fontsize=6, va="center")
    ax[1].set_xlabel("Mean rank over 43 regions (lower is better)")
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    fig.tight_layout(w_pad=2.0); fig.savefig(f"{OUT}/fig4_forecast.png"); plt.close(fig)


def fig5():
    import vectra_data as vd, vectra_models as VM, vectra_train as TR
    D = vd.load_meso(); A = vd.coupling_graph(D); W = VM.Window(D, A)
    _, _, te = TR.origin_sets(D); end = D["split"]["end"]
    m = VM.VECTRA(43, A, n_rhat=1); m.load_state_dict(torch.load("cache_SE-Brazil_VECTRA_s0_m1.pt"))
    r = TR.predict_torch(m, W, te, end, n=200, seed=0)
    g = VM.GRUNB(43, A); g.load_state_dict(torch.load("cache_SE-Brazil_GRU-NB_s0_m1.pt"))
    rg = TR.predict_torch(g, W, te, end, n=200, seed=0)
    tot = D["y"][te[0]:].sum(0); picks = np.argsort(-tot)[[0, 3, 9]]
    hh = 4
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.1))
    for k, i in enumerate(picks):
        wk = np.array([t + hh for t in te if t + hh <= end]); o = np.arange(len(wk))
        Q = r["Q"][o, i, hh - 1]; x = wk - te[0]
        ax[k].fill_between(x, Q[:, 0], Q[:, 8], color="0.88", lw=0, label="95% band")
        ax[k].fill_between(x, Q[:, 2], Q[:, 6], color="0.7", lw=0, label="80% band")
        ax[k].plot(x, Q[:, 4], color="black", lw=1.1, label="VECTRA median")
        ax[k].plot(x, rg["Q"][o, i, hh - 1, 4], color="0.3", lw=0.9, ls="--", label="GRU-NB median")
        ax[k].plot(x, D["y"][wk, i], "o", ms=1.8, color="black", label="observed")
        ax[k].set_title(f"Mesoregion {D['units'][i]}", fontsize=7.2)
        ax[k].set_xlabel("Epidemiological week, 2023")
        panel_label(ax[k], "(" + "abc"[k] + ")")
    ax[0].set_ylabel("Probable cases (4-week-ahead)")
    ax[0].legend(fontsize=5.6, loc="upper right", handlelength=1.8)
    fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig5_fans.png"); plt.close(fig)


def fig6():
    A = R("analysis"); dd = A["decision_summary"]; S = R("decision")["_sensitivity"]
    order = ["Reactive (current-week excess)", "Persistence", "NB-GLM", "LightGBM-Q", "LSTM-GBM", "Mechanistic-only",
             "Graph-GRU-NB", "GRU-NB", "VECTRA"]
    lab = {"Reactive (current-week excess)": "Reactive"}
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    y = np.arange(len(order))
    mu = [dd[m]["regret"][0] * 100 for m in order]; sd = [dd[m]["regret"][1] * 100 for m in order]
    ax[0].barh(y, mu, xerr=sd, color=["black" if m == "VECTRA" else ("0.45" if m.startswith("Reactive") else "0.75") for m in order],
               height=0.6, error_kw=dict(lw=0.7, capsize=1.5))
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab.get(m, m) for m in order], fontsize=6.3); ax[0].invert_yaxis()
    ax[0].set_xlabel("Mean allocation regret (%)")
    for k, (key, xl) in enumerate([("channel_pct", "Endemic channel percentile"), ("budget", "Weekly budget (team-weeks)")]):
        a = ax[k + 1]; xs = list(S[key].keys())
        for j, (m, nm) in enumerate([("Reactive", "Reactive"), ("LightGBM-Q", "LightGBM-Q"), ("GRU-NB", "GRU-NB"), ("VECTRA", "VECTRA")]):
            a.plot(range(len(xs)), [S[key][v][m] * 100 for v in xs], marker=MK[j], ms=3.5, color=GR[j] if m != "VECTRA" else "black",
                   ls=LS[j], lw=1.5 if m == "VECTRA" else 0.9, label=nm)
        a.set_xticks(range(len(xs))); a.set_xticklabels([str(int(float(v))) for v in xs]); a.set_xlabel(xl)
        a.set_ylabel("Regret (%)")
    ax[1].legend(fontsize=5.8, loc="lower right", borderaxespad=0.2); ax[1].set_ylim(6, 37)
    for i in range(3):
        panel_label(ax[i], "(" + "abc"[i] + ")")
    fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/fig6_decision.png"); plt.close(fig)


def fig7():
    C = R("analysis")["calibration_summary"]["VECTRA"]
    modes = [("raw@0.2", "Model-native"), ("split@0.2", "Split conformal"), ("aci_h@0.2", "ACI per horizon"),
             ("aci_region_h@0.2", "ACI per region-horizon")]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    h = np.arange(1, 9)
    for k, (m, nm) in enumerate(modes):
        ax[0].plot(h, C[m]["cov_h"], marker=MK[k], ms=3, color=GR[k] if k < 3 else "black", ls=LS[k], lw=1.5 if k == 3 else 0.9, label=nm)
    ax[0].axhline(0.8, color="black", lw=0.6, ls=":"); ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("Horizon $h$"); ax[0].set_ylabel("Empirical coverage (80% target)")
    ax[0].legend(fontsize=5.6, loc="lower left")
    x = np.arange(len(modes)); w = 0.36
    ax[1].bar(x - w / 2, [C[m]["cov_growth_h4"] for m, _ in modes], w, color="black", label="growth weeks")
    ax[1].bar(x + w / 2, [C[m]["cov_decline_h4"] for m, _ in modes], w, color="0.7", label="decline weeks")
    ax[1].axhline(0.8, color="black", lw=0.6, ls=":"); ax[1].set_ylim(0.4, 1.13); ax[1].set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax[1].set_xticks(x); ax[1].set_xticklabels(["native", "split", "ACI-h", "ACI-rh"], fontsize=6.4)
    ax[1].set_ylabel("Coverage at $h=4$"); ax[1].legend(fontsize=5.6, loc="upper left", ncol=2, borderaxespad=0.1, columnspacing=1.2)
    ax[2].bar(x, [C[m]["region_cov_sd"] for m, _ in modes], 0.55, color=["0.75", "0.75", "0.75", "black"])
    ax[2].set_xticks(x); ax[2].set_xticklabels(["native", "split", "ACI-h", "ACI-rh"], fontsize=6.4)
    ax[2].set_ylabel("SD of regional coverage")
    for i in range(3):
        panel_label(ax[i], "(" + "abc"[i] + ")")
    fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/fig7_calibration.png"); plt.close(fig)


def fig8():
    cf = R("counterfactual"); acts = R("decision")["_actions"]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    names = ["none"] + list(M.ACTIONS)
    for k, j in enumerate([0, 1]):
        a = ax[k]; hist = np.array(cf["history"][j]); xh = np.arange(-11, 1); xf = np.arange(1, 9)
        a.plot(xh, hist, color="black", lw=1.0, marker="o", ms=1.8)
        q = np.array(cf["none"]["q"])[:, j]
        a.fill_between(xf, q[0], q[2], color="0.85", lw=0, label="no action, 80% band")
        for n_, nm in enumerate(names):
            qq = np.array(cf[nm]["q"])[:, j]
            a.plot(xf, qq[1], color=GR[n_] if n_ else "black", ls=LS[n_], lw=1.1, label=nm.replace("_", " "))
        a.plot(xf, np.array(cf["truth"][j]), "o", ms=2.2, color="black", mfc="white", label="observed")
        a.axvline(0.5, color="0.5", lw=0.6, ls=":")
        a.set_xlabel("Weeks relative to forecast origin"); a.set_title(f"Mesoregion {cf['regions'][j]}", fontsize=7.2)
        panel_label(a, "(" + "ab"[k] + ")")
    ax[0].set_ylabel("Probable cases"); ax[0].legend(fontsize=5.4, loc="upper left", handlelength=2.0)
    T = np.array([a_["temp"] for a_ in acts]); C_ = np.array([a_["action"] for a_ in acts])
    bins = np.arange(np.floor(T.min()), np.ceil(T.max()) + 1, 1.0)
    bottom = np.zeros(len(bins) - 1)
    for k in range(3):
        c, _ = np.histogram(T[C_ == k], bins)
        ax[2].bar(bins[:-1] + 0.5, c, 0.9, bottom=bottom, color=["black", "0.55", "0.85"][k], edgecolor="black", lw=0.3,
                  label=M.ACTIONS[k].replace("_", " "))
        bottom += c
    ax[2].set_xlabel("Mean forecast temperature (°C)"); ax[2].set_ylabel("Allocated region-weeks")
    ax[2].set_ylim(0, bottom.max() * 1.62); ax[2].legend(fontsize=5.6, loc="upper right", borderaxespad=0.1); panel_label(ax[2], "(c)")
    fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig8_counterfactual.png"); plt.close(fig)


def fig9():
    Dp = R("deployment")["policies"]
    names = [("cloud", "Cloud-centric"), ("fog", "Fog-centric"), ("edge", "Edge-only"), ("greedy", "Latency-greedy"), ("vectra", "VECTRA placement")]
    ps = sorted(Dp["cloud"].keys(), key=float)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for k, (p, nm) in enumerate(names):
        st = dict(color="black" if p == "vectra" else GR[k % 4], ls=LS[k % 4], marker=MK[k], ms=3, lw=1.5 if p == "vectra" else 0.9, label=nm)
        ax[0].plot([float(x) for x in ps], [Dp[p][x]["new_plan"] for x in ps], **st)
        ax[1].plot([float(x) for x in ps], [Dp[p][x]["exp_regret"] * 100 for x in ps], **st)
    ax[0].set_xlabel("Backhaul outage probability"); ax[0].set_ylabel("Weeks with a fresh plan")
    ax[0].set_ylim(0.4, 1.04)
    ax[1].set_xlabel("Backhaul outage probability"); ax[1].set_ylabel("Expected regret (%)")
    ax[0].legend(fontsize=5.6, loc="lower left")
    x = np.arange(len(names))
    e = [Dp[p]["0.2"]["energy_J"] for p, _ in names]
    ax[2].bar(x, e, 0.6, color=["0.75"] * 4 + ["black"])
    ax[2].set_xticks(x); ax[2].set_xticklabels(["Cloud", "Fog", "Edge", "Greedy", "VECTRA"], fontsize=5.6)
    ax[2].set_ylabel("Energy per weekly cycle (J)")
    for i in range(3):
        panel_label(ax[i], "(" + "abc"[i] + ")")
    fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/fig9_deployment.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["3", "4", "5", "6", "7", "8", "9"]
    for w in which:
        globals()["fig" + w](); print("fig", w)
