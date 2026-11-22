"""Deadline- and energy-aware placement of the weekly VECTRA workload over an edge-fog-cloud continuum.

Workloads (CPU cycles) come from timings measured while running the actual pipeline; tier speeds,
power constants, link rates and outage probabilities are scenario assumptions (see DEFAULTS)."""
import numpy as np

DEFAULTS = dict(
    f_edge=(0.6e9, 1.5e9), f_fog=(1.2e9, 3.0e9), f_cloud=(3.0e9, 3.0e9),
    cores_edge=4, cores_fog=8, cores_cloud=32,
    kappa_edge=1.0e-27, kappa_fog=1.5e-27, kappa_cloud=2.0e-27,
    p_tx_edge=0.8,                       # W, edge radio while transmitting
    rate_ef=5e6, rtt_ef=0.04,            # edge<->fog (bit/s, s)
    rate_fc=50e6, rtt_fc=0.08,           # fog<->cloud
    p_out_ef=0.05, deadline=2.0,
)


def dvfs_freq(cycles, cores, slack, fmin, fmax):
    """Lowest frequency that finishes `cycles` (parallel over `cores`) within `slack` seconds."""
    if slack <= 0:
        return None
    f = cycles / (cores * slack)
    if f > fmax:
        return None
    return max(f, fmin)


def task_cost(cycles, cores, f, kappa):
    t = cycles / (cores * f)
    e = kappa * cycles * f ** 2          # dynamic energy  E = kappa * C * f^2
    return t, e


class Simulator:
    def __init__(self, W, n_regions, groups, cfg=None):
        """W: dict of measured workloads in cycles and payload sizes in bits:
        W['infer'] per-region inference (S paths), W['calib'] per-region calibration update,
        W['plan_global'] global allocation LP, W['plan_state'] per-state LP,
        W['in_bits'] per-region input, W['out_bits'] per-region scenario output, W['plan_bits']."""
        self.W, self.N, self.groups = W, n_regions, np.asarray(groups)
        self.c = dict(DEFAULTS, **(cfg or {}))
        self.states = sorted(set(groups))

    def run_week(self, policy, link_fc, link_ef, rng):
        """One weekly cycle. Returns the plan mode actually delivered, energy (J) and latency (s).
        Modes: global / partial (cloud LP, partial = some regions reused last forecasts), state_share (fog LPs
        with last coordinated state shares), state_equal (fog LPs with equal shares), stale (last plan reused),
        reactive (local threshold rule)."""
        c, W = self.c, self.W
        E = 0.0
        ef_up, fc_up, D = link_ef, link_fc, c["deadline"]
        n_up = int(ef_up.sum())
        if policy == "edge":
            t, e = task_cost(W["infer"] + W["calib"], c["cores_edge"], c["f_edge"][1], c["kappa_edge"])
            return dict(ok=t <= D, mode="reactive", energy=self.N * e, latency=t)
        if policy == "cloud":
            E += n_up * c["p_tx_edge"] * W["in_bits"] / c["rate_ef"]
            if not fc_up:
                return dict(ok=False, mode="stale", energy=E, latency=np.nan)
            comm = c["rtt_ef"] + W["in_bits"] / c["rate_ef"] + c["rtt_fc"] + n_up * W["in_bits"] / c["rate_fc"]
            t, e = task_cost(self.N * (W["infer"] + W["calib"]) + W["plan_global"], c["cores_cloud"], c["f_cloud"][1], c["kappa_cloud"])
            L = comm + t + c["rtt_fc"] + W["plan_bits"] / c["rate_fc"] + c["rtt_ef"] + W["plan_bits"] / c["rate_ef"]
            return dict(ok=L <= D, mode=("global" if n_up == self.N else "partial") if L <= D else "stale", energy=E + e, latency=L)
        # fog, greedy and vectra share the regional tier
        if policy == "vectra":
            f = dvfs_freq(W["infer"] + W["calib"], c["cores_edge"], 0.5 * D, *c["f_edge"]) or c["f_edge"][1]
            t_inf, e = task_cost(W["infer"] + W["calib"], c["cores_edge"], f, c["kappa_edge"])
            E += self.N * e + n_up * c["p_tx_edge"] * W["out_bits"] / c["rate_ef"]
            t_ready = t_inf + c["rtt_ef"] + W["out_bits"] / c["rate_ef"]
        else:
            E += n_up * c["p_tx_edge"] * W["in_bits"] / c["rate_ef"]
            per_state = max(np.sum(self.groups == s) for s in self.states)
            t_inf, e = task_cost(per_state * (W["infer"] + W["calib"]), c["cores_fog"], c["f_fog"][1], c["kappa_fog"])
            E += len(self.states) * e
            t_ready = c["rtt_ef"] + W["in_bits"] / c["rate_ef"] + t_inf
        if policy == "fog":
            t_p, e = task_cost(W["plan_state"], c["cores_fog"], c["f_fog"][1], c["kappa_fog"])
            L = t_ready + t_p + c["rtt_ef"]
            return dict(ok=L <= D, mode="state_equal", energy=E + len(self.states) * e, latency=L)
        if fc_up:
            comm = c["rtt_fc"] + n_up * (W["out_bits"] if policy == "vectra" else W["in_bits"]) / c["rate_fc"]
            t_p, e = task_cost(W["plan_global"], c["cores_cloud"], c["f_cloud"][1], c["kappa_cloud"])
            L = t_ready + comm + t_p + c["rtt_fc"] + W["plan_bits"] / c["rate_fc"] + c["rtt_ef"]
            if L <= D:
                return dict(ok=True, mode="global" if n_up == self.N else "partial", energy=E + e, latency=L)
        if policy == "greedy":
            return dict(ok=False, mode="stale", energy=E, latency=np.nan)
        slack = D - t_ready - c["rtt_ef"]
        f = dvfs_freq(W["plan_state"], c["cores_fog"], max(slack, 1e-3), *c["f_fog"]) or c["f_fog"][1]
        t_p, e = task_cost(W["plan_state"], c["cores_fog"], f, c["kappa_fog"])
        L = t_ready + t_p + c["rtt_ef"]
        return dict(ok=L <= D, mode="state_share", energy=E + len(self.states) * e, latency=L)

    def simulate(self, policy, p_out_fc, weeks=52, reps=200, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for r in range(reps):
            for w in range(weeks):
                fc = rng.random() > p_out_fc
                ef = rng.random(self.N) > self.c["p_out_ef"]
                res = self.run_week(policy, fc, ef, rng)
                out.append(res)
        return out
