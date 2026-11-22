"""Schematic figures (Fig. 1 system overview, Fig. 2 forecaster and decision layer).
Black and white, orthogonally routed arrows, no overlapping blocks or labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
                     "mathtext.fontset": "dejavuserif"})
DASH = (0, (4, 2.5)); DOT = (0, (1, 1.5))


def box(ax, x0, y0, x1, y1, text, fs=6.6, lw=0.9, fc="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                lw=lw, ec="black", fc=fc, ls=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def route(ax, pts, ls="-", lw=0.9, label=None, seg=0, off=(0, 0), fs=6.0, ha="center"):
    """Polyline with an arrowhead on the last segment; optional label next to segment `seg`."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color="black", lw=lw, ls=ls, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=lw, ls=ls,
                                 color="black", shrinkA=0, shrinkB=0))
    if label:
        (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], label, fontsize=fs, ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=0.4))


def tier(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0", lw=0.7, ec="black",
                                fc="#efefef", ls=DASH))
    ax.text(x0 + 0.8, y1 - 0.8, title, fontsize=6.8, fontweight="bold", ha="left", va="top")


def fig1(path):
    fig, ax = plt.subplots(figsize=(7.16, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 59); ax.axis("off")
    tier(ax, 18, 1.5, 43, 52, "EDGE (district)")
    tier(ax, 47, 1.5, 70, 52, "FOG (regional)")
    tier(ax, 74, 1.5, 99.5, 52, "CLOUD")
    R1, R2, R3 = (35, 46), (19.5, 30.5), (4, 15)
    box(ax, 0.5, R1[0], 15, R1[1], "Weekly probable\ncases (SINAN)", fs=6.1)
    box(ax, 0.5, R2[0], 15, R2[1], "Station climate\n(INMET): $T$,\nhumidity, rain", fs=6.1)
    box(ax, 0.5, R3[0], 15, R3[1], "Coupling graph\n$\\mathbf{A}$ (training\nco-movement)", fs=6.1)
    box(ax, 20, R1[0], 41, R1[1], "12-week feature\nwindow $\\mathbf{x}_{i,t}$")
    box(ax, 20, R2[0], 41, R2[1], "Thermally propagated\nrenewal forecaster", lw=1.4)
    box(ax, 20, R3[0], 41, R3[1], "Scenario sampler\n($S$ paths, factual and\nintervention-shifted)")
    box(ax, 49, R1[0], 68, R1[1], "Adaptive conformal\ncalibration per\n(region, horizon)")
    box(ax, 49, R2[0], 68, R2[1], "State-level\nallocation LP\n(fallback)")
    box(ax, 49, R3[0], 68, R3[1], "Deadline/DVFS\nplacement and\nplan dispatch")
    box(ax, 76, R1[0], 97.5, R1[1], "Training: NB likelihood\n+ closure penalty\n(weekly refit)")
    box(ax, 76, R2[0], 97.5, R2[1], "Global excess-burden\nallocation LP", lw=1.4)
    box(ax, 76, R3[0], 97.5, R3[1], "Model registry and\nweekly control plan")
    # data -> edge
    route(ax, [(15, 40.5), (20, 40.5)])
    route(ax, [(15, 25), (20, 25)])
    route(ax, [(15, 9.5), (17.3, 9.5), (17.3, 22), (20, 22)])
    route(ax, [(30.5, 35), (30.5, 30.5)])
    route(ax, [(30.5, 19.5), (30.5, 15)])
    # edge -> fog
    route(ax, [(41, 9.5), (45, 9.5), (45, 38), (49, 38)], label="scenarios", seg=1, off=(0, 0), fs=5.8)
    route(ax, [(41, 43), (49, 43)], ls=DASH)
    ax.text(45, 44.4, "$y_{t}$", fontsize=6.2, ha="center", va="bottom")
    route(ax, [(58.5, 35), (58.5, 30.5)], label="calibrated", off=(0, 0), fs=5.8)
    route(ax, [(58.5, 19.5), (58.5, 15)], label="fallback plan", off=(0, 0), fs=5.8)
    # fog <-> cloud
    route(ax, [(68, 43), (76, 43)], ls=DASH)
    ax.text(72, 44.4, "$y_{t+h}$", fontsize=6.2, ha="center", va="bottom")
    route(ax, [(68, 38), (72, 38), (72, 25), (76, 25)], label="$\\Delta$", seg=1, off=(0, 0), fs=6.2)
    route(ax, [(86.75, 19.5), (86.75, 15)], label="plan $\\mathbf{x}^{\\star}$", off=(0, 0), fs=5.8)
    route(ax, [(76, 9.5), (68, 9.5)])
    route(ax, [(49, 6.5), (41, 6.5)])
    ax.text(45, 4.6, "plan", fontsize=5.8, ha="center", va="top")
    # cloud -> edge model weights (above the tiers)
    route(ax, [(86.75, 46), (86.75, 55.5), (38.5, 55.5), (38.5, 46)], label="updated model weights", seg=1,
          off=(0, 0), fs=5.8)
    # legend
    ax.plot([14, 20], [-3, -3], color="black", lw=0.9); ax.text(21, -3, "data and plan flow", va="center", fontsize=6)
    ax.plot([46, 52], [-3, -3], color="black", lw=0.9, ls=DASH); ax.text(53, -3, "outcome feedback", va="center", fontsize=6)
    fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)


def fig2(path):
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 57); ax.axis("off")
    R1, R2, R3 = (42, 54), (26, 38), (10, 22)
    A, B, C, Dd, E = (0.5, 16.5), (21, 40), (45, 60), (65, 81), (86, 99.5)
    fs = 6.1
    box(ax, *A[:1], R1[0], A[1], R1[1], "Case history\n$y_{i,t-11:t}$ and\n$(\\mathbf{A}\\mathbf{y})_{i}$", fs=fs)
    box(ax, A[0], R2[0], A[1], R2[1], "Trailing\ntemperature $\\bar{T}_{i,t}$\nand climatology", fs=fs)
    box(ax, A[0], R3[0], A[1], R3[1], "Feature window\n$\\mathbf{x}_{i,t}$ (cases,\nRH, rain, delay)", fs=fs)
    box(ax, B[0], R1[0], B[1], R1[1], "Origin estimate\n$\\hat{R}_{i,t}$ with\nkernel $w(\\bar{T})$", fs=fs)
    box(ax, B[0], R2[0], B[1], R2[1], "Thermal suitability\n$\\log S(\\bar{T})$ and\nEIP $n(\\bar{T})$", fs=fs)
    box(ax, B[0], R3[0], B[1], R3[1], "Bounded GRU\nclosure\n$\\delta_h$, $\\kappa_h$, $\\nu_h$", fs=fs)
    box(ax, C[0], R3[0], C[1], R1[1], "Horizon-wise\n$\\log R_{i,t+h}$\n\n$\\rho^{h}\\log\\hat{R}_{i,t}$\n$+(1-\\rho^{h})b_i$\n$+\\gamma\\,\\Delta\\log S$\n$+\\delta_h$", fs=fs, lw=1.4)
    box(ax, Dd[0], R1[0], Dd[1], R1[1], "Coupled\nrenewal recursion\n$\\lambda_{i,t+h}$ ($\\omega$, $\\mathbf{A}$)", fs=fs)
    box(ax, E[0], R1[0], E[1], R1[1], "NB\nobservation\n($\\nu_h$, $r_{i,h}$)", fs=fs)
    box(ax, Dd[0], R3[0], Dd[1], R3[1], "Intervention\nshift $\\phi_a(\\bar{T})$\non $\\log R$, $h\\geq 3$", fs=fs)
    box(ax, E[0], R3[0], E[1], R3[1], "Averted\nexcess $\\Delta_{i,a}$\nto the LP", fs=fs)
    box(ax, 21, 0, 81, 5.5, "Training loss: NB negative log-likelihood + closure penalty", ls=DOT, fs=fs)
    route(ax, [(A[1], 48), (B[0], 48)])
    route(ax, [(A[1], 32), (B[0], 32)])
    route(ax, [(A[1], 16), (B[0], 16)])
    route(ax, [(8.5, 38), (8.5, 40), (18.7, 40), (18.7, 45), (B[0], 45)])
    route(ax, [(B[1], 48), (C[0], 48)])
    route(ax, [(B[1], 32), (C[0], 32)])
    route(ax, [(B[1], 16), (C[0], 16)])
    route(ax, [(C[1], 48), (Dd[0], 48)])
    route(ax, [(Dd[1], 48), (E[0], 48)])
    route(ax, [(C[1], 16), (Dd[0], 16)])
    route(ax, [(73, 22), (73, 42)], label="counterfactual\nre-simulation", fs=5.6)
    route(ax, [(95.5, 42), (95.5, 22)], label="factual and\nshifted paths", fs=5.6)
    route(ax, [(89, 42), (89, 31), (83.5, 31), (83.5, 2.75), (81, 2.75)])
    ax.text(86.3, 32.6, "$y_{t+h}$", fontsize=6.0, ha="center", va="bottom")
    route(ax, [(30.5, 5.5), (30.5, 10)], ls=DASH)
    ax.text(31.7, 7.7, "$\\nabla_{\\theta}$ (reverse mode)", fontsize=5.8, ha="left", va="center")
    route(ax, [(52.5, 5.5), (52.5, 10)], ls=DASH)
    ax.text(53.7, 7.7, "$\\nabla_{b,\\gamma,\\rho,\\omega}$", fontsize=5.8, ha="left", va="center")
    fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig1("fig1_system.png"); fig2("fig2_model.png")
