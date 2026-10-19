"""Figures for the paper (vector PDF, written to ../manuscript as Fig1.pdf ... Fig6.pdf)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plbound as pc

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "manuscript"

# validated categorical slots (blue, orange, aqua) and chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LEV = ["N", "L", "M", "H", "P"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
W1, W2 = 3.3, 6.85      # single / double column widths (inches) for the 174 mm text block


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def fig1_heatmaps():
    t = pd.read_csv(RES / "bivariate_table.csv")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f9fe"] + SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), constrained_layout=True)
    for ax, a in zip(axes, (0.05, 0.01)):
        d = t[t.alpha == a]
        M = np.array([[d[(d.rho0 == r0) & (d.rho1 == r1)].PL.iloc[0] for r1 in LEV] for r0 in LEV])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.55, origin="upper")
        for i in range(5):
            for j in range(5):
                v = M[i, j]
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7.2,
                        color="white" if v > 0.25 else INK)
        ax.set_xticks(range(5), LEV)
        ax.set_yticks(range(5), LEV)
        ax.set_xlabel(r"$\rho_1$ (positive class)")
        ax.set_ylabel(r"$\rho_0$ (negative class)")
        ax.set_title(rf"$\mathrm{{PL}}({a})$", color=INK, fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5)
    save(fig, "Fig1.pdf")


def fig3_transposition():
    """ROC transposition under covariance swap (Theorem 3) and the two loss curves."""
    c = pd.read_csv(RES / "bivariate_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.35), constrained_layout=True)
    for ax, (r0, r1) in zip(axes[:2], [("N", "P"), ("P", "N")]):
        d = c[(c.rho0 == r0) & (c.rho1 == r1)].sort_values("alpha")
        ax.plot(d.alpha, d.TPR_f, color=BLUE, label="full ($Q_f$)")
        ax.plot(d.alpha, d.TPR_nb, color=ORANGE, ls="--", label="independence ($Q_{nb}$)")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 0.5)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(r"false positive rate $\alpha$")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"ROC, ({r0},{r1})", color=INK, fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7)
    ax = axes[2]
    for (r0, r1), col, ls in [(("N", "P"), BLUE, "-"), (("P", "N"), ORANGE, "--")]:
        d = c[(c.rho0 == r0) & (c.rho1 == r1)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({r0},{r1})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"loss curves (equal $\Delta\mathrm{AUC}$)", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "Fig3.pdf")


def fig2_curves():
    """Loss curves for symmetric (homoscedastic) and asymmetric scenarios, with closed forms."""
    c = pd.read_csv(RES / "bivariate_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), constrained_layout=True, sharey=False)
    ax = axes[0]
    mu0, mu1 = np.array([1.0, 2.0]), np.array([2.0, 0.5])
    for lev, col, ls in [("M", BLUE, "-"), ("H", ORANGE, "--"), ("P", AQUA, "-.")]:
        d = c[(c.rho0 == lev) & (c.rho1 == lev)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({lev},{lev}) error-bounded")
        rho = {"M": 0.5, "H": 0.75, "P": 0.99}[lev]
        S = np.array([[1, rho], [rho, 1.0]])
        grid = np.geomspace(1e-4, 0.5, 13)
        ax.plot(grid, pc.homoscedastic_pl(mu0, mu1, S, grid), ls="none", marker="o", ms=3.2,
                mfc="white", mec=col, mew=1.0)
    ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=INK2, label="closed form (Prop. 2)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"(a) shared covariance, $\rho_0=\rho_1$", color=INK, fontsize=9)
    ax.legend(fontsize=6.8, loc="upper right")
    ax = axes[1]
    for (r0, r1), col, ls in [(("N", "H"), BLUE, "-"), (("L", "P"), ORANGE, "--"), (("M", "P"), AQUA, "-.")]:
        d = c[(c.rho0 == r0) & (c.rho1 == r1)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({r0},{r1})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"(b) unequal covariance, $\rho_0<\rho_1$", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "Fig2.pdf")


def fig4_mc():
    mc = pd.read_csv(RES / "bivariate_mc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), constrained_layout=True)
    ax = axes[0]
    d = mc[mc.N == 10000]
    ax.plot([0, 0.31], [0, 0.31], color=AXIS, lw=0.8)
    ax.errorbar(d.PL_ref, d.PL_mc, yerr=1.972 * d.PL_se, fmt="o", ms=3.5, color=BLUE, ecolor=BLUE,
                elinewidth=0.8, mec="white", mew=0.6)
    ax.set_xlabel(r"error-bounded $\mathrm{PL}(0.05)$")
    ax.set_ylabel(r"Monte Carlo mean, $N=10^4$")
    ax.set_title("(a) agreement, 95% intervals", color=INK, fontsize=9)
    ax = axes[1]
    x = np.arange(25)
    for N, col, mk, off in [(1000, ORANGE, "s", -0.15), (10000, BLUE, "o", 0.15)]:
        d = mc[mc.N == N].reset_index(drop=True)
        ax.plot(x + off, d.z, ls="none", marker=mk, ms=3.5, color=col, mec="white", mew=0.5,
                label=rf"$N={N:,}$".replace(",", "\\,"))
    ax.axhspan(-1.972, 1.972, color=GRID, alpha=0.5, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    lab = mc[mc.N == 10000].reset_index(drop=True)
    ax.set_xticks(x[::5], [f"({a},{b})" for a, b in zip(lab.rho0[::5], lab.rho1[::5])], fontsize=7)
    ax.set_ylabel(r"$(\overline{\mathrm{PL}}_{MC}-\mathrm{PL})/\mathrm{SE}$")
    ax.set_xlabel("scenario")
    ax.set_title("(b) standardised discrepancy", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncols=2, markerscale=1.6)
    save(fig, "Fig4.pdf")


def fig5_highdim():
    r = pd.read_csv(RES / "highdim_random.csv")
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, (p, d) in zip(axes, r.groupby("p")):
        ax.plot(d.dAUC, d.PL01, ls="none", marker="o", ms=3.0, color=BLUE, mec="white", mew=0.4, alpha=0.9)
        rho = d[["dAUC", "PL01"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"$p={p}$  (Spearman {rho:.2f})", color=INK, fontsize=8)
        ax.set_xlabel(r"$\Delta\mathrm{AUC}$")
    axes[0].set_ylabel(r"$\mathrm{PL}(0.01)$")
    save(fig, "Fig5.pdf")


def fig6_bssr1():
    s = pd.read_csv(RES / "bssr1_summary.csv")
    configs = list(dict.fromkeys(s.config))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True, sharey=True)
    for ax, a in zip(axes, (0.001, 0.01)):
        d = s[s.alpha == a].set_index("config").loc[configs]
        y = np.arange(len(configs))[::-1]
        ax.errorbar(d.PL_pboot_mean, y + 0.18, xerr=1.96 * d.PL_pboot_sd, fmt="none", ecolor=AQUA, elinewidth=3,
                    alpha=0.55, label="Gaussian-model sampling range")
        ax.errorbar(d.PL_emp_mean, y - 0.12, xerr=[np.clip(d.PL_emp_mean - d.PL_emp_q025, 0, None), np.clip(d.PL_emp_q975 - d.PL_emp_mean, 0, None)],
                    fmt="s", ms=4, color=ORANGE, ecolor=ORANGE, elinewidth=1, mec="white", mew=0.5,
                    label="empirical (test halves)")
        ax.plot(d.PL_model_full, y + 0.18, ls="none", marker="o", ms=5, color=BLUE, mec="white", mew=0.6,
                label="error-bounded, fitted model")
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(y, configs)
        ax.set_xlabel(rf"$\mathrm{{PL}}({a})$")
        ax.set_title(f"FAR = {a:g}", color=INK, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncols=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, "Fig6.pdf")


if __name__ == "__main__":
    for f in (fig1_heatmaps, fig2_curves, fig3_transposition, fig4_mc, fig5_highdim, fig6_bssr1):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
