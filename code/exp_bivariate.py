"""Study I: the controlled bivariate covariance grid (Section 5 of the paper).

Outputs (in ../results):
  bivariate_table.csv        error-bounded AUC/TPR/PL at alpha = 0.05 and 0.01 for all 25 scenarios
  bivariate_homoscedastic.csv closed form (Proposition 3) versus error-bounded values on the diagonal
  bivariate_mc.csv           Monte Carlo validation, N in {1000, 10000}, B = 200
  bivariate_curves.csv       error-bounded PL(alpha) curves, alpha in [1e-4, 0.5], all 25 scenarios
"""
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import plbound as pc

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
MU0, MU1 = np.array([1.0, 2.0]), np.array([2.0, 0.5])
LEVELS = {"N": 0.0, "L": 0.25, "M": 0.50, "H": 0.75, "P": 0.99}


def sigma(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def model(l0, l1):
    return pc.GaussianModel(MU0, MU1, sigma(LEVELS[l0]), sigma(LEVELS[l1]))


def error_bounded_table(eps=1e-10):
    rows = []
    for l0, l1 in product(LEVELS, LEVELS):
        m = model(l0, l1)
        for alpha in (0.05, 0.01):
            o = pc.performance_loss(m, alpha, eps=eps, with_auc=(alpha == 0.05))
            dist = pc.score_distributions(m)
            K_f = pc.scores(m)["f"][3]
            row = dict(rho0=l0, rho1=l1, alpha=alpha,
                       TPR_f=o["TPR_f"].mid, TPR_f_hw=o["TPR_f"].half_width,
                       TPR_nb=o["TPR_nb"].mid, TPR_nb_hw=o["TPR_nb"].half_width,
                       PL=o["PL"].mid, PL_lo=o["PL"].lo, PL_hi=o["PL"].hi, PL_hw=o["PL"].half_width,
                       t_f=o["t_f"], t_nb=o["t_nb"], roc_slope_f=float(np.exp(K_f + o["t_f"] / 2)),
                       r_f0=dist[("f", 0)].r, r_f1=dist[("f", 1)].r, seconds=o["seconds"])
            if alpha == 0.05:
                row.update(AUC_f=o["AUC_f"].mid, AUC_f_hw=o["AUC_f"].half_width,
                           AUC_nb=o["AUC_nb"].mid, AUC_nb_hw=o["AUC_nb"].half_width,
                           dAUC=o["dAUC"].mid, dAUC_hw=o["dAUC"].half_width)
            rows.append(row)
            print(f"({l0},{l1}) alpha={alpha}: PL={o['PL']}  [{o['seconds']:.1f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bivariate_table.csv", index=False)
    return df


def homoscedastic_check(table):
    rows = []
    for l in LEVELS:
        S = sigma(LEVELS[l])
        df_, dnb = pc.homoscedastic_separations(MU0, MU1, S)
        mx, a_star = pc.homoscedastic_max_pl(MU0, MU1, S)
        for alpha in (0.05, 0.01):
            closed = float(pc.homoscedastic_pl(MU0, MU1, S, alpha))
            ref = table.query("rho0 == @l and rho1 == @l and alpha == @alpha").iloc[0]
            rows.append(dict(level=l, rho=LEVELS[l], alpha=alpha, Delta_f=df_, Delta_nb=dnb,
                             PL_closed=closed, PL_numeric=ref.PL, abs_diff=abs(closed - ref.PL),
                             PL_hw=ref.PL_hw, max_PL=mx, alpha_star=a_star,
                             kantorovich_bound=pc.kantorovich_bound(S, df_)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bivariate_homoscedastic.csv", index=False)
    print(df.to_string(), flush=True)


def monte_carlo(table, B=200, seed=20261013):
    rng = np.random.default_rng(seed)
    rows = []
    for (l0, l1), N in product(product(LEVELS, LEVELS), (1000, 10000)):
        t0 = time.perf_counter()
        pl, af, anb = pc.mc_performance_loss(model(l0, l1), 0.05, N, B, rng, with_auc=True)
        se = pl.std(ddof=1) / np.sqrt(B)
        ref = table.query("rho0 == @l0 and rho1 == @l1 and alpha == 0.05").iloc[0]
        half = 1.9720 * se                                    # t_{0.975, 199}
        rows.append(dict(rho0=l0, rho1=l1, N=N, B=B, PL_mc=pl.mean(), PL_se=se,
                         ci_lo=pl.mean() - half, ci_hi=pl.mean() + half, PL_ref=ref.PL,
                         covered=bool(pl.mean() - half <= ref.PL <= pl.mean() + half),
                         z=(pl.mean() - ref.PL) / se if se > 0 else 0.0,
                         AUC_f_mc=af.mean(), AUC_f_ref=ref.AUC_f, AUC_nb_mc=anb.mean(),
                         AUC_nb_ref=ref.AUC_nb, seconds=time.perf_counter() - t0))
        print(f"MC ({l0},{l1}) N={N}: {pl.mean():.4f} +/- {se:.4f}  ref={ref.PL:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bivariate_mc.csv", index=False)


def curves():
    alphas = np.geomspace(1e-4, 0.5, 60)
    rows = []
    for l0, l1 in product(LEVELS, LEVELS):
        t0 = time.perf_counter()
        c = pc.pl_curve(model(l0, l1), alphas, eps=1e-9)
        for a, f, nb, pl in zip(alphas, c["TPR_f"], c["TPR_nb"], c["PL"]):
            rows.append(dict(rho0=l0, rho1=l1, alpha=a, TPR_f=f.mid, TPR_nb=nb.mid, PL=pl.mid,
                             PL_hw=pl.half_width))
        print(f"curve ({l0},{l1}) [{time.perf_counter()-t0:.1f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bivariate_curves.csv", index=False)


if __name__ == "__main__":
    tab = error_bounded_table()
    homoscedastic_check(tab)
    curves()
    monte_carlo(tab)
