"""Computational benchmark (Section 6 of the paper).

Part A  CDF level (reference: the inversion at eps=1e-12, or for the three bivariate laws an
        independent conditioning integral, see indep_bivariate.py): error-bounded inversion (this paper) versus Davies (1980) and Imhof (1961) as
        implemented in the R package CompQuadForm, the Lugannani-Rice saddlepoint approximation
        (Kuonen 1999) and crude Monte Carlo, on generalized chi-square laws arising in the studies.
Part B  PL level: error-bounded PL(alpha) versus the Monte Carlo plug-in estimator, as a function of
        the attainable precision.

Run this script alone (timings are wall-clock on one core).
Outputs: ../results/bench_cdf.csv, ../results/bench_pl.csv
"""
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import designs as ds
import plbound as pc

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
LEVELS = (1e-4, 0.01, 0.5, 0.99, 0.9999)


def bivariate(r0, r1):
    S = lambda r: np.array([[1, r], [r, 1.0]])
    return pc.GaussianModel(np.array([1.0, 2.0]), np.array([2.0, 0.5]), S(r0), S(r1))


def cases():
    out = {}
    d = pc.score_distributions(bivariate(0.0, 0.99))
    out["biv (N,P) Q_f|Y=0, r=2"] = d[("f", 0)]
    out["biv (N,P) Q_f|Y=1, r=2"] = d[("f", 1)]
    out["biv (L,H) Q_f|Y=0, r=2"] = pc.score_distributions(bivariate(0.25, 0.75))[("f", 0)]
    out["biv (N,P) AUC difference, r=4"] = pc.difference_distribution(bivariate(0.0, 0.99), "f")
    m10, _ = ds.random_model(10, np.random.default_rng(10001))
    m50, _ = ds.random_model(50, np.random.default_rng(50001))
    out["random p=10 Q_f|Y=0, r=10"] = pc.score_distributions(m10)[("f", 0)]
    out["random p=10 Q_nb|Y=1, r=10"] = pc.score_distributions(m10)[("nb", 1)]
    out["random p=50 Q_f|Y=0, r=50"] = pc.score_distributions(m50)[("f", 0)]
    return out


def timed(fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        v = fn()
    return v, (time.perf_counter() - t0) / reps


def part_a():
    rows, rcases = [], []
    for name, g in cases().items():
        for lev in LEVELS:
            x = pc.saddlepoint_quantile(g, lev)
            ref_ev = pc.ErrorBoundedCDF(g, (x, x), eps=1e-12, max_terms=2 ** 25)
            ref, ref_err = ref_ev(x)
            base = dict(case=name, level=lev, x=x, ref=ref, ref_bound=ref_err)
            for eps in (1e-6, 1e-8, 1e-10):
                (F, e), sec = timed(lambda: pc.ErrorBoundedCDF(g, (x, x), eps=eps)(x))
                rows.append(dict(base, method="error-bounded", setting=f"eps={eps:g}", value=F, bound=e, seconds=sec))
            F, sec = timed(lambda: pc.saddlepoint_cdf(g, x), reps=20)
            rows.append(dict(base, method="saddlepoint", setting="", value=F, bound=np.nan, seconds=sec))
            rng = np.random.default_rng(1)
            F, sec = timed(lambda: float(pc.mc_cdf(g, x, 10 ** 6, rng)[0]))
            rows.append(dict(base, method="Monte Carlo", setting="n=1e6", value=F, bound=np.nan, seconds=sec))
            rcases.append(dict(case=name, level=lev, x=x, q=x - g.kappa, sigma=g.sigma0,
                               lam=";".join(f"{v:.17g}" for v in g.lam), delta=";".join(f"{v:.17g}" for v in g.delta)))
            print(f"{name} level={lev}: ref={ref:.15f} (+/- {ref_err:.1e})", flush=True)
    pd.DataFrame(rcases).to_csv(OUT / "bench_cases_for_R.csv", index=False)
    subprocess.run(["Rscript", str(HERE / "benchmark_cdf.R"), str(OUT / "bench_cases_for_R.csv"),
                    str(OUT / "bench_R.csv")], check=True)
    r = pd.read_csv(OUT / "bench_R.csv")
    py = pd.DataFrame(rows)
    ref = py.drop_duplicates(["case", "level"])[["case", "level", "x", "ref", "ref_bound"]]
    r = r.merge(ref, on=["case", "level"], how="left")
    df = pd.concat([py, r], ignore_index=True)
    df["abs_error"] = (df["value"] - df["ref"]).abs()
    df.to_csv(OUT / "bench_cdf.csv", index=False)
    summary = (df.groupby(["method", "setting"])
                 .agg(max_abs_error=("abs_error", "max"), median_seconds=("seconds", "median"),
                      max_seconds=("seconds", "max"), failures=("ifault", lambda v: int(np.nansum(v > 0))))
                 .reset_index())
    print(summary.to_string(), flush=True)


def part_b():
    rows = []
    models = {"biv (N,P)": bivariate(0.0, 0.99), "biv (L,H)": bivariate(0.25, 0.75),
              "random p=10": ds.random_model(10, np.random.default_rng(10001))[0],
              "random p=50": ds.random_model(50, np.random.default_rng(50001))[0]}
    for name, m in models.items():
        for eps in (1e-6, 1e-8, 1e-10):
            o, sec = timed(lambda: pc.performance_loss(m, 0.05, eps=eps, with_auc=False))
            rows.append(dict(model=name, method="error-bounded", setting=f"eps={eps:g}", PL=o["PL"].mid,
                             precision=o["PL"].half_width, seconds=sec))
        rng = np.random.default_rng(3)
        for N in (10 ** 4, 10 ** 5):
            pl, sec = timed(lambda: pc.mc_performance_loss(m, 0.05, N, 20, rng))
            sd = pl.std(ddof=1)
            # time per replicate and the CPU time needed for a 95% half-width of 1e-4 or 1e-6
            per_rep = sec / 20
            rows.append({"model": name, "method": "Monte Carlo", "setting": f"N={N:.0e}", "PL": pl.mean(),
                         "precision": 1.96 * sd, "seconds": per_rep,
                         "seconds_for_1e-4": per_rep * (1.96 * sd / 1e-4) ** 2,
                         "seconds_for_1e-6": per_rep * (1.96 * sd / 1e-6) ** 2})
        print(f"part B {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bench_pl.csv", index=False)
    print(df.to_string(), flush=True)


def refresh_bivariate_reference():
    """Replace the reference of the three bivariate laws by the independent conditioning integral
    (indep_bivariate.py), which shares no code with plbound, and recompute the errors."""
    import indep_bivariate as ib
    S = lambda r: np.array([[1, r], [r, 1.0]])
    laws = {"biv (N,P) Q_f|Y=0, r=2": (ib.coef(S(0.0), S(0.99)), ib.MU0, S(0.0)),
            "biv (N,P) Q_f|Y=1, r=2": (ib.coef(S(0.0), S(0.99)), ib.MU1, S(0.99)),
            "biv (L,H) Q_f|Y=0, r=2": (ib.coef(S(0.25), S(0.75)), ib.MU0, S(0.25))}
    df = pd.read_csv(OUT / "bench_cdf.csv")
    df["ref_method"] = "inversion, eps=1e-12"
    for (case, level), idx in df.groupby(["case", "level"]).groups.items():
        if case not in laws:
            continue
        (A, a, b), mu, Sig = laws[case]
        x = float(df.loc[idx, "x"].iloc[0])
        ref = ib.cdf(A, a, b, mu, Sig, x)
        old = float(df.loc[idx, "ref"].iloc[0])
        bound = float(df.loc[idx, "ref_bound"].iloc[0])
        print(f"{case} level={level}: independent - inversion = {ref - old:+.2e} (inversion bound {bound:.1e})")
        df.loc[idx, "ref"] = ref
        df.loc[idx, "ref_method"] = "conditioning integral"
    df["abs_error"] = (df["value"] - df["ref"]).abs()
    df.to_csv(OUT / "bench_cdf.csv", index=False)


if __name__ == "__main__":
    import sys
    if "--refresh-reference" in sys.argv:
        refresh_bivariate_reference()
    else:
        part_a()
        refresh_bivariate_reference()
        part_b()
