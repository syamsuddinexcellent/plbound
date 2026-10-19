"""Study II: beyond two dimensions (Section 7 of the paper).

Outputs (in ../results):
  highdim_structured.csv   AR(1) designs, p in {2,5,10,20,50}, (rho0, rho1) in {0,.3,.6,.9}^2,
                           variance ratio in {1, 2}; error-bounded PL(0.05), PL(0.01), AUCs
  highdim_random.csv       100 random admissible models for each p in {5,10,20,50}
  highdim_mc.csv           Monte Carlo check of 10 random models per p
  highdim_timing.csv       error-bounded run time versus p and tolerance
"""
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import designs as ds
import plbound as pc

OUT = Path(__file__).resolve().parent.parent / "results"
RHOS = (0.0, 0.3, 0.6, 0.9)


def summarise(m, eps=1e-10):
    o5 = pc.performance_loss(m, 0.05, eps=eps, with_auc=True)
    o1 = pc.performance_loss(m, 0.01, eps=eps, with_auc=False)
    return dict(PL05=o5["PL"].mid, PL05_hw=o5["PL"].half_width, PL01=o1["PL"].mid,
                PL01_hw=o1["PL"].half_width, TPR_f05=o5["TPR_f"].mid, TPR_nb05=o5["TPR_nb"].mid,
                TPR_f01=o1["TPR_f"].mid, TPR_nb01=o1["TPR_nb"].mid, AUC_f=o5["AUC_f"].mid,
                AUC_nb=o5["AUC_nb"].mid, dAUC=o5["dAUC"].mid, dAUC_hw=o5["dAUC"].half_width,
                seconds=o5["seconds"] + o1["seconds"])


def structured_job(args):
    p, r0, r1, vr = args
    m = ds.structured_model(p, r0, r1, var_ratio=vr)
    return dict(p=p, rho0=r0, rho1=r1, var_ratio=vr, **summarise(m))


def random_job(args):
    p, seed = args
    rng = np.random.default_rng(seed)
    m, info = ds.random_model(p, rng)
    dist = pc.score_distributions(m)
    return dict(p=p, seed=seed, **info, r_nb=dist[("nb", 0)].r, **summarise(m))


def mc_job(args):
    p, seed = args
    m, _ = ds.random_model(p, np.random.default_rng(seed))
    ref = pc.performance_loss(m, 0.05, eps=1e-10, with_auc=False)["PL"]
    pl = pc.mc_performance_loss(m, 0.05, 10000, 100, np.random.default_rng(seed + 1))
    se = pl.std(ddof=1) / np.sqrt(pl.size)
    return dict(p=p, seed=seed, PL_ref=ref.mid, PL_ref_hw=ref.half_width, PL_mc=pl.mean(),
                PL_se=se, z=(pl.mean() - ref.mid) / se)


def timing_job(args):
    p, seed, eps = args
    m, _ = ds.random_model(p, np.random.default_rng(seed))
    t0 = time.perf_counter()
    o = pc.performance_loss(m, 0.05, eps=eps, with_auc=False)
    return dict(p=p, seed=seed, eps=eps, seconds=time.perf_counter() - t0, PL_hw=o["PL"].half_width)


def run(pool, job, tasks, name):
    t0 = time.perf_counter()
    rows = pool.map(job, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df)} rows in {time.perf_counter()-t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    with Pool(6) as pool:
        run(pool, structured_job,
            [(p, r0, r1, vr) for p in (2, 5, 10, 20, 50) for r0, r1 in product(RHOS, RHOS) for vr in (1.0, 2.0)],
            "highdim_structured.csv")
        run(pool, random_job, [(p, 1000 * p + i) for p in (5, 10, 20, 50) for i in range(100)],
            "highdim_random.csv")
        run(pool, mc_job, [(p, 1000 * p + i) for p in (5, 10, 20, 50) for i in range(10)], "highdim_mc.csv")
    # timings are measured serially, without competing worker processes
    rows = [timing_job((p, 7000 + i, eps)) for p in (2, 5, 10, 20, 50, 100) for i in range(5)
            for eps in (1e-6, 1e-10)]
    pd.DataFrame(rows).to_csv(OUT / "highdim_timing.csv", index=False)
    print("highdim_timing.csv done", flush=True)
