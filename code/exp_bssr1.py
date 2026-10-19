"""Study III: multimodal biometric score fusion on NIST BSSR1 (Section 8 of the paper).

Data (prepared by prepare_bssr1.py from https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1)
  Set 1  fing_x_face   517 subjects, one probe each, matchers face C, face G, finger LI, finger RI
  Set 2  face_x_face  3000 subjects, two probes each, matchers face C and face G
  Set 3  fing_x_fing  6000 subjects, one probe each, matchers finger LI and finger RI
Failed comparisons (face C score -1) are removed; probes or enrollees whose comparisons all
failed are removed with them.

Protocol
  * Each matcher score is mapped by a monotone Yeo-Johnson transformation (after log1p for the
    fingerprint matcher), its parameter chosen on training data to maximise the class-conditional
    Gaussian likelihood (classes weighted equally).
  * Full-data fit: Gaussian class models fitted to all subjects; error-bounded model-based PL(alpha).
  * Repeated 50/50 subject-disjoint splits: models fitted on training subjects; the full and the
    independence-based Gaussian LR fusion rules are applied to test subjects and the empirical TPR
    at empirical FAR alpha is recorded.
  * Parametric bootstrap: the same split protocol applied to data simulated from the full-data
    fitted Gaussian model, giving the sampling distribution of the empirical PL when the Gaussian
    model holds.

Outputs: ../results/bssr1_fullfit.csv, bssr1_splits.csv, bssr1_pboot.csv, bssr1_summary.csv
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import plbound as pc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
ALPHAS = (0.001, 0.01, 0.05)
FIT_SUBSAMPLE = 200_000


# ---------------------------------------------------------------------------------------------
# Data sets
# ---------------------------------------------------------------------------------------------
class ScoreSet:
    """Similarity scores S[probe, enrollee, matcher] with subject labels and a validity mask."""

    def __init__(self, name, S, probe_subj, enr_subj, matchers):
        self.name, self.S, self.matchers = name, S, matchers
        _, codes = np.unique(np.concatenate([probe_subj, enr_subj]), return_inverse=True)
        self.ps, self.es = codes[: len(probe_subj)], codes[len(probe_subj):]
        valid = np.ones(S.shape[:2], bool)
        if "face_C" in matchers:
            valid &= S[..., matchers.index("face_C")] != -1
        rows, cols = valid.any(axis=1), valid.any(axis=0)
        self.S, self.valid = self.S[rows][:, cols], valid[rows][:, cols]
        self.ps, self.es = self.ps[rows], self.es[cols]
        self.subjects = np.intersect1d(self.ps, self.es)

    def pairs(self, cols, subj=None):
        subj = self.subjects if subj is None else subj
        pr, er = np.isin(self.ps, subj), np.isin(self.es, subj)
        S = self.S[pr][:, er][..., cols].astype(float)
        V = self.valid[pr][:, er]
        same = self.ps[pr][:, None] == self.es[er][None, :]
        return S[same & V], S[~same & V]


def load_sets():
    d1 = np.load(ROOT / "data" / "bssr1_set1.npz")
    m1 = ["face_C", "face_G", "finger_LI", "finger_RI"]
    set1 = ScoreSet("Set 1", np.stack([d1[k] for k in m1], -1), d1["user_id"], d1["enrollee_id"], m1)
    d2 = np.load(ROOT / "data" / "bssr1_set2_face.npz")
    set2 = ScoreSet("Set 2", np.stack([d2["face_C"], d2["face_G"]], -1), d2["probe_id"], d2["enrollee_id"],
                    ["face_C", "face_G"])
    d3 = np.load(ROOT / "data" / "bssr1_set3_finger.npz")
    set3 = ScoreSet("Set 3", np.stack([d3["finger_LI"], d3["finger_RI"]], -1), d3["probe_id"],
                    d3["enrollee_id"], ["finger_LI", "finger_RI"])
    return {"Set 1": set1, "Set 2": set2, "Set 3": set3}


CONFIGS = [("face C + face G", "Set 2", ["face_C", "face_G"]),
           ("finger LI + finger RI", "Set 3", ["finger_LI", "finger_RI"]),
           ("face C + finger LI", "Set 1", ["face_C", "finger_LI"]),
           ("all four matchers", "Set 1", ["face_C", "face_G", "finger_LI", "finger_RI"])]


# ---------------------------------------------------------------------------------------------
# Marginal transformations and Gaussian fusion
# ---------------------------------------------------------------------------------------------
def _pre(x, name):
    return np.log1p(x) if name.startswith("finger") else x


def fit_transform(gen, imp, name, rng):
    if imp.size > FIT_SUBSAMPLE:
        imp = rng.choice(imp, FIT_SUBSAMPLE, replace=False)
    g, i = _pre(gen, name), _pre(imp, name)
    pooled = np.concatenate([g, i])
    m, s = pooled.mean(), pooled.std()
    zg, zi = (g - m) / s, (i - m) / s

    def nll(lmb):
        tot = 0.0
        for z in (zg, zi):
            y = stats.yeojohnson(z, lmbda=lmb)
            jac = (lmb - 1) * np.sign(z) * np.log1p(np.abs(z))
            tot -= np.mean(stats.norm.logpdf(y, y.mean(), y.std())) + np.mean(jac)
        return tot

    lmb = optimize.minimize_scalar(nll, bounds=(-3, 5), method="bounded").x
    return dict(name=name, m=m, s=s, lmb=lmb)


def apply_transforms(X, trs):
    return np.column_stack([stats.yeojohnson((_pre(X[:, j], tr["name"]) - tr["m"]) / tr["s"], lmbda=tr["lmb"])
                            for j, tr in enumerate(trs)])


def fit_model(tg, ti):
    return pc.GaussianModel(ti.mean(0), tg.mean(0), np.atleast_2d(np.cov(ti.T)), np.atleast_2d(np.cov(tg.T)))


def fused(model, X):
    return {m: pc.quad_form_values(X, A, a, b) for m, (A, a, b, _) in pc.scores(model).items()}


def empirical_tpr(sg, si, alpha):
    """TPR at the smallest threshold whose empirical FAR does not exceed alpha (ties rejected)."""
    k = int(math.floor(alpha * si.size))
    top = np.sort(np.partition(si, si.size - k - 1)[si.size - k - 1:])[::-1]   # k+1 largest, descending
    thr = top[k]
    return float(np.mean(sg > thr)), float(np.mean(si > thr))


def empirical_auc(sg, si, max_imp=2_000_000, rng=None):
    if si.size > max_imp:
        si = rng.choice(si, max_imp, replace=False)
    return float(stats.mannwhitneyu(sg, si).statistic / (sg.size * si.size))


def evaluate(model, tg_te, ti_te, rng):
    fg, fi = fused(model, tg_te), fused(model, ti_te)
    row = {f"AUC_{m}_emp": empirical_auc(fg[m], fi[m], rng=rng) for m in ("f", "nb")}
    for a in ALPHAS:
        tf, _ = empirical_tpr(fg["f"], fi["f"], a)
        tn, _ = empirical_tpr(fg["nb"], fi["nb"], a)
        row.update({f"TPR_f_emp_{a}": tf, f"TPR_nb_emp_{a}": tn, f"PL_emp_{a}": tf - tn})
    return row


def model_pl(model, eps):
    row = {}
    for a in ALPHAS:
        o = pc.performance_loss(model, a, eps=eps, with_auc=(a == ALPHAS[0]))
        row.update({f"PL_model_{a}": o["PL"].mid, f"PL_model_hw_{a}": o["PL"].half_width,
                    f"TPR_f_model_{a}": o["TPR_f"].mid, f"TPR_nb_model_{a}": o["TPR_nb"].mid})
        if a == ALPHAS[0]:
            row.update(AUC_f_model=o["AUC_f"].mid, AUC_nb_model=o["AUC_nb"].mid, dAUC_model=o["dAUC"].mid)
    return row


# ---------------------------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------------------------
def full_fit(sets, rng):
    rows, models = [], {}
    for name, sname, keys in CONFIGS:
        ss = sets[sname]
        cols = [ss.matchers.index(k) for k in keys]
        gen, imp = ss.pairs(cols)
        trs = [fit_transform(gen[:, j], imp[:, j], keys[j], rng) for j in range(len(cols))]
        tg, ti = apply_transforms(gen, trs), apply_transforms(imp, trs)
        model = fit_model(tg, ti)
        models[name] = (model, trs, gen.shape[0], imp.shape[0])
        Rg, Ri = np.corrcoef(tg.T), np.corrcoef(ti.T)
        iu = np.triu_indices_from(Rg, 1)
        row = dict(config=name, set=sname, subjects=len(ss.subjects), n_gen=gen.shape[0], n_imp=imp.shape[0],
                   corr_gen=";".join(f"{v:.3f}" for v in Rg[iu]), corr_imp=";".join(f"{v:.3f}" for v in Ri[iu]),
                   lambdas=";".join(f"{t['lmb']:.3f}" for t in trs), **model_pl(model, 1e-10))
        row.update({f"{k}_insample": v for k, v in evaluate(model, tg, ti, rng).items()})
        rows.append(row)
        print(f"[full] {name}: n_gen={gen.shape[0]} n_imp={imp.shape[0]} " +
              " ".join(f"PL({a})={row[f'PL_model_{a}']:.5f}/{row[f'PL_emp_{a}_insample']:.5f}" for a in ALPHAS),
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bssr1_fullfit.csv", index=False)
    return models


def repeated_splits(sets, R=50, R_model=20, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for name, sname, keys in CONFIGS:
        ss = sets[sname]
        cols = [ss.matchers.index(k) for k in keys]
        for r in range(R):
            t0 = time.perf_counter()
            perm = rng.permutation(ss.subjects)
            half = len(perm) // 2
            gen_tr, imp_tr = ss.pairs(cols, perm[:half])
            gen_te, imp_te = ss.pairs(cols, perm[half:])
            trs = [fit_transform(gen_tr[:, j], imp_tr[:, j], keys[j], rng) for j in range(len(cols))]
            model = fit_model(apply_transforms(gen_tr, trs), apply_transforms(imp_tr, trs))
            row = dict(config=name, split=r, n_gen_test=gen_te.shape[0], n_imp_test=imp_te.shape[0],
                       **evaluate(model, apply_transforms(gen_te, trs), apply_transforms(imp_te, trs), rng))
            if r < R_model:
                row.update(model_pl(model, 1e-8))
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
        print(f"[splits] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bssr1_splits.csv", index=False)
    return df


def parametric_bootstrap(models, B=50, seed=7):
    """Empirical PL of the split protocol when data are drawn from the fitted Gaussian model."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, (model, _, n_gen, n_imp) in models.items():
        L0, L1 = np.linalg.cholesky(model.S0), np.linalg.cholesky(model.S1)
        ng, ni = n_gen // 2, n_imp // 4                      # sizes of one half (impostor pairs ~ n^2)
        for b in range(B):
            draw = lambda n, mu, L: mu + rng.standard_normal((n, model.p)) @ L.T
            fitted = fit_model(draw(ng, model.mu1, L1), draw(ni, model.mu0, L0))
            row = evaluate(fitted, draw(ng, model.mu1, L1), draw(ni, model.mu0, L0), rng)
            rows.append(dict(config=name, rep=b, **row))
        print(f"[pboot] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bssr1_pboot.csv", index=False)
    return df


def summarise(full, splits, pboot):
    out = []
    for name, _, _ in CONFIGS:
        f = full[full.config == name].iloc[0]
        s, b = splits[splits.config == name], pboot[pboot.config == name]
        for a in ALPHAS:
            e = s[f"PL_emp_{a}"]
            out.append(dict(config=name, alpha=a, PL_model_full=f[f"PL_model_{a}"],
                            PL_model_split_mean=s[f"PL_model_{a}"].mean(),
                            PL_emp_mean=e.mean(), PL_emp_se=e.std(ddof=1) / math.sqrt(e.size),
                            PL_emp_q025=e.quantile(0.025), PL_emp_q975=e.quantile(0.975),
                            PL_pboot_mean=b[f"PL_emp_{a}"].mean(), PL_pboot_sd=b[f"PL_emp_{a}"].std(ddof=1),
                            TPR_f_model=f[f"TPR_f_model_{a}"], TPR_f_emp=s[f"TPR_f_emp_{a}"].mean(),
                            TPR_nb_model=f[f"TPR_nb_model_{a}"], TPR_nb_emp=s[f"TPR_nb_emp_{a}"].mean(),
                            dAUC_model=f["dAUC_model"], dAUC_emp=(s["AUC_f_emp"] - s["AUC_nb_emp"]).mean()))
    df = pd.DataFrame(out)
    df.to_csv(OUT / "bssr1_summary.csv", index=False)
    print(df.to_string(float_format=lambda v: f"{v:.5f}"), flush=True)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    sets = load_sets()
    for s in sets.values():
        print(s.name, "subjects:", len(s.subjects), "probes x enrollees:", s.S.shape[:2], flush=True)
    models = full_fit(sets, rng)
    splits = repeated_splits(sets)
    pboot = parametric_bootstrap(models)
    summarise(pd.read_csv(OUT / "bssr1_fullfit.csv"), splits, pboot)
