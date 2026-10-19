"""Covariance designs for the higher-dimensional studies (Section 7 of the paper)."""
import numpy as np

import plbound as pc


def ar1(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def unit_direction(p, seed=1):
    u = np.random.default_rng(seed).standard_normal(p)
    return u / np.linalg.norm(u)


def structured_model(p, rho0, rho1, var_ratio=1.0, sep=2.0, seed=1):
    """AR(1) correlation in each class; class-1 variances log-linear from 1 to var_ratio.

    var_ratio = 1 gives D0 = D1 = I (a linear independence score, as in the bivariate grid);
    var_ratio > 1 makes the independence score itself quadratic (A_nb != 0)."""
    v1 = var_ratio ** (np.arange(p) / max(p - 1, 1))
    S0 = ar1(p, rho0)
    S1 = np.sqrt(v1)[:, None] * ar1(p, rho1) * np.sqrt(v1)[None, :]
    return pc.GaussianModel(np.zeros(p), sep * unit_direction(p, seed), S0, S1)


def random_correlation(p, nu, rng):
    """Correlation matrix of a Wishart(I_p, p + nu) draw; small nu gives strong dependence."""
    G = rng.standard_normal((p + nu, p))
    W = G.T @ G
    d = np.sqrt(np.diag(W))
    return W / np.outer(d, d)


def random_model(p, rng, auc_range=(0.80, 0.99), max_tries=200):
    """Random two-class Gaussian model with heterogeneous dependence and variances.

    Sigma_1 = (1 - w) Sigma_0 + w Sigma_1*, with independent random correlation structures,
    and the draw is accepted only if AUC_f lies in auc_range (non-trivial difficulty)."""
    for _ in range(max_tries):
        nu0, nu1 = (int(v) for v in rng.integers(1, 2 * p + 1, size=2))
        sd0 = np.exp(0.25 * rng.standard_normal(p))
        sd1 = sd0 * np.exp(0.25 * rng.standard_normal(p))
        S0 = sd0[:, None] * random_correlation(p, nu0, rng) * sd0[None, :]
        S1s = sd1[:, None] * random_correlation(p, nu1, rng) * sd1[None, :]
        w = rng.uniform(0.0, 1.0)
        S1 = (1 - w) * S0 + w * S1s
        u = rng.standard_normal(p)
        u /= np.linalg.norm(u)
        sep = rng.uniform(0.5, 2.5)
        m = pc.GaussianModel(np.zeros(p), sep * u * sd0, S0, S1)
        auc = pc.error_bounded_auc(m, "f", eps=1e-6).mid
        if auc_range[0] <= auc <= auc_range[1]:
            return m, dict(nu0=nu0, nu1=nu1, w=w, sep=sep)
    raise RuntimeError("no admissible model found")
