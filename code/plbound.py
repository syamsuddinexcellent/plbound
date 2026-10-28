"""plbound: the fixed-false-positive-rate loss of the independence assumption in Gaussian
classification, computed with guaranteed error bounds and estimated from data.

Companion code for a manuscript by M. Syamsuddin Wisnubroto, Sumardi and Nanang Susyanto (under
review); the reference will be added on publication. Version 1.0.0, without the inference of
Section 6, is archived at https://doi.org/10.5281/zenodo.23472531.
Repository: https://github.com/syamsuddinexcellent/plbound (MIT licence)

This single file is both

* the Python module (``import plbound as pc``): generalized chi-square laws, error-bounded
  inversion of their distribution functions (Theorem 6), propagation of the bounds through the
  quantile (Proposition 7), enclosures of PL(alpha) = TPR_f(alpha) - TPR_nb(alpha), and the
  delta-method and bootstrap inference of Section 6; and
* the script that reproduces every number, table and figure of the paper in one run:

      python plbound.py

  It runs the exactness tests, the studies of Section 5, the simulation study of Section 6 and
  the application of Section 7, and writes the CSV files to ../results, the figures to
  ../results/figures and the LaTeX tables to ../results/tables. The full run takes about three and
  a half hours on a laptop (Apple M1, seven worker processes), most of it for the simulation study of
  Section 6, which can be interrupted and resumed; Section 7 needs about 8 GB of memory.

Requirements: Python >= 3.9 with numpy, scipy, pandas and matplotlib. The benchmark of
Section 5.3 also calls R (Rscript) with the package CompQuadForm, and Section 7 needs the NIST
BSSR1 scores: pass the downloaded bssr1.zip once with --bssr1 (see data/README.md). Steps whose
requirements are missing are skipped with a message.

Options
  --only STEPS   comma-separated subset of: tests, bivariate, explicit, highdim, benchmark, inference,
                 bssr1, figures, tables (default: all, in this order)
  --out DIR      folder for the results (default: ../results)
  --data DIR     folder for the prepared BSSR1 files (default: ../data)
  --bssr1 PATH   bssr1.zip from NIST, or the folder it was extracted to; converted into --data
  --workers N    worker processes for the parallel studies (default: 6)
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"            # CSV files; figures and tables go to OUT/figures and OUT/tables
DATA = REPO / "data"              # prepared NIST BSSR1 score matrices (Section 7)


# =============================================================================================
# Part 1. Generalized chi-square laws, error-bounded inversion and the loss
# =============================================================================================
U = np.finfo(float).eps / 2          # unit roundoff
LN2 = math.log(2.0)
N_BLOCKS = 160                        # dyadic blocks used in the tail bounds
NORMAL_CDF_ERR = 1e-15                # absolute accuracy of scipy's ndtr
NORMAL_TAIL_RELERR = 1e-13            # relative accuracy of ndtr in the tail (Cephes: 3.4e-14 on [-13, 0])


# ---------------------------------------------------------------------------------------------
# Generalized chi-square variable
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GenChi2:
    lam: np.ndarray
    eta: np.ndarray
    kappa0: float
    sigma0: float = 0.0

    def __post_init__(self):
        lam = np.atleast_1d(np.asarray(self.lam, float))
        eta = np.atleast_1d(np.asarray(self.eta, float))
        # Linear (lam = 0) components are merged into the normal part.
        quad = lam != 0.0
        s2 = float(self.sigma0) ** 2 + float(np.sum(eta[~quad] ** 2))
        object.__setattr__(self, "lam", lam[quad])
        object.__setattr__(self, "eta", eta[quad])
        object.__setattr__(self, "sigma0", math.sqrt(s2))
        object.__setattr__(self, "kappa0", float(self.kappa0))

    # -- basic quantities ---------------------------------------------------------------------
    @property
    def r(self):
        return self.lam.size

    @property
    def is_normal(self):
        return self.r == 0

    @property
    def has_closed_form(self):
        return self.r == 0 or (self.r == 1 and self.sigma0 == 0.0)

    def closed_form_cdf(self, x):
        """Exact CDF (up to ndtr accuracy) when r = 0, or r = 1 without a normal component."""
        if self.r == 0:
            return float(special.ndtr((x - self.kappa0) / self.sigma0))
        l, m = float(self.lam[0]), float(self.eta[0] / (2 * self.lam[0]))
        v = (x - self.kappa) / l                      # Q <= x  <=>  l (W + m)^2 <= x - kappa
        inside = 0.0 if v <= 0 else float(special.ndtr(math.sqrt(v) - m) - special.ndtr(-math.sqrt(v) - m))
        return inside if l > 0 else 1.0 - inside

    @property
    def delta(self):
        """Noncentrality parameters of the chi-square components."""
        return (self.eta / (2 * self.lam)) ** 2

    @property
    def kappa(self):
        """Constant of the classical form kappa + sigma0 G + sum lam_j chi2_1(delta_j)."""
        return self.kappa0 - float(np.sum(self.eta ** 2 / (4 * self.lam)))

    @property
    def mean(self):
        return self.kappa0 + float(np.sum(self.lam))

    @property
    def var(self):
        return self.sigma0 ** 2 + float(np.sum(2 * self.lam ** 2 + self.eta ** 2))

    @property
    def sd(self):
        return math.sqrt(self.var)

    # -- characteristic function ----------------------------------------------------------------
    def log_cf(self, t):
        t = np.asarray(t, float)
        out = 1j * self.kappa0 * t - 0.5 * self.sigma0 ** 2 * t ** 2
        for l, e in zip(self.lam, self.eta):
            z = 1.0 - 2j * l * t
            out = out - 0.5 * np.log(z) - 0.5 * e * e * t * t / z
        return out

    def cf(self, t):
        return np.exp(self.log_cf(t))

    def abs_cf(self, t):
        """Exact modulus |phi(t)|; non-increasing in |t|."""
        t = np.asarray(t, float)
        s = -0.5 * self.sigma0 ** 2 * t ** 2
        for l, e in zip(self.lam, self.eta):
            q = 4.0 * l * l * t * t
            s = s - 0.25 * np.log1p(q) - 0.5 * e * e * t * t / (1.0 + q)
        return np.exp(s)

    # -- moment generating function and Chernoff tail bounds -----------------------------------
    def s_range(self):
        pos, neg = self.lam[self.lam > 0], self.lam[self.lam < 0]
        s_hi = 1.0 / (2 * pos.max()) if pos.size else np.inf
        s_lo = 1.0 / (2 * neg.min()) if neg.size else -np.inf
        return s_lo, s_hi

    def log_mgf(self, s):
        s = np.asarray(s, float)
        out = self.kappa0 * s + 0.5 * self.sigma0 ** 2 * s ** 2
        for l, e in zip(self.lam, self.eta):
            z = 1.0 - 2.0 * l * s
            out = out - 0.5 * np.log(z) + 0.5 * e * e * s * s / z
        return out

    def _chernoff(self, c, upper):
        """log of the Chernoff bound on P(Q >= c) (upper) or P(Q <= c) (lower).
        Any admissible s gives a valid bound, so an imperfect minimisation is still rigorous."""
        s_lo, s_hi = self.s_range()
        scale = 1.0 / max(self.sd, 1e-300)
        if upper:
            if c <= self.mean:
                return 0.0
            top = s_hi * (1 - 1e-12) if np.isfinite(s_hi) else 1e6 * scale
            grid = np.geomspace(top * 1e-12, top, 400)
        else:
            if c >= self.mean:
                return 0.0
            bot = s_lo * (1 - 1e-12) if np.isfinite(s_lo) else -1e6 * scale
            grid = -np.geomspace(-bot * 1e-12, -bot, 400)
        with np.errstate(all="ignore"):
            vals = self.log_mgf(grid) - grid * c
        vals = np.where(np.isfinite(vals), vals, np.inf)
        i = int(np.argmin(vals))
        best = vals[i]
        # local refinement on the bracketing grid points (log M(s) - s c is convex in s)
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
        if a_ != b_:
            f = lambda s: float(self.log_mgf(s) - s * c)
            res = optimize.minimize_scalar(f, bounds=(min(a_, b_), max(a_, b_)), method="bounded",
                                           options={"xatol": 1e-14 * max(abs(a_), abs(b_))})
            if np.isfinite(res.fun):
                best = min(best, res.fun)
        return min(0.0, float(best))

    def tail_upper(self, c):
        """Rigorous upper bound on P(Q >= c)."""
        if self.is_normal:
            return float(special.ndtr(-(c - self.kappa0) / self.sigma0))
        return math.exp(self._chernoff(c, True))

    def tail_lower(self, c):
        """Rigorous upper bound on P(Q <= c)."""
        if self.is_normal:
            return float(special.ndtr((c - self.kappa0) / self.sigma0))
        return math.exp(self._chernoff(c, False))

    # -- tail-of-integral bounds used by the error-bounded inversion -------------------------------
    def _dyadic(self, a):
        return a * 2.0 ** np.arange(N_BLOCKS)

    def _log_c_phi(self):
        """log C_phi with |phi(t)| <= C_phi |t|^(-r/2), C_phi = prod_j (2|lam_j|)^(-1/2)."""
        return -0.5 * float(np.sum(np.log(2 * np.abs(self.lam))))

    def _remainder(self, tau_n, extra_power):
        """Bound on sum_{i>=0} C_phi (tau_n 2^i)^(-r/2-extra_power), the part of a dyadic sum
        beyond the evaluated blocks (geometric series)."""
        expo = 0.5 * self.r + extra_power
        return math.exp(self._log_c_phi() - expo * math.log(tau_n)) / (1.0 - 2.0 ** (-expo))

    def trunc_bound_modulus(self, a):
        """Bound on (1/pi) int_a^inf |phi(t)|/t dt  (Theorem 6(ii), modulus bound): the first
        N_BLOCKS dyadic terms exactly, the rest by |phi(t)| <= C_phi t^(-r/2)."""
        ai = self._dyadic(a)
        rem = self._remainder(a * 2.0 ** N_BLOCKS, 0.0)
        return LN2 / math.pi * (float(np.sum(self.abs_cf(ai))) + rem)

    def tv_bound(self, a):
        """Bound on the total variation of c(t) = phi(t) exp(-i kappa t) / t over [a, inf)."""
        ai = self._dyadic(a)
        lam, eta = self.lam[:, None], self.eta[:, None]
        q = 4 * lam ** 2 * ai[None, :] ** 2
        deriv = np.sum(np.abs(lam) / np.sqrt(1 + q) + eta ** 2 / (4 * np.abs(lam) * (1 + q)), axis=0)
        sup = deriv + 2 * ai * self.sigma0 ** 2 + 1.0 / ai
        # beyond the evaluated blocks: |lam|/sqrt(1+q) <= 1/(2 tau), eta^2/(4|lam|(1+q)) <= eta^2/(16|lam|^3 tau^2)
        tau_n = a * 2.0 ** N_BLOCKS
        c_delta = float(np.sum(self.eta ** 2 / (16 * np.abs(self.lam) ** 3)))
        rem = (1 + 0.5 * self.r) * self._remainder(tau_n, 1.0) + c_delta * self._remainder(tau_n, 2.0)
        if self.sigma0 > 0:
            if self.sigma0 * tau_n < 2:
                return np.inf
            rem += 4 * self.sigma0 ** 2 * math.exp(self._log_c_phi() + (1 - 0.5 * self.r) * math.log(tau_n)
                                                   - 0.5 * (self.sigma0 * tau_n) ** 2)
        return LN2 * (float(np.sum(self.abs_cf(ai) * sup)) + rem)


# ---------------------------------------------------------------------------------------------
# Gaussian quadratic expressions
# ---------------------------------------------------------------------------------------------
def represent(A, a, b, mu, Sigma, rel_tol=1e-13):
    """Distribution of Q = X'AX + a'X + b for X ~ N(mu, Sigma) as a GenChi2 (Lemma 5)."""
    A = np.asarray(A, float)
    a = np.asarray(a, float)
    mu = np.asarray(mu, float)
    L = np.linalg.cholesky(np.asarray(Sigma, float))
    B = L.T @ A @ L
    B = 0.5 * (B + B.T)
    lam, Uv = np.linalg.eigh(B)
    eta = Uv.T @ (L.T @ (2 * A @ mu + a))
    lam = np.where(np.abs(lam) <= rel_tol * max(np.abs(lam).max(), 1e-300), 0.0, lam)
    kappa0 = float(mu @ A @ mu + a @ mu + b)
    return GenChi2(lam, eta, kappa0)


@dataclass
class GaussianModel:
    """Two-class Gaussian model X | Y=y ~ N(mu_y, Sigma_y)."""
    mu0: np.ndarray
    mu1: np.ndarray
    S0: np.ndarray
    S1: np.ndarray

    def __post_init__(self):
        for k in ("mu0", "mu1", "S0", "S1"):
            setattr(self, k, np.asarray(getattr(self, k), float))

    @property
    def p(self):
        return self.mu0.size

    def swapped(self):
        """Model with the two class covariance matrices interchanged (Theorem 3)."""
        return GaussianModel(self.mu0, self.mu1, self.S1, self.S0)


def score_coefficients(mu0, mu1, S0, S1):
    """(A, a, b, K) with log-LR(x) = K + (x'Ax + a'x + b)/2."""
    P0, P1 = np.linalg.inv(S0), np.linalg.inv(S1)
    A = P0 - P1
    A = 0.5 * (A + A.T)
    a = 2 * (P1 @ mu1 - P0 @ mu0)
    b = float(mu0 @ P0 @ mu0 - mu1 @ P1 @ mu1)
    K = 0.5 * (np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])
    return A, a, b, K


def scores(model: GaussianModel):
    """Coefficients of the full (f) and independence-based (nb) quadratic scores."""
    D0, D1 = np.diag(np.diag(model.S0)), np.diag(np.diag(model.S1))
    return {"f": score_coefficients(model.mu0, model.mu1, model.S0, model.S1),
            "nb": score_coefficients(model.mu0, model.mu1, D0, D1)}


def score_distributions(model: GaussianModel):
    """GenChi2 laws of Q_m(X) | Y=y for m in {f, nb}, y in {0, 1}."""
    out = {}
    for m, (A, a, b, _) in scores(model).items():
        out[(m, 0)] = represent(A, a, b, model.mu0, model.S0)
        out[(m, 1)] = represent(A, a, b, model.mu1, model.S1)
    return out


def difference_distribution(model: GaussianModel, m):
    """Law of Q_m(X1) - Q_m(X0) for independent X1 ~ class 1, X0 ~ class 0 (AUC)."""
    A, a, b, _ = scores(model)[m]
    p = model.p
    Z = np.zeros((p, p))
    At = np.block([[A, Z], [Z, -A]])
    at = np.concatenate([a, -a])
    mut = np.concatenate([model.mu1, model.mu0])
    St = np.block([[model.S1, Z], [Z, model.S0]])
    return represent(At, at, 0.0, mut, St)


# ---------------------------------------------------------------------------------------------
# Error-bounded CDF evaluation (Theorem 6)
# ---------------------------------------------------------------------------------------------
class ErrorBoundedCDF:
    """Midpoint-rule Gil-Pelaez inversion of a GenChi2 law with a-priori error bounds.

    For each x the returned bound e(x) is the sum of
      * aliasing:   Chernoff bounds on P(Q >= x + 2 pi/h) and P(Q <= x - 2 pi/h)
      * truncation: min(modulus bound, oscillation (Abel) bound)
      * rounding:   a conservative floating-point allowance.
    """

    def __init__(self, g: GenChi2, window, eps=1e-10, max_terms=2 ** 24, chunk=2 ** 20):
        self.g, self.eps, self.chunk = g, float(eps), int(chunk)
        self.max_terms = int(max_terms)
        x_lo, x_hi = float(min(window)), float(max(window))
        self.window = (x_lo, x_hi)
        if g.has_closed_form:
            self.K_max = 0
            return
        self.L = self._choose_L(x_lo, x_hi, eps / 4)
        self.h = 2 * math.pi / self.L
        need = [self._K_needed(x) for x in (x_lo, x_hi)]
        if x_lo <= g.kappa <= x_hi:
            need.append(self._K_needed(g.kappa))
        self.K_max = int(min(max(need), self.max_terms))
        self._precompute()

    # -- design ---------------------------------------------------------------------------------
    def _choose_L(self, x_lo, x_hi, target):
        g = self.g
        L = max(4 * g.sd, x_hi - x_lo, 1e-12)
        while g.tail_upper(x_lo + L) > target or g.tail_lower(x_hi - L) > target:
            L *= 2.0
        lo, hi = L / 2, L
        for _ in range(30):                       # tighten: smallest admissible L (fewer terms)
            mid = 0.5 * (lo + hi)
            if g.tail_upper(x_lo + mid) <= target and g.tail_lower(x_hi - mid) <= target:
                hi = mid
            else:
                lo = mid
        return hi

    def _trunc_bound(self, K, x):
        g, h = self.g, self.h
        e_mod = g.trunc_bound_modulus((K - 0.5) * h)
        s = abs(math.sin((g.kappa - x) * h / 2))
        e_osc = (h / math.pi) * g.tv_bound((K + 0.5) * h) / s if s > 1e-300 else np.inf
        return min(e_mod, e_osc)

    def _K_needed(self, x):
        target = self.eps / 2
        K = 64
        while self._trunc_bound(K, x) > target:
            K *= 2
            if K > 4 * self.max_terms:
                return K
        lo, hi = K // 2, K
        while hi - lo > max(1, hi // 256):
            mid = (lo + hi) // 2
            if self._trunc_bound(mid, x) <= target:
                hi = mid
            else:
                lo = mid
        return hi

    def _precompute(self):
        K, h = self.K_max, self.h
        self.c = np.empty(K, complex)
        s_abs = s_abs_t = s_logabs = 0.0
        for i0 in range(0, K, self.chunk):
            t = (np.arange(i0, min(i0 + self.chunk, K)) + 0.5) * h
            lc = self.g.log_cf(t)
            ck = np.exp(lc) / t
            self.c[i0:i0 + t.size] = ck
            ak = np.abs(ck)
            s_abs += float(np.sum(ak))
            s_abs_t += float(np.sum(ak * t))
            s_logabs += float(np.sum(ak * (np.abs(lc) + t * abs(self.g.kappa0))))
        self._sum_abs, self._sum_abs_t, self._sum_logabs = s_abs, s_abs_t, s_logabs
        x_lo, x_hi = self.window
        self._alias_window = self.g.tail_upper(x_lo + self.L) + self.g.tail_lower(x_hi - self.L)

    def _round_bound(self, K, x):
        return self.h / math.pi * U * ((8 * math.log2(K + 1) + 16 * (self.g.r + 2)) * self._sum_abs
                                       + 4 * self._sum_logabs + 4 * abs(x) * self._sum_abs_t)

    # -- evaluation -----------------------------------------------------------------------------
    def _one(self, x):
        g = self.g
        if g.has_closed_form:
            return g.closed_form_cdf(x), 4 * NORMAL_CDF_ERR
        h = self.h
        K = min(self._K_needed(x), self.K_max)
        acc = 0.0
        for i0 in range(0, K, self.chunk):
            i1 = min(i0 + self.chunk, K)
            t = (np.arange(i0, i1) + 0.5) * h
            ph = t * x
            ck = self.c[i0:i1]
            acc += float(np.dot(np.cos(ph), ck.imag) - np.dot(np.sin(ph), ck.real))
        F_hat = 0.5 - h / math.pi * acc
        e_alias = g.tail_upper(x + self.L) + g.tail_lower(x - self.L)
        e_trunc = self._trunc_bound(K, x)
        return F_hat, e_alias + e_trunc + self._round_bound(K, x)

    def __call__(self, x):
        """Return (F_hat, error_bound); vectorised over x."""
        x = np.asarray(x, float)
        if x.ndim == 0:
            return self._one(float(x))
        res = np.array([self._one(float(v)) for v in x.ravel()])
        return res[:, 0].reshape(x.shape), res[:, 1].reshape(x.shape)

    def on_grid(self, min_points=2 ** 20):
        """F_hat and error-bounded bounds on a uniform grid covering the window, via one FFT.

        Grid: x_j = x_lo + j * L / M (j < M), restricted to the window. The same K_max terms
        are used at every grid point, so the bound is aliasing(window) + truncation(K_max, x)
        + rounding (with an extra FFT allowance)."""
        g = self.g
        x_lo, x_hi = self.window
        if g.has_closed_form:
            x = np.linspace(x_lo, x_hi, min_points)
            F = np.array([g.closed_form_cdf(v) for v in x]) if g.r else special.ndtr((x - g.kappa0) / g.sigma0)
            return x, F, np.full(x.size, 4 * NORMAL_CDF_ERR)
        K, h = self.K_max, self.h
        want = int(math.ceil(math.log2(min_points * self.L / max(x_hi - x_lo, 1e-300))))
        M = 1 << max(int(math.ceil(math.log2(max(K, 1)))), min(want, 24))
        d = np.zeros(M, complex)
        t = (np.arange(K) + 0.5) * h
        d[:K] = self.c * np.exp(-1j * t * x_lo)
        S = np.fft.fft(d) * np.exp(-1j * np.pi * np.arange(M) / M)
        x = x_lo + np.arange(M) * (self.L / M)
        keep = x <= x_hi
        x, S = x[keep], S[keep]
        F = 0.5 - h / math.pi * S.imag
        s = np.abs(np.sin((g.kappa - x) * h / 2))
        e_mod = g.trunc_bound_modulus((K - 0.5) * h)
        tv = (h / math.pi) * g.tv_bound((K + 0.5) * h)
        e_trunc = np.minimum(e_mod, np.where(s > 1e-300, tv / np.maximum(s, 1e-300), np.inf))
        e_round = np.array([self._round_bound(K, v) for v in (x_lo, x_hi)]).max()
        e_fft = h / math.pi * U * 8 * math.log2(M) * self._sum_abs
        return x, F, self._alias_window + e_trunc + e_round + e_fft


# ---------------------------------------------------------------------------------------------
# Saddlepoint approximation (used for starting values and as a benchmark competitor)
# ---------------------------------------------------------------------------------------------
def _cumulants_at(g: GenChi2, s):
    K = float(g.log_mgf(s))
    k1, k2, k3 = g.kappa0 + g.sigma0 ** 2 * s, g.sigma0 ** 2, 0.0
    for l, e in zip(g.lam, g.eta):
        z = 1 - 2 * l * s
        k1 += l / z + e * e * s * (1 - l * s) / z ** 2
        k2 += 2 * l * l / z ** 2 + e * e / z ** 3
        k3 += 8 * l ** 3 / z ** 3 + 6 * e * e * l / z ** 4
    return K, k1, k2, k3


def saddlepoint_cdf(g: GenChi2, x):
    """Lugannani-Rice saddlepoint approximation to P(Q <= x) (Kuonen 1999)."""
    if g.is_normal:
        return float(special.ndtr((x - g.kappa0) / g.sigma0))
    s_lo, s_hi = g.s_range()
    lo = s_lo * (1 - 1e-10) if np.isfinite(s_lo) else -1e8
    hi = s_hi * (1 - 1e-10) if np.isfinite(s_hi) else 1e8
    f = lambda s: _cumulants_at(g, s)[1] - x
    try:
        s_hat = optimize.brentq(f, lo, hi, xtol=1e-15, maxiter=500)
    except ValueError:
        return float(x > g.mean)
    K, k1, k2, k3 = _cumulants_at(g, s_hat)
    if abs(s_hat) * math.sqrt(k2) < 1e-5:
        _, _, c2, c3 = _cumulants_at(g, 0.0)
        return 0.5 + c3 / (6 * math.sqrt(2 * math.pi) * c2 ** 1.5)
    w = math.copysign(math.sqrt(max(2 * (s_hat * x - K), 0.0)), s_hat)
    u = s_hat * math.sqrt(k2)
    return float(special.ndtr(w) + math.exp(-0.5 * w * w) / math.sqrt(2 * math.pi) * (1 / w - 1 / u))


def saddlepoint_quantile(g: GenChi2, p):
    if g.is_normal:
        return g.kappa0 + g.sigma0 * float(special.ndtri(p))
    lo, hi = g.mean - 8 * g.sd, g.mean + 8 * g.sd
    while saddlepoint_cdf(g, lo) > p:
        lo -= 8 * g.sd
    while saddlepoint_cdf(g, hi) < p:
        hi += 8 * g.sd
    return optimize.brentq(lambda t: saddlepoint_cdf(g, t) - p, lo, hi, xtol=1e-12 * (1 + abs(g.mean)))


# ---------------------------------------------------------------------------------------------
# Error-bounded quantiles, TPRs, PL(alpha) and AUC (Proposition 7)
# ---------------------------------------------------------------------------------------------
@dataclass
class Enclosure:
    lo: float
    hi: float

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def half_width(self):
        return 0.5 * (self.hi - self.lo)

    def __sub__(self, other):
        return Enclosure(self.lo - other.hi, self.hi - other.lo)

    def __repr__(self):
        return f"{self.mid:.10f} +/- {self.half_width:.2e}"


def error_bounded_quantile(g: GenChi2, p, eps=1e-10, evaluator=None, max_terms=2 ** 24, tail=None):
    """Interval (t_lo, t_hi) error-bounded to contain F^{-1}(p), plus the evaluator used. tail = 1 - p,
    if given, is used exactly instead of the rounded p (this matters for small upper-tail levels)."""
    if g.is_normal:
        # normal law: the bracket is verified with the upper-tail probability; ndtr is accurate to
        # NORMAL_CDF_ERR absolutely and, in the tail, to NORMAL_TAIL_RELERR relatively
        a = (1 - p) if tail is None else tail
        t = g.kappa0 - g.sigma0 * float(special.ndtri(a))
        sf = lambda x: float(special.ndtr((g.kappa0 - x) / g.sigma0))
        err = lambda v: min(NORMAL_CDF_ERR, NORMAL_TAIL_RELERR * v)
        d = 1e-13 * (abs(t) + g.sigma0)
        for _ in range(200):
            lo, hi = sf(t - d), sf(t + d)
            if lo - err(lo) > a > hi + err(hi):
                return (t - d, t + d), t, None
            d *= 2
        raise RuntimeError("normal quantile bracketing failed")
    slack = 0.0 if tail is None else 2.0 ** -53      # p = fl(1 - tail) differs from 1 - tail by less
    t0 = saddlepoint_quantile(g, p)
    if evaluator is None:
        w = 0.02 * g.sd
        evaluator = ErrorBoundedCDF(g, (t0 - w, t0 + w), eps=eps, max_terms=max_terms)
    F = lambda x: evaluator(x)[0] - p
    step = 1e-3 * g.sd
    a_, fa = t0, F(t0)
    b_ = a_ + (step if fa < 0 else -step)
    fb = F(b_)
    while fa * fb > 0:
        step *= 2
        a_, fa = b_, fb
        b_ = a_ + (step if fa < 0 else -step)
        fb = F(b_)
    t_star = optimize.brentq(F, min(a_, b_), max(a_, b_), xtol=1e-14 * (1 + abs(t0)), rtol=1e-15)
    eta = 1e-6 * g.sd
    dens = max((evaluator(t_star + eta)[0] - evaluator(t_star - eta)[0]) / (2 * eta), 1e-300)
    e_star = evaluator(t_star)[1]
    d = max(2.0 * e_star / dens, 1e-14 * (1 + abs(t_star)))
    if d > 0.1 * g.sd:
        # At a coarse eps the difference quotient above can be swamped by the error bounds; the
        # bracket is verified below in any case, so d only needs to be a reasonable first guess.
        d = 1e-3 * g.sd
        for eta in g.sd * np.array([1e-4, 1e-3, 1e-2, 1e-1]):
            (F1, e1), (F2, e2) = evaluator(t_star + eta), evaluator(t_star - eta)
            if F1 - F2 > 4 * (e1 + e2):
                d = min(max(2.0 * e_star * 2 * eta / (F1 - F2), 1e-14 * (1 + abs(t_star))), 0.1 * g.sd)
                break
    for _ in range(80):
        Flo, elo = evaluator(t_star - d)
        Fhi, ehi = evaluator(t_star + d)
        if Flo + elo + slack < p < Fhi - ehi - slack:
            return (t_star - d, t_star + d), t_star, evaluator
        d *= 2
    raise RuntimeError("quantile bracketing failed")


def error_bounded_tpr(g0: GenChi2, g1: GenChi2, alpha, eps=1e-10, max_terms=2 ** 24):
    """Error-bounded enclosure of TPR(alpha) = 1 - F_1(F_0^{-1}(1 - alpha))."""
    (t_lo, t_hi), t_star, _ = error_bounded_quantile(g0, 1 - alpha, eps, max_terms=max_terms, tail=alpha)
    if g1.is_normal:
        hi = 1 - special.ndtr((t_lo - g1.kappa0) / g1.sigma0) + NORMAL_CDF_ERR
        lo = 1 - special.ndtr((t_hi - g1.kappa0) / g1.sigma0) - NORMAL_CDF_ERR
        return Enclosure(float(lo), float(hi)), t_star
    ev1 = ErrorBoundedCDF(g1, (t_lo, t_hi), eps=eps, max_terms=max_terms)
    F_lo, e_lo = ev1(t_lo)
    F_hi, e_hi = ev1(t_hi)
    return Enclosure(1 - F_hi - e_hi, 1 - F_lo + e_lo), t_star


def error_bounded_auc(model: GaussianModel, m, eps=1e-10, max_terms=2 ** 24):
    g = difference_distribution(model, m)
    if g.is_normal:
        v = 1 - special.ndtr(-g.kappa0 / g.sigma0)
        return Enclosure(v - NORMAL_CDF_ERR, v + NORMAL_CDF_ERR)
    ev = ErrorBoundedCDF(g, (0.0, 0.0), eps=eps, max_terms=max_terms)
    F, e = ev(0.0)
    return Enclosure(1 - F - e, 1 - F + e)


def performance_loss(model: GaussianModel, alpha, eps=1e-10, with_auc=True, max_terms=2 ** 24):
    """Error-bounded PL(alpha) = TPR_f(alpha) - TPR_nb(alpha) (and AUCs) for a Gaussian model."""
    t0 = time.perf_counter()
    dist = score_distributions(model)
    out = {"alpha": alpha}
    for m in ("f", "nb"):
        enc, t = error_bounded_tpr(dist[(m, 0)], dist[(m, 1)], alpha, eps, max_terms)
        out[f"TPR_{m}"], out[f"t_{m}"] = enc, t
    out["PL"] = out["TPR_f"] - out["TPR_nb"]
    if with_auc:
        for m in ("f", "nb"):
            out[f"AUC_{m}"] = error_bounded_auc(model, m, eps, max_terms)
        out["dAUC"] = out["AUC_f"] - out["AUC_nb"]
    out["seconds"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------------------------
# Closed forms for the homoscedastic case (Proposition 2)
# ---------------------------------------------------------------------------------------------
def homoscedastic_separations(mu0, mu1, S):
    d = np.asarray(mu1, float) - np.asarray(mu0, float)
    S = np.asarray(S, float)
    Dinv = 1.0 / np.diag(S)
    delta_f = math.sqrt(d @ np.linalg.solve(S, d))
    w = Dinv * d
    delta_nb = (d @ w) / math.sqrt(w @ S @ w)
    return delta_f, delta_nb


def homoscedastic_pl(mu0, mu1, S, alpha):
    df, dnb = homoscedastic_separations(mu0, mu1, S)
    z = special.ndtri(1 - np.asarray(alpha, float))
    return special.ndtr(df - z) - special.ndtr(dnb - z)


def homoscedastic_max_pl(mu0, mu1, S):
    df, dnb = homoscedastic_separations(mu0, mu1, S)
    return 2 * special.ndtr((df - dnb) / 2) - 1, float(special.ndtr(-(df + dnb) / 2))


def kantorovich_bound(S, delta_f):
    """Worst-case sup_alpha PL over mean shifts with Mahalanobis separation delta_f."""
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    ev = np.linalg.eigvalsh(R)
    kap = ev[-1] / ev[0]
    return 2 * special.ndtr(delta_f * (math.sqrt(kap) - 1) ** 2 / (2 * (1 + kap))) - 1


# ---------------------------------------------------------------------------------------------
# Monte Carlo reference estimators (validation only)
# ---------------------------------------------------------------------------------------------
def quad_form_values(X, A, a, b):
    return np.einsum("ij,jk,ik->i", X, A, X) + X @ a + b


def mc_performance_loss(model: GaussianModel, alpha, N, B, rng, with_auc=False):
    """B independent Monte Carlo replicates of the plug-in estimator of PL(alpha) (N per class)."""
    sc = scores(model)
    L0, L1 = np.linalg.cholesky(model.S0), np.linalg.cholesky(model.S1)
    pl, auc_f, auc_nb = np.empty(B), np.empty(B), np.empty(B)
    for i in range(B):
        X0 = model.mu0 + rng.standard_normal((N, model.p)) @ L0.T
        X1 = model.mu1 + rng.standard_normal((N, model.p)) @ L1.T
        tpr = {}
        for m, (A, a, b, _) in sc.items():
            s0, s1 = quad_form_values(X0, A, a, b), quad_form_values(X1, A, a, b)
            thr = np.quantile(s0, 1 - alpha, method="inverted_cdf")
            tpr[m] = float(np.mean(s1 > thr))
            if with_auc:
                from scipy.stats import rankdata
                rk = rankdata(np.concatenate([s1, s0]))
                v = (rk[:N].sum() - N * (N + 1) / 2) / (N * N)
                (auc_f if m == "f" else auc_nb)[i] = v
        pl[i] = tpr["f"] - tpr["nb"]
    if with_auc:
        return pl, auc_f, auc_nb
    return pl


def mc_cdf(g: GenChi2, x, n, rng):
    W = rng.standard_normal((n, g.r))
    Q = g.kappa0 + W ** 2 @ g.lam + W @ g.eta + g.sigma0 * rng.standard_normal(n)
    return np.mean(Q[:, None] <= np.atleast_1d(x)[None, :], axis=0)


# ---------------------------------------------------------------------------------------------
# Error-bounded loss curves alpha -> PL(alpha) via monotone grid enclosures
# ---------------------------------------------------------------------------------------------
def _grid_quantile_enclosure(x, F, e, p):
    """[t_lo, t_hi] containing F^{-1}(p), from error-bounded grid values and monotonicity of F."""
    below = np.nonzero(F + e < p)[0]
    above = np.nonzero(F - e > p)[0]
    if below.size == 0 or above.size == 0:
        return np.nan, np.nan
    return x[below[-1]], x[above[above > below[-1]][0]] if np.any(above > below[-1]) else np.nan


def _grid_tpr_enclosure(x1, F1, e1, t_lo, t_hi):
    """Enclosure of 1 - F_1(t) for t in [t_lo, t_hi], from an error-bounded grid for F_1."""
    i_up = np.searchsorted(x1, t_hi, side="left")          # first grid point >= t_hi
    i_dn = np.searchsorted(x1, t_lo, side="right") - 1      # last grid point <= t_lo
    if i_up >= x1.size or i_dn < 0:
        return Enclosure(np.nan, np.nan)
    return Enclosure(1 - (F1[i_up] + e1[i_up]), 1 - (F1[i_dn] - e1[i_dn]))


def pl_curve(model: GaussianModel, alphas, eps=1e-9, max_terms=2 ** 24, min_points=2 ** 21):
    """Error-bounded enclosures of TPR_f, TPR_nb and PL over a vector of alphas."""
    alphas = np.asarray(alphas, float)
    dist = score_distributions(model)
    out = {"alpha": alphas}
    for m in ("f", "nb"):
        g0, g1 = dist[(m, 0)], dist[(m, 1)]
        q = [saddlepoint_quantile(g0, 1 - a) for a in (alphas.min(), alphas.max())]
        pad = 0.1 * (max(q) - min(q)) + 0.25 * g0.sd
        window = (min(q) - pad, max(q) + pad)
        x0, F0, e0 = ErrorBoundedCDF(g0, window, eps, max_terms).on_grid(min_points)
        x1, F1, e1 = ErrorBoundedCDF(g1, window, eps, max_terms).on_grid(min_points)
        encs = []
        for a in alphas:
            t_lo, t_hi = _grid_quantile_enclosure(x0, F0, e0, 1 - a)
            encs.append(_grid_tpr_enclosure(x1, F1, e1, t_lo, t_hi))
        out[f"TPR_{m}"] = encs
    out["PL"] = [f - n for f, n in zip(out["TPR_f"], out["TPR_nb"])]
    return out


# ---------------------------------------------------------------------------------------------
# Estimating PL(alpha) from data: delta method and parametric bootstrap
# ---------------------------------------------------------------------------------------------
def fit_gaussian(X0, X1):
    """Gaussian class models fitted by sample means and unbiased sample covariance matrices."""
    X0, X1 = np.atleast_2d(np.asarray(X0, float)), np.atleast_2d(np.asarray(X1, float))
    return GaussianModel(X0.mean(0), X1.mean(0), np.atleast_2d(np.cov(X0.T)), np.atleast_2d(np.cov(X1.T)))


def model_parameters(model: GaussianModel):
    """theta = (mu0, mu1, vech S0, vech S1), vech taken over the lower triangle."""
    i, j = np.tril_indices(model.p)
    return np.concatenate([model.mu0, model.mu1, model.S0[i, j], model.S1[i, j]])


def model_from_parameters(theta, p):
    """Inverse of model_parameters."""
    i, j = np.tril_indices(p)
    q = i.size
    S = []
    for k in range(2):
        M = np.zeros((p, p))
        M[i, j] = M[j, i] = theta[2 * p + k * q: 2 * p + (k + 1) * q]
        S.append(M)
    return GaussianModel(theta[:p], theta[p:2 * p], S[0], S[1])


def parameter_covariance(model: GaussianModel, n0, n1):
    """Asymptotic covariance matrix of the estimated theta from n0 and n1 independent observations:
    Sigma_y / n_y for the means and 2 D+ (Sigma_y x Sigma_y) D+' / n_y for vech Sigma_y."""
    p = model.p
    i, j = np.tril_indices(p)
    D = np.zeros((p * p, i.size))
    for k, (a, b) in enumerate(zip(i, j)):
        D[a * p + b, k] = D[b * p + a, k] = 1.0
    Dp = np.linalg.pinv(D)
    blocks = [model.S0 / n0, model.S1 / n1, 2.0 / n0 * Dp @ np.kron(model.S0, model.S0) @ Dp.T,
              2.0 / n1 * Dp @ np.kron(model.S1, model.S1) @ Dp.T]
    size = sum(bl.shape[0] for bl in blocks)
    V, k = np.zeros((size, size)), 0
    for bl in blocks:
        V[k:k + bl.shape[0], k:k + bl.shape[0]] = bl
        k += bl.shape[0]
    return V


def pl_gradient(model: GaussianModel, alpha, eps=1e-8, rel_step=2e-3):
    """Central-difference gradient of PL(alpha) with respect to theta. Because every PL value is an
    enclosure, the rounding and inversion error of each difference quotient is bounded by
    (half-width up + half-width down) / (2 h); the O(h^2) truncation error is not included.
    Steps for covariance entries are scaled by the smallest eigenvalue of the class covariance, which
    keeps the perturbed matrices positive definite and the steps small when they are near-singular."""
    theta, p = model_parameters(model), model.p
    i, j = np.tril_indices(p)
    sd0, sd1 = np.sqrt(np.diag(model.S0)), np.sqrt(np.diag(model.S1))
    l0, l1 = np.linalg.eigvalsh(model.S0)[0], np.linalg.eigvalsh(model.S1)[0]
    h = rel_step * np.concatenate([sd0, sd1, np.minimum(sd0[i] * sd0[j], l0), np.minimum(sd1[i] * sd1[j], l1)])
    grad, err = np.empty(theta.size), np.empty(theta.size)
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = h[k]
        up = performance_loss(model_from_parameters(theta + step, p), alpha, eps, with_auc=False)["PL"]
        dn = performance_loss(model_from_parameters(theta - step, p), alpha, eps, with_auc=False)["PL"]
        grad[k] = (up.mid - dn.mid) / (2 * h[k])
        err[k] = (up.half_width + dn.half_width) / (2 * h[k])
    return grad, err


def pl_hessian(model: GaussianModel, alpha, eps=1e-10, rel_step=0.05):
    """Hessian of PL(alpha) with respect to theta by central second differences (2q^2 evaluations),
    with steps scaled as in pl_gradient. Used at a zero of the loss, where it gives the 1/n bias
    tr(H V)/2 of the plug-in estimate (Proposition 8(b))."""
    theta, p = model_parameters(model), model.p
    i, j = np.tril_indices(p)
    sd0, sd1 = np.sqrt(np.diag(model.S0)), np.sqrt(np.diag(model.S1))
    l0, l1 = np.linalg.eigvalsh(model.S0)[0], np.linalg.eigvalsh(model.S1)[0]
    h = rel_step * np.concatenate([sd0, sd1, np.minimum(sd0[i] * sd0[j], l0), np.minimum(sd1[i] * sd1[j], l1)])
    q = theta.size
    f = lambda x: performance_loss(model_from_parameters(theta + x, p), alpha, eps, with_auc=False)["PL"].mid
    E = np.diag(h)
    f0 = f(np.zeros(q))
    H = np.empty((q, q))
    for a in range(q):
        H[a, a] = (f(E[a]) - 2 * f0 + f(-E[a])) / h[a] ** 2
        for b in range(a):
            H[a, b] = H[b, a] = (f(E[a] + E[b]) - f(E[a] - E[b]) - f(-E[a] + E[b]) + f(-E[a] - E[b])) / (4 * h[a] * h[b])
    return H


def pl_delta_interval(model: GaussianModel, n0, n1, alpha, level=0.95, eps=1e-8, rel_step=2e-3,
                      se_rtol=None, eps_min=1e-12):
    """Plug-in estimate of PL(alpha) with a delta-method standard error and Wald interval.

    se_err bounds the change in se caused by the numerical error of the gradient: se is the norm
    |V^(1/2) g|, so by the triangle inequality the change is at most (e'|V| e)^(1/2), where e bounds
    the componentwise error of the difference quotients (pl_gradient). With se_rtol, eps is divided
    by 100 until se_err <= se_rtol * se or eps would fall below eps_min; the eps used is returned."""
    V = parameter_covariance(model, n0, n1)
    while True:
        grad, err = pl_gradient(model, alpha, eps, rel_step)
        se = float(np.sqrt(max(grad @ V @ grad, 0.0)))
        se_err = float(np.sqrt(err @ np.abs(V) @ err))
        if se_rtol is None or se_err <= se_rtol * se or eps / 100 < eps_min:
            break
        eps /= 100
    est = performance_loss(model, alpha, eps, with_auc=False)["PL"]
    z = float(special.ndtri(0.5 + level / 2))
    return dict(PL=est.mid, PL_hw=est.half_width, se=se, lo=est.mid - z * se, hi=est.mid + z * se,
                grad_err=float(np.max(err)), se_err=se_err, eps=eps)


def pl_bootstrap(model: GaussianModel, n0, n1, alpha, B, rng, level=0.95, eps=1e-6, est=None):
    """Parametric bootstrap of the plug-in estimator: B data sets of sizes n0, n1 are drawn from the
    fitted model, the model is refitted and PL(alpha) recomputed. Returns the percentile and basic
    (bias-reflecting) intervals, the bias-corrected estimate 2 PL_hat - mean(PL*) and the largest
    half-width of the PL* enclosures. est is the plug-in estimate PL_hat if already computed."""
    L0, L1 = np.linalg.cholesky(model.S0), np.linalg.cholesky(model.S1)
    if est is None:
        est = performance_loss(model, alpha, eps, with_auc=False)["PL"].mid
    stars, hw = np.empty(B), np.empty(B)
    for b in range(B):
        X0 = model.mu0 + rng.standard_normal((n0, model.p)) @ L0.T
        X1 = model.mu1 + rng.standard_normal((n1, model.p)) @ L1.T
        enc = performance_loss(fit_gaussian(X0, X1), alpha, eps, with_auc=False)["PL"]
        stars[b], hw[b] = enc.mid, enc.half_width
    a = (1 - level) / 2
    q_lo, q_hi = np.quantile(stars, [a, 1 - a])
    bias = float(stars.mean() - est)
    return dict(PL=est, bias=bias, PL_bc=est - bias, sd=float(stars.std(ddof=1)), lo_pct=float(q_lo),
                hi_pct=float(q_hi), lo_basic=float(2 * est - q_hi), hi_basic=float(2 * est - q_lo), hw_max=float(hw.max()),
                stars=stars)


# =============================================================================================
# Part 2. Covariance designs (Sections 5 and 6)
# =============================================================================================
# The bivariate design of Section 5.1 (also the setting of the independent check in Part 3)
MU0, MU1 = np.array([1.0, 2.0]), np.array([2.0, 0.5])
BIV_LEVELS = {"N": 0.0, "L": 0.25, "M": 0.50, "H": 0.75, "P": 0.99}
BIV_LABELS = list(BIV_LEVELS)


def biv_cov(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


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
    return GaussianModel(np.zeros(p), sep * unit_direction(p, seed), S0, S1)


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
        m = GaussianModel(np.zeros(p), sep * u * sd0, S0, S1)
        auc = error_bounded_auc(m, "f", eps=1e-6).mid
        if auc_range[0] <= auc <= auc_range[1]:
            return m, dict(nu0=nu0, nu1=nu1, w=w, sep=sep)
    raise RuntimeError("no admissible model found")


# =============================================================================================
# Part 3. Independent check: one-dimensional conditioning integral (bivariate design)
# =============================================================================================
def cond_coef(S0, S1):
    P0, P1 = np.linalg.inv(S0), np.linalg.inv(S1)
    return P0 - P1, 2 * (P1 @ MU1 - P0 @ MU0), float(MU0 @ P0 @ MU0 - MU1 @ P1 @ MU1)


def cond_tail(A, a, b, mu, S, t):
    """P(X'AX + a'X + b > t) for X ~ N_2(mu, S)."""
    s1 = math.sqrt(S[0, 0])
    beta = S[0, 1] / S[0, 0]
    s2c = math.sqrt(S[1, 1] - S[0, 1] ** 2 / S[0, 0])

    def cond(x1):
        m2 = mu[1] + beta * (x1 - mu[0])
        c2, c1 = A[1, 1], 2 * A[0, 1] * x1 + a[1]
        c0 = A[0, 0] * x1 * x1 + a[0] * x1 + b - t
        Phi = lambda z: special.ndtr((z - m2) / s2c)
        if abs(c2) < 1e-14:
            if abs(c1) < 1e-300:
                return float(c0 > 0)
            root = -c0 / c1
            return 1 - Phi(root) if c1 > 0 else Phi(root)
        disc = c1 * c1 - 4 * c2 * c0
        if disc <= 0:
            return 1.0 if c2 > 0 else 0.0
        r1, r2 = sorted([(-c1 - math.sqrt(disc)) / (2 * c2), (-c1 + math.sqrt(disc)) / (2 * c2)])
        inside = Phi(r2) - Phi(r1)
        return 1 - inside if c2 > 0 else inside

    f = lambda z: cond(mu[0] + s1 * z) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    pts = np.linspace(-12, 12, 97)
    return sum(integrate.quad(f, lo, hi, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
               for lo, hi in zip(pts[:-1], pts[1:]))


def cond_cdf(A, a, b, mu, S, t):
    return 1.0 - cond_tail(A, a, b, mu, S, t)


def cond_tpr(A, a, b, S0, S1, alpha):
    g = lambda t: cond_tail(A, a, b, MU0, S0, t) - alpha
    lo, hi = -50.0, 50.0
    while g(lo) < 0:
        lo *= 2
    while g(hi) > 0:
        hi *= 2
    t = optimize.brentq(g, lo, hi, xtol=1e-13, rtol=1e-14)
    return cond_tail(A, a, b, MU1, S1, t)


def bivariate_independent_check():
    """Table 1 recomputed by the conditioning integral (results/bivariate_independent.csv)."""
    rows = []
    for (l0, l1), alpha in product(product(BIV_LEVELS, BIV_LEVELS), (0.05, 0.01)):
        S0, S1 = biv_cov(BIV_LEVELS[l0]), biv_cov(BIV_LEVELS[l1])
        f = cond_tpr(*cond_coef(S0, S1), S0, S1, alpha)
        nb = cond_tpr(*cond_coef(np.eye(2), np.eye(2)), S0, S1, alpha)
        rows.append(dict(rho0=l0, rho1=l1, alpha=alpha, TPR_f=f, TPR_nb=nb, PL=f - nb))
        print(f"({l0},{l1}) alpha={alpha}: PL={f - nb:.10f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bivariate_independent.csv", index=False)
    table = OUT / "bivariate_table.csv"
    if table.exists():
        t = pd.read_csv(table).merge(df, on=["rho0", "rho1", "alpha"], suffixes=("", "_indep"))
        inside = ((t.PL_lo <= t.PL_indep) & (t.PL_indep <= t.PL_hi)).sum()
        print(f"independent check: {inside} of {len(t)} values inside the enclosures, "
              f"largest difference {np.abs(t.PL - t.PL_indep).max():.1e}", flush=True)


# =============================================================================================
# Part 4. Exactness tests
# =============================================================================================
def _check_cdf(name, g, xs, truth, eps=1e-10):
    t0 = time.perf_counter()
    ev = ErrorBoundedCDF(g, (min(xs), max(xs)), eps=eps)
    worst_ratio, worst_err, worst_bound = 0.0, 0.0, 0.0
    for x, F in zip(xs, truth):
        Fh, e = ev(x)
        err = abs(Fh - F)
        worst_err, worst_bound = max(worst_err, err), max(worst_bound, e)
        worst_ratio = max(worst_ratio, err / e)
        assert err <= e, (name, x, Fh, F, err, e)
    print(f"{name:38s} K_max={ev.K_max:>9d}  max|err|={worst_err:.1e}  max bound={worst_bound:.1e}  "
          f"max err/bound={worst_ratio:.2f}  [{time.perf_counter()-t0:.1f}s]")


def _prodnorm_cdf(y):
    """Distribution function of the product of two independent standard normals (K0 density)."""
    return 0.5 + np.sign(y) * special.iti0k0(np.abs(y))[1] / np.pi


def run_tests():
    """Error-bounded distribution functions against laws with known distribution functions."""
    # 1. central chi-square with one degree of freedom
    xp = np.array([0.01, 0.1, 0.5, 1, 2, 4, 8, 15])
    _check_cdf("chi2_1", GenChi2([1.0], [0.0], 0.0), xp, stats.chi2.cdf(xp, 1))
    # 2. noncentral: (W+1)^2 - 1 = W^2 + 2W
    xv = np.array([-0.9, -0.5, 0, 1, 3, 8, 20])
    _check_cdf("noncentral chi2_1 (r=1)", GenChi2([1.0], [2.0], 0.0), xv,
               special.ndtr(np.sqrt(xv + 1) - 1) - special.ndtr(-np.sqrt(xv + 1) - 1))
    # 3. indefinite, r = 2: Z1^2 - Z2^2 = 2 Z1' Z2' (product-normal law)
    xq = np.array([-8, -3, -1, -0.3, 0.2, 0.7, 2, 5, 9.0])
    _check_cdf("indefinite Z1^2 - Z2^2 (r=2)", GenChi2([1.0, -1.0], [0.0, 0.0], 0.0), xq, _prodnorm_cdf(xq / 2))
    # 4. scaled version with very unequal weights
    xq2 = 3 + 99 * np.array([-6, -1, -0.2, 0.1, 0.5, 3.0])
    _check_cdf("indefinite 99(Z1^2 - Z2^2)+3", GenChi2([99.0, -99.0], [0.0, 0.0], 3.0), xq2,
               _prodnorm_cdf((xq2 - 3) / 99 / 2))
    # 5. normal component only, through eta with lambda = 0
    xn = np.linspace(-5, 7, 9)
    _check_cdf("pure normal", GenChi2([0.0], [2.0], 1.0, 0.5), xn, special.ndtr((xn - 1) / math.sqrt(4.25)))
    # 6. exponential: 0.5 (Z1^2 + Z2^2) ~ Exp(1)
    xe = np.array([0.01, 0.2, 1, 3, 7, 15])
    _check_cdf("Exp(1) = (Z1^2+Z2^2)/2", GenChi2([0.5, 0.5], [0.0, 0.0], 0.0), xe, 1 - np.exp(-xe))
    # 7. noncentral chi-square with 5 degrees of freedom: (W1+1.3)^2 + W2^2 + ... + W5^2
    xc = np.array([0.5, 2, 5, 9, 20, 35])
    _check_cdf("ncx2(df=5, nc=1.69)", GenChi2([1.0] * 5, [2 * 1.3, 0, 0, 0, 0], 1.3 ** 2), xc,
               stats.ncx2.cdf(xc, 5, 1.69))
    print("ALL TESTS PASSED", flush=True)


# =============================================================================================
# Part 5. The bivariate design (Section 5.1)
# =============================================================================================
def bivariate_model(l0, l1):
    return GaussianModel(MU0, MU1, biv_cov(BIV_LEVELS[l0]), biv_cov(BIV_LEVELS[l1]))


def bivariate_error_bounded_table(eps=1e-10):
    rows = []
    for l0, l1 in product(BIV_LEVELS, BIV_LEVELS):
        m = bivariate_model(l0, l1)
        for alpha in (0.05, 0.01):
            o = performance_loss(m, alpha, eps=eps, with_auc=(alpha == 0.05))
            dist = score_distributions(m)
            K_f = scores(m)["f"][3]
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


def bivariate_homoscedastic_check(table):
    rows = []
    for l in BIV_LEVELS:
        S = biv_cov(BIV_LEVELS[l])
        df_, dnb = homoscedastic_separations(MU0, MU1, S)
        mx, a_star = homoscedastic_max_pl(MU0, MU1, S)
        for alpha in (0.05, 0.01):
            closed = float(homoscedastic_pl(MU0, MU1, S, alpha))
            ref = table.query("rho0 == @l and rho1 == @l and alpha == @alpha").iloc[0]
            rows.append(dict(level=l, rho=BIV_LEVELS[l], alpha=alpha, Delta_f=df_, Delta_nb=dnb,
                             PL_closed=closed, PL_numeric=ref.PL, abs_diff=abs(closed - ref.PL),
                             PL_hw=ref.PL_hw, max_PL=mx, alpha_star=a_star,
                             kantorovich_bound=kantorovich_bound(S, df_)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bivariate_homoscedastic.csv", index=False)
    print(df.to_string(), flush=True)


def bivariate_monte_carlo(table, B=200, seed=20261013):
    rng = np.random.default_rng(seed)
    rows = []
    for (l0, l1), N in product(product(BIV_LEVELS, BIV_LEVELS), (1000, 10000)):
        t0 = time.perf_counter()
        pl, af, anb = mc_performance_loss(bivariate_model(l0, l1), 0.05, N, B, rng, with_auc=True)
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


def bivariate_pl_curves():
    alphas = np.geomspace(1e-4, 0.5, 60)
    rows = []
    for l0, l1 in product(BIV_LEVELS, BIV_LEVELS):
        t0 = time.perf_counter()
        c = pl_curve(bivariate_model(l0, l1), alphas, eps=1e-9)
        for a, f, nb, pl in zip(alphas, c["TPR_f"], c["TPR_nb"], c["PL"]):
            rows.append(dict(rho0=l0, rho1=l1, alpha=a, TPR_f=f.mid, TPR_nb=nb.mid, PL=pl.mid,
                             PL_hw=pl.half_width))
        print(f"curve ({l0},{l1}) [{time.perf_counter()-t0:.1f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bivariate_curves.csv", index=False)


def run_bivariate():
    """Section 5.1: Table 1, the shared-covariance check, loss curves, Monte Carlo, independent check."""
    tab = bivariate_error_bounded_table()
    bivariate_homoscedastic_check(tab)
    bivariate_pl_curves()
    bivariate_monte_carlo(tab)
    bivariate_independent_check()


# =============================================================================================
# Part 5b. Explicit formulas: cumulant approximations and proportional covariance matrices
# (Online Resource 1, Sections S5 and S6)
# =============================================================================================
def genchi2_cumulants(g: GenChi2, rmax=4):
    """Cumulants kappa_1..kappa_rmax of Q = kappa0 + sum_j (lam_j W_j^2 + eta_j W_j) + sigma0 G."""
    out = []
    for r in range(1, rmax + 1):
        v = float(np.sum(2.0 ** (r - 1) * math.factorial(r - 1) * g.lam ** r))
        if r >= 2:
            v += float(np.sum(math.factorial(r) * g.eta ** 2 / 2 * (2 * g.lam) ** (r - 2)))
        v += g.kappa0 if r == 1 else 0.0
        v += g.sigma0 ** 2 if r == 2 else 0.0
        out.append(v)
    return out


def explicit_tpr(g0: GenChi2, g1: GenChi2, alpha, method):
    """Explicit approximation of TPR(alpha) = P_1{Q > F_0^{-1}(1 - alpha)} from cumulants: 'normal'
    (two moments) or 'cf' (Cornish-Fisher quantile in class 0, Edgeworth tail in class 1, four
    cumulants; Konishi, Niki and Gupta 1988)."""
    k0, k1 = genchi2_cumulants(g0), genchi2_cumulants(g1)
    z = float(special.ndtri(1 - alpha))
    s0, s1 = math.sqrt(k0[1]), math.sqrt(k1[1])
    if method == "normal":
        return float(special.ndtr(-(k0[0] + s0 * z - k1[0]) / s1))
    a3, a4 = k0[2] / s0 ** 3, k0[3] / s0 ** 4
    t = k0[0] + s0 * (z + (z ** 2 - 1) * a3 / 6 + (z ** 3 - 3 * z) * a4 / 24 - (2 * z ** 3 - 5 * z) * a3 ** 2 / 36)
    u, b3, b4 = (t - k1[0]) / s1, k1[2] / s1 ** 3, k1[3] / s1 ** 4
    cdf = special.ndtr(u) - stats.norm.pdf(u) * (b3 / 6 * (u ** 2 - 1) + b4 / 24 * (u ** 3 - 3 * u)
                                                 + b3 ** 2 / 72 * (u ** 5 - 10 * u ** 3 + 15 * u))
    return float(1 - cdf)


def explicit_cumulant_check():
    """Section S5: explicit cumulant approximations of PL(alpha) against the error-bounded values of
    the bivariate design (results/explicit_approx.csv)."""
    tab = pd.read_csv(OUT / "bivariate_table.csv")
    rows = []
    for _, r in tab.iterrows():
        m = GaussianModel(MU0, MU1, biv_cov(BIV_LEVELS[r.rho0]), biv_cov(BIV_LEVELS[r.rho1]))
        d = score_distributions(m)
        row = dict(rho0=r.rho0, rho1=r.rho1, alpha=r.alpha, PL=r.PL, PL_hw=r.PL_hw)
        for method in ("normal", "cf"):
            row[f"PL_{method}"] = (explicit_tpr(d[("f", 0)], d[("f", 1)], r.alpha, method)
                                   - explicit_tpr(d[("nb", 0)], d[("nb", 1)], r.alpha, method))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "explicit_approx.csv", index=False)
    return df


def proportional_tpr_full(mu0, mu1, S0, k, alpha):
    """Corollary S1: TPR of the full rule when Sigma_1 = k Sigma_0, k != 1, from noncentral chi-square
    distribution functions with p degrees of freedom and noncentralities D2/(k-1)^2 and k D2/(k-1)^2."""
    p, d = len(mu0), np.asarray(mu1, float) - np.asarray(mu0, float)
    D2 = float(d @ np.linalg.solve(S0, d))
    l0, l1 = D2 / (k - 1) ** 2, k * D2 / (k - 1) ** 2
    F = lambda q, lam: stats.ncx2.cdf(q, p, lam) if lam > 0 else stats.chi2.cdf(q, p)
    Finv = lambda u, lam: stats.ncx2.ppf(u, p, lam) if lam > 0 else stats.chi2.ppf(u, p)
    if k > 1:
        return float(1 - F(Finv(1 - alpha, l0) / k, l1))
    return float(F(Finv(alpha, l0) / k, l1))


PROPORTIONAL_CASES = [(0.5, (1.0, 2.0), 2.0), (0.5, (1.0, 2.0), 0.5), (-0.8, (1.0, 1.0), 3.0), (0.3, (2.0, 0.5), 1.5)]


def proportional_check(eps=1e-11):
    """Section S6: the explicit true positive rate of Corollary S1 against error-bounded enclosures
    (results/proportional.csv)."""
    rows = []
    for rho, var, k in PROPORTIONAL_CASES:
        s = np.sqrt(var)
        S0 = np.array([[var[0], rho * s[0] * s[1]], [rho * s[0] * s[1], var[1]]])
        for label, (m0, m1) in (("different", (MU0, MU1)), ("equal", (MU0, MU0))):
            m = GaussianModel(m0, m1, S0, k * S0)
            for a in (0.05, 0.01, 1e-4):
                out = performance_loss(m, a, eps, with_auc=False)
                ex = proportional_tpr_full(m0, m1, S0, k, a)
                enc = out["TPR_f"]
                rows.append(dict(rho=rho, var1=var[0], var2=var[1], k=k, means=label, alpha=a, TPR_f_explicit=ex,
                                 TPR_f_lo=enc.lo, TPR_f_hi=enc.hi, inside=bool(enc.lo <= ex <= enc.hi),
                                 PL=out["PL"].mid, PL_hw=out["PL"].half_width))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "proportional.csv", index=False)
    return df


def run_explicit():
    """Online Resource 1, Sections S5 and S6 (needs results/bivariate_table.csv)."""
    explicit_cumulant_check()
    proportional_check()


# =============================================================================================
# Part 6. Beyond two dimensions (Section 5.2)
# =============================================================================================
AR1_RHOS = (0.0, 0.3, 0.6, 0.9)


def highdim_summarise(m, eps=1e-10):
    o5 = performance_loss(m, 0.05, eps=eps, with_auc=True)
    o1 = performance_loss(m, 0.01, eps=eps, with_auc=False)
    return dict(PL05=o5["PL"].mid, PL05_hw=o5["PL"].half_width, PL01=o1["PL"].mid,
                PL01_hw=o1["PL"].half_width, TPR_f05=o5["TPR_f"].mid, TPR_nb05=o5["TPR_nb"].mid,
                TPR_f01=o1["TPR_f"].mid, TPR_nb01=o1["TPR_nb"].mid, AUC_f=o5["AUC_f"].mid,
                AUC_nb=o5["AUC_nb"].mid, dAUC=o5["dAUC"].mid, dAUC_hw=o5["dAUC"].half_width,
                seconds=o5["seconds"] + o1["seconds"])


def structured_job(args):
    p, r0, r1, vr = args
    m = structured_model(p, r0, r1, var_ratio=vr)
    return dict(p=p, rho0=r0, rho1=r1, var_ratio=vr, **highdim_summarise(m))


def random_job(args):
    p, seed = args
    rng = np.random.default_rng(seed)
    m, info = random_model(p, rng)
    dist = score_distributions(m)
    return dict(p=p, seed=seed, **info, r_nb=dist[("nb", 0)].r, **highdim_summarise(m))


def mc_job(args):
    p, seed = args
    m, _ = random_model(p, np.random.default_rng(seed))
    ref = performance_loss(m, 0.05, eps=1e-10, with_auc=False)["PL"]
    pl = mc_performance_loss(m, 0.05, 10000, 100, np.random.default_rng(seed + 1))
    se = pl.std(ddof=1) / np.sqrt(pl.size)
    return dict(p=p, seed=seed, PL_ref=ref.mid, PL_ref_hw=ref.half_width, PL_mc=pl.mean(),
                PL_se=se, z=(pl.mean() - ref.mid) / se)


def timing_job(args):
    p, seed, eps = args
    m, _ = random_model(p, np.random.default_rng(seed))
    t0 = time.perf_counter()
    o = performance_loss(m, 0.05, eps=eps, with_auc=False)
    return dict(p=p, seed=seed, eps=eps, seconds=time.perf_counter() - t0, PL_hw=o["PL"].half_width)


def _pool_run(pool, job, tasks, name):
    t0 = time.perf_counter()
    df = pd.DataFrame(pool.map(job, tasks, chunksize=1))
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df)} rows in {time.perf_counter()-t0:.0f}s", flush=True)
    return df


def highdim_mc_followup():
    """The two most discrepant Monte Carlo checks of Section 5.2, repeated with new random numbers
    and with both rules evaluated at their exact thresholds (results/highdim_mc_followup.csv)."""
    rows = []
    for p, seed in ((10, 10008), (20, 20009)):
        m, _ = random_model(p, np.random.default_rng(seed))
        o = performance_loss(m, 0.05, 1e-10, with_auc=False)
        ref = o["PL"].mid
        for N, B in ((10 ** 4, 400), (10 ** 5, 100)):
            pl = mc_performance_loss(m, 0.05, N, B, np.random.default_rng(seed + 7))
            se = pl.std(ddof=1) / np.sqrt(B)
            rows.append(dict(p=p, seed=seed, check=f"plug-in, N={N}, B={B}", PL_ref=ref, PL_mc=pl.mean(),
                             se=se, z=(pl.mean() - ref) / se))
        sc = scores(m)
        L0, L1 = np.linalg.cholesky(m.S0), np.linalg.cholesky(m.S1)
        rng = np.random.default_rng(99)
        cnt = {"f0": 0, "f1": 0, "nb0": 0, "nb1": 0}
        diff, n = [], 0
        for _ in range(40):                                   # 4e6 draws per class in batches
            X0 = m.mu0 + rng.standard_normal((100000, p)) @ L0.T
            X1 = m.mu1 + rng.standard_normal((100000, p)) @ L1.T
            hit = {}
            for k, (A, a, b, _) in sc.items():
                t = o["t_" + k]
                hit[k + "0"] = quad_form_values(X0, A, a, b) > t
                hit[k + "1"] = quad_form_values(X1, A, a, b) > t
            for key in cnt:
                cnt[key] += int(hit[key].sum())
            diff.append(hit["f1"].astype(float) - hit["nb1"])
            n += 100000
        diff = np.concatenate(diff)
        se = diff.std() / np.sqrt(n)
        rows.append(dict(p=p, seed=seed, check="exact thresholds, 4e6 draws per class", PL_ref=ref,
                         PL_mc=diff.mean(), se=se, z=(diff.mean() - ref) / se,
                         FPR_f=cnt["f0"] / n, FPR_nb=cnt["nb0"] / n))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "highdim_mc_followup.csv", index=False)


def run_highdim(workers=6):
    """Section 5.2: AR(1) and random designs, Monte Carlo checks and run times."""
    from multiprocessing import Pool
    _single_threaded_workers()
    with Pool(workers) as pool:
        _pool_run(pool, structured_job,
                  [(p, r0, r1, vr) for p in (2, 5, 10, 20, 50) for r0, r1 in product(AR1_RHOS, AR1_RHOS)
                   for vr in (1.0, 2.0)], "highdim_structured.csv")
        _pool_run(pool, random_job, [(p, 1000 * p + i) for p in (5, 10, 20, 50) for i in range(100)],
                  "highdim_random.csv")
        _pool_run(pool, mc_job, [(p, 1000 * p + i) for p in (5, 10, 20, 50) for i in range(10)], "highdim_mc.csv")
    highdim_mc_followup()
    # timings are measured serially, without competing worker processes
    rows = [timing_job((p, 7000 + i, eps)) for p in (2, 5, 10, 20, 50, 100) for i in range(5)
            for eps in (1e-6, 1e-10)]
    pd.DataFrame(rows).to_csv(OUT / "highdim_timing.csv", index=False)
    print("highdim_timing.csv done", flush=True)


# =============================================================================================
# Part 7. Computational benchmark (Section 5.3 and Online Resource 1)
# =============================================================================================
# The two R programs of the benchmark are kept here as text and written to a temporary folder
# when the benchmark runs (they need R with the package CompQuadForm).
R_SCRIPTS = {
"benchmark_cdf.R": r"""# Davies (1980) and Imhof (1961) CDF evaluations via CompQuadForm, for the benchmark of
# Section 5.3 (run by plbound.py).  Usage: Rscript benchmark_cdf.R cases.csv out.csv
#
# CompQuadForm evaluates P(Q > q) for Q = sum_j lambda_j chi2_1(delta_j) + sigma Z, so the
# constant kappa of the generalized chi-square law is absorbed in q = x - kappa.
suppressPackageStartupMessages(library(CompQuadForm))

args <- commandArgs(trailingOnly = TRUE)
cases <- read.csv(args[1], stringsAsFactors = FALSE)
parse <- function(s) as.numeric(strsplit(s, ";")[[1]])

timed <- function(expr_fn, min_time = 0.2) {
  reps <- 0; t0 <- proc.time()[["elapsed"]]
  repeat {
    val <- expr_fn(); reps <- reps + 1
    el <- proc.time()[["elapsed"]] - t0
    if (el >= min_time || reps >= 1000) break
  }
  list(val = val, sec = el / reps)
}

rows <- list()
for (i in seq_len(nrow(cases))) {
  cs <- cases[i, ]
  lam <- parse(cs$lam); del <- parse(cs$delta)
  for (acc in c(1e-6, 1e-8, 1e-10)) {
    r <- timed(function() suppressWarnings(davies(cs$q, lam, delta = del, sigma = cs$sigma,
                                                    lim = 1e8, acc = acc)))
    rows[[length(rows) + 1]] <- data.frame(case = cs$case, level = cs$level, method = "Davies",
      setting = sprintf("acc=%g", acc), value = 1 - r$val$Qq, bound = acc, seconds = r$sec,
      ifault = r$val$ifault)
  }
  for (lim in c(1e4)) {   # package default
    r <- timed(function() suppressWarnings(davies(cs$q, lam, delta = del, sigma = cs$sigma,
                                                    lim = lim, acc = 1e-10)))
    rows[[length(rows) + 1]] <- data.frame(case = cs$case, level = cs$level, method = "Davies",
      setting = "acc=1e-10, default lim", value = 1 - r$val$Qq, bound = 1e-10, seconds = r$sec,
      ifault = r$val$ifault)
  }
  if (cs$sigma == 0) {
    for (tol in c(1e-6, 1e-10)) {
      r <- timed(function() suppressWarnings(imhof(cs$q, lam, delta = del, epsabs = tol,
                                                   epsrel = tol, limit = 10000)))
      rows[[length(rows) + 1]] <- data.frame(case = cs$case, level = cs$level, method = "Imhof",
        setting = sprintf("eps=%g", tol), value = 1 - r$val$Qq, bound = r$val$abserr,
        seconds = r$sec, ifault = NA)
    }
  }
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
""",
"benchmark_pl_davies.R": r"""# PL(alpha) from Davies's (1980) algorithm combined with the propagation of Proposition 7, for
# the benchmark of Section 5.3 (run by plbound.py).  Usage: Rscript benchmark_pl_davies.R laws.csv out.csv [curve]
#
# Each distribution function is evaluated by CompQuadForm::davies with error at most acc; the
# threshold is bracketed as in Proposition 7(a) and the true positive rate enclosed as in 6(b).
# Normal laws (no quadratic part) use pnorm.
suppressPackageStartupMessages(library(CompQuadForm))

args <- commandArgs(trailingOnly = TRUE)
laws <- read.csv(args[1], stringsAsFactors = FALSE)
parse <- function(s) if (is.na(s) || s == "") numeric(0) else as.numeric(strsplit(s, ";")[[1]])
NORMAL_ERR <- 1e-15
n_calls <- 0

make_cdf <- function(L, acc) {
  lam <- parse(L$lam); del <- parse(L$delta)
  if (length(lam) == 0)
    return(list(F = function(x) pnorm((x - L$kappa0) / L$sigma), e = NORMAL_ERR))
  F <- function(x) {
    n_calls <<- n_calls + 1
    r <- suppressWarnings(davies(x - L$kappa, lam, delta = del, sigma = L$sigma, lim = 1e8, acc = acc))
    if (r$ifault != 0) stop(sprintf("davies ifault = %d", r$ifault))
    1 - r$Qq
  }
  list(F = F, e = acc)
}

tpr_enclosure <- function(L0, L1, alpha, acc) {
  c0 <- make_cdf(L0, acc); c1 <- make_cdf(L1, acc)
  # Proposition 7(a) needs F_0(t_hi) - e_0 > 1 - alpha, impossible when the error bound reaches alpha
  if (1 - alpha + c0$e >= 1) stop("acc >= alpha: the threshold cannot be bracketed")
  m <- L0$mean; s <- L0$sd
  t_star <- uniroot(function(t) c0$F(t) - (1 - alpha), c(m - 2 * s, m + 2 * s), extendInt = "upX",
                    tol = 1e-13 * (abs(m) + s), maxiter = 1000)$root
  eta <- 1e-4 * s
  dens <- max((c0$F(t_star + eta) - c0$F(t_star - eta)) / (2 * eta), 1e-300)
  d <- max(2 * c0$e / dens, 1e-14 * (1 + abs(t_star)))
  for (k in 0:200) {   # Proposition 7(a)
    lo <- t_star - d; hi <- t_star + d
    if (c0$F(lo) + c0$e < 1 - alpha && c0$F(hi) - c0$e > 1 - alpha) break
    if (k == 200) stop("bracketing failed")
    d <- 2 * d
  }
  c(1 - c1$F(hi) - c1$e, 1 - c1$F(lo) + c1$e)   # Proposition 7(b)
}

timed <- function(expr_fn, min_time = 0.2) {
  reps <- 0; t0 <- proc.time()[["elapsed"]]
  repeat {
    val <- expr_fn(); reps <- reps + 1
    el <- proc.time()[["elapsed"]] - t0
    if (el >= min_time || reps >= 100) break
  }
  list(val = val, sec = el / reps, reps = reps)
}

if (length(args) >= 3 && args[3] == "curve") {
  # one pass over a grid of alphas per model at acc = 1e-6: total time and largest half-width
  rows <- list()
  for (mod in unique(laws$model)) {
    lm <- laws[laws$model == mod, ]
    alphas <- sort(unique(lm$alpha)); hw <- numeric(0)
    t0 <- proc.time()[["elapsed"]]
    for (alpha in alphas) {
      g <- function(rule, cls) lm[lm$alpha == alpha & lm$rule == rule & lm$cls == cls, ][1, ]
      tf <- tpr_enclosure(g("f", 0), g("f", 1), alpha, 1e-6)
      tn <- tpr_enclosure(g("nb", 0), g("nb", 1), alpha, 1e-6)
      hw <- c(hw, 0.5 * ((tf[2] - tn[1]) - (tf[1] - tn[2])))
    }
    rows[[length(rows) + 1]] <- data.frame(model = mod, n_alpha = length(alphas), acc = 1e-6,
      seconds = proc.time()[["elapsed"]] - t0, max_half_width = max(hw))
    print(rows[[length(rows)]], digits = 6)
  }
  write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
  quit(save = "no")
}

rows <- list()
keys <- paste(laws$model, laws$alpha, sep = "|")
for (key in unique(keys)) {
  blk <- laws[keys == key, ]
  get <- function(rule, cls) blk[blk$rule == rule & blk$cls == cls, ][1, ]
  alpha <- blk$alpha[1]
  for (acc in c(1e-6, 1e-8, 1e-10)) {
    res <- tryCatch({
      n_calls <- 0
      r <- timed(function() {
        tf <- tpr_enclosure(get("f", 0), get("f", 1), alpha, acc)
        tn <- tpr_enclosure(get("nb", 0), get("nb", 1), alpha, acc)
        c(tf[1] - tn[2], tf[2] - tn[1])
      })
      data.frame(model = blk$model[1], alpha = alpha, acc = acc, PL_lo = r$val[1], PL_hi = r$val[2],
                 seconds = r$sec, calls = n_calls / r$reps, error = "")
    }, error = function(e) data.frame(model = blk$model[1], alpha = alpha, acc = acc, PL_lo = NA,
                                       PL_hi = NA, seconds = NA, calls = NA, error = conditionMessage(e)))
    rows[[length(rows) + 1]] <- res
    print(res, digits = 10)
  }
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
""",
}
_R_DIR = None


def _r_script(name):
    """Write an embedded R program to a temporary folder and return its path."""
    global _R_DIR
    if _R_DIR is None:
        _R_DIR = Path(tempfile.mkdtemp(prefix="plbound_R_"))
    path = _R_DIR / name
    path.write_text(R_SCRIPTS[name])
    return str(path)


BENCH_LEVELS = (1e-4, 0.01, 0.5, 0.99, 0.9999)


def bench_bivariate(r0, r1):
    S = lambda r: np.array([[1, r], [r, 1.0]])
    return GaussianModel(np.array([1.0, 2.0]), np.array([2.0, 0.5]), S(r0), S(r1))


def bench_cases():
    out = {}
    d = score_distributions(bench_bivariate(0.0, 0.99))
    out["biv (N,P) Q_f|Y=0, r=2"] = d[("f", 0)]
    out["biv (N,P) Q_f|Y=1, r=2"] = d[("f", 1)]
    out["biv (L,H) Q_f|Y=0, r=2"] = score_distributions(bench_bivariate(0.25, 0.75))[("f", 0)]
    out["biv (N,P) AUC difference, r=4"] = difference_distribution(bench_bivariate(0.0, 0.99), "f")
    m10, _ = random_model(10, np.random.default_rng(10001))
    m50, _ = random_model(50, np.random.default_rng(50001))
    out["random p=10 Q_f|Y=0, r=10"] = score_distributions(m10)[("f", 0)]
    out["random p=10 Q_nb|Y=1, r=10"] = score_distributions(m10)[("nb", 1)]
    out["random p=50 Q_f|Y=0, r=50"] = score_distributions(m50)[("f", 0)]
    return out


def timed(fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        v = fn()
    return v, (time.perf_counter() - t0) / reps


def benchmark_cdf():
    rows, rcases = [], []
    for name, g in bench_cases().items():
        for lev in BENCH_LEVELS:
            x = saddlepoint_quantile(g, lev)
            ref_ev = ErrorBoundedCDF(g, (x, x), eps=1e-12, max_terms=2 ** 25)
            ref, ref_err = ref_ev(x)
            base = dict(case=name, level=lev, x=x, ref=ref, ref_bound=ref_err)
            for eps in (1e-6, 1e-8, 1e-10):
                (F, e), sec = timed(lambda: ErrorBoundedCDF(g, (x, x), eps=eps)(x))
                rows.append(dict(base, method="error-bounded", setting=f"eps={eps:g}", value=F, bound=e, seconds=sec))
            F, sec = timed(lambda: saddlepoint_cdf(g, x), reps=20)
            rows.append(dict(base, method="saddlepoint", setting="", value=F, bound=np.nan, seconds=sec))
            rng = np.random.default_rng(1)
            F, sec = timed(lambda: float(mc_cdf(g, x, 10 ** 6, rng)[0]))
            rows.append(dict(base, method="Monte Carlo", setting="n=1e6", value=F, bound=np.nan, seconds=sec))
            rcases.append(dict(case=name, level=lev, x=x, q=x - g.kappa, sigma=g.sigma0,
                               lam=";".join(f"{v:.17g}" for v in g.lam), delta=";".join(f"{v:.17g}" for v in g.delta)))
            print(f"{name} level={lev}: ref={ref:.15f} (+/- {ref_err:.1e})", flush=True)
    pd.DataFrame(rcases).to_csv(OUT / "bench_cases_for_R.csv", index=False)
    subprocess.run(["Rscript", _r_script("benchmark_cdf.R"), str(OUT / "bench_cases_for_R.csv"),
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


def benchmark_pl():
    rows = []
    for name, m in bench_pl_models().items():
        for eps in (1e-6, 1e-8, 1e-10):
            o, sec = timed(lambda: performance_loss(m, 0.05, eps=eps, with_auc=False))
            rows.append(dict(model=name, method="error-bounded", setting=f"eps={eps:g}", PL=o["PL"].mid,
                             precision=o["PL"].half_width, seconds=sec))
        rng = np.random.default_rng(3)
        for N in (10 ** 4, 10 ** 5):
            pl, sec = timed(lambda: mc_performance_loss(m, 0.05, N, 20, rng))
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


def bench_pl_models():
    return {"biv (N,P)": bench_bivariate(0.0, 0.99), "biv (L,H)": bench_bivariate(0.25, 0.75),
            "random p=10": random_model(10, np.random.default_rng(10001))[0],
            "random p=50": random_model(50, np.random.default_rng(50001))[0]}


def bench_law_rows(name, a, dist):
    out = []
    for (rule, cls), g in dist.items():
        out.append(dict(model=name, alpha=a, rule=rule, cls=cls,
                        lam=";".join(f"{v:.17g}" for v in g.lam),
                        delta=";".join(f"{v:.17g}" for v in (g.eta / (2 * g.lam)) ** 2),
                        sigma=g.sigma0, kappa0=g.kappa0, kappa=g.kappa if g.r else g.kappa0,
                        mean=g.kappa0 + float(np.sum(g.lam)),
                        sd=float(np.sqrt(g.sigma0 ** 2 + np.sum(2 * g.lam ** 2 + g.eta ** 2)))))
    return out


def benchmark_davies(alphas=(0.05, 1e-4, 1e-6)):
    """PL(alpha) by Davies's algorithm combined with Proposition 7 (benchmark_pl_davies.R), and
    the error-bounded PL at strict operating points."""
    rows, laws = [], []
    for name, m in bench_pl_models().items():
        dist = score_distributions(m)
        for a in alphas:
            laws += bench_law_rows(name, a, dist)
            if a != 0.05:
                o, sec = timed(lambda: performance_loss(m, a, eps=1e-10, with_auc=False))
                rows.append(dict(model=name, alpha=a, method="error-bounded", setting="eps=1e-10",
                                 PL=o["PL"].mid, precision=o["PL"].half_width, seconds=sec))
                print(rows[-1], flush=True)
    pd.DataFrame(laws).to_csv(OUT / "bench_pl_laws_for_R.csv", index=False)
    subprocess.run(["Rscript", _r_script("benchmark_pl_davies.R"), str(OUT / "bench_pl_laws_for_R.csv"),
                    str(OUT / "bench_pl_davies_R.csv")], check=True)
    r = pd.read_csv(OUT / "bench_pl_davies_R.csv")
    for _, x in r.iterrows():
        rows.append(dict(model=x.model, alpha=x.alpha, method="Davies + Prop. 7", setting=f"acc={x.acc:g}",
                         PL=0.5 * (x.PL_lo + x.PL_hi), precision=0.5 * (x.PL_hi - x.PL_lo),
                         seconds=x.seconds, calls=x.calls, error=x.error))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bench_pl_davies.csv", index=False)
    print(df.to_string(), flush=True)
    # whole loss curves: one FFT per law (pl_curve) versus per-point Davies + Proposition 7
    grid = np.geomspace(1e-4, 0.5, 60)
    crow, claws = [], []
    for name, m in bench_pl_models().items():
        out, sec = timed(lambda: pl_curve(m, grid, eps=1e-9))
        crow.append(dict(model=name, method="error-bounded FFT grid", n_alpha=grid.size, seconds=sec,
                         max_half_width=max(e.half_width for e in out["PL"])))
        dist = score_distributions(m)
        for a in grid:
            claws += bench_law_rows(name, float(a), dist)
    pd.DataFrame(claws).to_csv(OUT / "bench_curve_laws_for_R.csv", index=False)
    subprocess.run(["Rscript", _r_script("benchmark_pl_davies.R"), str(OUT / "bench_curve_laws_for_R.csv"),
                    str(OUT / "bench_curve_davies_R.csv"), "curve"], check=True)
    r = pd.read_csv(OUT / "bench_curve_davies_R.csv")
    for _, x in r.iterrows():
        crow.append(dict(model=x.model, method="Davies + Prop. 7, acc=1e-6", n_alpha=x.n_alpha,
                         seconds=x.seconds, max_half_width=x.max_half_width))
    cdf = pd.DataFrame(crow)
    cdf.to_csv(OUT / "bench_curve.csv", index=False)
    print(cdf.to_string(), flush=True)


def refresh_bivariate_reference():
    """Replace the reference of the three bivariate laws by the independent conditioning integral
    (Part 3), which shares no code with the inversion, and recompute the errors."""
    S = biv_cov
    laws = {"biv (N,P) Q_f|Y=0, r=2": (cond_coef(S(0.0), S(0.99)), MU0, S(0.0)),
            "biv (N,P) Q_f|Y=1, r=2": (cond_coef(S(0.0), S(0.99)), MU1, S(0.99)),
            "biv (L,H) Q_f|Y=0, r=2": (cond_coef(S(0.25), S(0.75)), MU0, S(0.25))}
    df = pd.read_csv(OUT / "bench_cdf.csv")
    df["ref_method"] = "inversion, eps=1e-12"
    for (case, level), idx in df.groupby(["case", "level"]).groups.items():
        if case not in laws:
            continue
        (A, a, b), mu, Sig = laws[case]
        x = float(df.loc[idx, "x"].iloc[0])
        ref = cond_cdf(A, a, b, mu, Sig, x)
        old = float(df.loc[idx, "ref"].iloc[0])
        bound = float(df.loc[idx, "ref_bound"].iloc[0])
        print(f"{case} level={level}: independent - inversion = {ref - old:+.2e} (inversion bound {bound:.1e})")
        df.loc[idx, "ref"] = ref
        df.loc[idx, "ref_method"] = "conditioning integral"
    df["abs_error"] = (df["value"] - df["ref"]).abs()
    df.to_csv(OUT / "bench_cdf.csv", index=False)


def run_benchmark():
    """Section 5.3: distribution functions, the loss, Davies + Proposition 7, strict operating points."""
    if shutil.which("Rscript") is None:
        print("Rscript not found: the benchmark needs R with the package CompQuadForm; skipped", flush=True)
        return
    benchmark_cdf()
    refresh_bivariate_reference()
    benchmark_pl()
    benchmark_davies()


# =============================================================================================
# Part 7b. Estimating the loss from data (Section 6): simulation study of the delta-method and bootstrap
# intervals (aims: bias of the plug-in estimator and coverage of 95% intervals for PL(0.05);
# data: the bivariate design with n observations per class; estimand: PL(0.05) of the true model;
# methods: plug-in, bootstrap bias correction, delta-method, percentile and basic intervals;
# performance: bias, RMSE, coverage with its Monte Carlo standard error, interval width)
# =============================================================================================
INF_DESIGNS = {"(N,P)": (0.0, 0.99), "(L,H)": (0.25, 0.75), "(N,N)": (0.0, 0.0)}
INF_SIZES = (250, 2500)
INF_REPS, INF_BOOT, INF_ALPHA = 200, 100, 0.05
INF_SE_RTOL = 0.01      # eps of the delta method is refined until the bound on the error of se is below 1% of se
INF_BOUNDARY_REPS = 4000  # plug-in estimates only, to check the predicted bias where the loss vanishes


def inference_job(args):
    name, n, rep = args
    r0, r1 = INF_DESIGNS[name]
    m = GaussianModel(MU0, MU1, biv_cov(r0), biv_cov(r1))
    rng = np.random.default_rng([list(INF_DESIGNS).index(name), n, rep])
    L0, L1 = np.linalg.cholesky(m.S0), np.linalg.cholesky(m.S1)
    fit = fit_gaussian(m.mu0 + rng.standard_normal((n, 2)) @ L0.T, m.mu1 + rng.standard_normal((n, 2)) @ L1.T)
    t0 = time.perf_counter()
    d = pl_delta_interval(fit, n, n, INF_ALPHA, se_rtol=INF_SE_RTOL)
    # bootstrap values need only be accurate to well below their Monte Carlo spread, of the order se
    eps_boot = min(1e-4, 1e-3 * max(d["se"], 1e-9))
    b = pl_bootstrap(fit, n, n, INF_ALPHA, INF_BOOT, rng, eps=eps_boot, est=d["PL"])
    return dict(design=name, n=n, rep=rep, PL_hat=d["PL"], se_delta=d["se"], lo_delta=d["lo"], hi_delta=d["hi"],
                eps_delta=d["eps"], grad_err=d["grad_err"], se_err=d["se_err"], PL_bc=b["PL_bc"], bias_boot=b["bias"],
                sd_boot=b["sd"], lo_pct=b["lo_pct"], hi_pct=b["hi_pct"], lo_basic=b["lo_basic"], hi_basic=b["hi_basic"],
                eps_boot=eps_boot, hw_boot_max=b["hw_max"], seconds=time.perf_counter() - t0)


def boundary_job(args):
    """Plug-in estimate only, from the same random numbers as inference_job (Proposition 8(b) check)."""
    name, n, rep = args
    r0, r1 = INF_DESIGNS[name]
    m = GaussianModel(MU0, MU1, biv_cov(r0), biv_cov(r1))
    rng = np.random.default_rng([list(INF_DESIGNS).index(name), n, rep])
    L0, L1 = np.linalg.cholesky(m.S0), np.linalg.cholesky(m.S1)
    fit = fit_gaussian(m.mu0 + rng.standard_normal((n, 2)) @ L0.T, m.mu1 + rng.standard_normal((n, 2)) @ L1.T)
    return performance_loss(fit, INF_ALPHA, 1e-9, with_auc=False)["PL"].mid


def inference_summary(df):
    rows = []
    for (name, n), d in df.groupby(["design", "n"], sort=False):
        truth, R = d.PL_true.iloc[0], len(d)
        row = dict(design=name, n=n, reps=R, PL_true=truth, bias=d.PL_hat.mean() - truth,
                   bias_bc=d.PL_bc.mean() - truth, rmse=float(np.sqrt(((d.PL_hat - truth) ** 2).mean())),
                   rmse_bc=float(np.sqrt(((d.PL_bc - truth) ** 2).mean())), sd_hat=d.PL_hat.std(ddof=1),
                   se_delta_mean=d.se_delta.mean(), sd_boot_mean=d.sd_boot.mean())
        for k in ("delta", "pct", "basic"):
            cov = ((d[f"lo_{k}"] <= truth) & (truth <= d[f"hi_{k}"])).mean()
            row.update({f"cover_{k}": cov, f"cover_{k}_mcse": float(np.sqrt(cov * (1 - cov) / R)),
                        f"width_{k}": (d[f"hi_{k}"] - d[f"lo_{k}"]).mean()})
        row.update(se_err_rel_max=(d.se_err / d.se_delta).max(), refined=(d.eps_delta < 1e-8).mean(),
                   hw_boot_rel_max=(d.hw_boot_max / d.sd_boot).max(), pl_bc_neg=(d.PL_bc < 0).mean(),
                   seconds_mean=d.seconds.mean(), seconds_max=d.seconds.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _single_threaded_workers():
    """Worker processes inherit these settings, so that each uses one BLAS thread and the pool does
    not oversubscribe the cores."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")


def run_inference(workers=6):
    """Simulation study of Section 6 (results/inference_sim.csv and inference_summary.csv)."""
    from multiprocessing import Pool
    _single_threaded_workers()
    truth = {name: performance_loss(GaussianModel(MU0, MU1, biv_cov(r0), biv_cov(r1)), INF_ALPHA, 1e-10,
                                    with_auc=False)["PL"].mid for name, (r0, r1) in INF_DESIGNS.items()}
    tasks = [(name, n, rep) for name in INF_DESIGNS for n in INF_SIZES for rep in range(INF_REPS)]
    # replications are appended to a checkpoint file as they finish, so an interrupted run resumes
    ckpt = OUT / "inference_sim_partial.csv"
    done = pd.read_csv(ckpt) if ckpt.exists() else pd.DataFrame()
    finished = set(zip(done.design, done.n, done.rep)) if len(done) else set()
    todo = [t for t in tasks if t not in finished]
    print(f"inference: {len(finished)} replications from the checkpoint, {len(todo)} to run", flush=True)
    t0 = time.perf_counter()
    if todo:
        with Pool(workers) as pool, open(ckpt, "a") as fh:
            for k, row in enumerate(pool.imap_unordered(inference_job, todo, chunksize=1), 1):
                pd.DataFrame([row]).to_csv(fh, header=fh.tell() == 0, index=False)
                fh.flush()
                if k % 50 == 0:
                    print(f"inference: {k}/{len(todo)} replications [{time.perf_counter() - t0:.0f}s]", flush=True)
    df = pd.read_csv(ckpt).drop_duplicates(["design", "n", "rep"])
    df["design"] = pd.Categorical(df.design, list(INF_DESIGNS))
    df = df.sort_values(["design", "n", "rep"]).reset_index(drop=True)
    df["design"] = df.design.astype(str)
    df["PL_true"] = df.design.map(truth)
    df.to_csv(OUT / "inference_sim.csv", index=False)
    ckpt.unlink()
    summ = inference_summary(df)
    # Proposition 8(b): where the loss vanishes, the plug-in bias is approximately tr(H V)/2; checked
    # against the mean of INF_BOUNDARY_REPS plug-in estimates (the first INF_REPS are those of df)
    for c in ("bias_theory", "hessian_rank", "bias_large", "bias_large_mcse"):
        summ[c] = np.nan
    for name, (r0, r1) in INF_DESIGNS.items():
        if truth[name] == 0:
            m = GaussianModel(MU0, MU1, biv_cov(r0), biv_cov(r1))
            H = pl_hessian(m, INF_ALPHA)
            ev = np.linalg.eigvalsh(H)      # eigenvalues below 1e-3 of the largest are second-difference error
            for k in summ.index[summ.design == name]:
                n = int(summ.at[k, "n"])
                with Pool(workers) as pool:
                    est = np.array(pool.map(boundary_job, [(name, n, r) for r in range(INF_BOUNDARY_REPS)], chunksize=20))
                summ.loc[k, ["bias_theory", "hessian_rank", "bias_large", "bias_large_mcse"]] = [
                    0.5 * np.trace(H @ parameter_covariance(m, n, n)), int((np.abs(ev) > 1e-3 * np.abs(ev).max()).sum()),
                    est.mean(), est.std(ddof=1) / np.sqrt(est.size)]
    summ.to_csv(OUT / "inference_summary.csv", index=False)
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"), flush=True)


# =============================================================================================
# Part 8. Application: NIST BSSR1 score fusion (Section 7)
# =============================================================================================
MATCHERS = {"face_C": "face/C", "face_G": "face/G", "finger_LI": "li/V", "finger_RI": "ri/V"}


def read_signature_set(path):
    pat = re.compile(r'<signature name="([^"]+)" subject_id="([^"]+)"/>')
    return pat.findall(Path(path).read_text())


def read_similarity_file(path):
    lines = Path(path).read_text().splitlines()  # header line, count, then one score per line
    n = int(lines[1])
    scores = np.array(lines[2:2 + n], dtype=float)
    assert scores.size == n
    return scores


# ---------------------------------------------------------------------------------------------
# Sets 2 (face_x_face, 3000 subjects) and 3 (fing_x_fing, 6000 subjects)
# ---------------------------------------------------------------------------------------------
def read_fast(path):
    with open(path) as fh:
        fh.readline()
        n = int(fh.readline())
        v = np.array(fh.read().split()[:n], dtype=np.float32)   # files end with a footer line
    assert v.size == n
    return v


def prepare_large(root, out_face, out_finger):
    root = Path(root)
    # Set 2: two probes per subject, matchers C and G
    users = read_signature_set(root / "face_x_face/sets/dos/users.xml")
    enr = read_signature_set(root / "face_x_face/sets/dos/enrollees.xml")
    mats = {k: np.vstack([read_fast(root / "face_x_face/sims/dos/face" / k / n) for n, _ in users])
            for k in ("C", "G")}
    np.savez_compressed(out_face, face_C=mats["C"], face_G=mats["G"],
                        probe_id=np.array([s for _, s in users]), enrollee_id=np.array([s for _, s in enr]))
    print("set 2:", mats["C"].shape, "genuine:", (np.array([s for _, s in users])[:, None] ==
                                                  np.array([s for _, s in enr])[None, :]).sum())
    # Set 3: left and right index fingers, aligned by subject
    ids = {}
    mats = {}
    for f in ("li", "ri"):
        users = read_signature_set(root / f"fing_x_fing/sets/dos/{f}/users.xml")
        enr = read_signature_set(root / f"fing_x_fing/sets/dos/{f}/enrollees.xml")
        ids[f] = (np.array([s for _, s in users]), np.array([s for _, s in enr]))
        mats[f] = np.vstack([read_fast(root / f"fing_x_fing/sims/dos/{f}/V" / n) for n, _ in users])
    pu = np.argsort(ids["ri"][0])[np.argsort(np.argsort(ids["li"][0]))]   # align ri rows to li order
    pe = np.argsort(ids["ri"][1])[np.argsort(np.argsort(ids["li"][1]))]
    ri = mats["ri"][pu][:, pe]
    assert np.all(ids["ri"][0][pu] == ids["li"][0]) and np.all(ids["ri"][1][pe] == ids["li"][1])
    np.savez_compressed(out_finger, finger_LI=mats["li"], finger_RI=ri,
                        probe_id=ids["li"][0], enrollee_id=ids["li"][1])
    print("set 3:", mats["li"].shape)


def prepare_bssr1_set1(root, out):
    """Set 1 (fing_x_face): four 517 x 517 similarity matrices; rows follow users.xml (probes),
    columns follow enrollees.xml (gallery)."""
    root = Path(root)
    users = read_signature_set(root / "sets/dos/users.xml")
    enrollees = read_signature_set(root / "sets/dos/enrollees.xml")
    uid = np.array([s for _, s in users])
    eid = np.array([s for _, s in enrollees])
    genuine = uid[:, None] == eid[None, :]
    mats = {}
    for key, sub in MATCHERS.items():
        mats[key] = np.vstack([read_similarity_file(root / "sims/dos" / sub / name) for name, _ in users])
    np.savez_compressed(out, genuine=genuine, user_id=uid, enrollee_id=eid, **mats)
    print(f"users={len(users)} enrollees={len(enrollees)} genuine pairs={genuine.sum()}")
    for key, m in mats.items():
        print(f"{key}: shape={m.shape} min={m.min():.4g} max={m.max():.4g} "
              f"genuine mean={m[genuine].mean():.4g} impostor mean={m[~genuine].mean():.4g}")


def prepare_bssr1(source, data=None):
    """Convert the NIST BSSR1 download (bssr1.zip, or the folder it was extracted to) into the
    three .npz files used by Section 7. Source: https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1"""
    data = Path(data or DATA)
    data.mkdir(parents=True, exist_ok=True)
    source, tmp = Path(source), None
    if source.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="bssr1_"))
        with zipfile.ZipFile(source) as z:
            z.extractall(tmp)
        source = tmp
    try:
        root = next((p.parent for p in sorted(source.rglob("fing_x_face")) if p.is_dir()), None)
        if root is None:
            raise FileNotFoundError(f"no folder fing_x_face under {source}")
        prepare_bssr1_set1(root / "fing_x_face", data / "bssr1_set1.npz")
        prepare_large(root, data / "bssr1_set2_face.npz", data / "bssr1_set3_finger.npz")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


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
    d1 = np.load(DATA / "bssr1_set1.npz")
    m1 = ["face_C", "face_G", "finger_LI", "finger_RI"]
    set1 = ScoreSet("Set 1", np.stack([d1[k] for k in m1], -1), d1["user_id"], d1["enrollee_id"], m1)
    d2 = np.load(DATA / "bssr1_set2_face.npz")
    set2 = ScoreSet("Set 2", np.stack([d2["face_C"], d2["face_G"]], -1), d2["probe_id"], d2["enrollee_id"],
                    ["face_C", "face_G"])
    d3 = np.load(DATA / "bssr1_set3_finger.npz")
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
    return GaussianModel(ti.mean(0), tg.mean(0), np.atleast_2d(np.cov(ti.T)), np.atleast_2d(np.cov(tg.T)))


def fused(model, X):
    return {m: quad_form_values(X, A, a, b) for m, (A, a, b, _) in scores(model).items()}


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
        o = performance_loss(model, a, eps=eps, with_auc=(a == ALPHAS[0]))
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
                   corr_gen=";".join(f"{v:.6f}" for v in Rg[iu]), corr_imp=";".join(f"{v:.6f}" for v in Ri[iu]),
                   lambdas=";".join(f"{t['lmb']:.3f}" for t in trs), **model_pl(model, 1e-10))
        row.update({f"{k}_insample": v for k, v in evaluate(model, tg, ti, rng).items()})
        rows.append(row)
        print(f"[full] {name}: n_gen={gen.shape[0]} n_imp={imp.shape[0]} " +
              " ".join(f"PL({a})={row[f'PL_model_{a}']:.5f}/{row[f'PL_emp_{a}_insample']:.5f}" for a in ALPHAS),
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bssr1_fullfit.csv", index=False)
    return models


def repeated_splits(sets, R=50, R_model=50, seed=2026):
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


def bssr1_summarise(full, splits, pboot):
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


def run_bssr1():
    """Section 7: full-data fit, repeated subject-disjoint splits, parametric bootstrap and inference."""
    files = [DATA / f for f in ("bssr1_set1.npz", "bssr1_set2_face.npz", "bssr1_set3_finger.npz")]
    if not all(f.exists() for f in files):
        print(f"BSSR1 data not found in {DATA}: run once with --bssr1 <bssr1.zip>; Section 7 skipped", flush=True)
        return
    rng = np.random.default_rng(11)
    sets = load_sets()
    for s in sets.values():
        print(s.name, "subjects:", len(s.subjects), "probes x enrollees:", s.S.shape[:2], flush=True)
    models = full_fit(sets, rng)
    splits = repeated_splits(sets)
    pboot = parametric_bootstrap(models)
    bssr1_summarise(pd.read_csv(OUT / "bssr1_fullfit.csv"), splits, pboot)
    bssr1_inference(models, splits)


def bssr1_inference(models, splits, alphas=(0.001, 0.01)):
    """Standard errors of the model-based loss (results/bssr1_inference.csv). Half-sampling over
    subjects: the root mean squared deviation of the losses fitted on random halves of the subjects
    from their mean, a Monte Carlo version of the delete-d jackknife with d = n/2 (Shao and Wu 1989),
    which respects the dependence between comparisons that share a subject and includes the
    re-estimation of the transformations; and, for contrast, the delta method of pl_delta_interval,
    which treats all comparisons as independent Gaussian observations."""
    full = pd.read_csv(OUT / "bssr1_fullfit.csv").set_index("config")
    rows = []
    for name, (model, _, n_gen, n_imp) in models.items():
        s = splits[splits.config == name]
        for a in alphas:
            pl_full = float(full.loc[name, f"PL_model_{a}"])
            half = s[f"PL_model_{a}"].dropna().to_numpy()
            se_half = float(np.sqrt(np.mean((half - half.mean()) ** 2)))     # delete-d jackknife, d = n/2
            d = pl_delta_interval(model, n_imp, n_gen, a, se_rtol=0.01)
            rows.append(dict(config=name, alpha=a, PL_model=pl_full, n_half=half.size, se_half=se_half,
                             lo_half=pl_full - 1.96 * se_half, hi_half=pl_full + 1.96 * se_half,
                             se_delta_iid=d["se"], eps=d["eps"], grad_err=d["grad_err"], se_err=d["se_err"]))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bssr1_inference.csv", index=False)
    return df


# =============================================================================================
# Part 9. Figures
# =============================================================================================
plt = None          # matplotlib.pyplot, imported by _init_matplotlib() when the figures are drawn
FIG_DIR = None      # default OUT / "figures"


def _init_matplotlib():
    global plt
    if plt is None:
        import matplotlib.pyplot as pyplot
        pyplot.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.linewidth": 0.8,
            "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
            "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
        })
        plt = pyplot
    return plt


def _save_fig(fig, name):
    folder = FIG_DIR or OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name)
    plt.close(fig)
    print("wrote", folder / name)


# validated categorical slots (blue, orange, aqua) and chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


W1, W2 = 3.3, 4.8       # figure widths (inches); 4.8 in is the printed text width


def fig1_heatmaps():
    t = pd.read_csv(OUT / "bivariate_table.csv")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f9fe"] + SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.45), constrained_layout=True)
    for ax, a in zip(axes, (0.05, 0.01)):
        d = t[t.alpha == a]
        M = np.array([[d[(d.rho0 == r0) & (d.rho1 == r1)].PL.iloc[0] for r1 in BIV_LABELS] for r0 in BIV_LABELS])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.55, origin="upper")
        for i in range(5):
            for j in range(5):
                v = M[i, j]
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.25 else INK)
        ax.set_xticks(range(5), BIV_LABELS)
        ax.set_yticks(range(5), BIV_LABELS)
        ax.set_xlabel(r"$\rho_1$ (positive class)")
        ax.set_ylabel(r"$\rho_0$ (negative class)")
        ax.set_title(rf"$\mathrm{{PL}}({a})$", color=INK, fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5)
    _save_fig(fig, "Fig1.pdf")


def fig3_transposition():
    """ROC transposition under covariance swap (Theorem 3) and the two loss curves."""
    c = pd.read_csv(OUT / "bivariate_curves.csv")
    fig = plt.figure(figsize=(W2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
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
    axes[0].legend(loc="upper left", fontsize=6.5)
    ax = axes[2]
    for (r0, r1), col, ls in [(("N", "P"), BLUE, "-"), (("P", "N"), ORANGE, "--")]:
        d = c[(c.rho0 == r0) & (c.rho1 == r1)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({r0},{r1})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"loss curves (equal $\Delta\mathrm{AUC}$)", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    _save_fig(fig, "Fig3.pdf")


def fig2_curves():
    """Loss curves for symmetric (homoscedastic) and asymmetric scenarios, with closed forms."""
    c = pd.read_csv(OUT / "bivariate_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True, sharey=False)
    ax = axes[0]
    mu0, mu1 = np.array([1.0, 2.0]), np.array([2.0, 0.5])
    for lev, col, ls in [("M", BLUE, "-"), ("H", ORANGE, "--"), ("P", AQUA, "-.")]:
        d = c[(c.rho0 == lev) & (c.rho1 == lev)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({lev},{lev}) error-bounded")
        rho = {"M": 0.5, "H": 0.75, "P": 0.99}[lev]
        S = np.array([[1, rho], [rho, 1.0]])
        grid = np.geomspace(1e-4, 0.5, 13)
        ax.plot(grid, homoscedastic_pl(mu0, mu1, S, grid), ls="none", marker="o", ms=3.2,
                mfc="white", mec=col, mew=1.0)
    ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=INK2, label="closed form (Prop. 2)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"(a) shared covariance, $\rho_0=\rho_1$", color=INK, fontsize=9)
    ax.legend(fontsize=5.8, loc="upper right")
    ax = axes[1]
    for (r0, r1), col, ls in [(("N", "H"), BLUE, "-"), (("L", "P"), ORANGE, "--"), (("M", "P"), AQUA, "-.")]:
        d = c[(c.rho0 == r0) & (c.rho1 == r1)].sort_values("alpha")
        ax.plot(d.alpha, d.PL, color=col, ls=ls, label=f"({r0},{r1})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{PL}(\alpha)$")
    ax.set_title(r"(b) unequal covariance, $\rho_0<\rho_1$", color=INK, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    _save_fig(fig, "Fig2.pdf")


def fig4_mc():
    mc = pd.read_csv(OUT / "bivariate_mc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True)
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
    ax.set_title("(b) standardized discrepancy", color=INK, fontsize=9)
    ax.legend(fontsize=6, loc="lower left", ncols=2, markerscale=1.4)
    _save_fig(fig, "Fig4.pdf")


def fig5_highdim():
    r = pd.read_csv(OUT / "highdim_random.csv")
    fig, axs = plt.subplots(2, 2, figsize=(W2, 3.9), constrained_layout=True, sharex=True, sharey=True)
    axes = axs.ravel()
    for ax, (p, d) in zip(axes, r.groupby("p")):
        ax.plot(d.dAUC, d.PL01, ls="none", marker="o", ms=3.0, color=BLUE, mec="white", mew=0.4, alpha=0.9)
        rho = d[["dAUC", "PL01"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"$p={p}$  (Spearman {rho:.2f})", color=INK, fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel(r"$\Delta\mathrm{AUC}$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\mathrm{PL}(0.01)$")
    _save_fig(fig, "Fig5.pdf")


def fig6_bssr1():
    s = pd.read_csv(OUT / "bssr1_summary.csv")
    configs = list(dict.fromkeys(s.config))
    fig, axes = plt.subplots(2, 1, figsize=(W2, 4.3), constrained_layout=True, sharex=False)
    for ax, a in zip(axes, (0.001, 0.01)):
        d = s[s.alpha == a].set_index("config").loc[configs]
        y = np.arange(len(configs))[::-1]
        d = 100 * d.select_dtypes("number")                        # percentage points
        ax.errorbar(d.PL_pboot_mean, y + 0.18, xerr=1.96 * d.PL_pboot_sd, fmt="none", ecolor=AQUA, elinewidth=3,
                    alpha=0.55, label=r"parametric bootstrap, mean $\pm$ 1.96 SD")
        ax.errorbar(d.PL_emp_mean, y - 0.12, xerr=[np.clip(d.PL_emp_mean - d.PL_emp_q025, 0, None), np.clip(d.PL_emp_q975 - d.PL_emp_mean, 0, None)],
                    fmt="s", ms=4, color=ORANGE, ecolor=ORANGE, elinewidth=1, mec="white", mew=0.5,
                    label="held-out, mean and 2.5–97.5% quantiles")
        ax.plot(d.PL_model_full, y + 0.18, ls="none", marker="o", ms=5, color=BLUE, mec="white", mew=0.6,
                label="error-bounded, fitted model")
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(y, configs)
        ax.set_xlabel(rf"$\mathrm{{PL}}({a})$ (percentage points)")
        ax.set_title(f"FAR = {a:g}", color=INK, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncols=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.09))
    _save_fig(fig, "Fig6.pdf")


def make_figures():
    """Figures 1 to 5 of the paper and Figure S1 of Online Resource 1 (vector PDF). The files keep the
    numbering of an earlier draft: Fig4.pdf is Figure S1, and Fig5.pdf and Fig6.pdf are Figures 4 and 5."""
    _init_matplotlib()
    for fig in (fig1_heatmaps, fig2_curves, fig3_transposition, fig4_mc, fig5_highdim, fig6_bssr1):
        try:
            fig()
        except FileNotFoundError as e:
            print("skipped", fig.__name__, e)


# =============================================================================================
# Part 10. Tables (LaTeX, as printed in the paper)
# =============================================================================================
def _fmt(v, d):
    return f"{v:.{d}f}"


def sci_up(v, d=1):
    """LaTeX scientific notation rounded UP (for claims of the form 'below X')."""
    import math
    e = math.floor(math.log10(v))
    m = math.ceil(v / 10 ** e * 10 ** d - 1e-9) / 10 ** d
    if m >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{d}f}\\times10^{{{e}}}"


def sci(v, d=1):
    """LaTeX scientific notation, e.g. 1.1\\times10^{-9}."""
    m, e = f"{v:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def table_biv():
    t = pd.read_csv(OUT / "bivariate_table.csv")
    a5, a1 = t[t.alpha == 0.05].set_index(["rho0", "rho1"]), t[t.alpha == 0.01].set_index(["rho0", "rho1"])
    rows = []
    for i, r0 in enumerate(BIV_LABELS):
        for r1 in BIV_LABELS:
            x, y = a5.loc[(r0, r1)], a1.loc[(r0, r1)]
            rows.append(f"{r0} & {r1} & {_fmt(x.AUC_f, 6)} & {_fmt(x.AUC_nb, 6)} & {_fmt(x.dAUC, 6)} & "
                        f"{_fmt(x.TPR_f, 6)} & {_fmt(x.TPR_nb, 6)} & {_fmt(x.PL, 6)} & {_fmt(y.PL, 5)}\\\\")
        if i < 4:
            rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Error-bounded values for the bivariate design at $\\alpha=0.05$, and "
            "$\\PL(0.01)$. The correlations $\\rho_0$ and $\\rho_1$ of the negative and positive class take "
            "the values N, L, M, H, P $=0, 0.25, 0.5, 0.75, 0.99$. Half-widths of the enclosures are below "
            f"${sci_up(a5.PL_hw.max())}$ for $\\PL(0.05)$, below ${sci_up(max(a5.AUC_f_hw.max(), a5.AUC_nb_hw.max()))}$ for the AUCs and "
            f"below ${sci_up(a1.PL_hw.max())}$ for $\\PL(0.01)$}}\\label{{tab:biv}}\n"
            "\\footnotesize\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccccc@{}}\n\\toprule\n"
            "$\\rho_0$ & $\\rho_1$ & $\\AUC_f$ & $\\AUC_\\nb$ & $\\Delta\\AUC$ & $\\TPR_f(0.05)$ & "
            "$\\TPR_\\nb(0.05)$ & $\\PL(0.05)$ & $\\PL(0.01)$\\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_ar1():
    s = pd.read_csv(OUT / "highdim_structured.csv")
    combos = [(0.0, 0.0), (0.3, 0.3), (0.6, 0.6), (0.9, 0.9), (0.0, 0.9), (0.9, 0.0)]
    h0 = " & ".join(f"{a:g}" for a, _ in combos)
    h1 = " & ".join(f"{b:g}" for _, b in combos)
    rows = []
    for p in sorted(s.p.unique()):
        vals = []
        for vr in (1.0, 2.0):
            d = s[(s.p == p) & (s.var_ratio == vr)]
            for a, b in combos:
                vals.append(_fmt(d[(d.rho0 == a) & (d.rho1 == b)].PL05.iloc[0], 3))
        rows.append(f"$p={p}$ & " + " & ".join(vals) + "\\\\")
    return ("\\begin{table}[t]\n\\caption{Error-bounded $\\PL(0.05)$ in the AR(1) design for selected "
            "$(\\rho_0,\\rho_1)$, with class-1 variance ratio 1 (independence score linear) and 2 "
            "(independence score quadratic). Half-widths are below "
            f"${sci_up(s.PL05_hw.max())}$}}\\label{{tab:ar1}}\n\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{6}{c}{variance ratio 1} & \\multicolumn{6}{c}{variance ratio 2}\\\\\n"
            "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
            f"$\\rho_0$ & {h0} & {h0}\\\\\n$\\rho_1$ & {h1} & {h1}\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_bench_cdf():
    b = pd.read_csv(OUT / "bench_cdf.csv")
    order = [("error-bounded", "eps=1e-06"), ("error-bounded", "eps=1e-08"), ("error-bounded", "eps=1e-10"),
             ("Davies", "acc=1e-06"), ("Davies", "acc=1e-08"), ("Davies", "acc=1e-10"),
             ("Davies", "acc=1e-10, default lim"), ("Imhof", "eps=1e-06"), ("Imhof", "eps=1e-10"),
             ("saddlepoint", ""), ("Monte Carlo", "n=1e6")]
    label = {"eps=1e-06": "$\\varepsilon=10^{-6}$", "eps=1e-08": "$\\varepsilon=10^{-8}$",
             "eps=1e-10": "$\\varepsilon=10^{-10}$", "acc=1e-06": "acc $=10^{-6}$",
             "acc=1e-08": "acc $=10^{-8}$", "acc=1e-10": "acc $=10^{-10}$",
             "acc=1e-10, default lim": "acc $10^{-10}$, lim $10^4$", "": "Lugannani--Rice",
             "n=1e6": "$n=10^6$"}
    rows = []
    for meth, sett in order:
        d = b[(b.method == meth) & (b.setting.fillna("") == sett)]
        if d.empty:
            continue
        fails = int((d.ifault.fillna(0) > 0).sum()) if "ifault" in d else 0
        ok = d[(d.ifault.fillna(0) == 0)] if "ifault" in d else d
        err = ok.abs_error.max()
        within = "---"
        if meth == "error-bounded":
            within = f"{int((d.abs_error <= d.bound).sum())}/{len(d)}"
        def ms(v):
            v *= 1e3
            return f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
        rows.append(f"{meth} & {label.get(sett, sett)} & ${sci(err)}$ & {within} & {fails} & "
                    f"{ms(d.seconds.median())} & {ms(d.seconds.max())}\\\\")
    n = b.groupby(["case", "level"]).ngroups
    return ("\\begin{table}[t]\n\\caption{Distribution-function benchmark over " + f"{n}" +
            " evaluation points (seven generalized chi-square laws at five quantile levels from $10^{-4}$ to "
            "$1-10^{-4}$). Errors are absolute errors with respect to the reference values described in Section~\\ref{sec:bench}; for Davies's "
            "method, failures are calls returning a non-zero fault indicator and are excluded from the error. "
            "Times in milliseconds; acc is the accuracy requested from Davies's method and lim its maximal number of integration terms (default $10^4$, otherwise $10^8$)}\\label{tab:benchcdf}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccc@{}}\n\\toprule\n"
            "method & setting & max.\\ error & within bound & failures & median (ms) & max.\\ (ms)\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def _secs(v):
    """Seconds with two significant digits below 0.1 s (Davies's algorithm takes milliseconds)."""
    return f"{v:.2f}" if v >= 0.1 else f"{v:.2g}"


def _pow10(v):
    return f"10^{{{int(round(np.log10(v)))}}}"


def table_bench_pl():
    b = pd.read_csv(OUT / "bench_pl.csv")
    dv = pd.read_csv(OUT / "bench_pl_davies.csv")
    dv = dv[(dv.method == "Davies + Prop. 7") & (dv.alpha == 0.05)]
    rows = []
    for model, d in b.groupby("model", sort=False):
        for _, r in d.iterrows():
            if r.method == "error-bounded":
                eps = sci(float(r.setting.split("=")[1]), 0).replace("1\\times", "")
                rows.append(f"{model} & error-bounded, $\\varepsilon={eps}$ & {r.PL:.6f} & ${sci(r.precision)}$ & "
                            f"{r.seconds:.2f} & --- & --- \\\\")
                if r.setting == "eps=1e-10":
                    for _, x in dv[dv.model == model].iterrows():
                        acc = _pow10(float(x.setting.split("=")[1]))
                        rows.append(f"{model} & Davies + Prop.~7, acc $={acc}$ & {x.PL:.6f} & ${sci(x.precision)}$ & "
                                    f"{_secs(x.seconds)} & --- & --- \\\\")
            else:
                n = int(float(r.setting.split("=")[1]))
                mins = r["seconds_for_1e-4"] / 60
                days = r["seconds_for_1e-6"] / 86400
                rows.append(f"{model} & Monte Carlo, $N=10^{{{int(np.log10(n))}}}$ & {r.PL:.6f} & "
                            f"${sci(r.precision)}$ & {r.seconds:.3f} & {mins:.0f} & {days:,.0f} \\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Computing $\\PL(0.05)$: error-bounded enclosures, Davies's algorithm combined "
            "with Proposition~\\ref{prop:prop}, and the Monte Carlo "
            "plug-in estimator. Precision is the half-width of the enclosure or of a 95\\% interval "
            "from one replicate (Monte Carlo); time is per evaluation or per replicate in seconds. The last two "
            "columns give the CPU time Monte Carlo would need for a 95\\% half-width of $10^{-4}$ (minutes) "
            "and $10^{-6}$ (days), ignoring its $O(N^{-1})$ bias}"
            "\\label{tab:benchpl}\n\\footnotesize\\setlength{\\tabcolsep}{2pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccc@{}}\n\\toprule\n"
            "model & method & $\\PL(0.05)$ & precision & time (s) & $10^{-4}$ (min) & $10^{-6}$ (days)\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_bench_strict():
    d = pd.read_csv(OUT / "bench_pl_davies.csv")
    eb = d[d.method == "error-bounded"].set_index(["model", "alpha"])
    dv = d[(d.method == "Davies + Prop. 7")].copy()
    dv["acc"] = dv.setting.str.split("=").str[1].astype(float)
    rows = []
    for model in dict.fromkeys(d.model):
        for a in (1e-4, 1e-6):
            e = eb.loc[(model, a)]
            cells = [f"{model}", f"${_pow10(a)}$", f"{e.PL:.6f}", f"${sci(e.precision)}$", f"{e.seconds:.2f}"]
            for acc in (1e-6, 1e-10):
                x = dv[(dv.model == model) & (dv.alpha == a) & (dv.acc == acc)]
                if len(x) and np.isfinite(x.precision.iloc[0]):
                    cells += [f"${sci(x.precision.iloc[0])}$", _secs(x.seconds.iloc[0])]
                else:
                    cells += ["failed", "---"]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Strict operating points: error-bounded $\\PL(\\alpha)$ at $\\varepsilon=10^{-10}$ "
            "and Davies's algorithm combined with Proposition~\\ref{prop:prop} at accuracy acc. Half-widths of the "
            "enclosures and times in seconds}"
            "\\label{tab:strict}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccccc@{}}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{error-bounded, $\\varepsilon=10^{-10}$} & \\multicolumn{2}{c}{Davies, acc $=10^{-6}$} "
            "& \\multicolumn{2}{c}{Davies, acc $=10^{-10}$}\\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "model & $\\alpha$ & $\\PL(\\alpha)$ & half-width & time & half-width & time & half-width & time\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def _corr(text):
    vals = [float(v) for v in str(text).split(";")]
    return f"{vals[0]:.2f}" if len(vals) == 1 else f"{max(vals, key=abs):.2f}"


def table_bssr1(with_se=False):
    s = pd.read_csv(OUT / "bssr1_summary.csv")
    full = pd.read_csv(OUT / "bssr1_fullfit.csv").set_index("config")
    inf = pd.read_csv(OUT / "bssr1_inference.csv").set_index(["config", "alpha"]) if with_se else None
    short = {"face C + face G": "face C $+$ G", "finger LI + finger RI": "finger LI $+$ RI",
             "face C + finger LI": "face C $+$ finger LI", "all four matchers": "all four"}
    pp = lambda v: f"{100 * v:+.2f}".replace("+", "") if abs(v) >= 5e-5 else "0.00"
    rows = []
    for cfg in dict.fromkeys(s.config):
        fr = full.loc[cfg]
        for k, a in enumerate((0.001, 0.01)):
            r = s[(s.config == cfg) & (s.alpha == a)].iloc[0]
            if k == 0:
                lead = f"{short[cfg]} & {fr['set'][-1]} & {int(fr.n_gen)} & ${_corr(fr.corr_gen)}/{_corr(fr.corr_imp)}$"
            else:
                lead = " & & & "
            if with_se:
                model = f"${100 * r.PL_model_full:.3f}$ $({100 * inf.loc[(cfg, a), 'se_half']:.3f})$"
            else:
                model = f"${pp(r.PL_model_full)}$"
            rows.append(f"{lead} & {a:g} & {model} & ${pp(r.PL_emp_mean)}$ & "
                        f"$[{pp(r.PL_emp_q025)}, {pp(r.PL_emp_q975)}]$ & ${pp(r.PL_pboot_mean)}$ "
                        f"$({100 * r.PL_pboot_sd:.2f})$\\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{BSSR1 fusion; losses in percentage points. Correlations are those of "
            "the transformed genuine/impostor scores (largest absolute value for four matchers). Model: "
            "error-bounded $\\PL(\\alpha)$ of the Gaussian model fitted to all subjects (half-widths below "
            "$10^{-8}$)" + (", with its half-sampling standard error over subjects in parentheses" if with_se else "")
            + ". Held-out: mean empirical loss over 50 subject-disjoint 50/50 splits, with 2.5\\% and "
            "97.5\\% quantiles (a positive mean with both quantiles 0.00 means that most splits give a loss of exactly zero). Bootstrap: mean (standard deviation) of the same held-out loss for data "
            "simulated from the fitted Gaussian model}\\label{tab:bssr1}\n\\footnotesize\\setlength{\\tabcolsep}{"
            + ("2" if with_se else "3") + "pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccccc@{}}\n\\toprule\n"
            "fusion & set & genuine & corr. & FAR & model & held-out & 2.5--97.5\\% & bootstrap\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_bench_compact():
    d = pd.read_csv(OUT / "bench_pl_davies.csv")
    b = pd.read_csv(OUT / "bench_pl.csv")
    rows = []
    for model in dict.fromkeys(d.model):
        for k, a in enumerate((0.05, 1e-4, 1e-6)):
            if a == 0.05:
                e = b[(b.model == model) & (b.setting == "eps=1e-10")].iloc[0]
            else:
                e = d[(d.model == model) & (d.alpha == a) & (d.method == "error-bounded")].iloc[0]
            x = d[(d.model == model) & (d.alpha == a) & (d.method == "Davies + Prop. 7") & (d.setting == "acc=1e-10")]
            dv = (f"${sci(x.precision.iloc[0])}$ & {_secs(x.seconds.iloc[0])}"
                  if len(x) and np.isfinite(x.precision.iloc[0]) else "failed & ---")
            alab = "0.05" if a == 0.05 else f"${_pow10(a)}$"
            rows.append(f"{model if k == 0 else ''} & {alab} & {e.PL:.6f} & ${sci(e.precision)}$ & {_secs(e.seconds)} & {dv} \\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Computing $\\PL(\\alpha)$ with guaranteed error bounds: Algorithm~\\ref{alg:pl} "
            "at $\\varepsilon=10^{-10}$ and the algorithm of Davies (1980) combined with Proposition~\\ref{prop:prop} at "
            "acc $=10^{-10}$. Half-widths of the enclosures and times in seconds on one core, for the bivariate "
            "configurations (N,P) and (L,H) and two random models of Section~\\ref{sec:highdim}}\\label{tab:benchcompact}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccc@{}}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{Algorithm~\\ref{alg:pl}} & \\multicolumn{2}{c}{Davies + Prop.~\\ref{prop:prop}}\\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "model & $\\alpha$ & $\\PL(\\alpha)$ & half-width & time & half-width & time\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_inference():
    s = pd.read_csv(OUT / "inference_summary.csv")
    reps = int(s.reps.iloc[0])
    mcse = max(s[f"cover_{k}_mcse"].max() for k in ("delta", "pct", "basic"))
    num = lambda v, sign="": (f"{v:{sign}.4f}" if abs(v) >= 1e-3 or v == 0
                              else ("+" if sign and v > 0 else "") + sci(v, 1))
    rows = []
    for _, r in s.iterrows():
        cov = lambda k: f"{100 * r[f'cover_{k}']:.1f}"
        rows.append(f"{r.design} & {int(r.n)} & {r.PL_true:.4f} & ${num(r.bias, '+')}$ & ${num(r.bias_bc, '+')}$ & "
                    f"${num(r.sd_hat)}$ & ${num(r.se_delta_mean)}$ & {cov('delta')} & {cov('pct')} & {cov('basic')} \\\\")
    return ("\\begin{table}[t]\n\\caption{Estimating $\\PL(0.05)$ in the bivariate design from $n_y$ observations per "
            f"class ({reps} replications, {INF_BOOT} bootstrap samples). Bias of the plug-in and bootstrap "
            "bias-corrected estimates, standard deviation (SD) of the plug-in estimate and mean delta-method "
            "standard error (SE), and coverage (\\%) of nominal 95\\% intervals: Wald (delta method), percentile "
            f"and basic bootstrap; the Monte Carlo standard errors of the coverages are at most {100 * mcse:.1f} "
            "percentage points}\\label{tab:inference}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrcccccccc@{}}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{bias} & & & \\multicolumn{3}{c}{coverage (\\%)}\\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){8-10}\n"
            "design & $n_y$ & $\\PL(0.05)$ & plug-in & corrected & SD & SE & Wald & percentile & basic\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_explicit_approx():
    d = pd.read_csv(OUT / "explicit_approx.csv")
    pos = d[d.PL > 1e-6]
    rows = []
    for (r0, r1) in (("N", "P"), ("L", "H"), ("P", "N"), ("L", "M"), ("H", "H")):
        for a in (0.05, 0.01):
            r = d[(d.rho0 == r0) & (d.rho1 == r1) & (d.alpha == a)].iloc[0]
            rows.append(f"({r0},{r1}) & {a:g} & ${r.PL:.4f}$ & ${r.PL_normal:+.4f}$ & ${r.PL_cf:+.4f}$\\\\")
    rows.append("\\midrule")
    summ = []
    for name, fn in (("median absolute error", lambda e, x: f"{e.median():.4f}"),
                     ("largest absolute error", lambda e, x: f"{e.max():.4f}"),
                     ("negative values", lambda e, x: f"{int((x < 0).sum())}"),
                     ("relative error below 10\\%", lambda e, x: f"{int(((e / pos.PL) < 0.1).sum())}")):
        vals = [fn((pos[c] - pos.PL).abs(), pos[c]) for c in ("PL_normal", "PL_cf")]
        summ.append(f"\\multicolumn{{3}}{{l}}{{{name} ({len(pos)} cases)}} & {vals[0]} & {vals[1]}\\\\")
    return ("\\begin{table}[t]\n\\caption{Explicit cumulant approximations of $\\PL(\\alpha)$ in the bivariate design "
            "against the error-bounded values (half-widths below $10^{-9}$). Normal: two moments; Cornish--Fisher: "
            "Cornish--Fisher quantile of the class-0 score and Edgeworth tail of the class-1 score from four cumulants. "
            "The summary covers all configurations with a positive loss at $\\alpha\\in\\{0.05,0.01\\}$}"
            "\\label{tab:explicit}\n\\small\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccc@{}}\n\\toprule\n"
            "configuration & $\\alpha$ & error-bounded & normal & Cornish--Fisher\\\\\n\\midrule\n"
            + "\n".join(rows + summ) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_proportional():
    d = pd.read_csv(OUT / "proportional.csv")
    rows, prev = [], None
    for _, r in d.iterrows():
        key = (r.rho, r.var1, r.var2, r.k, r.means)
        lead = (f"${r.rho:g}$ & $({r.var1:g},{r.var2:g})$ & ${r.k:g}$ & {r.means}" if key != prev else " & & &")
        prev = key
        hw = (r.TPR_f_hi - r.TPR_f_lo) / 2
        loss = f"{r.PL:.6f}" if r.PL - r.PL_hw > 0 else f"\\le{sci_up(r.PL + r.PL_hw, 0)}"
        rows.append(f"{lead} & {r.alpha:g} & ${r.TPR_f_explicit:.10f}$ & ${sci(hw, 1)}$ & ${loss}$\\\\")
    return ("\\begin{table}[t]\n\\caption{Proportional covariance matrices $\\bm\\Sigma_1=k\\bm\\Sigma_0$ with correlation "
            "$\\rho$ and class-0 variances as shown: the true positive rate of the full rule from Corollary~S1 "
            "against the half-width of its error-bounded enclosure; all " + str(int(d.inside.sum())) + " of "
            + str(len(d)) + " explicit values lie inside their enclosures. The last column is the error-bounded "
            "loss, which is positive by Theorem~4 because $\\rho\\ne0$; where it is below the resolution of its enclosure, "
            "the upper end of the enclosure is shown}\\label{tab:proportional}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}cccccccc@{}}\n\\toprule\n"
            "$\\rho$ & variances & $k$ & means & $\\alpha$ & $\\TPR_f$ (Corollary S1) & half-width & $\\PL(\\alpha)$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def latex_tables():
    """LaTeX source of Tables 1 to 5 of the paper and S1 to S5 of Online Resource 1, built from the
    CSV files in OUT."""
    gens = {"table1_bivariate": table_biv, "table2_ar1": table_ar1, "table3_benchmark": table_bench_compact,
            "table4_inference": table_inference, "table5_bssr1": lambda: table_bssr1(with_se=True),
            "tableS1_benchmark_cdf": table_bench_cdf, "tableS2_benchmark_pl": table_bench_pl,
            "tableS3_strict_operating_points": table_bench_strict, "tableS4_explicit_approximations": table_explicit_approx,
            "tableS5_proportional": table_proportional}
    out = {}
    for name, gen in gens.items():
        try:
            out[name] = gen()
        except (FileNotFoundError, KeyError) as e:
            print("table", name, "skipped:", e)
    return out


def write_tables():
    folder = OUT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, tex in latex_tables().items():
        (folder / f"{name}.tex").write_text(tex + "\n")
        print("wrote", folder / f"{name}.tex")


# =============================================================================================
# Part 11. Command line: reproduce everything in one run
# =============================================================================================
STEPS = ("tests", "bivariate", "explicit", "highdim", "benchmark", "inference", "bssr1", "figures", "tables")


def main(argv=None):
    global OUT, DATA
    ap = argparse.ArgumentParser(description="Reproduce every table and figure of the paper in one run.")
    ap.add_argument("--only", default=",".join(STEPS), help="comma-separated subset of " + ", ".join(STEPS))
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the results (default: %(default)s)")
    ap.add_argument("--data", type=Path, default=DATA, help="folder for the prepared BSSR1 files")
    ap.add_argument("--bssr1", type=Path, default=None, help="bssr1.zip from NIST, or its extracted folder")
    ap.add_argument("--workers", type=int, default=6, help="worker processes for the parallel studies")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        ap.error("unknown step(s): " + ", ".join(unknown))
    OUT, DATA = args.out.resolve(), args.data.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if "figures" in steps:
        import matplotlib
        matplotlib.use("Agg")
    t_all = time.perf_counter()
    for step in STEPS:
        if step not in steps:
            continue
        print(f"\n=== {step} ===", flush=True)
        t0 = time.perf_counter()
        if step == "tests":
            run_tests()
        elif step == "bivariate":
            run_bivariate()
        elif step == "explicit":
            run_explicit()
        elif step == "highdim":
            run_highdim(args.workers)
        elif step == "benchmark":
            run_benchmark()
        elif step == "inference":
            run_inference(args.workers)
        elif step == "bssr1":
            if args.bssr1 is not None:
                prepare_bssr1(args.bssr1, DATA)
            run_bssr1()
        elif step == "figures":
            make_figures()
        elif step == "tables":
            write_tables()
        print(f"=== {step} done in {time.perf_counter() - t0:.0f} s ===", flush=True)
    print(f"\nAll done in {(time.perf_counter() - t_all) / 60:.1f} min; results in {OUT}", flush=True)


if __name__ == "__main__":
    main()
