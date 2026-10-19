"""plbound: error-bounded evaluation of the fixed-FPR performance loss of the independence
assumption in Gaussian likelihood-ratio classification.

Companion code for Wisnubroto, Sumardi and Susyanto, "Error-bounded computation of the
fixed-false-positive-rate cost of the independence assumption in Gaussian classification".

Main objects
------------
GenChi2          Q = kappa0 + sigma0*G + sum_j (lam_j W_j^2 + eta_j W_j), all components N(0,1)
                 and independent (a generalized chi-square variable in "raw" parametrisation,
                 which is numerically stable when some lam_j are small or zero).
represent        Gaussian quadratic expression x'Ax + a'x + b, X ~ N(mu, Sigma)  ->  GenChi2.
ErrorBoundedCDF     Midpoint-rule Gil-Pelaez inversion with a-priori bounds on aliasing,
                 truncation and rounding error: returns F_hat(x) and a bound e(x) such that
                 |F(x) - F_hat(x)| <= e(x).
error_bounded_quantile, performance_loss, auc
                 Error-bounded enclosures of thresholds, TPRs, PL(alpha) and AUC.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

U = np.finfo(float).eps / 2          # unit roundoff
LN2 = math.log(2.0)
N_BLOCKS = 160                        # dyadic blocks used in the tail bounds
NORMAL_CDF_ERR = 1e-15                # accuracy of scipy's ndtr / ndtri


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
        """Bound on (1/pi) int_a^inf |phi(t)|/t dt  (Theorem 5(ii), modulus bound): the first
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
    """Distribution of Q = X'AX + a'X + b for X ~ N(mu, Sigma) as a GenChi2 (Lemma 4)."""
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
# Error-bounded CDF evaluation (Theorem 5)
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
# Error-bounded quantiles, TPRs, PL(alpha) and AUC (Proposition 6)
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


def error_bounded_quantile(g: GenChi2, p, eps=1e-10, evaluator=None, max_terms=2 ** 24):
    """Interval (t_lo, t_hi) error-bounded to contain F^{-1}(p), plus the evaluator used."""
    if g.is_normal:
        t = g.kappa0 + g.sigma0 * float(special.ndtri(p))
        d = 1e-13 * (abs(t) + g.sigma0)
        return (t - d, t + d), t, None
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
    for _ in range(80):
        Flo, elo = evaluator(t_star - d)
        Fhi, ehi = evaluator(t_star + d)
        if Flo + elo < p < Fhi - ehi:
            return (t_star - d, t_star + d), t_star, evaluator
        d *= 2
    raise RuntimeError("quantile bracketing failed")


def error_bounded_tpr(g0: GenChi2, g1: GenChi2, alpha, eps=1e-10, max_terms=2 ** 24):
    """Error-bounded enclosure of TPR(alpha) = 1 - F_1(F_0^{-1}(1 - alpha))."""
    (t_lo, t_hi), t_star, _ = error_bounded_quantile(g0, 1 - alpha, eps, max_terms=max_terms)
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
