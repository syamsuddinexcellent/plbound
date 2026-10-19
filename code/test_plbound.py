"""Correctness tests for plbound against distributions with known CDFs."""
import math
import time

import numpy as np
from scipy import special, stats

import plbound as pc


def check(name, g, xs, truth, eps=1e-10):
    t0 = time.perf_counter()
    ev = pc.ErrorBoundedCDF(g, (min(xs), max(xs)), eps=eps)
    worst_ratio, worst_err, worst_bound = 0.0, 0.0, 0.0
    for x, F in zip(xs, truth):
        Fh, e = ev(x)
        err = abs(Fh - F)
        worst_err, worst_bound = max(worst_err, err), max(worst_bound, e)
        worst_ratio = max(worst_ratio, err / e)
        assert err <= e, (name, x, Fh, F, err, e)
    print(f"{name:38s} K_max={ev.K_max:>9d}  max|err|={worst_err:.1e}  max bound={worst_bound:.1e}  "
          f"max err/bound={worst_ratio:.2f}  [{time.perf_counter()-t0:.1f}s]")


xs = np.linspace(-6, 9, 13)
# 1. central chi-square with one degree of freedom
g = pc.GenChi2([1.0], [0.0], 0.0)
xp = np.array([0.01, 0.1, 0.5, 1, 2, 4, 8, 15])
check("chi2_1", g, xp, stats.chi2.cdf(xp, 1))
# 2. noncentral: (W+1)^2 - 1  = W^2 + 2W
g = pc.GenChi2([1.0], [2.0], 0.0)
xv = np.array([-0.9, -0.5, 0, 1, 3, 8, 20])
check("noncentral chi2_1 (r=1)", g, xv, special.ndtr(np.sqrt(xv + 1) - 1) - special.ndtr(-np.sqrt(xv + 1) - 1))
# 3. indefinite, r = 2: Z1^2 - Z2^2 = 2 Z1' Z2'  (product-normal law, K0 density)
g = pc.GenChi2([1.0, -1.0], [0.0, 0.0], 0.0)
def prodnorm_cdf(y):
    return 0.5 + np.sign(y) * special.iti0k0(np.abs(y))[1] / np.pi
xq = np.array([-8, -3, -1, -0.3, 0.2, 0.7, 2, 5, 9.0])
check("indefinite Z1^2 - Z2^2 (r=2)", g, xq, prodnorm_cdf(xq / 2))
# 4. scaled versions with very unequal weights (mimics lambda = (0.5, -99))
g = pc.GenChi2([99.0, -99.0], [0.0, 0.0], 3.0)
xq2 = 3 + 99 * np.array([-6, -1, -0.2, 0.1, 0.5, 3.0])
check("indefinite 99(Z1^2 - Z2^2)+3", g, xq2, prodnorm_cdf((xq2 - 3) / 99 / 2))
# 5. normal component only through eta with lambda = 0
g = pc.GenChi2([0.0], [2.0], 1.0, 0.5)
xn = np.linspace(-5, 7, 9)
check("pure normal", g, xn, special.ndtr((xn - 1) / math.sqrt(4.25)))
# 6. exponential: 0.5*(Z1^2 + Z2^2) ~ Exp(1)
g = pc.GenChi2([0.5, 0.5], [0.0, 0.0], 0.0)
xe = np.array([0.01, 0.2, 1, 3, 7, 15])
check("Exp(1) = (Z1^2+Z2^2)/2", g, xe, 1 - np.exp(-xe))
# 7. chi2_5 noncentral (r=5, equal weights)  vs scipy ncx2
g = pc.GenChi2([1.0] * 5, [2 * 1.3, 0, 0, 0, 0], 1.3 ** 2)   # W1^2+2.6W1+1.69 = (W1+1.3)^2
xc = np.array([0.5, 2, 5, 9, 20, 35])
check("ncx2(df=5, nc=1.69)", g, xc, stats.ncx2.cdf(xc, 5, 1.69))
print("ALL TESTS PASSED")
