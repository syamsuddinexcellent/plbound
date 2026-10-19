"""Independent check of the bivariate design (Section 5) without characteristic functions.

For X ~ N_2(mu, S) and Q(x) = x'Ax + a'x + b, condition on X_1: X_2 | X_1 is normal and Q is a
quadratic polynomial in X_2, so P(Q > t | X_1) is a finite sum of normal probabilities over the
intervals where that polynomial exceeds t. Integrating over X_1 with adaptive Gauss-Kronrod
quadrature gives P(Q > t) to about 1e-13, by a route that shares no code with plbound.

Usage:  python indep_bivariate.py      (writes ../results/bivariate_independent.csv)
"""
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special

MU0, MU1 = np.array([1.0, 2.0]), np.array([2.0, 0.5])
LEV = {"N": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "P": 0.99}
OUT = Path(__file__).resolve().parent.parent / "results"


def sigma(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def coef(S0, S1):
    P0, P1 = np.linalg.inv(S0), np.linalg.inv(S1)
    return P0 - P1, 2 * (P1 @ MU1 - P0 @ MU0), float(MU0 @ P0 @ MU0 - MU1 @ P1 @ MU1)


def tail(A, a, b, mu, S, t):
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


def cdf(A, a, b, mu, S, t):
    return 1.0 - tail(A, a, b, mu, S, t)


def tpr(A, a, b, S0, S1, alpha):
    g = lambda t: tail(A, a, b, MU0, S0, t) - alpha
    lo, hi = -50.0, 50.0
    while g(lo) < 0:
        lo *= 2
    while g(hi) > 0:
        hi *= 2
    t = optimize.brentq(g, lo, hi, xtol=1e-13, rtol=1e-14)
    return tail(A, a, b, MU1, S1, t)


def main():
    rows = []
    for (l0, l1), alpha in product(product(LEV, LEV), (0.05, 0.01)):
        S0, S1 = sigma(LEV[l0]), sigma(LEV[l1])
        f = tpr(*coef(S0, S1), S0, S1, alpha)
        nb = tpr(*coef(np.eye(2), np.eye(2)), S0, S1, alpha)
        rows.append(dict(rho0=l0, rho1=l1, alpha=alpha, TPR_f=f, TPR_nb=nb, PL=f - nb))
        print(f"({l0},{l1}) alpha={alpha}: PL={f - nb:.10f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bivariate_independent.csv", index=False)


if __name__ == "__main__":
    main()
