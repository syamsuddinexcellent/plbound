# plbound — fixed-FPR loss of the independence assumption with guaranteed error bounds

Online Resource 1 for *Computing the fixed-false-positive-rate loss from assuming independence in
Gaussian classification with guaranteed error bounds*, Computational Statistics.
Repository: <https://github.com/syamsuddinexcellent/plbound>.
Authors: M. Syamsuddin Wisnubroto, Sumardi, Nanang Susyanto.
Corresponding author: Sumardi, Department of Mathematics, Universitas Gadjah Mada, Yogyakarta,
Indonesia (mardimath@ugm.ac.id).
Questions about the code: M. Syamsuddin Wisnubroto (msyamsuddinwisnubroto@mail.ugm.ac.id).

`plbound.py` computes guaranteed enclosures of

    PL(alpha) = TPR_full(alpha) - TPR_independence(alpha)

for two Gaussian classes, where the full rule is the Gaussian likelihood-ratio (QDA) rule and the
independence rule replaces each class covariance matrix by its diagonal.

## Quick start

```python
import numpy as np
import plbound as pc

S = lambda r: np.array([[1, r], [r, 1.0]])
model = pc.GaussianModel(mu0=[1, 2], mu1=[2, 0.5], S0=S(0.0), S1=S(0.99))
out = pc.performance_loss(model, alpha=0.05, eps=1e-10)
print(out["PL"])          # 0.2924439279 +/- 1.5e-10
print(out["dAUC"])        # AUC_f - AUC_nb, also error-bounded
```

Main functions

| function | returns |
|---|---|
| `represent(A, a, b, mu, Sigma)` | generalized chi-square law of x'Ax + a'x + b, X ~ N(mu, Sigma) |
| `ErrorBoundedCDF(g, window, eps)` | callable returning (F_hat(x), bound) with \|F(x) - F_hat(x)\| <= bound |
| `error_bounded_quantile`, `error_bounded_tpr`, `error_bounded_auc` | enclosures with guaranteed error bounds |
| `performance_loss(model, alpha, eps)` | enclosures of TPR_f, TPR_nb, PL(alpha), AUC_f, AUC_nb |
| `pl_curve(model, alphas)` | error-bounded loss curves via one FFT per law |
| `homoscedastic_pl`, `homoscedastic_max_pl`, `kantorovich_bound` | closed forms of Proposition 2 |

## Reproducing the paper

Requirements: Python >= 3.9 with numpy, scipy, pandas, matplotlib (`pip install -r requirements.txt`);
R with the package CompQuadForm for the benchmark.

```bash
python test_plbound.py             # exactness tests against known distributions
python indep_bivariate.py          # independent check of Table 1 (no characteristic functions)
python exp_bivariate.py            # Section 5, Table 1, Figures 1-4   (~10 min)
python exp_highdim.py              # Section 7, Table 4, Figure 5      (~15 min, 6 processes)
python exp_benchmark.py            # Section 6, Tables 2 and 3 (run alone; timings)
python exp_bssr1.py                # Section 8, Table 5, Figure 6      (~1 h, needs 8 GB RAM)
python make_figures.py
python build_manuscript.py         # inserts the tables into ../manuscript/manuscript.tex (manuscript sources are not part of the repository)
```

### BSSR1 data

Download `bssr1.zip` (178 MB, MD5 `43f40a6383c8b7a485f17dcfc98ad7bb`) from
<https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1>, unzip it, and run

```bash
python prepare_bssr1.py <dir>/bssr1/fing_x_face ../data/bssr1_set1.npz
python prepare_bssr1.py --large <dir>/bssr1 ../data/bssr1_set2_face.npz ../data/bssr1_set3_finger.npz
```

## Files

| file | content |
|---|---|
| `plbound.py` | the method (Lemma 4, Theorem 5, Proposition 6, Algorithm 1 of the paper) |
| `indep_bivariate.py` | independent one-dimensional conditioning integral for the bivariate design (validation of Table 1; benchmark reference) |
| `designs.py` | covariance designs of Section 7 |
| `exp_*.py` | experiments; results are written to `../results` |
| `benchmark_cdf.R` | Davies and Imhof evaluations via CompQuadForm |
| `prepare_bssr1.py` | conversion of the NIST score files |
| `make_figures.py`, `build_manuscript.py` | figures and tables of the paper |
