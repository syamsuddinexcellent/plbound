# plbound — fixed-FPR loss of the independence assumption with guaranteed error bounds

Companion code for a manuscript under review (the reference will be added on publication).
Repository: <https://github.com/syamsuddinexcellent/plbound>; archived version 1.0.0:
<https://doi.org/10.5281/zenodo.23472531>.
Authors: M. Syamsuddin Wisnubroto, Sumardi, Nanang Susyanto.
Corresponding author: Sumardi, Department of Mathematics, Universitas Gadjah Mada, Yogyakarta,
Indonesia (mardimath@ugm.ac.id).
Questions about the code: M. Syamsuddin Wisnubroto (msyamsuddinwisnubroto@mail.ugm.ac.id).

Everything is in one file, `plbound.py`. It is both the Python module and the script that
reproduces the paper.

## Reproducing the paper in one run

```bash
pip install -r requirements.txt
python plbound.py
```

This runs the exactness tests, the checks of Sections 3.4 and 3.5, Sections 5 to 7 and Sections S5,
S6 and S8 of Online Resource 1, and writes every number, figure and table of the paper: the CSV files to `../results`, the figures to `../results/figures` (`Fig1.pdf` … `Fig6.pdf`)
and the LaTeX tables to `../results/tables`. The full run takes about three and a half hours on a laptop
(Apple M1, seven worker processes), most of it for the simulation study of Section 6, which writes a
checkpoint and resumes where it stopped if it is interrupted. Section 7 needs about 8 GB of memory.

Two parts need resources outside Python. Without them, those steps are skipped with a message.

* **Section 5.3 (benchmark):** needs R with the package CompQuadForm. The R code is contained in
  `plbound.py`.
* **Section 7 (application):** needs the NIST BSSR1 scores. Download `bssr1.zip` (178 MB,
  MD5 `43f40a6383c8b7a485f17dcfc98ad7bb`) from
  <https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1> and pass it once:

  ```bash
  python plbound.py --bssr1 path/to/bssr1.zip
  ```

  The scores are converted into `../data/*.npz` and used in later runs.

Options:

| option | meaning |
|---|---|
| `--only tests,bivariate` | run a subset of `tests`, `bivariate`, `explicit`, `structure`, `highdim`, `benchmark`, `inference`, `bssr1`, `figures`, `tables` |
| `--out DIR` | write the results elsewhere (default `../results`) |
| `--data DIR` | folder for the prepared BSSR1 files (default `../data`) |
| `--workers N` | worker processes for Sections 5.2 and 6 (default 6) |

Timings in the benchmark depend on the machine. All other results are deterministic (fixed random
seeds).

## Using the module

```python
import numpy as np
import plbound as pc

S = lambda r: np.array([[1, r], [r, 1.0]])
model = pc.GaussianModel(mu0=[1, 2], mu1=[2, 0.5], S0=S(0.0), S1=S(0.99))
out = pc.performance_loss(model, alpha=0.05, eps=1e-10)
print(out["PL"])          # 0.2924439279 +/- 1.5e-10
print(out["dAUC"])        # AUC_f - AUC_nb, also error-bounded
```

| function | returns |
|---|---|
| `represent(A, a, b, mu, Sigma)` | generalized chi-square law of x'Ax + a'x + b, X ~ N(mu, Sigma) (Lemma 6) |
| `ErrorBoundedCDF(g, window, eps)` | callable returning (F_hat(x), bound) with \|F(x) - F_hat(x)\| <= bound (Theorem 7) |
| `error_bounded_quantile`, `error_bounded_tpr`, `error_bounded_auc` | enclosures with guaranteed error bounds (Proposition 8) |
| `performance_loss(model, alpha, eps)` | enclosures of TPR_f, TPR_nb, PL(alpha), AUC_f, AUC_nb (Algorithm 1) |
| `pl_curve(model, alphas)` | error-bounded loss curves via one FFT per law |
| `homoscedastic_pl`, `homoscedastic_max_pl`, `kantorovich_bound` | closed forms of Proposition 2 |
| `fit_gaussian(X0, X1)` | plug-in model from two samples (rows are observations) |
| `pl_delta_interval(model, n0, n1, alpha)` | plug-in estimate with delta-method standard error and Wald interval; the numerical error of the gradient is bounded (Section 6) |
| `pl_bootstrap(model, n0, n1, alpha, B, rng)` | parametric bootstrap: bias-corrected estimate, percentile and basic intervals |
| `genchi2_cumulants(g)`, `explicit_tpr(g0, g1, alpha, method)` | cumulants of a generalized chi-square law; normal and Cornish–Fisher approximations of a true positive rate (Online Resource 1, Section S5) |
| `proportional_tpr_full(mu0, mu1, S0, k, alpha)` | explicit true positive rate of the full rule when Sigma1 = k Sigma0 (Online Resource 1, Section S6) |
| `zero_loss_constant(mu0, mu1, S0, S1)` | the constant c and the residual of the zero-loss criterion (Theorem 4(iv)) |
| `weak_equal_var(mu0, mu1, E0, E1, alpha)` | second-order approximation of the loss near independence for equal marginal variances (Corollary 2) |

## Layout of `plbound.py`

| part | content |
|---|---|
| 1 | generalized chi-square laws, error-bounded inversion, propagation through the quantile, the loss |
| 2 | covariance designs of Sections 5 and 6 |
| 3 | independent check of Table 1 by a one-dimensional conditioning integral |
| 4 | exactness tests against laws with known distribution functions |
| 5–8 | Sections 5.1 (bivariate design; part 5b: Sections S5 and S6 of Online Resource 1; part 5c: checks of Sections 3.4 and 3.5, Section S8), 5.2 (higher dimensions), 5.3 (benchmark, with the R code), 6 (simulation study of the inference, part 7b) and 7 (BSSR1) |
| 9–10 | figures and LaTeX tables |
| 11 | command line |
