# plbound

Code and numerical results for

> M. Syamsuddin Wisnubroto, Sumardi and Nanang Susyanto, *Computing the fixed-false-positive-rate loss
> from assuming independence in Gaussian classification with guaranteed error bounds*
> (submitted to *Computational Statistics*).

For two Gaussian classes, `plbound` computes guaranteed enclosures of the loss in true positive rate at a
fixed false positive rate, PL(alpha) = TPR_full(alpha) - TPR_independence(alpha), incurred when a
likelihood-ratio classifier replaces the class covariance matrices by their diagonals. The generalized
chi-square distribution functions involved are evaluated by midpoint-rule inversion of the characteristic
function with explicit bounds on aliasing, truncation and rounding error, and the bounds are propagated
through the quantile that defines the threshold.

Everything is in one file, [`code/plbound.py`](code/plbound.py): the module, and the script that
reproduces every number, table and figure of the paper in one run.

```bash
git clone https://github.com/syamsuddinexcellent/plbound.git
cd plbound/code && pip install -r requirements.txt
python plbound.py                              # tests and Sections 5-8, about 25 minutes
python plbound.py --bssr1 path/to/bssr1.zip    # first run only: converts the NIST scores for Section 8
```

```python
import numpy as np
import plbound as pc            # run from the code/ folder

S = lambda r: np.array([[1, r], [r, 1.0]])
model = pc.GaussianModel(mu0=[1, 2], mu1=[2, 0.5], S0=S(0.0), S1=S(0.99))
print(pc.performance_loss(model, alpha=0.05, eps=1e-10)["PL"])   # 0.2924439279 +/- 1.5e-10
```

| folder | content |
|---|---|
| [`code/`](code) | `plbound.py` (module and reproduction script) and its documentation, [`code/README.md`](code/README.md) |
| [`results/`](results) | every number reported in the paper as CSV files, with the figures (`results/figures`) and LaTeX tables (`results/tables`) |
| [`data/`](data) | instructions for obtaining the NIST BSSR1 scores used in Section 8 |

Requirements: Python 3.9 or later with numpy, scipy, pandas and matplotlib; R with CompQuadForm for
the benchmark. Released under the MIT licence.
