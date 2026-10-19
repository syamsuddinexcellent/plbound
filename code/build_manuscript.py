"""Assemble manuscript.tex from manuscript_src.tex and the result tables in ../results.

Every number in the generated tables is read from the CSV files written by the exp_*.py scripts.
Text blocks (<<NAME>>) are read from ../manuscript/blocks/NAME.tex when that file exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, MAN = ROOT / "results", ROOT / "manuscript"
LEV = ["N", "L", "M", "H", "P"]


def f(v, d):
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
    t = pd.read_csv(RES / "bivariate_table.csv")
    a5, a1 = t[t.alpha == 0.05].set_index(["rho0", "rho1"]), t[t.alpha == 0.01].set_index(["rho0", "rho1"])
    rows = []
    for i, r0 in enumerate(LEV):
        for r1 in LEV:
            x, y = a5.loc[(r0, r1)], a1.loc[(r0, r1)]
            rows.append(f"{r0} & {r1} & {f(x.AUC_f, 6)} & {f(x.AUC_nb, 6)} & {f(x.dAUC, 6)} & "
                        f"{f(x.TPR_f, 6)} & {f(x.TPR_nb, 6)} & {f(x.PL, 6)} & {f(y.PL, 5)}\\\\")
        if i < 4:
            rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Error-bounded values for the bivariate design at $\\alpha=0.05$, and "
            "$\\PL(0.01)$. $\\rho_0$ and $\\rho_1$ are the correlations of the negative and positive class "
            "(N, L, M, H, P $=0, 0.25, 0.5, 0.75, 0.99$). Half-widths of the enclosures are below "
            f"${sci_up(a5.PL_hw.max())}$ for $\\PL(0.05)$, below ${sci_up(max(a5.AUC_f_hw.max(), a5.AUC_nb_hw.max()))}$ for the AUCs and "
            f"below ${sci_up(a1.PL_hw.max())}$ for $\\PL(0.01)$}}\\label{{tab:biv}}\n"
            "\\footnotesize\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccccc@{}}\n\\toprule\n"
            "$\\rho_0$ & $\\rho_1$ & $\\AUC_f$ & $\\AUC_\\nb$ & $\\Delta\\AUC$ & $\\TPR_f(0.05)$ & "
            "$\\TPR_\\nb(0.05)$ & $\\PL(0.05)$ & $\\PL(0.01)$\\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_ar1():
    s = pd.read_csv(RES / "highdim_structured.csv")
    combos = [(0.0, 0.0), (0.3, 0.3), (0.6, 0.6), (0.9, 0.9), (0.0, 0.9), (0.9, 0.0)]
    h0 = " & ".join(f"{a:g}" for a, _ in combos)
    h1 = " & ".join(f"{b:g}" for _, b in combos)
    rows = []
    for p in sorted(s.p.unique()):
        vals = []
        for vr in (1.0, 2.0):
            d = s[(s.p == p) & (s.var_ratio == vr)]
            for a, b in combos:
                vals.append(f(d[(d.rho0 == a) & (d.rho1 == b)].PL05.iloc[0], 3))
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
    b = pd.read_csv(RES / "bench_cdf.csv")
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
            "$1-10^{-4}$). Errors are relative to the reference values described in Section~\\ref{sec:bench}; for Davies's "
            "method, failures are calls returning a non-zero fault indicator and are excluded from the error. "
            "Times in milliseconds; lim is the maximal number of integration terms (default $10^4$, otherwise $10^8$)}\\label{tab:benchcdf}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccc@{}}\n\\toprule\n"
            "method & setting & max.\\ error & within bound & failures & median (ms) & max.\\ (ms)\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def table_bench_pl():
    b = pd.read_csv(RES / "bench_pl.csv")
    rows = []
    for model, d in b.groupby("model", sort=False):
        for _, r in d.iterrows():
            if r.method == "error-bounded":
                eps = sci(float(r.setting.split("=")[1]), 0).replace("1\\times", "")
                rows.append(f"{model} & error-bounded, $\\varepsilon={eps}$ & {r.PL:.6f} & ${sci(r.precision)}$ & "
                            f"{r.seconds:.2f} & --- & --- \\\\")
            else:
                n = int(float(r.setting.split("=")[1]))
                mins = r["seconds_for_1e-4"] / 60
                days = r["seconds_for_1e-6"] / 86400
                rows.append(f"{model} & Monte Carlo, $N=10^{{{int(np.log10(n))}}}$ & {r.PL:.6f} & "
                            f"${sci(r.precision)}$ & {r.seconds:.3f} & {mins:.0f} & {days:,.0f} \\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{Computing $\\PL(0.05)$: error-bounded enclosures versus the Monte Carlo "
            "plug-in estimator. Precision is the half-width of the enclosure (error-bounded) or of a 95\\% interval "
            "from one replicate (Monte Carlo); time is per evaluation or per replicate in seconds. The last two "
            "columns give the CPU time Monte Carlo would need for a 95\\% half-width of $10^{-4}$ (minutes) "
            "and $10^{-6}$ (days), ignoring its $O(N^{-1})$ bias}"
            "\\label{tab:benchpl}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llccccc@{}}\n\\toprule\n"
            "model & method & $\\PL(0.05)$ & precision & time (s) & $10^{-4}$ (min) & $10^{-6}$ (days)\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


def _corr(text):
    vals = [float(v) for v in str(text).split(";")]
    return f"{vals[0]:.2f}" if len(vals) == 1 else f"{max(vals, key=abs):.2f}"


def table_bssr1():
    s = pd.read_csv(RES / "bssr1_summary.csv")
    full = pd.read_csv(RES / "bssr1_fullfit.csv").set_index("config")
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
            rows.append(f"{lead} & {a:g} & ${pp(r.PL_model_full)}$ & ${pp(r.PL_emp_mean)}$ & "
                        f"$[{pp(r.PL_emp_q025)}, {pp(r.PL_emp_q975)}]$ & ${pp(r.PL_pboot_mean)}$ "
                        f"$({100 * r.PL_pboot_sd:.2f})$\\\\")
        rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\caption{BSSR1 fusion; losses in percentage points. Correlations are those of "
            "the transformed genuine/impostor scores (largest absolute value for four matchers). Model: "
            "error-bounded $\\PL(\\alpha)$ of the Gaussian model fitted to all subjects (half-widths below "
            "$10^{-8}$). Held-out: mean empirical loss over 50 subject-disjoint 50/50 splits, with 2.5\\% and "
            "97.5\\% quantiles. Bootstrap: mean (standard deviation) of the same held-out loss for data "
            "simulated from the fitted Gaussian model}\\label{tab:bssr1}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccccc@{}}\n\\toprule\n"
            "fusion & set & genuine & corr. & FAR & model & held-out & range & bootstrap\\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\botrule\n\\end{tabular*}\n\\end{table}")


# Indonesian wording for the generated tables (internal translation of the manuscript)
TABLE_ID = [
    ("Error-bounded values for the bivariate design at $\\alpha=0.05$, and $\\PL(0.01)$.",
     "Nilai berbatas galat untuk desain bivariat pada $\\alpha=0.05$, serta $\\PL(0.01)$."),
    ("$\\rho_0$ and $\\rho_1$ are the correlations of the negative and positive class",
     "$\\rho_0$ dan $\\rho_1$ adalah korelasi kelas negatif dan kelas positif"),
    ("Half-widths of the enclosures are below", "Setengah lebar selang penutup di bawah"),
    ("for $\\PL(0.05)$, below", "untuk $\\PL(0.05)$, di bawah"),
    ("for the AUCs and below", "untuk AUC dan di bawah"),
    ("for $\\PL(0.01)$}", "untuk $\\PL(0.01)$}"),
    ("Error-bounded $\\PL(0.05)$ in the AR(1) design for selected $(\\rho_0,\\rho_1)$, with class-1 variance ratio 1 "
     "(independence score linear) and 2 (independence score quadratic). Half-widths are below",
     "$\\PL(0.05)$ berbatas galat pada desain AR(1) untuk $(\\rho_0,\\rho_1)$ terpilih, dengan rasio varians kelas 1 "
     "sebesar 1 (skor independensi linear) dan 2 (skor independensi kuadratik). Setengah lebar di bawah"),
    ("variance ratio 1", "rasio varians 1"), ("variance ratio 2", "rasio varians 2"),
    ("Distribution-function benchmark over 35 evaluation points (seven generalized chi-square laws at five quantile "
     "levels from $10^{-4}$ to $1-10^{-4}$). Errors are relative to the reference values described in Section~\\ref{sec:bench}; "
     "for Davies's method, failures are calls returning a non-zero fault indicator and are excluded from the error. "
     "Times in milliseconds; lim is the maximal number of integration terms (default $10^4$, otherwise $10^8$)",
     "Benchmark fungsi distribusi pada 35 titik evaluasi (tujuh hukum chi-square umum pada lima tingkat kuantil dari "
     "$10^{-4}$ hingga $1-10^{-4}$). Galat dihitung terhadap nilai rujukan yang dijelaskan di Bagian~\\ref{sec:bench}; untuk "
     "metode Davies, kegagalan adalah pemanggilan yang mengembalikan indikator kesalahan tidak nol dan tidak dihitung "
     "dalam galat. Waktu dalam milidetik; lim adalah banyak suku integrasi maksimum (bawaan $10^4$, selain itu $10^8$)"),
    ("method & setting & max.\\ error & within bound & failures & median (ms) & max.\\ (ms)",
     "metode & setelan & galat maks. & dalam batas & gagal & median (ms) & maks.\\ (ms)"),
    ("\nerror-bounded &", "\nberbatas galat &"), ("Lugannani--Rice", "Lugannani--Rice"),
    ("Computing $\\PL(0.05)$: error-bounded enclosures versus the Monte Carlo plug-in estimator. Precision is the "
     "half-width of the enclosure (error-bounded) or of a 95\\% interval from one replicate (Monte Carlo); time is per "
     "evaluation or per replicate in seconds. The last two columns give the CPU time Monte Carlo would need for a "
     "95\\% half-width of $10^{-4}$ (minutes) and $10^{-6}$ (days), ignoring its $O(N^{-1})$ bias",
     "Menghitung $\\PL(0.05)$: selang penutup berbatas galat dibanding penduga plug-in Monte Carlo. Presisi adalah "
     "setengah lebar selang penutup (berbatas galat) atau selang 95\\% dari satu ulangan (Monte Carlo); waktu per "
     "evaluasi atau per ulangan dalam detik. Dua kolom terakhir memberikan waktu CPU yang diperlukan Monte Carlo "
     "untuk setengah lebar 95\\% sebesar $10^{-4}$ (menit) dan $10^{-6}$ (hari), tanpa memperhitungkan bias "
     "$O(N^{-1})$-nya"),
    ("model & method & $\\PL(0.05)$ & precision & time (s) & $10^{-4}$ (min) & $10^{-6}$ (days)",
     "model & metode & $\\PL(0.05)$ & presisi & waktu (s) & $10^{-4}$ (menit) & $10^{-6}$ (hari)"),
    ("& error-bounded,", "& berbatas galat,"), ("random p=", "acak p="),
    ("BSSR1 fusion; losses in percentage points. Correlations are those of the transformed genuine/impostor scores "
     "(largest absolute value for four matchers). Model: error-bounded $\\PL(\\alpha)$ of the Gaussian model fitted to all "
     "subjects (half-widths below $10^{-8}$). Held-out: mean empirical loss over 50 subject-disjoint 50/50 splits, "
     "with 2.5\\% and 97.5\\% quantiles. Bootstrap: mean (standard deviation) of the same held-out loss for data "
     "simulated from the fitted Gaussian model",
     "Fusi BSSR1; kerugian dalam poin persen. Korelasi adalah korelasi skor genuine/impostor yang telah "
     "ditransformasi (nilai mutlak terbesar untuk empat matcher). Model: $\\PL(\\alpha)$ berbatas galat dari model "
     "Gaussian yang diduga dari semua subjek (setengah lebar di bawah $10^{-8}$). Held-out: rata-rata kerugian "
     "empiris atas 50 pembagian 50/50 yang disjoin menurut subjek, dengan kuantil 2.5\\% dan 97.5\\%. Bootstrap: "
     "rata-rata (simpangan baku) kerugian held-out yang sama untuk data yang disimulasikan dari model Gaussian "
     "hasil pendugaan"),
    ("fusion & set & genuine & corr. & FAR & model & held-out & range & bootstrap",
     "fusi & set & genuine & korelasi & FAR & model & held-out & rentang & bootstrap"),
    ("face C $+$ finger LI", "wajah C $+$ sidik jari LI"), ("face C $+$ G", "wajah C $+$ G"),
    ("finger LI $+$ RI", "sidik jari LI $+$ RI"), ("\nall four &", "\nkeempatnya &"),
]


def to_indonesian(tex):
    for en, idn in TABLE_ID:
        tex = tex.replace(en, idn)
    return tex


def inline_numbers(lang="en"):
    """Numbers quoted in the text blocks, read from the result files."""
    out = {}
    try:
        full = pd.read_csv(RES / "bssr1_fullfit.csv").set_index("config")
        summ = pd.read_csv(RES / "bssr1_summary.csv")
        splits = pd.read_csv(RES / "bssr1_splits.csv")
    except FileNotFoundError:
        return out
    key = {"FACE": "face C + face G", "FINGER": "finger LI + finger RI", "CROSS": "face C + finger LI",
           "ALL": "all four matchers"}
    for k, cfg in key.items():
        fr = full.loc[cfg]
        conj = " dan " if lang == "id" else " and "
        out[f"CORR_{k}"] = f"${float(str(fr.corr_gen).split(';')[0]):.2f}${conj}${float(str(fr.corr_imp).split(';')[0]):.2f}$"
        r = summ[(summ.config == cfg) & (summ.alpha == 0.001)].iloc[0]
        pl = r.PL_model_full
        out[f"PL_{k}"] = f"{pl:.4f}" if pl >= 1e-4 else f"${sci(pl, 0)}$"
        out[f"EMP_{k}"] = f"{-r.PL_emp_mean:.4f}"
        out[f"PB_{k}"] = f"${r.PL_pboot_mean:.4f}\\pm{r.PL_pboot_sd:.4f}$"
        out[f"TPR_MODEL_{k}"] = f"{r.TPR_f_model:.3f}"
        out[f"TPR_EMP_{k}"] = f"{r.TPR_f_emp:.3f}"
    return out


def main(lang="en"):
    suffix = "" if lang == "en" else f"_{lang}"
    src = (MAN / f"manuscript{suffix}_src.tex").read_text()
    gens = {"TABLE_BIV": table_biv, "TABLE_AR1": table_ar1, "TABLE_BENCH_CDF": table_bench_cdf,
            "TABLE_BENCH_PL": table_bench_pl, "TABLE_BSSR1": table_bssr1}
    blocks = MAN / ("blocks" if lang == "en" else f"blocks_{lang}")
    for _ in range(2):                          # blocks may contain table markers
        for name in list(gens) + [p.stem for p in blocks.glob("*.tex")] if blocks.exists() else gens:
            key = f"<<{name}>>"
            if key not in src:
                continue
            if (blocks / f"{name}.tex").exists():
                src = src.replace(key, (blocks / f"{name}.tex").read_text().strip())
            else:
                try:
                    table = gens[name]()
                    src = src.replace(key, to_indonesian(table) if lang == "id" else table)
                except (FileNotFoundError, KeyError) as e:
                    print("left placeholder", name, e)
    for k, v in inline_numbers(lang).items():
        src = src.replace(f"<<{k}>>", v)
    left = sorted(set(__import__("re").findall(r"<<([A-Z_0-9]+)>>", src)))
    if left:
        print("unresolved placeholders:", left)
    (MAN / f"manuscript{suffix}.tex").write_text(src)
    print(f"wrote manuscript{suffix}.tex")


if __name__ == "__main__":
    import sys
    main("id" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "id" else "en")
