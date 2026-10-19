# PL(alpha) from Davies's (1980) algorithm combined with the propagation of Proposition 6, for
# the benchmark in exp_benchmark.py (part C).  Usage: Rscript benchmark_pl_davies.R laws.csv out.csv [curve]
#
# Each distribution function is evaluated by CompQuadForm::davies with error at most acc; the
# threshold is bracketed as in Proposition 6(a) and the true positive rate enclosed as in 6(b).
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
  # Proposition 6(a) needs F_0(t_hi) - e_0 > 1 - alpha, impossible when the error bound reaches alpha
  if (1 - alpha + c0$e >= 1) stop("acc >= alpha: the threshold cannot be bracketed")
  m <- L0$mean; s <- L0$sd
  t_star <- uniroot(function(t) c0$F(t) - (1 - alpha), c(m - 2 * s, m + 2 * s), extendInt = "upX",
                    tol = 1e-13 * (abs(m) + s), maxiter = 1000)$root
  eta <- 1e-4 * s
  dens <- max((c0$F(t_star + eta) - c0$F(t_star - eta)) / (2 * eta), 1e-300)
  d <- max(2 * c0$e / dens, 1e-14 * (1 + abs(t_star)))
  for (k in 0:200) {   # Proposition 6(a)
    lo <- t_star - d; hi <- t_star + d
    if (c0$F(lo) + c0$e < 1 - alpha && c0$F(hi) - c0$e > 1 - alpha) break
    if (k == 200) stop("bracketing failed")
    d <- 2 * d
  }
  c(1 - c1$F(hi) - c1$e, 1 - c1$F(lo) + c1$e)   # Proposition 6(b)
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
