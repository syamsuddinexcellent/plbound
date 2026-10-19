# Davies (1980) and Imhof (1961) CDF evaluations via CompQuadForm, for the benchmark in
# exp_benchmark.py.  Usage: Rscript benchmark_cdf.R cases.csv out.csv
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
