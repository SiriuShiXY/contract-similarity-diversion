# Distance-bin and reduced-form regressions. Paths are passed by Run_All.py.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop("Run through Run_All.py, or provide data_dir result_dir [bins|reduced_form]")
data_dir <- normalizePath(args[1], mustWork=TRUE)
res_dir <- normalizePath(args[2], mustWork=TRUE)
mode <- if (length(args) >= 3) args[3] else "reduced_form"
if (!mode %in% c("bins", "reduced_form")) stop("Unknown regression mode")
if (mode == "bins") {
## Distance-bin regressions on the Mahalanobis baseline (similarity_diversion_design.pdf, eq. 8).
##
##   (i)  y_jt = sum_b beta_b 1{d_jt in bin b} [+ gamma N_jt] + mu_c(j) + delta_t + e_jt
##        d_jt = Mahalanobis distance from existing cell j to the nearest new cell (same product, type,
##        day); six bins with equal shares of pre-event volume, farthest bin = reference.
##   (ii) y_jt = sum_b beta_b Z_jtb [+ gamma N_jt] + mu_c(j) + delta_t + e_jt         (addition)
##        Z_jtb = new-cell volume at Mahalanobis distance b from j / Vbar_pre_j, bins [0,.5), [.5,1),
##        [1,1.5), [1.5,2), [2,3), [3,inf).  beta_b = existing volume lost per unit of new volume at b.
##
## y_jt = (V_jt - V_cf_jt) / Vbar_pre_j (design eq. 15; counterfactual eq. 14, matching DTE bucket).
## Sample: existing cells of the treated ETFs, post-event days (2022: Mon/Wed 7-30 and 31+ DTE cells
## excluded).  mu_c(j): DTE bucket x delta bucket x call/put (x product x event); delta_t: product x day.
## Weights Vbar_pre_j; standard errors two-way clustered by cell and trading day (design Table 5 notes).
## Per event and pooled (event-specific fixed effects, pooled bins).
##
## Usage: Rscript bin_regression.R
## Output: results/bin_regression_coefs.csv, results/bin_regression_tables.txt,
##         results/bin_regression_diagnostics.csv

suppressMessages({library(fixest); library(data.table)})
d <- fread(file.path(data_dir, "reg_panel_all.csv"), colClasses = list(character = c("event", "trade_date")))
d <- d[in_reg == TRUE]
d[, `:=`(celltype = paste(event, underlying_symbol, dte_bucket, delta_bucket, option_type, sep = "|"),
         cell = paste(event, underlying_symbol, eff_wd, dte_bucket, delta_bucket, option_type, sep = "|"),
         pday = paste(event, underlying_symbol, trade_date, sep = "|"),
         day = paste(event, trade_date, sep = "|"))]
Z <- paste0("Z", 1:6)
samples <- c(sort(unique(d$event)), "pooled")

fits <- list(); coefs <- list(); diag <- list()
for (ev in samples) {
  s <- if (ev == "pooled") copy(d) else d[event == ev]
  s[, bin := if (ev == "pooled") bin_pool else bin_ev]
  fml <- list(
    i   = y ~ i(bin, ref = 6) | celltype + pday,
    iN  = y ~ i(bin, ref = 6) + N | celltype + pday,
    ii  = as.formula(paste("y ~", paste(Z, collapse = " + "), "| celltype + pday")),
    iiN = as.formula(paste("y ~", paste(c(Z, "N"), collapse = " + "), "| celltype + pday")))
  for (nm in names(fml)) {
    m <- feols(fml[[nm]], data = s, weights = ~Vpre, cluster = ~cell + day)
    fits[[paste(ev, nm)]] <- m
    ct <- as.data.table(coeftable(m), keep.rownames = "term")
    setnames(ct, c("term", "estimate", "se", "t", "p"))
    coefs[[paste(ev, nm)]] <- ct[, `:=`(sample = ev, model = nm, nobs = nobs(m),
                                        n_cells = uniqueN(s$cell), n_days = uniqueN(s$day),
                                        r2_within = fitstat(m, "wr2")[[1]])]
  }
  ## identifying variation left after the fixed effects
  w_d <- fitstat(feols(d_nearest ~ 1 | celltype + pday, s, weights = ~Vpre), "r2")[[1]]
  w_n <- fitstat(feols(N ~ 1 | celltype + pday, s, weights = ~Vpre), "r2")[[1]]
  mixed <- s[, .(n1 = sum(N == 1), n0 = sum(N == 0)), by = celltype][n1 > 0 & n0 > 0]
  diag[[ev]] <- data.table(sample = ev, nobs = nrow(s), n_cells = uniqueN(s$cell), n_days = uniqueN(s$day),
                           n_N1 = sum(s$N == 1),
                           N1_in_celltypes_with_both = sum(mixed$n1), N0_in_those_celltypes = sum(mixed$n0),
                           r2_fe_on_d_nearest = w_d, r2_fe_on_N = w_n)
}
coefs <- rbindlist(coefs)
fwrite(coefs, file.path(res_dir, "bin_regression_coefs.csv"))
fwrite(rbindlist(diag), file.path(res_dir, "bin_regression_diagnostics.csv"))

sink(file.path(res_dir, "bin_regression_tables.txt"))
cat("Version (i): bins of the distance to the nearest new cell (bin 1 = nearest, bin 6 = reference)\n")
print(etable(fits[paste(samples, rep(c("i", "iN"), each = length(samples)))], se.below = TRUE, digits = 3,
             fitstat = ~n + wr2))
cat("\nVersion (ii): new-cell volume by Mahalanobis distance (Z1 = [0,.5), ..., Z6 = [3,inf)) / Vbar_pre\n")
print(etable(fits[paste(samples, rep(c("ii", "iiN"), each = length(samples)))], se.below = TRUE, digits = 3,
             fitstat = ~n + wr2))
cat("\nIdentifying variation after the fixed effects\n")
print(rbindlist(diag))
sink()
cat("done:", nrow(coefs), "coefficients\n")
} else {
## Reduced form (similarity_diversion_design.pdf eqs. 17-18; table_shells Tables 3 and 5).
##
## Table 5, columns (1)-(3) and (5) -- eq. (17), treated products, post-event days (the eq. 8 sample):
##   y_jt = beta Sbar_jt + gamma N_jt + delta_t [+ mu_c(j)] + e_jt
##   y = (V - V_cf) / Vbar_pre (Counterfactual 1); Sbar = sum_k V_k exp(-theta_hat d_jk^2) / sum_k V_k over the new
##   cells of the same product, type and day, theta_hat from Table 6 col. 1 (structural.py, logistic lambda);
##   delta_t: product x day; mu_c: cell type (DTE bucket x delta bucket x call/put, x product x event) in col. 5.
##   (Column 5 follows the shell's notes: treated products only; the header's "control families: yes" is not used.)
## Table 5, column (4) -- eq. (18), stacked treated and control families, pre- and post-event (phantom.py):
##   log V_jt = alpha_j + mu_{p,t} + nu_{b,t} + beta Post x Treat x S_jt + gamma Post x Treat x N_jt
##              + S + N + Post x S + Treat x S + Post x N + Treat x N + e_jt
##   alpha_j cell, mu_{p,t} product x day (absorbs Post x Treat), nu_{b,t} DTE bucket x day (user's choice,
##   2026-10-29); S: similarity to the new cells on the calendar (actual for treated products after the event,
##   the same dates for the control products after the event, phantom before), one calendar-position profile
##   of characteristics and weights per event; N on the same calendar.
## Table 3 -- the same stacked panel, N only:  log V = ... + gamma Post x Treat x N + N + Post x N + Treat x N;
##   columns: all cells, ATM, OTM, ITM.
## Lower-order terms that the fixed effects absorb completely in a sample (N and Post x N in 2016 and 2022: the
## cell holding the nearest expiration is the only cell of its DTE bucket that day; R^2 on the fixed effects >=
## 0.9999) are left out of that sample's regression; the reported coefficients are identical either way.
## All: weights Vbar_pre_j; SEs two-way clustered by cell and trading day; 2022 Monday/Wednesday 7-30 and
## 31+ DTE cells excluded; log V needs V > 0 (zero-volume cell-days dropped, counted in the diagnostics).
## Pooled across events (event-specific fixed effects) and event by event (appendix A5).
## Robustness: Smax (eq. 16) and theta in {1, 2, 4} for (3); theta grid and distance shares for (4).
##
## Usage: Rscript reduced_form.R          (after similarity_exposure.py and phantom.py)
## Output: results/reduced_form_coefs.csv, results/reduced_form_tables.txt, results/reduced_form_diagnostics.csv

suppressMessages({library(fixest); library(data.table)})
setFixest_notes(FALSE)

## ---------------------------------------------------------------- Table 5, cols (1)-(3), (5)
r <- fread(file.path(data_dir, "reg_panel_rf.csv"), colClasses = list(character = c("event", "trade_date")))
r <- r[in_reg == TRUE]
r[, `:=`(celltype = paste(event, underlying_symbol, dte_bucket, delta_bucket, option_type, sep = "|"),
         cell = paste(event, underlying_symbol, eff_wd, dte_bucket, delta_bucket, option_type, sep = "|"),
         pday = paste(event, underlying_symbol, trade_date, sep = "|"),
         day = paste(event, trade_date, sep = "|"))]

## ---------------------------------------------------------------- stacked panel for (4) and Table 3
d <- fread(file.path(data_dir, "did18_all.csv"), colClasses = list(character = c("event", "trade_date")))
n_all <- nrow(d)
d <- d[cut_cell == FALSE & Vpre > 0]
zero <- d[, .(n_zero = sum(V <= 0), n = .N, w_zero = sum(Vpre[V <= 0]) / sum(Vpre)), by = .(event, treated)]
d <- d[V > 0]
d[, `:=`(logV = log(V), post = as.integer(period == "post"), treat = as.integer(treated),
         cell = paste(event, underlying_symbol, eff_wd, dte_bucket, delta_bucket, option_type, sep = "|"),
         pday = paste(event, underlying_symbol, trade_date, sep = "|"),
         bday = paste(event, dte_bucket, trade_date, sep = "|"),
         day = paste(event, trade_date, sep = "|"))]
mk <- function(dt, s) {
  dt[, `:=`(S = get(s), N = N_cal)]
  dt[, `:=`(PTS = post * treat * S, PTN = post * treat * N, PS = post * S, TS = treat * S, PN = post * N, TN = treat * N)]
  dt
}

## lower-order terms not absorbed by the stacked panel's fixed effects
free_terms <- function(dt, terms) {
  terms[sapply(terms, function(v) fitstat(feols(as.formula(paste(v, "~ 1 | cell + pday + bday")), dt,
                                                weights = ~Vpre), "r2")[[1]] < 0.9999)]
}
absorbed <- list()

samples <- c(sort(unique(r$event)), "pooled")
fits <- list(); coefs <- list()
keep <- function(m, sample, model, s, extra = list()) {
  ct <- as.data.table(coeftable(m), keep.rownames = "term")
  setnames(ct, c("term", "estimate", "se", "t", "p"))
  ct[, `:=`(sample = sample, model = model, nobs = nobs(m), r2 = fitstat(m, "r2")[[1]],
            wr2 = fitstat(m, "wr2")[[1]], n_cells = uniqueN(s$cell), n_days = uniqueN(s$day))]
  for (k in names(extra)) ct[, (k) := extra[[k]]]
  coefs[[paste(sample, model)]] <<- ct
  fits[[paste(sample, model)]] <<- m
}

for (ev in samples) {
  s <- if (ev == "pooled") copy(r) else r[event == ev]
  ## Table 5 (1)-(3), (5)
  keep(feols(y ~ N | pday, s, weights = ~Vpre, cluster = ~cell + day), ev, "t5_1", s)
  keep(feols(y ~ Sbar | pday, s, weights = ~Vpre, cluster = ~cell + day), ev, "t5_2", s)
  keep(feols(y ~ Sbar + N | pday, s, weights = ~Vpre, cluster = ~cell + day), ev, "t5_3", s)
  keep(feols(y ~ Sbar + N | celltype + pday, s, weights = ~Vpre, cluster = ~cell + day), ev, "t5_5", s)
  ## robustness of (3): max similarity, theta grid
  keep(feols(y ~ Smax + N | pday, s, weights = ~Vpre, cluster = ~cell + day), ev, "t5_3_max", s)
  for (t in c(1, 2, 4))
    keep(feols(as.formula(sprintf("y ~ Sbar_t%d + N | pday", t)), s, weights = ~Vpre, cluster = ~cell + day),
         ev, sprintf("t5_3_theta%d", t), s)

  ## stacked DiD: Table 5 (4) and Table 3
  dd <- if (ev == "pooled") copy(d) else d[event == ev]
  low_N <- free_terms(mk(copy(dd), "S_hat"), c("N", "PN", "TN"))
  absorbed[[ev]] <- data.table(sample = ev, absorbed = paste(setdiff(c("N", "PN", "TN"), low_N), collapse = ","))
  for (sv in c("S_hat", "S1", "S2", "S4")) {
    x <- mk(copy(dd), sv)
    nm <- if (sv == "S_hat") "t5_4" else paste0("t5_4_", sv)
    keep(feols(as.formula(paste("logV ~", paste(c("PTS", "PTN", "S", "PS", "TS", low_N), collapse = " + "),
                                "| cell + pday + bday")), x, weights = ~Vpre, cluster = ~cell + day), ev, nm, x)
  }
  ## distance shares instead of S (user's choice, 2026-10-29): share of the calendar weight within [0,.5), [.5,1), [1,2)
  x <- mk(copy(dd), "S_hat")
  for (i in 1:3) x[, (paste0("PTs", i)) := post * treat * get(paste0("s", i))]
  for (i in 1:3) x[, (paste0("Ps", i)) := post * get(paste0("s", i))]
  for (i in 1:3) x[, (paste0("Ts", i)) := treat * get(paste0("s", i))]
  keep(feols(as.formula(paste("logV ~", paste(c("PTs1", "PTs2", "PTs3", "PTN", "s1", "s2", "s3", "Ps1", "Ps2", "Ps3",
                                                "Ts1", "Ts2", "Ts3", low_N), collapse = " + "), "| cell + pday + bday")),
             x, weights = ~Vpre, cluster = ~cell + day), ev, "t5_4_shares", x)
  ## Table 3
  for (col in c("all", "ATM", "OTM", "ITM")) {
    x <- mk(copy(if (col == "all") dd else dd[delta_bucket == col]), "S_hat")
    lo <- free_terms(x, c("N", "PN", "TN"))
    keep(feols(as.formula(paste("logV ~", paste(c("PTN", lo), collapse = " + "), "| cell + pday + bday")), x,
               weights = ~Vpre, cluster = ~cell + day), ev, paste0("t3_", col), x,
         list(absorbed = paste(setdiff(c("N", "PN", "TN"), lo), collapse = ",")))
  }
}
coefs <- rbindlist(coefs, fill = TRUE)
fwrite(coefs, file.path(res_dir, "reduced_form_coefs.csv"))
zero[, n_did_all := n_all]
fwrite(zero, file.path(res_dir, "reduced_form_diagnostics.csv"))
fwrite(rbindlist(absorbed), file.path(res_dir, "reduced_form_absorbed.csv"))

sink(file.path(res_dir, "reduced_form_tables.txt"))
for (ev in samples) {
  cat("\n====", ev, "====\nTable 5 (1)-(3), (5)\n")
  print(etable(fits[paste(ev, c("t5_1", "t5_2", "t5_3", "t5_5"))], se.below = TRUE, digits = 3, fitstat = ~n + r2 + wr2))
  cat("\nTable 5 (4), eq. 18 (S at theta_hat, then theta = 1, 2, 4, distance shares)\n")
  print(etable(fits[paste(ev, c("t5_4", "t5_4_S1", "t5_4_S2", "t5_4_S4", "t5_4_shares"))], se.below = TRUE, digits = 3,
               fitstat = ~n + r2, keep = "PT"))
  cat("\nTable 3 (all, ATM, OTM, ITM)\n")
  print(etable(fits[paste(ev, c("t3_all", "t3_ATM", "t3_OTM", "t3_ITM"))], se.below = TRUE, digits = 3, fitstat = ~n + r2))
}
cat("\nZero-volume cell-days dropped from the log regressions\n")
print(zero)
sink()
cat("done:", nrow(coefs), "coefficients\n")
}
