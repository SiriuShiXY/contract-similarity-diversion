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
root <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
variant <- Sys.getenv("SIM_VARIANT")       # "" = baseline; "vendor_delta" = robustness (build_cells.py)
data_dir <- if (variant == "") file.path(root, "data") else file.path(root, "data", "robustness", variant)
res_dir <- if (variant == "") file.path(root, "results") else file.path(root, "results", "robustness", variant)
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
