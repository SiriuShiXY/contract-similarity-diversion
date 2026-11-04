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
root <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
variant <- Sys.getenv("SIM_VARIANT")       # "" = baseline; "vendor_delta" = robustness (build_cells.py)
data_dir <- if (variant == "") file.path(root, "data") else file.path(root, "data", "robustness", variant)
res_dir <- if (variant == "") file.path(root, "results") else file.path(root, "results", "robustness", variant)
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
