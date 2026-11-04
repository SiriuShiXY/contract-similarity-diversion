"""
Mahalanobis-distance baseline (similarity_diversion_design.pdf, Sections 4 and 11.1) and the regressors
of the distance-bin regression (design eq. 8, plus an addition).

Sigma_p, per event x product (treated and control ETFs) x call/put: existing cells' pre-event cell-days
with characteristics (trades in 09:55-15:45; early closes excluded), weighted by delta notional V.
Weighted mean and covariance of x = (log trading hours, |delta| at the common volatility); the
characteristics are standardized by their weighted s.d. and the correlation matrix is shrunk toward the
identity with a Ledoit-Wolf-type intensity, computed with the Schafer-Strimmer formula using the
observation weights (w normalized to sum to one, u = z_i z_j):
    Var(r_ij) = sum(w^2) x sum w (u - r_ij)^2 / (1 - sum(w^2)),   delta = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2
    R_LW = (1 - delta) R + delta I,   Sigma_LW = D^1/2 R_LW D^1/2.
Held fixed after the event.  The condition number is reported for R_LW (scale-free); the condition
number of Sigma itself mostly reflects the units of the two characteristics.

Distances: d_M(j,k,t) = sqrt((x_j - x_k)' Sigma_LW^-1 (x_j - x_k)) for every listed existing cell j of a
treated product (characteristics imputed where it did not trade in the liquidity window, see panel.py)
and every new cell k with trades in the liquidity window, same product, option type and day
(post-event and 2022 transition days).

Regressors per existing cell-day:
  d_nearest  distance to the nearest new cell (design eq. 8).  Version (i) bins it into six bins with
             equal shares of pre-event volume: quantiles of d_nearest weighted by Vbar_pre_j over the
             regression sample, per event (bin_ev) and pooled across events (bin_pool).
  Z1..Z6     new-cell volume located at distance b from j, sum_k V_k 1{d_jk in b} / Vbar_pre_j, with
             bins [0, .5), [.5, 1), [1, 1.5), [1.5, 2), [2, 3), [3, inf) in M-units (version ii, an addition
             to the design).
Regression sample (in_reg): existing cells of the treated ETFs on post-event days with a counterfactual;
2022 Monday/Wednesday cells at 7-30 and 31+ DTE excluded (listing cut, design Section 7.4).  Deep ITM cells
have no characteristics (design 3.6), so they have no distances and are not in the sample.

Usage: python mahalanobis.py
Output: results/sigma.csv, data/<event>/pairs.parquet, data/reg_panel_all.csv
"""
from __future__ import annotations

import numpy as np
import polars as pl

from build_cells import EVENTS, OUT_DIR, RES_DIR

X = ["x_logtau", "x_absd"]
Z_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
N_BINS = 6
J = ["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]


def shrunk_cov(X_: np.ndarray, w: np.ndarray) -> dict:
    w = w / w.sum()
    m = w @ X_
    Zc = X_ - m
    sd = np.sqrt(w @ Zc ** 2)
    Z = Zc / sd
    R = (w[:, None] * Z).T @ Z
    sw2 = float((w ** 2).sum())
    num = den = 0.0
    M = X_.shape[1]
    for i in range(M):
        for j in range(M):
            if i != j:
                u = Z[:, i] * Z[:, j]
                num += sw2 * float(w @ (u - R[i, j]) ** 2) / (1 - sw2)
                den += R[i, j] ** 2
    delta = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0
    R_lw = (1 - delta) * R + delta * np.eye(M)
    S_lw = np.diag(sd) @ R_lw @ np.diag(sd)
    return dict(mean=m, sd=sd, R=R, delta=delta, R_lw=R_lw, S_lw=S_lw, S_inv=np.linalg.inv(S_lw),
                n=len(w), n_eff=1 / sw2, cond_R=float(np.linalg.cond(R_lw)),
                cond_S=float(np.linalg.cond(S_lw)))


def sigmas(ev: str, cells: pl.DataFrame) -> tuple[dict, list[dict]]:
    pre = cells.filter(~pl.col("is_new") & (pl.col("period") == "pre") & ~pl.col("early_close") & (pl.col("V_char") > 0))
    out, rows = {}, []
    for (sym, typ), g in pre.group_by(["underlying_symbol", "option_type"]):
        s = shrunk_cov(g.select(X).to_numpy(), g["V"].to_numpy())
        out[(sym, typ)] = s
        rows.append(dict(event=ev, underlying_symbol=sym, option_type=typ, n_cell_days=s["n"], n_eff=s["n_eff"],
                         mean_logtau=s["mean"][0], mean_absd=s["mean"][1], sd_logtau=s["sd"][0], sd_absd=s["sd"][1],
                         rho=s["R"][0, 1], lw_delta=s["delta"], rho_lw=s["R_lw"][0, 1], cond_R_lw=s["cond_R"],
                         cond_Sigma_lw=s["cond_S"],
                         ## M-units of one unit of log tau and of one delta bucket (+0.25), other fixed
                         m_per_logtau=float(np.sqrt(s["S_inv"][0, 0])),
                         m_per_bucket=0.25 * float(np.sqrt(s["S_inv"][1, 1]))))
    return out, rows


def distances(ev: str, cells: pl.DataFrame, panel: pl.DataFrame, sig: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    days = ["post", "transition"]
    ex = panel.filter(pl.col("treated") & pl.col("period").is_in(days))
    nw = (cells.filter(pl.col("is_new") & pl.col("period").is_in(days) & ~pl.col("early_close") & (pl.col("V_char") > 0))
          .select("underlying_symbol", "trade_date", "option_type", pl.col("eff_wd").alias("eff_wd_k"),
                  pl.col("dte_bucket").alias("dte_bucket_k"), pl.col("delta_bucket").alias("delta_bucket_k"),
                  pl.col("V").alias("V_k"), pl.col("x_logtau").alias("x_logtau_k"), pl.col("x_absd").alias("x_absd_k")))
    pairs = ex.select(J + X + ["Vpre"]).join(nw, on=["underlying_symbol", "trade_date", "option_type"])
    inv = pl.DataFrame([dict(underlying_symbol=s, option_type=t, i11=v["S_inv"][0, 0], i12=v["S_inv"][0, 1],
                             i22=v["S_inv"][1, 1]) for (s, t), v in sig.items()])
    a, b = pl.col("x_logtau") - pl.col("x_logtau_k"), pl.col("x_absd") - pl.col("x_absd_k")
    pairs = (pairs.join(inv, on=["underlying_symbol", "option_type"])
             .with_columns((pl.col("i11") * a ** 2 + 2 * pl.col("i12") * a * b + pl.col("i22") * b ** 2).sqrt().alias("d"))
             .drop("i11", "i12", "i22"))
    zb = pl.col("d").cut(Z_EDGES[1:-1], labels=[f"Z{i + 1}" for i in range(N_BINS)], left_closed=True)
    near = (pairs.group_by(J).agg(
        pl.col("d").min().alias("d_nearest"),
        pl.col("V_k").sort_by("d").first().alias("Vk_nearest"),
        pl.concat_str([pl.col("eff_wd_k").cast(pl.Utf8), pl.col("dte_bucket_k"), pl.col("delta_bucket_k")], separator="|")
        .sort_by("d").first().alias("nearest_k"),
        pl.col("V_k").sum().alias("Vnew_type"),
        (pl.col("V_k") * pl.col("d") ** 2).sum().alias("_vd2")))
    z = (pairs.with_columns(zb.alias("zbin")).group_by(J + ["zbin"]).agg(pl.col("V_k").sum())
         .pivot(on="zbin", index=J, values="V_k"))
    for i in range(N_BINS):
        if f"Z{i + 1}" not in z.columns:
            z = z.with_columns(pl.lit(0.0).alias(f"Z{i + 1}"))
    near = (near.join(z, on=J, how="left")
            .with_columns([pl.col(f"Z{i + 1}").fill_null(0) for i in range(N_BINS)])
            .with_columns((pl.col("_vd2") / pl.col("Vnew_type")).sqrt().alias("d_volw")).drop("_vd2"))
    return pairs, near


def weighted_bins(d: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Bin index 1..k with (about) equal shares of total weight, by increasing d."""
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    b = np.empty(len(d), dtype=int)
    b[order] = np.minimum(np.ceil(cum * k - 1e-12).astype(int), k).clip(min=1)
    return b


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    srows, panels = [], []
    for ev in EVENTS:
        cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet")
        panel = pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet").filter(pl.col("delta_bucket") != "DITM")
        sig, rows = sigmas(ev, cells)
        srows.extend(rows)
        pairs, near = distances(ev, cells, panel, sig)
        pairs.write_parquet(OUT_DIR / ev / "pairs.parquet")
        p = (panel.filter(pl.col("treated") & pl.col("period").is_in(["post", "transition"]))
             .join(near, on=J, how="left"))
        assert p["d_nearest"].null_count() == 0, f"{ev}: existing cell-days without a new cell of the same type"
        z = [f"Z{i + 1}" for i in range(N_BINS)]
        p = p.with_columns([(pl.col(c) / pl.col("Vpre")).alias(c) for c in z] +
                           [(pl.col("Vnew_type") / pl.col("Vpre")).alias("Vnew_type_rel"), pl.lit(ev).alias("event")])
        p = p.with_columns(((pl.col("period") == "post") & ~pl.col("cut_cell") & pl.col("y").is_not_null()
                            & (pl.col("Vpre") > 0)).alias("in_reg"))
        r = p.filter(pl.col("in_reg"))
        p = p.join(r.select(J).with_columns(pl.Series("bin_ev", weighted_bins(r["d_nearest"].to_numpy(),
                                                                                  r["Vpre"].to_numpy(), N_BINS))),
                   on=J, how="left")
        panels.append(p)
        print(f"[{ev}] pairs {pairs.height:,}; regression cell-days {r.height:,}", flush=True)
    allp = pl.concat(panels, how="diagonal_relaxed")
    r = allp.filter(pl.col("in_reg"))
    allp = allp.join(r.select(["event"] + J).with_columns(
        pl.Series("bin_pool", weighted_bins(r["d_nearest"].to_numpy(), r["Vpre"].to_numpy(), N_BINS))),
        on=["event"] + J, how="left")
    allp.drop("is_new").write_csv(OUT_DIR / "reg_panel_all.csv")
    s = pl.DataFrame(srows).sort("event", "underlying_symbol", "option_type")
    s.write_csv(RES_DIR / "sigma.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=3):
        print(s.drop("mean_logtau", "mean_absd", "cond_Sigma_lw"))


if __name__ == "__main__":
    main()
