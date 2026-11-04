"""Estimate distances, distance bins, and fitted-theta similarity exposures."""

from __future__ import annotations

import sys


# Mahalanobis
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
import numpy as np
import polars as pl
from Build_Panel import EVENTS, OUT_DIR, RES_DIR

mahalanobis_X = ["x_logtau", "x_absd"]
Z_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
N_BINS = 6
J = [
    "underlying_symbol",
    "trade_date",
    "eff_wd",
    "dte_bucket",
    "delta_bucket",
    "option_type",
]


def shrunk_cov(X_: np.ndarray, w: np.ndarray) -> dict:
    w = w / w.sum()
    m = w @ X_
    Zc = X_ - m
    sd = np.sqrt(w @ Zc**2)
    Z = Zc / sd
    R = (w[:, None] * Z).T @ Z
    sw2 = float((w**2).sum())
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
    return dict(
        mean=m,
        sd=sd,
        R=R,
        delta=delta,
        R_lw=R_lw,
        S_lw=S_lw,
        S_inv=np.linalg.inv(S_lw),
        n=len(w),
        n_eff=1 / sw2,
        cond_R=float(np.linalg.cond(R_lw)),
        cond_S=float(np.linalg.cond(S_lw)),
    )


def sigmas(ev: str, cells: pl.DataFrame) -> tuple[dict, list[dict]]:
    pre = cells.filter(
        ~pl.col("is_new")
        & (pl.col("period") == "pre")
        & ~pl.col("early_close")
        & (pl.col("V_char") > 0)
    )
    out, rows = ({}, [])
    for (sym, typ), g in pre.group_by(["underlying_symbol", "option_type"]):
        s = shrunk_cov(g.select(mahalanobis_X).to_numpy(), g["V"].to_numpy())
        out[sym, typ] = s
        rows.append(
            dict(
                event=ev,
                underlying_symbol=sym,
                option_type=typ,
                n_cell_days=s["n"],
                n_eff=s["n_eff"],
                mean_logtau=s["mean"][0],
                mean_absd=s["mean"][1],
                sd_logtau=s["sd"][0],
                sd_absd=s["sd"][1],
                rho=s["R"][0, 1],
                lw_delta=s["delta"],
                rho_lw=s["R_lw"][0, 1],
                cond_R_lw=s["cond_R"],
                cond_Sigma_lw=s["cond_S"],
                m_per_logtau=float(np.sqrt(s["S_inv"][0, 0])),
                m_per_bucket=0.25 * float(np.sqrt(s["S_inv"][1, 1])),
            )
        )
    return (out, rows)


def distances(
    ev: str, cells: pl.DataFrame, panel: pl.DataFrame, sig: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    days = ["post", "transition"]
    ex = panel.filter(pl.col("treated") & pl.col("period").is_in(days))
    nw = cells.filter(
        pl.col("is_new")
        & pl.col("period").is_in(days)
        & ~pl.col("early_close")
        & (pl.col("V_char") > 0)
    ).select(
        "underlying_symbol",
        "trade_date",
        "option_type",
        pl.col("eff_wd").alias("eff_wd_k"),
        pl.col("dte_bucket").alias("dte_bucket_k"),
        pl.col("delta_bucket").alias("delta_bucket_k"),
        pl.col("V").alias("V_k"),
        pl.col("x_logtau").alias("x_logtau_k"),
        pl.col("x_absd").alias("x_absd_k"),
    )
    pairs = ex.select(J + mahalanobis_X + ["Vpre"]).join(
        nw, on=["underlying_symbol", "trade_date", "option_type"]
    )
    inv = pl.DataFrame(
        [
            dict(
                underlying_symbol=s,
                option_type=t,
                i11=v["S_inv"][0, 0],
                i12=v["S_inv"][0, 1],
                i22=v["S_inv"][1, 1],
            )
            for (s, t), v in sig.items()
        ]
    )
    a, b = (
        pl.col("x_logtau") - pl.col("x_logtau_k"),
        pl.col("x_absd") - pl.col("x_absd_k"),
    )
    pairs = (
        pairs.join(inv, on=["underlying_symbol", "option_type"])
        .with_columns(
            (pl.col("i11") * a**2 + 2 * pl.col("i12") * a * b + pl.col("i22") * b**2)
            .sqrt()
            .alias("d")
        )
        .drop("i11", "i12", "i22")
    )
    zb = pl.col("d").cut(
        Z_EDGES[1:-1], labels=[f"Z{i + 1}" for i in range(N_BINS)], left_closed=True
    )
    near = pairs.group_by(J).agg(
        pl.col("d").min().alias("d_nearest"),
        pl.col("V_k").sort_by("d").first().alias("Vk_nearest"),
        pl.concat_str(
            [
                pl.col("eff_wd_k").cast(pl.Utf8),
                pl.col("dte_bucket_k"),
                pl.col("delta_bucket_k"),
            ],
            separator="|",
        )
        .sort_by("d")
        .first()
        .alias("nearest_k"),
        pl.col("V_k").sum().alias("Vnew_type"),
        (pl.col("V_k") * pl.col("d") ** 2).sum().alias("_vd2"),
    )
    z = (
        pairs.with_columns(zb.alias("zbin"))
        .group_by(J + ["zbin"])
        .agg(pl.col("V_k").sum())
        .pivot(on="zbin", index=J, values="V_k")
    )
    for i in range(N_BINS):
        if f"Z{i + 1}" not in z.columns:
            z = z.with_columns(pl.lit(0.0).alias(f"Z{i + 1}"))
    near = (
        near.join(z, on=J, how="left")
        .with_columns([pl.col(f"Z{i + 1}").fill_null(0) for i in range(N_BINS)])
        .with_columns((pl.col("_vd2") / pl.col("Vnew_type")).sqrt().alias("d_volw"))
        .drop("_vd2")
    )
    return (pairs, near)


def weighted_bins(d: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Bin index 1..k with (about) equal shares of total weight, by increasing d."""
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    b = np.empty(len(d), dtype=int)
    b[order] = np.minimum(np.ceil(cum * k - 1e-12).astype(int), k).clip(min=1)
    return b


def mahalanobis_main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    srows, panels = ([], [])
    for ev in EVENTS:
        cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet")
        panel = pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet").filter(
            pl.col("delta_bucket") != "DITM"
        )
        sig, rows = sigmas(ev, cells)
        srows.extend(rows)
        pairs, near = distances(ev, cells, panel, sig)
        pairs.write_parquet(OUT_DIR / ev / "pairs.parquet")
        p = panel.filter(
            pl.col("treated") & pl.col("period").is_in(["post", "transition"])
        ).join(near, on=J, how="left")
        assert (
            p["d_nearest"].null_count() == 0
        ), f"{ev}: existing cell-days without a new cell of the same type"
        z = [f"Z{i + 1}" for i in range(N_BINS)]
        p = p.with_columns(
            [(pl.col(c) / pl.col("Vpre")).alias(c) for c in z]
            + [
                (pl.col("Vnew_type") / pl.col("Vpre")).alias("Vnew_type_rel"),
                pl.lit(ev).alias("event"),
            ]
        )
        p = p.with_columns(
            (
                (pl.col("period") == "post")
                & ~pl.col("cut_cell")
                & pl.col("y").is_not_null()
                & (pl.col("Vpre") > 0)
            ).alias("in_reg")
        )
        r = p.filter(pl.col("in_reg"))
        p = p.join(
            r.select(J).with_columns(
                pl.Series(
                    "bin_ev",
                    weighted_bins(
                        r["d_nearest"].to_numpy(), r["Vpre"].to_numpy(), N_BINS
                    ),
                )
            ),
            on=J,
            how="left",
        )
        panels.append(p)
        print(
            f"[{ev}] pairs {pairs.height:,}; regression cell-days {r.height:,}",
            flush=True,
        )
    allp = pl.concat(panels, how="diagonal_relaxed")
    r = allp.filter(pl.col("in_reg"))
    allp = allp.join(
        r.select(["event"] + J).with_columns(
            pl.Series(
                "bin_pool",
                weighted_bins(r["d_nearest"].to_numpy(), r["Vpre"].to_numpy(), N_BINS),
            )
        ),
        on=["event"] + J,
        how="left",
    )
    allp.drop("is_new").write_csv(OUT_DIR / "reg_panel_all.csv")
    s = pl.DataFrame(srows).sort("event", "underlying_symbol", "option_type")
    s.write_csv(RES_DIR / "sigma.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=3):
        print(s.drop("mean_logtau", "mean_absd", "cond_Sigma_lw"))


# Similarity Exposure
"""
Similarity of each existing cell-day to the new cells trading that day (similarity_diversion_design.pdf
eq. 16), the regressor of the reduced form eq. (17) / table_shells Table 5 columns (1)-(3) and (5).

    S_jk  = exp(-theta d_M(j,k)^2)                       (Table 5 notes: theta from Table 6, column 1)
    Sbar_jt = sum_k V_kt S_jk / sum_k V_kt              (volume-weighted, baseline)
    Smax_jt = max_k S_jk                                 (eq. 16 alternative, robustness)
k: new cells of the same product, option type and day with characteristics (mahalanobis.py pairs);
theta_hat = the logistic-lambda structural estimate (structural.py, version L).  Robustness: theta in
{1, 2, 4}, the grid of phantom.py.

Usage: python similarity_exposure.py          (after structural.py)
Output: data/reg_panel_rf.csv (reg_panel_all.csv + Sbar, Smax, Sbar_t1, Sbar_t2, Sbar_t4)
"""
import polars as pl
from Build_Panel import EVENTS, OUT_DIR

GRID = [1, 2, 4]


def theta_hat() -> float:
    e = pl.read_csv(RES_DIR / "structural_estimates.csv")
    return float(e.filter(pl.col("version") == "L")["theta"][0])


def exposure(ev: str, theta: float) -> pl.DataFrame:
    p = pl.read_parquet(OUT_DIR / ev / "pairs.parquet")
    d2 = pl.col("d") ** 2
    vk = pl.col("V_k")
    return (
        p.group_by(J)
        .agg(
            [
                ((vk * (-theta * d2).exp()).sum() / vk.sum()).alias("Sbar"),
                (-theta * d2).exp().max().alias("Smax"),
            ]
            + [
                ((vk * (-t * d2).exp()).sum() / vk.sum()).alias(f"Sbar_t{t}")
                for t in GRID
            ]
        )
        .with_columns(pl.lit(ev).alias("event"))
    )


def similarity_exposure_main() -> None:
    th = theta_hat()
    reg = pl.read_csv(
        OUT_DIR / "reg_panel_all.csv",
        schema_overrides={"event": pl.Utf8},
        try_parse_dates=True,
    )
    ex = pl.concat([exposure(ev, th) for ev in EVENTS])
    out = reg.join(ex, on=["event"] + J, how="left").with_columns(
        pl.lit(th).alias("theta_hat")
    )
    r = out.filter(pl.col("in_reg"))
    assert r["Sbar"].null_count() == 0
    out.write_csv(OUT_DIR / "reg_panel_rf.csv")
    w = pl.col("Vpre")
    print(f"theta_hat {th:.3f}; regression cell-days {r.height:,}")
    print(
        r.group_by("event")
        .agg(
            pl.len(),
            ((pl.col("Sbar") * w).sum() / w.sum()).alias("Sbar_wmean"),
            pl.col("Sbar").quantile(0.9).alias("Sbar_p90"),
            (pl.col("Sbar") > 0.01).mean().alias("share>0.01"),
            ((pl.col("Sbar_t1") * w).sum() / w.sum()).alias("Sbar_t1_wmean"),
        )
        .sort("event")
    )


# Phantom
"""
Phantom new cells and similarity exposure for the stacked difference-in-differences of
similarity_diversion_design.pdf eq. (18):

    log V_jt = alpha_j + mu_{p(j),t} + nu_{b(j),t} + beta Post x Treat x S_jt + gamma Post x Treat x N_jt
               + lower-order terms + e_jt

New-expiration calendar, per ETF and day (expiration level):
  treated ETF, post-event   the new expirations it actually lists that day (data/events/<event>/expirations.csv)
  control ETF, post-event   the same dates the treated ETF lists that day (2022: union of SPY and QQQ)
  control ETF that already lists the new weekdays (2021: SPY for QQQ; SPY and QQQ for IWM), before and after
                            the event: its own expirations on those weekdays up to the same horizon; they are left
                            out of its existing cells, as the treated ETF's new cells are (user's choice 2026-10-31)
  pre-event, both groups    phantom: sessions on the new weekday(s) from day t up to H(weekday of t) calendar
                            days ahead, H = the longest new expiration the treated ETF lists on that weekday
                            after the event; dates that are already an expiration of the ETF (Monday-holiday
                            Tuesdays in 2022) are skipped.
New cells (actual and phantom alike) = (expiration weekday, DTE bucket) of these expirations x delta bucket x
call/put.  Their characteristics and weights come from one calendar-position profile per event: the treated
ETFs' actual new cells after the event, by (weekday of t, expiration weekday, DTE bucket, delta bucket, type):
x = delta-notional-weighted mean of the cell-day characteristics, w = mean share of the day's same-type new
volume (fallbacks: pooled over the weekday of t, then by DTE bucket x delta bucket x type).  So similarity
depends on the calendar and contract terms only; realized new volume does not enter.

Per existing cell-day (Deep ITM cells, which have no characteristics, left out; so the eq. 18 panel and Table 3
exclude them) (characteristics as in panel.py, distance with the ETF's own Sigma_p from
mahalanobis.py, same option type):
  S_theta  = sum_k w_k exp(-theta d_jk^2) / sum_k w_k: S_hat at the structural estimate theta_hat (structural.py,
             version L; Table 5 notes), and theta in {1, 2, 4} (robustness grid)
  s1..s3   = share of w within d in [0, .5), [.5, 1), [1, 2)  (reference [2, inf))
  N        = 1 for the existing cell holding the nearest existing expiration when a new expiration on this
             calendar is nearer (equals panel.py's N for the treated ETF after the event).

Usage: python phantom.py          (after structural.py)
Output: data/<event>/did_panel.parquet, data/did18_all.csv
"""
from datetime import timedelta
import numpy as np
import polars as pl
from Build_Panel import (
    EVENT_SPEC,
    EVENTS,
    EV_DIR,
    OUT_DIR,
    SESSION_IDX,
    effective_weekday,
    last_session,
)
from Build_Event_Panel import CFG, dte_bucket, listed_expirations

NEW_WD = {ev: s["new_wd"] for ev, s in EVENT_SPEC.items()}
OWN = {ev: s["control_treated"] for ev, s in EVENT_SPEC.items()}
THETAS = [1, 2, 4]
SHARE_EDGES = [0.5, 1.0, 2.0]
phantom_X = ["x_logtau", "x_absd"]


def theta_from_structural() -> float:
    e = pl.read_csv(RES_DIR / "structural_estimates.csv")
    return float(e.filter(pl.col("version") == "L")["theta"][0])


def new_calendar(ev: str, days: pl.DataFrame, lst: pl.DataFrame) -> pl.DataFrame:
    cfg, new_wd = (CFG[ev], NEW_WD[ev])
    treated, control = (cfg["treated"], cfg["control"])
    act = (
        lst.filter(pl.col("is_new"))
        .join(days.select("trade_date", "period", "wd"), on="trade_date")
        .filter(pl.col("period") == "post")
    )
    horizon = dict(act.group_by("wd").agg(pl.col("dte").max()).iter_rows())
    existing = {
        s: set(
            lst.filter((pl.col("underlying_symbol") == s) & ~pl.col("is_new"))[
                "last_session"
            ].to_list()
        )
        for s in treated + control
    }
    own = OWN[ev]
    rows = []
    for sym, t, e in act.select(
        "underlying_symbol", "trade_date", "last_session"
    ).iter_rows():
        rows.append((sym, t, e, "actual"))
    mirror = act.select("trade_date", "last_session").unique()
    for sym in control:
        if sym not in own:
            rows += [(sym, t, e, "mirror") for t, e in mirror.iter_rows()]
    for t, wd in (
        days.filter(pl.col("period") == "pre").select("trade_date", "wd").iter_rows()
    ):
        for k in range(horizon[wd] + 1):
            d = t + timedelta(k)
            if d.isoweekday() in new_wd and d in SESSION_IDX:
                rows += [
                    (sym, t, d, "phantom")
                    for sym in treated + control
                    if sym not in own and d not in existing[sym]
                ]
    hz = max(horizon.values())
    o = lst.filter(
        pl.col("underlying_symbol").is_in(own) & pl.col("eff_wd").is_in(new_wd)
    ).join(days.select("trade_date", "wd"), on="trade_date")
    for sym, t, wd, e, dte in o.select(
        "underlying_symbol", "trade_date", "wd", "last_session", "dte"
    ).iter_rows():
        if dte <= horizon.get(wd, hz):
            rows.append((sym, t, e, "own"))
    cal = pl.DataFrame(
        rows,
        schema=["underlying_symbol", "trade_date", "last_session", "source"],
        orient="row",
    )
    return (
        cal.unique(["underlying_symbol", "trade_date", "last_session"])
        .with_columns(
            # Match the effective-weekday key used by the cell profiles.
            pl.col("last_session")
            .map_elements(effective_weekday, return_dtype=pl.Int64)
            .alias("eff_wd_k"),
            (pl.col("last_session") - pl.col("trade_date"))
            .dt.total_days()
            .alias("dte_k"),
        )
        .with_columns(dte_bucket(pl.col("dte_k")).alias("dte_bucket_k"))
    )


def profile(ev: str, cells: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    nw = (
        cells.filter(
            pl.col("is_new")
            & (pl.col("period") == "post")
            & ~pl.col("early_close")
            & (pl.col("V_char") > 0)
        )
        .with_columns(
            pl.col("trade_date").dt.weekday().alias("wd"),
            (
                pl.col("V")
                / pl.col("V")
                .sum()
                .over("underlying_symbol", "trade_date", "option_type")
            ).alias("share"),
        )
        .rename(
            {
                "eff_wd": "eff_wd_k",
                "dte_bucket": "dte_bucket_k",
                "delta_bucket": "delta_bucket_k",
            }
        )
    )
    out = []
    for key in (
        ["wd", "eff_wd_k", "dte_bucket_k"],
        ["eff_wd_k", "dte_bucket_k"],
        ["dte_bucket_k"],
    ):
        k = key + ["delta_bucket_k", "option_type"]
        out.append(
            nw.group_by(k).agg(
                [
                    ((pl.col(c) * pl.col("V")).sum() / pl.col("V").sum()).alias(
                        f"{c}_k"
                    )
                    for c in phantom_X
                ]
                + [pl.col("share").mean().alias("w_k")]
            )
        )
    return tuple(out)


def build(ev: str, sig: pl.DataFrame, theta_hat: float) -> pl.DataFrame:
    cfg = CFG[ev]
    days = (
        pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
        .filter(
            pl.col("has_file")
            & ~pl.col("early_close")
            & pl.col("period").is_in(["pre", "post"])
        )
        .select("trade_date", "period", pl.col("trade_date").dt.weekday().alias("wd"))
    )
    lst = listed_expirations(ev, days)
    lst = lst.filter(pl.col("underlying_symbol").is_in(cfg["treated"] + cfg["control"]))
    own_new = pl.col("underlying_symbol").is_in(OWN[ev]) & pl.col("eff_wd").is_in(
        NEW_WD[ev]
    )
    cal = new_calendar(ev, days, lst)
    cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet")
    p1, p2, p3 = profile(ev, cells)
    e_star = (
        lst.filter(~pl.col("is_new") & ~own_new)
        .group_by("underlying_symbol", "trade_date")
        .agg(
            pl.col("last_session").min().alias("e_star"),
            pl.col("eff_wd").sort_by("last_session").first(),
            pl.col("dte_bucket").sort_by("last_session").first(),
        )
    )
    n_star = cal.group_by("underlying_symbol", "trade_date").agg(
        pl.col("last_session").min().alias("n_star")
    )
    nflag = (
        e_star.join(n_star, on=["underlying_symbol", "trade_date"])
        .filter(pl.col("n_star") < pl.col("e_star"))
        .select(
            "underlying_symbol",
            "trade_date",
            "eff_wd",
            "dte_bucket",
            pl.lit(1).alias("N_cal"),
        )
    )
    nc = (
        cal.select("underlying_symbol", "trade_date", "eff_wd_k", "dte_bucket_k")
        .unique()
        .join(
            pl.DataFrame({"delta_bucket_k": ["DOTM", "OTM", "ATM", "ITM"]}), how="cross"
        )
        .join(pl.DataFrame({"option_type": ["C", "P"]}), how="cross")
        .with_columns(pl.col("trade_date").dt.weekday().alias("wd"))
    )
    cols = [f"{c}_k" for c in phantom_X] + ["w_k"]
    k1, k2, k3 = (
        ["wd", "eff_wd_k", "dte_bucket_k"],
        ["eff_wd_k", "dte_bucket_k"],
        ["dte_bucket_k"],
    )
    nc = (
        nc.join(p1, on=k1 + ["delta_bucket_k", "option_type"], how="left")
        .join(
            p2.rename({c: f"{c}_2" for c in cols}),
            on=k2 + ["delta_bucket_k", "option_type"],
            how="left",
        )
        .join(
            p3.rename({c: f"{c}_3" for c in cols}),
            on=k3 + ["delta_bucket_k", "option_type"],
            how="left",
        )
        .with_columns(pl.col("w_k").is_null().alias("profile_fallback"))
        .with_columns([pl.coalesce(c, f"{c}_2", f"{c}_3").alias(c) for c in cols])
        .drop([f"{c}_{i}" for c in cols for i in (2, 3)])
    )
    assert nc["w_k"].null_count() == 0, f"{ev}: new cells without a profile"
    ex = pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet").filter(
        pl.col("period").is_in(["pre", "post"])
        & pl.col("underlying_symbol").is_in(cfg["treated"] + cfg["control"])
        & ~own_new
        & (pl.col("delta_bucket") != "DITM")
    )
    ex = ex.join(
        nflag,
        on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket"],
        how="left",
    ).with_columns(pl.col("N_cal").fill_null(0))
    pairs = ex.select(
        "underlying_symbol",
        "trade_date",
        "eff_wd",
        "dte_bucket",
        "delta_bucket",
        "option_type",
        *phantom_X,
    ).join(nc, on=["underlying_symbol", "trade_date", "option_type"])
    inv = sig.filter(pl.col("event") == ev).select(
        "underlying_symbol", "option_type", "sd_logtau", "sd_absd", "rho_lw"
    )
    a, b = (
        pl.col("x_logtau") - pl.col("x_logtau_k"),
        pl.col("x_absd") - pl.col("x_absd_k"),
    )
    det = 1 - pl.col("rho_lw") ** 2
    za, zb = (a / pl.col("sd_logtau"), b / pl.col("sd_absd"))
    pairs = pairs.join(inv, on=["underlying_symbol", "option_type"]).with_columns(
        ((za**2 - 2 * pl.col("rho_lw") * za * zb + zb**2) / det).alias("d2")
    )
    J = [
        "underlying_symbol",
        "trade_date",
        "eff_wd",
        "dte_bucket",
        "delta_bucket",
        "option_type",
    ]
    w = pl.col("w_k")
    agg = pairs.group_by(J).agg(
        [((w * (-theta_hat * pl.col("d2")).exp()).sum() / w.sum()).alias("S_hat")]
        + [
            ((w * (-th * pl.col("d2")).exp()).sum() / w.sum()).alias(f"S{th}")
            for th in THETAS
        ]
        + [
            (
                w.filter(pl.col("d2").sqrt().is_between(lo, hi, closed="left")).sum()
                / w.sum()
            ).alias(f"s{i + 1}")
            for i, (lo, hi) in enumerate(zip([0.0] + SHARE_EDGES[:-1], SHARE_EDGES))
        ]
        + [
            pl.col("d2").min().sqrt().alias("d_min"),
            pl.col("profile_fallback").any().alias("profile_fallback"),
        ]
    )
    out = ex.join(agg, on=J, how="left").with_columns(pl.lit(ev).alias("event"))
    assert (
        out["S1"].null_count() == 0
    ), f"{ev}: existing cell-days without a new cell of the same type"
    chk = out.filter(pl.col("treated") & (pl.col("period") == "post"))
    assert (chk["N"] == chk["N_cal"]).all(), f"{ev}: calendar N differs from panel.py N"
    src = cal.group_by("source").len().sort("source").rows()
    print(
        f"[{ev}] new-expiration calendar rows by source {src}; new cells {nc.height:,} (profile fallback {int(nc['profile_fallback'].sum())}); existing cell-days {out.height:,}",
        flush=True,
    )
    return out


def phantom_main() -> None:
    sig = pl.read_csv(RES_DIR / "sigma.csv", schema_overrides={"event": pl.Utf8})
    theta_hat = theta_from_structural()
    parts = []
    for ev in EVENTS:
        out = build(ev, sig, theta_hat)
        out.write_parquet(OUT_DIR / ev / "did_panel.parquet")
        parts.append(out)
    keep = (
        [
            "event",
            "underlying_symbol",
            "treated",
            "trade_date",
            "period",
            "wd",
            "eff_wd",
            "dte_bucket",
            "delta_bucket",
            "option_type",
            "cut_cell",
            "V",
            "Vpre",
            "N_cal",
            "d_min",
            "profile_fallback",
            "S_hat",
        ]
        + [f"S{t}" for t in THETAS]
        + [f"s{i + 1}" for i in range(len(SHARE_EDGES))]
    )
    pl.concat([p.select(keep) for p in parts]).write_csv(OUT_DIR / "did18_all.csv")


# Stage orchestration
def run(paths, cfg, log):
    from Run_All import invoke, invoke_r

    invoke(paths, log, "Measure_Similarity", "mahalanobis")
    invoke_r(paths, log, "bins")
    return (
        [paths.data / "reg_panel_all.csv", paths.results / "sigma.csv"]
        + sorted(paths.data.glob("*/pairs.parquet"))
        + sorted(paths.results.glob("bin_regression_*"))
    )


TASKS = {
    "mahalanobis": mahalanobis_main,
    "similarity_exposure": similarity_exposure_main,
    "phantom": phantom_main,
}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        TASKS[task]()
    else:
        from Run_All import main

        main("Measure_Similarity")
