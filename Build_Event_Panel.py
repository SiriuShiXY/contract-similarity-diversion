"""Build listed-cell panels and control-adjusted counterfactuals."""

from __future__ import annotations

import sys


# Panel
"""
Existing-cell panel for the similarity analysis (similarity_diversion_design.pdf, Sections 6.4, 7.3, 8.1):
the grid of listed cells (zero volume where a listed cell does not trade), imputed characteristics for
cell-days without trades in the liquidity window, the nearest-expiry indicator N_{j,t}, Counterfactual 1
and the outcome y_{j,t}.

Grid.  An expiration is listed on day t if its first trade in the window (ListingDynamics
expirations.csv) is on or before t and its last trading session is on or after t.  A cell (product x
effective expiration weekday x DTE bucket x new/existing) exists on day t if one of its expirations is
listed; it is crossed with all five delta buckets (Deep OTM, OTM, ATM, ITM, Deep ITM) and call/put.  Sessions: pre, post and the 2022 transition days; early closes are dropped.

Characteristics of cell-days with no trade in 09:55-15:45 (V_char = 0): the cell's mean over its
pre-event cell-days on the same weekday, else over all its pre-event cell-days, else over all its
cell-days (flag x_imputed). Deep ITM cells use the same characteristic and imputation rules as other cells.

N_{j,t} = 1 for the existing cell that holds the nearest expiration of the pre-listing calendar (the
nearest existing expiration) when a listed new expiration expires earlier on day t (treated products).

Counterfactual 1 (design eq. 14, matching-DTE refinement):
    V_cf_{j,t} = Vbar_pre_{j,wd(t)} x V_ctrl_{t,b(j)} / Vbar_ctrl_pre_{wd(t),b(j)}
Vbar_pre_{j,wd}: mean of V_{j,t} over pre-event days on weekday wd on which j is listed (zeros included).
When j is listed on day t but on no pre-event day of that weekday -- an expiration moved by a holiday,
e.g. in 2018 the Friday series that expired on Thursday 2018-03-29 (Good Friday) sits in the Friday
0-DTE cell on a Thursday -- the cell's mean over all its pre-event days is used instead (flag wd_fallback;
design Section 7.4, holiday-shifted expirations treated explicitly).
V_ctrl_{t,b}: delta notional of the event's control ETFs (other index families) in DTE bucket b, on days
on which b is listed for them; its pre-event mean is taken over the same weekday.  When b is not
listed for the control ETFs on day t or on weekday wd(t) before the event (2018: SPY's Wednesday
expirations at 1 DTE on Tuesdays; QQQ and IWM had Fridays only), the ratio uses the control ETFs'
total volume (flag cf_fallback).
y_{j,t} = (V_{j,t} - V_cf_{j,t}) / Vbar_pre_j, Vbar_pre_j = mean over the pre-event days on which j is listed.
y_raw uses the weekday-matched baseline without control scaling (reference only).

2022: Cboe stopped listing new Monday and Wednesday series beyond 14 days for SPY, QQQ and IWM on
2022-11-10, a mechanical supply change (design Section 7.4).  Monday/Wednesday cells at 7-30 and 31+
DTE are flagged (cut_cell) and left out of the control volume.

Usage: python panel.py [event ...]
Output: data/<event>/existing_panel.parquet
"""
import sys
from pathlib import Path
import polars as pl
from Build_Panel import (
    BUILD_EVENTS,
    DELTA_BUCKETS,
    EVENT_SPEC,
    EV_DIR,
    OUT_DIR,
    last_session,
)

CFG = {
    ev: dict(treated=s["treated"], control=s["control"], cut=s["cut"])
    for ev, s in EVENT_SPEC.items()
}
DELTA = DELTA_BUCKETS
TYPES = ["C", "P"]
CELL = ["underlying_symbol", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
X = ["x_logtau", "x_absd"]


def dte_bucket(dte: pl.Expr) -> pl.Expr:
    return (
        pl.when(dte == 0)
        .then(pl.lit("D0"))
        .when(dte == 1)
        .then(pl.lit("D1"))
        .when(dte <= 6)
        .then(pl.lit("D2-6"))
        .when(dte <= 30)
        .then(pl.lit("D7-30"))
        .otherwise(pl.lit("D31+"))
    )


def listed_expirations(ev: str, days: pl.DataFrame) -> pl.DataFrame:
    expi = pl.read_csv(EV_DIR / ev / "expirations.csv", try_parse_dates=True).select(
        "underlying_symbol", "expiration", "first_trade", "eff_wd", "is_new"
    )
    expi = expi.with_columns(
        pl.col("expiration")
        .map_elements(last_session, return_dtype=pl.Date)
        .alias("last_session")
    )
    lst = (
        days.select("trade_date")
        .join(expi, how="cross")
        .filter(
            (pl.col("first_trade") <= pl.col("trade_date"))
            & (pl.col("last_session") >= pl.col("trade_date"))
        )
    )
    return lst.with_columns(
        (pl.col("last_session") - pl.col("trade_date")).dt.total_days().alias("dte")
    ).with_columns(dte_bucket(pl.col("dte")).alias("dte_bucket"))


def nearest_flags(lst: pl.DataFrame, treated: list[str]) -> pl.DataFrame:
    """Existing cell holding the nearest pre-listing expiration, flagged when a new expiration is nearer."""
    t = lst.filter(pl.col("underlying_symbol").is_in(treated))
    ex = (
        t.filter(~pl.col("is_new"))
        .group_by("underlying_symbol", "trade_date")
        .agg(
            pl.col("last_session").min().alias("e_star"),
            pl.col("eff_wd").sort_by("last_session").first(),
            pl.col("dte_bucket").sort_by("last_session").first(),
        )
    )
    nw = (
        t.filter(pl.col("is_new"))
        .group_by("underlying_symbol", "trade_date")
        .agg(pl.col("last_session").min().alias("n_star"))
    )
    f = ex.join(nw, on=["underlying_symbol", "trade_date"], how="left")
    return f.filter(
        pl.col("n_star").is_not_null() & (pl.col("n_star") < pl.col("e_star"))
    ).select(
        "underlying_symbol", "trade_date", "eff_wd", "dte_bucket", pl.lit(1).alias("N")
    )


def build(ev: str) -> pl.DataFrame:
    cfg = CFG[ev]
    days = (
        pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
        .filter(
            pl.col("has_file")
            & ~pl.col("early_close")
            & pl.col("period").is_in(["pre", "post", "transition"])
        )
        .select(
            "trade_date",
            pl.col("tau").alias("event_time"),
            "period",
            pl.col("trade_date").dt.weekday().alias("wd"),
        )
    )
    cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet").filter(
        ~pl.col("early_close")
    )
    lst = listed_expirations(ev, days)
    grid = (
        lst.select("underlying_symbol", "trade_date", "eff_wd", "is_new", "dte_bucket")
        .unique()
        .join(pl.DataFrame({"delta_bucket": DELTA}), how="cross")
        .join(pl.DataFrame({"option_type": TYPES}), how="cross")
        .join(days, on="trade_date")
    )
    keys = [
        "underlying_symbol",
        "trade_date",
        "eff_wd",
        "is_new",
        "dte_bucket",
        "delta_bucket",
        "option_type",
    ]
    orphan = cells.join(grid, on=keys, how="anti")
    assert (
        orphan.height == 0
    ), f"{ev}: {orphan.height} traded cell-days outside the listed grid"
    vol = ["V", "contracts", "premium", "trades", "V_char", "n_char"]
    p = grid.join(
        cells.select(keys + vol + X + ["sigbar"]), on=keys, how="left"
    ).with_columns([pl.col(c).fill_null(0) for c in vol])
    nf = nearest_flags(lst, cfg["treated"])
    p = p.join(
        nf, on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket"], how="left"
    ).with_columns(
        pl.when(~pl.col("is_new"))
        .then(pl.col("N").fill_null(0))
        .otherwise(0)
        .alias("N")
    )
    cut = (
        pl.lit(cfg["cut"])
        & pl.col("eff_wd").is_in([1, 3])
        & pl.col("dte_bucket").is_in(["D7-30", "D31+"])
    )
    p = p.with_columns(cut.alias("cut_cell"))
    ex = p.filter(~pl.col("is_new"))
    pre = ex.filter(pl.col("period") == "pre")
    has_x = pl.col("V_char") > 0
    m_wd = (
        pre.filter(has_x)
        .group_by(CELL + ["wd"])
        .agg([pl.col(c).mean().alias(f"{c}_wd") for c in X])
    )
    m_pre = (
        pre.filter(has_x)
        .group_by(CELL)
        .agg([pl.col(c).mean().alias(f"{c}_pre") for c in X])
    )
    m_all = (
        ex.filter(has_x)
        .group_by(CELL)
        .agg([pl.col(c).mean().alias(f"{c}_all") for c in X])
    )
    ex = (
        ex.join(m_wd, on=CELL + ["wd"], how="left")
        .join(m_pre, on=CELL, how="left")
        .join(m_all, on=CELL, how="left")
        .with_columns((pl.col("V_char") == 0).alias("x_imputed"))
        .with_columns(
            [
                pl.when(pl.col("x_imputed"))
                .then(pl.coalesce(f"{c}_wd", f"{c}_pre", f"{c}_all"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in X
            ]
        )
        .drop([f"{c}_{s}" for c in X for s in ("wd", "pre", "all")])
    )
    vpre = pre.group_by(CELL).agg(
        pl.col("V").mean().alias("Vpre"), pl.len().alias("n_pre_days")
    )
    vpre_wd = pre.group_by(CELL + ["wd"]).agg(pl.col("V").mean().alias("Vpre_wd"))
    ex = ex.join(vpre, on=CELL, how="left").join(vpre_wd, on=CELL + ["wd"], how="left")
    ctrl = ex.filter(
        pl.col("underlying_symbol").is_in(cfg["control"]) & ~pl.col("cut_cell")
    )
    vb = ctrl.group_by("trade_date", "period", "wd", "dte_bucket").agg(
        pl.col("V").sum().alias("Vc")
    )
    vb_pre = (
        vb.filter(pl.col("period") == "pre")
        .group_by("wd", "dte_bucket")
        .agg(pl.col("Vc").mean().alias("Vc_pre"))
    )
    vt = ctrl.group_by("trade_date", "period", "wd").agg(pl.col("V").sum().alias("Vct"))
    vt_pre = (
        vt.filter(pl.col("period") == "pre")
        .group_by("wd")
        .agg(pl.col("Vct").mean().alias("Vct_pre"))
    )
    a_b = vb.join(vb_pre, on=["wd", "dte_bucket"], how="left").select(
        "trade_date", "dte_bucket", (pl.col("Vc") / pl.col("Vc_pre")).alias("A_b")
    )
    a_t = vt.join(vt_pre, on="wd", how="left").select(
        "trade_date", (pl.col("Vct") / pl.col("Vct_pre")).alias("A_t")
    )
    ex = (
        ex.join(a_b, on=["trade_date", "dte_bucket"], how="left")
        .join(a_t, on="trade_date", how="left")
        .with_columns(pl.col("A_b").is_null().alias("cf_fallback"))
        .with_columns(
            pl.coalesce("A_b", "A_t").alias("A"),
            (pl.col("Vpre_wd").is_null() & pl.col("Vpre").is_not_null()).alias(
                "wd_fallback"
            ),
            pl.coalesce("Vpre_wd", "Vpre").alias("Vbase"),
        )
        .with_columns((pl.col("Vbase") * pl.col("A")).alias("V_cf"))
        .with_columns(
            ((pl.col("V") - pl.col("V_cf")) / pl.col("Vpre")).alias("y"),
            ((pl.col("V") - pl.col("Vbase")) / pl.col("Vpre")).alias("y_raw"),
        )
        .with_columns(
            pl.col("underlying_symbol").is_in(cfg["treated"]).alias("treated")
        )
    )
    return ex.sort(
        "underlying_symbol",
        "trade_date",
        "eff_wd",
        "dte_bucket",
        "delta_bucket",
        "option_type",
    )


def panel_main() -> None:
    for ev in sys.argv[1:] or BUILD_EVENTS:
        ex = build(ev)
        ex.write_parquet(OUT_DIR / ev / "existing_panel.parquet")
        post = ex.filter((pl.col("period") == "post") & pl.col("treated"))
        print(
            f"[{ev}] existing cell-days {ex.height:,}; treated post {post.height:,} (x imputed {post['x_imputed'].sum()}, cf fallback {post['cf_fallback'].sum()}, no weekday baseline {post['Vpre_wd'].is_null().sum()} (all-day fallback {post['wd_fallback'].sum()}), N=1 {post['N'].sum()}, cut {post['cut_cell'].sum()})",
            flush=True,
        )


# Stage orchestration
def run(paths, cfg, log):
    from Run_All import invoke

    invoke(paths, log, "Build_Event_Panel", "panel")
    return sorted(paths.data.glob("*/existing_panel.parquet"))


TASKS = {"panel": panel_main}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        TASKS[task]()
    else:
        from Run_All import main

        main("Build_Event_Panel")
