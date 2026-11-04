"""
Randomization inference for lambda with placebo event dates (similarity_diversion_design.pdf Section 8.4;
table_shells Table 4 "RI p-value", Table 9 Panel C).

Placebo dates.  Every session t_p of the full-sample panel (build_full_panel.py, 2016-01-06 to 2023-12-29)
whose window -- the 60 sessions with data before t_p, t_p and the 59 after, as for the events -- contains
no listing of a new expiration weekday for SPY, QQQ or IWM (first trades found in the data: SPY Wednesday
2016-08-30, SPY Monday 2018-02-16, QQQ Monday/Wednesday 2021-04-23/27, IWM Wednesday/Monday 2021-10-05/08,
SPY/QQQ Tuesday 2022-11-14 and Thursday 2022-11-16), no Cboe listing cut (2022-11-10), and does not straddle
the July-August 2020 gap in OptionMetrics (17 sessions without the common volatility).

Pre-event placebo dates (design 8.4, "placebo event dates in pre-event periods"; user's choice 2026-11-02, the rule
of placebo_beta.py): each event uses only the dates whose whole window lies before the event's new weekday was listed
for the treated ETFs (2022: before the 2022-11-10 cut); for 2018 also after SPY's Wednesday listing (2016-08-30), so
that the placebo windows have the event's pre-event calendar.  2016: 45 dates, 2018: 244, 2021 QQQ and IWM: 819 each,
2022: 972; IWM 2024 (Table 2 only): all.

At each placebo date the event's statistic is recomputed with exactly the event code (lambda_est.py): the
event's treated and control products, listed existing cells, weekday-matched baselines from the 60 placebo
pre-event sessions, Counterfactual 1 with the control families' matching-DTE volume (2022 configuration:
Monday/Wednesday 7-30 and 31+ DTE cells left out, as in the event), and the existing cells' change over
placebo weeks 1-12 in % of the weekday baseline, x_p = 100 (E - CF) / B.  There are no new contracts at a
placebo date, so the placebo lambda divides by the actual event's new volume (same % of B):
    lambda_p = -x_p / new_pct_e,      RI p-value = share of placebo dates with |lambda_p| >= |lambda_e|
(= share with |x_p| >= |x_e|: how often an ordinary 12-week window shows an existing-volume change, relative
to the control-adjusted counterfactual, as large as the event's).  Also reported: the percentile of lambda_e
in the placebo distribution (Table 9 Panel C), and the same for the weekday-baseline-only version.

Density slope (Table 4 Panel B; an addition): 999 random triples of placebo dates, one per event, give placebo
lambdas for the three events; their slope on n_e (identity Lambda, OLS) forms the null distribution of b.

Usage: python placebo.py
Output: results/placebo_lambda.parquet (one row per placebo date x configuration x unit),
        results/lambda_ri.csv (RI p-values and percentiles per event x unit; every Table 2 event, IWM 2024 for the shells'
        Table 4 Panel A),
        results/placebo_lambda_events.parquet (each event's pre-event placebo rows),
        results/table2_placebo_sd.csv (s.d. of the placebo control-adjusted change, every Table 2 event),
        results/lambda_density_ri.csv (RI p-value of the density slope b, delta notional)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import date
from multiprocessing import Pool

import numpy as np
import polars as pl

from build_cells import BUILD_EVENTS, EVENT_SPEC, EVENTS, INCOMPLETE_DAYS, last_session
from build_full_panel import FULL_DIR
from lambda_est import RES_DIR, UNITS, counterfactual, derived, make_window, sums, with_blocks
from panel import CFG, DELTA, TYPES, dte_bucket

LISTINGS = [date(2016, 8, 30), date(2018, 2, 16), date(2021, 4, 23), date(2021, 4, 27), date(2021, 10, 5),
            date(2021, 10, 8), date(2022, 11, 10), date(2022, 11, 14), date(2022, 11, 16)]
OM_GAP = (date(2020, 7, 27), date(2020, 8, 18))
N_PRE, N_POST = 60, 59
CONFIGS = {"SPY|QQQ+IWM": dict(treated=["SPY"], control=["QQQ", "IWM"], cut=False),
           "QQQ|SPY+IWM": dict(treated=["QQQ"], control=["SPY", "IWM"], cut=False),
           "IWM|SPY+QQQ": dict(treated=["IWM"], control=["SPY", "QQQ"], cut=False),
           "SPY+QQQ|IWM": dict(treated=["SPY", "QQQ"], control=["IWM"], cut=True)}
EVENT_CONFIG = {"2016-08-30": "SPY|QQQ+IWM", "2018-02-16": "SPY|QQQ+IWM", "2021-04-27": "QQQ|SPY+IWM",
                "2021-10-08": "IWM|SPY+QQQ", "2022-11-16": "SPY+QQQ|IWM", "2024-04-18": "IWM|SPY+QQQ"}
## pre-event placebo windows: before the event's new weekday was listed (2022: the cut); 2018 also after SPY's Wednesday
## listing, so that the placebo calendar is the event's pre-event calendar (also used by placebo_beta.py)
FIRST_LISTED = {ev: date(2022, 11, 10) if s["cut"] else s["start"] for ev, s in EVENT_SPEC.items()}
LO_BOUND = {ev: date(2016, 8, 30) if ev == "2018-02-16" else date(2016, 1, 1) for ev in EVENT_SPEC}


def full_cells() -> pl.DataFrame:
    """The full-sample panel without the sessions whose file lacks an ETF (build_cells.INCOMPLETE_DAYS)."""
    return pl.concat([pl.read_parquet(FULL_DIR / f"cells_{y}.parquet") for y in range(2016, 2024)]).filter(
        ~pl.col("trade_date").is_in(list(INCOMPLETE_DAYS)))


def full_grid(cells: pl.DataFrame) -> pl.DataFrame:
    """Listed existing cells on every panel day (as panel.py: an expiration is listed from its first trade to
    its last session), with zero volume where a listed cell does not trade."""
    ex = pl.concat([pl.read_parquet(FULL_DIR / f"expirations_{y}.parquet") for y in range(2016, 2024)])
    expi = (ex.group_by("underlying_symbol", "expiration", "eff_wd").agg(pl.col("trade_date").min().alias("first_trade"))
            .with_columns(pl.col("expiration").map_elements(last_session, return_dtype=pl.Date).alias("last_session")))
    days = cells.select("trade_date").unique()
    lst = (days.join(expi, how="cross")
           .filter((pl.col("first_trade") <= pl.col("trade_date")) & (pl.col("last_session") >= pl.col("trade_date")))
           .with_columns(dte_bucket((pl.col("last_session") - pl.col("trade_date")).dt.total_days()).alias("dte_bucket")))
    grid = (lst.select("underlying_symbol", "trade_date", "eff_wd", "dte_bucket").unique()
            .join(pl.DataFrame({"delta_bucket": DELTA}), how="cross").join(pl.DataFrame({"option_type": TYPES}), how="cross"))
    keys = ["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
    vol = ["V", "contracts", "premium", "trades"]
    orphan = cells.join(grid, on=keys, how="anti")
    assert orphan.height == 0, f"{orphan.height} traded cell-days outside the listed grid"
    return (grid.join(cells.select(keys + vol), on=keys, how="left").with_columns([pl.col(c).fill_null(0) for c in vol])
            .sort("trade_date"))


def sessions(cells: pl.DataFrame) -> pl.DataFrame:
    return cells.group_by("trade_date").agg(pl.col("early_close").first()).sort("trade_date")


def eligible(sess: pl.DataFrame) -> list[int]:
    d = sess["trade_date"].to_list()
    out = []
    for i in range(N_PRE, len(d) - N_POST):
        lo, hi = d[i - N_PRE], d[i + N_POST]
        if any(lo <= x <= hi for x in LISTINGS) or (lo <= OM_GAP[1] and hi >= OM_GAP[0]):
            continue
        out.append(i)
    return out


def placebo_days(sess: pl.DataFrame, i: int) -> pl.DataFrame:
    w = sess.slice(i - N_PRE, N_PRE + 1 + N_POST).with_row_index("k")
    w = (w.with_columns((pl.col("k").cast(pl.Int64) - N_PRE).alias("tau"))
         .with_columns(pl.when(pl.col("tau") < 0).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
                       pl.col("trade_date").dt.weekday().alias("wd"))
         .filter(~pl.col("early_close")).select("trade_date", "tau", "period", "wd"))
    return with_blocks(w)


_G = {}


def _init(grid, sess):
    _G["grid"], _G["sess"] = grid, sess


def one(args) -> list[dict]:
    i, cname = args
    cfg = CONFIGS[cname]
    days = placebo_days(_G["sess"], i)
    lo, hi = days["trade_date"].min(), days["trade_date"].max()
    g = _G["grid"].filter(pl.col("trade_date").is_between(lo, hi)).with_columns(
        (pl.lit(cfg["cut"]) & pl.col("eff_wd").is_in([1, 3]) & pl.col("dte_bucket").is_in(["D7-30", "D31+"])).alias("cut_cell"))
    new = pl.DataFrame(schema={"underlying_symbol": pl.Utf8, "trade_date": pl.Date, **{u: pl.Float64 for u in UNITS}})
    rows = []
    for unit in UNITS:
        win = make_window(g, new, days, cfg["treated"], cfg["control"], unit)
        s = sums(win, counterfactual(win))
        rows.append(dict(config=cname, t_p=_G["sess"]["trade_date"][i], unit=unit,
                         x_adj=100 * (s["E"] - s["CF"]) / s["B"], x_raw=100 * (s["E"] - s["B"]) / s["B"],
                         E_nobase_pct=100 * s["E_nobase"] / max(s["E"], 1.0)))
    return rows


def main() -> None:
    cells = full_cells()
    grid = full_grid(cells)
    sess = sessions(cells)
    idx = eligible(sess)
    print(f"panel sessions {sess.height}; eligible placebo dates {len(idx)} "
          f"({sess['trade_date'][idx[0]]} to {sess['trade_date'][idx[-1]]})", flush=True)
    jobs = [(i, c) for c in CONFIGS for i in idx]
    with Pool(12, initializer=_init, initargs=(grid, sess)) as pool:
        rows = [r for rr in pool.imap(one, jobs, chunksize=20) for r in rr]
    pl_ = pl.DataFrame(rows)
    pl_.write_parquet(RES_DIR / "placebo_lambda.parquet")
    ## each event's pre-event placebo rows (window [w_lo, w_hi] inside the event's pre-event period)
    dl = sess["trade_date"].to_list()
    win = pl.DataFrame({"t_p": [dl[i] for i in idx], "w_lo": [dl[i - N_PRE] for i in idx], "w_hi": [dl[i + N_POST] for i in idx]})
    pe = pl.concat([pl_.filter(pl.col("config") == EVENT_CONFIG[ev]).join(win, on="t_p")
                    .filter((pl.col("w_lo") > LO_BOUND[ev]) & (pl.col("w_hi") < FIRST_LISTED[ev]))
                    .with_columns(pl.lit(ev).alias("event")) for ev in BUILD_EVENTS])
    pe.write_parquet(RES_DIR / "placebo_lambda_events.parquet")
    print(pe.filter(pl.col("unit") == "V").group_by("event").agg(pl.len().alias("placebo_dates"),
                                                                 pl.col("t_p").min().alias("first"), pl.col("t_p").max().alias("last")).sort("event"))

    ## the events' lambda and new volume: table2_volume.csv (lambda_est.py) has every Table 2 event, IWM 2024 included,
    ## with the same values as lambda_event.csv's "all" split for the baseline events
    act = pl.read_csv(RES_DIR / "table2_volume.csv", schema_overrides={"event": pl.Utf8})
    out = []
    for ev in BUILD_EVENTS:
        for unit in UNITS:
            a = act.filter((pl.col("event") == ev) & (pl.col("unit") == unit)).row(0, named=True)
            p = pe.filter((pl.col("event") == ev) & (pl.col("unit") == unit))
            lam_p = -p["x_adj"].to_numpy() / a["new_pct"]
            lam_rp = -p["x_raw"].to_numpy() / a["new_pct"]
            out.append(dict(event=ev, unit=unit, n_placebo=p.height, lam=a["lam"], lam_raw=a["lam_raw"],
                            ri_p=float(np.mean(np.abs(lam_p) >= abs(a["lam"]))),
                            pctile=float(100 * np.mean(lam_p <= a["lam"])),
                            ri_p_raw=float(np.mean(np.abs(lam_rp) >= abs(a["lam_raw"]))),
                            pctile_raw=float(100 * np.mean(lam_rp <= a["lam_raw"])),
                            placebo_sd_x=float(p["x_adj"].std()), placebo_mean_x=float(p["x_adj"].mean()),
                            placebo_sd_x_raw=float(p["x_raw"].std()), placebo_mean_x_raw=float(p["x_raw"].mean()),
                            nobase_pct_max=float(p["E_nobase_pct"].max())))
    out = pl.DataFrame(out)
    out.write_csv(RES_DIR / "lambda_ri.csv")
    ## Table 2's placebo s.d. row for every event of the table (pre-event dates; IWM 2024: all, 2021 IWM configuration)
    pl.DataFrame([dict(event=ev, unit=u, n_placebo=p.height, placebo_sd_x=float(p["x_adj"].std()),
                       placebo_mean_x=float(p["x_adj"].mean()))
                  for ev in BUILD_EVENTS for u in UNITS
                  for p in [pe.filter((pl.col("event") == ev) & (pl.col("unit") == u))]]
                 ).write_csv(RES_DIR / "table2_placebo_sd.csv")

    ## density slope: placebo lambdas for the three events from random placebo dates, slope on n_e
    rng = np.random.default_rng(20261103)
    n_e = np.array([float(EVENT_SPEC[ev]["n_e"]) for ev in EVENTS])
    lam_a = np.array([act.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["lam"][0] for ev in EVENTS])
    newp = [act.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["new_pct"][0] for ev in EVENTS]
    xs = [pe.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["x_adj"].to_numpy() for ev in EVENTS]
    b_act = float(np.polyfit(n_e, lam_a, 1)[0])
    bp = np.array([np.polyfit(n_e, [-x[rng.integers(len(x))] / q for x, q in zip(xs, newp)], 1)[0] for _ in range(999)])
    pl.DataFrame([dict(b_lin=b_act, ri_p=float(np.mean(np.abs(bp) >= abs(b_act))), placebo_sd=float(bp.std(ddof=1)),
                       pctile=float(100 * np.mean(bp <= b_act)), n=len(bp))]).write_csv(RES_DIR / "lambda_density_ri.csv")
    print(f"density slope b = {b_act:.3f}: RI p = {np.mean(np.abs(bp) >= abs(b_act)):.3f}, placebo s.d. {bp.std(ddof=1):.3f}")
    with pl.Config(tbl_rows=20, tbl_width_chars=220, float_precision=3):
        print(out)


if __name__ == "__main__":
    main()
