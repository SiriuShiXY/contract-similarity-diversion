"""Build full-sample cells and run the complete randomization inference."""

from __future__ import annotations

import sys


# Build Full Panel
"""
Full-sample cell-day panel, 2016-01-06 to 2023-12-29, for randomization inference with placebo event
dates (similarity_diversion_design.pdf Section 8.4; table_shells Table 4 "RI p-value").

Source: the complete daily Cboe trade files on the external T7 drive (OptionsTradeData.zip, one zipped CSV
per session, 2012-01-03 to 2025-07-14; same files and fields as the local raw directory, the same four
sessions missing in the event windows).  The span is set by the baseline inputs: OptionMetrics 30-day
ATM IV (common volatility) 2016-01-04 to 2023-12-29, OIS curve from 2016-01-06.

Per session, exactly the event panels' per-day code (build_cells.cells_for_day: trade filters, trading-hour
clock, common-volatility delta, delta notional, characteristics, Deep ITM cells without characteristics) with every expiration
classified by the weekday it stands for (ListingDynamics rule: Saturday-dated -> Friday, Tuesday after a
Monday holiday -> Monday, Thursday before a Friday holiday -> Friday; added here, for SPY: Tuesday before a
Wednesday holiday -> Wednesday, which the event windows do not contain) and is_new = False (no listing
is assigned here; placebo.py does that).  Also written per session: every traded expiration with its
contracts, so that first trades (and hence listed expirations and new-weekday listings) can be derived.

Usage: python build_full_panel.py [n_processes]          (about 10 minutes with 6 processes)
Output: data/full/cells_<year>.parquet, data/full/expirations_<year>.parquet, data/full/sigbar_<year>.csv
"""
import io
import json
import sys
import time
import zipfile
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path
import polars as pl
from Build_Panel import archive_days_all
from Build_Panel import (
    ARCHIVE,
    CLOSE_MIN,
    DELTA_BUCKETS,
    DELTA_SOURCE,
    OUT_DIR,
    SESSION_IDX,
    SIGMA_SOURCE,
    SYMBOLS,
    VARIANT,
    cells_for_day,
    dividend_yields,
    ois_curve,
    optionmetrics_vol,
)

FULL_DIR = OUT_DIR / "full"
SPEC = dict(
    variant=VARIANT,
    delta_source=DELTA_SOURCE,
    delta_buckets=DELTA_BUCKETS,
    sigma_source=SIGMA_SOURCE,
)
START, END = (date(2016, 1, 6), date(2023, 12, 29))
CLOSED = {date(2015, 1, 1) + timedelta(i) for i in range(4018)} - set(SESSION_IDX)


def effective_weekday(d: date, sym: str) -> int:
    """Weekday an expiration stands for (ListingDynamics/analyze_listing_dynamics.py), plus one case that
    does not occur in the three event windows: for SPY, the only one of the three ETFs with Wednesday
    expirations when Wednesday holidays fell in 2016-2023 (2018-07-04, 2018-12-05, 2019-12-25, 2020-01-01),
    a Tuesday before a Wednesday holiday stands for Wednesday.  (A Tuesday month-end of QQQ or IWM before
    such a holiday, e.g. 2019-12-31, stays a Tuesday.)"""
    wd = d.isoweekday()
    if wd == 6:
        return 5
    if wd == 2 and d - timedelta(1) in CLOSED:
        return 1
    if wd == 2 and d + timedelta(1) in CLOSED and (sym == "SPY"):
        return 3
    if wd == 4 and d + timedelta(1) in CLOSED:
        return 5
    return wd


def archive_days() -> list[date]:
    return [d for d in archive_days_all() if START <= d <= END]


def read_day(z: zipfile.ZipFile, d: date) -> bytes:
    inner = zipfile.ZipFile(
        io.BytesIO(z.read(f"OptionsTradeData/UnderlyingOptionsTradesCalcs_{d}.zip"))
    )
    return inner.read(f"UnderlyingOptionsTradesCalcs_{d}.csv")


def build_year(args: tuple[int, bool]) -> str:
    year, reuse = args
    out = FULL_DIR / f"cells_{year}.parquet"
    if reuse and out.exists():
        return f"{year}: exists"
    t0 = time.time()
    ois, divs = (ois_curve(), dividend_yields())
    days = [d for d in archive_days() if d.year == year]
    om = optionmetrics_vol()
    cells, expis, diags, skipped = ([], [], [], [])
    with zipfile.ZipFile(ARCHIVE) as z:
        for d in days:
            if not all(((s, d) in om for s in SYMBOLS)):
                skipped.append(d)
                continue
            raw = read_day(z, d)
            ex = (
                pl.read_csv(
                    raw,
                    columns=[
                        "underlying_symbol",
                        "expiration",
                        "trade_size",
                        "canceled_trade_condition_id",
                    ],
                    infer_schema_length=0,
                    truncate_ragged_lines=True,
                )
                .filter(
                    pl.col("underlying_symbol").is_in(SYMBOLS)
                    & (pl.col("canceled_trade_condition_id").cast(pl.Int64) == 0)
                )
                .group_by("underlying_symbol", "expiration")
                .agg(pl.col("trade_size").cast(pl.Float64).sum().alias("contracts"))
                .with_columns(pl.col("expiration").str.to_date())
            )
            ex = ex.with_columns(
                pl.struct("expiration", "underlying_symbol")
                .map_elements(
                    lambda r: effective_weekday(
                        r["expiration"], r["underlying_symbol"]
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("eff_wd"),
                pl.lit(False).alias("is_new"),
                pl.lit(d).alias("first_trade"),
            )
            c, g = cells_for_day(
                d,
                ex.select(
                    "underlying_symbol", "expiration", "eff_wd", "is_new", "first_trade"
                ),
                ois,
                divs,
                src=raw,
                strict=False,
            )
            cells.append(
                c.with_columns(pl.lit(CLOSE_MIN[d] < 16 * 60).alias("early_close"))
            )
            expis.append(
                ex.select(
                    "underlying_symbol",
                    "expiration",
                    "eff_wd",
                    "contracts",
                    pl.lit(d).alias("trade_date"),
                )
            )
            diags.extend(g)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(expis).write_parquet(FULL_DIR / f"expirations_{year}.parquet")
    pl.DataFrame(diags).write_csv(FULL_DIR / f"sigbar_{year}.csv")
    pl.concat(cells).write_parquet(out)
    return f"{year}: {len(cells)} sessions, {sum((c.height for c in cells)):,} cell-days, skipped (no OptionMetrics) {skipped} ({time.time() - t0:.0f}s)"


def build_full_panel_main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = FULL_DIR / "spec.json"
    reuse = stamp.exists() and json.loads(stamp.read_text()) == SPEC
    if not reuse:
        print(
            f"cell settings {SPEC} differ from the files' (or no spec.json): every year is rebuilt",
            flush=True,
        )
    with Pool(n) as pool:
        for msg in pool.imap_unordered(
            build_year, [(y, reuse) for y in range(START.year, END.year + 1)]
        ):
            print(msg, flush=True)
    stamp.write_text(json.dumps(SPEC))


# Placebo
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
import os

for placebo__v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(placebo__v, "1")
from datetime import date
from multiprocessing import Pool
import numpy as np
import polars as pl
from Build_Panel import BUILD_EVENTS, EVENT_SPEC, EVENTS, INCOMPLETE_DAYS, last_session
from Estimate_Lambda import (
    RES_DIR,
    UNITS,
    counterfactual,
    derived,
    make_window,
    sums,
    with_blocks,
)
from Build_Event_Panel import CFG, DELTA, TYPES, dte_bucket

LISTINGS = [
    date(2016, 8, 30),
    date(2018, 2, 16),
    date(2021, 4, 23),
    date(2021, 4, 27),
    date(2021, 10, 5),
    date(2021, 10, 8),
    date(2022, 11, 10),
    date(2022, 11, 14),
    date(2022, 11, 16),
]
OM_GAP = (date(2020, 7, 27), date(2020, 8, 18))
N_PRE, N_POST = (60, 59)
CONFIGS = {
    "SPY|QQQ+IWM": dict(treated=["SPY"], control=["QQQ", "IWM"], cut=False),
    "QQQ|SPY+IWM": dict(treated=["QQQ"], control=["SPY", "IWM"], cut=False),
    "IWM|SPY+QQQ": dict(treated=["IWM"], control=["SPY", "QQQ"], cut=False),
    "SPY+QQQ|IWM": dict(treated=["SPY", "QQQ"], control=["IWM"], cut=True),
}
EVENT_CONFIG = {
    "2016-08-30": "SPY|QQQ+IWM",
    "2018-02-16": "SPY|QQQ+IWM",
    "2021-04-27": "QQQ|SPY+IWM",
    "2021-10-08": "IWM|SPY+QQQ",
    "2022-11-16": "SPY+QQQ|IWM",
    "2024-04-18": "IWM|SPY+QQQ",
}
FIRST_LISTED = {
    ev: date(2022, 11, 10) if s["cut"] else s["start"] for ev, s in EVENT_SPEC.items()
}
LO_BOUND = {
    ev: date(2016, 8, 30) if ev == "2018-02-16" else date(2016, 1, 1)
    for ev in EVENT_SPEC
}


def full_cells() -> pl.DataFrame:
    """The full-sample panel without the sessions whose file lacks an ETF (build_cells.INCOMPLETE_DAYS)."""
    return pl.concat(
        [pl.read_parquet(FULL_DIR / f"cells_{y}.parquet") for y in range(2016, 2024)]
    ).filter(~pl.col("trade_date").is_in(list(INCOMPLETE_DAYS)))


def full_grid(cells: pl.DataFrame) -> pl.DataFrame:
    """Listed existing cells on every panel day (as panel.py: an expiration is listed from its first trade to
    its last session), with zero volume where a listed cell does not trade."""
    ex = pl.concat(
        [
            pl.read_parquet(FULL_DIR / f"expirations_{y}.parquet")
            for y in range(2016, 2024)
        ]
    )
    expi = (
        ex.group_by("underlying_symbol", "expiration", "eff_wd")
        .agg(pl.col("trade_date").min().alias("first_trade"))
        .with_columns(
            pl.col("expiration")
            .map_elements(last_session, return_dtype=pl.Date)
            .alias("last_session")
        )
    )
    days = cells.select("trade_date").unique()
    lst = (
        days.join(expi, how="cross")
        .filter(
            (pl.col("first_trade") <= pl.col("trade_date"))
            & (pl.col("last_session") >= pl.col("trade_date"))
        )
        .with_columns(
            dte_bucket(
                (pl.col("last_session") - pl.col("trade_date")).dt.total_days()
            ).alias("dte_bucket")
        )
    )
    grid = (
        lst.select("underlying_symbol", "trade_date", "eff_wd", "dte_bucket")
        .unique()
        .join(pl.DataFrame({"delta_bucket": DELTA}), how="cross")
        .join(pl.DataFrame({"option_type": TYPES}), how="cross")
    )
    keys = [
        "underlying_symbol",
        "trade_date",
        "eff_wd",
        "dte_bucket",
        "delta_bucket",
        "option_type",
    ]
    vol = ["V", "contracts", "premium", "trades"]
    orphan = cells.join(grid, on=keys, how="anti")
    assert (
        orphan.height == 0
    ), f"{orphan.height} traded cell-days outside the listed grid"
    return (
        grid.join(cells.select(keys + vol), on=keys, how="left")
        .with_columns([pl.col(c).fill_null(0) for c in vol])
        .sort("trade_date")
    )


def sessions(cells: pl.DataFrame) -> pl.DataFrame:
    return (
        cells.group_by("trade_date")
        .agg(pl.col("early_close").first())
        .sort("trade_date")
    )


def eligible(sess: pl.DataFrame) -> list[int]:
    d = sess["trade_date"].to_list()
    out = []
    for i in range(N_PRE, len(d) - N_POST):
        lo, hi = (d[i - N_PRE], d[i + N_POST])
        if any((lo <= x <= hi for x in LISTINGS)) or (
            lo <= OM_GAP[1] and hi >= OM_GAP[0]
        ):
            continue
        out.append(i)
    return out


def placebo_days(sess: pl.DataFrame, i: int) -> pl.DataFrame:
    w = sess.slice(i - N_PRE, N_PRE + 1 + N_POST).with_row_index("k")
    w = (
        w.with_columns((pl.col("k").cast(pl.Int64) - N_PRE).alias("tau"))
        .with_columns(
            pl.when(pl.col("tau") < 0)
            .then(pl.lit("pre"))
            .otherwise(pl.lit("post"))
            .alias("period"),
            pl.col("trade_date").dt.weekday().alias("wd"),
        )
        .filter(~pl.col("early_close"))
        .select("trade_date", "tau", "period", "wd")
    )
    return with_blocks(w)


placebo__G = {}


def placebo__init(grid, sess):
    placebo__G["grid"], placebo__G["sess"] = (grid, sess)


def placebo_one(args) -> list[dict]:
    i, cname = args
    cfg = CONFIGS[cname]
    days = placebo_days(placebo__G["sess"], i)
    lo, hi = (days["trade_date"].min(), days["trade_date"].max())
    g = (
        placebo__G["grid"]
        .filter(pl.col("trade_date").is_between(lo, hi))
        .with_columns(
            (
                pl.lit(cfg["cut"])
                & pl.col("eff_wd").is_in([1, 3])
                & pl.col("dte_bucket").is_in(["D7-30", "D31+"])
            ).alias("cut_cell")
        )
    )
    new = pl.DataFrame(
        schema={
            "underlying_symbol": pl.Utf8,
            "trade_date": pl.Date,
            **{u: pl.Float64 for u in UNITS},
        }
    )
    rows = []
    for unit in UNITS:
        win = make_window(g, new, days, cfg["treated"], cfg["control"], unit)
        s = sums(win, counterfactual(win))
        rows.append(
            dict(
                config=cname,
                t_p=placebo__G["sess"]["trade_date"][i],
                unit=unit,
                x_adj=100 * (s["E"] - s["CF"]) / s["B"],
                x_raw=100 * (s["E"] - s["B"]) / s["B"],
                E_nobase_pct=100 * s["E_nobase"] / max(s["E"], 1.0),
            )
        )
    return rows


def placebo_main() -> None:
    cells = full_cells()
    grid = full_grid(cells)
    sess = sessions(cells)
    idx = eligible(sess)
    print(
        f"panel sessions {sess.height}; eligible placebo dates {len(idx)} ({sess['trade_date'][idx[0]]} to {sess['trade_date'][idx[-1]]})",
        flush=True,
    )
    jobs = [(i, c) for c in CONFIGS for i in idx]
    with Pool(12, initializer=placebo__init, initargs=(grid, sess)) as pool:
        rows = [r for rr in pool.imap(placebo_one, jobs, chunksize=20) for r in rr]
    pl_ = pl.DataFrame(rows)
    pl_.write_parquet(RES_DIR / "placebo_lambda.parquet")
    dl = sess["trade_date"].to_list()
    win = pl.DataFrame(
        {
            "t_p": [dl[i] for i in idx],
            "w_lo": [dl[i - N_PRE] for i in idx],
            "w_hi": [dl[i + N_POST] for i in idx],
        }
    )
    pe = pl.concat(
        [
            pl_.filter(pl.col("config") == EVENT_CONFIG[ev])
            .join(win, on="t_p")
            .filter(
                (pl.col("w_lo") > LO_BOUND[ev]) & (pl.col("w_hi") < FIRST_LISTED[ev])
            )
            .with_columns(pl.lit(ev).alias("event"))
            for ev in BUILD_EVENTS
        ]
    )
    pe.write_parquet(RES_DIR / "placebo_lambda_events.parquet")
    print(
        pe.filter(pl.col("unit") == "V")
        .group_by("event")
        .agg(
            pl.len().alias("placebo_dates"),
            pl.col("t_p").min().alias("first"),
            pl.col("t_p").max().alias("last"),
        )
        .sort("event")
    )
    act = pl.read_csv(
        RES_DIR / "table2_volume.csv", schema_overrides={"event": pl.Utf8}
    )
    out = []
    for ev in BUILD_EVENTS:
        for unit in UNITS:
            a = act.filter((pl.col("event") == ev) & (pl.col("unit") == unit)).row(
                0, named=True
            )
            p = pe.filter((pl.col("event") == ev) & (pl.col("unit") == unit))
            lam_p = -p["x_adj"].to_numpy() / a["new_pct"]
            lam_rp = -p["x_raw"].to_numpy() / a["new_pct"]
            out.append(
                dict(
                    event=ev,
                    unit=unit,
                    n_placebo=p.height,
                    lam=a["lam"],
                    lam_raw=a["lam_raw"],
                    ri_p=float(np.mean(np.abs(lam_p) >= abs(a["lam"]))),
                    pctile=float(100 * np.mean(lam_p <= a["lam"])),
                    ri_p_raw=float(np.mean(np.abs(lam_rp) >= abs(a["lam_raw"]))),
                    pctile_raw=float(100 * np.mean(lam_rp <= a["lam_raw"])),
                    placebo_sd_x=float(p["x_adj"].std()),
                    placebo_mean_x=float(p["x_adj"].mean()),
                    placebo_sd_x_raw=float(p["x_raw"].std()),
                    placebo_mean_x_raw=float(p["x_raw"].mean()),
                    nobase_pct_max=float(p["E_nobase_pct"].max()),
                )
            )
    out = pl.DataFrame(out)
    out.write_csv(RES_DIR / "lambda_ri.csv")
    pl.DataFrame(
        [
            dict(
                event=ev,
                unit=u,
                n_placebo=p.height,
                placebo_sd_x=float(p["x_adj"].std()),
                placebo_mean_x=float(p["x_adj"].mean()),
            )
            for ev in BUILD_EVENTS
            for u in UNITS
            for p in [pe.filter((pl.col("event") == ev) & (pl.col("unit") == u))]
        ]
    ).write_csv(RES_DIR / "table2_placebo_sd.csv")
    rng = np.random.default_rng(20261103)
    n_e = np.array([float(EVENT_SPEC[ev]["n_e"]) for ev in EVENTS])
    lam_a = np.array(
        [
            act.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["lam"][0]
            for ev in EVENTS
        ]
    )
    newp = [
        act.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["new_pct"][0]
        for ev in EVENTS
    ]
    xs = [
        pe.filter((pl.col("event") == ev) & (pl.col("unit") == "V"))["x_adj"].to_numpy()
        for ev in EVENTS
    ]
    b_act = float(np.polyfit(n_e, lam_a, 1)[0])
    bp = np.array(
        [
            np.polyfit(
                n_e, [-x[rng.integers(len(x))] / q for x, q in zip(xs, newp)], 1
            )[0]
            for _ in range(999)
        ]
    )
    pl.DataFrame(
        [
            dict(
                b_lin=b_act,
                ri_p=float(np.mean(np.abs(bp) >= abs(b_act))),
                placebo_sd=float(bp.std(ddof=1)),
                pctile=float(100 * np.mean(bp <= b_act)),
                n=len(bp),
            )
        ]
    ).write_csv(RES_DIR / "lambda_density_ri.csv")
    print(
        f"density slope b = {b_act:.3f}: RI p = {np.mean(np.abs(bp) >= abs(b_act)):.3f}, placebo s.d. {bp.std(ddof=1):.3f}"
    )
    with pl.Config(tbl_rows=20, tbl_width_chars=220, float_precision=3):
        print(out)


# Placebo Beta
"""
Randomization inference for the similarity and nearest-expiry coefficients with placebo event dates
(similarity_diversion_design.pdf Section 8.4: "recompute lambda and beta"; table_shells Table 9 Panel C,
appendix A8).

Statistic: the reduced form eq. (17) on the treated products' existing cells, post-event days 0-59,
    y_jt = beta S_jt + gamma N_jt + product x day FE,  weights Vbar_pre_j      (Table 5 columns 1-3)
with y from Counterfactual 1 and S, N on the event's new-expiration calendar.  At a placebo date there is
no new volume, so S is the calendar similarity of eq. (18) (phantom.py): the new weekday's expirations from
day t up to the event's longest new expiration for that weekday of t, skipping dates that already are
expirations of the product, each (expiration weekday, DTE bucket, delta bucket, type) cell carrying the
event's calendar-position profile of characteristics and weights; S_jt = sum_k w_k exp(-theta_hat d_jk^2) /
sum_k w_k with Sigma_p from the placebo window's pre-event sessions; N on the same calendar.  The actual
statistic is computed the same way (phantom.py's S_hat and N for the treated products after the event) so
that actual and placebo are comparable; it differs from Table 5, whose S weights the realized new volume.

Placebo dates: the dates of placebo.py whose whole window lies before the event's new weekday was listed
for the treated products (2016: SPY Wednesday, before 2016-08-30; 2018: SPY Monday, between the 2016 and
2018 listings; 2022: Tuesday/Thursday, before 2022-11-10), so that the phantom weekday is not traded.
Pooled: 999 random draws of one placebo date per event, the pooled regression on the stacked samples.
RI p-value = share of placebo coefficients at least as large in absolute value as the actual one.

Usage: python placebo_beta.py          (after placebo.py, structural.py, phantom.py)
Output: results/placebo_beta.parquet, results/beta_ri.csv
"""
import os

for placebo_beta__v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(placebo_beta__v, "1")
from datetime import date, timedelta
from multiprocessing import Pool
import numpy as np
import polars as pl
from Build_Panel import EVENT_SPEC, EVENTS, EV_DIR, OUT_DIR, SESSION_IDX, last_session
from Estimate_Lambda import RES_DIR, counterfactual, make_window
from Measure_Similarity import shrunk_cov
from Build_Event_Panel import CFG, dte_bucket, listed_expirations
from Measure_Similarity import NEW_WD, profile, theta_from_structural

SPECS = {"t5_1": ["N"], "t5_2": ["S"], "t5_3": ["S", "N"]}
X = ["x_logtau", "x_absd"]
CELL = ["underlying_symbol", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
N_POOL = 999
SEED = 20261102


def within_xprods(y, Xm, w, grp) -> dict:
    """Weighted one-way within transformation by group; returns X'WX and X'Wy for the pooled solve."""
    ng = grp.max() + 1
    sw = np.bincount(grp, w, ng)
    yt = y - np.bincount(grp, w * y, ng)[grp] / sw[grp]
    Xt = np.column_stack([x - np.bincount(grp, w * x, ng)[grp] / sw[grp] for x in Xm.T])
    return dict(XtWX=(Xt * w[:, None]).T @ Xt, XtWy=(Xt * w[:, None]).T @ yt)


def solve(xp: dict) -> np.ndarray:
    return np.linalg.solve(xp["XtWX"], xp["XtWy"])


def stats(frame: pl.DataFrame) -> dict:
    """frame: y, S, N, Vpre, grp (product x day) on the regression sample."""
    y, w = (frame["y"].to_numpy(), frame["Vpre"].to_numpy())
    grp = frame["grp"].to_numpy()
    out = {}
    for nm, cols in SPECS.items():
        xp = within_xprods(y, frame.select(cols).to_numpy(), w, grp)
        b = solve(xp)
        out[nm] = dict(xp=xp, coef=dict(zip(cols, b)))
    return out


placebo_beta__G = {}


def placebo_beta__init(shared):
    placebo_beta__G.update(shared)


def horizon(ev: str) -> dict[int, int]:
    days = (
        pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
        .filter(
            pl.col("has_file") & ~pl.col("early_close") & (pl.col("period") == "post")
        )
        .select("trade_date", pl.col("trade_date").dt.weekday().alias("wd"))
    )
    lst = listed_expirations(ev, days.select("trade_date"))
    act = lst.filter(pl.col("is_new")).join(days, on="trade_date")
    return dict(act.group_by("wd").agg(pl.col("dte").max()).iter_rows())


def placebo_sample(ev: str, i: int) -> pl.DataFrame:
    """Regression sample (y, S, N, Vpre, product x day) of event ev's design at placebo date index i."""
    cfg, new_wd, H = (CFG[ev], NEW_WD[ev], placebo_beta__G["H"][ev])
    days = placebo_days(placebo_beta__G["sess"], i)
    lo, hi = (days["trade_date"].min(), days["trade_date"].max())
    cut = (
        pl.lit(cfg["cut"])
        & pl.col("eff_wd").is_in([1, 3])
        & pl.col("dte_bucket").is_in(["D7-30", "D31+"])
    )
    g = (
        placebo_beta__G["grid"]
        .filter(pl.col("trade_date").is_between(lo, hi))
        .with_columns(cut.alias("cut_cell"))
    )
    new = pl.DataFrame(
        schema={"underlying_symbol": pl.Utf8, "trade_date": pl.Date, "V": pl.Float64}
    )
    win = make_window(g, new, days, cfg["treated"], cfg["control"], "V")
    c = counterfactual(win)
    post = days.filter(pl.col("period") == "post")
    ex = (
        g.filter(
            pl.col("underlying_symbol").is_in(cfg["treated"])
            & (pl.col("delta_bucket") != "DITM")
        )
        .join(
            placebo_beta__G["cells"]
            .filter(pl.col("trade_date").is_between(lo, hi))
            .select(
                [
                    "underlying_symbol",
                    "trade_date",
                    "eff_wd",
                    "dte_bucket",
                    "delta_bucket",
                    "option_type",
                    "V_char",
                ]
                + X
            ),
            on=[
                "underlying_symbol",
                "trade_date",
                "eff_wd",
                "dte_bucket",
                "delta_bucket",
                "option_type",
            ],
            how="left",
        )
        .join(days.select("trade_date", "period", "wd"), on="trade_date")
        .with_columns(pl.col("V_char").fill_null(0))
    )
    pre = ex.filter(pl.col("period") == "pre")
    has_x = pl.col("V_char") > 0
    m_wd = (
        pre.filter(has_x)
        .group_by(CELL + ["wd"])
        .agg([pl.col(x).mean().alias(f"{x}_wd") for x in X])
    )
    m_pre = (
        pre.filter(has_x)
        .group_by(CELL)
        .agg([pl.col(x).mean().alias(f"{x}_pre") for x in X])
    )
    m_all = (
        ex.filter(has_x)
        .group_by(CELL)
        .agg([pl.col(x).mean().alias(f"{x}_all") for x in X])
    )
    ex = (
        ex.filter(pl.col("period") == "post")
        .join(m_wd, on=CELL + ["wd"], how="left")
        .join(m_pre, on=CELL, how="left")
        .join(m_all, on=CELL, how="left")
        .with_columns(
            [
                pl.when(pl.col("V_char") > 0)
                .then(pl.col(x))
                .otherwise(pl.coalesce(f"{x}_wd", f"{x}_pre", f"{x}_all"))
                .alias(x)
                for x in X
            ]
        )
    )
    inv = {}
    for (sym, typ), gg in pre.filter(has_x).group_by(
        ["underlying_symbol", "option_type"]
    ):
        inv[sym, typ] = shrunk_cov(gg.select(X).to_numpy(), gg["V"].to_numpy())["S_inv"]
    lst = placebo_beta__G["lst"].filter(
        pl.col("underlying_symbol").is_in(cfg["treated"])
        & pl.col("trade_date").is_between(lo, hi)
    )
    existing = {
        s: set(lst.filter(pl.col("underlying_symbol") == s)["last_session"].to_list())
        for s in cfg["treated"]
    }
    rows = []
    for t, wd in post.select("trade_date", "wd").iter_rows():
        for k in range(H[wd] + 1):
            d = t + timedelta(k)
            if d.isoweekday() in new_wd and d in SESSION_IDX:
                rows += [(s, t, d) for s in cfg["treated"] if d not in existing[s]]
    cal = pl.DataFrame(
        rows, schema=["underlying_symbol", "trade_date", "last_session"], orient="row"
    ).with_columns(
        # Use the same product-specific classification as the full-sample cells.
        pl.struct("last_session", "underlying_symbol")
        .map_elements(
            lambda row: effective_weekday(
                row["last_session"], row["underlying_symbol"]
            ),
            return_dtype=pl.Int64,
        )
        .alias("eff_wd_k"),
        dte_bucket(
            (pl.col("last_session") - pl.col("trade_date")).dt.total_days()
        ).alias("dte_bucket_k"),
    )
    e_star = (
        lst.filter(pl.col("trade_date").is_in(post["trade_date"].implode()))
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
            pl.lit(1).alias("N"),
        )
    )
    p1, p2, p3 = placebo_beta__G["prof"][ev]
    nc = (
        cal.select("underlying_symbol", "trade_date", "eff_wd_k", "dte_bucket_k")
        .unique()
        .join(
            pl.DataFrame({"delta_bucket_k": ["DOTM", "OTM", "ATM", "ITM"]}), how="cross"
        )
        .join(pl.DataFrame({"option_type": ["C", "P"]}), how="cross")
        .with_columns(pl.col("trade_date").dt.weekday().alias("wd"))
    )
    cols = [f"{x}_k" for x in X] + ["w_k"]
    nc = (
        nc.join(
            p1,
            on=["wd", "eff_wd_k", "dte_bucket_k", "delta_bucket_k", "option_type"],
            how="left",
        )
        .join(
            p2.rename({c: f"{c}_2" for c in cols}),
            on=["eff_wd_k", "dte_bucket_k", "delta_bucket_k", "option_type"],
            how="left",
        )
        .join(
            p3.rename({c: f"{c}_3" for c in cols}),
            on=["dte_bucket_k", "delta_bucket_k", "option_type"],
            how="left",
        )
        .with_columns([pl.coalesce(c, f"{c}_2", f"{c}_3").alias(c) for c in cols])
        .filter(pl.col("w_k").is_not_null())
    )
    pairs = ex.select(CELL + ["trade_date"] + X).join(
        nc.select("underlying_symbol", "trade_date", "option_type", *cols),
        on=["underlying_symbol", "trade_date", "option_type"],
    )
    i11 = pl.Series(
        [
            inv[s, t][0, 0]
            for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()
        ]
    )
    i12 = pl.Series(
        [
            inv[s, t][0, 1]
            for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()
        ]
    )
    i22 = pl.Series(
        [
            inv[s, t][1, 1]
            for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()
        ]
    )
    a, b = (
        pl.col("x_logtau") - pl.col("x_logtau_k"),
        pl.col("x_absd") - pl.col("x_absd_k"),
    )
    pairs = pairs.with_columns(i11=i11, i12=i12, i22=i22).with_columns(
        (
            pl.col("i11") * a**2 + 2 * pl.col("i12") * a * b + pl.col("i22") * b**2
        ).alias("d2")
    )
    th = placebo_beta__G["theta"]
    S = pairs.group_by(CELL + ["trade_date"]).agg(
        (
            (pl.col("w_k") * (-th * pl.col("d2")).exp()).sum() / pl.col("w_k").sum()
        ).alias("S")
    )
    ci = {r: j for j, r in enumerate(win.cells.iter_rows())}
    di = {d: k for k, d in enumerate(days["trade_date"].to_list())}
    smp = (
        ex.select(CELL + ["trade_date"])
        .join(S, on=CELL + ["trade_date"], how="left")
        .join(
            nflag,
            on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket"],
            how="left",
        )
        .with_columns(
            pl.col("N").fill_null(0).cast(pl.Float64), pl.col("S").fill_null(0.0)
        )
    )
    jr = np.array([ci.get(r, -1) for r in smp.select(CELL).iter_rows()])
    dc = np.array([di[d] for d in smp["trade_date"].to_list()])
    ok = jr >= 0
    vpre = np.where(ok, c["mean_all"][np.where(ok, jr, 0)], np.nan)
    cf = np.where(ok, c["cf"][np.where(ok, jr, 0), dc], np.nan)
    V = np.where(ok, win.T[np.where(ok, jr, 0), dc], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = (V - cf) / vpre
    smp = smp.with_columns(pl.Series("y", y), pl.Series("Vpre", vpre)).filter(
        pl.col("y").is_finite() & (pl.col("Vpre") > 0)
    )
    return smp.with_columns(
        pl.concat_str(["underlying_symbol", pl.col("trade_date").cast(pl.Utf8)]).alias(
            "pday"
        )
    )


def placebo_beta_one(args) -> dict:
    ev, i = args
    smp = placebo_sample(ev, i)
    smp = smp.with_columns(pl.col("pday").rank("dense").cast(pl.Int64).alias("grp") - 1)
    st = stats(smp)
    return dict(
        event=ev,
        t_p=placebo_beta__G["sess"]["trade_date"][i],
        n=smp.height,
        S_mean=float(smp["S"].mean()),
        N_sum=int(smp["N"].sum()),
        st=st,
    )


def actual(ev: str) -> dict:
    d = pl.read_parquet(OUT_DIR / ev / "did_panel.parquet").filter(
        pl.col("treated")
        & (pl.col("period") == "post")
        & ~pl.col("cut_cell")
        & pl.col("y").is_not_null()
        & (pl.col("Vpre") > 0)
    )
    d = d.with_columns(
        pl.col("S_hat").alias("S"),
        pl.col("N_cal").cast(pl.Float64).alias("N"),
        pl.concat_str(["underlying_symbol", pl.col("trade_date").cast(pl.Utf8)])
        .rank("dense")
        .cast(pl.Int64)
        .alias("grp")
        - 1,
    )
    return dict(event=ev, n=d.height, st=stats(d))


def placebo_beta_main() -> None:
    cells = full_cells()
    grid = full_grid(cells)
    sess = sessions(cells)
    expi = (
        pl.concat(
            [
                pl.read_parquet(FULL_DIR / f"expirations_{y}.parquet")
                for y in range(2016, 2024)
            ]
        )
        .group_by("underlying_symbol", "expiration", "eff_wd")
        .agg(pl.col("trade_date").min().alias("first_trade"))
        .with_columns(
            pl.col("expiration")
            .map_elements(last_session, return_dtype=pl.Date)
            .alias("last_session")
        )
    )
    lst = (
        sess.select("trade_date")
        .join(expi, how="cross")
        .filter(
            (pl.col("first_trade") <= pl.col("trade_date"))
            & (pl.col("last_session") >= pl.col("trade_date"))
        )
        .with_columns(
            dte_bucket(
                (pl.col("last_session") - pl.col("trade_date")).dt.total_days()
            ).alias("dte_bucket")
        )
    )
    idx = eligible(sess)
    dl = sess["trade_date"].to_list()
    jobs = []
    for ev in EVENTS:
        lo_bound = LO_BOUND[ev]
        ok = [i for i in idx if dl[i - 60] > lo_bound and dl[i + 59] < FIRST_LISTED[ev]]
        jobs += [(ev, i) for i in ok]
        print(
            f"[{ev}] eligible placebo dates {len(ok)}"
            + (f" ({dl[ok[0]]} to {dl[ok[-1]]})" if ok else ""),
            flush=True,
        )
    shared = dict(
        grid=grid,
        cells=cells,
        sess=sess,
        lst=lst,
        theta=theta_from_structural(),
        H={ev: horizon(ev) for ev in EVENTS},
        prof={
            ev: profile(ev, pl.read_parquet(OUT_DIR / ev / "cells.parquet"))
            for ev in EVENTS
        },
    )
    with Pool(12, initializer=placebo_beta__init, initargs=(shared,)) as pool:
        res = pool.map(placebo_beta_one, jobs, chunksize=4)
    act = {ev: actual(ev) for ev in EVENTS}
    rows = []
    for r in res:
        for nm, v in r["st"].items():
            for term, b in v["coef"].items():
                rows.append(
                    dict(
                        event=r["event"],
                        t_p=r["t_p"],
                        spec=nm,
                        term=term,
                        coef=float(b),
                        n=r["n"],
                        S_mean=r["S_mean"],
                        N_sum=r["N_sum"],
                    )
                )
    rng = np.random.default_rng(SEED)
    by_ev = {ev: [r for r in res if r["event"] == ev] for ev in EVENTS}
    for draw in range(N_POOL):
        pick = [by_ev[ev][rng.integers(len(by_ev[ev]))] for ev in EVENTS]
        for nm, cols in SPECS.items():
            xp = dict(
                XtWX=sum((p["st"][nm]["xp"]["XtWX"] for p in pick)),
                XtWy=sum((p["st"][nm]["xp"]["XtWy"] for p in pick)),
            )
            for term, b in zip(cols, solve(xp)):
                rows.append(
                    dict(
                        event="pooled",
                        t_p=None,
                        spec=nm,
                        term=term,
                        coef=float(b),
                        n=sum((p["n"] for p in pick)),
                        S_mean=None,
                        N_sum=None,
                        draw=draw,
                    )
                )
    pb = pl.DataFrame(rows, infer_schema_length=None)
    pb.write_parquet(RES_DIR / "placebo_beta.parquet")
    out = []
    for ev in list(EVENTS) + ["pooled"]:
        for nm, cols in SPECS.items():
            if ev == "pooled":
                xp = dict(
                    XtWX=sum((act[e]["st"][nm]["xp"]["XtWX"] for e in EVENTS)),
                    XtWy=sum((act[e]["st"][nm]["xp"]["XtWy"] for e in EVENTS)),
                )
                coef = dict(zip(cols, solve(xp)))
            else:
                coef = act[ev]["st"][nm]["coef"]
            for term in cols:
                p = pb.filter(
                    (pl.col("event") == ev)
                    & (pl.col("spec") == nm)
                    & (pl.col("term") == term)
                )["coef"].to_numpy()
                out.append(
                    dict(
                        event=ev,
                        spec=nm,
                        term=term,
                        actual=float(coef[term]),
                        n_placebo=len(p),
                        ri_p=(
                            float(np.mean(np.abs(p) >= abs(coef[term])))
                            if len(p)
                            else None
                        ),
                        pctile=(
                            float(100 * np.mean(p <= coef[term])) if len(p) else None
                        ),
                        placebo_mean=float(p.mean()) if len(p) else None,
                        placebo_sd=float(p.std(ddof=1)) if len(p) > 1 else None,
                    )
                )
    out = pl.DataFrame(out)
    out.write_csv(RES_DIR / "beta_ri.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=3):
        print(out)


# Stage orchestration
def run(paths, cfg, log):
    from Run_All import invoke

    invoke(paths, log, "Inference", "build_full_panel", [cfg["full_panel_workers"]])
    invoke(paths, log, "Inference", "placebo")
    invoke(paths, log, "Inference", "placebo_beta")
    return sorted(paths.full.iterdir()) + [
        paths.results / f
        for f in [
            "placebo_lambda.parquet",
            "placebo_lambda_events.parquet",
            "lambda_ri.csv",
            "table2_placebo_sd.csv",
            "lambda_density_ri.csv",
            "placebo_beta.parquet",
            "beta_ri.csv",
        ]
    ]


TASKS = {
    "build_full_panel": build_full_panel_main,
    "placebo": placebo_main,
    "placebo_beta": placebo_beta_main,
}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        TASKS[task]()
    else:
        from Run_All import main

        main("Inference")
