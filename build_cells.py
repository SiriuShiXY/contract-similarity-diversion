"""
Cell-day panel for the Mahalanobis-distance baseline of the similarity / diversion design
(similarity_diversion_design.pdf, Sections 2-4 and 11.1), built from the raw Cboe trade files.

  2016-08-30  SPY adds Wednesday expirations      control QQQ, IWM
  2018-02-16  SPY adds Monday expirations         control QQQ, IWM
  2021-04-27  QQQ adds Monday and Wednesday       control SPY, IWM   (Monday expirations trade from 2021-04-23)
  2021-10-08  IWM adds Wednesday and Monday       control SPY, QQQ   (Wednesday expirations trade from 2021-10-05)
  2022-11-16  SPY, QQQ add Tuesday and Thursday   control IWM   (Tuesday expirations trade from 2022-11-14)

Window and new/existing classification: data/events/<event>/days.csv (60 pre-event sessions with a raw
file, transition sessions, day 0 and 59 more, early-close flags) and expirations.csv (effective expiration
weekday, is_new), from event_windows.py, which reproduces ListingDynamics' files for 2016, 2018 and 2022.
Trades are read from the local raw files, or from the T7 archive where there is none (2021).

Trades: as in build_listing_agg.py -- not cancelled, size > 0, price > 0, 09:30-16:15, records with
the paper's trade key merged -- and underlying mid > 0.

Characteristics (G1), per trade:
  tau_h  trading hours from the trade to the close of the last session on or before the expiration
         date (sessions 09:30-16:00, early closes 13:00; overnight, weekends and holidays count zero).
  |delta| Black-Scholes forward delta N(d1) for calls, N(-d1) for puts, at the product's common
         volatility sigma_bar on the same trading-time clock: d1 = (ln(F/K) + sigma_bar^2 tau / 2) /
         (sigma_bar sqrt(tau)), tau = tau_h / (252 x 6.5), F = S exp((r - q) tau_cal), r from the OIS curve
         (Raw Data/sofr.csv, interpolated at the calendar maturity), q = trailing dividend yield
         (Raw Data/Dividends.xlsx), the paper's inputs.
  sigma_bar  same-day 30-day at-the-money implied volatility of the product.  Baseline
         (SIGMA_SOURCE = "optionmetrics"): OptionMetrics Volatility Surface (WRDS vsurfd), 30 days,
         average of the delta 50 call and delta -50 put implied volatilities, as in An, Ang, Bali and
         Cakici (2014, JF); its 30 calendar days are put on the trading-time clock,
         iv x sqrt((30/365) / tau(close of t -> close of the last session on or before t + 30)).
         Robustness (SIGMA_SOURCE = "trades"): from the product's own trades, 09:55-15:45, vendor |delta|
         in (0.375, 0.625], IV > 0, expirations with 7-120 calendar DTE; each IV put on the trading-time
         clock (iv sqrt(tau_cal / tau)), total variance interpolated between the two expirations around
         30 calendar days.  Both are written to sigbar.csv.

Cells: product x day x effective expiration weekday x calendar DTE bucket {0, 1, 2-6, 7-30, 31+}
(DTE counted to the last trading session, which differs from the expiration date only for the one
Saturday-dated SPY series in 2016) x delta bucket at the common volatility (Deep OTM <= 0.125 < OTM
<= 0.375 < ATM <= 0.625 < ITM <= 0.875 < Deep ITM, design 2.2) x call/put, split into new and existing.

Per cell-day: volume over all trades (delta notional |delta| x S x 100 x size at the common-vol delta,
contracts, trades; design 2.3) and premium excluding deep-ITM trades (shell Table 2 notes; premium_all keeps
them), and characteristics over the liquidity window 09:55-15:45: delta-notional-weighted means of log tau_h
and |delta| (contract-weighted versions kept as robustness), after winsorizing both at 0.25% / 99.75% within
product x DTE bucket x day.  Deep-ITM trades are left out of the characteristics (design 3.6), so Deep ITM
cells have volume but no characteristics: they enter the volume tables, lambda and the control volume, not
the distances.  (Until 2026-11-02 deep-ITM trades were dropped altogether; SIM_VARIANT=no_ditm.)

Robustness (design 3.2, own implied volatilities; user's choice 2026-11-01): SIM_VARIANT=vendor_delta takes
|delta| from Cboe's trade delta (trade_delta, at the contract's own implied volatility trade_iv) for the delta
bucket, the delta notional and the |delta| characteristic; trades without one (trade_iv <= 0 or delta 0) are
dropped (design 3.6).  Every pipeline output then goes under data/robustness/vendor_delta/ and
results/robustness/vendor_delta/.  SIM_VARIANT=no_ditm drops deep-ITM trades from every volume measure (the
baseline until 2026-11-02, kept as a robustness check).

Usage: python build_cells.py [event ...]          (SIM_VARIANT=vendor_delta python build_cells.py: robustness)
Output: data/<event>/cells.parquet, data/<event>/sigbar.csv
"""
from __future__ import annotations

import math
import os
import sys
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import polars as pl
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
LD = ROOT.parent / "ListingDynamics"
RAW_DIR = Path("/Users/amadeus/Desktop/0DTE Options/0DTEOptions_TradeData")
RAW_DATA = ROOT.parents[1] / "Raw Data"
ARCHIVE = Path("/Volumes/T7/Options/OptionsTradeData.zip")   # complete daily trade files (2012-01 to 2025-07)
## SIM_VARIANT (environment): "" = baseline; "vendor_delta" = Cboe's own trade delta; "no_ditm" = deep-ITM trades
## dropped from every volume measure (robustness, see above).
VARIANT = os.environ.get("SIM_VARIANT", "")
assert VARIANT in ("", "vendor_delta", "no_ditm"), f"unknown SIM_VARIANT {VARIANT!r}"
DELTA_SOURCE = "vendor" if VARIANT == "vendor_delta" else "common"
DROP_DITM = VARIANT == "no_ditm"
DELTA_BUCKETS = ["DOTM", "OTM", "ATM", "ITM"] + ([] if DROP_DITM else ["DITM"])
OUT_DIR = ROOT / "data" / "robustness" / VARIANT if VARIANT else ROOT / "data"
RES_DIR = ROOT / "results" / "robustness" / VARIANT if VARIANT else ROOT / "results"
EV_DIR = ROOT / "data" / "events"                            # windows and expiration classes (event_windows.py)
SYMBOLS = ["SPY", "QQQ", "IWM"]

## The events (table_shells Table 1 rows 1, 2, 3, 4, 6), keyed by day 0.  pre_end: first listing day (the
## pre-event window ends the session before); start: first listing day for the new/existing classification;
## new_wd: weekdays added; n_e: weekday expirations per week before the event; control: other index families
## (Table 1); control_treated: controls that already list the added weekdays (their own expirations on those
## weekdays play the new contracts' part in eq. 18 and Table 3; user's choice 2026-10-31); cut: 2022-11-10
## listing cut of Monday/Wednesday series beyond 14 days (design 7.4).  tables_only: shown in Tables 1 and 2
## only, not in lambda, the distances or the regressions (IWM 2024, shell Table 1 row 7; user's choice 2026-11-02;
## first trades of the new Tuesday and Thursday series 2024-04-16 and 2024-04-18).
EVENT_SPEC = {
    "2016-08-30": dict(day0=date(2016, 8, 30), pre_end=date(2016, 8, 30), start=date(2016, 8, 30), treated=["SPY"],
                       control=["QQQ", "IWM"], control_treated=[], new_wd=[3], n_e=1, cut=False,
                       label="2016: SPY adds Wednesday", short="2016", tab="SPY 2016 (Wed)"),
    "2018-02-16": dict(day0=date(2018, 2, 16), pre_end=date(2018, 2, 16), start=date(2018, 2, 16), treated=["SPY"],
                       control=["QQQ", "IWM"], control_treated=[], new_wd=[1], n_e=2, cut=False,
                       label="2018: SPY adds Monday", short="2018", tab="SPY 2018 (Mon)"),
    "2021-04-27": dict(day0=date(2021, 4, 27), pre_end=date(2021, 4, 23), start=date(2021, 4, 23), treated=["QQQ"],
                       control=["SPY", "IWM"], control_treated=["SPY"], new_wd=[1, 3], n_e=1, cut=False,
                       label="2021: QQQ adds Mon/Wed", short="2021 QQQ", tab="QQQ 2021 (Mon, Wed)"),
    "2021-10-08": dict(day0=date(2021, 10, 8), pre_end=date(2021, 10, 5), start=date(2021, 10, 5), treated=["IWM"],
                       control=["SPY", "QQQ"], control_treated=["SPY", "QQQ"], new_wd=[1, 3], n_e=1, cut=False,
                       label="2021: IWM adds Mon/Wed", short="2021 IWM", tab="IWM 2021 (Mon, Wed)"),
    "2022-11-16": dict(day0=date(2022, 11, 16), pre_end=date(2022, 11, 14), start=date(2022, 11, 14),
                       treated=["SPY", "QQQ"], control=["IWM"], control_treated=[], new_wd=[2, 4], n_e=3, cut=True,
                       label="2022: SPY, QQQ add Tue/Thu", short="2022", tab="SPY/QQQ 2022 (Tue, Thu)"),
    "2024-04-18": dict(day0=date(2024, 4, 18), pre_end=date(2024, 4, 16), start=date(2024, 4, 16), treated=["IWM"],
                       control=["SPY", "QQQ"], control_treated=["SPY", "QQQ"], new_wd=[2, 4], n_e=3, cut=False,
                       label="2024: IWM adds Tue/Thu", short="2024 IWM", tab="IWM 2024 (Tue, Thu)", tables_only=True),
}
EVENTS = [ev for ev, s in EVENT_SPEC.items() if not s.get("tables_only")]     # the baseline events (every step)
TABLE_EVENTS = list(EVENT_SPEC)                                                 # Tables 1 and 2
BUILD_EVENTS = TABLE_EVENTS if not VARIANT else EVENTS                          # events whose cells a run builds
## sessions whose trade file lacks one of the three ETFs (2021-09-29: no QQQ trades at all), treated like
## sessions without a file: skipped in the event windows and in the full-sample panel
INCOMPLETE_DAYS = {date(2021, 9, 29)}

COLS = ["underlying_symbol", "quote_datetime", "sequence_number", "expiration", "strike", "option_type",
        "exchange_id", "trade_size", "trade_price", "trade_condition_id", "canceled_trade_condition_id",
        "best_bid", "best_ask", "trade_iv", "trade_delta", "underlying_bid", "underlying_ask"]
FLOATS = ["strike", "trade_size", "trade_price", "best_bid", "best_ask", "trade_iv", "trade_delta",
          "underlying_bid", "underlying_ask"]
TRADE_KEY = ["underlying_symbol", "expiration", "strike", "option_type", "quote_dt", "sequence_number",
             "exchange_id", "trade_price", "trade_condition_id"]
CELL = ["underlying_symbol", "trade_date", "eff_wd", "is_new", "dte_bucket", "delta_bucket", "option_type"]

OIS_TENORS = [("OIS1W", 7), ("OIS2W", 14), ("OIS3W", 21), ("OIS1M", 30), ("OIS2M", 60), ("OIS3M", 90),
              ("OIS4M", 120), ("OIS5M", 150), ("OIS6M", 180), ("OIS8M", 240), ("OIS9M", 270),
              ("OIS10M", 300), ("OIS11M", 330), ("OIS12M", 360), ("OIS2Y", 720), ("OIS3Y", 1080),
              ("OIS4Y", 1440)]
HOURS_PER_YEAR = 252 * 6.5
MIN_TAU_H = 1 / 60                       # one minute: floor for trades at or after the close
CHAR_WINDOW = (9 * 60 + 55, 15 * 60 + 45)
WINSOR_Q = (0.0025, 0.9975)
ATM_DTE = (7, 120)
ATM_MIN_TRADES = 10
TARGET_DAYS = 30
SIGMA_SOURCE = "optionmetrics"           # "optionmetrics" (baseline) or "trades"
## WRDS vsurfd downloads: 2016-2023 and 2020-2025-08 (the overlap 2020-2023 is identical, checked 2026-11-02)
OM_FILES = sorted((ROOT / "data" / "optionmetrics").glob("vsurfd_30d_SPY_QQQ_IWM_*.csv"))
OM_SECID = {109820: "SPY", 107899: "QQQ", 106445: "IWM"}


## ---------------------------------------------------------------- calendar
_sch = mcal.get_calendar("NYSE").schedule("2015-01-01", "2025-12-31")
_close = _sch["market_close"].dt.tz_convert("America/New_York")
SESSIONS = [d.date() for d in _sch.index]
SESSION_IDX = {d: i for i, d in enumerate(SESSIONS)}
CLOSE_MIN = {d: c.hour * 60 + c.minute for d, c in zip(SESSIONS, _close)}
CUM_HOURS = np.cumsum([(c.hour * 60 + c.minute - 570) / 60 for c in _close])


def last_session(d: date) -> date:
    while d not in SESSION_IDX:
        d -= timedelta(1)
    return d


def hours_after(day: date, last: date) -> float:
    """Trading hours in the sessions after `day` up to and including `last`."""
    return float(CUM_HOURS[SESSION_IDX[last]] - CUM_HOURS[SESSION_IDX[day]]) if last > day else 0.0


## ---------------------------------------------------------------- rates and dividends
def ois_curve() -> pd.DataFrame:
    c = pd.read_csv(RAW_DATA / "sofr.csv")
    c["trade_date"] = pd.to_datetime(c["Date"], format="%m/%d/%y").dt.date
    return c.set_index("trade_date")[[t for t, _ in OIS_TENORS]].astype(float).sort_index() / 100.0


def dividend_yields() -> pd.DataFrame:
    d = pd.read_excel(RAW_DATA / "Dividends.xlsx", sheet_name="Dividends_TTM")
    d["trade_date"] = pd.to_datetime(d["date"]).dt.date
    d = d.set_index("trade_date")[[f"Dividend_TTM_{s}" for s in SYMBOLS]].sort_index() / 100.0
    d.columns = SYMBOLS
    return d


def asof(frame: pd.DataFrame, day: date) -> pd.Series:
    sub = frame[[d <= day for d in frame.index]]
    assert len(sub), f"no rate/dividend observation on or before {day}"
    return sub.ffill().iloc[-1]


def delta_bucket(absd: pl.Expr) -> pl.Expr:
    return (pl.when(absd <= 0.125).then(pl.lit("DOTM")).when(absd <= 0.375).then(pl.lit("OTM"))
            .when(absd <= 0.625).then(pl.lit("ATM")).when(absd <= 0.875).then(pl.lit("ITM"))
            .otherwise(pl.lit("DITM")))


## ---------------------------------------------------------------- one session
def raw_csv(day: date):
    """The day's trade file: the local raw file if there is one, else the CSV bytes from the T7 archive."""
    local = RAW_DIR / f"UnderlyingOptionsTradesCalcs_{day}.csv"
    if local.exists():
        return local
    import io
    import zipfile
    with zipfile.ZipFile(ARCHIVE) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read(f"OptionsTradeData/UnderlyingOptionsTradesCalcs_{day}.zip")))
        return inner.read(f"UnderlyingOptionsTradesCalcs_{day}.csv")


def load_trades(day: date, src=None) -> pl.DataFrame:
    """src: the day's CSV as bytes (full-sample panel, read from the archive); default raw_csv(day)."""
    df = (pl.read_csv(raw_csv(day) if src is None else src, columns=COLS,
                      infer_schema_length=0, truncate_ragged_lines=True)
          .filter(pl.col("underlying_symbol").is_in(SYMBOLS))
          .with_columns([pl.col(c).cast(pl.Float64) for c in FLOATS]
                        + [pl.col("canceled_trade_condition_id").cast(pl.Int64),
                           pl.col("quote_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f").alias("quote_dt"),
                           pl.col("expiration").str.to_date()])
          .filter((pl.col("canceled_trade_condition_id") == 0) & (pl.col("trade_size") > 0)
                  & (pl.col("trade_price") > 0))
          ## minutes after midnight; dt.hour() is Int8, so cast before multiplying
          .with_columns((pl.col("quote_dt").dt.hour().cast(pl.Int32) * 60 + pl.col("quote_dt").dt.minute().cast(pl.Int32)
                         + pl.col("quote_dt").dt.second().cast(pl.Float64) / 60
                         + pl.col("quote_dt").dt.microsecond().cast(pl.Float64) / 6e7).alias("tmin"))
          .filter(pl.col("tmin").is_between(9 * 60 + 30, 16 * 60 + 15))
          .group_by(TRADE_KEY)
          .agg(pl.col("trade_size").sum(), pl.exclude(TRADE_KEY + ["trade_size"]).first())
          .with_columns(((pl.col("underlying_bid") + pl.col("underlying_ask")) / 2).alias("S"))
          .filter(pl.col("S") > 0))
    assert df.height and (df["quote_dt"].dt.date() == day).all(), f"{day}: trades from another date"
    return df


def expiration_table(day: date, df: pl.DataFrame, expi: pl.DataFrame, ois: pd.Series, q: pd.Series) -> pl.DataFrame:
    """One row per (product, expiration) traded on `day`: classification, last session, trading hours after
    today, calendar DTE to the last session, OIS rate at the calendar maturity, dividend yield."""
    ex = df.select("underlying_symbol", "expiration").unique()
    ex = ex.join(expi, on=["underlying_symbol", "expiration"], how="left")
    assert ex["is_new"].null_count() == 0, f"{day}: expirations missing from ListingDynamics expirations.csv"
    tenor_days = np.array([d for _, d in OIS_TENORS], dtype=float)
    tenor_rates = ois[[t for t, _ in OIS_TENORS]].to_numpy(dtype=float)
    rows = []
    for sym, e in ex.select("underlying_symbol", "expiration").iter_rows():
        last = last_session(e)
        rows.append((sym, e, last, hours_after(day, last), (last - day).days,
                     float(np.interp(max((e - day).days, 0.5), tenor_days, tenor_rates)), float(q[sym])))
    extra = pl.DataFrame(rows, schema=["underlying_symbol", "expiration", "last_session", "hours_after", "dte",
                                       "r", "q"], orient="row")
    return ex.join(extra, on=["underlying_symbol", "expiration"])


@lru_cache(maxsize=1)
def optionmetrics_vol() -> dict[tuple[str, date], tuple[float, float]]:
    """(product, day) -> (OptionMetrics 30-day delta-50 call/put average IV, same on the trading-time clock)."""
    key = ["secid", "date", "days", "delta", "cp_flag"]
    v = pl.concat([pl.read_csv(f, try_parse_dates=True, infer_schema_length=100000).select(key + ["impl_volatility"])
                   for f in OM_FILES]).unique(subset=key, keep="first")
    a = (v.filter((pl.col("days") == 30) & (pl.col("delta").abs() == 50) & pl.col("secid").is_in(list(OM_SECID)))
         .group_by("secid", "date")
         .agg(pl.col("impl_volatility").filter(pl.col("cp_flag") == "C").first().alias("c"),
              pl.col("impl_volatility").filter(pl.col("cp_flag") == "P").first().alias("p")))
    out = {}
    for secid, d, c, p in a.select("secid", "date", "c", "p").iter_rows():
        if c is None or p is None:
            continue
        iv = (c + p) / 2
        tau_tr = hours_after(d, last_session(d + timedelta(TARGET_DAYS))) / HOURS_PER_YEAR
        out[(OM_SECID[secid], d)] = (iv, iv * math.sqrt((TARGET_DAYS / 365) / tau_tr))
    return out


def common_vol(day: date, df: pl.DataFrame) -> tuple[dict[str, float], list[dict]]:
    """Same-day 30-calendar-day ATM IV per product on the trading-time clock."""
    noon_target = (CLOSE_MIN[day] - 12 * 60) / 60 + hours_after(day, last_session(day + timedelta(TARGET_DAYS)))
    tau_star = noon_target / HOURS_PER_YEAR
    atm = (df.filter(pl.col("tmin").is_between(*CHAR_WINDOW) & (pl.col("trade_iv") > 0)
                     & pl.col("trade_delta").abs().is_between(0.375, 0.625, closed="right")
                     & pl.col("dte").is_between(*ATM_DTE))
           .with_columns((pl.col("trade_iv") * (pl.col("tau_cal") / pl.col("tau")).sqrt()).alias("iv_tr"))
           .group_by("underlying_symbol", "expiration", "dte")
           .agg(pl.len().alias("n"),
                ((pl.col("iv_tr") * pl.col("trade_size")).sum() / pl.col("trade_size").sum()).alias("iv_tr"),
                ((pl.col("tau") * pl.col("trade_size")).sum() / pl.col("trade_size").sum()).alias("tau"))
           .filter(pl.col("n") >= ATM_MIN_TRADES).sort("underlying_symbol", "tau"))
    sig, diag = {}, []
    for sym in SYMBOLS:
        a = atm.filter(pl.col("underlying_symbol") == sym)
        assert a.height, f"{day} {sym}: no ATM trades for the common volatility"
        lo, hi = a.filter(pl.col("tau") <= tau_star).tail(1), a.filter(pl.col("tau") > tau_star).head(1)
        if lo.height and hi.height:
            w_lo, w_hi = lo["iv_tr"][0] ** 2 * lo["tau"][0], hi["iv_tr"][0] ** 2 * hi["tau"][0]
            w = w_lo + (w_hi - w_lo) * (tau_star - lo["tau"][0]) / (hi["tau"][0] - lo["tau"][0])
            sig[sym] = float(np.sqrt(w / tau_star))
        else:
            sig[sym] = float((lo if lo.height else hi)["iv_tr"][0])
        diag.append(dict(underlying_symbol=sym, trade_date=day, sigbar=sig[sym],
                         dte_lo=lo["dte"][0] if lo.height else None, iv_lo=lo["iv_tr"][0] if lo.height else None,
                         n_lo=lo["n"][0] if lo.height else None,
                         dte_hi=hi["dte"][0] if hi.height else None, iv_hi=hi["iv_tr"][0] if hi.height else None,
                         n_hi=hi["n"][0] if hi.height else None))
    return sig, diag


def cells_for_day(day: date, expi: pl.DataFrame, ois: pd.DataFrame, divs: pd.DataFrame, src=None,
                  strict: bool = True) -> tuple[pl.DataFrame, list[dict]]:
    """strict=False (full-sample panel): a day on which the trade-based sigma_bar (a diagnostic, not the
    baseline) cannot be computed is kept, with that diagnostic left empty."""
    df = load_trades(day, src)
    ex = expiration_table(day, df, expi, asof(ois, day), asof(divs, day))
    df = df.join(ex, on=["underlying_symbol", "expiration"], how="left")
    df = df.with_columns(
        ((pl.lit(CLOSE_MIN[day]) - pl.col("tmin")).clip(lower_bound=0) / 60 + pl.col("hours_after"))
        .clip(lower_bound=MIN_TAU_H).alias("tau_h"),
        ((pl.col("expiration").cast(pl.Datetime("us")) + pl.duration(hours=16) - pl.col("quote_dt"))
         .dt.total_seconds() / (365 * 86400)).clip(lower_bound=MIN_TAU_H / (365 * 24)).alias("tau_cal"),
    ).with_columns((pl.col("tau_h") / HOURS_PER_YEAR).alias("tau"))
    try:
        sig_trades, diag = common_vol(day, df)
    except AssertionError:
        if strict:
            raise
        sig_trades, diag = {}, []
    om = optionmetrics_vol()
    assert all((s, day) in om for s in SYMBOLS), f"{day}: OptionMetrics 30-day ATM IV missing"
    sig = {s: om[(s, day)][1] for s in SYMBOLS} if SIGMA_SOURCE == "optionmetrics" else sig_trades
    for g in diag:
        s = g["underlying_symbol"]
        g.update(sigbar_trades=g.pop("sigbar"), sigbar_om_raw=om[(s, day)][0], sigbar_om=om[(s, day)][1], sigbar=sig[s])
    df = df.with_columns(pl.col("underlying_symbol").replace_strict(sig, return_dtype=pl.Float64).alias("sigbar"))
    ## forward Black-Scholes |delta| at the common volatility
    F = (df["S"] * ((df["r"] - df["q"]) * df["tau_cal"]).exp()).to_numpy()
    sg, tau = df["sigbar"].to_numpy(), df["tau"].to_numpy()
    d1 = (np.log(F / df["strike"].to_numpy()) + 0.5 * sg ** 2 * tau) / (sg * np.sqrt(tau))
    absd_cv = np.where((df["option_type"] == "C").to_numpy(), ndtr(d1), ndtr(-d1))
    ## Cboe's trade delta (own implied volatility); missing when Cboe has no implied volatility for the trade
    absd_v = (pl.when((pl.col("trade_iv") > 0) & (pl.col("trade_delta") != 0))
              .then(pl.col("trade_delta").abs().clip(upper_bound=1.0)))
    df = df.with_columns(pl.Series("absd_cv", absd_cv), absd_v.alias("absd_v"), pl.lit(day).alias("trade_date"))
    if DELTA_SOURCE == "vendor":
        df = df.filter(pl.col("absd_v").is_not_null())
    df = df.with_columns(pl.col("absd_v" if DELTA_SOURCE == "vendor" else "absd_cv").alias("absd"))
    df = df.with_columns(
        delta_bucket(pl.col("absd")).alias("delta_bucket"),
        delta_bucket(pl.col("absd_cv")).alias("common_bucket"),
        pl.when(pl.col("absd_v").is_not_null()).then(delta_bucket(pl.col("absd_v"))).alias("vendor_bucket"),
        pl.when(pl.col("dte") == 0).then(pl.lit("D0")).when(pl.col("dte") == 1).then(pl.lit("D1"))
        .when(pl.col("dte") <= 6).then(pl.lit("D2-6")).when(pl.col("dte") <= 30).then(pl.lit("D7-30"))
        .otherwise(pl.lit("D31+")).alias("dte_bucket"),
        (pl.col("absd") * pl.col("S") * 100 * pl.col("trade_size")).alias("dn"),
        pl.col("tau_h").log().alias("logtau"),
        ## characteristics: liquidity window, deep-ITM trades left out (design 3.6)
        (pl.col("tmin").is_between(*CHAR_WINDOW) & (delta_bucket(pl.col("absd")) != "DITM")).alias("char"),
    )
    if DROP_DITM:
        df = df.filter(pl.col("delta_bucket") != "DITM")
    ## winsorize the characteristics within product x DTE bucket x day (liquidity window only)
    wgrp = ["underlying_symbol", "dte_bucket"]
    caps = (df.filter(pl.col("char")).group_by(wgrp)
            .agg([pl.col(c).quantile(q, interpolation="nearest").alias(f"{c}_{i}")
                  for c in ("logtau", "absd") for i, q in enumerate(WINSOR_Q)]))
    df = (df.join(caps, on=wgrp, how="left")
          .with_columns([pl.col(c).clip(pl.col(f"{c}_0"), pl.col(f"{c}_1")).alias(f"{c}_w") for c in ("logtau", "absd")]))
    ch, dn, ct = pl.col("char"), pl.col("dn"), pl.col("trade_size")
    cells = df.group_by(CELL).agg(
        dn.sum().alias("V"),
        ct.sum().alias("contracts"),
        (ct * pl.col("trade_price") * 100).filter(pl.col("delta_bucket") != "DITM").sum().alias("premium"),
        (ct * pl.col("trade_price") * 100).sum().alias("premium_all"),
        pl.len().alias("trades"),
        pl.col("expiration").n_unique().alias("n_expirations"),
        ch.sum().alias("n_char"),
        dn.filter(ch).sum().alias("V_char"),
        ((dn * pl.col("logtau_w")).filter(ch).sum() / dn.filter(ch).sum()).alias("x_logtau"),
        ((dn * pl.col("absd_w")).filter(ch).sum() / dn.filter(ch).sum()).alias("x_absd"),
        ((ct * pl.col("logtau_w")).filter(ch).sum() / ct.filter(ch).sum()).alias("x_logtau_ct"),
        ((ct * pl.col("absd_w")).filter(ch).sum() / ct.filter(ch).sum()).alias("x_absd_ct"),
        ## diagnostic: contracts with a vendor delta, and those whose vendor-delta bucket differs from the
        ## common-volatility bucket
        ct.filter(pl.col("vendor_bucket").is_not_null()).sum().alias("contracts_vendor"),
        ct.filter(pl.col("vendor_bucket") != pl.col("common_bucket")).sum().alias("contracts_vendor_diff"),
        pl.col("sigbar").first(),
    )
    return cells, diag


def main() -> None:
    ois, divs = ois_curve(), dividend_yields()
    for ev in sys.argv[1:] or BUILD_EVENTS:
        t0 = time.time()
        days = pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
        expi = (pl.read_csv(EV_DIR / ev / "expirations.csv", try_parse_dates=True)
                .select("underlying_symbol", "expiration", "eff_wd", "is_new", "first_trade"))
        parts, diags = [], []
        for d in days.filter(pl.col("has_file"))["trade_date"]:
            c, g = cells_for_day(d, expi, ois, divs)
            parts.append(c)
            diags.extend(g)
        cells = pl.concat(parts).join(days.select("trade_date", "tau", "period", "early_close"), on="trade_date", how="left")
        assert cells["period"].null_count() == 0
        out = OUT_DIR / ev
        out.mkdir(parents=True, exist_ok=True)
        cells.rename({"tau": "event_time"}).write_parquet(out / "cells.parquet")
        pl.DataFrame(diags).write_csv(out / "sigbar.csv")
        print(f"[{ev}] days={len(parts)} cell-days={cells.height:,} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
