"""
Event windows and the new/existing classification of expirations for every event of the baseline, built
the same way as ListingDynamics (build_listing_agg.window, analyze_listing_dynamics.classify) from the
complete trade files on the T7 drive (build_full_panel.ARCHIVE).

  Window: event time counts sessions with a raw file (the paper's convention; a file without one of the
          three ETFs, 2021-09-29, counts as missing); 60 sessions before
          pre_end, the transition sessions [pre_end, day0) and day 0 plus 59 sessions.
  Expirations: every (ETF, expiration) traded in the window (not cancelled, size > 0, price > 0,
          09:30-16:15); first_trade = first trade in the window; eff_wd = the weekday it stands for
          (Saturday -> Friday, Tuesday after a Monday holiday -> Monday, Thursday before a Friday
          holiday -> Friday); is_new = treated ETF, eff_wd a weekday the event adds, first trade on or
          after the first listing day.

The three events of the paper (2016-08-30, 2018-02-16, 2022-11-16) must reproduce ListingDynamics'
days.csv and expirations.csv exactly (checked in main); the 2021 QQQ and IWM listings are added here.

Usage: python event_windows.py
Output: data/events/<event>/days.csv, data/events/<event>/expirations.csv
"""
from __future__ import annotations

import io
import zipfile
from multiprocessing import Pool

import polars as pl

from datetime import timedelta

from build_cells import ARCHIVE, CLOSE_MIN, EVENT_SPEC, EV_DIR, INCOMPLETE_DAYS, LD, SESSION_IDX, SYMBOLS, TABLE_EVENTS
from build_full_panel import archive_days_all

N_PRE, N_POST = 60, 60
CLOSED = {min(SESSION_IDX) + timedelta(i) for i in range((max(SESSION_IDX) - min(SESSION_IDX)).days)} - set(SESSION_IDX)


def effective_weekday(d) -> int:
    """ListingDynamics' rule (analyze_listing_dynamics.effective_weekday)."""
    wd = d.isoweekday()
    if wd == 6:
        return 5
    if wd == 2 and d - timedelta(1) in CLOSED:
        return 1
    if wd == 4 and d + timedelta(1) in CLOSED:
        return 5
    return wd


def window(ev: str, days: list) -> pl.DataFrame:
    spec = EVENT_SPEC[ev]
    i_pre, i0 = days.index(spec["pre_end"]), days.index(spec["day0"])
    pre, trans, post = days[i_pre - N_PRE:i_pre], days[i_pre:i0], days[i0:i0 + N_POST]
    assert len(pre) == N_PRE and len(post) == N_POST, (ev, len(pre), len(post))
    return pl.DataFrame({
        "trade_date": pre + trans + post,
        "tau": list(range(-N_PRE - len(trans), N_POST)),
        "period": ["pre"] * len(pre) + ["transition"] * len(trans) + ["post"] * len(post),
        "early_close": [CLOSE_MIN[d] < 16 * 60 for d in pre + trans + post],
        "has_file": [True] * (len(pre) + len(trans) + len(post)),
    })


def traded_expirations(d) -> pl.DataFrame:
    with zipfile.ZipFile(ARCHIVE) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read(f"OptionsTradeData/UnderlyingOptionsTradesCalcs_{d}.zip")))
        raw = inner.read(f"UnderlyingOptionsTradesCalcs_{d}.csv")
    return (pl.read_csv(raw, columns=["underlying_symbol", "quote_datetime", "expiration", "trade_size", "trade_price",
                                      "canceled_trade_condition_id"], infer_schema_length=0, truncate_ragged_lines=True)
            .filter(pl.col("underlying_symbol").is_in(SYMBOLS))
            .with_columns(pl.col("trade_size").cast(pl.Float64), pl.col("trade_price").cast(pl.Float64),
                          pl.col("canceled_trade_condition_id").cast(pl.Int64),
                          pl.col("quote_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f").dt.time().alias("t"))
            .filter((pl.col("canceled_trade_condition_id") == 0) & (pl.col("trade_size") > 0) & (pl.col("trade_price") > 0)
                    & pl.col("t").is_between(pl.time(9, 30), pl.time(16, 15)))
            .group_by("underlying_symbol", "expiration").agg(pl.col("trade_size").sum().alias("contracts"))
            .with_columns(pl.col("expiration").str.to_date(), pl.lit(d).alias("trade_date")))


def classify(ev: str, win: pl.DataFrame) -> pl.DataFrame:
    spec = EVENT_SPEC[ev]
    with Pool(8) as pool:
        parts = pool.map(traded_expirations, win["trade_date"].to_list())
    a = pl.concat(parts)
    e = (a.group_by("underlying_symbol", "expiration")
         .agg(pl.col("trade_date").min().alias("first_trade"), pl.col("contracts").sum().alias("contracts_window"))
         .with_columns(pl.col("expiration").dt.weekday().alias("exp_wd"),
                       pl.col("expiration").map_elements(effective_weekday, return_dtype=pl.Int64).alias("eff_wd")))
    e = e.with_columns((pl.col("eff_wd") != pl.col("exp_wd")).alias("holiday_shift"))
    is_new = (pl.col("underlying_symbol").is_in(spec["treated"]) & pl.col("eff_wd").is_in(spec["new_wd"])
              & (pl.col("first_trade") >= spec["start"]))
    return e.with_columns(is_new.alias("is_new")).sort("underlying_symbol", "expiration")


def main() -> None:
    days = [d for d in archive_days_all() if d not in INCOMPLETE_DAYS]
    for ev in TABLE_EVENTS:
        win = window(ev, days)
        ex = classify(ev, win)
        out = EV_DIR / ev
        out.mkdir(parents=True, exist_ok=True)
        win.write_csv(out / "days.csv")
        ex.write_csv(out / "expirations.csv")
        msg = f"[{ev}] window {win['trade_date'].min()}..{win['trade_date'].max()} ({win.height} sessions); " \
              f"expirations {ex.height}, new {int(ex['is_new'].sum())}"
        ## the paper's events: identical to ListingDynamics
        ld_days, ld_exp = LD / "data" / ev / "days.csv", LD / "results" / ev / "expirations.csv"
        if ld_days.exists():
            a = pl.read_csv(ld_days, try_parse_dates=True)
            b = pl.read_csv(out / "days.csv", try_parse_dates=True)
            assert a.equals(b.select(a.columns)), f"{ev}: days differ from ListingDynamics"
            a = pl.read_csv(ld_exp, try_parse_dates=True).sort("underlying_symbol", "expiration")
            b = pl.read_csv(out / "expirations.csv", try_parse_dates=True).select(a.columns)
            assert a.equals(b), f"{ev}: expirations differ from ListingDynamics"
            msg += "; identical to ListingDynamics"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
