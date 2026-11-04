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
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from build_cells import (ARCHIVE, CLOSE_MIN, DELTA_BUCKETS, DELTA_SOURCE, OUT_DIR, SESSION_IDX, SIGMA_SOURCE, SYMBOLS,
                         VARIANT, cells_for_day, dividend_yields, ois_curve, optionmetrics_vol)

FULL_DIR = OUT_DIR / "full"
## the cell settings the year files depend on (full/spec.json); files built under other settings are rebuilt
SPEC = dict(variant=VARIANT, delta_source=DELTA_SOURCE, delta_buckets=DELTA_BUCKETS, sigma_source=SIGMA_SOURCE)
START, END = date(2016, 1, 6), date(2023, 12, 29)
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
    if wd == 2 and d + timedelta(1) in CLOSED and sym == "SPY":
        return 3
    if wd == 4 and d + timedelta(1) in CLOSED:
        return 5
    return wd


def archive_days_all() -> list[date]:
    with zipfile.ZipFile(ARCHIVE) as z:
        names = [n.split("_")[-1][:10] for n in z.namelist() if n.endswith(".zip")]
    return sorted(date.fromisoformat(n) for n in names)


def archive_days() -> list[date]:
    return [d for d in archive_days_all() if START <= d <= END]


def read_day(z: zipfile.ZipFile, d: date) -> bytes:
    inner = zipfile.ZipFile(io.BytesIO(z.read(f"OptionsTradeData/UnderlyingOptionsTradesCalcs_{d}.zip")))
    return inner.read(f"UnderlyingOptionsTradesCalcs_{d}.csv")


def build_year(args: tuple[int, bool]) -> str:
    year, reuse = args
    out = FULL_DIR / f"cells_{year}.parquet"
    if reuse and out.exists():
        return f"{year}: exists"
    t0 = time.time()
    ois, divs = ois_curve(), dividend_yields()
    days = [d for d in archive_days() if d.year == year]
    om = optionmetrics_vol()
    cells, expis, diags, skipped = [], [], [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        for d in days:
            if not all((s, d) in om for s in SYMBOLS):
                skipped.append(d)               # no OptionMetrics common volatility that day
                continue
            raw = read_day(z, d)
            ## expirations traded today (same trade filters as the cells: load_trades runs again inside)
            ex = (pl.read_csv(raw, columns=["underlying_symbol", "expiration", "trade_size", "canceled_trade_condition_id"],
                              infer_schema_length=0, truncate_ragged_lines=True)
                  .filter(pl.col("underlying_symbol").is_in(SYMBOLS)
                          & (pl.col("canceled_trade_condition_id").cast(pl.Int64) == 0))
                  .group_by("underlying_symbol", "expiration").agg(pl.col("trade_size").cast(pl.Float64).sum().alias("contracts"))
                  .with_columns(pl.col("expiration").str.to_date()))
            ex = ex.with_columns(pl.struct("expiration", "underlying_symbol")
                                 .map_elements(lambda r: effective_weekday(r["expiration"], r["underlying_symbol"]),
                                               return_dtype=pl.Int64).alias("eff_wd"),
                                 pl.lit(False).alias("is_new"), pl.lit(d).alias("first_trade"))
            c, g = cells_for_day(d, ex.select("underlying_symbol", "expiration", "eff_wd", "is_new", "first_trade"),
                                 ois, divs, src=raw, strict=False)
            cells.append(c.with_columns(pl.lit(CLOSE_MIN[d] < 16 * 60).alias("early_close")))
            expis.append(ex.select("underlying_symbol", "expiration", "eff_wd", "contracts", pl.lit(d).alias("trade_date")))
            diags.extend(g)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(expis).write_parquet(FULL_DIR / f"expirations_{year}.parquet")
    pl.DataFrame(diags).write_csv(FULL_DIR / f"sigbar_{year}.csv")
    pl.concat(cells).write_parquet(out)            # written last: marks the year as done
    return (f"{year}: {len(cells)} sessions, {sum(c.height for c in cells):,} cell-days, skipped (no OptionMetrics) "
            f"{skipped} ({time.time() - t0:.0f}s)")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = FULL_DIR / "spec.json"
    reuse = stamp.exists() and json.loads(stamp.read_text()) == SPEC
    if not reuse:
        print(f"cell settings {SPEC} differ from the files' (or no spec.json): every year is rebuilt", flush=True)
    with Pool(n) as pool:
        for msg in pool.imap_unordered(build_year, [(y, reuse) for y in range(START.year, END.year + 1)]):
            print(msg, flush=True)
    stamp.write_text(json.dumps(SPEC))


if __name__ == "__main__":
    main()
