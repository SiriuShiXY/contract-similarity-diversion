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
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from build_cells import EVENT_SPEC, EVENTS, EV_DIR, OUT_DIR, SESSION_IDX, last_session
from mahalanobis import RES_DIR
from panel import CFG, dte_bucket, listed_expirations

NEW_WD = {ev: s["new_wd"] for ev, s in EVENT_SPEC.items()}
OWN = {ev: s["control_treated"] for ev, s in EVENT_SPEC.items()}       # controls already listing the new weekdays
THETAS = [1, 2, 4]
SHARE_EDGES = [0.5, 1.0, 2.0]
X = ["x_logtau", "x_absd"]


def theta_from_structural() -> float:
    e = pl.read_csv(RES_DIR / "structural_estimates.csv")
    return float(e.filter(pl.col("version") == "L")["theta"][0])


def new_calendar(ev: str, days: pl.DataFrame, lst: pl.DataFrame) -> pl.DataFrame:
    cfg, new_wd = CFG[ev], NEW_WD[ev]
    treated, control = cfg["treated"], cfg["control"]
    act = (lst.filter(pl.col("is_new")).join(days.select("trade_date", "period", "wd"), on="trade_date")
           .filter(pl.col("period") == "post"))
    horizon = dict(act.group_by("wd").agg(pl.col("dte").max()).iter_rows())
    existing = {s: set(lst.filter((pl.col("underlying_symbol") == s) & ~pl.col("is_new"))["last_session"].to_list())
                for s in treated + control}
    own = OWN[ev]
    rows = []
    for sym, t, e in act.select("underlying_symbol", "trade_date", "last_session").iter_rows():
        rows.append((sym, t, e, "actual"))
    mirror = act.select("trade_date", "last_session").unique()
    for sym in control:
        if sym not in own:
            rows += [(sym, t, e, "mirror") for t, e in mirror.iter_rows()]
    for t, wd in days.filter(pl.col("period") == "pre").select("trade_date", "wd").iter_rows():
        for k in range(horizon[wd] + 1):
            d = t + timedelta(k)
            if d.isoweekday() in new_wd and d in SESSION_IDX:
                rows += [(sym, t, d, "phantom") for sym in treated + control if sym not in own and d not in existing[sym]]
    ## a control that already lists the new weekdays: its own expirations on those weekdays, before and after
    ## the event, within the same horizon (user's choice 2026-10-31)
    hz = max(horizon.values())
    o = (lst.filter(pl.col("underlying_symbol").is_in(own) & pl.col("eff_wd").is_in(new_wd))
         .join(days.select("trade_date", "wd"), on="trade_date"))
    for sym, t, wd, e, dte in o.select("underlying_symbol", "trade_date", "wd", "last_session", "dte").iter_rows():
        if dte <= horizon.get(wd, hz):
            rows.append((sym, t, e, "own"))
    cal = pl.DataFrame(rows, schema=["underlying_symbol", "trade_date", "last_session", "source"], orient="row")
    return (cal.unique(["underlying_symbol", "trade_date", "last_session"])
            .with_columns(pl.col("last_session").dt.weekday().alias("eff_wd_k"),
                          (pl.col("last_session") - pl.col("trade_date")).dt.total_days().alias("dte_k"))
            .with_columns(dte_bucket(pl.col("dte_k")).alias("dte_bucket_k")))


def profile(ev: str, cells: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    nw = (cells.filter(pl.col("is_new") & (pl.col("period") == "post") & ~pl.col("early_close") & (pl.col("V_char") > 0))
          .with_columns(pl.col("trade_date").dt.weekday().alias("wd"),
                        (pl.col("V") / pl.col("V").sum().over("underlying_symbol", "trade_date", "option_type")).alias("share"))
          .rename({"eff_wd": "eff_wd_k", "dte_bucket": "dte_bucket_k", "delta_bucket": "delta_bucket_k"}))
    out = []
    for key in (["wd", "eff_wd_k", "dte_bucket_k"], ["eff_wd_k", "dte_bucket_k"], ["dte_bucket_k"]):
        k = key + ["delta_bucket_k", "option_type"]
        out.append(nw.group_by(k).agg([((pl.col(c) * pl.col("V")).sum() / pl.col("V").sum()).alias(f"{c}_k") for c in X]
                                      + [pl.col("share").mean().alias("w_k")]))
    return tuple(out)


def build(ev: str, sig: pl.DataFrame, theta_hat: float) -> pl.DataFrame:
    cfg = CFG[ev]
    days = (pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
            .filter(pl.col("has_file") & ~pl.col("early_close") & pl.col("period").is_in(["pre", "post"]))
            .select("trade_date", "period", pl.col("trade_date").dt.weekday().alias("wd")))
    lst = listed_expirations(ev, days)
    lst = lst.filter(pl.col("underlying_symbol").is_in(cfg["treated"] + cfg["control"]))
    ## an already-treated control's expirations on the new weekdays play the new contracts' part
    own_new = pl.col("underlying_symbol").is_in(OWN[ev]) & pl.col("eff_wd").is_in(NEW_WD[ev])
    cal = new_calendar(ev, days, lst)
    cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet")
    p1, p2, p3 = profile(ev, cells)

    ## nearest-expiry indicator on this calendar
    e_star = (lst.filter(~pl.col("is_new") & ~own_new).group_by("underlying_symbol", "trade_date")
              .agg(pl.col("last_session").min().alias("e_star"),
                   pl.col("eff_wd").sort_by("last_session").first(), pl.col("dte_bucket").sort_by("last_session").first()))
    n_star = cal.group_by("underlying_symbol", "trade_date").agg(pl.col("last_session").min().alias("n_star"))
    nflag = (e_star.join(n_star, on=["underlying_symbol", "trade_date"])
             .filter(pl.col("n_star") < pl.col("e_star")).select("underlying_symbol", "trade_date", "eff_wd", "dte_bucket",
                                                                 pl.lit(1).alias("N_cal")))

    ## new cells with profile characteristics and weights
    nc = (cal.select("underlying_symbol", "trade_date", "eff_wd_k", "dte_bucket_k").unique()
          .join(pl.DataFrame({"delta_bucket_k": ["DOTM", "OTM", "ATM", "ITM"]}), how="cross")
          .join(pl.DataFrame({"option_type": ["C", "P"]}), how="cross")
          .with_columns(pl.col("trade_date").dt.weekday().alias("wd")))
    cols = [f"{c}_k" for c in X] + ["w_k"]
    k1, k2, k3 = (["wd", "eff_wd_k", "dte_bucket_k"], ["eff_wd_k", "dte_bucket_k"], ["dte_bucket_k"])
    nc = (nc.join(p1, on=k1 + ["delta_bucket_k", "option_type"], how="left")
          .join(p2.rename({c: f"{c}_2" for c in cols}), on=k2 + ["delta_bucket_k", "option_type"], how="left")
          .join(p3.rename({c: f"{c}_3" for c in cols}), on=k3 + ["delta_bucket_k", "option_type"], how="left")
          .with_columns(pl.col("w_k").is_null().alias("profile_fallback"))
          .with_columns([pl.coalesce(c, f"{c}_2", f"{c}_3").alias(c) for c in cols])
          .drop([f"{c}_{i}" for c in cols for i in (2, 3)]))
    assert nc["w_k"].null_count() == 0, f"{ev}: new cells without a profile"

    ## existing cells (actual characteristics; imputed where no liquidity-window trade; Deep ITM cells have none)
    ex = (pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet")
          .filter(pl.col("period").is_in(["pre", "post"]) & pl.col("underlying_symbol").is_in(cfg["treated"] + cfg["control"])
                  & ~own_new & (pl.col("delta_bucket") != "DITM")))
    ex = (ex.join(nflag, on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket"], how="left")
          .with_columns(pl.col("N_cal").fill_null(0)))
    pairs = ex.select("underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type", *X) \
              .join(nc, on=["underlying_symbol", "trade_date", "option_type"])
    inv = sig.filter(pl.col("event") == ev).select("underlying_symbol", "option_type", "sd_logtau", "sd_absd", "rho_lw")
    a, b = pl.col("x_logtau") - pl.col("x_logtau_k"), pl.col("x_absd") - pl.col("x_absd_k")
    det = 1 - pl.col("rho_lw") ** 2
    za, zb = a / pl.col("sd_logtau"), b / pl.col("sd_absd")
    pairs = (pairs.join(inv, on=["underlying_symbol", "option_type"])
             .with_columns(((za ** 2 - 2 * pl.col("rho_lw") * za * zb + zb ** 2) / det).alias("d2")))
    J = ["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
    w = pl.col("w_k")
    agg = pairs.group_by(J).agg(
        [((w * (-theta_hat * pl.col("d2")).exp()).sum() / w.sum()).alias("S_hat")]
        + [((w * (-th * pl.col("d2")).exp()).sum() / w.sum()).alias(f"S{th}") for th in THETAS]
        + [(w.filter(pl.col("d2").sqrt().is_between(lo, hi, closed="left")).sum() / w.sum()).alias(f"s{i + 1}")
           for i, (lo, hi) in enumerate(zip([0.0] + SHARE_EDGES[:-1], SHARE_EDGES))]
        + [pl.col("d2").min().sqrt().alias("d_min"), pl.col("profile_fallback").any().alias("profile_fallback")])
    out = ex.join(agg, on=J, how="left").with_columns(pl.lit(ev).alias("event"))
    assert out["S1"].null_count() == 0, f"{ev}: existing cell-days without a new cell of the same type"
    ## the calendar N must reproduce panel.py's N for the treated ETFs after the event
    chk = out.filter(pl.col("treated") & (pl.col("period") == "post"))
    assert (chk["N"] == chk["N_cal"]).all(), f"{ev}: calendar N differs from panel.py N"
    src = cal.group_by("source").len().sort("source").rows()
    print(f"[{ev}] new-expiration calendar rows by source {src}; new cells {nc.height:,} "
          f"(profile fallback {int(nc['profile_fallback'].sum())}); existing cell-days {out.height:,}", flush=True)
    return out


def main() -> None:
    sig = pl.read_csv(RES_DIR / "sigma.csv", schema_overrides={"event": pl.Utf8})
    theta_hat = theta_from_structural()
    parts = []
    for ev in EVENTS:
        out = build(ev, sig, theta_hat)
        out.write_parquet(OUT_DIR / ev / "did_panel.parquet")
        parts.append(out)
    keep = ["event", "underlying_symbol", "treated", "trade_date", "period", "wd", "eff_wd", "dte_bucket", "delta_bucket",
            "option_type", "cut_cell", "V", "Vpre", "N_cal", "d_min", "profile_fallback", "S_hat"] + [f"S{t}" for t in THETAS] + \
           [f"s{i + 1}" for i in range(len(SHARE_EDGES))]
    pl.concat([p.select(keep) for p in parts]).write_csv(OUT_DIR / "did18_all.csv")


if __name__ == "__main__":
    main()
