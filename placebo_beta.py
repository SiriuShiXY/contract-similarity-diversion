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
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import polars as pl

from build_cells import EVENT_SPEC, EVENTS, EV_DIR, OUT_DIR, SESSION_IDX, last_session
from build_full_panel import FULL_DIR
from lambda_est import RES_DIR, counterfactual, make_window
from mahalanobis import shrunk_cov
from panel import CFG, dte_bucket, listed_expirations
from phantom import NEW_WD, profile, theta_from_structural
from placebo import FIRST_LISTED, LO_BOUND, full_cells, full_grid, sessions, eligible, placebo_days

## placebo windows: FIRST_LISTED and LO_BOUND from placebo.py (the same pre-event dates as the lambda RI; 2018: SPY
## lists Wednesdays, the calendar the Monday joined)
SPECS = {"t5_1": ["N"], "t5_2": ["S"], "t5_3": ["S", "N"]}
X = ["x_logtau", "x_absd"]
CELL = ["underlying_symbol", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
N_POOL = 999
SEED = 20261102


## ------------------------------------------------------------------ regression pieces
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
    y, w = frame["y"].to_numpy(), frame["Vpre"].to_numpy()
    grp = frame["grp"].to_numpy()
    out = {}
    for nm, cols in SPECS.items():
        xp = within_xprods(y, frame.select(cols).to_numpy(), w, grp)
        b = solve(xp)
        out[nm] = dict(xp=xp, coef=dict(zip(cols, b)))
    return out


## ------------------------------------------------------------------ placebo samples
_G = {}


def _init(shared):
    _G.update(shared)


def horizon(ev: str) -> dict[int, int]:
    days = (pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
            .filter(pl.col("has_file") & ~pl.col("early_close") & (pl.col("period") == "post"))
            .select("trade_date", pl.col("trade_date").dt.weekday().alias("wd")))
    lst = listed_expirations(ev, days.select("trade_date"))
    act = lst.filter(pl.col("is_new")).join(days, on="trade_date")
    return dict(act.group_by("wd").agg(pl.col("dte").max()).iter_rows())


def placebo_sample(ev: str, i: int) -> pl.DataFrame:
    """Regression sample (y, S, N, Vpre, product x day) of event ev's design at placebo date index i."""
    cfg, new_wd, H = CFG[ev], NEW_WD[ev], _G["H"][ev]
    days = placebo_days(_G["sess"], i)
    lo, hi = days["trade_date"].min(), days["trade_date"].max()
    cut = (pl.lit(cfg["cut"]) & pl.col("eff_wd").is_in([1, 3]) & pl.col("dte_bucket").is_in(["D7-30", "D31+"]))
    g = _G["grid"].filter(pl.col("trade_date").is_between(lo, hi)).with_columns(cut.alias("cut_cell"))
    new = pl.DataFrame(schema={"underlying_symbol": pl.Utf8, "trade_date": pl.Date, "V": pl.Float64})
    win = make_window(g, new, days, cfg["treated"], cfg["control"], "V")
    c = counterfactual(win)
    post = days.filter(pl.col("period") == "post")
    ## existing treated cell-days (listed cells with characteristics, i.e. not Deep ITM; imputed as in panel.py)
    ex = (g.filter(pl.col("underlying_symbol").is_in(cfg["treated"]) & (pl.col("delta_bucket") != "DITM"))
          .join(_G["cells"].filter(pl.col("trade_date").is_between(lo, hi)).select(
              ["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type", "V_char"] + X),
              on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type"], how="left")
          .join(days.select("trade_date", "period", "wd"), on="trade_date")
          .with_columns(pl.col("V_char").fill_null(0)))
    pre = ex.filter(pl.col("period") == "pre")
    has_x = pl.col("V_char") > 0
    m_wd = pre.filter(has_x).group_by(CELL + ["wd"]).agg([pl.col(x).mean().alias(f"{x}_wd") for x in X])
    m_pre = pre.filter(has_x).group_by(CELL).agg([pl.col(x).mean().alias(f"{x}_pre") for x in X])
    m_all = ex.filter(has_x).group_by(CELL).agg([pl.col(x).mean().alias(f"{x}_all") for x in X])
    ex = (ex.filter(pl.col("period") == "post")
          .join(m_wd, on=CELL + ["wd"], how="left").join(m_pre, on=CELL, how="left").join(m_all, on=CELL, how="left")
          .with_columns([pl.when(pl.col("V_char") > 0).then(pl.col(x)).otherwise(pl.coalesce(f"{x}_wd", f"{x}_pre", f"{x}_all"))
                         .alias(x) for x in X]))
    ## Sigma_p from the placebo pre-event sessions (treated products, V-weighted, shrunk)
    inv = {}
    for (sym, typ), gg in pre.filter(has_x).group_by(["underlying_symbol", "option_type"]):
        inv[(sym, typ)] = shrunk_cov(gg.select(X).to_numpy(), gg["V"].to_numpy())["S_inv"]
    ## new-weekday calendar after the placebo date, and the nearest-expiry flag
    lst = _G["lst"].filter(pl.col("underlying_symbol").is_in(cfg["treated"]) & pl.col("trade_date").is_between(lo, hi))
    existing = {s: set(lst.filter(pl.col("underlying_symbol") == s)["last_session"].to_list()) for s in cfg["treated"]}
    rows = []
    for t, wd in post.select("trade_date", "wd").iter_rows():
        for k in range(H[wd] + 1):
            d = t + timedelta(k)
            if d.isoweekday() in new_wd and d in SESSION_IDX:
                rows += [(s, t, d) for s in cfg["treated"] if d not in existing[s]]
    cal = (pl.DataFrame(rows, schema=["underlying_symbol", "trade_date", "last_session"], orient="row")
           .with_columns(pl.col("last_session").dt.weekday().alias("eff_wd_k"),
                         dte_bucket((pl.col("last_session") - pl.col("trade_date")).dt.total_days()).alias("dte_bucket_k")))
    e_star = (lst.filter(pl.col("trade_date").is_in(post["trade_date"].implode()))
              .group_by("underlying_symbol", "trade_date")
              .agg(pl.col("last_session").min().alias("e_star"), pl.col("eff_wd").sort_by("last_session").first(),
                   pl.col("dte_bucket").sort_by("last_session").first()))
    n_star = cal.group_by("underlying_symbol", "trade_date").agg(pl.col("last_session").min().alias("n_star"))
    nflag = (e_star.join(n_star, on=["underlying_symbol", "trade_date"]).filter(pl.col("n_star") < pl.col("e_star"))
             .select("underlying_symbol", "trade_date", "eff_wd", "dte_bucket", pl.lit(1).alias("N")))
    p1, p2, p3 = _G["prof"][ev]
    nc = (cal.select("underlying_symbol", "trade_date", "eff_wd_k", "dte_bucket_k").unique()
          .join(pl.DataFrame({"delta_bucket_k": ["DOTM", "OTM", "ATM", "ITM"]}), how="cross")
          .join(pl.DataFrame({"option_type": ["C", "P"]}), how="cross")
          .with_columns(pl.col("trade_date").dt.weekday().alias("wd")))
    cols = [f"{x}_k" for x in X] + ["w_k"]
    nc = (nc.join(p1, on=["wd", "eff_wd_k", "dte_bucket_k", "delta_bucket_k", "option_type"], how="left")
          .join(p2.rename({c: f"{c}_2" for c in cols}), on=["eff_wd_k", "dte_bucket_k", "delta_bucket_k", "option_type"], how="left")
          .join(p3.rename({c: f"{c}_3" for c in cols}), on=["dte_bucket_k", "delta_bucket_k", "option_type"], how="left")
          .with_columns([pl.coalesce(c, f"{c}_2", f"{c}_3").alias(c) for c in cols])
          .filter(pl.col("w_k").is_not_null()))
    pairs = ex.select(CELL + ["trade_date"] + X).join(nc.select("underlying_symbol", "trade_date", "option_type", *cols),
                                                       on=["underlying_symbol", "trade_date", "option_type"])
    i11 = pl.Series([inv[(s, t)][0, 0] for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()])
    i12 = pl.Series([inv[(s, t)][0, 1] for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()])
    i22 = pl.Series([inv[(s, t)][1, 1] for s, t in pairs.select("underlying_symbol", "option_type").iter_rows()])
    a, b = pl.col("x_logtau") - pl.col("x_logtau_k"), pl.col("x_absd") - pl.col("x_absd_k")
    pairs = pairs.with_columns(i11=i11, i12=i12, i22=i22).with_columns(
        (pl.col("i11") * a ** 2 + 2 * pl.col("i12") * a * b + pl.col("i22") * b ** 2).alias("d2"))
    th = _G["theta"]
    S = pairs.group_by(CELL + ["trade_date"]).agg(
        ((pl.col("w_k") * (-th * pl.col("d2")).exp()).sum() / pl.col("w_k").sum()).alias("S"))
    ## y and Vbar_pre from the counterfactual engine
    ci = {r: j for j, r in enumerate(win.cells.iter_rows())}
    di = {d: k for k, d in enumerate(days["trade_date"].to_list())}
    smp = (ex.select(CELL + ["trade_date"]).join(S, on=CELL + ["trade_date"], how="left")
           .join(nflag, on=["underlying_symbol", "trade_date", "eff_wd", "dte_bucket"], how="left")
           .with_columns(pl.col("N").fill_null(0).cast(pl.Float64), pl.col("S").fill_null(0.0)))
    jr = np.array([ci.get(r, -1) for r in smp.select(CELL).iter_rows()])
    dc = np.array([di[d] for d in smp["trade_date"].to_list()])
    ok = jr >= 0
    vpre = np.where(ok, c["mean_all"][np.where(ok, jr, 0)], np.nan)
    cf = np.where(ok, c["cf"][np.where(ok, jr, 0), dc], np.nan)
    V = np.where(ok, win.T[np.where(ok, jr, 0), dc], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = (V - cf) / vpre
    smp = smp.with_columns(pl.Series("y", y), pl.Series("Vpre", vpre)).filter(
        pl.col("y").is_finite() & (pl.col("Vpre") > 0))
    return smp.with_columns(pl.concat_str(["underlying_symbol", pl.col("trade_date").cast(pl.Utf8)]).alias("pday"))


def one(args) -> dict:
    ev, i = args
    smp = placebo_sample(ev, i)
    smp = smp.with_columns(pl.col("pday").rank("dense").cast(pl.Int64).alias("grp") - 1)
    st = stats(smp)
    return dict(event=ev, t_p=_G["sess"]["trade_date"][i], n=smp.height, S_mean=float(smp["S"].mean()),
                N_sum=int(smp["N"].sum()), st=st)


## ------------------------------------------------------------------ actual statistic, same construction
def actual(ev: str) -> dict:
    d = pl.read_parquet(OUT_DIR / ev / "did_panel.parquet").filter(
        pl.col("treated") & (pl.col("period") == "post") & ~pl.col("cut_cell") & pl.col("y").is_not_null() & (pl.col("Vpre") > 0))
    d = d.with_columns(pl.col("S_hat").alias("S"), pl.col("N_cal").cast(pl.Float64).alias("N"),
                       pl.concat_str(["underlying_symbol", pl.col("trade_date").cast(pl.Utf8)]).rank("dense")
                       .cast(pl.Int64).alias("grp") - 1)
    return dict(event=ev, n=d.height, st=stats(d))


def main() -> None:
    cells = full_cells()
    grid = full_grid(cells)
    sess = sessions(cells)
    expi = (pl.concat([pl.read_parquet(FULL_DIR / f"expirations_{y}.parquet") for y in range(2016, 2024)])
            .group_by("underlying_symbol", "expiration", "eff_wd").agg(pl.col("trade_date").min().alias("first_trade"))
            .with_columns(pl.col("expiration").map_elements(last_session, return_dtype=pl.Date).alias("last_session")))
    lst = (sess.select("trade_date").join(expi, how="cross")
           .filter((pl.col("first_trade") <= pl.col("trade_date")) & (pl.col("last_session") >= pl.col("trade_date")))
           .with_columns(dte_bucket((pl.col("last_session") - pl.col("trade_date")).dt.total_days()).alias("dte_bucket")))
    idx = eligible(sess)
    dl = sess["trade_date"].to_list()
    jobs = []
    for ev in EVENTS:
        lo_bound = LO_BOUND[ev]
        ok = [i for i in idx if dl[i - 60] > lo_bound and dl[i + 59] < FIRST_LISTED[ev]]
        jobs += [(ev, i) for i in ok]
        print(f"[{ev}] eligible placebo dates {len(ok)}" + (f" ({dl[ok[0]]} to {dl[ok[-1]]})" if ok else ""), flush=True)
    shared = dict(grid=grid, cells=cells, sess=sess, lst=lst, theta=theta_from_structural(),
                  H={ev: horizon(ev) for ev in EVENTS},
                  prof={ev: profile(ev, pl.read_parquet(OUT_DIR / ev / "cells.parquet")) for ev in EVENTS})
    with Pool(12, initializer=_init, initargs=(shared,)) as pool:
        res = pool.map(one, jobs, chunksize=4)
    act = {ev: actual(ev) for ev in EVENTS}

    rows = []
    for r in res:
        for nm, v in r["st"].items():
            for term, b in v["coef"].items():
                rows.append(dict(event=r["event"], t_p=r["t_p"], spec=nm, term=term, coef=float(b), n=r["n"],
                                 S_mean=r["S_mean"], N_sum=r["N_sum"]))
    ## pooled: random triples of placebo dates
    rng = np.random.default_rng(SEED)
    by_ev = {ev: [r for r in res if r["event"] == ev] for ev in EVENTS}
    for draw in range(N_POOL):
        pick = [by_ev[ev][rng.integers(len(by_ev[ev]))] for ev in EVENTS]
        for nm, cols in SPECS.items():
            xp = dict(XtWX=sum(p["st"][nm]["xp"]["XtWX"] for p in pick), XtWy=sum(p["st"][nm]["xp"]["XtWy"] for p in pick))
            for term, b in zip(cols, solve(xp)):
                rows.append(dict(event="pooled", t_p=None, spec=nm, term=term, coef=float(b), n=sum(p["n"] for p in pick),
                                 S_mean=None, N_sum=None, draw=draw))
    pb = pl.DataFrame(rows, infer_schema_length=None)
    pb.write_parquet(RES_DIR / "placebo_beta.parquet")

    out = []
    for ev in list(EVENTS) + ["pooled"]:
        for nm, cols in SPECS.items():
            if ev == "pooled":
                xp = dict(XtWX=sum(act[e]["st"][nm]["xp"]["XtWX"] for e in EVENTS),
                          XtWy=sum(act[e]["st"][nm]["xp"]["XtWy"] for e in EVENTS))
                coef = dict(zip(cols, solve(xp)))
            else:
                coef = act[ev]["st"][nm]["coef"]
            for term in cols:
                p = pb.filter((pl.col("event") == ev) & (pl.col("spec") == nm) & (pl.col("term") == term))["coef"].to_numpy()
                out.append(dict(event=ev, spec=nm, term=term, actual=float(coef[term]), n_placebo=len(p),
                                ri_p=float(np.mean(np.abs(p) >= abs(coef[term]))) if len(p) else None,
                                pctile=float(100 * np.mean(p <= coef[term])) if len(p) else None,
                                placebo_mean=float(p.mean()) if len(p) else None,
                                placebo_sd=float(p.std(ddof=1)) if len(p) > 1 else None))
    out = pl.DataFrame(out)
    out.write_csv(RES_DIR / "beta_ri.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=3):
        print(out)


if __name__ == "__main__":
    main()
