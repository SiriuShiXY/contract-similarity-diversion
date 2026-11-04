"""
Structural diversion model with the Mahalanobis baseline (similarity_diversion_design.pdf eqs. 19-21,
Sections 8.2-8.4; table_shells Table 6 column 1 and Table 4 Panel B).

    -dV_hat_jt = sum_k lambda_e V_kt S_jkt / sum_j' S_j'kt,   S_jkt = exp(-theta d_jkt^2 + gamma N_jt)     (19)
    min sum_e sum_{t in post} sum_j Vbar_pre_j ((-dV_jt - (-dV_hat_jt)) / Vbar_pre_j)^2                    (20)
    lambda_e = Lambda(a + b n_e), n_e = 1 (2016), 2 (2018), 1 (2021 QQQ, IWM), 3 (2022), Lambda logistic      (21)

j: existing cells of the treated products; k: new cells of the same product, option type and day (calls and
puts separately, design 11.1); d: Mahalanobis distance on (log trading hours, |delta| at the common
volatility) with the pre-event, volume-weighted, shrunk Sigma_p of the product and type (mahalanobis.py);
N: nearest-expiry indicator (panel.py).  The choice set in the denominators is every listed existing cell of
the product and type that day (2022 Monday/Wednesday 7-30 and 31+ DTE cells included as sources, left out of
the fit, design 7.4) except the Deep ITM cells, which have no characteristics (design 3.6): neither existing
Deep ITM cells nor new Deep ITM volume enter the model.  dV_jt = V_jt - V_cf_jt (Counterfactual 1, panel.py); fit sample = existing cells of the
treated products on post-event days (day 0-59, early closes out) with a counterfactual and Vbar_pre > 0, the
regression sample of eq. 8.

Versions: (L) lambda_e = logistic(a + b n_e) (Table 6 col. 1, Table 4 Panel B); (F) a free lambda_e per event
(the unrestricted alternative of Section 8.2; given theta and gamma it is concentrated out by weighted least
squares); (I) lambda_e = a + b n_e (Lambda = identity, which allows lambda outside [0, 1]).
Fit: in-sample weighted RMSE of y = dV / Vbar_pre (and of the no-loss prediction for scale).  Out of sample
(Section 8.3; IWM 2024 has no OptionMetrics volatility): (L) and (I) estimated on 2016, 2018 and the 2021
QQQ and IWM events predict 2022 with lambda_2022 = Lambda(a + 3b): weighted RMSE and the rank (Spearman) correlation between predicted
and realized cell-level losses (cell = product x expiration weekday x DTE bucket x delta bucket x type;
loss = mean over post-event days of -y, predicted mean of the model's -dV_hat / Vbar_pre).

Block bootstrap (8.4): per event, five-session blocks resampled with replacement within the pre- and the
post-event window (lambda_est.boot_weights); each draw recomputes the weekday baselines, Counterfactual 1, y,
Vbar_pre, Sigma_p (with its shrinkage) and the distances, and re-estimates every version.
Also: Table 4 Panel B, lambda_e = logistic(a + b n_e) fitted by NLS to the three event-level lambdas of
Table 4 Panel A (lambda_est.py), with the same fit on each of its bootstrap draws; and lambda by new weekday
(2022: Tuesday vs Thursday) with each existing cell-day's Delta V attributed to the new weekdays in
proportion to the model's predicted diversion from them.

Usage: python structural.py [n_boot]
Output: results/structural_estimates.csv, results/structural_boot.parquet, results/lambda_density.csv,
        results/lambda_newweekday.csv, data/<event>/model_fit.parquet
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from multiprocessing import Pool

import numpy as np
import polars as pl
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import spearmanr

from build_cells import EVENT_SPEC, EVENTS, OUT_DIR
from lambda_est import RES_DIR, boot_weights, counterfactual, event_inputs, make_window
from mahalanobis import X, shrunk_cov
from panel import CFG

N_E = {ev: s["n_e"] for ev, s in EVENT_SPEC.items()}
KEY = {ev: s["short"].replace(" ", "") for ev, s in EVENT_SPEC.items()}     # 2016, 2018, 2021QQQ, 2021IWM, 2022
J = ["underlying_symbol", "trade_date", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
CELLKEY = ["underlying_symbol", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
N_BOOT = 499
SEED = 20261101
THETA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
GAMMA_GRID = [-1.0, 0.0, 1.0, 2.0]


## ------------------------------------------------------------------ data
def event_data(ev: str) -> dict:
    """Arrays for one event: menu cell-days, pairs, fit rows, and the engine window for the bootstrap."""
    cfg = CFG[ev]
    days, grid, new = event_inputs(ev)
    win = make_window(grid, new, days, cfg["treated"], cfg["control"], "V")
    panel = pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet").filter(pl.col("treated") & (pl.col("period") == "post"))
    pairs = pl.read_parquet(OUT_DIR / ev / "pairs.parquet").filter(
        pl.col("trade_date").is_in(panel["trade_date"].unique().implode()))
    menu = pairs.select(J).unique().sort(J).with_row_index("m")
    menu = menu.join(panel.select(J + ["y", "Vpre", "N", "cut_cell", "V", "V_cf"]), on=J, how="left")
    assert menu["N"].null_count() == 0
    menu = menu.with_columns((~pl.col("cut_cell") & pl.col("y").is_not_null() & (pl.col("Vpre") > 0)).alias("fit"))
    pr = pairs.join(menu.select(J + ["m"]), on=J).with_columns(
        pl.concat_str([pl.col("underlying_symbol"), pl.col("option_type"), pl.col("trade_date").cast(pl.Utf8),
                       pl.col("eff_wd_k").cast(pl.Utf8), pl.col("dte_bucket_k"), pl.col("delta_bucket_k")],
                      separator="|").alias("kkey"))
    kk = pr.select("kkey").unique().sort("kkey").with_row_index("g")
    pr = pr.join(kk, on="kkey")
    ## engine positions of the menu cell-days (treated, non-cut cells only exist in the engine)
    ci = {r: i for i, r in enumerate(win.cells.iter_rows())}
    di = {d: i for i, d in enumerate(days["trade_date"].to_list())}
    jrow = np.array([ci.get(r, -1) for r in menu.select(CELLKEY).iter_rows()])
    dcol = np.array([di[d] for d in menu["trade_date"].to_list()])
    ## pre-event cells for Sigma_p
    cells = pl.read_parquet(OUT_DIR / ev / "cells.parquet")
    pre = cells.filter(~pl.col("is_new") & (pl.col("period") == "pre") & ~pl.col("early_close") & (pl.col("V_char") > 0)
                       & pl.col("underlying_symbol").is_in(cfg["treated"]))
    sig_groups = []
    for (sym, typ), g in pre.group_by(["underlying_symbol", "option_type"]):
        sig_groups.append(dict(sym=sym, typ=typ, X=g.select(X).to_numpy(), V=g["V"].to_numpy(),
                               dcol=np.array([di[d] for d in g["trade_date"].to_list()]),
                               pidx=np.flatnonzero(((pr["underlying_symbol"] == sym) & (pr["option_type"] == typ)).to_numpy())))
    return dict(ev=ev, days=days, win=win, menu=menu, n_menu=menu.height, n_groups=kk.height,
                m_idx=pr["m"].to_numpy(), g_idx=pr["g"].to_numpy(), Vk=pr["V_k"].to_numpy(),
                da=(pr["x_logtau"] - pr["x_logtau_k"]).to_numpy(), db=(pr["x_absd"] - pr["x_absd_k"]).to_numpy(),
                d2=pr["d"].to_numpy() ** 2, N=menu["N"].to_numpy().astype(float), fit=menu["fit"].to_numpy(),
                y=menu["y"].fill_null(0).to_numpy(), Vpre=menu["Vpre"].fill_null(0).to_numpy(),
                w=np.ones(menu.height), jrow=jrow, dcol=dcol, sig_groups=sig_groups, pairs=pr)


def resample(ed: dict, rng: np.random.Generator) -> dict:
    """Bootstrap copy: day multiplicities, recomputed counterfactual, y, Vbar_pre, Sigma_p and distances."""
    win = ed["win"]
    w = boot_weights(win, rng)
    c = counterfactual(win, w)
    ok = ed["jrow"] >= 0
    y = np.zeros(ed["n_menu"])
    vpre = np.zeros(ed["n_menu"])
    jr, dc = ed["jrow"][ok], ed["dcol"][ok]
    vpre[ok] = c["mean_all"][jr]
    with np.errstate(invalid="ignore", divide="ignore"):
        y[ok] = (win.T[jr, dc] - c["cf"][jr, dc]) / vpre[ok]
    fit = ed["fit"] & ok & np.isfinite(y) & (vpre > 0)
    d2 = ed["d2"].copy()
    for g in ed["sig_groups"]:
        s = shrunk_cov(g["X"], g["V"] * w[g["dcol"]])
        i = s["S_inv"]
        a, b = ed["da"][g["pidx"]], ed["db"][g["pidx"]]
        d2[g["pidx"]] = i[0, 0] * a * a + 2 * i[0, 1] * a * b + i[1, 1] * b * b
    out = dict(ed)
    out.update(y=np.where(fit, y, 0.0), Vpre=np.where(fit, vpre, 0.0), fit=fit, d2=d2, w=w[ed["dcol"]])
    return out


## ------------------------------------------------------------------ model
def G_of(ed: dict, theta: float, gamma: float) -> np.ndarray:
    """sum_k V_k S_jk / sum_j' S_j'k for every menu cell-day (log-sum-exp within each new cell)."""
    z = -theta * ed["d2"] + gamma * ed["N"][ed["m_idx"]]
    zmax = np.full(ed["n_groups"], -np.inf)
    np.maximum.at(zmax, ed["g_idx"], z)
    s = np.exp(z - zmax[ed["g_idx"]])
    den = np.bincount(ed["g_idx"], s, ed["n_groups"])
    return np.bincount(ed["m_idx"], ed["Vk"] * s / den[ed["g_idx"]], ed["n_menu"])


def pieces(eds: list[dict], theta: float, gamma: float):
    """Per event: -y, g = G / Vbar_pre and observation weights on the fit rows."""
    out = []
    for ed in eds:
        f = ed["fit"] & (ed["w"] > 0)
        G = G_of(ed, theta, gamma)
        out.append((-ed["y"][f], G[f] / ed["Vpre"][f], ed["Vpre"][f] * ed["w"][f], f))
    return out


def lam_of(version: str, p, eds: list[dict]) -> np.ndarray:
    n = np.array([N_E[ed["ev"]] for ed in eds], dtype=float)
    if version == "L":
        return expit(p[0] + p[1] * n)
    return p[0] + p[1] * n


def resid(params, version: str, eds: list[dict]) -> np.ndarray:
    theta, gamma = np.exp(params[0]), params[1]
    pc = pieces(eds, theta, gamma)
    if version == "F":
        lam = [float((w * ly * g).sum() / (w * g * g).sum()) for ly, g, w, _ in pc]
    else:
        lam = lam_of(version, params[2:], eds)
    return np.concatenate([np.sqrt(w) * (ly - l * g) for (ly, g, w, _), l in zip(pc, lam)])


def free_lambdas(eds, theta, gamma) -> list[float]:
    return [float((w * ly * g).sum() / (w * g * g).sum()) for ly, g, w, _ in pieces(eds, theta, gamma)]


def fit(version: str, eds: list[dict], start=None) -> dict:
    """Weighted NLS; theta = exp(phi) > 0; multi-start over a (theta, gamma) grid for the concentrated free model."""
    if start is None:
        best = None
        for th in THETA_GRID:
            for ga in GAMMA_GRID:
                r = resid([np.log(th), ga], "F", eds)
                v = float(r @ r)
                if best is None or v < best[0]:
                    best = (v, th, ga)
        start = [np.log(best[1]), best[2]]
    if version == "F":
        x0 = start[:2]
    else:
        lam = np.clip(free_lambdas(eds, np.exp(start[0]), start[1]), 0.02, 0.98)
        n = np.array([N_E[ed["ev"]] for ed in eds], dtype=float)
        tgt = logit(lam) if version == "L" else np.array(free_lambdas(eds, np.exp(start[0]), start[1]))
        if len(eds) > 1:
            b, a = np.polyfit(n, tgt, 1)
        else:
            a, b = tgt[0], 0.0
        x0 = [start[0], start[1], a, b]
    r = least_squares(resid, x0, args=(version, eds), method="trf", x_scale="jac", max_nfev=2000)
    theta, gamma = float(np.exp(r.x[0])), float(r.x[1])
    lam = free_lambdas(eds, theta, gamma) if version == "F" else list(lam_of(version, r.x[2:], eds))
    ssr = float(r.fun @ r.fun)
    wsum = sum(float((ed["Vpre"] * ed["w"])[ed["fit"] & (ed["w"] > 0)].sum()) for ed in eds)
    ss0 = sum(float(((ed["Vpre"] * ed["w"] * ed["y"] ** 2)[ed["fit"] & (ed["w"] > 0)]).sum()) for ed in eds)
    out = dict(version=version, theta=theta, gamma=gamma, ssr=ssr, rmse=np.sqrt(ssr / wsum), rmse0=np.sqrt(ss0 / wsum),
               success=bool(r.success), nfev=int(r.nfev), **{f"lam_{KEY[ed['ev']]}": l for ed, l in zip(eds, lam)})
    if version != "F":
        out.update(a=float(r.x[2]), b=float(r.x[3]))
    return out


def predict(ed: dict, theta: float, gamma: float, lam: float) -> np.ndarray:
    """Predicted -dV / Vbar_pre for every menu cell-day."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return lam * G_of(ed, theta, gamma) / ed["Vpre"]


def oos(version: str, train: list[dict], test: dict, start=None) -> dict:
    f = fit(version, train, start)
    lam = float(lam_of(version, [f["a"], f["b"]], [test])[0])
    m = predict(test, f["theta"], f["gamma"], lam)
    k = test["fit"] & (test["w"] > 0)
    w = test["Vpre"][k] * test["w"][k]
    r = -test["y"][k] - m[k]
    ## cell-level realized and predicted losses (means over post-event days)
    cells = test["menu"].filter(pl.Series(k)).select(J[:1] + J[2:]).with_columns(
        pl.Series("loss", -test["y"][k]), pl.Series("pred", m[k]), pl.Series("w", test["w"][k]))
    cl = cells.group_by(J[:1] + J[2:]).agg(((pl.col("loss") * pl.col("w")).sum() / pl.col("w").sum()).alias("loss"),
                                           ((pl.col("pred") * pl.col("w")).sum() / pl.col("w").sum()).alias("pred"))
    rho = float(spearmanr(cl["loss"], cl["pred"]).statistic)
    return dict(version=f"{version}_oos", theta=f["theta"], gamma=f["gamma"], a=f["a"], b=f["b"], lam_2022_pred=lam,
                **{k: v for k, v in f.items() if k.startswith("lam_")},
                rmse_oos=float(np.sqrt((w * r * r).sum() / w.sum())),
                rmse0_oos=float(np.sqrt((w * test["y"][k] ** 2).sum() / w.sum())), rank_corr_oos=rho,
                n_cells_oos=cl.height)


def all_fits(eds: list[dict], start=None) -> list[dict]:
    rows = []
    fF = fit("F", eds, start)
    st = [np.log(fF["theta"]), fF["gamma"]]
    rows.append(fF)
    for v in ("L", "I"):
        rows.append(fit(v, eds, st))
    for v in ("L", "I"):
        rows.append(oos(v, eds[:-1], eds[-1], st))          # design 8.3: earlier events predict 2022
    return rows


## ------------------------------------------------------------------ Table 4 Panel B
def density_fit(lams: np.ndarray, n=np.array([float(N_E[ev]) for ev in EVENTS])) -> tuple[float, float, float, float]:
    """lambda_e = logistic(a + b n_e) by NLS on the event-level lambdas, and the identity version (OLS)."""
    def r(p):
        return expit(p[0] + p[1] * n) - lams
    lc = np.clip(lams, 0.02, 0.98)
    b0, a0 = np.polyfit(n, logit(lc), 1)
    s = least_squares(r, [a0, b0], method="lm", max_nfev=5000)
    bi, ai = np.polyfit(n, lams, 1)
    return float(s.x[0]), float(s.x[1]), float(ai), float(bi)


## ------------------------------------------------------------------ bootstrap worker
_E = {}


def _init(eds):
    _E["eds"] = eds


def boot_one(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    eds = [resample(ed, rng) for ed in _E["eds"]]
    start = _E.get("start")
    try:
        rows = all_fits(eds, start)
    except Exception as e:                                  # pragma: no cover - reported, not hidden
        return [dict(draw=seed, error=str(e))]
    return [dict(draw=seed, **r) for r in rows]


def main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else N_BOOT
    eds = [event_data(ev) for ev in EVENTS]
    for ed in eds:
        print(f"[{ed['ev']}] menu cell-days {ed['n_menu']:,}, fit {int(ed['fit'].sum()):,}, pairs {len(ed['d2']):,}, "
              f"new cells {ed['n_groups']:,}", flush=True)
    rows = all_fits(eds)
    est = pl.DataFrame(rows, infer_schema_length=None)
    with pl.Config(tbl_width_chars=250, float_precision=4, tbl_cols=30):
        print(est)

    ## model fit per cell-day (point estimates) and lambda by new weekday
    fL = [r for r in rows if r["version"] == "L"][0]
    for ed in eds:
        lamL = float(fL[f"lam_{KEY[ed['ev']]}"])
        m = predict(ed, fL["theta"], fL["gamma"], lamL)
        ed["menu"].with_columns(pl.Series("pred_loss_L", m), pl.lit(lamL).alias("lam_L")).write_parquet(
            OUT_DIR / ed["ev"] / "model_fit.parquet")
    nw = []
    for ed in eds:
        pr = ed["pairs"]
        z = -fL["theta"] * ed["d2"] + fL["gamma"] * ed["N"][ed["m_idx"]]
        zmax = np.full(ed["n_groups"], -np.inf)
        np.maximum.at(zmax, ed["g_idx"], z)
        s = np.exp(z - zmax[ed["g_idx"]])
        share = s / np.bincount(ed["g_idx"], s, ed["n_groups"])[ed["g_idx"]]
        contrib = ed["Vk"] * share
        tot = np.bincount(ed["m_idx"], contrib, ed["n_menu"])
        menu = ed["menu"].with_columns(pl.Series("dV", (ed["menu"]["V"] - ed["menu"]["V_cf"]).fill_null(0)))
        days = ed["days"].filter((pl.col("tau") >= 1) & (pl.col("tau") <= 59))["trade_date"]
        inwk = menu["trade_date"].is_in(days.implode()).to_numpy() & ed["fit"]
        for wdk in sorted(pr["eff_wd_k"].unique().to_list()):
            mk = (pr["eff_wd_k"] == wdk).to_numpy()
            part = np.bincount(ed["m_idx"], np.where(mk, contrib, 0.0), ed["n_menu"])
            with np.errstate(invalid="ignore", divide="ignore"):
                pi = np.where(tot > 0, part / tot, 0.0)
            newv = pl.read_parquet(OUT_DIR / ed["ev"] / "cells.parquet").filter(
                pl.col("is_new") & (pl.col("eff_wd") == wdk) & pl.col("trade_date").is_in(days.implode()))["V"].sum()
            dv = float((menu["dV"].to_numpy() * pi)[inwk].sum())
            nw.append(dict(event=ed["ev"], new_weekday=int(wdk), new_V=float(newv), dV_attributed=dv,
                           lam=-dv / float(newv) if newv > 0 else np.nan))
    pl.DataFrame(nw).write_csv(RES_DIR / "lambda_newweekday.csv")
    print(pl.DataFrame(nw))

    ## Table 4 Panel B on the event-level lambdas of lambda_est.py (delta notional)
    le = pl.read_csv(RES_DIR / "lambda_event.csv", schema_overrides={"event": pl.Utf8}).filter(
        (pl.col("split") == "all") & (pl.col("unit") == "V")).sort("event")
    lams = le["lam"].to_numpy()
    a, b, ai, bi = density_fit(lams)
    lb = pl.read_parquet(RES_DIR / "lambda_boot.parquet").filter((pl.col("split") == "all") & (pl.col("unit") == "V"))
    piv = lb.pivot(on="event", index="draw", values="lam").sort("draw")
    bd = np.array([density_fit(np.array(r[1:])) for r in piv.select(["draw"] + list(EVENTS)).iter_rows()])
    dens = dict(a=a, b=b, se_a=float(np.std(bd[:, 0], ddof=1)), se_b=float(np.std(bd[:, 1], ddof=1)),
                a_lin=ai, b_lin=bi, se_a_lin=float(np.std(bd[:, 2], ddof=1)), se_b_lin=float(np.std(bd[:, 3], ddof=1)),
                b_med=float(np.median(bd[:, 1])), share_b_pos=float(np.mean(bd[:, 1] > 0)),
                share_blin_pos=float(np.mean(bd[:, 3] > 0)), n_boot=len(bd), **{f"lam_{KEY[ev]}": float(l) for ev, l in zip(EVENTS, lams)})
    dens["p_b0"] = float(2 * min(np.mean(bd[:, 1] <= 0), np.mean(bd[:, 1] >= 0)))
    dens["p_b0_lin"] = float(2 * min(np.mean(bd[:, 3] <= 0), np.mean(bd[:, 3] >= 0)))
    pl.DataFrame([dens]).write_csv(RES_DIR / "lambda_density.csv")
    print("Table 4 Panel B:", {k: round(v, 3) if isinstance(v, float) else v for k, v in dens.items()})

    ## bootstrap of the structural model
    _E["start"] = None
    with Pool(12, initializer=_init, initargs=(eds,)) as pool:
        draws = [r for rr in pool.imap_unordered(boot_one, range(SEED, SEED + n_boot)) for r in rr]
    bt = pl.DataFrame(draws, infer_schema_length=None)
    bt.write_parquet(RES_DIR / "structural_boot.parquet")
    ses = []
    for r in rows:
        v = r["version"]
        b_ = bt.filter(pl.col("version") == v)
        se = {f"se_{c}": float(b_[c].std()) for c in b_.columns if c in r and isinstance(r[c], float) and
              (c in ("theta", "gamma", "a", "b", "rmse_oos", "rank_corr_oos") or c.startswith("lam_"))}
        se.update({f"p05_{c}": float(b_[c].quantile(0.05)) for c in ("theta", "gamma") if c in b_.columns},
                  **{f"p95_{c}": float(b_[c].quantile(0.95)) for c in ("theta", "gamma") if c in b_.columns},
                  n_boot_ok=b_.height)
        ses.append({**r, **se})
    out = pl.DataFrame(ses, infer_schema_length=None)
    out.write_csv(RES_DIR / "structural_estimates.csv")
    with pl.Config(tbl_width_chars=250, float_precision=4, tbl_cols=40):
        print(out)


if __name__ == "__main__":
    main()
