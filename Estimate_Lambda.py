"""Estimate event-level diversion and the definition bridge."""

from __future__ import annotations

import sys


# Lambda Est
"""
Counterfactual 1, the volume table and the reallocated share lambda (similarity_diversion_design.pdf
Sections 7.1-7.4 and 8.4; table_shells Tables 2 and 4, Panel A).

Existing cell j of a treated product on day t (cells and grid as in panel.py: listed existing cells, zero
volume where a listed cell does not trade; Deep ITM cells included, their premium excluded (shell Table 2);
2022 Monday/Wednesday cells at 7-30 and 31+ DTE excluded, design Section 7.4):
    base_jt = Vbar_pre_{j,wd(t)}   weekday-matched pre-event mean (the cell's all-day pre mean when it has no
                                   pre-event day of that weekday: holiday-shifted expirations, panel.py)
    cf_jt   = base_jt x A_{b(j),t}, A = control families' volume in the matching DTE bucket relative to its
              weekday-matched pre-event mean (their total volume when the bucket is not listed for them)
Sums over post-event weeks 1-12 (event time 1-59; day 0, the 2022 transition days and early closes out):
    E = sum V_jt,  B = sum base_jt,  CF = sum cf_jt,  NEW = sum of the new cells' volume V_kt
    lambda = -(E - CF) / NEW          (design eq. 13; 1 - lambda = (E + NEW - CF) / NEW)
    lambda_raw = -(E - B) / NEW       (weekday baseline only: the design's 0.65 / 0.27 / 0.91 in contracts)
Table 2, Panel B, in % of B: existing, weekday baseline only (E - B); existing, control-adjusted (E - CF); new
contracts NEW; total, control-adjusted (E + NEW - CF); pre-event weekly s.d. of existing volume: s.d. over the
twelve pre-event weeks of sum_week V / sum_week base - 1 (in sample, a benchmark for normal fluctuations).
Panel A: share of the treated products' volume (all existing cells, including the 2022 cut cells) in the new
cells, weeks 1-12, in contracts, delta notional, trades and premium.

Block bootstrap (design 8.4; Table 4 notes): blocks of five consecutive sessions in event time (pre-event
weeks -60..-56, ..., -5..-1; post weeks 1-5, ..., 56-59; day 0 its own block), resampled with replacement
within the pre-event and within the post-event window; every draw recomputes the weekday-matched baselines,
the control ratios, Counterfactual 1, Delta V and lambda.

Usage: python lambda_est.py [n_boot]
Output: results/lambda_event.csv (Table 4 Panel A and splits), results/table2_volume.csv,
        results/lambda_boot.parquet (draws), data/<event>/window_<unit>.npz is not written (all in memory)
"""
import sys
from dataclasses import dataclass, field
import numpy as np
import polars as pl
from Build_Panel import BUILD_EVENTS, EVENTS, EV_DIR, OUT_DIR, RES_DIR
from Build_Event_Panel import CFG

UNITS = {"V": "delta notional", "contracts": "contracts", "premium": "premium"}
BUCKETS = ["D0", "D1", "D2-6", "D7-30", "D31+"]
CELL = ["underlying_symbol", "eff_wd", "dte_bucket", "delta_bucket", "option_type"]
N_BOOT = 999
SEED = 20261031
BLOCKS_LAMBDA = {"w1-4": (1, 20), "w5-8": (21, 40), "w9-12": (41, 59)}


@dataclass
class Window:
    """One event (or placebo) window for one volume unit: existing treated cells x days, control buckets."""

    days: pl.DataFrame
    cells: pl.DataFrame
    T: np.ndarray
    bucket: np.ndarray
    product: np.ndarray
    C: np.ndarray
    Ct: np.ndarray
    NEW: dict
    wd: np.ndarray = field(init=False)
    pre: np.ndarray = field(init=False)
    tau: np.ndarray = field(init=False)
    block: np.ndarray = field(init=False)

    def __post_init__(self):
        self.wd = self.days["wd"].to_numpy()
        self.pre = (self.days["period"] == "pre").to_numpy()
        self.tau = self.days["tau"].to_numpy()
        self.block = self.days["block"].to_numpy()


def event_days(ev: str) -> pl.DataFrame:
    d = (
        pl.read_csv(EV_DIR / ev / "days.csv", try_parse_dates=True)
        .filter(
            pl.col("has_file")
            & ~pl.col("early_close")
            & pl.col("period").is_in(["pre", "post"])
        )
        .select(
            "trade_date", "tau", "period", pl.col("trade_date").dt.weekday().alias("wd")
        )
    )
    return with_blocks(d)


def with_blocks(d: pl.DataFrame) -> pl.DataFrame:
    """Five-session blocks in event time: pre-event by rank from the window start, post-event day 0 = block 0,
    weeks 1-12 = blocks 1-12 (tau 1-5, ..., 56-59)."""
    t0 = d.filter(pl.col("period") == "pre")["tau"].min()
    return d.with_columns(
        pl.when(pl.col("period") == "pre")
        .then(-1 - (pl.col("tau") - t0) // 5)
        .when(pl.col("tau") == 0)
        .then(0)
        .otherwise((pl.col("tau") - 1) // 5 + 1)
        .alias("block")
    ).sort("trade_date")


def make_window(
    grid: pl.DataFrame,
    new: pl.DataFrame,
    days: pl.DataFrame,
    treated: list[str],
    control: list[str],
    unit: str,
) -> Window:
    """grid: listed existing cell-days (CELL, trade_date, cut_cell, unit column); new: treated new cells
    (underlying_symbol, trade_date, unit column)."""
    di = {d: i for i, d in enumerate(days["trade_date"].to_list())}
    g = grid.filter(pl.col("trade_date").is_in(days["trade_date"].implode()))
    tr = g.filter(pl.col("underlying_symbol").is_in(treated) & ~pl.col("cut_cell"))
    cells = tr.select(CELL).unique().sort(CELL)
    ci = {r: i for i, r in enumerate(cells.iter_rows())}
    T = np.full((cells.height, days.height), np.nan)
    rows = [ci[r] for r in tr.select(CELL).iter_rows()]
    cols = [di[d] for d in tr["trade_date"].to_list()]
    T[rows, cols] = tr[unit].to_numpy()
    bucket = np.array([BUCKETS.index(b) for b in cells["dte_bucket"]])
    ct = (
        g.filter(pl.col("underlying_symbol").is_in(control) & ~pl.col("cut_cell"))
        .group_by("trade_date", "dte_bucket")
        .agg(pl.col(unit).sum())
    )
    C = np.full((len(BUCKETS), days.height), np.nan)
    C[
        [BUCKETS.index(b) for b in ct["dte_bucket"]], [di[d] for d in ct["trade_date"]]
    ] = ct[unit].to_numpy()
    Ct = np.nansum(C, axis=0)
    NEW = {}
    for p in treated:
        v = np.zeros(days.height)
        n = (
            new.filter(
                (pl.col("underlying_symbol") == p)
                & pl.col("trade_date").is_in(days["trade_date"].implode())
            )
            .group_by("trade_date")
            .agg(pl.col(unit).sum())
        )
        v[[di[d] for d in n["trade_date"]]] = n[unit].to_numpy()
        NEW[p] = v
    return Window(
        days=days,
        cells=cells,
        T=T,
        bucket=bucket,
        product=cells["underlying_symbol"].to_numpy(),
        C=C,
        Ct=Ct,
        NEW=NEW,
    )


def counterfactual(win: Window, w: np.ndarray | None = None) -> dict:
    """Weekday-matched baselines and Counterfactual 1 under day weights w (bootstrap multiplicities)."""
    D = win.T.shape[1]
    w = np.ones(D) if w is None else w
    L = ~np.isnan(win.T)
    T0 = np.where(L, win.T, 0.0)
    pw = w * win.pre
    mean_wd = np.full((win.T.shape[0], 6), np.nan)
    cm = np.full((len(BUCKETS), 6), np.nan)
    ctm = np.full(6, np.nan)
    LC = ~np.isnan(win.C)
    C0 = np.where(LC, win.C, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, 6):
            m = pw * (win.wd == k)
            mean_wd[:, k] = T0 * L @ m / (L @ m)
            cm[:, k] = C0 * LC @ m / (LC @ m)
            ctm[k] = win.Ct @ m / m.sum()
        mean_all = T0 * L @ pw / (L @ pw)
        base = mean_wd[:, win.wd]
        base = np.where(np.isnan(base), mean_all[:, None], base)
        base = np.where(L, base, np.nan)
        ab = win.C / cm[:, win.wd]
        at = win.Ct / ctm[win.wd]
        A = np.where(np.isfinite(ab), ab, at[None, :])
        cf = base * A[win.bucket, :]
    return dict(L=L, T0=T0, base=base, cf=cf, mean_all=mean_all, A=A)


def sums(
    win: Window,
    cfr: dict,
    w: np.ndarray | None = None,
    day_mask: np.ndarray | None = None,
    products: list[str] | None = None,
) -> dict:
    """E, B, CF, NEW over the post-event days in day_mask (default weeks 1-12), for the given products."""
    D = win.T.shape[1]
    w = np.ones(D) if w is None else w
    q = (win.tau >= 1) & (win.tau <= 59) & ~win.pre if day_mask is None else day_mask
    qw = w * q
    prods = list(win.NEW) if products is None else products
    jm = np.isin(win.product, prods)
    ok = cfr["L"] & ~np.isnan(cfr["base"]) & jm[:, None]
    nobase = cfr["L"] & np.isnan(cfr["base"]) & jm[:, None]
    E = float(np.where(ok, cfr["T0"], 0.0).sum(axis=0) @ qw)
    B = float(np.where(ok, cfr["base"], 0.0).sum(axis=0) @ qw)
    CF = float(np.where(ok, cfr["cf"], 0.0).sum(axis=0) @ qw)
    NEW = float(sum((win.NEW[p] for p in prods)) @ qw)
    E_nobase = float(np.where(nobase, cfr["T0"], 0.0).sum(axis=0) @ qw)
    return dict(E=E, B=B, CF=CF, NEW=NEW, E_nobase=E_nobase)


def derived(s: dict) -> dict:
    B, E, CF, NEW = (s["B"], s["E"], s["CF"], s["NEW"])
    return dict(
        existing_raw_pct=100 * (E - B) / B,
        existing_adj_pct=100 * (E - CF) / B,
        new_pct=100 * NEW / B,
        total_adj_pct=100 * (E + NEW - CF) / B,
        lam=-(E - CF) / NEW if NEW > 0 else np.nan,
        lam_raw=-(E - B) / NEW if NEW > 0 else np.nan,
        created=(E + NEW - CF) / NEW if NEW > 0 else np.nan,
    )


def pre_weekly_sd(win: Window, cfr: dict) -> float:
    """s.d. over pre-event weeks of the week's existing volume relative to its weekday-matched baseline (%)."""
    ok = cfr["L"] & ~np.isnan(cfr["base"])
    e = np.where(ok, cfr["T0"], 0.0).sum(axis=0)
    b = np.where(ok, cfr["base"], 0.0).sum(axis=0)
    dev = []
    for blk in np.unique(win.block[win.pre]):
        m = win.pre & (win.block == blk)
        dev.append(100 * (e[m].sum() / b[m].sum() - 1))
    return float(np.std(dev, ddof=1))


def boot_weights(win: Window, rng: np.random.Generator, post_blocks=None) -> np.ndarray:
    """Day multiplicities: pre-event blocks and post-event blocks resampled with replacement (separately)."""
    w = np.zeros(win.T.shape[1])
    pre_b = np.unique(win.block[win.pre])
    post_b = (
        np.unique(win.block[~win.pre])
        if post_blocks is None
        else np.asarray(post_blocks)
    )
    for blocks in (pre_b, post_b):
        for b in rng.choice(blocks, size=len(blocks), replace=True):
            w += win.block == b
    return w


def event_inputs(ev: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    days = event_days(ev)
    grid = pl.read_parquet(OUT_DIR / ev / "existing_panel.parquet")
    new = (
        pl.read_parquet(OUT_DIR / ev / "cells.parquet")
        .filter(pl.col("is_new") & ~pl.col("early_close"))
        .select(
            "underlying_symbol", "trade_date", "V", "contracts", "premium", "trades"
        )
    )
    return (days, grid, new)


def panel_a(ev: str, days: pl.DataFrame, grid: pl.DataFrame, new: pl.DataFrame) -> dict:
    """New cells' share of the treated products' volume, weeks 1-12 (all existing cells, cut cells included)."""
    q = days.filter(
        (pl.col("tau") >= 1) & (pl.col("tau") <= 59) & (pl.col("period") == "post")
    )["trade_date"].implode()
    tr = CFG[ev]["treated"]
    ex = grid.filter(
        pl.col("underlying_symbol").is_in(tr) & pl.col("trade_date").is_in(q)
    )
    nw = new.filter(
        pl.col("underlying_symbol").is_in(tr) & pl.col("trade_date").is_in(q)
    )
    return {
        f"share_{u}": 100 * nw[u].sum() / (nw[u].sum() + ex[u].sum())
        for u in ("contracts", "V", "trades", "premium")
    }


def lambda_est_main() -> None:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else N_BOOT
    rng = np.random.default_rng(SEED)
    rows, t2, draws = ([], [], [])
    for ev in BUILD_EVENTS:
        cfg = CFG[ev]
        days, grid, new = event_inputs(ev)
        pa = panel_a(ev, days, grid, new)
        for unit in UNITS:
            win = make_window(grid, new, days, cfg["treated"], cfg["control"], unit)
            cfr = counterfactual(win)
            s = sums(win, cfr)
            d = derived(s)
            sd = pre_weekly_sd(win, cfr)
            splits = {"all": dict(products=None, day_mask=None)}
            if len(cfg["treated"]) > 1:
                splits.update(
                    {p: dict(products=[p], day_mask=None) for p in cfg["treated"]}
                )
            post_blocks = {}
            for nm, (a, b) in BLOCKS_LAMBDA.items():
                splits[nm] = dict(
                    products=None, day_mask=(win.tau >= a) & (win.tau <= b) & ~win.pre
                )
                post_blocks[nm] = np.unique(
                    win.block[(win.tau >= a) & (win.tau <= b) & ~win.pre]
                )
            est = {nm: derived(sums(win, cfr, **kw)) for nm, kw in splits.items()}
            bd = {nm: [] for nm in splits}
            for i in range(n_boot):
                w = boot_weights(win, rng)
                c = counterfactual(win, w)
                for nm, kw in splits.items():
                    if nm in post_blocks:
                        wb = boot_weights(win, rng, post_blocks[nm])
                        bd[nm].append(
                            derived(sums(win, counterfactual(win, wb), w=wb, **kw))
                        )
                    else:
                        bd[nm].append(derived(sums(win, c, w=w, **kw)))
            for nm in splits:
                b = pl.DataFrame(bd[nm])
                if ev not in EVENTS:
                    continue
                draws.append(
                    b.with_columns(
                        pl.lit(ev).alias("event"),
                        pl.lit(unit).alias("unit"),
                        pl.lit(nm).alias("split"),
                        pl.int_range(pl.len()).alias("draw"),
                    )
                )
                r = dict(
                    event=ev,
                    unit=unit,
                    split=nm,
                    **est[nm],
                    **{
                        f"se_{k}": float(b[k].std())
                        for k in (
                            "lam",
                            "lam_raw",
                            "existing_adj_pct",
                            "total_adj_pct",
                            "existing_raw_pct",
                            "new_pct",
                        )
                    },
                    lam_ci_lo=float(b["lam"].quantile(0.025)),
                    lam_ci_hi=float(b["lam"].quantile(0.975)),
                )
                rows.append(r)
            ba = pl.DataFrame(bd["all"])
            t2.append(
                dict(
                    event=ev,
                    unit=unit,
                    **d,
                    pre_weekly_sd_pct=sd,
                    E=s["E"],
                    B=s["B"],
                    CF=s["CF"],
                    NEW=s["NEW"],
                    E_nobase=s["E_nobase"],
                    se_lam=float(ba["lam"].std()),
                    se_existing_adj_pct=float(ba["existing_adj_pct"].std()),
                    se_total_adj_pct=float(ba["total_adj_pct"].std()),
                    **{} if unit != "V" else pa,
                )
            )
            print(
                f"[{ev}] {unit:9s} lambda {d['lam']:.3f} (raw {d['lam_raw']:.3f}); existing raw {d['existing_raw_pct']:.1f}% adj {d['existing_adj_pct']:.1f}%, new {d['new_pct']:.1f}%, total adj {d['total_adj_pct']:.1f}%, pre weekly sd {sd:.1f}%",
                flush=True,
            )
    RES_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(RES_DIR / "lambda_event.csv")
    pl.DataFrame(t2).write_csv(RES_DIR / "table2_volume.csv")
    pl.concat(draws).write_parquet(RES_DIR / "lambda_boot.parquet")
    with pl.Config(tbl_rows=60, tbl_width_chars=220, float_precision=3):
        print(
            pl.DataFrame(rows).select(
                "event",
                "unit",
                "split",
                "lam",
                "se_lam",
                "lam_ci_lo",
                "lam_ci_hi",
                "lam_raw",
                "existing_adj_pct",
                "se_existing_adj_pct",
                "new_pct",
            )
        )


# Stage orchestration
def run(paths, cfg, log):
    from Run_All import invoke

    invoke(paths, log, "Estimate_Lambda", "lambda_est", [cfg["lambda_draws"]])
    return [
        paths.results / f
        for f in [
            "lambda_event.csv",
            "table2_volume.csv",
            "lambda_boot.parquet",
        ]
    ]


TASKS = {"lambda_est": lambda_est_main}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        TASKS[task]()
    else:
        from Run_All import main

        main("Estimate_Lambda")
