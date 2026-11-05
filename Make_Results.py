"""Fill the table shells and compile the PDF."""

from __future__ import annotations

import sys


# Fill Table Shells
"""
Fill the table shells with the baseline estimates: Tables 1-5, Table 6 column (1),
Table 6B (baseline similarity, alternative lambda specifications), and Table 9 Panel C.
Reads the pipeline's outputs (results/, data/) and the untouched shell
(Table_Shells_Overleaf/main_shell_template.tex) and writes Table_Shells_Overleaf/main.tex.  Cells the baseline does not
cover keep their grey placeholders; notes are amended where the estimates differ from the shell's plan (fixed effects,
placebo dates, sample).  Every replacement checks that the shell text it replaces occurs exactly once.

Usage: python fill_table_shells.py          (after the pipeline, logs/run_baseline.sh)
Output: Table_Shells_Overleaf/main.tex
"""
from config import PATHS, EVENT_SPEC as ORIGINAL_EVENTS
from math import erf, sqrt
from pathlib import Path
import polars as pl
from Build_Panel import EVENT_SPEC, EVENTS, TABLE_EVENTS

ROOT = PATHS.output
RES = PATHS.results
SHELL = PATHS.shells
T2_COLS = [
    "2016-08-30",
    "2018-02-16",
    "2021-04-27",
    "2021-10-08",
    "SPX",
    "2022-11-16",
    "2024-04-18",
]
NA = "--"


def num(x: float, d: int = 2) -> str:
    x = 0.0 if round(x, d) == 0 else x
    return f"${x:.{d}f}$" if x < 0 else f"{x:.{d}f}"


def se(x: float, d: int = 2) -> str:
    return f"({x:.{d}f})"


def tv(est: float, s: float) -> str:
    """t-statistic in parentheses: the estimate over its standard error (user's choice 2026-11-02)."""
    return num(est / s).join("()")


def stars(p: float) -> str:
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def bstar(est: float, s: float, d: int = 2) -> str:
    """Estimate with stars from its bootstrap t, two-sided normal approximation (user's choice 2026-11-02)."""
    x = 0.0 if round(est, d) == 0 else est
    return f"${x:.{d}f}{stars(1 - erf(abs(est / s) / sqrt(2)))}$"


BSTAR_NOTE = "Stars from the two-sided normal approximation to these $t$-statistics: $^{*}$, $^{**}$, $^{***}$: 10\\%, 5\\%, 1\\%. The bootstrap resamples days within the event window and does not capture shifts between windows; the randomization inference of Table 4 does."


def rd(name: str, **over) -> pl.DataFrame:
    return pl.read_csv(RES / name, schema_overrides=over or None)


def row(df: pl.DataFrame, **eq) -> dict:
    for k, v in eq.items():
        df = df.filter(pl.col(k) == v)
    assert df.height == 1, eq
    return df.row(0, named=True)


class Shell:

    def __init__(self, text: str):
        self.lines = text.split("\n")

    def _find(self, pred) -> int:
        idx = [i for i, l in enumerate(self.lines) if pred(l)]
        assert len(idx) == 1, f"{len(idx)} matching lines"
        return idx[0]

    def line(
        self, prefix: str, new: str, then: str | None = None, amp: int | None = None
    ) -> None:
        """Replace the line starting with prefix (and, with amp, having that many &), and optionally the next one."""
        i = self._find(
            lambda l: l.startswith(prefix) and (amp is None or l.count("&") == amp)
        )
        self.lines[i] = new
        if then is not None:
            self.lines[i + 1] = then

    def sub(self, old: str, new: str) -> None:
        text = "\n".join(self.lines)
        assert text.count(old) == 1, ("not found once", old[:70])
        self.lines = text.replace(old, new).split("\n")

    def text(self) -> str:
        return "\n".join(self.lines)


def table1(sh: Shell) -> None:
    vals = {}
    for ev in TABLE_EVENTS:
        c = pl.read_parquet(ROOT / "data" / ev / "cells.parquet").filter(
            ~pl.col("early_close")
        )
        tr = EVENT_SPEC[ev]["treated"]
        pre = c.filter(
            (pl.col("period") == "pre") & pl.col("underlying_symbol").is_in(tr)
        )
        post = c.filter(
            (pl.col("period") == "post") & pl.col("underlying_symbol").is_in(tr)
        )
        vol = (
            pre.group_by("underlying_symbol", "trade_date")
            .agg(pl.col("V").sum())
            .group_by("underlying_symbol")
            .agg(pl.col("V").mean())
        )
        cid = ["eff_wd", "dte_bucket", "delta_bucket", "option_type"]
        ex = dict(
            pre.filter(~pl.col("is_new"))
            .group_by("underlying_symbol")
            .agg(pl.struct(cid).n_unique())
            .iter_rows()
        )
        nw = dict(
            post.filter(pl.col("is_new"))
            .group_by("underlying_symbol")
            .agg(pl.struct(cid).n_unique())
            .iter_rows()
        )
        cells = (
            f"{ex[tr[0]]} / {nw[tr[0]]}"
            if len(tr) == 1 or (ex[tr[0]] == ex[tr[1]] and nw[tr[0]] == nw[tr[1]])
            else f"{ex[tr[0]]}, {ex[tr[1]]} / {nw[tr[0]]}, {nw[tr[1]]}"
        )
        vals[ev] = f"{vol['V'].sum() / 1000000000.0:.2f} & {cells}"
    nd = "\\multicolumn{2}{c}{\\textit{no index option data}}"
    sh.line(
        "1  & SPY      & Wed",
        f"1  & SPY      & Wed      & 2016-08-30               & 1           & N100, R2000      & {vals['2016-08-30']} \\\\",
    )
    sh.line(
        "2  & SPY      & Mon",
        f"2  & SPY      & Mon      & 2018-02-16               & 2           & N100, R2000      & {vals['2018-02-16']} \\\\",
    )
    sh.line(
        "3  & QQQ      & Mon, Wed",
        f"3  & QQQ      & Mon, Wed & 2021-04-23/27            & 1           & S\\&P\\,500, R2000  & {vals['2021-04-27']} \\\\",
    )
    sh.line(
        "4  & IWM      & Mon, Wed",
        f"4  & IWM      & Mon, Wed & 2021-10-05/08            & 1           & S\\&P\\,500, N100   & {vals['2021-10-08']} \\\\",
    )
    sh.line(
        "5  & SPX      & Tue, Thu",
        f"5  & SPX      & Tue, Thu & 2022-04-18, 05-11        & 3           & N100, R2000      & {nd} \\\\",
    )
    sh.line(
        "6  & SPY, QQQ & Tue, Thu",
        f"6  & SPY, QQQ & Tue, Thu & 2022-11-14/16            & 3           & R2000            & {vals['2022-11-16']} \\\\",
    )
    sh.line(
        "7  & IWM      & Tue, Thu",
        f"7  & IWM      & Tue, Thu & 2024-04-16/18            & 3           & S\\&P\\,500, N100   & {vals['2024-04-18']} \\\\",
    )
    sh.line(
        "8+ & NDX, XSP",
        f"8  & NDX      & Tue, Thu & 2022-08-08/10            & 3           & S\\&P\\,500, R2000  & {nd} \\\\"
        + "\n"
        + f"9  & XSP      & Tue, Thu & 2022-10-03/12            & 3           & N100, R2000      & {nd} \\\\",
    )
    sh.sub(
        "Pre-event volume is the mean daily delta notional over the 60 pre-event sessions.",
        "Pre-event volume is the mean daily delta notional over the 60 pre-event sessions (early closes excluded; all trades, $|\\Delta|$ at the common volatility; event 6 sums SPY and QQQ).",
    )
    sh.sub(
        "Cells are defined by expiration weekday $\\times$ days-to-expiration bucket $\\times$ delta bucket $\\times$ call/put.",
        "Cells are defined by expiration weekday $\\times$ days-to-expiration bucket $\\times$ delta bucket $\\times$ call/put (five delta buckets, Deep ITM included); the column counts the distinct existing cells traded before the event and the distinct new cells traded after it, per treated product.",
    )
    sh.sub(
        "Listing dates for ETFs are from the existing paper's Table 1; $^\\dagger$ marks dates and calendars to be verified against Cboe and Nasdaq notices.",
        "Listing dates of the ETF events are the first trades in the Cboe trade files; those of SPX (SPXW), NDX (NDXP) and XSP are first trading days from exchange notices (Cboe C2022030800 and C2022050401; Nasdaq Options Trader Alert 2022-26; Cboe C2022091900); each of the three already had Monday, Wednesday and Friday expirations. Event 7 is shown in Tables 1 and 2 and in Panel A of Table 4 only. Baseline estimates: events 1--4 and 6.",
    )


def table2(sh: Shell) -> None:
    t2 = rd("table2_volume.csv", event=pl.Utf8).filter(pl.col("unit") == "V")
    V = {e: row(t2, event=e) for e in TABLE_EVENTS}

    def r(fn):
        return " & ".join((NA if e == "SPX" else fn(V[e]) for e in T2_COLS))

    sh.line(
        "Contracts        &",
        "Contracts        & " + r(lambda v: f"{v['share_contracts']:.1f}") + " \\\\",
    )
    sh.line(
        "Delta notional   &",
        "Delta notional   & " + r(lambda v: f"{v['share_V']:.1f}") + " \\\\",
    )
    sh.line(
        "Number of trades &",
        "Number of trades & " + r(lambda v: f"{v['share_trades']:.1f}") + " \\\\",
    )
    sh.line(
        "Premium (\\$)     &",
        "Premium (\\$)     & " + r(lambda v: f"{v['share_premium']:.1f}") + " \\\\",
    )
    sh.line(
        "Existing, weekday baseline only &",
        "Existing, weekday baseline only & "
        + r(lambda v: num(v["existing_raw_pct"], 1))
        + " \\\\",
    )
    sh.line(
        "Existing, control-adjusted      &",
        "Existing, control-adjusted      & "
        + r(lambda v: bstar(v["existing_adj_pct"], v["se_existing_adj_pct"], 1))
        + " \\\\",
        then="                                & "
        + r(lambda v: tv(v["existing_adj_pct"], v["se_existing_adj_pct"]))
        + " \\\\",
    )
    sh.line(
        "New contracts                   &",
        "New contracts                   & "
        + r(lambda v: num(v["new_pct"], 1))
        + " \\\\",
    )
    sh.line(
        "Total, control-adjusted         &",
        "Total, control-adjusted         & "
        + r(lambda v: bstar(v["total_adj_pct"], v["se_total_adj_pct"], 1))
        + " \\\\",
        then="                                & "
        + r(lambda v: tv(v["total_adj_pct"], v["se_total_adj_pct"]))
        + " \\\\",
    )
    sh.line(
        "Pre-event weekly s.d.\\ of existing volume (\\%) &",
        "Pre-event weekly s.d.\\ of existing volume (\\%) & "
        + r(lambda v: f"{v['pre_weekly_sd_pct']:.1f}")
        + " \\\\",
    )
    sh.sub(
        "Volumes in Panel B are in delta notional (black entries are contract counts from the existing report and will be replaced). Block-bootstrap standard errors (resampling trading days) in parentheses.",
        "Volumes in Panel B are in delta notional. $t$-statistics in parentheses: the estimate over its block-bootstrap standard error (five-session blocks resampled within the pre- and post-event windows, counterfactual recomputed; 999 draws). "
        + BSTAR_NOTE
        + " Contracts, delta notional and trades count every trade. 2022 SPX: no index option data; 2024 IWM is shown in Tables 1 and 2 and in Panel A of Table 4 only.",
    )


def table3(sh: Shell) -> None:
    rf = rd("reduced_form_coefs.csv", sample=pl.Utf8)
    cols = [
        row(rf, sample="pooled", model=f"t3_{c}", term="PTN")
        for c in ("all", "ATM", "OTM", "ITM")
    ]
    sh.line(
        "Post $\\times$ Treat                    &",
        "Post $\\times$ Treat                    & \\multicolumn{4}{c}{\\textit{absorbed by the ETF $\\times$ day fixed effects}} \\\\",
        then="                                       &       &       &       &       \\\\",
    )
    sh.line(
        "Post $\\times$ Treat $\\times$ $N_{j,t}$ &",
        "Post $\\times$ Treat $\\times$ $N_{j,t}$ & "
        + " & ".join((f"${c['estimate']:.3f}{stars(c['p'])}$" for c in cols))
        + " \\\\",
        then="                                       & "
        + " & ".join((tv(c["estimate"], c["se"]) for c in cols))
        + " \\\\",
    )
    sh.line(
        "Family $\\times$ DTE bucket $\\times$ day fixed effects &",
        "ETF $\\times$ day fixed effects                        & Yes & Yes & Yes & Yes \\\\"
        + "\n"
        + "DTE bucket $\\times$ day fixed effects (pooled)       & Yes & Yes & Yes & Yes \\\\",
    )
    sh.line(
        "Observations (cell-days)",
        "Observations (cell-days)                             & "
        + " & ".join((f"{c['nobs']:,}" for c in cols))
        + " \\\\",
        amp=4,
    )
    sh.sub(
        "Standard errors two-way clustered by cell and day; randomization-inference $p$-values in the appendix.",
        "Fixed effects as estimated: cell, ETF $\\times$ day and DTE bucket $\\times$ day, the last common to an event's treated and control ETFs. The coefficient therefore compares, within the same DTE bucket and day, treated cells that lose nearest-expiry status with the control cells in the same calendar position. The planned family $\\times$ DTE bucket $\\times$ day effects would absorb it: in the treated post-event sample $N_{j,t}$ is constant within every such group in 2016, 2021 IWM and 2022, and varies in one group in 2021 QQQ and twelve in 2018. Deep ITM cells (no characteristics) are not in the sample; zero-volume cell-days dropped; weighted by pre-event volume. $t$-statistics in parentheses, from standard errors two-way clustered by cell and day; $^{*}$, $^{**}$, $^{***}$: 10\\%, 5\\%, 1\\%; randomization-inference $p$-values in the appendix.",
    )


def table4(sh: Shell) -> None:
    le = rd("table2_volume.csv", event=pl.Utf8)
    ri = rd("lambda_ri.csv", event=pl.Utf8).filter(pl.col("unit") == "V")
    dn = rd("lambda_density.csv").row(0, named=True)
    dri = rd("lambda_density_ri.csv").row(0, named=True)

    def ev_row(lab: str, ev: str) -> str:
        lam = {
            u: row(le, event=ev, unit=u)["lam"] for u in ("V", "contracts", "premium")
        }
        r = row(ri, event=ev)
        p = f"$<1/{r['n_placebo']}$" if r["ri_p"] == 0 else f"{r['ri_p']:.2f}"
        s_v = row(le, event=ev, unit="V")["se_lam"]
        return (
            f"{lab} & {EVENT_SPEC[ev]['n_e']} & {bstar(lam['V'], s_v)} & "
            + " & ".join((num(lam[u]) for u in ("contracts", "premium")))
            + f" & {tv(lam['V'], s_v)} & {p} "
            + "\\\\"
        )

    sh.sub(
        "& Bootstrap s.e. & RI $p$-value \\\\", "& Bootstrap $t$ & RI $p$-value \\\\"
    )
    sh.line("SPY 2016 (Wed)", ev_row("SPY 2016 (Wed)         ", "2016-08-30"))
    sh.line("SPY 2018 (Mon)", ev_row("SPY 2018 (Mon)         ", "2018-02-16"))
    sh.line("QQQ 2021 (Mon, Wed)", ev_row("QQQ 2021 (Mon, Wed)    ", "2021-04-27"))
    sh.line("IWM 2021 (Mon, Wed)", ev_row("IWM 2021 (Mon, Wed)    ", "2021-10-08"))
    sh.line(
        "SPX 2022 (Tue, Thu)",
        "SPX 2022 (Tue, Thu)     & 3 & \\multicolumn{5}{c}{\\textit{no index option data}} \\\\",
    )
    sh.line("SPY/QQQ 2022 (Tue, Thu)", ev_row("SPY/QQQ 2022 (Tue, Thu)", "2022-11-16"))
    sh.line("IWM 2024 (Tue, Thu)", ev_row("IWM 2024 (Tue, Thu)    ", "2024-04-18"))
    i = sh._find(lambda l: "\\multicolumn{2}{c}{$\\hat a$}" in l)
    sh.lines[i] = (
        "         & \\multicolumn{2}{c}{$\\hat a$} & \\multicolumn{2}{c}{$\\hat b$} & \\multicolumn{2}{c}{Test: $b = 0$} \\\\"
    )
    pb = lambda x: "$<0.01$" if x < 0.01 else f"{x:.2f}"
    sh.line(
        "Estimate & \\multicolumn{2}{c}{\\ph}",
        f"Logistic $\\Lambda$ (design) & \\multicolumn{{2}}{{c}}{{{bstar(dn['a'], dn['se_a'], 1)}}} & \\multicolumn{{2}}{{c}}{{{bstar(dn['b'], dn['se_b'], 1)}}} & \\multicolumn{{2}}{{c}}{{$p =$ {pb(dn['p_b0'])}}} \\\\",
        then=f"         & \\multicolumn{{2}}{{c}}{{{tv(dn['a'], dn['se_a'])}}} & \\multicolumn{{2}}{{c}}{{{tv(dn['b'], dn['se_b'])}}} & \\multicolumn{{2}}{{c}}{{}} \\\\"
        + "\n"
        + f"Identity $\\Lambda$ & \\multicolumn{{2}}{{c}}{{{bstar(dn['a_lin'], dn['se_a_lin'])}}} & \\multicolumn{{2}}{{c}}{{{bstar(dn['b_lin'], dn['se_b_lin'])}}} & \\multicolumn{{2}}{{c}}{{$p =$ {pb(dn['p_b0_lin'])}; RI $p =$ {dri['ri_p']:.2f}}} \\\\"
        + "\n"
        + f"         & \\multicolumn{{2}}{{c}}{{{tv(dn['a_lin'], dn['se_a_lin'])}}} & \\multicolumn{{2}}{{c}}{{{tv(dn['b_lin'], dn['se_b_lin'])}}} & \\multicolumn{{2}}{{c}}{{}} \\\\",
    )
    n_pl = (
        ", ".join(
            (f"{row(ri, event=ev)['n_placebo']:,}".replace(",", "{,}") for ev in EVENTS)
        )
        + "; IWM 2024: all "
        + f"{row(ri, event='2024-04-18')['n_placebo']:,}".replace(",", "{,}")
        + ", each before 2024"
    )
    sh.sub(
        "RI $p$-value: share of placebo event dates with $|\\hat\\lambda|$ at least as large.",
        "RI $p$-value: share of the event's pre-event placebo dates (whole 120-session window before the event's new weekday was listed, no listing in it; 2018 also after SPY's Wednesday listing; "
        + n_pl
        + " dates) with $|\\hat\\lambda|$ at least as large; the 2016 windows start within two months of each other and overlap almost entirely.",
    )
    sh.sub(
        "In Panel B, $\\Lambda$ is the logistic function, estimated by nonlinear least squares across events with bootstrap standard errors.",
        "In Panel B, $\\Lambda$ is the logistic function, estimated by nonlinear least squares across events with bootstrap standard errors (999 draws) on the five events of 2016--2022 (IWM 2024 is in Panel A only); with $\\lambda_{2016}>1$ and $\\lambda_{2022}<0$ it saturates, so the identity $\\Lambda$ (linear least squares) is also shown. $p$: bootstrap test of $b=0$; RI $p$: share of 999 random draws of one placebo date per event with a slope at least as large in absolute value. All figures in parentheses are $t$-statistics, the estimate over its bootstrap standard error (Panel A: $\\lambda$ in delta notional, null $\\lambda=0$). Stars on $\\lambda$ (delta notional), $\\hat a$ and $\\hat b$: two-sided normal approximation to these $t$-statistics, 10\\%, 5\\%, 1\\%; the randomization inference is the test across windows.",
    )


def table5(sh: Shell) -> None:
    rf = rd("reduced_form_coefs.csv", sample=pl.Utf8)
    models = ["t5_1", "t5_2", "t5_3", "t5_4", "t5_5"]
    sterm = {"t5_2": "Sbar", "t5_3": "Sbar", "t5_4": "PTS", "t5_5": "Sbar"}
    nterm = {"t5_1": "N", "t5_3": "N", "t5_4": "PTN", "t5_5": "N"}

    def cells(terms, what):
        out = []
        for m in models:
            if m not in terms:
                out.append("")
                continue
            r = row(rf, sample="pooled", model=m, term=terms[m])
            out.append(
                f"${r['estimate']:.3f}{stars(r['p'])}$"
                if what == "est"
                else tv(r["estimate"], r["se"])
            )
        return " & ".join(out)

    meta = {
        m: rf.filter((pl.col("sample") == "pooled") & (pl.col("model") == m)).row(
            0, named=True
        )
        for m in models
    }
    sh.line(
        "Similarity $\\bar S_{j,t}$",
        "Similarity $\\bar S_{j,t}$         & " + cells(sterm, "est") + " \\\\",
        then="                                  & " + cells(sterm, "se") + " \\\\",
    )
    sh.line(
        "Nearest-expiry loss $N_{j,t}$",
        "Nearest-expiry loss $N_{j,t}$     & " + cells(nterm, "est") + " \\\\",
        then="                                  & " + cells(nterm, "se") + " \\\\",
    )
    sh.line(
        "Day fixed effects                    &",
        "Day fixed effects (ETF $\\times$ day)  & Yes & Yes & Yes & Yes & Yes \\\\"
        + "\n"
        + "Cell and DTE bucket $\\times$ day fixed effects & No & No & No & Yes & No \\\\",
    )
    sh.line(
        "Control families (phantom similarity) &",
        "Control families (phantom similarity) & No  & No  & No  & Yes & No \\\\",
    )
    sh.line(
        "Observations (cell-days)",
        "Observations (cell-days)             & "
        + " & ".join((f"{meta[m]['nobs']:,}" for m in models))
        + " \\\\",
        amp=5,
    )
    sh.line(
        "$R^2$                                &",
        "$R^2$                                & "
        + " & ".join((f"{meta[m]['r2']:.3f}" for m in models))
        + " \\\\",
    )
    th = rd("structural_estimates.csv").filter(pl.col("version") == "L")["theta"][0]
    sh.sub(
        "Regressions weighted by pre-event volume. Standard errors two-way clustered by cell and day.",
        "Here $\\hat\\theta="
        + f"{th:.1f}"
        + "$. Day fixed effects are ETF $\\times$ day; column (4) adds cell and DTE bucket $\\times$ day effects (the latter common to treated and control ETFs). Column (5) adds cell-type effects to column (3) on the treated products only (the shell's header lists control families for it); the design's eq.~(17) itself includes the cell-type effects. Deep ITM cells (no characteristics) are not in the samples. Regressions weighted by pre-event volume. $t$-statistics in parentheses, from standard errors two-way clustered by cell and day; $^{*}$, $^{**}$, $^{***}$: 10\\%, 5\\%, 1\\%.",
    )


def table6(sh: Shell) -> None:
    s = rd("structural_estimates.csv")
    L, O = (row(s, version="L"), row(s, version="L_oos"))

    def first(line_: str, value: str) -> str:
        """Replace the first placeholder of a shell row (column 1)."""
        for ph in ("\\phse", "\\ph"):
            j = line_.find("& " + ph)
            if j >= 0:
                return line_[:j] + "& " + value + line_[j + 2 + len(ph) :]
        raise AssertionError(line_)

    for prefix, est, err in [
        ("$a$ (level of $\\lambda$)", bstar(L["a"], L["se_a"]), tv(L["a"], L["se_a"])),
        (
            "$b$ (density slope of $\\lambda$)",
            bstar(L["b"], L["se_b"]),
            tv(L["b"], L["se_b"]),
        ),
        (
            "$\\theta$ (bandwidth)",
            f"{L['theta']:.1f}",
            f"[{L['p05_theta']:.0f}, {L['p95_theta']:.0f}]",
        ),
        (
            "$\\gamma$ (nearest-expiry)",
            bstar(L["gamma"], L["se_gamma"], 1),
            tv(L["gamma"], L["se_gamma"]),
        ),
    ]:
        i = sh._find(lambda l, p=prefix: l.startswith(p))
        sh.lines[i] = first(sh.lines[i], est)
        sh.lines[i + 1] = first(sh.lines[i + 1], err)
    for prefix, value in [
        ("In-sample weighted RMSE", f"{L['rmse']:.3f} [{L['rmse0']:.3f}]"),
        ("Out-of-sample weighted RMSE", f"{O['rmse_oos']:.3f} [{O['rmse0_oos']:.3f}]"),
        (
            "Out-of-sample rank correlation",
            f"{bstar(O['rank_corr_oos'], O['se_rank_corr_oos'])} {tv(O['rank_corr_oos'], O['se_rank_corr_oos'])}",
        ),
    ]:
        i = sh._find(lambda l, p=prefix: l.startswith(p))
        sh.lines[i] = first(sh.lines[i], value)
    sh.sub(
        "Rank correlation is between predicted and realized cell-level losses. Block-bootstrap standard errors.",
        "Rank correlation is between predicted and realized cell-level losses. Block-bootstrap standard errors. Column (1) is the baseline, with $t$-statistics in parentheses and stars from their two-sided normal approximation (10\\%, 5\\%, 1\\%): the estimate over its block-bootstrap standard error (499 draws, each re-estimating $\\Sigma_p$ and the distances); for $\\theta$ the bootstrap 5th and 95th percentiles in brackets (its bootstrap distribution has a long right tail); RMSE of $y$ with the no-loss RMSE in brackets; out of sample: estimated on 2016, 2018 and the two 2021 events, evaluated on 2022 only (IWM 2024 is not estimated). The model's choice set and new volume exclude Deep ITM cells, which have no characteristics.",
    )


def table9(sh: Shell) -> None:
    ri = rd("lambda_ri.csv", event=pl.Utf8).filter(pl.col("unit") == "V")
    br = rd("beta_ri.csv").filter((pl.col("spec") == "t5_3") & (pl.col("term") == "S"))
    lam = [f"{row(ri, event=e)['pctile']:.0f}" for e in ("2016-08-30", "2022-11-16")]
    beta = [f"{row(br, event=e)['pctile']:.1f}" for e in ("2016-08-30", "2022-11-16")]
    sh.line(
        "Percentile of actual $\\hat\\lambda$ among placebo dates",
        "Percentile of actual $\\hat\\lambda$ among placebo dates          & "
        + " & ".join(lam)
        + f" & {NA} "
        + "\\\\",
    )
    sh.line(
        "Percentile of actual similarity coefficient among placebo dates",
        "Percentile of actual similarity coefficient among placebo dates & "
        + " & ".join(beta)
        + f" & {NA} "
        + "\\\\",
    )
    n16, n22 = (row(ri, event=e)["n_placebo"] for e in ("2016-08-30", "2022-11-16"))
    sh.sub(
        "Panel C reports the position of the actual estimates in the distribution of estimates from placebo event dates drawn from non-event periods.",
        "Panel C reports the position of the actual estimates in the distribution of estimates from placebo event dates drawn from non-event periods: baseline, pre-event placebo dates ("
        + f"{n16} for 2016, whose windows overlap almost entirely; "
        + f"{n22:,}".replace(",", "{,}")
        + " for 2022); percentile = share of placebo estimates at or below the actual one; the similarity coefficient is that of eq.~(17) with $N$, on the calendar similarity. 2024 is not estimated.",
    )


def baseline_lambda_table() -> str:
    """Report the three full-sample fits at the same baseline similarity."""
    estimates = rd("structural_estimates.csv")
    versions = ("L", "I", "F")
    fits = {v: row(estimates, version=v) for v in versions}
    forms = {"L": "logistic", "I": "identity", "F": "free"}
    records = []

    def estimate_rows(label, term, section, digits=3, event=None):
        values, statistics = [], []
        for version in versions:
            fit = fits[version]
            value = fit.get(term)
            if value is None:
                values.append(NA)
                statistics.append("")
                continue
            error = fit["se_" + term]
            t_stat = value / error
            normal_p = 1 - erf(abs(t_stat) / sqrt(2))
            values.append(bstar(value, error, d=digits))
            statistics.append(tv(value, error))
            records.append(
                dict(
                    version=version,
                    lambda_form=forms[version],
                    section=section,
                    term=term,
                    event=event,
                    n_e=EVENT_SPEC[event]["n_e"] if event else None,
                    estimate=value,
                    bootstrap_se=error,
                    t_stat=t_stat,
                    normal_p_value=normal_p,
                    significance_stars=stars(normal_p).strip("^{}"),
                    lambda_zero_boundary=version == "L" and section == "event_lambda",
                    n_boot=fit["n_boot_ok"],
                )
            )
        return (
            label + " & " + " & ".join(values) + r" \\"
            + "\n & " + " & ".join(statistics) + r" \\"
        )

    parameters = "\n".join(
        estimate_rows(label, term, "parameters")
        for label, term in (
            (r"$a$ (level)", "a"),
            (r"$b$ (calendar-density slope)", "b"),
            (r"$\theta$ (bandwidth)", "theta"),
            (r"$\gamma$ (nearest-expiry)", "gamma"),
        )
    )
    intervals = (
        r"$\theta$: bootstrap 5th--95th percentiles & "
        + " & ".join(
            f"{fits[v]['p05_theta']:.2f}--{fits[v]['p95_theta']:.2f}"
            for v in versions
        )
        + r" \\"
    )
    event_rows = "\n".join(
        estimate_rows(
            label + f" ($n_e={EVENT_SPEC[event]['n_e']}$)",
            "lam_" + key,
            "event_lambda",
            digits=4,
            event=event,
        )
        for label, event, key in (
            ("SPY 2016", "2016-08-30", "2016"),
            ("SPY 2018", "2018-02-16", "2018"),
            ("QQQ 2021", "2021-04-27", "2021QQQ"),
            ("IWM 2021", "2021-10-08", "2021IWM"),
            ("SPY/QQQ 2022", "2022-11-16", "2022"),
        )
    )
    body = r"""
\clearpage
\section*{Table 6B: Baseline Diversion Estimates}
\begingroup
\renewcommand{\thetable}{6B}
\renewcommand{\theHtable}{baseline-lambda-models}
\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{threeparttable}
\caption{Baseline Mahalanobis Similarity: Alternative Diversion Specifications}
\begin{tabularx}{\textwidth}{l*{3}{C}}
\toprule
 & Logistic L$^\dagger$ & Identity I & Free F \\
$\lambda_e$ specification & $\Lambda(a+b n_e)$ & $a+b n_e$ & Event-specific \\
Permitted range of $\lambda_e$ & $(0,1)$ & Unrestricted & Unrestricted \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Model-specific parameters}} \\
@@PARAMETERS@@
@@INTERVALS@@
\midrule
\multicolumn{4}{l}{\textit{Panel B: Event-level diversion estimates}} \\
@@EVENTS@@
\bottomrule
\end{tabularx}
\begin{tablenotes}
\footnotesize
\item[] \textit{Notes:} All three columns use the same baseline similarity:
Mahalanobis distance on log remaining trading hours and $|\Delta|$, with common-volatility
characteristics and a pre-event, volume-weighted, shrunk covariance matrix.
The columns differ only in the parameterization of $\lambda_e$ and are fitted separately:
each has its own $\theta$ and $\gamma$; L and I also have their own $a$ and $b$.
$\Lambda$ is the logistic function. F estimates one unrestricted $\lambda_e$ per event,
so $a$ and $b$ do not apply. Within each column, $\theta$ and $\gamma$ are shared across events.
\item[] All entries are full-sample, in-sample estimates using the five ETF events of 2016--2022
and post-event sessions $\tau=0,\ldots,59$ (early closes excluded).
IWM 2024 has no structural fit. Deep ITM cells are excluded from the structural choice set and
new volume; cells affected by the 2022 expiration cut are excluded from the fit.
$n_e$ is the number of regular expiration weekdays before the event.
L and I imply identical $\lambda_e$ for events with the same $n_e$; those rows are not independent estimates.
\item[] Parentheses below estimates contain $t$-statistics: the estimate divided by its
block-bootstrap standard error. Stars use a two-sided standard-normal approximation
for a zero coefficient: $^{*}$, $^{**}$, $^{***}$ denote 10\%, 5\%, and 1\%.
These stars do not use RI $p$-values.
The explicitly labeled percentile row gives the 5th--95th percentile range for $\theta$.
The bootstrap resamples five-session blocks separately before and after each event and re-estimates
the covariance matrices, distances, and model parameters.
Valid bootstrap draws: @@BOOTSTRAP_COUNTS@@.
$^\dagger$ L restricts $\lambda_e$ to $(0,1)$; its $\lambda_e$ stars are only the displayed
normal-approximation Wald calculation. Because zero is a boundary value, they are not a
regular test of $\lambda_e=0$. No structural RI test has been computed.
Table 4 reports the separate aggregate-volume ratio; these structural estimates fit the
weighted pattern of cell-level losses. Out-of-sample results are reported separately in Table 6.
\end{tablenotes}
\end{threeparttable}
\end{table}
\addtocounter{table}{-1}
\endgroup
"""
    for marker, replacement in (
        ("@@PARAMETERS@@", parameters),
        ("@@INTERVALS@@", intervals),
        ("@@EVENTS@@", event_rows),
        (
            "@@BOOTSTRAP_COUNTS@@",
            ", ".join(f"{v}={fits[v]['n_boot_ok']}" for v in versions),
        ),
    ):
        assert body.count(marker) == 1
        body = body.replace(marker, replacement)
    pl.DataFrame(records, infer_schema_length=None).write_csv(
        SHELL / "baseline_lambda_models.csv"
    )
    preamble = (SHELL / "main_shell_template.tex").read_text().split(
        r"\begin{document}", 1
    )[0]
    (SHELL / "baseline_lambda_models.tex").write_text(
        preamble + "\n" + r"\begin{document}" + "\n" + body
        + "\n" + r"\end{document}" + "\n"
    )
    return body


def fill_table_shells_main() -> None:
    sh = Shell((SHELL / "main_shell_template.tex").read_text())
    sh.sub(
        "\\date{October 2026}",
        "\\date{October 2026; baseline estimates filled in November 2026}",
    )
    sh.sub(
        "The tables follow the paper's chain of argument. Grey \\ph{} entries are placeholders. Numbers in black are taken from the existing reports (\\textit{Existing and New Contracts after the Weekly-Expiration Listings} and \\textit{Is the Control Group Affected by the New Expirations?}) and must be recomputed under the new definitions (delta notional, control-family adjustment) before use. Entries marked $^\\dagger$ must be verified against exchange notices.",
        "The tables follow the paper's chain of argument. Black entries in Tables 1--6, Table 6B and Panel C of Table 9 are the baseline estimates (design steps 2--6: the five ETF events with data, delta notional at a common volatility, the control-adjusted counterfactual, Mahalanobis similarity on log trading hours and $|\\Delta|$); the baseline report gives the details. Grey \\ph{} entries are steps the baseline does not cover: Table 6, columns (2)--(5), Tables 7, 8 and 10, and Panels A--B of Table 9, whose black entries come from the existing control-group report. ``--'' marks events without data or not estimated. Listing dates are verified against exchange notices. Figures in parentheses are $t$-statistics (the estimate over its standard error), as in Panel A of Table 9; the notes name the standard errors behind them. Table 6B compares logistic, identity and free-event diversion at baseline similarity, with t-statistics in parentheses and stars from a two-sided normal approximation; its notes identify the logistic boundary restriction.",
    )
    for fn in (table1, table2, table3, table4, table5, table6, table9):
        fn(sh)
    overview_row = (
        "6  & 3. Mechanism           & Do investors substitute on contract terms, "
        "exposure, or payoffs? Is there a salience effect? & New \\\\"
    )
    sh.sub(
        overview_row,
        overview_row
        + "\n6B & 3. Mechanism & How does the diversion estimate change across "
        "logistic, identity and free-event specifications at baseline similarity? & New \\\\",
    )
    sh.sub(
        "\\begin{table}[H]\n\\small\n\\begin{tabularx}{\\textwidth}{clLl}",
        "\\begin{table}[H]\n\\footnotesize\n\\begin{tabularx}{\\textwidth}{clLl}",
    )
    next_table = r"\clearpage" + "\n" + r"\section*{Table 7: Cross-Product Substitution}"
    sh.sub(next_table, baseline_lambda_table() + "\n" + next_table)
    (SHELL / "main.tex").write_text(sh.text())
    print("wrote", SHELL / "main.tex")


# Stage orchestration
def run(paths, cfg, log):
    import shutil
    import subprocess
    from Run_All import invoke

    shutil.copy2(paths.shell_template, paths.shell_file("main_shell_template.tex"))
    invoke(paths, log, "Make_Results", "fill_table_shells")
    for stem in ("main", "baseline_lambda_models"):
        for _ in range(3):
            proc = subprocess.run(
                [
                    paths.executable("pdflatex"),
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    stem + ".tex",
                ],
                cwd=paths.shells,
                capture_output=True,
                text=True,
            )
            if proc.returncode:
                raise RuntimeError(proc.stdout[-5000:])
    return [
        paths.shell_file("main.tex"),
        paths.shell_file("main.pdf"),
        paths.shell_file("main.log"),
        paths.shell_file("baseline_lambda_models.tex"),
        paths.shell_file("baseline_lambda_models.pdf"),
        paths.shell_file("baseline_lambda_models.csv"),
        paths.shell_file("baseline_lambda_models.log"),
    ]


TASKS = {"fill_table_shells": fill_table_shells_main}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        TASKS[task]()
    else:
        from Run_All import main

        main("Make_Results")
