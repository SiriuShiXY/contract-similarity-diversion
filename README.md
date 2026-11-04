# Contract similarity and diversion: the baseline (design steps 2–6, five ETF events)

Implements the baseline of *Measuring Contract Similarity and Diversion Ratios* (similarity_diversion_design.pdf)
with the specification of Section 11.1, steps 2–6 of the implementation order (Section 11.2), and the tables of
table_shells.pdf:

- **Step 2:** the cell panel.
- **Step 3:** Counterfactual 1 and the reallocated share λ (shell Tables 2 and 4).
- **Step 4:** Mahalanobis similarity on G1 = (log trading hours τ, |delta| at a common volatility), and the distance-bin
  regression of eq. 8 with its figure.
- **Step 5:** the reduced form, eqs. 16–18 (shell Tables 3 and 5).
- **Step 6:** the structural diversion model, eqs. 19–21 (shell Table 6, column 1; Table 4, Panel B).

Inference uses block bootstrap and randomization inference with placebo event dates.

| Event (day 0) | New expirations | Treated | Control families | n_e |
|---|---|---|---|---|
| 2016-08-30 | SPY Wednesday | SPY | QQQ, IWM | 1 |
| 2018-02-16 | SPY Monday | SPY | QQQ, IWM | 2 |
| 2021-04-27 | QQQ Monday (from 04-23) and Wednesday | QQQ | SPY, IWM | 1 |
| 2021-10-08 | IWM Wednesday (from 10-05) and Monday | IWM | SPY, QQQ | 1 |
| 2022-11-16 | SPY, QQQ Tuesday (from 11-14) and Thursday | SPY, QQQ | IWM | 3 |

These are events 1–4 and 6 of shell Table 1: every ETF event with the OptionMetrics volatility.

- IWM 2024 (event 7) is in the trade files, but the OptionMetrics file ends 2023-12-29.
- SPX, NDX and XSP are not in the trade files.

**Trade data.**

- The complete daily Cboe trade files (SPY, QQQ, IWM) are on the external T7 drive:
  `/Volumes/T7/Options/OptionsTradeData.zip`, 3,348 sessions from 2012-01-03 to 2025-07-14.
- The paper's three event windows are also on local disk. `build_cells.raw_csv` reads a day locally if possible,
  otherwise from T7.
- 2021-09-29 has a file without QQQ trades. It is treated like a missing session (`build_cells.INCOMPLETE_DAYS`).

The first version (three events, 2026-10-31) is kept in `results/three_events/`, with its report PDF. Report Table S11
compares the two versions.

## Pipeline

```
python event_windows.py          # windows + expiration classes -> data/events/<event>/{days,expirations}.csv
                                 #   (five events + IWM 2024; reproduces ListingDynamics for 2016, 2018, 2022)
python build_cells.py            # step 2: trades -> data/<event>/cells.parquet, sigbar.csv            (~1 min/event)
python panel.py                  # grid, N, Counterfactual 1, y -> data/<event>/existing_panel.parquet
python mahalanobis.py            # step 4: Sigma_p, distances -> results/sigma.csv, data/<event>/pairs.parquet,
                                 #         data/reg_panel_all.csv
Rscript bin_regression.R         # eq. 8 (i) and (ii) -> results/bin_regression_*
python make_bin_figure.py        # results/figures/fig_distance_bins*.{png,pdf}
python make_distance_map.py      # results/figures/fig_distance_map.{png,pdf}
python lambda_est.py             # step 3: Table 2, lambda, bootstrap -> results/lambda_event.csv,
                                 #         table2_volume.csv, lambda_boot.parquet
python lambda_bridge.py          # raw 0.65/0.27/0.91 -> baseline lambda -> results/lambda_bridge.csv
python build_full_panel.py 8     # T7 files -> data/full/cells_<year>.parquet, 2016-01-06..2023-12-29        (~5 min)
python placebo.py                # RI for lambda and the density slope -> results/placebo_lambda.parquet,
                                 #         lambda_ri.csv, lambda_density_ri.csv                             (~20 s)
python structural.py 499         # step 6: Table 6 col. 1, bootstrap, Table 4 Panel B, lambda by new weekday
                                 #         -> results/structural_*.{csv,parquet}, lambda_density.csv,
                                 #            lambda_newweekday.csv                                        (~4 min)
python similarity_exposure.py    # Sbar at theta_hat -> data/reg_panel_rf.csv
python phantom.py                # phantom calendars, eq. 18 panel -> data/did18_all.csv
Rscript reduced_form.R           # step 5: Tables 3 and 5 -> results/reduced_form_*
python placebo_beta.py           # RI for the similarity and N coefficients -> results/beta_ri.csv           (~1 min)
python make_placebo_figure.py    # results/figures/fig_placebo_{lambda,beta}.{png,pdf}
python make_baseline_report.py   # overleaf/tables/*.tex, overleaf/numbers.tex, figures
python fill_table_shells.py      # baseline numbers into ../../Table_Shells_Overleaf/main.tex (from main_shell_template.tex)
```

All event metadata (dates, treated/control, added weekdays, n_e, already-treated controls, the 2022 cut) live in
`build_cells.EVENT_SPEC`. Run the numerical steps with `OMP_NUM_THREADS=1` (the bootstraps are much faster
single-threaded). `logs/run_baseline.sh` runs the whole list above except the report (~25 min);
`logs/run_five_events.sh` runs λ → structural → reduced form → RI only.

**IWM 2024 (shell Table 1 row 7; user's choice 2026-11-02): Tables 1 and 2 only.** `EVENT_SPEC["2024-04-18"]` has
`tables_only=True`: `build_cells.EVENTS` (every estimate) are the five baseline events, `TABLE_EVENTS` add IWM 2024.
Windows, cells, panel and `lambda_est.py` run for it (Table 2 → `table2_volume.csv`, which also carries the bootstrap
s.e. of Panel B and of λ), `placebo.py` gives its placebo s.d. (`table2_placebo_sd.csv`, the 2021 IWM configuration)
and its λ RI (`lambda_ri.csv`, all 1,129 dates; user asked 2026-11-02 to show it in Table 4 Panel A of the table shells,
not in Panel B's density fit); nothing else uses it. First trades of the new IWM Tuesday and Thursday series: 2024-04-16 and 2024-04-18 (day 0); controls SPY
and QQQ, which list Tuesdays and Thursdays since 2022-11.

**Robustness without deep-ITM trades** (`SIM_VARIANT=no_ditm`): the baseline until 2026-11-01, which dropped |delta| >
0.875 trades from every volume measure. Its outputs, as produced then, are in `data/robustness/no_ditm/` and
`results/robustness/no_ditm/`; the variant code path reproduces its cells exactly. `python compare_variant.py no_ditm`
writes `results/robustness/no_ditm/comparison.csv` (base = baseline with deep ITM, alt = without).

**Robustness with Cboe's own trade delta** (design 3.2, own implied volatilities; user's choice 2026-11-01):
`logs/run_vendor_delta.sh` reruns the whole pipeline (cells, steps 2–6, full-sample panel, RI) with
`SIM_VARIANT=vendor_delta`. The vendor |delta| replaces the common-volatility |delta| in the delta bucket, the delta
notional and the |delta| characteristic; trades without one (trade_iv ≤ 0 or delta 0) are dropped (design 3.6). Outputs
go under `data/robustness/vendor_delta/` and `results/robustness/vendor_delta/`; the baseline files are untouched.
`python compare_variant.py` writes the side-by-side comparison `results/robustness/vendor_delta/comparison.csv`.
Rerun 2026-11-02 under the corrected rules (deep ITM by the vendor delta as the fifth bucket; pre-event RI dates).
`build_full_panel.py` rebuilds its year files whenever the cell settings in `full/spec.json` differ.

## Settings

### Inputs and cells (chosen 2026-10-29)

- **Common volatility sigma_bar_{p,t}:** OptionMetrics Volatility Surface (WRDS vsurfd), 30 days, the average of the
  delta 50 call and delta −50 put IVs, same day (An, Ang, Bali and Cakici 2014, JF).
  - Files: `data/optionmetrics/vsurfd_30d_SPY_QQQ_IWM_2016_2023.csv` and `vsurfd_30d_SPY_QQQ_IWM_2020_2025.csv`
    (2020-01-02 to 2025-08-29, downloaded 2026-11-02; identical to the first file over 2020–2023); secids 109820 SPY,
    107899 QQQ, 106445 IWM. `build_cells.OM_FILES` reads every `vsurfd_30d_SPY_QQQ_IWM_*.csv` in the folder.
  - Converted to the trading-time clock: iv·sqrt((30/365) / tau_trading(close of t → close of the last session ≤ t+30)).
  - Robustness: the trade-based version (`SIGMA_SOURCE = "trades"`), run for the three paper events only, in
    `results/robustness/sigbar_trades/`.
- **Delta used for volume:** the same common-volatility |delta|, not the vendor delta (vendor delta: robustness above).
- **tau:** trading hours from the trade to the close of the last session on or before the expiration (09:30–16:00,
  early close 13:00; nights, weekends and holidays count zero).
  - Forward |delta| = N(±d1), with F = S·exp((r − q)·tau_cal).
  - r = OIS (`Raw Data/sofr.csv`), q = trailing dividend yield (`Raw Data/Dividends.xlsx`).
- **Cells:** ETF × day × effective expiration weekday × DTE bucket {0, 1, 2–6, 7–30, 31+} × delta bucket at the common
  volatility × call/put, new vs existing.
  - Deep ITM (|delta| > 0.875) is the fifth delta bucket (design 2.2): in delta notional, contracts and trades (2.3);
    not in premium (`premium`; shell Table 2 notes; `premium_all` keeps it) and not in the characteristics (3.6). So
    Deep ITM cells have no x: they are in Tables 1–2, λ and the control volume of Counterfactual 1, but not in
    Sigma_p, the distances, eq. 8, Tables 3 and 5 or the structural model. (Corrected 2026-11-02; until then deep-ITM
    trades were dropped from everything, see `no_ditm` above.)
  - V = delta notional over 09:30–16:15.
  - Characteristics x = delta-notional-weighted means of (log tau, |delta|) over 09:55–15:45, winsorized 0.25/99.75%
    within ETF × DTE bucket × day, deep-ITM trades left out.
- **Sigma_p:** per event × ETF × call/put, from existing pre-event cell-days weighted by V. Correlations are shrunk
  toward the identity (weighted Schäfer–Strimmer); fixed after the event.

### The 2021 events (user's choices 2026-10-31)

- **Windows:** the paper's convention (sessions with a raw file).
  - Day 0 is the second listing day (QQQ 2021-04-27, IWM 2021-10-08).
  - The pre-event window ends before the first listing (2021-04-23, 2021-10-05).
  - The days in between are transition days, as in 2022.
- **Controls:** per shell Table 1 (QQQ: SPY + IWM; IWM: SPY + QQQ).
  - Some controls already list the added weekdays. Their calendar is constant in the window, so they stay in
    Counterfactual 1.
  - Check: QQQ's Mon/Wed share is 32–39% across IWM's window; its DTE mix is stable pre vs post.
- **Eq. 18 / Table 3:** a control that already lists the new weekdays (2021: SPY for QQQ; SPY and QQQ for IWM) has its
  own Mon/Wed expirations play the new contracts' part, pre and post alike.
  - These expirations define its similarity calendar and N; their cells are dropped from its existing cells.
  - Implemented in `phantom.py`, flag "own".
- **Out of sample (design 8.3):** estimate on 2016, 2018 and both 2021 events, predict 2022.

### Step 3: counterfactual and λ

- **Counterfactual 1 (eq. 14):** V_cf = weekday-matched pre-event mean of the cell × the control families' delta notional
  in the same DTE bucket relative to its weekday-matched pre-event mean. Two fallbacks:
  - the control's total volume when the bucket is not listed for it (`cf_fallback`);
  - the cell's all-day pre mean when it is listed on a weekday never seen pre-event (`wd_fallback`; the Good Friday 2018
    and Christmas 2021 weeks).
- **Outcome:** y = (V − V_cf) / Vbar_pre_j.
- **2022 cut:** Mon/Wed 7–30 and 31+ DTE cells are excluded.
- **λ (eq. 13):** −Σ(V − V_cf) / Σ V_new over weeks 1–12 (event time 1–59), cell by cell.
  - Units: delta notional, contracts, premium.
  - Splits: products, blocks, new weekday (structural attribution).
- **Bootstrap:** five-session blocks resampled separately pre and post; 999 draws for λ, 499 for the structural model.
- **RI:**
  - Eligible dates: 1,129 sessions in 2016–2023 whose ±60-session window has no new-weekday listing (any of the three
    ETFs), no 2022-11-10 cut, and is not in the 2020-07-27..08-18 OptionMetrics gap.
  - Pre-event dates only (design 8.4; user's choice 2026-11-02): each event uses the eligible dates whose whole window
    lies before its new weekday was listed (2022: the cut); 2018 also after 2016-08-30 (same calendar). The rule of
    `placebo_beta.py` (`placebo.FIRST_LISTED`, `LO_BOUND`): 45 (2016), 244 (2018), 819 (each 2021 event), 972 (2022);
    IWM 2024 (Table 2 only): all 1,129. The 2016 windows start 2016-04-04..06-06 and overlap almost entirely.
    `results/placebo_lambda_events.parquet` holds each event's rows.
  - Placebo λ = −(existing change, % of B) / (the event's new volume, % of B).
  - Density slope: random draws of one pre-event placebo date per event.

### Step 6: structural model

- **Model:** −dV_jt = Σ_k λ_e V_k S_jk / Σ_j' S_j'k, with S = exp(−θ d² + γ N); NLS weighted by Vbar_pre on the eq. 8
  sample.
- **Choice set:** all listed existing cells of the ETF, type and day, except Deep ITM cells (no characteristics).
- **Versions:** L = logistic(a + b n_e); F = free λ_e; I = identity Λ.
- **Bootstrap:** also re-estimates Sigma_p and the distances.

### Step 5: reduced form

- **Table 5, cols. 1–3:** y on Sbar (θ_hat from L) and N | ETF × day FE.
- **Table 5, col. 5:** adds cell-type FE (treated only).
- **Table 5, col. 4 (eq. 18):** stacked treated + control panel, log V.
  - FE: cell + ETF × day + DTE × day, the DTE × day effects common to an event's treated and control ETFs. So
    Post×Treat×N compares treated cells losing nearest-expiry status with control cells in the same DTE bucket and day.
    The design's family (= ETF) × DTE × day FE would leave N constant within groups in 2016, 2021 IWM and 2022, one
    varying group in 2021 QQQ and 12 in 2018 (96 N=1 cell-days): identified by 2018 alone (checked 2026-11-02, also by
    the user; stated in the report's Step 5 and Table 3 text).
  - Calendar-position profile; θ grid and distance shares as robustness.
  - Absorbed N terms are dropped (2016, 2021 IWM, 2022).
- **Table 3:** the same panel with N only.
- **RI for β and γ:** eq. 17 with the calendar similarity, at placebo windows before each event's new weekday.
  Placebo dates: 45 (2016), 244 (2018), 819 (2021 QQQ), 819 (2021 IWM), 972 (2022).

## Results snapshot (2026-11-02: deep ITM in volume, pre-event RI dates)

### Table 2 (delta notional, weeks 1–12)

| | 2016 | 2018 | 2021 QQQ | 2021 IWM | 2022 | 2024 IWM |
|---|---|---|---|---|---|---|
| New share (%) | 16.2 | 10.2 | 27.4 | 21.5 | 24.8 | 14.5 |
| Existing, control-adjusted (%) | −20.2 | −5.9 | −16.3 | −9.1 | +2.8 | +3.7 |
| Placebo s.d. of the adjusted change (%) | 2.9 | 15.7 | 23.5 | 18.3 | 18.4 | 27.1 |

The 2016 placebo s.d. comes from 45 nearly identical windows; IWM 2024's from all 1,129 dates.

### λ (Table 4)

| | 2016 | 2018 | 2021 QQQ | 2021 IWM | 2022 |
|---|---|---|---|---|---|
| λ, delta notional | 1.08 | 0.56 | 0.51 | 0.32 | −0.10 |
| RI p (pre-event dates) | < 1/45 | 0.61 | 0.41 | 0.62 | 0.88 |

- 2016 against all 1,129 dates would be p 0.16.
- Panel B: the logistic does not fit. With identity Λ, b = −0.33 (bootstrap s.e. 0.17; RI p 0.40).

### Table 3: Post×Treat×N (unchanged by the deep-ITM correction)

- Pooled: −0.18 (0.05); ATM −0.21, OTM −0.11, ITM −0.28.
- By event: −0.19 / −0.04 / −0.09 / −0.31 / −0.36.

### Table 5 (pooled)

| Column | Estimate |
|---|---|
| (1) N | −0.22 (0.05) |
| (2) Sbar | −1.04 (0.60) |
| (3) Sbar | −1.18 (0.62) |
| (4) Post×Treat×S | −1.50 (0.73) |
| (4) Post×Treat×N | −0.18 (0.05) |
| (5) Sbar, cell-type FE | +0.65 (0.61) |

- Share of weight within 0.5 M-units: −0.67 (0.22).
- RI for the calendar Sbar (col. 3): pooled p < 0.001; 2016 p < 1/45; 2022 0.006; QQQ 0.17; IWM 0.15; 2018 0.67.
- RI for N, pooled: p 0.001.
- Col. 5: cell-type FE leave ~73% of Sbar's variation (net of ETF × day FE); within types the relation is not
  negative. Design eq. 17 itself has the cell-type FE; the shell puts them in col. 5 only.

### Table 6 (col. 1, L)

- θ = 44.2 (bootstrap 5–95%: 18–77): 97% of each new cell's diverted volume goes to its nearest existing cell.
- γ is not identified (−18 to 12).
- λ = 0.20 (n_e = 1), 0.23 (2018), 0.28 (2022); b = 0.22 (s.e. 0.49).
- Free λ = 0.40 / 0.12 / 0.22 / 0.04 / 0.28.
- Weighted RMSE 0.647 vs 0.662 for no loss.
- Out of sample (2016, 2018, 2021 → 2022): λ_2022 = 0.07; RMSE 0.677 vs 0.684; rank correlation 0.57 (0.38–0.64).

### Eq. 8

- (i) has no decay.
- (ii) Z1: 2022 −0.31, 2018 −0.23, pooled −0.29; 2016 and both 2021 events are positive and imprecise.

### Deep ITM: with vs without (`results/robustness/no_ditm/comparison.csv`, report Tables S11–S12)

- Deep ITM is 5.0–10.0% of the treated ETFs' existing pre-event delta notional (0DTE cells 8–28%); spikes of 25–50%
  on the day before SPY ex-dividend dates (calls) and monthly expiries; 2022 spikes are deep-ITM puts.
- New shares move ≤ 0.7 pp; λ moves most in 2018 (0.22 → 0.56, s.e. 0.44), ≤ 0.09 elsewhere.
- Distance-based estimates move little (Table 3 identical; eq. 8 Z1 −0.25 → −0.29; θ 47.5 → 44.2); T5 col. 5 Sbar
  0.30 → 0.65.

## Checks

Independent re-computations (scratch code, not kept; * = redone 2026-11-02 after the deep-ITM correction):

- * Cells other than Deep ITM identical to the previous version's, and `SIM_VARIANT=no_ditm` reproduces the previous
  cells exactly (2016-09-12, 2021-11-03, 2022-12-01).
- * The full panel reproduces the 2016 event cells exactly (Deep ITM included); the placebo code at 2016-08-30 returns
  the event's −20.231.
- * λ from panel columns equals lambda_est for all five events; the λ engine equals panel.py's V_cf to 1e-15.
- * λ and β RI use the same pre-event dates (45/244/819/819/972).

- **Inputs:**
  - `event_windows.py` reproduces ListingDynamics' days and expirations files for 2016/2018/2022 exactly;
  - the full panel reproduces the 2016 event cells exactly.
- **λ and RI:** for all five events, λ recomputed from panel columns and RI p-values by hand match.
- **Engines:**
  - the λ engine equals panel.py's V_cf to 1e-15;
  - the structural inputs match the pipeline, and G matches a hand calculation;
  - the RI regression helper equals fixest;
  - the placebo calendar S reproduces phantom.py.
- **Eq. 18 panel:** already-treated controls carry no new-weekday cells.

## Report

`overleaf/` (main.tex, numbers.tex, tables/, figures/) is the full-baseline report.

- Tables 1–6 carry the shells' numbers; the other tables are S1–S17 (S11–S12: deep ITM with vs without).
- Everything is generated by `python make_baseline_report.py`.
- `baseline_report_overleaf.zip` is the same folder for upload to Overleaf.
- `baseline_report.pdf`: 45 pages, no LaTeX warnings or overfull boxes (2026-11-02).
- Compiled with TinyTeX (`~/Library/TinyTeX/bin/universal-darwin/pdflatex`, three passes, in a scratch copy of
  `overleaf/`).
