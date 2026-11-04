# Contract Similarity and Option-Demand Diversion

Standalone production pipeline for the ETF expiration-listing baseline. All calculations live in the ten scripts below. The repository contains code, pinned dependency definitions, and the LaTeX table-shell source template. Raw data, estimated results, previous implementations, audit files, development tests, caches, and generated reports are not included.

## Files

| File | Purpose |
|---|---|
| `config.py` | Event definitions, panel parameters, inference settings |
| `Build_Panel.py` | Raw trade loading, event windows, expiration classification, cell-day volumes and characteristics |
| `Build_Event_Panel.py` | Listed-cell grids, weekday baselines and control-adjusted counterfactuals |
| `Estimate_Lambda.py` | Event-level reallocation ratios and block bootstrap |
| `Measure_Similarity.py` | Covariance shrinkage, Mahalanobis distances, similarity exposures and calendar profiles |
| `Reduced_Form.R` | Distance-bin and reduced-form regressions |
| `Structural_Model.py` | Structural parameters, event-level density fits, holdout predictions and bootstrap |
| `Inference.py` | Full-sample raw-data panels and date-placebo randomization inference |
| `Make_Results.py` | Populate table shells and compile the PDF |
| `Run_All.py` | All input/output paths, execution order, subprocesses and cache verification |

`setup_R.R` installs/checks pinned R packages. `assets/templates/main_shell_template.tex` is a required source template, not a generated result.

## Environment

Reference environment: CPython **3.12.2**, R **4.5.1**, and pdfLaTeX from **TeX Live 2026**. Python runtime packages and their transitive dependencies are pinned in `requirements.txt`; R package versions are pinned in `setup_R.R`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript setup_R.R
Rscript setup_R.R --check
```

Use an R installation with a writable package library, or configure `R_LIBS_USER` before starting R. Installing pinned R packages from source requires suitable C/C++/Fortran compilers. Existing packages with the correct versions are reused.

`Rscript` and `pdflatex` must be on `PATH`. The template requires `geometry`, `amsmath`, `array`, `booktabs`, `tabularx`, `threeparttable`, `float`, `enumitem`, `xcolor`, and `hyperref`; these are available in a full TeX Live installation. Python-only installation does not supply R or LaTeX.

## Supply the original inputs

The input datasets are external to this repository. In the quoted path section at the top of `Run_All.py`, paste each file's actual address:

```python
trade_archive_path = "/your/data/OptionsTradeData.zip"
trade_csv_directory_path = ""
optionmetrics_2016_2023_path = "/your/data/optionmetrics_early.csv"
optionmetrics_2020_2025_path = "/your/data/optionmetrics_late.csv"
rates_path = "/your/data/rates.csv"
dividends_path = "/your/data/dividends.xlsx"
table_shell_template_path = "assets/templates/main_shell_template.tex"
output_directory_path = "runs/baseline"
configuration_path = "config.py"
```

Each input can be in a different directory and have a different file name. Paths may be absolute or relative to `Run_All.py`; they are independent of the shell's current working directory. Keep the data columns, workbook sheet name, and archive member naming conventions described below. An optional daily CSV directory can supply event-window files, but a complete trade archive is still required for the full inference stage.

| Input | Required format |
|---|---|
| Trade archive | Outer ZIP with `OptionsTradeData/UnderlyingOptionsTradesCalcs_YYYY-MM-DD.zip`; each inner ZIP contains `UnderlyingOptionsTradesCalcs_YYYY-MM-DD.csv`. Preserve the original Cboe columns, including those listed in `Build_Panel.COLS`. The full run uses event windows through IWM 2024 and 2016-01-06 through 2023-12-29 for RI. |
| OptionMetrics CSVs | `secid`, `date`, `days`, `delta`, `cp_flag`, `impl_volatility`; include 30-day, absolute delta-50 call and put IVs for SPY/QQQ/IWM. The two inputs jointly provide the reference coverage. Full-panel sessions without the required IV are logged and skipped; event-window inputs are checked strictly. The files may overlap; earlier-file records have priority. One input can be empty if the other covers the full sample. |
| Rate CSV | `Date` in `%m/%d/%y` format and the OIS tenor columns listed in `config.OIS_TENORS`; rates in percentage points. The original file happens to be called `sofr.csv`. |
| Dividend workbook | Sheet `Dividends_TTM`; columns `date`, `Dividend_TTM_SPY`, `Dividend_TTM_QQQ`, `Dividend_TTM_IWM`; yields in percentage points. |
| Table template | The bundled `assets/templates/main_shell_template.tex`, or an identical external copy. |

The reference input identities are recorded below so another researcher can confirm they have the same files. File names themselves are not mandatory.

| Original input | SHA-256 |
|---|---|
| `vsurfd_30d_SPY_QQQ_IWM_2016_2023.csv` | `ef603a841ea5e77f9d7b80ad102c2cddc4db5aef4ef9478b628cf11a059c3394` |
| `vsurfd_30d_SPY_QQQ_IWM_2020_2025.csv` | `40614a813502eb78487907b60242fe5c45c45568600b0c362faf20dd8e3e075f` |
| `sofr.csv` | `a084000f942369f41163e047076cdf34da1cf48cbc67d6b5f2cf4d61264917e1` |
| `Dividends.xlsx` | `700103beb2dfb247912051adbc335f55972399d0bcbabec9aefb2d5975338f18` |
| Table template | `a7138b1c0fde2703685e2b659af1146fd6b6154020d915bb90dbbbdab883c705` |

The reference trade ZIP is 36,706,026,688 bytes and contains 3,348 non-directory members. Its inventory checksum is `ea8f78f8455f5867af49f233b134dc88545e94215d901b0363de2cbe61796d1e`, computed as SHA-256 of UTF-8 `json.dumps(sorted((i.filename, i.file_size, i.CRC) for i in zipfile.ZipFile(path).infolist() if not i.is_dir()), separators=(",", ":"))`. This identifies the archive inventory, not a SHA-256 of all archive bytes.

## Run from raw data

After setting paths:

```bash
python Run_All.py
```

Every path can also be supplied on the command line:

```bash
python Run_All.py \
  --archive /your/data/OptionsTradeData.zip \
  --optionmetrics-early /your/data/optionmetrics_early.csv \
  --optionmetrics-late /your/data/optionmetrics_late.csv \
  --rates-file /your/data/rates.csv \
  --dividends-file /your/data/dividends.xlsx \
  --output runs/baseline
```

Execution order:

```text
Build_Panel -> Build_Event_Panel -> Estimate_Lambda -> Measure_Similarity
            -> Structural_Model -> Reduced_Form -> Inference -> Make_Results
```

The structural estimate of theta is used to construct the main similarity exposures, so structural estimation runs before the main reduced-form regressions.

The default inference uses **999 lambda bootstrap draws**, **499 structural bootstrap draws**, the complete eligible date-placebo sets, and **999 pooled coefficient-RI draws**. The seeds and event definitions are in the source/configuration; preserve them when reproducing the baseline. Draw counts are not reduced for a normal run.

Successful outputs are generated under the selected output directory:

```text
data/                   Generated event and full-sample panels
results/                Numerical estimates and bootstrap/RI draws
table_shells/main.tex    Filled table-shell source
table_shells/main.pdf    Compiled table shells
manifest.json           Stage status and hashes of generated outputs
settings.json           Settings, event definitions and resolved input paths
runtime_context.json    Paths supplied to the standalone stages
run.log                 Execution log
```

These generated files stay local and are ignored by Git. A clean run requires no historical panels, prior estimates, parent-project modules, or legacy scripts. For an existing run, stages reuse outputs only when the run fingerprint and recorded output hashes match. After code or input changes, use a new output directory or `--force`.

Useful commands:

```bash
python Run_All.py --show-paths
python Run_All.py --stage Structural_Model --archive /your/data/OptionsTradeData.zip
python Run_All.py --results-only
```

`--stage` runs the selected stage and its prerequisites. Running a stage file directly also invokes its prerequisites. `--results-only` requires a completed numerical run. A fresh output directory is the simplest way to verify complete reproduction rather than cache reuse.

## Release scope

This snapshot contains the approved **F02 effective-weekday calendar/profile correction**, including coefficient RI. Actual expiration dates remain the inputs to the trading-time and delta calculations.

It reproduces the current baseline implementation. The remaining methodological audit findings have not been silently changed for publication. In particular:

- Structural OOS initialization still uses a full-event free-lambda warm start (F04).
- The auxiliary new-weekday lambda decomposition uses a different sample from the overall event ratio (F03).
- The trading calendar currently ends on 2025-12-31, affecting expirations beyond that boundary (F09).
- IWM 2024 has descriptive/event-lambda outputs and lambda RI, but no structural OOS fit; its existing lambda-RI percentile is not yet filled into Table 9 (F08/F55).
- G1 Mahalanobis is the implemented structural baseline. Other similarity definitions and parts of the manual remain unimplemented. Tables 7/8/10 and other unestimated shell cells retain placeholders; inherited black entries in Table 9 Panels A/B are identified by the template notes.

The production code does not import reference estimates or execute old-versus-new comparison reports. Reproduction requires the same original inputs and compatible pinned numerical environment; floating-point/optimizer differences can affect unstable bootstrap draws across platforms.


## Clean-run verification

On 2026-11-04, the production tree was executed from the original raw inputs in a newly created Python virtual environment installed only from `requirements.txt`, using the pinned R packages and a fresh output directory. No precomputed panels, cached estimates, or legacy code were supplied. All eight stages completed, all 98 recorded output hashes verified, and the table shells compiled to a 13-page PDF. The run used 999 lambda bootstrap draws, 499 draws for each of the five structural versions, all eligible event-level date placebos, and 999 pooled coefficient-RI draws.

Main displayed point estimates reproduced at their reported precision. Some long-tail bootstrap standard errors differed slightly between numerical environments; fixed seeds do not remove sensitivity to floating-point reductions and weakly identified nonlinear fits. Reproduction of the complete workflow is verified; byte-identical bootstrap estimates across executions/platforms are not asserted.
