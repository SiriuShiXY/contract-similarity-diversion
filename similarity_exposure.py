"""
Similarity of each existing cell-day to the new cells trading that day (similarity_diversion_design.pdf
eq. 16), the regressor of the reduced form eq. (17) / table_shells Table 5 columns (1)-(3) and (5).

    S_jk  = exp(-theta d_M(j,k)^2)                       (Table 5 notes: theta from Table 6, column 1)
    Sbar_jt = sum_k V_kt S_jk / sum_k V_kt              (volume-weighted, baseline)
    Smax_jt = max_k S_jk                                 (eq. 16 alternative, robustness)
k: new cells of the same product, option type and day with characteristics (mahalanobis.py pairs);
theta_hat = the logistic-lambda structural estimate (structural.py, version L).  Robustness: theta in
{1, 2, 4}, the grid of phantom.py.

Usage: python similarity_exposure.py          (after structural.py)
Output: data/reg_panel_rf.csv (reg_panel_all.csv + Sbar, Smax, Sbar_t1, Sbar_t2, Sbar_t4)
"""
from __future__ import annotations

import polars as pl

from build_cells import EVENTS, OUT_DIR
from mahalanobis import J, RES_DIR

GRID = [1, 2, 4]


def theta_hat() -> float:
    e = pl.read_csv(RES_DIR / "structural_estimates.csv")
    return float(e.filter(pl.col("version") == "L")["theta"][0])


def exposure(ev: str, theta: float) -> pl.DataFrame:
    p = pl.read_parquet(OUT_DIR / ev / "pairs.parquet")
    d2 = pl.col("d") ** 2
    vk = pl.col("V_k")
    return p.group_by(J).agg(
        [((vk * (-theta * d2).exp()).sum() / vk.sum()).alias("Sbar"), (-theta * d2).exp().max().alias("Smax")]
        + [((vk * (-t * d2).exp()).sum() / vk.sum()).alias(f"Sbar_t{t}") for t in GRID]
    ).with_columns(pl.lit(ev).alias("event"))


def main() -> None:
    th = theta_hat()
    reg = pl.read_csv(OUT_DIR / "reg_panel_all.csv", schema_overrides={"event": pl.Utf8}, try_parse_dates=True)
    ex = pl.concat([exposure(ev, th) for ev in EVENTS])
    out = reg.join(ex, on=["event"] + J, how="left").with_columns(pl.lit(th).alias("theta_hat"))
    r = out.filter(pl.col("in_reg"))
    assert r["Sbar"].null_count() == 0
    out.write_csv(OUT_DIR / "reg_panel_rf.csv")
    w = pl.col("Vpre")
    print(f"theta_hat {th:.3f}; regression cell-days {r.height:,}")
    print(r.group_by("event").agg(pl.len(), ((pl.col("Sbar") * w).sum() / w.sum()).alias("Sbar_wmean"),
                                  pl.col("Sbar").quantile(0.9).alias("Sbar_p90"), (pl.col("Sbar") > 0.01).mean().alias("share>0.01"),
                                  ((pl.col("Sbar_t1") * w).sum() / w.sum()).alias("Sbar_t1_wmean")).sort("event"))


if __name__ == "__main__":
    main()
