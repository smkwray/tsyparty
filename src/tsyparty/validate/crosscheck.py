"""Cross-validate harmonized panel against EFA and TIC sources."""

from __future__ import annotations

import pandas as pd


def crosscheck_sector(
    panel_series: pd.DataFrame,
    external_series: pd.DataFrame,
    panel_col: str = "holdings",
    external_col: str = "external",
) -> pd.DataFrame:
    """Compare a panel sector's holdings against an external source.

    Parameters
    ----------
    panel_series : DataFrame with columns: date, {panel_col}
    external_series : DataFrame with columns: date, {external_col}

    Returns
    -------
    DataFrame with columns: date, panel, external, diff, diff_pct
    """
    merged = panel_series.merge(external_series, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["date", "panel", "external", "diff", "diff_pct"])

    merged = merged.rename(columns={panel_col: "panel", external_col: "external"})
    merged["diff"] = merged["panel"] - merged["external"]
    merged["diff_pct"] = merged["diff"] / merged["external"].replace(0, float("nan")) * 100
    return merged[["date", "panel", "external", "diff", "diff_pct"]]


def run_crosschecks(
    panel: pd.DataFrame,
    efa_bank: pd.DataFrame | None = None,
    efa_mmf: pd.DataFrame | None = None,
    tic_foreign: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all available cross-validation checks.

    Returns a summary DataFrame with columns: sector, mean_diff_pct, max_abs_diff_pct, quarters.
    """
    results = []

    if efa_bank is not None and not efa_bank.empty:
        panel_banks = panel[panel["sector"] == "banks"].groupby("date", as_index=False)["holdings"].sum()
        check = crosscheck_sector(panel_banks, efa_bank, "holdings", "bank_treasury_holdings")
        if not check.empty:
            results.append(("banks", check["diff_pct"].mean(), check["diff_pct"].abs().max(), len(check)))

    if efa_mmf is not None and not efa_mmf.empty:
        panel_mmf = panel[panel["sector"] == "money_market_funds"].groupby("date", as_index=False)["holdings"].sum()
        check = crosscheck_sector(panel_mmf, efa_mmf, "holdings", "mmf_treasury_holdings")
        if not check.empty:
            results.append(("money_market_funds", check["diff_pct"].mean(), check["diff_pct"].abs().max(), len(check)))

    if tic_foreign is not None and not tic_foreign.empty:
        panel_foreign = panel[panel["sector"] == "foreigners_official"].groupby("date", as_index=False)["holdings"].sum()
        check = crosscheck_sector(panel_foreign, tic_foreign, "holdings", "tic_foreign_treasury")
        if not check.empty:
            results.append(("foreigners_official", check["diff_pct"].mean(), check["diff_pct"].abs().max(), len(check)))

    if not results:
        return pd.DataFrame(columns=["sector", "mean_diff_pct", "max_abs_diff_pct", "quarters"])

    return pd.DataFrame(results, columns=["sector", "mean_diff_pct", "max_abs_diff_pct", "quarters"])
