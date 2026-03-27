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


def compare_inference_to_fwtw(
    flows: pd.DataFrame,
    fwtw: pd.DataFrame,
) -> pd.DataFrame:
    """Compare inferred seller totals per quarter against FWTW holding levels.

    FWTW provides issuer-holder levels. We compare the total inferred flow
    per quarter against the FWTW-implied holding change for available sectors.

    Returns a DataFrame with columns: date, sector, inferred_total, fwtw_delta, diff, diff_pct
    """
    if flows.empty or fwtw.empty:
        return pd.DataFrame(columns=["date", "sector", "inferred_total", "fwtw_delta", "diff", "diff_pct"])

    # FWTW has: date, sector, holdings. Compute FWTW holding changes.
    fwtw_sorted = fwtw.sort_values(["sector", "date"])
    fwtw_sorted["fwtw_delta"] = fwtw_sorted.groupby("sector")["holdings"].diff()
    fwtw_changes = fwtw_sorted.dropna(subset=["fwtw_delta"])

    # Inference flows: sum inferred amounts per (date, buyer) for dense method
    dense = flows[flows["method"] == "dense"]
    if dense.empty:
        dense = flows
    inferred_by_buyer = dense.groupby(["date", "buyer"], as_index=False)["amount"].sum()
    inferred_by_buyer = inferred_by_buyer.rename(columns={"buyer": "sector", "amount": "inferred_total"})

    merged = inferred_by_buyer.merge(
        fwtw_changes[["date", "sector", "fwtw_delta"]],
        on=["date", "sector"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["date", "sector", "inferred_total", "fwtw_delta", "diff", "diff_pct"])

    merged["diff"] = merged["inferred_total"] - merged["fwtw_delta"]
    merged["diff_pct"] = merged["diff"] / merged["fwtw_delta"].replace(0, float("nan")) * 100
    return merged[["date", "sector", "inferred_total", "fwtw_delta", "diff", "diff_pct"]]


def compare_inference_to_auction(
    flows: pd.DataFrame,
    auction_allotments: pd.DataFrame,
) -> pd.DataFrame:
    """Compare total inferred quarterly buying against auction allotment totals.

    If total inferred net buying in a quarter greatly exceeds primary-market
    issuance, it signals that secondary-market activity dominates.

    Returns a DataFrame with columns: date, total_inferred_buying, auction_total, ratio
    """
    if flows.empty or auction_allotments.empty:
        return pd.DataFrame(columns=["date", "total_inferred_buying", "auction_total", "ratio"])

    dense = flows[flows["method"] == "dense"]
    if dense.empty:
        dense = flows
    quarterly_buying = dense.groupby("date", as_index=False)["amount"].sum()
    quarterly_buying = quarterly_buying.rename(columns={"amount": "total_inferred_buying"})

    # Auction data may use 'amount' or 'allotment_amount'
    amount_col = "amount" if "amount" in auction_allotments.columns else "allotment_amount"
    if amount_col not in auction_allotments.columns:
        return pd.DataFrame(columns=["date", "total_inferred_buying", "auction_total", "ratio"])

    # Quarterly-aggregate auction allotments
    auction_copy = auction_allotments.copy()
    auction_copy["date"] = pd.to_datetime(auction_copy["date"])
    auction_copy["quarter"] = auction_copy["date"].dt.to_period("Q").dt.to_timestamp("Q")
    auction_q = auction_copy.groupby("quarter", as_index=False)[amount_col].sum()
    auction_q = auction_q.rename(columns={"quarter": "date", amount_col: "auction_total"})

    merged = quarterly_buying.merge(auction_q, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["date", "total_inferred_buying", "auction_total", "ratio"])

    merged["ratio"] = merged["total_inferred_buying"] / merged["auction_total"].replace(0, float("nan"))
    return merged


def compare_foreign_inference_to_tic(
    flows: pd.DataFrame,
    tic_foreign: pd.DataFrame,
) -> pd.DataFrame:
    """Compare inferred flows to/from foreigners against TIC foreign holding changes.

    Returns a DataFrame with columns: date, inferred_foreign_net, tic_delta, diff, diff_pct
    """
    if flows.empty or tic_foreign.empty:
        return pd.DataFrame(columns=["date", "inferred_foreign_net", "tic_delta", "diff", "diff_pct"])

    dense = flows[flows["method"] == "dense"]
    if dense.empty:
        dense = flows

    # Net foreign buying = sum of flows where foreigners_official is buyer
    # minus sum where foreigners_official is seller
    foreign_buying = dense[dense["buyer"] == "foreigners_official"].groupby("date")["amount"].sum()
    foreign_selling = dense[dense["seller"] == "foreigners_official"].groupby("date")["amount"].sum()
    foreign_net = (foreign_buying - foreign_selling).reset_index()
    foreign_net.columns = ["date", "inferred_foreign_net"]

    # TIC holding changes
    tic_sorted = tic_foreign.sort_values("date")
    tic_sorted["tic_delta"] = tic_sorted["tic_foreign_treasury"].diff()
    tic_changes = tic_sorted.dropna(subset=["tic_delta"])

    merged = foreign_net.merge(tic_changes[["date", "tic_delta"]], on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["date", "inferred_foreign_net", "tic_delta", "diff", "diff_pct"])

    merged["diff"] = merged["inferred_foreign_net"] - merged["tic_delta"]
    merged["diff_pct"] = merged["diff"] / merged["tic_delta"].replace(0, float("nan")) * 100
    return merged
