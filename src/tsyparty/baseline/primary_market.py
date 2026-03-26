"""Primary-market allocation: Treasury → sector flows from auction allotments.

This module models how new issuance reaches sectors through the primary market.
It is kept separate from secondary-market reallocation per project constraints.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_primary_allocation(
    bills_allotments: pd.DataFrame | None = None,
    coupon_allotments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine bill and coupon allotments into a unified primary-market allocation table.

    Returns a DataFrame with columns:
        date, instrument, buyer_class, allotment_amount, share_of_instrument
    aggregated to quarterly frequency.
    """
    frames = []

    for allotments, instrument in [
        (bills_allotments, "bills"),
        (coupon_allotments, "nominal_coupons"),
    ]:
        if allotments is None or allotments.empty:
            continue
        df = allotments.copy()
        df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp("Q")
        quarterly = df.groupby(["quarter", "buyer_class"], as_index=False)["allotment_amount"].sum()
        totals = quarterly.groupby("quarter")["allotment_amount"].transform("sum")
        quarterly["share_of_instrument"] = quarterly["allotment_amount"] / totals.replace(0, float("nan"))
        quarterly["instrument"] = instrument
        quarterly = quarterly.rename(columns={"quarter": "date"})
        frames.append(quarterly)

    if not frames:
        return pd.DataFrame(columns=["date", "instrument", "buyer_class", "allotment_amount", "share_of_instrument"])

    combined = pd.concat(frames, ignore_index=True)

    # Add combined "all_instruments" aggregation
    all_instr = combined.groupby(["date", "buyer_class"], as_index=False)["allotment_amount"].sum()
    totals = all_instr.groupby("date")["allotment_amount"].transform("sum")
    all_instr["share_of_instrument"] = all_instr["allotment_amount"] / totals.replace(0, float("nan"))
    all_instr["instrument"] = "all_instruments"
    combined = pd.concat([combined, all_instr], ignore_index=True)

    return combined.sort_values(["date", "instrument", "buyer_class"]).reset_index(drop=True)


def primary_allocation_summary(allocation: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary-market allocation: average shares by buyer class across all quarters."""
    if allocation.empty:
        return pd.DataFrame()

    all_instr = allocation[allocation["instrument"] == "all_instruments"]
    if all_instr.empty:
        return pd.DataFrame()

    summary = all_instr.groupby("buyer_class", as_index=False).agg(
        mean_share=("share_of_instrument", "mean"),
        mean_amount=("allotment_amount", "mean"),
        quarters=("date", "nunique"),
    )
    return summary.sort_values("mean_share", ascending=False).reset_index(drop=True)
