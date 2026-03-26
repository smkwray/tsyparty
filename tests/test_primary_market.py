"""Tests for primary-market allocation module."""

import pandas as pd

from tsyparty.baseline.primary_market import build_primary_allocation, primary_allocation_summary


def _sample_allotments() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-01-22", "2024-02-15", "2024-04-10"]),
        "security_type": ["bills"] * 4,
        "term": ["4-Week"] * 4,
        "buyer_class": ["dealers", "dealers", "foreign_official", "dealers"],
        "allotment_amount": [100.0, 150.0, 50.0, 200.0],
    })


def test_build_primary_allocation_basic():
    allotments = _sample_allotments()
    result = build_primary_allocation(bills_allotments=allotments)

    assert not result.empty
    assert "all_instruments" in result["instrument"].values
    assert "date" in result.columns
    assert "share_of_instrument" in result.columns


def test_primary_allocation_shares_sum_to_one():
    allotments = _sample_allotments()
    result = build_primary_allocation(bills_allotments=allotments)

    for (date, instr), grp in result.groupby(["date", "instrument"]):
        total = grp["share_of_instrument"].sum()
        assert abs(total - 1.0) < 0.01, f"Shares for {date}/{instr} sum to {total}"


def test_primary_allocation_summary():
    allotments = _sample_allotments()
    allocation = build_primary_allocation(bills_allotments=allotments)
    summary = primary_allocation_summary(allocation)

    assert not summary.empty
    assert "dealers" in summary["buyer_class"].values
