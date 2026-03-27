"""Tests for baseline/flows.py — holding changes and buyer/seller margins."""

import pandas as pd
import pytest

from tsyparty.baseline.flows import holdings_changes_from_levels, buyer_seller_margins


def test_holdings_changes_basic():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"]),
        "sector": ["banks", "banks", "dealers", "dealers"],
        "instrument": ["treasury"] * 4,
        "holdings": [100.0, 120.0, 200.0, 180.0],
    })
    result = holdings_changes_from_levels(df, group_cols=["sector", "instrument"])
    banks = result[(result["sector"] == "banks") & (result["date"] == pd.Timestamp("2024-06-30"))]
    assert abs(banks["delta_holdings"].iloc[0] - 20.0) < 0.01


def test_holdings_changes_first_quarter_is_nan():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "sector": ["banks", "banks"],
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 120.0],
    })
    result = holdings_changes_from_levels(df, group_cols=["sector", "instrument"])
    first = result[result["date"] == pd.Timestamp("2024-03-31")]
    assert first["delta_holdings"].isna().all()


def test_buyer_seller_margins_basic():
    df = pd.DataFrame({
        "sector": ["banks", "dealers", "insurers"],
        "net_flow": [50.0, -30.0, -20.0],
    })
    buyers, sellers = buyer_seller_margins(df)
    assert "banks" in buyers.index
    assert buyers["banks"] == 50.0
    assert "dealers" in sellers.index
    assert sellers["dealers"] == 30.0  # sellers are positive


def test_buyer_seller_margins_requires_sector():
    df = pd.DataFrame({"name": ["x"], "net_flow": [10.0]})
    with pytest.raises(ValueError, match="sector"):
        buyer_seller_margins(df)


def test_buyer_seller_margins_all_buyers():
    df = pd.DataFrame({"sector": ["a", "b"], "net_flow": [10.0, 20.0]})
    buyers, sellers = buyer_seller_margins(df)
    assert len(buyers) == 2
    assert sellers.empty
