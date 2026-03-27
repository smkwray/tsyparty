"""Tests for the 3 new inference validation checks in crosscheck.py."""

import pandas as pd

from tsyparty.validate.crosscheck import (
    compare_inference_to_fwtw,
    compare_inference_to_auction,
    compare_foreign_inference_to_tic,
)


def _make_flows():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31"] * 4 + ["2024-06-30"] * 4),
        "seller": ["dealers", "insurers"] * 4,
        "buyer": ["banks", "foreigners_official"] * 4,
        "amount": [50.0, 30.0, 40.0, 20.0, 60.0, 35.0, 45.0, 25.0],
        "method": ["dense"] * 8,
        "converged": [True] * 8,
    })


def test_compare_inference_to_fwtw():
    flows = _make_flows()
    fwtw = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30"] * 2),
        "sector": ["banks", "banks", "foreigners_official", "foreigners_official"],
        "holdings": [1000.0, 1055.0, 500.0, 525.0],
    })
    result = compare_inference_to_fwtw(flows, fwtw)
    assert not result.empty
    assert "diff_pct" in result.columns


def test_compare_inference_to_fwtw_empty():
    result = compare_inference_to_fwtw(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_compare_inference_to_auction():
    flows = _make_flows()
    auction = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "amount": [200.0, 250.0],
    })
    result = compare_inference_to_auction(flows, auction)
    assert not result.empty
    assert "ratio" in result.columns
    # Inferred buying should be some ratio of auction total
    assert all(result["ratio"] > 0)


def test_compare_inference_to_auction_empty():
    result = compare_inference_to_auction(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_compare_foreign_inference_to_tic():
    flows = _make_flows()
    tic = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "tic_foreign_treasury": [500.0, 530.0],
    })
    result = compare_foreign_inference_to_tic(flows, tic)
    assert not result.empty
    assert "diff_pct" in result.columns


def test_compare_foreign_inference_to_tic_empty():
    result = compare_foreign_inference_to_tic(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_compare_foreign_inference_to_tic_with_private():
    """Foreign inference validation should sum both official and private flows."""
    flows = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31"] * 4 + ["2024-06-30"] * 4),
        "seller": ["dealers", "insurers", "dealers", "insurers"] * 2,
        "buyer": ["foreigners_official", "foreigners_official", "foreigners_private", "foreigners_private"] * 2,
        "amount": [30.0, 20.0, 10.0, 5.0, 35.0, 25.0, 12.0, 8.0],
        "method": ["dense"] * 8,
        "converged": [True] * 8,
    })
    tic = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "tic_foreign_treasury": [500.0, 570.0],
    })
    result = compare_foreign_inference_to_tic(flows, tic)
    assert not result.empty
    # Net foreign buying for Q2 = (35+25+12+8) - 0 sellers = 80
    # But no foreign sellers in this data, so net = total buying
    row = result[result["date"] == pd.Timestamp("2024-06-30")]
    assert len(row) == 1
    # inferred_foreign_net should include both official + private buying
    assert row.iloc[0]["inferred_foreign_net"] == 80.0
