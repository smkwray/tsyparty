"""Tests for cross-validation checks."""

import pandas as pd

from tsyparty.validate.crosscheck import crosscheck_sector, run_crosschecks


def test_crosscheck_sector_basic():
    panel = pd.DataFrame({"date": [pd.Timestamp("2024-03-31")], "holdings": [1000.0]})
    external = pd.DataFrame({"date": [pd.Timestamp("2024-03-31")], "external": [900.0]})

    result = crosscheck_sector(panel, external, "holdings", "external")
    assert len(result) == 1
    assert abs(result.iloc[0]["diff_pct"] - 11.11) < 0.1


def test_crosscheck_sector_no_overlap():
    panel = pd.DataFrame({"date": [pd.Timestamp("2024-03-31")], "holdings": [1000.0]})
    external = pd.DataFrame({"date": [pd.Timestamp("2023-03-31")], "external": [900.0]})

    result = crosscheck_sector(panel, external, "holdings", "external")
    assert result.empty


def test_run_crosschecks_with_bank_data():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-31")] * 2,
        "sector": ["banks", "money_market_funds"],
        "holdings": [1000.0, 500.0],
    })
    efa_bank = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-31")],
        "bank_treasury_holdings": [900.0],
    })

    summary = run_crosschecks(panel, efa_bank=efa_bank)
    assert len(summary) == 1
    assert summary.iloc[0]["sector"] == "banks"
