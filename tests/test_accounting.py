"""Tests for reconcile/accounting.py — reconciliation math."""

import pandas as pd
import pytest

from tsyparty.reconcile.accounting import reconcile_public_debt, summarize_gap_frame


def test_reconcile_public_debt():
    result = reconcile_public_debt(
        public_debt=1000.0,
        soma_holdings=300.0,
        sector_total=600.0,
    )
    assert result.residual_gap == 100.0
    assert result.public_debt == 1000.0


def test_reconcile_public_debt_exact():
    result = reconcile_public_debt(
        public_debt=1000.0,
        soma_holdings=400.0,
        sector_total=600.0,
    )
    assert result.residual_gap == 0.0


def test_summarize_gap_frame():
    df = pd.DataFrame({
        "public_debt": [1000.0, 1100.0],
        "soma_holdings": [300.0, 320.0],
        "sector_total": [600.0, 650.0],
    })
    result = summarize_gap_frame(df)
    assert "residual_gap" in result.columns
    assert result["residual_gap"].iloc[0] == 100.0
    assert result["residual_gap"].iloc[1] == 130.0


def test_summarize_gap_frame_missing_cols():
    df = pd.DataFrame({"public_debt": [1000.0]})
    with pytest.raises(ValueError, match="Missing columns"):
        summarize_gap_frame(df)
