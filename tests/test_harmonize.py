"""Tests for harmonized panel builder and reconciliation."""

import pandas as pd
import pytest

from tsyparty.reconcile.harmonize import (
    HarmonizedPanel,
    ReconciliationReport,
    build_harmonized_panel,
    panel_to_wide,
    reconcile_panel,
)


def _sample_z1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-03-31"] * 3 + ["2023-06-30"] * 3),
            "sector": ["banks", "money_market_funds", "fed"] * 2,
            "instrument": ["all_treasuries"] * 6,
            "holdings": [1200.0, 800.0, 5000.0, 1250.0, 820.0, 5100.0],
        }
    )


def _sample_fwtw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-03-31"] * 2 + ["2023-06-30"] * 2),
            "sector": ["banks", "foreigners_official"] * 2,
            "instrument": ["all_treasuries"] * 4,
            "holdings": [1190.0, 3000.0, 1240.0, 3100.0],
        }
    )


def test_build_panel_z1_only():
    panel = build_harmonized_panel(z1_holdings=_sample_z1())
    assert isinstance(panel, HarmonizedPanel)
    assert panel.sources_used == ["z1"]
    assert not panel.panel.empty
    assert (panel.panel["source"] == "z1").all()


def test_build_panel_fwtw_only():
    panel = build_harmonized_panel(fwtw_holdings=_sample_fwtw())
    assert panel.sources_used == ["fwtw"]
    assert (panel.panel["source"] == "fwtw").all()


def test_build_panel_z1_priority():
    panel = build_harmonized_panel(_sample_z1(), _sample_fwtw(), priority="z1")
    assert set(panel.sources_used) == {"z1", "fwtw"}

    # Banks should come from z1 (priority)
    banks_q1 = panel.panel[
        (panel.panel["sector"] == "banks") & (panel.panel["date"] == pd.Timestamp("2023-03-31"))
    ]
    assert len(banks_q1) == 1
    assert banks_q1.iloc[0]["source"] == "z1"
    assert banks_q1.iloc[0]["holdings"] == 1200.0

    # Foreigners should come from fwtw (only source)
    foreign = panel.panel[panel.panel["sector"] == "foreigners_official"]
    assert not foreign.empty
    assert (foreign["source"] == "fwtw").all()


def test_build_panel_fwtw_priority():
    panel = build_harmonized_panel(_sample_z1(), _sample_fwtw(), priority="fwtw")

    # Banks should come from fwtw now
    banks_q1 = panel.panel[
        (panel.panel["sector"] == "banks") & (panel.panel["date"] == pd.Timestamp("2023-03-31"))
    ]
    assert len(banks_q1) == 1
    assert banks_q1.iloc[0]["source"] == "fwtw"
    assert banks_q1.iloc[0]["holdings"] == 1190.0


def test_build_panel_empty():
    panel = build_harmonized_panel()
    assert panel.panel.empty
    assert panel.sources_used == []
    assert panel.date_range is None


def test_panel_to_wide():
    panel = build_harmonized_panel(z1_holdings=_sample_z1())
    wide = panel_to_wide(panel)
    assert "banks" in wide.columns
    assert "fed" in wide.columns
    assert len(wide) == 2  # 2 quarters


def test_build_panel_excludes_meta_rows():
    z1 = _sample_z1()
    meta = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-03-31", "2023-03-31"]),
            "sector": ["_total", "_discrepancy"],
            "instrument": ["all_treasuries", "all_treasuries"],
            "holdings": [7000.0, 25.0],
        }
    )

    panel = build_harmonized_panel(z1_holdings=pd.concat([z1, meta], ignore_index=True))
    assert "_total" not in set(panel.panel["sector"])
    assert "_discrepancy" not in set(panel.panel["sector"])


def test_reconcile_panel_no_debt_totals():
    panel = build_harmonized_panel(z1_holdings=_sample_z1())
    report = reconcile_panel(panel)
    assert isinstance(report, ReconciliationReport)
    assert len(report.quarters) == 2  # 2 quarters
    assert "sector_total" in report.quarters.columns
    assert "soma_holdings" in report.quarters.columns


def test_reconcile_panel_with_debt_totals():
    panel = build_harmonized_panel(z1_holdings=_sample_z1())
    debt = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-03-31", "2023-06-30"]),
            "public_debt": [25000.0, 25500.0],
        }
    )
    report = reconcile_panel(panel, debt)
    assert not report.quarters["public_debt"].isna().any()
    assert not report.quarters["residual_gap"].isna().any()
    assert report.summary["quarters_with_debt_total"] == 2
