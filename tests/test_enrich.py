"""Tests for foreign enrichment module."""

import json
import pandas as pd

from tsyparty.reconcile.enrich import enrich_foreign_split, estimate_official_share, write_enrichment_metadata


def test_enrich_foreign_split_basic():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-31")] * 3,
        "sector": ["banks", "foreigners_official", "money_market_funds"],
        "instrument": ["all_treasuries"] * 3,
        "holdings": [1000.0, 5000.0, 800.0],
        "source": ["z1"] * 3,
    })

    result = enrich_foreign_split(panel, default_official_share=0.6)

    # Should have 4 rows: banks, foreigners_official, foreigners_private, mmf
    assert len(result) == 4
    sectors = set(result["sector"])
    assert "foreigners_official" in sectors
    assert "foreigners_private" in sectors

    official = result[result["sector"] == "foreigners_official"]["holdings"].iloc[0]
    private = result[result["sector"] == "foreigners_private"]["holdings"].iloc[0]
    assert abs(official - 3000.0) < 0.01  # 60% of 5000
    assert abs(private - 2000.0) < 0.01   # 40% of 5000


def test_enrich_preserves_other_sectors():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-31")] * 2,
        "sector": ["banks", "foreigners_official"],
        "instrument": ["all_treasuries"] * 2,
        "holdings": [1000.0, 5000.0],
        "source": ["z1"] * 2,
    })

    result = enrich_foreign_split(panel, default_official_share=0.5)
    banks = result[result["sector"] == "banks"]["holdings"].iloc[0]
    assert banks == 1000.0


def test_estimate_official_share():
    countries = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15"] * 3),
        "country": ["Japan", "China, Mainland", "United Kingdom"],
        "treasury": [1000.0, 800.0, 500.0],
    })

    share = estimate_official_share(countries)
    assert not share.empty
    # Japan + China = 1800 / 2300 total ≈ 78.3%
    assert share.iloc[0] > 0.7


def test_write_enrichment_metadata_with_tic(tmp_path):
    share = pd.Series([0.78, 0.75], index=pd.to_datetime(["2024-03-31", "2024-06-30"]))
    path = write_enrichment_metadata(tmp_path / "meta.json", share)
    assert path.exists()
    with open(path) as f:
        meta = json.load(f)
    assert meta["split_method"] == "tic_country_heuristic"
    assert meta["quarters_with_tic_data"] == 2
    assert 0.75 <= meta["avg_official_share"] <= 0.78


def test_write_enrichment_metadata_default(tmp_path):
    path = write_enrichment_metadata(tmp_path / "meta.json", None)
    with open(path) as f:
        meta = json.load(f)
    assert meta["split_method"] == "default_constant"
    assert meta["avg_official_share"] == 0.65
    assert meta["quarters_with_tic_data"] == 0
