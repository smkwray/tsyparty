"""Tests for Z.1 CSV parser."""

import io
import zipfile

import pandas as pd
import pytest

from tsyparty.ingest.z1_parser import (
    Z1_SERIES_SECTOR_MAP,
    Z1_TOTAL_SERIES,
    Z1ParseResult,
    parse_z1_zip,
    z1_holdings_wide,
)


def _make_z1_zip(tmp_path, csv_text: str, filename: str = "l210.csv") -> str:
    """Create a minimal Z.1 zip with the given CSV content."""
    zip_path = tmp_path / "z1_csv_files.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"z1_csv_files/{filename}", csv_text)
    return str(zip_path)


SAMPLE_CSV = """\
Series Description,FL153061105.Q,FL763061105.Q,FL633061105.Q,FL263061105.Q,FL893061105.Q
Households,Banks,Money Market Funds,Rest of World,All Sectors
2023:Q1,500.0,1200.0,800.0,3000.0,6000.0
2023:Q2,510.0,1250.0,820.0,3100.0,6200.0
2023:Q3,490.0,1180.0,790.0,3050.0,6100.0
"""


def test_parse_z1_zip_basic(tmp_path):
    zip_path = _make_z1_zip(tmp_path, SAMPLE_CSV)
    result = parse_z1_zip(zip_path)

    assert isinstance(result, Z1ParseResult)
    assert result.source_file.endswith("l210.csv")
    assert not result.holdings.empty

    # Should have mapped to canonical sectors
    sectors = set(result.holdings["sector"].unique())
    assert "households_residual" in sectors
    assert "banks" in sectors
    assert "money_market_funds" in sectors
    assert "foreigners_official" in sectors
    assert "_total" in sectors


def test_parse_z1_zip_aggregates_sectors(tmp_path):
    # FL763061105 and FL753061105 both map to "banks"
    csv = """\
Series Description,FL763061105.Q,FL753061105.Q
US Banks,Foreign Banking Offices
2023:Q1,1000.0,200.0
"""
    zip_path = _make_z1_zip(tmp_path, csv)
    result = parse_z1_zip(zip_path)

    banks = result.holdings[
        (result.holdings["sector"] == "banks") & (result.holdings["date"] == pd.Timestamp("2023-03-31"))
    ]
    assert len(banks) == 1
    assert banks.iloc[0]["holdings"] == 1200.0


def test_parse_z1_zip_dates_are_quarter_end(tmp_path):
    csv = """\
Series,FL153061105.Q
Desc,Households
2024:Q1,100.0
2024:Q4,110.0
"""
    zip_path = _make_z1_zip(tmp_path, csv)
    result = parse_z1_zip(zip_path)

    dates = sorted(result.holdings["date"].unique())
    assert dates[0] == pd.Timestamp("2024-03-31")
    assert dates[1] == pd.Timestamp("2024-12-31")


def test_z1_holdings_wide(tmp_path):
    zip_path = _make_z1_zip(tmp_path, SAMPLE_CSV)
    result = parse_z1_zip(zip_path)
    wide = z1_holdings_wide(result)

    assert not wide.empty
    assert "_total" not in wide.columns
    assert "banks" in wide.columns
    assert len(wide) == 3  # 3 quarters


def test_parse_z1_zip_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_z1_zip("/nonexistent/z1.zip")


def test_parse_z1_zip_no_l210(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("other_table.csv", "a,b\n1,2\n")
    with pytest.raises(FileNotFoundError, match="L.210"):
        parse_z1_zip(zip_path)


def test_series_map_covers_key_sectors():
    """Verify the series map includes entries for all key analysis sectors."""
    mapped_sectors = set(Z1_SERIES_SECTOR_MAP.values())
    required = {"banks", "money_market_funds", "foreigners_official", "fed", "dealers", "insurers", "pensions"}
    assert required.issubset(mapped_sectors)
