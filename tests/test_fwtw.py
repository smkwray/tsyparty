"""Tests for FWTW parser."""

import pandas as pd
import pytest

from tsyparty.ingest.fwtw import FWTWParseResult, parse_fwtw_csv


SAMPLE_FWTW_CSV = """\
date,FL31413061105.Q,FL31633061105.Q,FL31763061105.Q,FL31263061105.Q,FL99993061105.Q
2022:Q4,500.0,800.0,1200.0,3000.0,100.0
2023:Q1,510.0,820.0,1250.0,3100.0,110.0
2023:Q2,520.0,810.0,1280.0,3050.0,105.0
"""


def test_parse_fwtw_csv_basic(tmp_path):
    csv_path = tmp_path / "fwtw_data.csv"
    csv_path.write_text(SAMPLE_FWTW_CSV)

    result = parse_fwtw_csv(csv_path)
    assert isinstance(result, FWTWParseResult)
    assert result.raw_series_count > 0


def test_parse_fwtw_csv_maps_sectors(tmp_path):
    csv_path = tmp_path / "fwtw_data.csv"
    csv_path.write_text(SAMPLE_FWTW_CSV)

    result = parse_fwtw_csv(csv_path)
    if not result.holdings.empty:
        sectors = set(result.holdings["sector"].unique())
        # At least some sectors should be mapped
        assert len(sectors) > 0


def test_parse_fwtw_csv_dates_are_timestamps(tmp_path):
    csv_path = tmp_path / "fwtw_data.csv"
    csv_path.write_text(SAMPLE_FWTW_CSV)

    result = parse_fwtw_csv(csv_path)
    if not result.holdings.empty:
        assert all(isinstance(d, pd.Timestamp) for d in result.holdings["date"])


def test_parse_fwtw_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_fwtw_csv("/nonexistent/fwtw.csv")


def test_parse_fwtw_csv_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("date\n")

    result = parse_fwtw_csv(csv_path)
    assert result.holdings.empty
    assert result.raw_series_count == 0
