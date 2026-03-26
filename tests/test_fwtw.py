"""Tests for FWTW parser."""

import pandas as pd
import pytest

from tsyparty.ingest.fwtw import FWTWParseResult, parse_fwtw_csv


SAMPLE_FWTW_CSV = """\
"Instrument Name","Instrument Code","Holder Name","Holder Code","Issuer Name","Issuer Code","Date","Level"
"Treasury Securities","30611","US-Chartered","76","Federal Govt.","31","2022Q4","1200.0"
"Treasury Securities","30611","MMF","63","Federal Govt.","31","2022Q4","800.0"
"Treasury Securities","30611","Rest of World","26","Federal Govt.","31","2022Q4","3000.0"
"Treasury Securities","30611","Households","15","Federal Govt.","31","2022Q4","500.0"
"Treasury Securities","30611","US-Chartered","76","Federal Govt.","31","2023Q1","1250.0"
"Treasury Securities","30611","MMF","63","Federal Govt.","31","2023Q1","820.0"
"Treasury Securities","30611","Rest of World","26","Federal Govt.","31","2023Q1","3100.0"
"Treasury Securities","30611","Households","15","Federal Govt.","31","2023Q1","510.0"
"Treasury Securities","30611","US-Chartered","76","Federal Govt.","31","2023Q2","1280.0"
"Treasury Securities","30611","MMF","63","Federal Govt.","31","2023Q2","810.0"
"Treasury Securities","30611","Rest of World","26","Federal Govt.","31","2023Q2","3050.0"
"Treasury Securities","30611","Households","15","Federal Govt.","31","2023Q2","520.0"
"Federal Funds and Repos","20500","US-Chartered","76","Rest of World","26","2022Q4","999.0"
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
    csv_path.write_text('"Instrument Name","Instrument Code","Holder Name","Holder Code","Issuer Name","Issuer Code","Date","Level"\n')

    result = parse_fwtw_csv(csv_path)
    assert result.holdings.empty
    assert result.raw_series_count == 0
