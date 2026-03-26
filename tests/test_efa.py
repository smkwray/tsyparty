"""Tests for EFA parsers."""

import pandas as pd
import pytest

from tsyparty.ingest.efa import parse_efa_mmf_treasury, parse_efa_bank_treasury


SAMPLE_MMF_CSV = """\
Date,Worldwide,North America; USA; Treasury
2023 Dec,3500000.0,2800000.0
2024 Jan,3550000.0,2850000.0
2024 Feb,3600000.0,2900000.0
2024 Mar,3650000.0,2950000.0
"""

SAMPLE_BANK_CSV = """\
Date,Assets: Total,Assets: Treasury securities
2023:Q4,20000.0,1200.5
2024:Q1,20500.0,1250.3
2024:Q2,21000.0,1300.1
"""


def test_parse_efa_mmf_treasury(tmp_path):
    path = tmp_path / "mmf.csv"
    path.write_text(SAMPLE_MMF_CSV)

    result = parse_efa_mmf_treasury(path)
    assert not result.empty
    assert "mmf_treasury_holdings" in result.columns
    # Should aggregate to quarters
    assert all(isinstance(d, pd.Timestamp) for d in result["date"])


def test_parse_efa_bank_treasury(tmp_path):
    path = tmp_path / "banks.csv"
    path.write_text(SAMPLE_BANK_CSV)

    result = parse_efa_bank_treasury(path)
    assert not result.empty
    assert len(result) == 3
    # Values should be in millions (input is billions * 1000)
    assert result.iloc[0]["bank_treasury_holdings"] == 1200.5 * 1000


def test_parse_efa_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_efa_mmf_treasury("/nonexistent/mmf.csv")
