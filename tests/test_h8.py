"""Tests for H.8 bank balance sheet parser."""

import pandas as pd
import pytest

from tsyparty.context.h8 import parse_h8_csv
from tsyparty.context.weekly_series import validate_weekly_series


def _make_h8_csv(tmp_path, n_weeks=20):
    """Create a synthetic H.8 CSV file matching Fed DDP format."""
    lines = []
    lines.append('"Series Description","All commercial banks: Treasury and agency securities"')
    lines.append('"H8/B1058NCBAM",""')  # series ID row
    dates = pd.date_range("2023-01-06", periods=n_weeks, freq="W-FRI")
    for date in dates:
        lines.append(f'"{date.strftime("%Y-%m-%d")}","1234567"')
    path = tmp_path / "h8.csv"
    path.write_text("\n".join(lines))
    return path


def test_parse_h8_csv_basic(tmp_path):
    path = _make_h8_csv(tmp_path)
    result = parse_h8_csv(path)
    assert not result.weekly.empty
    validate_weekly_series(result.weekly)
    assert result.weekly["source_key"].iloc[0] == "h8_release_page"


def test_parse_h8_csv_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_h8_csv(tmp_path / "nonexistent.csv")


def test_parse_h8_csv_date_range(tmp_path):
    path = _make_h8_csv(tmp_path, n_weeks=52)
    result = parse_h8_csv(path)
    if result.date_range:
        assert result.date_range[0] < result.date_range[1]
