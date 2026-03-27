"""Tests for primary dealer statistics parser."""

import json
import pandas as pd
import pytest

from tsyparty.context.dealer import parse_dealer_json
from tsyparty.context.weekly_series import validate_weekly_series


def _make_dealer_json(tmp_path, n_weeks=20):
    """Create a synthetic primary dealer JSON file."""
    records = []
    base = pd.Timestamp("2023-01-04")
    for i in range(n_weeks):
        date = base + pd.Timedelta(weeks=i)
        records.append({
            "asOfDate": date.strftime("%Y-%m-%d"),
            "description": "Treasury Net Position",
            "value": 50000 + i * 100,
        })
        records.append({
            "asOfDate": date.strftime("%Y-%m-%d"),
            "description": "Treasury Financing Repo",
            "value": 120000 + i * 200,
        })
    data = {"pd": {"timeseries": records}}
    path = tmp_path / "primary_dealer_statistics.json"
    path.write_text(json.dumps(data))
    return path


def test_parse_dealer_json_basic(tmp_path):
    path = _make_dealer_json(tmp_path)
    result = parse_dealer_json(path)
    assert not result.weekly.empty
    validate_weekly_series(result.weekly)
    assert result.weekly["source_key"].iloc[0] == "primary_dealer_statistics"


def test_parse_dealer_json_multiple_series(tmp_path):
    path = _make_dealer_json(tmp_path)
    result = parse_dealer_json(path)
    assert result.n_series >= 1


def test_parse_dealer_json_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_dealer_json(tmp_path / "nonexistent.json")


def test_parse_dealer_json_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"pd": {"timeseries": []}}))
    result = parse_dealer_json(path)
    assert result.weekly.empty
    assert result.n_series == 0


def test_parse_dealer_json_flat_array(tmp_path):
    records = [
        {"asOfDate": "2023-01-04", "description": "Treasury Net Position", "value": 50000},
        {"asOfDate": "2023-01-11", "description": "Treasury Net Position", "value": 50100},
    ]
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(records))
    result = parse_dealer_json(path)
    assert result.n_series >= 1
