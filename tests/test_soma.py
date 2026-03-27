"""Tests for SOMA parser and weekly-series contract."""

import json
import pandas as pd
import pytest

from tsyparty.context.soma import parse_soma_json
from tsyparty.context.weekly_series import validate_weekly_series, quarter_end_aggregate, REQUIRED_COLUMNS


def _make_soma_json(tmp_path, n_weeks=52):
    """Create a synthetic SOMA JSON file."""
    records = []
    base = pd.Timestamp("2023-01-04")
    for i in range(n_weeks):
        date = base + pd.Timedelta(weeks=i)
        records.append({
            "asOfDate": date.strftime("%Y-%m-%d"),
            "parValue": 5_000_000 + i * 10_000,  # raw dollars
        })
    data = {"soma": {"holdings": records}}
    path = tmp_path / "soma_holdings_page.json"
    path.write_text(json.dumps(data))
    return path


def test_parse_soma_json_basic(tmp_path):
    path = _make_soma_json(tmp_path)
    result = parse_soma_json(path)
    assert result.n_records == 52
    assert not result.weekly.empty
    assert "soma_treasury_total" in result.weekly["series_id"].values
    validate_weekly_series(result.weekly)


def test_parse_soma_json_quarterly_delta(tmp_path):
    path = _make_soma_json(tmp_path, n_weeks=104)
    result = parse_soma_json(path)
    assert not result.quarterly_delta.empty
    assert "date" in result.quarterly_delta.columns
    assert "delta_soma" in result.quarterly_delta.columns


def test_parse_soma_json_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"soma": {"holdings": []}}))
    result = parse_soma_json(path)
    assert result.n_records == 0
    assert result.weekly.empty


def test_parse_soma_json_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_soma_json(tmp_path / "nonexistent.json")


def test_parse_soma_json_flat_array(tmp_path):
    """Should handle a flat array of records."""
    records = [
        {"asOfDate": "2023-01-04", "parValue": 5_000_000},
        {"asOfDate": "2023-01-11", "parValue": 5_010_000},
    ]
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(records))
    result = parse_soma_json(path)
    assert result.n_records == 2


def test_weekly_series_validate_rejects_missing_cols():
    df = pd.DataFrame({"date": [pd.Timestamp("2023-01-04")], "value": [1.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_weekly_series(df)


def test_weekly_series_validate_rejects_non_datetime():
    df = pd.DataFrame({
        "date": ["2023-01-04"],
        "series_id": ["x"],
        "value": [1.0],
        "frequency": ["weekly"],
        "units": ["millions_usd"],
        "source_key": ["test"],
    })
    with pytest.raises(ValueError, match="datetime64"):
        validate_weekly_series(df)


def test_quarter_end_aggregate():
    dates = pd.date_range("2023-01-04", periods=52, freq="W-WED")
    df = pd.DataFrame({
        "date": dates,
        "series_id": "test",
        "value": range(52),
        "frequency": "weekly",
        "units": "millions_usd",
        "source_key": "test",
    })
    quarterly = quarter_end_aggregate(df, agg="last")
    assert not quarterly.empty
    assert quarterly["frequency"].iloc[0] == "quarterly"
    # Should have ~4 quarters
    assert len(quarterly) >= 3
