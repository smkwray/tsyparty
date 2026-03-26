"""Tests for Fiscal Data parsers."""

import json

import pandas as pd
import pytest

from tsyparty.ingest.fiscaldata import parse_debt_to_penny


def test_parse_debt_to_penny_basic(tmp_path):
    data = {
        "data": [
            {"record_date": "2024-03-29", "debt_held_public_amt": "27000000000000.00", "intragov_hold_amt": "0", "tot_pub_debt_out_amt": "0"},
            {"record_date": "2024-03-28", "debt_held_public_amt": "26900000000000.00", "intragov_hold_amt": "0", "tot_pub_debt_out_amt": "0"},
            {"record_date": "2024-01-31", "debt_held_public_amt": "26500000000000.00", "intragov_hold_amt": "0", "tot_pub_debt_out_amt": "0"},
        ],
    }
    path = tmp_path / "debt.json"
    path.write_text(json.dumps(data))

    result = parse_debt_to_penny(str(path))
    assert not result.empty
    assert "date" in result.columns
    assert "public_debt" in result.columns
    # Should have 1 quarter (Q1 2024)
    assert len(result) == 1
    assert result.iloc[0]["public_debt"] == 27_000_000  # millions


def test_parse_debt_to_penny_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"data": []}))

    result = parse_debt_to_penny(str(path))
    assert result.empty


def test_parse_debt_to_penny_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_debt_to_penny("/nonexistent/debt.json")
