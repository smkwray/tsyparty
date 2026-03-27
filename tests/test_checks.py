"""Tests for validate/checks.py — tolerance assertions and market clearing."""

import pandas as pd
import pytest

from tsyparty.validate.checks import assert_close_series, validate_market_clearing


def test_assert_close_series_passes():
    left = pd.Series({"a": 1.0, "b": 2.0})
    right = pd.Series({"a": 1.0, "b": 2.0})
    assert_close_series(left, right)


def test_assert_close_series_fails():
    left = pd.Series({"a": 1.0, "b": 2.0})
    right = pd.Series({"a": 1.0, "b": 3.0})
    with pytest.raises(AssertionError):
        assert_close_series(left, right)


def test_validate_market_clearing_passes():
    matrix = pd.DataFrame(
        [[10.0, 20.0], [30.0, 40.0]],
        index=["a", "b"],
        columns=["x", "y"],
    )
    row_targets = pd.Series({"a": 30.0, "b": 70.0})
    col_targets = pd.Series({"x": 40.0, "y": 60.0})
    result = validate_market_clearing(matrix, row_targets, col_targets)
    assert result["passes"]


def test_validate_market_clearing_fails():
    matrix = pd.DataFrame(
        [[10.0, 20.0], [30.0, 40.0]],
        index=["a", "b"],
        columns=["x", "y"],
    )
    row_targets = pd.Series({"a": 50.0, "b": 70.0})  # doesn't match
    col_targets = pd.Series({"x": 40.0, "y": 60.0})
    result = validate_market_clearing(matrix, row_targets, col_targets)
    assert not result["passes"]
    assert result["max_row_gap"] > 0
