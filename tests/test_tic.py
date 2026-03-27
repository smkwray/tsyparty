"""Tests for ingest/tic.py — TIC SLT parsing."""

import pandas as pd
import pytest

from tsyparty.ingest.tic import parse_slt_global, parse_slt_table1_countries


def _make_slt_global_csv(tmp_path, n_months=12):
    """Create a minimal synthetic SLT global CSV."""
    lines = [""] * 14  # skip header rows
    base_date = pd.Timestamp("2023-01-15")
    for i in range(n_months):
        date = base_date + pd.DateOffset(months=i)
        date_str = date.strftime("%Y-%m")
        # Grand Total row (country_code 99996)
        lines.append(f"Grand Total,99996,{date_str},500000,{300000 + i * 1000},100000,50000,50000")
        # Some country rows
        lines.append(f"Japan,59440,{date_str},100000,80000,5000,5000,10000")
    path = tmp_path / "slt1d_globl.csv"
    path.write_text("\n".join(lines))
    return path


def test_parse_slt_global_basic(tmp_path):
    path = _make_slt_global_csv(tmp_path)
    result = parse_slt_global(path)
    assert not result.empty
    assert "date" in result.columns
    assert "tic_foreign_treasury" in result.columns
    assert len(result) >= 3  # at least 3 quarters from 12 months


def test_parse_slt_global_quarterly(tmp_path):
    path = _make_slt_global_csv(tmp_path, n_months=24)
    result = parse_slt_global(path)
    # Should be quarterly resampled
    diffs = result["date"].diff().dropna()
    # Quarter gaps should be ~90 days
    assert all(d.days >= 80 for d in diffs)


def test_parse_slt_global_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_slt_global(tmp_path / "nonexistent.csv")


def _make_slt_table1_txt(tmp_path, n_months=6):
    """Create a minimal synthetic SLT table1.txt."""
    lines = []
    base_date = pd.Timestamp("2023-01-15")
    for i in range(n_months):
        date = base_date + pd.DateOffset(months=i)
        date_str = date.strftime("%Y-%m-%d")
        # Tab-separated: country, code, date, col3, col4, col5, col6(treasury), col7, col8
        lines.append(f"Japan\t59440\t{date_str}\t100\t200\t300\t80000\t400\t500")
        lines.append(f"United Kingdom\t11200\t{date_str}\t50\t100\t150\t40000\t200\t250")
    path = tmp_path / "slt_table1.txt"
    path.write_text("\n".join(lines))
    return path


def test_parse_slt_table1_countries_basic(tmp_path):
    path = _make_slt_table1_txt(tmp_path)
    result = parse_slt_table1_countries(path)
    assert not result.empty
    assert set(result.columns) == {"date", "country", "country_code", "treasury"}
    assert "Japan" in result["country"].values


def test_parse_slt_table1_countries_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_slt_table1_countries(tmp_path / "nonexistent.txt")
