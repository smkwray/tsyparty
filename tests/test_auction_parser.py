"""Tests for auction investor-class parser."""

import pandas as pd
import pytest

from tsyparty.ingest.auction_parser import AuctionParseResult, parse_investor_class_xls


def _make_simple_xls(tmp_path, filename: str = "IC-Bills.xls") -> str:
    """Create a minimal investor-class XLS for testing."""
    df = pd.DataFrame(
        {
            "Auction Date": ["01/15/2024", "02/15/2024", "03/15/2024"],
            "Term": ["4-Week", "4-Week", "4-Week"],
            "Primary Dealer": [5000, 6000, 5500],
            "Direct Bidder": [2000, 2500, 2200],
            "Indirect Bidder": [3000, 3500, 3300],
        }
    )
    path = tmp_path / filename
    df.to_excel(path, index=False)
    return str(path)


def test_parse_xls_basic(tmp_path):
    xls_path = _make_simple_xls(tmp_path)
    result = parse_investor_class_xls(xls_path)

    assert isinstance(result, AuctionParseResult)
    assert result.instrument == "bills"
    assert not result.allotments.empty
    assert len(result.allotments) == 9  # 3 auctions x 3 buyer classes


def test_parse_xls_detects_coupons(tmp_path):
    xls_path = _make_simple_xls(tmp_path, "IC-Coupons.xls")
    result = parse_investor_class_xls(xls_path)
    assert result.instrument == "nominal_coupons"


def test_quarterly_composition_sums_to_one(tmp_path):
    xls_path = _make_simple_xls(tmp_path)
    result = parse_investor_class_xls(xls_path)

    qc = result.quarterly_composition
    if not qc.empty:
        for _, grp in qc.groupby("date"):
            total_share = grp["share"].sum()
            assert abs(total_share - 1.0) < 1e-8, f"Shares sum to {total_share}, expected 1.0"


def test_parse_xls_dates_are_timestamps(tmp_path):
    xls_path = _make_simple_xls(tmp_path)
    result = parse_investor_class_xls(xls_path)

    assert all(isinstance(d, pd.Timestamp) for d in result.allotments["date"])


def test_parse_xls_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_investor_class_xls("/nonexistent/IC-Bills.xls")
