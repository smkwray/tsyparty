"""Parse Treasury investor-class auction allotment XLS files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Investor-class XLS structure
#
# Treasury publishes separate XLS files for Coupons and Bills.
# Each file has auction-level rows with allotment amounts by buyer class:
#   - FIMA (Foreign and International Monetary Authorities)
#   - Primary Dealers
#   - Direct Bidders
#   - Indirect Bidders (includes foreign central banks, investment managers)
#
# We map these to canonical sectors where possible and preserve the
# primary-market buyer-class labels for composition analysis.
# ---------------------------------------------------------------------------

# Map from auction buyer-class column patterns to canonical labels.
# These are matched case-insensitively against XLS column headers.
BUYER_CLASS_MAP: dict[str, str] = {
    "fima": "foreign_official",
    "foreign": "foreign_official",
    "primary dealer": "dealers",
    "dealer": "dealers",
    "direct": "direct_bidders",
    "indirect": "indirect_bidders",
    "competitive": "competitive_total",
    "noncompetitive": "noncompetitive",
    "soma": "fed",
}

# Map from buyer-class labels to canonical sectors (where unambiguous)
BUYER_CLASS_SECTOR_MAP: dict[str, str] = {
    "foreign_official": "foreigners_official",
    "dealers": "dealers",
    "fed": "fed",
}

# Instrument detection from filename
_BILL_RE = re.compile(r"bill", re.IGNORECASE)
_COUPON_RE = re.compile(r"coupon", re.IGNORECASE)


@dataclass(slots=True)
class AuctionParseResult:
    """Parsed auction investor-class allotment data."""

    allotments: pd.DataFrame
    """Columns: date, security_type, term, buyer_class, allotment_amount."""

    quarterly_composition: pd.DataFrame
    """Quarterly aggregated buyer-class shares. Columns: date, instrument, buyer_class, share."""

    instrument: str
    """'bills' or 'nominal_coupons'."""


def _detect_instrument(path: Path) -> str:
    """Detect whether the file is for bills or coupons from the filename."""
    name = path.stem
    if _BILL_RE.search(name):
        return "bills"
    if _COUPON_RE.search(name):
        return "nominal_coupons"
    return "all_treasuries"


def _normalize_buyer_class(col_name: str) -> str | None:
    """Map a raw column header to a standardized buyer-class label."""
    lower = col_name.strip().lower()
    for pattern, label in BUYER_CLASS_MAP.items():
        if pattern in lower:
            return label
    return None


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Find the row index that contains column headers in an XLS sheet.

    Treasury XLS files often have title rows before the actual data header.
    We look for a row containing keywords like 'Date', 'CUSIP', 'Term', etc.
    """
    keywords = {"date", "cusip", "term", "type", "security", "issue", "maturity"}
    for i in range(min(20, len(df_raw))):
        row_vals = [str(v).strip().lower() for v in df_raw.iloc[i] if pd.notna(v)]
        hits = sum(1 for v in row_vals if v in keywords)
        if hits >= 2:
            return i
    return 0


def parse_investor_class_xls(xls_path: str | Path) -> AuctionParseResult:
    """Parse an investor-class allotment XLS into structured auction data.

    Parameters
    ----------
    xls_path : path to the downloaded XLS (bills or coupons)

    Returns
    -------
    AuctionParseResult with per-auction allotments and quarterly composition.
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(xls_path)

    instrument = _detect_instrument(xls_path)

    # Read all sheets; usually the first sheet has the data
    df_raw = pd.read_excel(xls_path, sheet_name=0, header=None, dtype=str)

    header_row = _find_header_row(df_raw)
    headers = [str(v).strip() if pd.notna(v) else f"col_{i}" for i, v in enumerate(df_raw.iloc[header_row])]
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    # Find date column
    date_col = _find_date_col(headers)
    if date_col is None:
        raise ValueError(f"Could not find date column in {xls_path.name}")

    # Find term/type column for labeling
    term_col = _find_col_by_keywords(headers, ["term", "type", "security"])

    # Find buyer-class amount columns
    buyer_cols: dict[str, str] = {}  # original_header -> buyer_class label
    for h in headers:
        label = _normalize_buyer_class(h)
        if label is not None:
            buyer_cols[h] = label

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_date = str(row[date_col]).strip()
        date = _parse_auction_date(raw_date)
        if date is None:
            continue

        term = str(row[term_col]).strip() if term_col else ""

        for col_header, buyer_class in buyer_cols.items():
            raw_val = str(row[col_header]).strip()
            if not raw_val or raw_val in ("nan", "None", "-", ""):
                continue
            # Remove commas, dollar signs
            raw_val = raw_val.replace(",", "").replace("$", "").strip()
            try:
                amount = float(raw_val)
            except ValueError:
                continue
            if amount <= 0:
                continue

            records.append(
                {
                    "date": date,
                    "security_type": instrument,
                    "term": term,
                    "buyer_class": buyer_class,
                    "allotment_amount": amount,
                }
            )

    allotments = pd.DataFrame(records)
    quarterly_comp = _compute_quarterly_composition(allotments, instrument)

    return AuctionParseResult(
        allotments=allotments,
        quarterly_composition=quarterly_comp,
        instrument=instrument,
    )


def _find_date_col(headers: list[str]) -> str | None:
    """Find date column by keyword."""
    for h in headers:
        if h.strip().lower() in ("date", "auction date", "issue date", "settlement date"):
            return h
    for h in headers:
        if "date" in h.lower():
            return h
    return None


def _find_col_by_keywords(headers: list[str], keywords: list[str]) -> str | None:
    """Return the first header matching any keyword."""
    for h in headers:
        lower = h.strip().lower()
        for kw in keywords:
            if kw in lower:
                return h
    return None


def _parse_auction_date(raw: str) -> pd.Timestamp | None:
    """Parse various date formats from auction XLS files."""
    if not raw or raw in ("nan", "None"):
        return None
    # Try standard date parsing
    try:
        return pd.Timestamp(raw)
    except Exception:
        pass
    # Try MM/DD/YYYY
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw)
    if match:
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except Exception:
            pass
    return None


def _compute_quarterly_composition(
    allotments: pd.DataFrame, instrument: str
) -> pd.DataFrame:
    """Aggregate auction allotments into quarterly buyer-class shares."""
    if allotments.empty:
        return pd.DataFrame(columns=["date", "instrument", "buyer_class", "share"])

    df = allotments.copy()
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp("Q")

    quarterly = (
        df.groupby(["quarter", "buyer_class"], as_index=False)["allotment_amount"]
        .sum()
    )
    totals = quarterly.groupby("quarter")["allotment_amount"].transform("sum")
    quarterly["share"] = quarterly["allotment_amount"] / totals.replace(0, float("nan"))
    quarterly = quarterly.rename(columns={"quarter": "date"})
    quarterly["instrument"] = instrument

    return quarterly[["date", "instrument", "buyer_class", "share"]].sort_values(
        ["date", "buyer_class"]
    ).reset_index(drop=True)
