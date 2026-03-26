"""Parse Z.1 Financial Accounts CSV zip into quarterly sector Treasury holdings."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Z.1 series-code → canonical sector mapping
#
# The Z.1 CSV tables use series mnemonics of the form
#   {prefix}{sector_code}{instrument_code}.Q
# where prefix is FL (flow/level), LM (market-value level), etc.
# Table L.210 contains Treasury-securities levels by sector.
#
# The mapping below covers the instrument suffix 3061105.Q
# (total Treasury securities) in the L.210 table.  Sector codes
# follow the Financial Accounts coding guide.
# ---------------------------------------------------------------------------

# Map from Z.1 sector code (string) to canonical sector key.
# Sector codes vary in length (2-3 digits).  We match on the series
# mnemonic directly.
Z1_SERIES_SECTOR_MAP: dict[str, str] = {
    # Households and nonprofit organizations (residual category)
    "FL153061105.Q": "households_residual",
    # Nonfinancial corporate business
    "FL113061105.Q": "nonfinancial_corporates",
    # Nonfinancial noncorporate business (fold into corporates)
    "FL143061105.Q": "nonfinancial_corporates",
    # State and local governments
    "FL213061105.Q": "state_local_governments",
    # Federal government retirement funds (fold into pensions)
    "FL343061105.Q": "pensions",
    # Monetary authority (Fed/SOMA)
    "FL713061105.Q": "fed",
    # U.S.-chartered depository institutions
    "FL763061105.Q": "banks",
    # Foreign banking offices in U.S. (fold into banks)
    "FL753061105.Q": "banks",
    # Credit unions (fold into banks)
    "FL473061105.Q": "banks",
    # Property-casualty insurance
    "FL513061105.Q": "insurers",
    # Life insurance companies
    "FL543061105.Q": "insurers",
    # Private pension funds
    "FL573061105.Q": "pensions",
    # State and local government retirement funds
    "FL223061105.Q": "pensions",
    # Money market mutual funds
    "FL633061105.Q": "money_market_funds",
    # Mutual funds (open-end, ex money-market)
    "FL653061105.Q": "mutual_funds_etfs",
    # Exchange-traded funds
    "FL563061105.Q": "mutual_funds_etfs",
    # Government-sponsored enterprises (fold into other_financial)
    "FL403061105.Q": "other_financial",
    # Security brokers and dealers
    "FL663061105.Q": "dealers",
    # Rest of the world
    "FL263061105.Q": "foreigners_official",
    # Holding companies (fold into other_financial)
    "FL733061105.Q": "other_financial",
    # Funding corporations (fold into other_financial)
    "FL503061105.Q": "other_financial",
    # Finance companies (fold into other_financial)
    "FL613061105.Q": "other_financial",
    # REITs (fold into other_financial)
    "FL643061105.Q": "other_financial",
    # ABS issuers (fold into other_financial)
    "FL673061105.Q": "other_financial",
}

# Additional series for total Treasury securities outstanding
# (used for reconciliation denominator).
Z1_TOTAL_SERIES = "FL893061105.Q"  # all sectors total

# Pattern: table L.210 file inside the Z.1 zip
_L210_FILENAME_RE = re.compile(r"l210", re.IGNORECASE)

# Pattern: quarterly date string in Z.1 CSVs (e.g. "2024:Q1" or "2024q1")
_DATE_RE = re.compile(r"(\d{4})[:\-]?[Qq](\d)")


@dataclass(slots=True)
class Z1ParseResult:
    """Parsed quarterly sector Treasury holdings from Z.1."""

    holdings: pd.DataFrame
    """Columns: date, sector, instrument, holdings (billions USD)."""

    unmapped_series: list[str]
    """Series codes found in the table but not in the sector map."""

    source_file: str
    """Name of the table file parsed inside the zip."""


def _quarter_to_date(raw: str) -> pd.Timestamp | None:
    """Convert '2024:Q1' style strings to quarter-end Timestamp."""
    match = _DATE_RE.search(str(raw))
    if not match:
        return None
    year, quarter = int(match.group(1)), int(match.group(2))
    return pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)


def _find_l210_in_zip(zf: zipfile.ZipFile) -> str | None:
    """Return the first entry matching the L.210 table."""
    for name in zf.namelist():
        if _L210_FILENAME_RE.search(Path(name).stem) and name.endswith(".csv"):
            return name
    return None


def parse_z1_zip(zip_path: str | Path) -> Z1ParseResult:
    """Extract and parse table L.210 from a Z.1 CSV zip.

    Parameters
    ----------
    zip_path : path to the downloaded z1_csv_files.zip

    Returns
    -------
    Z1ParseResult with a long-form holdings DataFrame, unmapped series, and source info.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        table_name = _find_l210_in_zip(zf)
        if table_name is None:
            # Fall back: try to find any CSV with "210" in the name
            for name in zf.namelist():
                if "210" in name and name.endswith(".csv"):
                    table_name = name
                    break
        if table_name is None:
            raise FileNotFoundError("No L.210 table found in zip")

        raw_bytes = zf.read(table_name)

    text = raw_bytes.decode("utf-8", errors="replace")
    return _parse_l210_csv(text, table_name)


def _parse_l210_csv(text: str, source_file: str) -> Z1ParseResult:
    """Parse the L.210 CSV text into structured holdings."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV")

    # Find the header row: the one containing series codes (FL*.Q pattern)
    header_idx = None
    series_pattern = re.compile(r"^[A-Z]{2}\d+\.\w$")
    for i, row in enumerate(rows):
        matches = sum(1 for cell in row if series_pattern.match(cell.strip()))
        if matches >= 1:
            header_idx = i
            break

    if header_idx is None:
        # Try simpler: first row with a year-quarter pattern in col 0
        # is a data row; use the row before it as the header.
        for i, row in enumerate(rows):
            if row and re.match(r"\d{4}", row[0].strip()):
                header_idx = max(0, i - 1)
                break
            if row and "date" in row[0].lower():
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not locate header row in L.210 CSV")

    headers = [h.strip() for h in rows[header_idx]]
    data_rows = rows[header_idx + 1 :]

    # If the header row itself looks like data, step back
    if _DATE_RE.match(headers[0]):
        data_rows = [rows[header_idx]] + data_rows
        # Use previous row as header
        if header_idx > 0:
            headers = [h.strip() for h in rows[header_idx - 1]]
        else:
            raise ValueError("Date found in first row with no preceding header")

    # Identify series columns
    series_cols: dict[int, str] = {}
    for col_idx, header in enumerate(headers):
        if series_pattern.match(header):
            series_cols[col_idx] = header

    unmapped: list[str] = []
    records: list[dict[str, Any]] = []

    for row in data_rows:
        if not row or not row[0].strip():
            continue
        date = _quarter_to_date(row[0])
        if date is None:
            continue

        for col_idx, series_code in series_cols.items():
            if col_idx >= len(row):
                continue
            raw_val = row[col_idx].strip()
            if not raw_val or raw_val in ("ND", "nd", "-"):
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue

            sector = Z1_SERIES_SECTOR_MAP.get(series_code)
            if sector is None:
                if series_code not in unmapped and series_code != Z1_TOTAL_SERIES:
                    unmapped.append(series_code)
                if series_code == Z1_TOTAL_SERIES:
                    records.append(
                        {
                            "date": date,
                            "sector": "_total",
                            "instrument": "all_treasuries",
                            "holdings": value,
                        }
                    )
                continue

            records.append(
                {
                    "date": date,
                    "sector": sector,
                    "instrument": "all_treasuries",
                    "holdings": value,
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        # Aggregate sectors that map to the same canonical key
        df = (
            df.groupby(["date", "sector", "instrument"], as_index=False)["holdings"]
            .sum()
            .sort_values(["date", "sector"])
            .reset_index(drop=True)
        )

    return Z1ParseResult(holdings=df, unmapped_series=unmapped, source_file=source_file)


def z1_holdings_wide(result: Z1ParseResult) -> pd.DataFrame:
    """Pivot parsed Z.1 holdings into a date x sector wide table (billions)."""
    df = result.holdings
    if df.empty:
        return pd.DataFrame()
    sectors = df[df["sector"] != "_total"]
    return sectors.pivot_table(
        index="date", columns="sector", values="holdings", aggfunc="sum"
    ).sort_index()
