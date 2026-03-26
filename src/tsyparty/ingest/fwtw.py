"""Download and parse the Enhanced Financial Accounts From-Whom-To-Whom data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tsyparty.config import data_root
from tsyparty.ingest.base import download_with_manifest
from tsyparty.registry import load_sources
from tsyparty.utils_http import fetch_text

# ---------------------------------------------------------------------------
# FWTW series-code structure
#
# The FWTW CSV has series codes encoding:
#   {prefix}{issuer}{holder}{instrument}{valuation}.Q
#
# For Treasury securities, the issuer is the federal government (sector 31)
# and instruments are coded with prefix 3061 (Treasury securities).
#
# We parse the holder portion of the mnemonic to map to canonical sectors.
# ---------------------------------------------------------------------------

# Map from FWTW holder code fragment to canonical sector.
# The FWTW uses Financial Accounts sector codes embedded in series names.
# These are matched against the holder portion of the series mnemonic.
FWTW_HOLDER_MAP: dict[str, str] = {
    "10": "households_residual",
    "15": "households_residual",
    "11": "nonfinancial_corporates",
    "14": "nonfinancial_corporates",
    "21": "state_local_governments",
    "26": "foreigners_official",
    "31": "fed",        # federal government retirement/general
    "34": "pensions",   # federal retirement
    "40": "other_financial",
    "41": "banks",      # U.S.-chartered depository institutions
    "47": "banks",      # credit unions
    "50": "other_financial",
    "51": "insurers",   # property-casualty
    "54": "insurers",   # life insurance
    "56": "mutual_funds_etfs",  # closed-end + ETFs
    "57": "pensions",   # private pension funds
    "22": "pensions",   # state/local retirement
    "61": "other_financial",  # finance companies
    "63": "money_market_funds",
    "64": "other_financial",  # REITs
    "65": "mutual_funds_etfs",
    "66": "dealers",
    "67": "other_financial",  # ABS issuers
    "71": "fed",        # monetary authority
    "73": "other_financial",  # holding companies
    "75": "banks",      # foreign banking offices
    "76": "banks",      # banks n.e.c.
    "86": "dealers",
}

# Series-code pattern for FWTW data: identifies Treasury-related series
_FWTW_SERIES_RE = re.compile(
    r"^(?P<prefix>[A-Z]{2})"
    r"(?P<issuer>\d{2,3})"
    r"(?P<holder>\d{2,3})"
    r"(?P<instrument>3061\d+)"
    r"\.Q$"
)

# Simpler pattern: just match any quarterly series
_QUARTERLY_SERIES_RE = re.compile(r"^[A-Z]{2}\d+\.Q$")


@dataclass(slots=True)
class FWTWParseResult:
    """Parsed FWTW quarterly Treasury holdings by holder sector."""

    holdings: pd.DataFrame
    """Columns: date, sector, instrument, holdings (billions USD)."""

    raw_series_count: int
    """Number of quarterly series found in the raw CSV."""

    unmapped_series: list[str]
    """Series codes that could not be mapped to canonical sectors."""


def download_fwtw(dest_dir: str | Path | None = None) -> Path:
    """Download the FWTW CSV and its data dictionary."""
    sources = load_sources()
    spec = sources["fwtw_csv"]
    if dest_dir is None:
        dest_dir = data_root() / "raw_public" / "fwtw"
    dest_dir = Path(dest_dir)

    dest = dest_dir / "fwtw_data.csv"
    download_with_manifest(
        spec.direct_url,
        dest,
        {"source": spec.key, "landing_url": spec.landing_url},
    )

    # Also grab the data dictionary for reference
    dict_url = spec.raw.get("dictionary_url")
    if dict_url:
        dict_dest = dest_dir / "fwtw_data_dictionary.txt"
        try:
            text = fetch_text(dict_url)
            dict_dest.write_text(text, encoding="utf-8")
        except Exception:
            pass  # dictionary is nice-to-have, not critical

    return dest


def parse_fwtw_csv(csv_path: str | Path) -> FWTWParseResult:
    """Parse the FWTW data CSV into quarterly sector Treasury holdings.

    The FWTW CSV uses a wide format with series codes as columns and
    quarterly dates as rows.  We filter for Treasury-security series
    and map holder codes to canonical sectors.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Read the CSV - FWTW format has a header row with series codes
    # and a date column.  Try common column-name patterns.
    df_raw = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Identify the date column (often first column or named 'date'/'Date')
    date_col = _find_date_column(df_raw)
    if date_col is None:
        raise ValueError("Could not identify date column in FWTW CSV")

    # Parse dates
    df_raw[date_col] = df_raw[date_col].apply(_parse_fwtw_date)
    df_raw = df_raw.dropna(subset=[date_col])

    # Find all quarterly series columns
    series_cols = [
        c for c in df_raw.columns
        if c != date_col and _QUARTERLY_SERIES_RE.match(c.strip())
    ]

    raw_series_count = len(series_cols)
    unmapped: list[str] = []
    records: list[dict[str, Any]] = []

    for col in series_cols:
        col_clean = col.strip()

        # Try to parse the series code for holder info
        sector = _map_fwtw_series_to_sector(col_clean)
        if sector is None:
            unmapped.append(col_clean)
            continue

        for _, row in df_raw.iterrows():
            date = row[date_col]
            raw_val = str(row[col]).strip()
            if not raw_val or raw_val in ("ND", "nd", "-", "nan", "None"):
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue

            records.append(
                {
                    "date": date,
                    "sector": sector,
                    "instrument": "all_treasuries",
                    "holdings": value,
                }
            )

    holdings = pd.DataFrame(records)
    if not holdings.empty:
        holdings = (
            holdings
            .groupby(["date", "sector", "instrument"], as_index=False)["holdings"]
            .sum()
            .sort_values(["date", "sector"])
            .reset_index(drop=True)
        )

    return FWTWParseResult(
        holdings=holdings,
        raw_series_count=raw_series_count,
        unmapped_series=unmapped,
    )


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Find the date column by name or content heuristics."""
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("date", "observation_date", "obs_date", "time_period"):
            return col
    # Fall back: first column with year-quarter patterns
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if sample.apply(lambda x: bool(re.search(r"\d{4}[:\-]?[Qq]\d", str(x)))).any():
            return col
    return None


def _parse_fwtw_date(raw: str) -> pd.Timestamp | None:
    """Parse FWTW date strings to quarter-end timestamps."""
    raw = str(raw).strip()
    # Try "YYYY:Q1" or "YYYY-Q1" format
    match = re.search(r"(\d{4})[:\-]?[Qq](\d)", raw)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        return pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)
    # Try ISO date
    try:
        ts = pd.Timestamp(raw)
        return ts
    except Exception:
        return None


def _map_fwtw_series_to_sector(series_code: str) -> str | None:
    """Map a FWTW series mnemonic to a canonical sector key.

    The FWTW series encodes issuer and holder.  For Treasury securities
    the issuer is always federal government.  We extract the holder
    code and look it up in FWTW_HOLDER_MAP.
    """
    match = _FWTW_SERIES_RE.match(series_code)
    if not match:
        return None

    holder = match.group("holder")
    return FWTW_HOLDER_MAP.get(holder)
