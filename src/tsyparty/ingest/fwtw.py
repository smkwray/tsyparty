"""Download and parse the Enhanced Financial Accounts From-Whom-To-Whom data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from tsyparty.config import data_root
from tsyparty.ingest.base import download_with_manifest
from tsyparty.registry import load_sources
from tsyparty.utils_http import fetch_text

# ---------------------------------------------------------------------------
# FWTW data format
#
# The FWTW CSV uses a long format with columns:
#   Instrument Name, Instrument Code, Holder Name, Holder Code,
#   Issuer Name, Issuer Code, Date, Level
#
# Treasury securities have Instrument Code = "30611" and are
# issued by the Federal Government (Issuer Code = "31").
# ---------------------------------------------------------------------------

# Treasury securities instrument code in FWTW data.
FWTW_TREASURY_INSTRUMENT = "30611"
FWTW_FEDERAL_GOVT_ISSUER = "31"

# Map from FWTW Holder Code to canonical sector.
FWTW_HOLDER_MAP: dict[str, str] = {
    "10": "nonfinancial_corporates",  # Nonfin Corp Bus
    "11": "nonfinancial_corporates",  # Nonfin Noncorp Bus
    "15": "households_residual",
    "21": "state_local_governments",
    "26": "foreigners_official",      # Rest of World
    "42": "other_financial",          # GSE and Agency
    "47": "banks",                    # Credit Unions
    "50": "other_financial",          # Other Fin. Bus.
    "51": "insurers",                 # PC Insurance
    "54": "insurers",                 # Life Insurance
    "55": "other_financial",          # Closed-End Funds
    "56": "mutual_funds_etfs",        # ETFs
    "59": "pensions",                 # Pensions (consolidated)
    "63": "money_market_funds",       # MMF
    "65": "mutual_funds_etfs",        # Mutual Funds
    "66": "dealers",                  # Broker/Dealers
    "67": "other_financial",          # ABS
    "71": "fed",                      # Monetary Authority
    "73": "other_financial",          # Holding Companies
    "74": "banks",                    # Banks in U.S.-Affiliated Areas
    "75": "banks",                    # FBOs
    "76": "banks",                    # US-Chartered
    "89": "_total",                   # All sectors total
    "90": "_discrepancy",             # Statistical discrepancy
}


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

    The FWTW CSV uses a long format with columns:
        Instrument Name, Instrument Code, Holder Name, Holder Code,
        Issuer Name, Issuer Code, Date, Level
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df_raw = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Normalize column names for robustness
    col_map = {c: c.strip() for c in df_raw.columns}
    df_raw = df_raw.rename(columns=col_map)

    required = {"Instrument Code", "Holder Code", "Issuer Code", "Date", "Level"}
    if not required.issubset(df_raw.columns):
        # Empty or incompatible file
        return FWTWParseResult(holdings=pd.DataFrame(), raw_series_count=0, unmapped_series=[])

    # Filter to Treasury securities issued by Federal Government
    tsy = df_raw[
        (df_raw["Instrument Code"].str.strip() == FWTW_TREASURY_INSTRUMENT)
        & (df_raw["Issuer Code"].str.strip() == FWTW_FEDERAL_GOVT_ISSUER)
    ].copy()

    raw_series_count = len(tsy)

    if tsy.empty:
        return FWTWParseResult(holdings=pd.DataFrame(), raw_series_count=0, unmapped_series=[])

    # Parse dates
    tsy["date"] = tsy["Date"].apply(_parse_fwtw_date)
    tsy = tsy.dropna(subset=["date"])

    # Map holder codes to canonical sectors
    tsy["holder_code"] = tsy["Holder Code"].str.strip()
    tsy["sector"] = tsy["holder_code"].map(FWTW_HOLDER_MAP)

    unmapped_codes = sorted(tsy.loc[tsy["sector"].isna(), "holder_code"].unique().tolist())
    tsy = tsy.dropna(subset=["sector"])

    # Parse level values
    tsy["holdings"] = pd.to_numeric(tsy["Level"].str.strip(), errors="coerce")
    tsy = tsy.dropna(subset=["holdings"])

    tsy["instrument"] = "all_treasuries"

    holdings = (
        tsy.groupby(["date", "sector", "instrument"], as_index=False)["holdings"]
        .sum()
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )

    return FWTWParseResult(
        holdings=holdings,
        raw_series_count=raw_series_count,
        unmapped_series=unmapped_codes,
    )


def _parse_fwtw_date(raw: str) -> pd.Timestamp | None:
    """Parse FWTW date strings to quarter-end timestamps.

    Handles formats: '2024Q1', '2024:Q1', '2024-Q1'.
    """
    raw = str(raw).strip()
    match = re.search(r"(\d{4})[:\-]?[Qq]?(\d)$", raw)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        if 1 <= quarter <= 4:
            return pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None
