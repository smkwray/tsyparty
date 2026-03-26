"""Parse Enhanced Financial Accounts (EFA) data sources."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def parse_efa_mmf_treasury(csv_path: str | Path) -> pd.DataFrame:
    """Parse EFA MMF holdings CSV, extracting quarterly Treasury holdings.

    Returns a DataFrame with columns: date, mmf_treasury_holdings (millions USD).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype=str)
    col = "North America; USA; Treasury"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in EFA MMF CSV")

    records = []
    for _, row in df.iterrows():
        date_raw = str(row["Date"]).strip()
        match = re.match(r"(\d{4})\s+(\w+)", date_raw)
        if match:
            ts = pd.Timestamp(f"{match.group(2)} {match.group(1)}")
        else:
            try:
                ts = pd.Timestamp(date_raw)
            except Exception:
                continue
        val = str(row[col]).replace(",", "").strip()
        try:
            records.append({"date": ts, "mmf_treasury_holdings": float(val)})
        except (ValueError, TypeError):
            continue

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["date", "mmf_treasury_holdings"])

    result = result.sort_values("date").set_index("date")
    quarterly = result.resample("QE").last().dropna().reset_index()
    return quarterly


def parse_efa_bank_treasury(csv_path: str | Path) -> pd.DataFrame:
    """Parse EFA consolidated bank balance sheet, extracting Treasury holdings.

    Returns a DataFrame with columns: date, bank_treasury_holdings (millions USD).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype=str)
    col = "Assets: Treasury securities"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in EFA banks CSV")

    records = []
    for _, row in df.iterrows():
        date_raw = str(row["Date"]).strip()
        match = re.match(r"(\d{4}):Q(\d)", date_raw)
        if not match:
            continue
        year, q = int(match.group(1)), int(match.group(2))
        ts = pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
        val = str(row[col]).replace(",", "").strip()
        try:
            # EFA banks data is in billions → convert to millions
            records.append({"date": ts, "bank_treasury_holdings": float(val) * 1000})
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def parse_efa_international(csv_path: str | Path) -> pd.DataFrame:
    """Parse EFA international portfolio investment, extracting total foreign holdings.

    Returns a DataFrame with columns: date, foreign_total_lt_securities (millions USD).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype=str)
    if "Worldwide" not in df.columns:
        raise ValueError("Expected column 'Worldwide' not found in EFA international CSV")

    records = []
    for _, row in df.iterrows():
        date_raw = str(row["Date"]).strip()
        try:
            ts = pd.Timestamp(date_raw)
        except Exception:
            continue
        val = str(row["Worldwide"]).replace(",", "").strip()
        try:
            # EFA international is in billions → convert to millions
            records.append({"date": ts, "foreign_total_lt_securities": float(val) * 1000})
        except (ValueError, TypeError):
            continue

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["date", "foreign_total_lt_securities"])

    result = result.sort_values("date").set_index("date")
    quarterly = result.resample("QE").last().dropna().reset_index()
    return quarterly
