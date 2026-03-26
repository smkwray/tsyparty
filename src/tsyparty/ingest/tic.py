"""Parse TIC SLT data for foreign Treasury holdings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_slt_global(csv_path: str | Path) -> pd.DataFrame:
    """Parse TIC SLT historical global CSV into quarterly foreign Treasury holdings.

    Uses the Grand Total row to avoid double-counting regional aggregates.

    Returns a DataFrame with columns: date, tic_foreign_treasury (millions USD).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df_raw = pd.read_csv(
        csv_path, dtype=str, skiprows=14, header=None,
        on_bad_lines="skip", low_memory=False,
    )
    if df_raw.shape[1] < 5:
        raise ValueError("Unexpected TIC SLT format: too few columns")

    df_raw.columns = [
        "country", "country_code", "date", "total_longterm",
        "treasury", "agency", "corp_bonds", "corp_stocks",
    ][:df_raw.shape[1]]

    # Use Grand Total (99996) or All Countries (69995) for aggregate
    totals = df_raw[df_raw["country_code"].str.strip() == "99996"].copy()
    if totals.empty:
        totals = df_raw[df_raw["country_code"].str.strip() == "69995"].copy()
    if totals.empty:
        raise ValueError("Could not find aggregate total row in TIC SLT data")

    totals["treasury_val"] = pd.to_numeric(
        totals["treasury"].str.replace(",", "").str.strip(), errors="coerce"
    )
    totals["date_ts"] = pd.to_datetime(totals["date"].str.strip(), format="%Y-%m", errors="coerce")
    totals = totals.dropna(subset=["date_ts", "treasury_val"])

    monthly = totals[["date_ts", "treasury_val"]].copy()
    monthly = monthly.sort_values("date_ts").set_index("date_ts")
    quarterly = monthly.resample("QE").last().dropna().reset_index()
    quarterly.columns = ["date", "tic_foreign_treasury"]
    return quarterly


def parse_slt_table1_countries(txt_path: str | Path) -> pd.DataFrame:
    """Parse TIC SLT table1.txt into country-level monthly Treasury holdings.

    Returns a DataFrame with columns: date, country, country_code, treasury.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    records = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        date_str = parts[2].strip()
        if not date_str or not date_str[:4].isdigit():
            continue
        country = parts[0].strip()
        country_code = parts[1].strip()
        tsy_holdings = parts[6].strip().replace(",", "")
        try:
            ts = pd.Timestamp(date_str)
            val = float(tsy_holdings)
        except (ValueError, TypeError):
            continue
        records.append({
            "date": ts,
            "country": country,
            "country_code": country_code,
            "treasury": val,
        })

    return pd.DataFrame(records)
