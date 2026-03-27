"""SOMA (System Open Market Account) holdings parser.

Parses NY Fed SOMA Treasury holdings data into the common weekly-series
schema, plus a quarterly delta_soma aggregator for the behavior layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd

from tsyparty.context.weekly_series import validate_weekly_series, quarter_end_aggregate


@dataclass(slots=True)
class SomaParseResult:
    """Result of parsing SOMA holdings data."""

    weekly: pd.DataFrame  # weekly-series schema
    quarterly_delta: pd.DataFrame  # date, delta_soma columns
    n_records: int


def parse_soma_json(path: str | Path) -> SomaParseResult:
    """Parse SOMA Treasury holdings JSON (from NY Fed API) into weekly series.

    Expected JSON structure from the NY Fed Markets API:
    {
      "soma": {
        "holdings": [
          {"asOfDate": "2024-01-03", "parValue": ..., "percentOutstanding": ..., ...},
          ...
        ]
      }
    }

    Or a flat array of holding records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SOMA JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle different JSON structures from the NY Fed API
    if isinstance(raw, dict):
        if "soma" in raw and isinstance(raw["soma"], dict):
            # summary endpoint: soma.summary, holdings endpoint: soma.holdings
            inner = raw["soma"]
            records = inner.get("summary", inner.get("holdings", []))
        elif "soma" in raw and isinstance(raw["soma"], list):
            records = raw["soma"]
        else:
            records = raw.get("holdings", raw.get("data", raw.get("summary", [])))
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected SOMA JSON structure: {type(raw)}")

    if not records:
        return SomaParseResult(
            weekly=pd.DataFrame(columns=["date", "series_id", "value", "frequency", "units", "source_key"]),
            quarterly_delta=pd.DataFrame(columns=["date", "delta_soma"]),
            n_records=0,
        )

    df = pd.DataFrame(records)

    # Normalize column names — NY Fed uses camelCase
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "date" in lower and "asof" in lower:
            col_map[col] = "as_of_date"
        elif lower in ("asofdate", "as_of_date"):
            col_map[col] = "as_of_date"
        elif "parvalue" in lower or "par_value" in lower:
            col_map[col] = "par_value"
        elif "currentfacevalue" in lower:
            col_map[col] = "face_value"
    df = df.rename(columns=col_map)

    if "as_of_date" not in df.columns:
        # Try the first date-like column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "as_of_date"})
        else:
            raise ValueError(f"No date column found in SOMA data. Columns: {list(df.columns)}")

    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df = df.dropna(subset=["as_of_date"])

    # Determine value column — prefer Treasury-specific total over grand total
    # Summary endpoint has: total, notesbonds, bills, tips, frn, mbs, agencies, etc.
    # We want Treasury total = notesbonds + bills + tips + frn (excluding MBS/agencies)
    treasury_cols = [c for c in ("notesbonds", "bills", "tips", "frn") if c in df.columns]
    if treasury_cols:
        # Compute Treasury-only total from components
        for tc in treasury_cols:
            df[tc] = pd.to_numeric(df[tc], errors="coerce").fillna(0)
        df["_tsy_total"] = sum(df[tc] for tc in treasury_cols)
        value_col = "_tsy_total"
    else:
        value_col = None
        for candidate in ("total", "par_value", "face_value", "parValue", "currentFaceValue"):
            if candidate in df.columns:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]
            else:
                raise ValueError(f"No numeric value column found in SOMA data. Columns: {list(df.columns)}")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Aggregate to weekly totals (usually already one row per date from summary)
    weekly_totals = (
        df.groupby("as_of_date")[value_col]
        .sum()
        .reset_index()
        .rename(columns={"as_of_date": "date", value_col: "value"})
        .sort_values("date")
    )

    # Convert to millions if values look like they're in raw dollars
    if weekly_totals["value"].median() > 1e9:
        weekly_totals["value"] = weekly_totals["value"] / 1e6

    weekly_totals["series_id"] = "soma_treasury_total"
    weekly_totals["frequency"] = "weekly"
    weekly_totals["units"] = "millions_usd"
    weekly_totals["source_key"] = "soma_holdings_page"

    validate_weekly_series(weekly_totals)

    # Compute quarterly delta_soma
    quarterly = quarter_end_aggregate(weekly_totals, agg="last")
    quarterly = quarterly.sort_values("date")
    quarterly["delta_soma"] = quarterly["value"].diff()
    quarterly_delta = quarterly[["date", "delta_soma"]].dropna().reset_index(drop=True)

    return SomaParseResult(
        weekly=weekly_totals.reset_index(drop=True),
        quarterly_delta=quarterly_delta,
        n_records=len(records),
    )
