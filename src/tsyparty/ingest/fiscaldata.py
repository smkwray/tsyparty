"""Parse Fiscal Data API responses (debt-to-penny, etc.)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def parse_debt_to_penny(json_path: str | Path) -> pd.DataFrame:
    """Parse debt-to-penny API JSON into quarterly debt totals.

    Returns a DataFrame with columns: date, public_debt (in millions USD).
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with open(json_path) as f:
        data = json.load(f)

    records = data.get("data", [])
    rows = []
    for rec in records:
        date_str = rec.get("record_date", "")
        held_public = rec.get("debt_held_public_amt", "null")
        if held_public == "null" or not held_public:
            continue
        try:
            ts = pd.Timestamp(date_str)
            val = float(held_public)
        except (ValueError, TypeError):
            continue
        rows.append({"date": ts, "public_debt": val / 1_000_000})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "public_debt"])

    df = df.sort_values("date").set_index("date")
    quarterly = df.resample("QE").last().dropna().reset_index()
    return quarterly
