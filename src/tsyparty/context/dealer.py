"""Primary dealer statistics parser.

Parses NY Fed primary dealer statistics JSON/CSV into the common
weekly-series schema. Focuses on:
  - Treasury net positions
  - Treasury financing (repo/reverse repo)
  - Treasury fails to deliver/receive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import pandas as pd

from tsyparty.context.weekly_series import validate_weekly_series


# Series of interest from the primary dealer data
DEALER_SERIES_PATTERNS = {
    "net_position": {
        "keywords": ["treasury", "net position", "net_position"],
        "series_id_prefix": "pd_treasury_net_position",
    },
    "financing": {
        "keywords": ["treasury", "financing", "repo"],
        "series_id_prefix": "pd_treasury_financing",
    },
    "fails": {
        "keywords": ["treasury", "fail"],
        "series_id_prefix": "pd_treasury_fails",
    },
}


@dataclass(slots=True)
class DealerParseResult:
    """Result of parsing primary dealer statistics."""

    weekly: pd.DataFrame  # common weekly-series schema
    n_series: int
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    source_periods: list[str]  # track structure changes across FR 2004 regimes


def parse_dealer_json(path: str | Path) -> DealerParseResult:
    """Parse primary dealer statistics JSON from the NY Fed API.

    The JSON structure varies across time periods. This parser handles
    the common formats and tags records with their source period.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dealer statistics JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle different JSON structures
    if isinstance(raw, dict):
        records = raw.get("pd", raw.get("data", raw.get("timeseries", [])))
        if isinstance(records, dict):
            records = records.get("timeseries", records.get("data", []))
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    if not records:
        empty = pd.DataFrame(columns=list({"date", "series_id", "value", "frequency", "units", "source_key"}))
        return DealerParseResult(weekly=empty, n_series=0, date_range=None, source_periods=[])

    df = pd.DataFrame(records)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("asofdate", "asof_date", "as_of_date", "report_date"):
            col_map[col] = "report_date"
        elif lower == "date":
            col_map[col] = "report_date"
        elif lower in ("keyid", "key_id"):
            col_map[col] = "series_key"
        elif lower in ("description", "series_name"):
            col_map[col] = "series_name"
        elif lower in ("value", "amount", "val"):
            col_map[col] = "value"
        elif lower == "name" or lower == "label":
            col_map[col] = "series_name"
    df = df.rename(columns=col_map)

    # Try to find date and value columns
    date_col = "report_date" if "report_date" in df.columns else None
    if date_col is None:
        date_candidates = [c for c in df.columns if "date" in c.lower()]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise ValueError(f"No date column found. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    value_col = "value" if "value" in df.columns else None
    if value_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            raise ValueError(f"No numeric column found. Columns: {list(df.columns)}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Determine series identity
    name_col = None
    for candidate in ("series_name", "series_key", "description"):
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        name_candidates = [c for c in df.columns if c.lower() in ("name", "label", "series")]
        if name_candidates:
            name_col = name_candidates[0]

    # Build weekly-series output
    rows = []
    source_periods = set()

    if name_col:
        for series_name, grp in df.groupby(name_col):
            series_lower = str(series_name).lower()
            series_id = _classify_series(series_lower, str(series_name))

            # Track source period metadata
            period_col = next((c for c in grp.columns if "period" in c.lower()), None)
            if period_col:
                source_periods.update(grp[period_col].dropna().unique())

            for _, row in grp.iterrows():
                val = row.get(value_col)
                if pd.isna(val):
                    continue
                rows.append({
                    "date": row[date_col],
                    "series_id": series_id,
                    "value": float(val),
                    "frequency": "weekly",
                    "units": "millions_usd",
                    "source_key": "primary_dealer_statistics",
                })
    else:
        # No series name — treat as single series
        for _, row in df.iterrows():
            val = row.get(value_col)
            if pd.isna(val):
                continue
            rows.append({
                "date": row[date_col],
                "series_id": "pd_treasury_aggregate",
                "value": float(val),
                "frequency": "weekly",
                "units": "millions_usd",
                "source_key": "primary_dealer_statistics",
            })

    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return DealerParseResult(weekly=weekly, n_series=0, date_range=None, source_periods=[])

    validate_weekly_series(weekly)
    date_range = (weekly["date"].min(), weekly["date"].max())

    return DealerParseResult(
        weekly=weekly.sort_values("date").reset_index(drop=True),
        n_series=weekly["series_id"].nunique(),
        date_range=date_range,
        source_periods=sorted(source_periods),
    )


def _classify_series(name_lower: str, name_raw: str) -> str:
    """Classify a dealer series into one of the target categories."""
    for category, info in DEALER_SERIES_PATTERNS.items():
        if all(kw in name_lower for kw in info["keywords"]):
            return info["series_id_prefix"]
    # Fallback: sanitize the name
    safe = name_lower.replace(" ", "_").replace("/", "_")[:50]
    return f"pd_{safe}"
