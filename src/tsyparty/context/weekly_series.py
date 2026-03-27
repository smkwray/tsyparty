"""Common weekly-series normalization contract.

All higher-frequency context parsers produce DataFrames with this schema:
    date          : datetime64  — observation date
    series_id     : str         — unique series identifier within source
    value         : float       — observed value
    frequency     : str         — "weekly" | "daily" | "monthly"
    units         : str         — e.g. "millions_usd", "percent", "count"
    source_key    : str         — matches sources.yml key

Optional columns (added by specific parsers):
    instrument    : str         — e.g. "treasury_notes_bonds", "treasury_bills"
    security_type : str         — e.g. "nominal", "tips"
    maturity_bucket : str       — e.g. "0-1y", "1-5y", "5-10y", "10y+"
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"date", "series_id", "value", "frequency", "units", "source_key"}
OPTIONAL_COLUMNS = {"instrument", "security_type", "maturity_bucket"}


def validate_weekly_series(df: pd.DataFrame) -> None:
    """Raise ValueError if the DataFrame doesn't match the weekly-series contract."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime64")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("'value' column must be numeric")
    if df["series_id"].isna().any():
        raise ValueError("'series_id' column must not contain null values")


def quarter_end_aggregate(
    df: pd.DataFrame,
    agg: str = "last",
) -> pd.DataFrame:
    """Aggregate weekly series to quarter-end values.

    Parameters
    ----------
    df : weekly-series DataFrame
    agg : aggregation method — "last" (quarter-end snapshot) or "mean"

    Returns a DataFrame with the same schema but quarterly frequency.
    """
    validate_weekly_series(df)
    out = df.copy()
    out["quarter"] = out["date"].dt.to_period("Q").dt.to_timestamp("Q")
    if agg == "last":
        result = out.sort_values("date").groupby(["quarter", "series_id"], as_index=False).last()
    elif agg == "mean":
        result = out.groupby(["quarter", "series_id"], as_index=False).agg(
            {"value": "mean", "frequency": "first", "units": "first", "source_key": "first"}
        )
    else:
        raise ValueError(f"Unsupported agg method: {agg!r}. Use 'last' or 'mean'.")
    result["date"] = result["quarter"]
    result["frequency"] = "quarterly"
    return result.drop(columns=["quarter"])
