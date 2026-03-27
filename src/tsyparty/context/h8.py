"""H.8 bank balance sheet parser.

Parses Federal Reserve H.8 statistical release CSV data into the common
weekly-series schema. Focuses on Treasury and agency securities holdings
for all commercial banks, domestically chartered banks, and foreign-related
institutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tsyparty.context.weekly_series import validate_weekly_series


# H.8 series IDs for Treasury & agency holdings (seasonally adjusted)
# These are the Data Download Program series codes
H8_SERIES = {
    # All commercial banks — Treasury & agency securities
    "H8/B1058NCBAM": {
        "series_id": "h8_all_banks_treasury_agency",
        "description": "All commercial banks: Treasury and agency securities",
        "bank_group": "all_commercial_banks",
    },
    # Domestically chartered — Treasury & agency
    "H8/B1058NCLAM": {
        "series_id": "h8_domestic_banks_treasury_agency",
        "description": "Domestically chartered banks: Treasury and agency securities",
        "bank_group": "domestically_chartered",
    },
    # Foreign-related institutions — Treasury & agency
    "H8/B1058NFRAM": {
        "series_id": "h8_foreign_banks_treasury_agency",
        "description": "Foreign-related institutions: Treasury and agency securities",
        "bank_group": "foreign_related",
    },
}


@dataclass(slots=True)
class H8ParseResult:
    """Result of parsing H.8 data."""

    weekly: pd.DataFrame  # common weekly-series schema
    n_series: int
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None


def parse_h8_csv(path: str | Path) -> H8ParseResult:
    """Parse H.8 CSV from the Fed's Data Download Program.

    The CSV has a header section followed by data rows with dates in the
    first column and series values in subsequent columns. Column headers
    contain series identifiers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H.8 CSV not found: {path}")

    # The Fed DDP CSV has variable header rows. Read all, skip metadata.
    raw = pd.read_csv(path, header=None, dtype=str)

    # Find the row that contains "Time Period" or a date pattern
    header_row = None
    for i, row in raw.iterrows():
        first_val = str(row.iloc[0]).strip().lower()
        if first_val in ("time period", "series description", ""):
            continue
        # Check if it looks like a date (YYYY-MM-DD)
        if len(first_val) >= 8 and first_val[:4].isdigit():
            header_row = i
            break

    if header_row is None:
        # Try assuming the first row is headers and second is data
        header_row = 1

    # Use the row before data as column names if it contains series IDs
    if header_row > 0:
        potential_headers = raw.iloc[header_row - 1]
        col_names = ["date"] + [str(c).strip() for c in potential_headers.iloc[1:]]
    else:
        col_names = ["date"] + [f"series_{i}" for i in range(raw.shape[1] - 1)]

    data = raw.iloc[header_row:].copy()
    data.columns = col_names[:data.shape[1]]

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    if data.empty:
        return H8ParseResult(
            weekly=pd.DataFrame(columns=sorted({"date", "series_id", "value", "frequency", "units", "source_key"})),
            n_series=0,
            date_range=None,
        )

    # Melt to long format
    value_cols = [c for c in data.columns if c != "date"]
    rows = []
    for col in value_cols:
        series_info = H8_SERIES.get(col)
        series_id = series_info["series_id"] if series_info else f"h8_{col}"

        col_data = data[["date", col]].copy()
        col_data[col] = pd.to_numeric(col_data[col], errors="coerce")
        col_data = col_data.dropna(subset=[col])
        if col_data.empty:
            continue

        for _, row in col_data.iterrows():
            rows.append({
                "date": row["date"],
                "series_id": series_id,
                "value": float(row[col]),
                "frequency": "weekly",
                "units": "millions_usd",
                "source_key": "h8_release_page",
            })

    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return H8ParseResult(weekly=weekly, n_series=0, date_range=None)

    validate_weekly_series(weekly)
    date_range = (weekly["date"].min(), weekly["date"].max())

    return H8ParseResult(
        weekly=weekly.sort_values("date").reset_index(drop=True),
        n_series=weekly["series_id"].nunique(),
        date_range=date_range,
    )
