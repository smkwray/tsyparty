"""Enrichment: split foreign holdings into official vs private sectors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Countries most likely to hold US Treasuries via official/sovereign accounts.
OFFICIAL_HOLDER_COUNTRIES = frozenset({
    "Japan", "China, Mainland", "Saudi Arabia", "India", "South Korea",
    "Taiwan", "Brazil", "Singapore", "Hong Kong", "Thailand",
    "Israel", "Kuwait", "United Arab Emirates", "Qatar", "Norway",
    "Russia", "Mexico", "Turkey", "Indonesia", "Philippines",
})


def estimate_official_share(tic_countries: pd.DataFrame) -> pd.Series:
    """Estimate the official share of foreign Treasury holdings per quarter.

    Uses TIC SLT country-level data and a heuristic classifier based on
    known official holders. Returns a quarterly Series indexed by date.

    Parameters
    ----------
    tic_countries : DataFrame with columns: date, country, treasury
    """
    df = tic_countries.copy()
    df["is_official"] = df["country"].isin(OFFICIAL_HOLDER_COUNTRIES)

    monthly = df.groupby(["date", "is_official"], as_index=False)["treasury"].sum()
    totals = monthly.groupby("date")["treasury"].transform("sum")
    monthly["share"] = monthly["treasury"] / totals.replace(0, float("nan"))

    official = monthly[monthly["is_official"]].set_index("date")["share"]
    return official.resample("QE").mean().dropna()


def enrich_foreign_split(
    panel: pd.DataFrame,
    official_share: pd.Series | None = None,
    default_official_share: float = 0.65,
) -> pd.DataFrame:
    """Split the base-panel foreign total into official and private sectors.

    Parameters
    ----------
    panel : harmonized panel DataFrame (date, sector, instrument, holdings, source).
            In the base panel, the historical `foreigners_official` label is the
            combined rest-of-world total from Z.1/FWTW before TIC enrichment.
    official_share : quarterly Series of official-share fractions, or None for default
    default_official_share : fallback share when no TIC data available

    Returns
    -------
    Enriched panel with foreigners_official and foreigners_private rows.
    """
    # Historical naming kept for downstream compatibility: this sector is the
    # combined foreign total before enrichment, not already-official holdings.
    foreign_mask = panel["sector"] == "foreigners_official"
    foreign_rows = panel[foreign_mask].copy()
    other_rows = panel[~foreign_mask].copy()

    enriched = []
    for _, row in foreign_rows.iterrows():
        q = row["date"]
        share = default_official_share
        if official_share is not None and q in official_share.index:
            share = float(official_share.loc[q])

        official_val = row["holdings"] * share
        private_val = row["holdings"] * (1 - share)
        enriched.append({**row.to_dict(), "sector": "foreigners_official", "holdings": official_val})
        enriched.append({**row.to_dict(), "sector": "foreigners_private", "holdings": private_val})

    result = pd.concat([other_rows, pd.DataFrame(enriched)], ignore_index=True)
    return result.sort_values(["date", "sector"]).reset_index(drop=True)


def write_enrichment_metadata(
    out_path: Path,
    official_share: pd.Series | None,
    default_official_share: float = 0.65,
) -> Path:
    """Write enrichment metadata JSON describing the split method and shares."""
    if official_share is not None and not official_share.empty:
        method = "tic_country_heuristic"
        avg_share = float(official_share.mean())
        min_share = float(official_share.min())
        max_share = float(official_share.max())
        n_quarters = len(official_share)
    else:
        method = "default_constant"
        avg_share = default_official_share
        min_share = default_official_share
        max_share = default_official_share
        n_quarters = 0

    metadata = {
        "split_method": method,
        "official_holder_countries": sorted(OFFICIAL_HOLDER_COUNTRIES),
        "avg_official_share": round(avg_share, 4),
        "min_official_share": round(min_share, 4),
        "max_official_share": round(max_share, 4),
        "quarters_with_tic_data": n_quarters,
        "default_official_share": default_official_share,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path
