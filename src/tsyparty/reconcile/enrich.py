"""Enrichment: split foreign holdings into official vs private sectors."""

from __future__ import annotations

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
    """Split the foreigners_official sector into official and private.

    Parameters
    ----------
    panel : harmonized panel DataFrame (date, sector, instrument, holdings, source)
    official_share : quarterly Series of official-share fractions, or None for default
    default_official_share : fallback share when no TIC data available

    Returns
    -------
    Enriched panel with foreigners_official and foreigners_private rows.
    """
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
