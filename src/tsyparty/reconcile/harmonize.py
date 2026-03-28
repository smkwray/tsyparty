"""Build a harmonized quarterly holder panel from parsed sources and run stock reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tsyparty.config import load_yaml
from tsyparty.reconcile.accounting import ReconciliationResult, reconcile_public_debt


@dataclass(slots=True)
class HarmonizedPanel:
    """Quarterly sector-level Treasury holdings panel."""

    panel: pd.DataFrame
    """Columns: date, sector, instrument, holdings, source."""

    sources_used: list[str]
    """Names of contributing sources."""

    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    """Earliest and latest quarter in the panel."""


@dataclass(slots=True)
class ReconciliationReport:
    """Stock reconciliation results for each quarter in the panel."""

    quarters: pd.DataFrame
    """Columns: date, public_debt, soma_holdings, sector_total, residual_gap, gap_pct."""

    summary: dict[str, Any]
    """Aggregate statistics across all quarters."""


def build_harmonized_panel(
    z1_holdings: pd.DataFrame | None = None,
    fwtw_holdings: pd.DataFrame | None = None,
    priority: str = "z1",
) -> HarmonizedPanel:
    """Merge Z.1 and FWTW holdings into a single quarterly panel.

    When both sources cover the same sector-quarter, one is chosen based
    on the priority parameter.  Z.1 is the primary backbone because it
    covers all sectors; FWTW provides a cross-check and fills gaps.

    Parameters
    ----------
    z1_holdings : DataFrame with columns date, sector, instrument, holdings
    fwtw_holdings : DataFrame with columns date, sector, instrument, holdings
    priority : which source wins on overlap ('z1' or 'fwtw')
    """
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    if z1_holdings is not None and not z1_holdings.empty:
        z1 = z1_holdings.copy()
        z1["source"] = "z1"
        frames.append(z1)
        sources.append("z1")

    if fwtw_holdings is not None and not fwtw_holdings.empty:
        fwtw = fwtw_holdings.copy()
        fwtw["source"] = "fwtw"
        frames.append(fwtw)
        sources.append("fwtw")

    if not frames:
        return HarmonizedPanel(
            panel=pd.DataFrame(columns=["date", "sector", "instrument", "holdings", "source"]),
            sources_used=[],
            date_range=None,
        )

    combined = pd.concat(frames, ignore_index=True)

    # Filter out meta rows used for diagnostics rather than the canonical panel.
    combined = combined[~combined["sector"].isin(["_total", "_discrepancy"])].copy()

    # Resolve overlaps: keep the priority source when both exist
    if len(sources) > 1:
        combined = _resolve_overlaps(combined, priority)

    combined = combined.sort_values(["date", "sector"]).reset_index(drop=True)

    date_range = (combined["date"].min(), combined["date"].max()) if not combined.empty else None

    return HarmonizedPanel(
        panel=combined,
        sources_used=sources,
        date_range=date_range,
    )


def _resolve_overlaps(df: pd.DataFrame, priority: str) -> pd.DataFrame:
    """When both Z.1 and FWTW cover the same (date, sector, instrument), keep priority."""
    key_cols = ["date", "sector", "instrument"]
    # Mark duplicates
    df = df.copy()
    df["_priority"] = df["source"].map({"z1": 0, "fwtw": 1} if priority == "z1" else {"fwtw": 0, "z1": 1})
    df = df.sort_values(key_cols + ["_priority"])
    df = df.drop_duplicates(subset=key_cols, keep="first")
    return df.drop(columns=["_priority"])


def reconcile_panel(
    panel: HarmonizedPanel,
    debt_totals: pd.DataFrame | None = None,
) -> ReconciliationReport:
    """Run stock reconciliation for each quarter in the harmonized panel.

    Parameters
    ----------
    panel : HarmonizedPanel from build_harmonized_panel
    debt_totals : optional DataFrame with columns date, public_debt, soma_holdings.
                  If None, reconciliation reports sector totals only.
    """
    df = panel.panel
    if df.empty:
        return ReconciliationReport(
            quarters=pd.DataFrame(
                columns=["date", "public_debt", "soma_holdings", "sector_total", "residual_gap", "gap_pct"]
            ),
            summary={},
        )

    # Compute sector totals per quarter (excluding fed/SOMA)
    private_sectors = df[df["sector"] != "fed"]
    sector_totals = private_sectors.groupby("date", as_index=False)["holdings"].sum()
    sector_totals = sector_totals.rename(columns={"holdings": "sector_total"})

    # Get SOMA holdings per quarter from the panel
    soma = df[df["sector"] == "fed"].groupby("date", as_index=False)["holdings"].sum()
    soma = soma.rename(columns={"holdings": "soma_holdings"})

    merged = sector_totals.merge(soma, on="date", how="outer").fillna(0.0)

    if debt_totals is not None and not debt_totals.empty:
        # Align debt totals to quarter-end dates
        dt = debt_totals.copy()
        if "public_debt" in dt.columns:
            merged = merged.merge(dt[["date", "public_debt"]], on="date", how="left")
        else:
            merged["public_debt"] = float("nan")
    else:
        merged["public_debt"] = float("nan")

    merged["residual_gap"] = merged["public_debt"] - merged["soma_holdings"] - merged["sector_total"]
    merged["gap_pct"] = (
        merged["residual_gap"] / merged["public_debt"].replace(0, float("nan")) * 100
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    # Summary statistics
    valid = merged.dropna(subset=["residual_gap"])
    summary: dict[str, Any] = {
        "quarters_covered": len(merged),
        "quarters_with_debt_total": int(valid.shape[0]),
    }
    if not valid.empty:
        summary["mean_gap_pct"] = float(valid["gap_pct"].mean())
        summary["max_abs_gap_pct"] = float(valid["gap_pct"].abs().max())
        summary["mean_residual_gap"] = float(valid["residual_gap"].mean())

    return ReconciliationReport(quarters=merged, summary=summary)


def panel_to_wide(panel: HarmonizedPanel) -> pd.DataFrame:
    """Pivot the harmonized panel into a date x sector wide table."""
    df = panel.panel
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index="date", columns="sector", values="holdings", aggfunc="sum"
    ).sort_index()


def save_panel_csv(panel: HarmonizedPanel, dest: str | Path) -> Path:
    """Write the harmonized panel to a CSV file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.panel.to_csv(dest, index=False)
    return dest


def save_reconciliation_csv(report: ReconciliationReport, dest: str | Path) -> Path:
    """Write the reconciliation report to a CSV file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    report.quarters.to_csv(dest, index=False)
    return dest
