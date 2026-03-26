from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ReconciliationResult:
    public_debt: float
    soma_holdings: float
    sector_total: float
    residual_gap: float


def reconcile_public_debt(public_debt: float, soma_holdings: float, sector_total: float) -> ReconciliationResult:
    residual_gap = public_debt - soma_holdings - sector_total
    return ReconciliationResult(
        public_debt=public_debt,
        soma_holdings=soma_holdings,
        sector_total=sector_total,
        residual_gap=residual_gap,
    )


def summarize_gap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"public_debt", "soma_holdings", "sector_total"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = frame.copy()
    out["residual_gap"] = out["public_debt"] - out["soma_holdings"] - out["sector_total"]
    return out
