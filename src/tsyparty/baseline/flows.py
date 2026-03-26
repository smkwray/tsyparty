from __future__ import annotations

import pandas as pd


def holdings_changes_from_levels(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = frame.sort_values(group_cols + ["date"]).copy()
    out["delta_holdings"] = out.groupby(group_cols, observed=True)["holdings"].diff()
    return out


def buyer_seller_margins(frame: pd.DataFrame, value_col: str = "net_flow") -> tuple[pd.Series, pd.Series]:
    if "sector" not in frame.columns:
        raise ValueError("frame must contain a 'sector' column")
    series = frame.set_index("sector")[value_col].astype(float)
    buyers = series[series > 0]
    sellers = -series[series < 0]
    return buyers, sellers
