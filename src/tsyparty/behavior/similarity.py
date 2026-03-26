from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_features(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame - frame.mean(axis=0)
    scaled = centered / frame.std(axis=0, ddof=0).replace(0, np.nan)
    return scaled.fillna(0.0)


def cosine_distance_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    x = standardize_features(frame).to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    sim = x @ x.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=frame.index, columns=frame.index)


def closest_sectors(distance_matrix: pd.DataFrame, target: str, top_n: int = 5) -> pd.Series:
    if target not in distance_matrix.index:
        raise KeyError(target)
    series = distance_matrix.loc[target].drop(index=target).sort_values()
    return series.head(top_n)


def rolling_absorption_beta(
    frame: pd.DataFrame,
    sector_col: str = "sector",
    y_col: str = "delta_holdings",
    x_cols: list[str] | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """
    Estimate simple rolling OLS betas by sector with numpy.linalg.lstsq.

    The returned frame has one row per sector-window endpoint.
    """
    if x_cols is None:
        x_cols = ["net_public_supply", "delta_soma"]
    required = {sector_col, y_col, "date", *x_cols}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for sector, grp in frame.sort_values("date").groupby(sector_col, observed=True):
        grp = grp.reset_index(drop=True)
        if len(grp) < window:
            continue
        for end in range(window, len(grp) + 1):
            sub = grp.iloc[end - window:end]
            y = sub[y_col].to_numpy(dtype=float)
            x = sub[x_cols].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(sub)), x])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            row = {
                "sector": sector,
                "date": sub["date"].iloc[-1],
                "alpha": beta[0],
            }
            for idx, name in enumerate(x_cols, start=1):
                row[f"beta_{name}"] = beta[idx]
            rows.append(row)
    return pd.DataFrame(rows)
