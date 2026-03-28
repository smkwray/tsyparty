from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection


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
            sub = grp.iloc[end - window:end][["date", y_col, *x_cols]].dropna()
            if len(sub) <= len(x_cols) + 1:
                continue
            y = sub[y_col].to_numpy(dtype=float)
            x = sub[x_cols].to_numpy(dtype=float)
            if not np.isfinite(y).all() or not np.isfinite(x).all():
                continue
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


def _residualize(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """Return OLS residuals after partialing out the supplied controls."""
    if controls.empty:
        x = pd.DataFrame(index=y.index)
    else:
        x = controls
    design = sm.add_constant(x, has_constant="add")
    result = sm.OLS(y.to_numpy(dtype=float), design.to_numpy(dtype=float)).fit()
    return pd.Series(result.resid, index=y.index)


def partial_corr_row(
    y1: pd.Series,
    y2: pd.Series,
    controls: pd.DataFrame | None = None,
) -> dict[str, float | int | bool] | None:
    """Estimate a factor-adjusted Pearson correlation row for one window."""
    if controls is None:
        controls = pd.DataFrame(index=y1.index)

    df = pd.concat(
        [y1.rename("y1"), y2.rename("y2"), controls],
        axis=1,
    ).dropna()
    if df.empty:
        return None

    control_cols = [c for c in df.columns if c not in {"y1", "y2"}]
    if len(df) <= len(control_cols) + 3:
        return None

    resid_1 = _residualize(df["y1"], df[control_cols])
    resid_2 = _residualize(df["y2"], df[control_cols])
    corr = float(np.corrcoef(resid_1, resid_2)[0, 1])
    if not np.isfinite(corr):
        return None

    corr = float(np.clip(corr, -0.999999, 0.999999))
    n_obs = int(len(df))
    n_controls = len(control_cols)
    dof = n_obs - n_controls - 2
    if dof <= 0:
        return None

    denom = max(1e-12, 1.0 - corr * corr)
    t_stat = corr * np.sqrt(dof / denom)
    p_value = float(2.0 * st.t.sf(abs(t_stat), dof))

    z_value = np.arctanh(corr)
    z_se = 1.0 / np.sqrt(max(n_obs - n_controls - 3, 1))
    z_crit = float(st.norm.ppf(0.975))
    ci_low = float(np.tanh(z_value - z_crit * z_se))
    ci_high = float(np.tanh(z_value + z_crit * z_se))

    return {
        "value": corr,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(not (ci_low <= 0.0 <= ci_high)),
        "n_obs": n_obs,
    }


def rolling_partial_correlations(
    wide: pd.DataFrame,
    context: pd.DataFrame | None = None,
    x_cols: list[str] | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling factor-adjusted pairwise comovement for all sector pairs."""
    if wide.empty or wide.shape[1] < 2 or len(wide) < window:
        return pd.DataFrame(
            columns=[
                "date",
                "sector_1",
                "sector_2",
                "metric",
                "value",
                "p_value",
                "q_value",
                "ci_low",
                "ci_high",
                "significant",
                "fdr_reject",
                "window",
                "n_obs",
                "controls",
            ]
        )

    if x_cols is None:
        x_cols = []

    context_frame = pd.DataFrame(index=wide.index)
    if context is not None and not context.empty:
        indexed = context.copy()
        if "date" in indexed.columns:
            indexed = indexed.set_index("date")
        indexed.index = pd.to_datetime(indexed.index)
        available = [c for c in x_cols if c in indexed.columns]
        if available:
            context_frame = indexed[available].reindex(wide.index)
            x_cols = available
        else:
            x_cols = []

    sectors = sorted(str(col) for col in wide.columns)
    rows: list[dict[str, object]] = []
    for end in range(window - 1, len(wide)):
        window_slice = wide.iloc[end - window + 1 : end + 1]
        controls_slice = context_frame.reindex(window_slice.index)
        row_indices: list[int] = []
        p_values: list[float] = []

        for idx, sector_1 in enumerate(sectors):
            for sector_2 in sectors[idx + 1 :]:
                result = partial_corr_row(
                    window_slice[sector_1],
                    window_slice[sector_2],
                    controls_slice,
                )
                if result is None:
                    continue
                row_indices.append(len(rows))
                p_values.append(float(result["p_value"]))
                rows.append(
                    {
                        "date": pd.Timestamp(window_slice.index[-1]),
                        "sector_1": sector_1,
                        "sector_2": sector_2,
                        "metric": "partial_pearson",
                        "value": result["value"],
                        "p_value": result["p_value"],
                        "q_value": np.nan,
                        "ci_low": result["ci_low"],
                        "ci_high": result["ci_high"],
                        "significant": result["significant"],
                        "fdr_reject": False,
                        "window": int(window),
                        "n_obs": result["n_obs"],
                        "controls": ",".join(x_cols),
                    }
                )

        if p_values:
            reject, q_values = fdrcorrection(p_values, alpha=0.05, method="indep")
            for row_idx, q_value, is_reject in zip(row_indices, q_values, reject):
                rows[row_idx]["q_value"] = float(q_value)
                rows[row_idx]["fdr_reject"] = bool(is_reject)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "sector_1",
                "sector_2",
                "metric",
                "value",
                "p_value",
                "q_value",
                "ci_low",
                "ci_high",
                "significant",
                "fdr_reject",
                "window",
                "n_obs",
                "controls",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "sector_1", "sector_2"]).reset_index(drop=True)
