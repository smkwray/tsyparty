"""Quarterly holder-response sidecar for issuance-mix shocks.

This module keeps descriptive holder-base evidence inside
``tsyparty`` while treating QRA/maturity-tilt series as explicit input
artifacts. It does not infer exact bilateral counterparties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import datetime
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tsyparty.baseline.flows import holdings_changes_from_levels
from tsyparty.config import transaction_scale_to_quarterly_billions


COEFFICIENT_COLUMNS = [
    "sector",
    "horizon",
    "shock_col",
    "shock_scale_bn",
    "response",
    "std_error",
    "ci_low",
    "ci_high",
    "p_value",
    "n_obs",
    "r_squared",
    "outcome",
    "outcome_source",
    "controls",
    "claim_label",
    "covariance",
    "maxlags",
    "transaction_basis",
    "outcome_units",
]


@dataclass(slots=True)
class HolderResponseConfig:
    """Typed configuration for the holder-response sidecar."""

    shock_col: str = "ati_baseline_bn"
    shock_scale_bn: float = 100.0
    max_horizon_quarters: int = 4
    min_observations: int = 8
    covariance: str = "HAC"
    transaction_basis: str | None = None
    total_matches_holder_perimeter: bool = False
    preferred_transaction_columns: list[str] = field(
        default_factory=lambda: ["transactions", "treasury_transactions", "net_transactions", "net_flow"]
    )
    fallback_to_holdings_change: bool = True
    panel_units: str = "millions"
    report_units: str = "billions"
    exclude_sectors: list[str] = field(default_factory=lambda: ["_discrepancy"])
    total_sector: str = "_total"
    residual_sector: str = "_residual"
    policy_legs: list[str] = field(default_factory=lambda: ["fed"])
    claim_label: str = "reduced-form holder-base evidence"
    caveat: str = "mechanism evidence for the TDC buyer-mix channel, not a settled causal policy elasticity"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> HolderResponseConfig:
        shock = cfg.get("shock", {})
        estimation = cfg.get("estimation", {})
        outcomes = cfg.get("outcomes", {})
        sectors = cfg.get("sectors", {})
        claims = cfg.get("claims", {})
        return cls(
            shock_col=shock.get("default_column", "ati_baseline_bn"),
            shock_scale_bn=float(shock.get("scale_bn", 100.0)),
            max_horizon_quarters=int(estimation.get("max_horizon_quarters", 4)),
            min_observations=int(estimation.get("min_observations", 8)),
            covariance=estimation.get("covariance", "HAC"),
            transaction_basis=outcomes.get("transaction_basis"),
            total_matches_holder_perimeter=sectors.get("total_matches_holder_perimeter", False),
            preferred_transaction_columns=outcomes.get(
                "preferred_transaction_columns",
                ["transactions", "treasury_transactions", "net_transactions", "net_flow"],
            ),
            fallback_to_holdings_change=bool(outcomes.get("fallback_to_holdings_change", True)),
            panel_units=outcomes.get("panel_units", "millions"),
            report_units=outcomes.get("report_units", "billions"),
            exclude_sectors=sectors.get("exclude", ["_discrepancy"]),
            total_sector=sectors.get("total_sector", "_total"),
            residual_sector=sectors.get("residual_sector", "_residual"),
            policy_legs=sectors.get("policy_legs", ["fed"]),
            claim_label=claims.get("label", "reduced-form holder-base evidence"),
            caveat=claims.get(
                "caveat",
                "mechanism evidence for the TDC buyer-mix channel, not a settled causal policy elasticity",
            ),
        )

    @classmethod
    def from_yaml(cls) -> HolderResponseConfig:
        from tsyparty.config import load_yaml

        return cls.from_dict(load_yaml("configs/holder_response.yml"))


@dataclass(slots=True)
class HolderResponseResult:
    response_panel: pd.DataFrame
    coefficients: pd.DataFrame
    cumulative: pd.DataFrame
    shock_summary: dict[str, Any]
    date_range: dict[str, str]
    outcome_source: str
    residual_method: str
    controls: list[str]


def _quarter_to_date(value: Any) -> pd.Timestamp:
    text = str(value)
    if "Q" in text:
        return pd.Period(text, freq="Q").to_timestamp(how="end").normalize()
    return pd.Timestamp(value)


def load_shock_artifact(path: str | Path, shock_col: str) -> pd.DataFrame:
    """Load a quarterly shock artifact from CSV."""
    shock_path = Path(path)
    shock = pd.read_csv(shock_path)
    if shock.empty:
        raise ValueError(f"Shock artifact is empty: {shock_path}")
    if shock_col not in shock.columns:
        raise ValueError(f"Shock artifact missing column {shock_col!r}")

    if "date" in shock.columns:
        dates = pd.to_datetime(shock["date"])
    elif "quarter" in shock.columns:
        dates = shock["quarter"].map(_quarter_to_date)
    else:
        raise ValueError("Shock artifact must contain either 'date' or 'quarter'")

    out = pd.DataFrame({"date": pd.to_datetime(dates), shock_col: pd.to_numeric(shock[shock_col], errors="coerce")})
    out = out.dropna(subset=["date", shock_col])
    out["date"] = out["date"].dt.to_period("Q").dt.to_timestamp("Q")
    if out["date"].duplicated().any():
        raise ValueError("Shock artifact must have exactly one row per quarter")
    if out.empty:
        raise ValueError(f"Shock artifact has no usable values for {shock_col!r}")
    return out.sort_values("date").reset_index(drop=True)


def build_response_panel(
    panel: pd.DataFrame,
    shock: pd.DataFrame,
    config: HolderResponseConfig | None = None,
    context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Build sector-quarter outcomes aligned to a quarterly shock."""
    if config is None:
        config = HolderResponseConfig()

    if type(config.total_matches_holder_perimeter) is not bool:
        raise ValueError("total_matches_holder_perimeter must be a literal boolean")
    if config.report_units != "billions":
        raise ValueError("Holder response report_units must be billions")

    required = {"date", "sector"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    shock_work = shock.copy()
    shock_work["date"] = pd.to_datetime(shock_work["date"]).dt.to_period("Q").dt.to_timestamp("Q")
    work["date"] = work["date"].dt.to_period("Q").dt.to_timestamp("Q")
    if shock_work["date"].duplicated().any():
        raise ValueError("Shock artifact must have exactly one row per quarter")

    outcome_col = _select_outcome_column(work, config)
    if outcome_col is None:
        if "holdings" not in work.columns:
            raise ValueError("Panel needs a transaction column or holdings for fallback changes")
        group_cols = ["sector", "instrument"] if "instrument" in work.columns else ["sector"]
        work = holdings_changes_from_levels(work, group_cols=group_cols)
        quarter_number = work["date"].dt.to_period("Q").astype("int64")
        previous = quarter_number.groupby([work[col] for col in group_cols]).shift(1)
        work.loc[quarter_number - previous != 1, "delta_holdings"] = np.nan
        outcome_col = "delta_holdings"
        outcome_source = "holdings_change_proxy"
    else:
        outcome_source = "reported_transactions"

    if outcome_source == "reported_transactions":
        scale = transaction_scale_to_quarterly_billions(config.transaction_basis)
        if "transaction_basis" in work and not work["transaction_basis"].eq(config.transaction_basis).all():
            raise ValueError("Mixed or conflicting transaction_basis in holder input")
    else:
        if config.panel_units not in {"millions", "billions"}:
            raise ValueError("Holdings proxy panel_units must be millions or billions")
        scale = 0.001 if config.panel_units == "millions" else 1.0
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="raise") * scale
    excludes = set(config.exclude_sectors)
    total_frame = work[work["sector"] == config.total_sector].copy()
    sectors = work[~work["sector"].isin(excludes | {config.total_sector})].copy()
    sector_outcomes = (
        sectors.groupby(["date", "sector"], as_index=False)[outcome_col]
        .sum(min_count=1)
        .rename(columns={outcome_col: "outcome"})
    )

    residual = _build_residual_outcomes(total_frame, sector_outcomes, outcome_col, config)
    residual_method = str(residual.attrs.get("residual_method", "unavailable_no_total"))
    sector_outcomes = pd.concat([sector_outcomes, residual], ignore_index=True)

    merged = sector_outcomes.merge(shock_work, on="date", how="inner")
    if context is not None and not context.empty:
        ctx = context.copy()
        ctx["date"] = pd.to_datetime(ctx["date"])
        merged = merged.merge(ctx, on="date", how="left")

    merged["outcome_source"] = outcome_source
    merged["transaction_basis"] = config.transaction_basis if outcome_source == "reported_transactions" else "not_applicable_holdings_change"
    merged["outcome_units"] = "quarterly_billions"
    merged["sector_role"] = np.where(
        merged["sector"].isin(config.policy_legs),
        "policy_leg",
        np.where(merged["sector"] == config.residual_sector, "residual", "holder_sector"),
    )
    return merged.sort_values(["sector", "date"]).reset_index(drop=True), outcome_source, residual_method


def _select_outcome_column(panel: pd.DataFrame, config: HolderResponseConfig) -> str | None:
    for col in config.preferred_transaction_columns:
        if col in panel.columns:
            return col
    if config.fallback_to_holdings_change:
        return None
    raise ValueError("No configured transaction column found and holdings-change fallback is disabled")


def _build_residual_outcomes(
    total_frame: pd.DataFrame,
    sector_outcomes: pd.DataFrame,
    outcome_col: str,
    config: HolderResponseConfig,
) -> pd.DataFrame:
    sector_sum = sector_outcomes.groupby("date", as_index=False)["outcome"].sum(min_count=1)

    complete = sector_outcomes.groupby("date")["outcome"].count().eq(sector_outcomes["sector"].nunique())
    sector_sum.loc[~sector_sum["date"].map(complete), "outcome"] = np.nan
    if config.total_matches_holder_perimeter and not total_frame.empty and outcome_col in total_frame.columns:
        total = (
            total_frame.groupby("date", as_index=False)[outcome_col]
            .sum(min_count=1)
            .rename(columns={outcome_col: "total_outcome"})
        )
        residual = total.merge(sector_sum, on="date", how="left")
        residual["outcome"] = residual["total_outcome"] - residual["outcome"]
        method = "source_total"
    else:
        residual = sector_sum[["date"]].copy()
        residual["outcome"] = np.nan
        method = "unavailable_no_total"

    out = residual[["date", "outcome"]].copy()
    out["sector"] = config.residual_sector
    out.attrs["residual_method"] = method
    return out[["date", "sector", "outcome"]]


def estimate_holder_response(
    response_panel: pd.DataFrame,
    config: HolderResponseConfig | None = None,
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate horizon-specific sector responses to the shock."""
    if config is None:
        config = HolderResponseConfig()
    if controls is None:
        controls = []

    required = {"date", "sector", "outcome", config.shock_col}
    missing = required.difference(response_panel.columns)
    if missing:
        raise ValueError(f"Response panel missing columns: {sorted(missing)}")

    if config.covariance != "HAC":
        raise ValueError("Holder overlapping horizons require covariance=HAC")
    rows: list[dict[str, Any]] = []
    for sector, group in response_panel.sort_values("date").groupby("sector", observed=True):
        base = group.reset_index(drop=True).copy()
        for horizon in range(config.max_horizon_quarters + 1):
            frame = _horizon_frame(base, horizon, config, controls)
            if len(frame) < config.min_observations:
                rows.append(_empty_coefficient_row(sector, horizon, config, controls, base))
                continue
            if frame[config.shock_col].nunique(dropna=True) < 2 or frame["cumulative_outcome"].nunique(dropna=True) < 2:
                rows.append(_empty_coefficient_row(sector, horizon, config, controls, base, n_obs=len(frame)))
                continue
            rows.append(_estimate_one(frame, sector, horizon, config, controls, base))

    coefficients = pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)
    cumulative = coefficients[coefficients["horizon"] == config.max_horizon_quarters].copy()
    return coefficients, cumulative.reset_index(drop=True)


def _horizon_frame(base: pd.DataFrame, horizon: int, config: HolderResponseConfig, controls: list[str]) -> pd.DataFrame:
    frame = base[["date", "outcome", config.shock_col, *controls]].copy()
    frame.index = pd.DatetimeIndex(frame["date"]).to_period("Q")
    if frame.index.has_duplicates:
        raise ValueError("Response panel must have one row per sector-quarter")
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq="Q"))
    cumul = np.zeros(len(frame), dtype=float)
    valid = np.ones(len(frame), dtype=bool)
    for step in range(horizon + 1):
        shifted = frame["outcome"].shift(-step)
        cumul += shifted.fillna(0.0).to_numpy(dtype=float)
        valid &= shifted.notna().to_numpy()
    frame["cumulative_outcome"] = cumul
    return frame.loc[valid].dropna(subset=["cumulative_outcome", config.shock_col, *controls])


def _estimate_one(
    frame: pd.DataFrame,
    sector: str,
    horizon: int,
    config: HolderResponseConfig,
    controls: list[str],
    base: pd.DataFrame,
) -> dict[str, Any]:
    y = frame["cumulative_outcome"].astype(float)
    x = sm.add_constant(frame[[config.shock_col, *controls]].astype(float), has_constant="add")
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})
    coef = float(fit.params[config.shock_col])
    se = float(fit.bse[config.shock_col])
    response = coef * config.shock_scale_bn
    std_error = se * config.shock_scale_bn
    return {
        "sector": sector,
        "horizon": int(horizon),
        "shock_col": config.shock_col,
        "shock_scale_bn": config.shock_scale_bn,
        "response": response,
        "std_error": std_error,
        "ci_low": response - 1.96 * std_error,
        "ci_high": response + 1.96 * std_error,
        "p_value": float(fit.pvalues[config.shock_col]),
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "outcome": f"{horizon + 1}q_cumulative_" + (
            "treasury_transaction_flow" if "outcome_source" in base and base["outcome_source"].iloc[0] == "reported_transactions"
            else "treasury_holdings_change_proxy"
        ),
        "outcome_source": str(base["outcome_source"].iloc[0]) if "outcome_source" in base.columns else "",
        "controls": ",".join(controls),
        "claim_label": config.claim_label,
        "covariance": "HAC",
        "maxlags": max(horizon, 1),
        "transaction_basis": str(base["transaction_basis"].iloc[0]) if "transaction_basis" in base else config.transaction_basis,
        "outcome_units": "quarterly_billions",
    }


def _empty_coefficient_row(
    sector: str,
    horizon: int,
    config: HolderResponseConfig,
    controls: list[str],
    base: pd.DataFrame,
    n_obs: int = 0,
) -> dict[str, Any]:
    return {
        "sector": sector,
        "horizon": int(horizon),
        "shock_col": config.shock_col,
        "shock_scale_bn": config.shock_scale_bn,
        "response": np.nan,
        "std_error": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "n_obs": int(n_obs),
        "r_squared": np.nan,
        "outcome": f"{horizon + 1}q_cumulative_" + (
            "treasury_transaction_flow" if "outcome_source" in base and base["outcome_source"].iloc[0] == "reported_transactions"
            else "treasury_holdings_change_proxy"
        ),
        "outcome_source": str(base["outcome_source"].iloc[0]) if "outcome_source" in base.columns else "",
        "controls": ",".join(controls),
        "claim_label": config.claim_label,
        "covariance": "HAC",
        "maxlags": max(horizon, 1),
        "transaction_basis": str(base["transaction_basis"].iloc[0]) if "transaction_basis" in base else config.transaction_basis,
        "outcome_units": "quarterly_billions",
    }


def run_holder_response(
    panel: pd.DataFrame,
    shock: pd.DataFrame,
    config: HolderResponseConfig | None = None,
    context: pd.DataFrame | None = None,
    controls: list[str] | None = None,
) -> HolderResponseResult:
    if config is None:
        config = HolderResponseConfig()
    if controls is None:
        controls = []

    missing_controls = [c for c in controls if context is None or c not in context.columns]
    if missing_controls:
        raise ValueError(f"Requested controls unavailable: {missing_controls}")
    usable_controls = controls
    response_panel, outcome_source, residual_method = build_response_panel(panel, shock, config=config, context=context)
    coefficients, cumulative = estimate_holder_response(response_panel, config=config, controls=usable_controls)
    shock_values = response_panel[[config.shock_col, "date"]].drop_duplicates()[config.shock_col]
    date_values = response_panel["date"].dropna()
    return HolderResponseResult(
        response_panel=response_panel,
        coefficients=coefficients,
        cumulative=cumulative,
        shock_summary={
            "shock_col": config.shock_col,
            "shock_scale_bn": config.shock_scale_bn,
            "n_quarters": int(shock_values.shape[0]),
            "mean": float(shock_values.mean()) if not shock_values.empty else np.nan,
            "std": float(shock_values.std()) if not shock_values.empty else np.nan,
        },
        date_range={
            "min": str(pd.Timestamp(date_values.min()).date()) if not date_values.empty else "",
            "max": str(pd.Timestamp(date_values.max()).date()) if not date_values.empty else "",
        },
        outcome_source=outcome_source,
        residual_method=residual_method,
        controls=usable_controls,
    )


def write_outputs(
    result: HolderResponseResult,
    out_dir: str | Path,
    config: HolderResponseConfig | None = None,
    shock_path: str | Path | None = None,
) -> dict[str, Path]:
    if config is None:
        config = HolderResponseConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    response_panel_path = out / "holder_response_panel.csv"
    result.response_panel.to_csv(response_panel_path, index=False)
    paths["response_panel"] = response_panel_path

    coeff_path = out / "sector_response_coefficients.csv"
    result.coefficients.to_csv(coeff_path, index=False)
    paths["coefficients"] = coeff_path

    cumulative_path = out / "sector_response_cumulative.csv"
    result.cumulative.to_csv(cumulative_path, index=False)
    paths["cumulative"] = cumulative_path

    bundle = {
        "schema_version": 1,
        "pipeline": "holder_response",
        "claim_label": config.claim_label,
        "caveat": config.caveat,
        "date_range": result.date_range,
        "shock_summary": result.shock_summary,
        "shock_path": str(shock_path) if shock_path is not None else None,
        "outcome_source": result.outcome_source,
        "residual_method": result.residual_method,
        "controls": result.controls,
        "headline_horizon": config.max_horizon_quarters,
        "report_units": "billions",
        "transaction_basis": config.transaction_basis if result.outcome_source == "reported_transactions" else "not_applicable_holdings_change",
        "holdings_input_units": config.panel_units,
        "total_matches_holder_perimeter": config.total_matches_holder_perimeter,
        "covariance": "HAC",
        "maxlags_by_horizon": {str(h): max(h, 1) for h in range(config.max_horizon_quarters + 1)},
        "files_written": [
            "holder_response_panel.csv",
            "sector_response_coefficients.csv",
            "sector_response_cumulative.csv",
            "holder_response_bundle.json",
        ],
    }
    bundle_path = out / "holder_response_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    paths["bundle"] = bundle_path

    manifest = {
        "schema_version": 1,
        "pipeline": "holder_response",
        "status": "ok",
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **bundle,
    }
    manifest["files_written"] = sorted([p.name for p in paths.values()] + ["manifest.json"])
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def write_chart(
    result: HolderResponseResult,
    out_dir: str | Path,
    config: HolderResponseConfig | None = None,
) -> Path | None:
    if config is None:
        config = HolderResponseConfig()
    frame = result.cumulative.dropna(subset=["response"]).copy()
    if frame.empty:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    priority = {"banks": 0, "foreigners_official": 1, "foreigners_private": 2, "money_market_funds": 3, "dealers": 4, config.residual_sector: 99}
    frame["_sort"] = frame["sector"].map(priority).fillna(50)
    frame = frame.sort_values(["_sort", "sector"])

    colors = [
        "#B8341F" if s == "banks" else "#2D5D7B" if str(s).startswith("foreigners") else "#8B8B8B"
        for s in frame["sector"]
    ]
    yerr = np.vstack(
        [
            (frame["response"] - frame["ci_low"]).clip(lower=0).to_numpy(dtype=float),
            (frame["ci_high"] - frame["response"]).clip(lower=0).to_numpy(dtype=float),
        ]
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "holder_response_chart.png"

    source_label = (
        "Treasury transaction flows"
        if result.outcome_source == "reported_transactions"
        else "Treasury holdings-change proxy"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(frame["sector"], frame["response"], yerr=yerr, capsize=4, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Reduced-form holder-base response to bill-heavy issuance\n"
        f"{source_label}; {config.max_horizon_quarters + 1}-quarter cumulative response"
    )
    ax.set_ylabel(f"{config.max_horizon_quarters + 1}-quarter cumulative response ({config.report_units} USD)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.text(
        0.0,
        -0.24,
        "Reduced-form holder-base evidence; not exact immediate counterparties or a settled causal policy elasticity.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
