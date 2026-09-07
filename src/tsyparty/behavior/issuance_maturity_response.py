"""Issuance-maturity response diagnostics for descriptive analysis.

The module estimates whether sector Treasury absorption changes when new
Treasury issuance is longer or shorter across the curve. It is intentionally
reduced-form: realized issuance WAM is not treated as a standalone causal
policy shock.
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


DEFAULT_SECTOR_GROUPS = {
    "Banks": ["bank_us_chartered", "bank_foreign_banking_offices_us", "bank_us_affiliated_areas"],
    "Foreign holders": ["foreigners_total"],
    "Money funds": ["money_market_funds"],
    "Mutual funds & ETFs": ["mutual_funds", "exchange_traded_funds", "closed_end_funds"],
    "Dealers": ["security_brokers_and_dealers"],
    "Pensions & insurers": [
        "federal_defined_benefit_pensions",
        "federal_defined_contribution_pensions",
        "private_defined_benefit_pensions",
        "private_defined_contribution_pensions",
        "state_local_employee_defined_benefit_pensions",
        "life_insurers",
        "property_casualty_insurers",
    ],
    "Households & nonprofits": [
        "households_nonprofits",
    ],
    "State/local governments": [
        "state_local_governments",
    ],
    "Nonfinancial businesses": [
        "nonfinancial_corporates",
        "nonfinancial_noncorporate_business",
    ],
    "Other domestic financials": [
        "other_financial_business",
        "holding_companies",
        "government_sponsored_enterprises",
        "asset_backed_securities_issuers",
        "credit_unions_marketable_proxy",
    ],
}


@dataclass(slots=True)
class IssuanceMaturityResponseConfig:
    trailing_expectation_quarters: int = 8
    min_expectation_quarters: int = 4
    horizons: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    share_scale: float = 100.0
    min_positive_absorption_bn: float = 1.0
    core_controls: list[str] = field(
        default_factory=lambda: [
            "issuance_volume_gap_bn",
            "dgs10_l1",
            "slope_10y2y_l1",
            "term_premium_10y_l1",
            "fed_treasury_change_l1",
            "tga_change_l1",
            "onrrp_change_l1",
            "vix_l1",
        ]
    )
    factor_controls_enabled: bool = True
    k_grid: list[int] = field(default_factory=lambda: [100, 200, 300])
    factor_count: int = 4
    min_coverage: float = 0.6
    control_universe_path: str | None = None
    min_observations: int = 40
    claim_label: str = "reduced-form maturity-response evidence"
    caveat: str = "supporting evidence about holder absorption, not a standalone causal policy elasticity"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> IssuanceMaturityResponseConfig:
        treatment = cfg.get("treatment", {})
        outcomes = cfg.get("outcomes", {})
        controls = cfg.get("controls", {})
        factors = cfg.get("factor_controls", {})
        inference = cfg.get("inference", {})
        claims = cfg.get("claims", {})
        return cls(
            trailing_expectation_quarters=int(treatment.get("trailing_expectation_quarters", 8)),
            min_expectation_quarters=int(treatment.get("min_expectation_quarters", 4)),
            horizons=[int(item) for item in outcomes.get("horizons", [0, 1, 2, 4])],
            share_scale=float(outcomes.get("share_scale", 100.0)),
            min_positive_absorption_bn=float(outcomes.get("min_positive_absorption_bn", 1.0)),
            core_controls=controls.get("core", []),
            factor_controls_enabled=bool(factors.get("enabled", True)),
            k_grid=[int(item) for item in factors.get("k_grid", [100, 200, 300])],
            factor_count=int(factors.get("factor_count", 4)),
            min_coverage=float(factors.get("min_coverage", 0.6)),
            control_universe_path=factors.get("control_universe_path"),
            min_observations=int(inference.get("min_observations", 40)),
            claim_label=claims.get("label", "reduced-form maturity-response evidence"),
            caveat=claims.get("caveat", "supporting evidence about holder absorption, not a standalone causal policy elasticity"),
        )

    @classmethod
    def from_yaml(cls) -> IssuanceMaturityResponseConfig:
        from tsyparty.config import load_yaml

        return cls.from_dict(load_yaml("configs/issuance_maturity_response.yml"))


@dataclass(slots=True)
class IssuanceMaturityResponseResult:
    treatment_panel: pd.DataFrame
    sector_flows: pd.DataFrame
    outcome_panel: pd.DataFrame
    controls_panel: pd.DataFrame
    estimates: pd.DataFrame
    placebo: pd.DataFrame
    factor_summary: pd.DataFrame
    design_summary: dict[str, Any]


def build_issuance_maturity_treatment(
    auctions: pd.DataFrame,
    config: IssuanceMaturityResponseConfig,
) -> pd.DataFrame:
    """Construct quarterly auction-weighted issuance maturity."""
    required = {"issue_date", "maturity_date"}
    missing = required.difference(auctions.columns)
    if missing:
        raise ValueError(f"Auctions missing columns: {sorted(missing)}")
    frame = auctions.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="coerce")
    frame["maturity_date"] = pd.to_datetime(frame["maturity_date"], errors="coerce")
    if not {"total_accepted", "offering_amt"}.intersection(frame.columns):
        raise ValueError("Auctions require total_accepted or offering_amt")
    amount = pd.to_numeric(frame.get("total_accepted", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if "offering_amt" in frame.columns:
        amount = amount.fillna(pd.to_numeric(frame.get("offering_amt"), errors="coerce"))
    frame["amount_bn"] = amount / 1e9
    frame["term_years"] = (frame["maturity_date"] - frame["issue_date"]).dt.days / 365.25
    frame = frame.dropna(subset=["issue_date", "amount_bn", "term_years"])
    frame = frame[(frame["amount_bn"] > 0) & (frame["term_years"] > 0)].copy()
    frame["date"] = frame["issue_date"].dt.to_period("Q").dt.to_timestamp("Q")

    def _quarter_row(group: pd.DataFrame) -> pd.Series:
        amount_sum = float(group["amount_bn"].sum())
        return pd.Series(
            {
                "issuance_wam_years": float(np.average(group["term_years"], weights=group["amount_bn"])),
                "issuance_volume_bn": amount_sum,
                "bill_share": float(group.loc[group["term_years"] <= 1.01, "amount_bn"].sum() / amount_sum),
                "coupon_share": float(group.loc[group["term_years"] > 1.01, "amount_bn"].sum() / amount_sum),
                "long_7y_plus_share": float(group.loc[group["term_years"] >= 6.5, "amount_bn"].sum() / amount_sum),
            }
        )

    out = frame.groupby("date").apply(_quarter_row, include_groups=False).reset_index()
    out = out.sort_values("date").reset_index(drop=True)
    _require_contiguous_quarters(out["date"])
    trailing = out["issuance_wam_years"].rolling(
        config.trailing_expectation_quarters,
        min_periods=config.min_expectation_quarters,
    ).mean().shift(1)
    volume_trailing = out["issuance_volume_bn"].rolling(
        config.trailing_expectation_quarters,
        min_periods=config.min_expectation_quarters,
    ).mean().shift(1)
    out["issuance_wam_expected_years"] = trailing
    out["issuance_wam_gap_years"] = out["issuance_wam_years"] - trailing
    out["issuance_volume_gap_bn"] = out["issuance_volume_bn"] - volume_trailing
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    return out


def build_sector_flows(
    z1_sector_panel: pd.DataFrame,
    sector_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate Z.1 Treasury transactions into reporting sectors."""
    if sector_groups is None:
        sector_groups = DEFAULT_SECTOR_GROUPS
    required = {"date", "sector_key", "transactions"}
    missing = required.difference(z1_sector_panel.columns)
    if missing:
        raise ValueError(f"Z.1 sector panel missing columns: {sorted(missing)}")
    z1 = z1_sector_panel.copy()
    z1["date"] = pd.to_datetime(z1["date"])
    rows: list[pd.DataFrame] = []
    for label, keys in sector_groups.items():
        sub = z1[z1["sector_key"].isin(keys)]
        grouped = sub.groupby("date", as_index=False)["transactions"].sum(min_count=1)
        grouped["sector"] = label
        grouped["transactions_bn"] = grouped["transactions"] / 1000.0
        rows.append(grouped[["date", "sector", "transactions_bn"]])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "sector"]).reset_index(drop=True)


def build_outcome_panel(
    sector_flows: pd.DataFrame,
    treatment: pd.DataFrame,
    config: IssuanceMaturityResponseConfig,
) -> pd.DataFrame:
    """Create sector and aggregate absorption-share outcomes."""
    wide = sector_flows.pivot_table(index="date", columns="sector", values="transactions_bn", aggfunc="sum").sort_index()
    sectors = list(wide.columns)
    rows: list[dict[str, Any]] = []
    for date, values in wide.iterrows():
        positive_total = float(values.clip(lower=0).sum())
        for sector in sectors:
            flow = float(values.get(sector, np.nan))
            share = np.nan if positive_total < config.min_positive_absorption_bn else flow / positive_total * config.share_scale
            rows.append(
                {
                    "date": date,
                    "outcome": f"{sector} absorption share",
                    "value": share,
                    "flow_bn": flow,
                    "positive_absorption_bn": positive_total,
                }
            )
        bank_foreign_flow = float(values.get("Banks", 0.0) + values.get("Foreign holders", 0.0))
        bank_foreign_share = np.nan if positive_total < config.min_positive_absorption_bn else bank_foreign_flow / positive_total * config.share_scale
        domestic_nonbank_sectors = [
            sector for sector in sectors
            if sector not in {"Banks", "Foreign holders", "Dealers"}
        ]
        money_nonbank_flow = float(sum(values.get(sector, 0.0) for sector in domestic_nonbank_sectors))
        money_nonbank_share = np.nan if positive_total < config.min_positive_absorption_bn else money_nonbank_flow / positive_total * config.share_scale
        rows.extend(
            [
                {
                    "date": date,
                    "outcome": "Banks + foreign holders absorption share",
                    "value": bank_foreign_share,
                    "flow_bn": bank_foreign_flow,
                    "positive_absorption_bn": positive_total,
                },
                {
                    "date": date,
                    "outcome": "Money funds + domestic nonbanks absorption share",
                    "value": money_nonbank_share,
                    "flow_bn": money_nonbank_flow,
                    "positive_absorption_bn": positive_total,
                },
                {
                    "date": date,
                    "outcome": "Banks + foreign minus money/domestic share",
                    "value": bank_foreign_share - money_nonbank_share,
                    "flow_bn": bank_foreign_flow - money_nonbank_flow,
                    "positive_absorption_bn": positive_total,
                },
            ]
        )
    panel = pd.DataFrame(rows)
    treatment_cols = ["date", "quarter", "issuance_wam_gap_years", "issuance_volume_gap_bn", "issuance_wam_years", "bill_share"]
    return panel.merge(treatment[treatment_cols], on="date", how="inner").sort_values(["outcome", "date"]).reset_index(drop=True)


def _quarterize_fred(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", value_name])
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "value" not in frame.columns:
        return pd.DataFrame(columns=["date", value_name])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[value_name] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", value_name]).sort_values("date")
    frame["date"] = frame["date"].dt.to_period("Q").dt.to_timestamp("Q")
    return frame.groupby("date", as_index=False)[value_name].last()


def build_core_controls(fred_dir: str | Path) -> pd.DataFrame:
    """Build lagged macro/Treasury controls from local FRED-style seed files."""
    root = Path(fred_dir)
    series = {
        "dgs10": "DGS10.csv",
        "dgs2": "DGS2.csv",
        "term_premium_10y": "THREEFYTP10.csv",
        "fed_treasury": "TREAST.csv",
        "tga": "WDTGAL.csv",
        "onrrp": "RRPONTSYD.csv",
        "vix": "VIXCLS.csv",
    }
    frames = [_quarterize_fred(root / filename, name) for name, filename in series.items()]
    controls = frames[0]
    for frame in frames[1:]:
        controls = controls.merge(frame, on="date", how="outer")
    if controls.empty:
        return pd.DataFrame()
    controls = controls.sort_values("date")
    controls["slope_10y2y"] = controls["dgs10"] - controls["dgs2"]
    for col in ["fed_treasury", "tga", "onrrp"]:
        controls[f"{col}_change"] = controls[col].diff()
    lag_cols = {
        "dgs10": "dgs10_l1",
        "slope_10y2y": "slope_10y2y_l1",
        "term_premium_10y": "term_premium_10y_l1",
        "fed_treasury_change": "fed_treasury_change_l1",
        "tga_change": "tga_change_l1",
        "onrrp_change": "onrrp_change_l1",
        "vix": "vix_l1",
    }
    for src, dest in lag_cols.items():
        controls[dest] = controls[src].shift(1)
    controls["quarter"] = controls["date"].dt.to_period("Q").astype(str)
    return controls[["date", "quarter", *lag_cols.values()]]


def _require_contiguous_quarters(dates: pd.Series) -> None:
    quarters = pd.to_datetime(dates).dt.to_period("Q").astype("int64")
    if not quarters.diff().dropna().eq(1).all():
        raise ValueError("Quarterly estimation requires contiguous, unique calendar quarters")


def _build_cumulative_target(values: pd.Series, horizon: int) -> pd.Series:
    target = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    valid = pd.Series(True, index=values.index)
    for step in range(horizon + 1):
        shifted = values.shift(-step)
        target = target + shifted.fillna(0.0)
        valid &= shifted.notna()
    target[~valid] = np.nan
    return target


def _build_cumulative_absorption_share(sub: pd.DataFrame, horizon: int, scale: float) -> pd.Series:
    """Cumulative sector flow share over the horizon.

    This is not the sum of quarterly percentages. It is the cumulative net
    sector flow divided by cumulative positive absorption across sectors,
    which is the object needed for a holder-base response.
    """
    _require_contiguous_quarters(sub["date"])
    numerator = _build_cumulative_target(sub["flow_bn"].astype(float), horizon)
    denominator = _build_cumulative_target(sub["positive_absorption_bn"].astype(float), horizon)
    target = numerator / denominator.replace(0.0, np.nan) * scale
    return target.replace([np.inf, -np.inf], np.nan)


def _build_lagged_absorption_share(sub: pd.DataFrame, lead: int, scale: float) -> pd.Series:
    _require_contiguous_quarters(sub["date"])
    numerator = pd.Series(np.zeros(len(sub)), index=sub.index, dtype=float)
    denominator = pd.Series(np.zeros(len(sub)), index=sub.index, dtype=float)
    valid = pd.Series(True, index=sub.index)
    for step in range(1, lead + 1):
        flow = sub["flow_bn"].shift(step)
        denom = sub["positive_absorption_bn"].shift(step)
        numerator = numerator + flow.fillna(0.0)
        denominator = denominator + denom.fillna(0.0)
        valid &= flow.notna() & denom.notna()
    target = numerator / denominator.replace(0.0, np.nan) * scale
    target[~valid] = np.nan
    return target.replace([np.inf, -np.inf], np.nan)


def _fit_lp(
    frame: pd.DataFrame,
    outcome: str,
    horizon: int,
    treatment_col: str,
    control_cols: list[str],
    model: str,
    min_observations: int,
    share_scale: float,
) -> dict[str, Any] | None:
    sub = frame[frame["outcome"] == outcome].sort_values("date").copy()
    sub["target"] = _build_cumulative_absorption_share(sub, horizon, scale=share_scale)
    cols = ["target", treatment_col, *control_cols]
    sub = sub.dropna(subset=cols)
    if len(sub) < min_observations or sub[treatment_col].nunique() < 2:
        return None
    _require_contiguous_quarters(sub["date"])
    x = sm.add_constant(sub[[treatment_col, *control_cols]].astype(float), has_constant="add")
    try:
        fit = sm.OLS(sub["target"].astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})
    except np.linalg.LinAlgError:
        return None
    beta = float(fit.params[treatment_col])
    se = float(fit.bse[treatment_col])
    return {
        "model": model,
        "outcome": outcome,
        "horizon": horizon,
        "treatment": treatment_col,
        "beta_pp_per_1y": beta,
        "std_error": se,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "p_value": float(fit.pvalues[treatment_col]),
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "controls": ",".join(control_cols),
        "covariance": "Newey-West/HAC",
        "dependent_variable": "cumulative sector net transactions / cumulative positive absorption",
    }


def estimate_local_projections(
    panel: pd.DataFrame,
    config: IssuanceMaturityResponseConfig,
    controls: pd.DataFrame | None = None,
    factor_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate local projections for all outcomes and model tiers."""
    frame = panel.copy()
    if controls is not None and not controls.empty:
        frame = frame.merge(controls.drop(columns=["quarter"], errors="ignore"), on="date", how="left")
    estimates: list[dict[str, Any]] = []
    outcomes = sorted(frame["outcome"].dropna().unique())
    models = [("no_controls", [])]
    core_controls = [col for col in config.core_controls if col in frame.columns]
    if core_controls and set(core_controls) == set(config.core_controls):
        models.append(("core_controls", core_controls))
    if factor_controls is not None and not factor_controls.empty:
        frame = frame.merge(factor_controls, on="date", how="left")
        for k in config.k_grid:
            factor_cols = [c for c in frame.columns if c.startswith(f"factor_k{k}_")]
            if factor_cols and set(core_controls) == set(config.core_controls):
                models.append((f"factor_controls_k{k}", [*core_controls, *factor_cols]))
    for outcome in outcomes:
        for horizon in config.horizons:
            for model, control_cols in models:
                result = _fit_lp(
                    frame,
                    outcome,
                    horizon,
                    "issuance_wam_gap_years",
                    control_cols,
                    model,
                    config.min_observations,
                    config.share_scale,
                )
                if result is not None:
                    estimates.append(result)
    return pd.DataFrame(estimates)


def estimate_placebos(
    panel: pd.DataFrame,
    config: IssuanceMaturityResponseConfig,
    controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Regress pre-treatment cumulative outcomes on current treatment."""
    frame = panel.copy()
    if controls is not None and not controls.empty:
        frame = frame.merge(controls.drop(columns=["quarter"], errors="ignore"), on="date", how="left")
    core_controls = [col for col in config.core_controls if col in frame.columns]
    rows: list[dict[str, Any]] = []
    if set(core_controls) != set(config.core_controls):
        return pd.DataFrame()
    for outcome in sorted(frame["outcome"].dropna().unique()):
        sub = frame[frame["outcome"] == outcome].sort_values("date").copy()
        for lead in [1, 2, 4]:
            placebo_frame = sub.copy()
            placebo_frame["target"] = _build_lagged_absorption_share(placebo_frame, lead, scale=config.share_scale)
            cols = ["target", "issuance_wam_gap_years", *core_controls]
            fit_frame = placebo_frame.dropna(subset=cols)
            if len(fit_frame) < config.min_observations or fit_frame["issuance_wam_gap_years"].nunique() < 2:
                continue
            _require_contiguous_quarters(fit_frame["date"])
            x = sm.add_constant(fit_frame[["issuance_wam_gap_years", *core_controls]].astype(float), has_constant="add")
            try:
                fit = sm.OLS(fit_frame["target"].astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": max(lead, 1)})
            except np.linalg.LinAlgError:
                continue
            beta = float(fit.params["issuance_wam_gap_years"])
            se = float(fit.bse["issuance_wam_gap_years"])
            rows.append(
                {
                    "model": f"pretrend_placebo_{lead}q",
                    "outcome": outcome,
                    "horizon": 0,
                    "placebo_lead_quarters": lead,
                    "treatment": "issuance_wam_gap_years",
                    "beta_pp_per_1y": beta,
                    "std_error": se,
                    "ci_low": beta - 1.96 * se,
                    "ci_high": beta + 1.96 * se,
                    "p_value": float(fit.pvalues["issuance_wam_gap_years"]),
                    "n_obs": int(fit.nobs),
                    "r_squared": float(fit.rsquared),
                    "controls": ",".join(core_controls),
                    "covariance": "Newey-West/HAC",
                    "dependent_variable": "pre-treatment cumulative sector absorption share",
                }
            )
    return pd.DataFrame(rows)


def build_factor_controls(
    outcome_panel: pd.DataFrame,
    control_universe_path: str | Path | None,
    config: IssuanceMaturityResponseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen high-dimensional controls and compress them to factors."""
    if not config.factor_controls_enabled:
        return pd.DataFrame(), pd.DataFrame()
    path = Path(control_universe_path) if control_universe_path is not None else None
    if path is None or not path.is_file():
        raise ValueError("Requested factor-control universe is unavailable; provide it or explicitly disable factors")
    universe = pd.read_csv(path)
    if "quarter" not in universe.columns:
        raise ValueError("Factor-control universe requires a quarter column")
    if universe["quarter"].duplicated().any():
        raise ValueError("Factor-control universe must have unique quarters")
    base = outcome_panel[["quarter", "date", "issuance_wam_gap_years"]].drop_duplicates().dropna()
    # Use the key reported outcomes for screening.
    wide_outcomes = outcome_panel.pivot_table(index="quarter", columns="outcome", values="value", aggfunc="first")
    merged = base.merge(wide_outcomes.reset_index(), on="quarter", how="left").merge(universe, on="quarter", how="left")
    feature_ids = [c for c in universe.columns if c != "quarter"]
    y_cols = [
        c for c in [
            "Banks absorption share",
            "Foreign holders absorption share",
            "Banks + foreign holders absorption share",
            "Money funds absorption share",
        ]
        if c in merged.columns
    ]
    screen_rows: list[dict[str, Any]] = []
    for feature in feature_ids:
        values = pd.to_numeric(merged[feature], errors="coerce")
        coverage = float(values.notna().mean())
        if coverage < config.min_coverage or values.nunique(dropna=True) < 3:
            continue
        correlations: list[float] = []
        for target in ["issuance_wam_gap_years", *y_cols]:
            pair = pd.concat([values, pd.to_numeric(merged[target], errors="coerce")], axis=1).dropna()
            if len(pair) >= 20 and pair.iloc[:, 0].nunique() > 2 and pair.iloc[:, 1].nunique() > 2:
                correlations.append(abs(float(pair.corr().iloc[0, 1])))
        if correlations:
            screen_rows.append({"feature_id": feature, "score": max(correlations), "coverage": coverage})
    screen = pd.DataFrame(screen_rows, columns=["feature_id", "score", "coverage"]).sort_values("score", ascending=False).reset_index(drop=True)
    if screen.empty:
        return pd.DataFrame(), screen

    factor_frame = base[["date", "quarter"]].copy()
    meta_rows: list[dict[str, Any]] = []
    for k in config.k_grid:
        selected = screen.head(k)["feature_id"].tolist()
        matrix = merged[selected].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
        std = matrix.std(ddof=0).replace(0, np.nan)
        standardized = ((matrix - matrix.mean()) / std).fillna(0.0)
        if standardized.empty:
            continue
        u, s, _ = np.linalg.svd(standardized.to_numpy(dtype=float), full_matrices=False)
        n_factors = min(config.factor_count, u.shape[1])
        for idx in range(n_factors):
            col = f"factor_k{k}_{idx + 1}"
            factor_frame[col] = u[:, idx] * s[idx]
            meta_rows.append({"k": k, "factor": col, "selected_features": len(selected), "singular_value": float(s[idx])})
    return factor_frame.drop(columns=["quarter"]), pd.DataFrame(meta_rows)


def run_issuance_maturity_response(
    auctions: pd.DataFrame,
    z1_sector_panel: pd.DataFrame,
    config: IssuanceMaturityResponseConfig,
    fred_dir: str | Path | None = None,
    control_universe_path: str | Path | None = None,
) -> IssuanceMaturityResponseResult:
    treatment = build_issuance_maturity_treatment(auctions, config)
    flows = build_sector_flows(z1_sector_panel)
    outcomes = build_outcome_panel(flows, treatment, config)
    controls = build_core_controls(fred_dir) if fred_dir is not None else pd.DataFrame()
    if not controls.empty:
        outcomes = outcomes.merge(treatment[["date"]], on="date", how="inner")
    factor_controls, factor_summary = build_factor_controls(
        outcomes,
        control_universe_path or config.control_universe_path,
        config,
    )
    estimates = estimate_local_projections(outcomes, config, controls=controls, factor_controls=factor_controls)
    placebo = estimate_placebos(outcomes, config, controls=controls)
    design_summary = {
        "treatment": "auction-weighted issuance WAM gap versus trailing expectation",
        "outcome": "sector Treasury transaction absorption shares",
        "share_units": "percentage points",
        "sample": {
            "start": str(pd.Timestamp(outcomes["date"].min()).date()) if not outcomes.empty else "",
            "end": str(pd.Timestamp(outcomes["date"].max()).date()) if not outcomes.empty else "",
            "quarters": int(outcomes["date"].nunique()) if not outcomes.empty else 0,
        },
        "core_controls": [c for c in config.core_controls if c in outcomes.columns or c in controls.columns],
        "missing_core_controls": [c for c in config.core_controls if c not in outcomes.columns and c not in controls.columns],
        "factor_controls_enabled": bool(not factor_controls.empty),
        "claim_label": config.claim_label,
        "caveat": config.caveat,
    }
    return IssuanceMaturityResponseResult(
        treatment_panel=treatment,
        sector_flows=flows,
        outcome_panel=outcomes,
        controls_panel=controls,
        estimates=estimates,
        placebo=placebo,
        factor_summary=factor_summary,
        design_summary=design_summary,
    )


def write_outputs(result: IssuanceMaturityResponseResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in [
        ("quarterly_issuance_maturity.csv", result.treatment_panel),
        ("sector_transaction_groups.csv", result.sector_flows),
        ("maturity_response_panel.csv", result.outcome_panel),
        ("core_controls.csv", result.controls_panel),
        ("maturity_response_estimates.csv", result.estimates),
        ("maturity_response_placebos.csv", result.placebo),
        ("factor_control_summary.csv", result.factor_summary),
    ]:
        path = out / name
        frame.to_csv(path, index=False)
        paths[name] = path
    bundle = {
        "schema_version": 1,
        "pipeline": "issuance_maturity_response",
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **result.design_summary,
        "files_written": sorted(paths),
    }
    bundle_path = out / "maturity_response_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    paths["maturity_response_bundle.json"] = bundle_path
    return paths


def write_charts(result: IssuanceMaturityResponseResult, out_dir: str | Path) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    bg = "#FAF7F2"
    ink = "#162033"
    muted = "#68707D"
    red = "#B8341F"
    blue = "#2D5D7B"
    green = "#126C43"
    gray = "#8A8175"
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg})

    def clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E5DED2", linewidth=0.8)
        ax.set_axisbelow(True)

    treatment = result.treatment_panel.dropna(subset=["issuance_wam_years"])
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.plot(treatment["date"], treatment["issuance_wam_years"], color=blue, linewidth=2.5)
    ax.plot(treatment["date"], treatment["issuance_wam_expected_years"], color=gray, linewidth=2.0)
    ax.set_title("Treasury issuance moves along the maturity curve", loc="left", fontsize=17, fontweight="bold", color=ink)
    ax.text(0, 1.01, "Auction-weighted maturity of newly issued marketable Treasury securities", transform=ax.transAxes, color=muted)
    ax.set_ylabel("Years to maturity")
    clean(ax)
    path = out / "issuance_weighted_average_maturity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["issuance_wam_chart"] = path

    estimates = result.estimates
    if not estimates.empty:
        subset = estimates[
            (estimates["model"] == "core_controls")
            & (estimates["horizon"] == max(result.estimates["horizon"]))
            & estimates["outcome"].isin(
                [
                    "Banks absorption share",
                    "Foreign holders absorption share",
                    "Money funds absorption share",
                    "Mutual funds & ETFs absorption share",
                    "Dealers absorption share",
                    "Pensions & insurers absorption share",
                    "Households & nonprofits absorption share",
                    "State/local governments absorption share",
                    "Nonfinancial businesses absorption share",
                    "Other domestic financials absorption share",
                ]
            )
        ].copy()
        order = [
            "Banks absorption share",
            "Foreign holders absorption share",
            "Money funds absorption share",
            "Mutual funds & ETFs absorption share",
            "Dealers absorption share",
            "Pensions & insurers absorption share",
            "Households & nonprofits absorption share",
            "State/local governments absorption share",
            "Nonfinancial businesses absorption share",
            "Other domestic financials absorption share",
        ]
        subset["order"] = subset["outcome"].map({name: idx for idx, name in enumerate(order)})
        subset = subset.sort_values("order")
        labels = [x.replace(" absorption share", "") for x in subset["outcome"]]
        x = subset["beta_pp_per_1y"]
        xerr = np.vstack([(x - subset["ci_low"]).clip(lower=0), (subset["ci_high"] - x).clip(lower=0)])
        colors = [red if "Banks" in label else blue if "Foreign" in label else green if "Money" in label else gray for label in labels]
        fig, ax = plt.subplots(figsize=(12, 7.2))
        ax.barh(labels[::-1], x.iloc[::-1], xerr=xerr[:, ::-1], capsize=4, color=colors[::-1])
        ax.axvline(0, color=ink, linewidth=0.9)
        fig.text(0.24, 0.965, "Who absorbs more when issuance gets longer?", fontsize=17, fontweight="bold", color=ink)
        fig.text(
            0.24,
            0.925,
            "Sector absorption-share response to a one-year longer-than-expected issuance maturity; core controls included",
            fontsize=10.5,
            color=muted,
        )
        ax.set_xlabel("Percentage points of Treasury absorption")
        ax.grid(axis="x", color="#E5DED2", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        path = out / "sector_absorption_share_response.png"
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.10, top=0.88)
        fig.savefig(path, dpi=220)
        split_path = out / "sector_absorption_share_response_split.png"
        fig.savefig(split_path, dpi=220)
        plt.close(fig)
        paths["sector_response_chart"] = path
        paths["sector_response_split_chart"] = split_path
    return paths
