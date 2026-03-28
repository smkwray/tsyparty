"""Behavior similarity pipeline.

Extracts feature construction, distance computation, and output writing
from cli.cmd_similarity into reusable, testable functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsyparty.baseline.flows import holdings_changes_from_levels
from tsyparty.behavior.similarity import (
    cosine_distance_matrix,
    closest_sectors,
    rolling_partial_correlations,
    rolling_absorption_beta,
)


@dataclass(slots=True)
class SimilarityConfig:
    """Typed configuration for the behavior similarity pipeline."""

    targets: list[str] = field(
        default_factory=lambda: ["banks", "foreigners_official", "money_market_funds"]
    )
    top_n: int = 5
    rolling_window: int = 20
    min_date: str = "2000-01-01"
    exclude_sectors: list[str] = field(
        default_factory=lambda: ["_total", "_discrepancy", "fed"]
    )
    distance_metric: str = "cosine"
    x_cols: list[str] = field(
        default_factory=lambda: ["net_public_supply", "delta_soma"]
    )
    minimum_observations: int = 20

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> SimilarityConfig:
        return cls(
            targets=cfg.get("targets", ["banks", "foreigners_official", "money_market_funds"]),
            top_n=cfg.get("top_n", 5),
            rolling_window=cfg.get("rolling_window", 20),
            min_date=cfg.get("min_date", "2000-01-01"),
            exclude_sectors=cfg.get("exclude_sectors", ["_total", "_discrepancy", "fed"]),
            distance_metric=cfg.get("distance_metric", "cosine"),
            x_cols=cfg.get("x_cols", ["net_public_supply", "delta_soma"]),
            minimum_observations=cfg.get("minimum_observations", 20),
        )

    @classmethod
    def from_sectors_yml(cls, **overrides) -> SimilarityConfig:
        """Load target groups from configs/sectors.yml analysis_groups."""
        try:
            from tsyparty.config import load_yaml
            sectors_cfg = load_yaml("configs/sectors.yml")
            groups = sectors_cfg.get("analysis_groups", {})
            # Flatten all analysis group members as targets
            targets = []
            for group_name in ("bank_like_target", "foreign_like_target"):
                targets.extend(groups.get(group_name, []))
            # Add money_market_funds if present in canonical sectors
            canonical = sectors_cfg.get("canonical_sectors", {})
            if "money_market_funds" in canonical and "money_market_funds" not in targets:
                targets.append("money_market_funds")
            if not targets:
                targets = ["banks", "foreigners_official", "money_market_funds"]
        except (FileNotFoundError, KeyError, TypeError):
            import logging
            logging.warning("Failed to load targets from configs/sectors.yml, using defaults")
            targets = ["banks", "foreigners_official", "money_market_funds"]
        return cls(targets=targets, **overrides)

    @classmethod
    def from_behavior_yml(cls, **overrides) -> SimilarityConfig:
        """Load from behavior.yml for analysis params, sectors.yml for targets."""
        # Get targets from sectors.yml
        base = cls.from_sectors_yml()
        targets = base.targets

        # Get behavior params from behavior.yml
        try:
            from tsyparty.config import load_yaml
            bcfg = load_yaml("configs/behavior.yml")
            rolling = bcfg.get("rolling_similarity", {})
            distance = bcfg.get("distance_metric", {})
        except (FileNotFoundError, KeyError):
            return cls(targets=targets, **overrides)

        return cls(
            targets=targets,
            rolling_window=rolling.get("window_quarters", 20),
            min_date=rolling.get("min_date", "2000-01-01"),
            distance_metric=distance.get("default", "cosine") if isinstance(distance, dict) else "cosine",
            x_cols=rolling.get("x_cols", ["net_public_supply", "delta_soma"]),
            minimum_observations=rolling.get("minimum_observations", 20),
            **overrides,
        )


@dataclass(slots=True)
class SimilarityResult:
    """Full similarity pipeline output."""

    features: pd.DataFrame
    distance_matrix: pd.DataFrame
    closest: dict[str, pd.Series]  # target -> closest sectors
    rolling_comovement: pd.DataFrame | None = None
    rolling_correlations: pd.DataFrame | None = None
    absorption_betas: pd.DataFrame | None = None
    date_range: dict[str, str] | None = None  # {"min": "YYYY-MM-DD", "max": "YYYY-MM-DD"}


def build_behavior_context(
    interim_dir: str | Path,
    config: SimilarityConfig | None = None,
) -> pd.DataFrame | None:
    """Build quarterly context factors for behavior estimation from interim artifacts."""
    if config is None:
        config = SimilarityConfig()

    interim_path = Path(interim_dir)
    series_frames: list[pd.DataFrame] = []

    if "net_public_supply" in config.x_cols:
        debt_path = interim_path / "debt_totals.csv"
        if debt_path.exists():
            debt = pd.read_csv(debt_path, parse_dates=["date"])
            if {"date", "public_debt"}.issubset(debt.columns):
                debt = debt.sort_values("date")[["date", "public_debt"]].copy()
                debt["net_public_supply"] = debt["public_debt"].diff()
                series_frames.append(debt[["date", "net_public_supply"]])

    if "delta_soma" in config.x_cols:
        soma_path = interim_path / "soma_quarterly_delta.csv"
        if soma_path.exists():
            soma = pd.read_csv(soma_path, parse_dates=["date"])
            if {"date", "delta_soma"}.issubset(soma.columns):
                series_frames.append(soma[["date", "delta_soma"]].copy())

    if not series_frames:
        return None

    context = series_frames[0]
    for frame in series_frames[1:]:
        context = context.merge(frame, on="date", how="outer")

    keep_cols = ["date"] + [col for col in config.x_cols if col in context.columns]
    context = context[keep_cols].sort_values("date").reset_index(drop=True)
    value_cols = [col for col in context.columns if col != "date"]
    if not value_cols:
        return None
    context = context.dropna(how="all", subset=value_cols)
    return context if not context.empty else None


def build_features(
    panel: pd.DataFrame,
    config: SimilarityConfig | None = None,
) -> pd.DataFrame:
    """Build sector-level feature frame from the harmonized panel."""
    if config is None:
        config = SimilarityConfig()

    private = panel[~panel["sector"].isin(config.exclude_sectors)].copy()

    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes = changes.dropna(subset=["delta_holdings"])

    wide = changes.pivot_table(
        index="date", columns="sector", values="delta_holdings", aggfunc="sum"
    ).fillna(0)

    wide_recent = wide.loc[wide.index >= config.min_date]
    if wide_recent.empty or wide_recent.shape[1] < 3 or wide_recent.shape[0] < config.minimum_observations:
        return pd.DataFrame()

    features = pd.DataFrame(index=wide_recent.columns)
    features["mean_delta"] = wide_recent.mean()
    features["volatility"] = wide_recent.std()
    features["share_of_total_change"] = (
        wide_recent.abs().mean() / wide_recent.abs().mean().sum()
    )

    # Holdings share from latest available levels
    levels = private.pivot_table(
        index="date", columns="sector", values="holdings", aggfunc="sum"
    )
    if not levels.empty:
        latest_levels = levels.iloc[-1]
        total = latest_levels.sum()
        if total > 0:
            features["share_of_total_holdings"] = latest_levels / total

    return features.dropna()


def run_similarity(
    panel: pd.DataFrame,
    config: SimilarityConfig | None = None,
    context: pd.DataFrame | None = None,
) -> SimilarityResult | None:
    """Run the full similarity pipeline.

    Parameters
    ----------
    panel : harmonized panel DataFrame
    config : pipeline configuration
    context : optional quarterly context factors (date, net_public_supply, delta_soma, ...).
              Joined to changes for absorption beta estimation when available.
    """
    if config is None:
        config = SimilarityConfig()

    features = build_features(panel, config)
    if features.empty or len(features) < 3:
        return None

    if config.distance_metric != "cosine":
        raise ValueError(f"Unsupported distance metric: {config.distance_metric!r}. Only 'cosine' is implemented.")
    dist = cosine_distance_matrix(features)

    closest_map: dict[str, pd.Series] = {}
    for target in config.targets:
        if target in dist.index:
            closest_map[target] = closest_sectors(dist, target, top_n=config.top_n)

    # Rolling correlations for key pairs
    private = panel[~panel["sector"].isin(config.exclude_sectors)].copy()
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes = changes.dropna(subset=["delta_holdings"])
    wide_raw = changes.pivot_table(
        index="date", columns="sector", values="delta_holdings", aggfunc="sum"
    ).sort_index()
    wide_recent = wide_raw.loc[wide_raw.index >= config.min_date]
    wide_recent_filled = wide_recent.fillna(0)

    rolling_comovement = rolling_partial_correlations(
        wide_recent,
        context=context,
        x_cols=config.x_cols,
        window=config.rolling_window,
    )
    if rolling_comovement.empty:
        rolling_comovement = None

    rolling_corr = None
    corr_pairs: list[tuple[str, str]] = []
    for i, t1 in enumerate(config.targets):
        for t2 in config.targets[i + 1:]:
            if t1 in wide_recent_filled.columns and t2 in wide_recent_filled.columns:
                corr_pairs.append((t1, t2))

    if corr_pairs:
        corr_dict: dict[str, pd.Series] = {}
        for t1, t2 in corr_pairs:
            key = f"{t1}_vs_{t2}"
            corr_dict[key] = wide_recent_filled[t1].rolling(config.rolling_window).corr(
                wide_recent_filled[t2]
            )
        rolling_corr = pd.DataFrame(corr_dict).dropna()
        if rolling_corr.empty:
            rolling_corr = None

    # Rolling absorption betas (when context factors are available)
    absorption_betas = None
    if context is not None and not context.empty:
        x_cols = [c for c in context.columns if c != "date" and c in config.x_cols]
        if x_cols:
            # Build long-form frame for rolling_absorption_beta
            sector_changes = changes.groupby(["date", "sector"], as_index=False)["delta_holdings"].sum()
            merged = sector_changes.merge(context, on="date", how="inner")
            if len(merged) >= config.rolling_window:
                try:
                    absorption_betas = rolling_absorption_beta(
                        merged,
                        sector_col="sector",
                        y_col="delta_holdings",
                        x_cols=x_cols,
                        window=config.rolling_window,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    import logging
                    logging.warning("Rolling absorption beta estimation failed: %s", exc)

    # Compute date range from the data window used
    date_range = None
    if not wide_recent.empty:
        date_range = {
            "min": str(pd.Timestamp(wide_recent.index.min()).date()),
            "max": str(pd.Timestamp(wide_recent.index.max()).date()),
        }

    return SimilarityResult(
        features=features,
        distance_matrix=dist,
        closest=closest_map,
        rolling_comovement=rolling_comovement,
        rolling_correlations=rolling_corr,
        absorption_betas=absorption_betas,
        date_range=date_range,
    )


def write_outputs(result: SimilarityResult, out_dir: Path, config: SimilarityConfig | None = None) -> dict[str, Path]:
    """Write similarity artifacts to disk. Returns paths written.

    Always writes all artifact files (empty if no data) so consumers
    can distinguish 'no data' from 'pipeline did not run'.
    """
    import datetime
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    features_path = out_dir / "sector_features.csv"
    result.features.to_csv(features_path)
    paths["features"] = features_path

    dist_path = out_dir / "sector_distance_matrix.csv"
    result.distance_matrix.to_csv(dist_path)
    paths["distance_matrix"] = dist_path

    for target, series in result.closest.items():
        p = out_dir / f"closest_to_{target}.csv"
        series.to_csv(p, header=True)
        paths[f"closest_{target}"] = p

    # Always write rolling correlations (schemaful empty CSV if None)
    comove_path = out_dir / "rolling_comovement.csv"
    if result.rolling_comovement is not None and not result.rolling_comovement.empty:
        result.rolling_comovement.to_csv(comove_path, index=False)
    else:
        pd.DataFrame(
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
        ).to_csv(comove_path, index=False)
    paths["rolling_comovement"] = comove_path

    corr_path = out_dir / "rolling_correlations.csv"
    if result.rolling_correlations is not None:
        result.rolling_correlations.to_csv(corr_path)
    else:
        pd.DataFrame(columns=["date"]).to_csv(corr_path, index=False)
    paths["rolling_correlations"] = corr_path

    # Always write absorption betas (schemaful empty CSV if None)
    beta_path = out_dir / "rolling_absorption_betas.csv"
    if result.absorption_betas is not None and not result.absorption_betas.empty:
        result.absorption_betas.to_csv(beta_path, index=False)
    else:
        pd.DataFrame(columns=["date", "sector"]).to_csv(beta_path, index=False)
    paths["absorption_betas"] = beta_path

    # Manifest
    manifest = {
        "schema_version": 1,
        "pipeline": "similarity",
        "status": "ok",
        "headline_behavior_metric": "partial_pearson",
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "date_range": result.date_range or {},
        "n_sectors": int(result.distance_matrix.shape[0]),
        "targets_configured": config.targets if config else [],
        "targets_found": list(result.closest.keys()),
        "files_written": sorted([p.name for p in paths.values()] + ["manifest.json"]),
    }
    if config is not None:
        manifest["config"] = {
            "distance_metric": config.distance_metric,
            "rolling_window": config.rolling_window,
            "min_date": config.min_date,
            "top_n": config.top_n,
            "minimum_observations": config.minimum_observations,
            "x_cols": config.x_cols,
            "exclude_sectors": config.exclude_sectors,
            "targets": config.targets,
        }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    paths["manifest"] = manifest_path

    return paths


def write_no_data_outputs(out_dir: Path, config: SimilarityConfig) -> dict[str, Path]:
    """Write empty artifact set when insufficient data for similarity analysis.

    Ensures the CLI always produces a manifest and empty files so consumers
    can distinguish 'no data' from 'pipeline did not run'.
    """
    import datetime
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Schemaful empty CSVs — headers match success-path schemas so consumers
    # get a well-typed empty DataFrame rather than a headerless file.
    features_path = out_dir / "sector_features.csv"
    pd.DataFrame(
        columns=["mean_delta", "volatility", "share_of_total_change", "share_of_total_holdings"]
    ).to_csv(features_path)
    paths["features"] = features_path

    dist_path = out_dir / "sector_distance_matrix.csv"
    pd.DataFrame().to_csv(dist_path)
    paths["distance_matrix"] = dist_path

    corr_path = out_dir / "rolling_correlations.csv"
    pd.DataFrame(columns=["date"]).to_csv(corr_path, index=False)
    paths["rolling_correlations"] = corr_path

    comove_path = out_dir / "rolling_comovement.csv"
    pd.DataFrame(
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
    ).to_csv(comove_path, index=False)
    paths["rolling_comovement"] = comove_path

    beta_path = out_dir / "rolling_absorption_betas.csv"
    pd.DataFrame(columns=["date", "sector"]).to_csv(beta_path, index=False)
    paths["absorption_betas"] = beta_path

    manifest = {
        "schema_version": 1,
        "pipeline": "similarity",
        "status": "no_data",
        "headline_behavior_metric": "partial_pearson",
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "date_range": {},
        "n_sectors": 0,
        "targets_configured": config.targets,
        "targets_found": [],
        "files_written": sorted([p.name for p in paths.values()] + ["manifest.json"]),
        "config": {
            "distance_metric": config.distance_metric,
            "rolling_window": config.rolling_window,
            "min_date": config.min_date,
            "top_n": config.top_n,
            "minimum_observations": config.minimum_observations,
            "x_cols": config.x_cols,
            "exclude_sectors": config.exclude_sectors,
            "targets": config.targets,
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    paths["manifest"] = manifest_path

    return paths


def write_charts(result: SimilarityResult, out_dir: Path, config: SimilarityConfig | None = None) -> dict[str, Path]:
    """Write similarity charts. Separated from write_outputs for testability."""
    import matplotlib
    matplotlib.use("Agg")
    from tsyparty.viz.charts import save_heatmap, save_line_chart

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    metric = config.distance_metric if config else "cosine"

    # Distance heatmap
    heatmap_path = out_dir / "sector_distance_heatmap.png"
    save_heatmap(result.distance_matrix, f"Sector Behavior Distance ({metric})", heatmap_path, label="Distance")
    paths["heatmap"] = heatmap_path

    # Rolling correlation charts
    if result.rolling_correlations is not None and not result.rolling_correlations.empty:
        corr_path = out_dir / "rolling_correlations.png"
        save_line_chart(
            result.rolling_correlations,
            "Rolling Correlations",
            corr_path,
            ylabel="Correlation",
        )
        paths["rolling_correlations_chart"] = corr_path

    if result.rolling_comovement is not None and not result.rolling_comovement.empty:
        comove_path = out_dir / "rolling_comovement.png"
        pair_labels = (
            result.rolling_comovement["sector_1"] + "_vs_" + result.rolling_comovement["sector_2"]
        )
        pivot = result.rolling_comovement.assign(pair=pair_labels).pivot_table(
            index="date",
            columns="pair",
            values="value",
        )
        if not pivot.empty:
            save_line_chart(
                pivot,
                "Rolling Factor-Adjusted Comovement",
                comove_path,
                ylabel="Partial Pearson Correlation",
            )
            paths["rolling_comovement_chart"] = comove_path

    # Absorption beta charts
    if result.absorption_betas is not None and not result.absorption_betas.empty:
        beta_path = out_dir / "rolling_absorption_betas.png"
        # Pivot betas for chart: date x sector for each beta column
        beta_cols = [c for c in result.absorption_betas.columns if c.startswith("beta_")]
        if beta_cols:
            for bcol in beta_cols:
                pivot = result.absorption_betas.pivot_table(
                    index="date", columns="sector", values=bcol
                )
                if not pivot.empty:
                    col_path = out_dir / f"rolling_{bcol}.png"
                    save_line_chart(pivot, f"Rolling {bcol}", col_path, ylabel="Beta")
                    paths[f"chart_{bcol}"] = col_path

    return paths
