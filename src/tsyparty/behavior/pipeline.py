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
        default_factory=lambda: ["_total", "_discrepancy"]
    )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> SimilarityConfig:
        return cls(
            targets=cfg.get("targets", cls.targets),
            top_n=cfg.get("top_n", 5),
            rolling_window=cfg.get("rolling_window", 20),
            min_date=cfg.get("min_date", "2000-01-01"),
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
        except Exception:
            targets = ["banks", "foreigners_official", "money_market_funds"]
        return cls(targets=targets, **overrides)


@dataclass(slots=True)
class SimilarityResult:
    """Full similarity pipeline output."""

    features: pd.DataFrame
    distance_matrix: pd.DataFrame
    closest: dict[str, pd.Series]  # target -> closest sectors
    rolling_correlations: pd.DataFrame | None = None
    absorption_betas: pd.DataFrame | None = None


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
    if wide_recent.empty or wide_recent.shape[1] < 3:
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

    dist = cosine_distance_matrix(features)

    closest_map: dict[str, pd.Series] = {}
    for target in config.targets:
        if target in dist.index:
            closest_map[target] = closest_sectors(dist, target, top_n=config.top_n)

    # Rolling correlations for key pairs
    private = panel[~panel["sector"].isin(config.exclude_sectors)].copy()
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes = changes.dropna(subset=["delta_holdings"])
    wide = changes.pivot_table(
        index="date", columns="sector", values="delta_holdings", aggfunc="sum"
    ).fillna(0)
    wide_recent = wide.loc[wide.index >= config.min_date]

    rolling_corr = None
    corr_pairs: list[tuple[str, str]] = []
    for i, t1 in enumerate(config.targets):
        for t2 in config.targets[i + 1:]:
            if t1 in wide_recent.columns and t2 in wide_recent.columns:
                corr_pairs.append((t1, t2))

    if corr_pairs:
        corr_dict: dict[str, pd.Series] = {}
        for t1, t2 in corr_pairs:
            key = f"{t1}_vs_{t2}"
            corr_dict[key] = wide_recent[t1].rolling(config.rolling_window).corr(
                wide_recent[t2]
            )
        rolling_corr = pd.DataFrame(corr_dict).dropna()
        if rolling_corr.empty:
            rolling_corr = None

    # Rolling absorption betas (when context factors are available)
    absorption_betas = None
    if context is not None and not context.empty:
        x_cols = [c for c in context.columns if c != "date" and c in ("net_public_supply", "delta_soma")]
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
                except (ValueError, np.linalg.LinAlgError):
                    pass

    return SimilarityResult(
        features=features,
        distance_matrix=dist,
        closest=closest_map,
        rolling_correlations=rolling_corr,
        absorption_betas=absorption_betas,
    )


def write_outputs(result: SimilarityResult, out_dir: Path) -> dict[str, Path]:
    """Write similarity artifacts to disk. Returns paths written."""
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

    if result.rolling_correlations is not None:
        corr_path = out_dir / "rolling_correlations.csv"
        result.rolling_correlations.to_csv(corr_path)
        paths["rolling_correlations"] = corr_path

    if result.absorption_betas is not None and not result.absorption_betas.empty:
        beta_path = out_dir / "rolling_absorption_betas.csv"
        result.absorption_betas.to_csv(beta_path, index=False)
        paths["absorption_betas"] = beta_path

    return paths


def write_charts(result: SimilarityResult, out_dir: Path) -> dict[str, Path]:
    """Write similarity charts. Separated from write_outputs for testability."""
    import matplotlib
    matplotlib.use("Agg")
    from tsyparty.viz.charts import save_heatmap, save_line_chart

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Distance heatmap
    heatmap_path = out_dir / "sector_distance_heatmap.png"
    save_heatmap(result.distance_matrix, "Sector Behavior Distance (cosine)", heatmap_path, label="Distance")
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
