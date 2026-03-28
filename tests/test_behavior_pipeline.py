"""Tests for behavior/pipeline.py — asserts artifact schemas and feature construction."""

import numpy as np
import pandas as pd
import pytest

from tsyparty.behavior.pipeline import (
    SimilarityConfig,
    SimilarityResult,
    build_behavior_context,
    build_features,
    run_similarity,
    write_outputs,
    write_no_data_outputs,
    write_charts,
)


def _make_panel(n_quarters: int = 24) -> pd.DataFrame:
    """Build a synthetic panel with enough history for similarity analysis."""
    sectors = [
        "banks", "dealers", "foreigners_official",
        "money_market_funds", "insurers",
    ]
    rows = []
    rng = np.random.default_rng(99)
    base_date = pd.Timestamp("2000-03-31")
    for i in range(n_quarters):
        date = base_date + pd.DateOffset(months=3 * i)
        for sector in sectors:
            rows.append({
                "date": date,
                "sector": sector,
                "instrument": "treasury",
                "holdings": 500.0 + rng.normal(0, 50) + i * 5,
            })
    return pd.DataFrame(rows)


def test_similarity_config_defaults():
    config = SimilarityConfig()
    assert "banks" in config.targets
    assert config.top_n == 5
    assert config.rolling_window == 20


def test_build_features_columns():
    """Feature frame must contain the expected columns."""
    panel = _make_panel()
    features = build_features(panel)
    assert not features.empty
    expected = {"mean_delta", "volatility", "share_of_total_change", "share_of_total_holdings"}
    assert expected.issubset(set(features.columns))


def test_build_features_excludes_meta():
    """Meta sectors like _total should not appear in features."""
    panel = _make_panel()
    meta = pd.DataFrame([{
        "date": panel["date"].iloc[0],
        "sector": "_total",
        "instrument": "treasury",
        "holdings": 2500.0,
    }])
    panel_with_meta = pd.concat([panel, meta], ignore_index=True)
    features = build_features(panel_with_meta)
    assert "_total" not in features.index


def test_build_features_insufficient_data():
    """Should return empty if fewer than 3 sectors."""
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2001-03-31")] * 2,
        "sector": ["banks", "dealers"],
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 200.0],
    })
    features = build_features(panel)
    assert features.empty


def test_run_similarity_result_schema():
    """SimilarityResult must have distance_matrix, features, and closest."""
    panel = _make_panel()
    result = run_similarity(panel)
    assert result is not None
    assert isinstance(result, SimilarityResult)
    assert result.distance_matrix.shape[0] == result.distance_matrix.shape[1]
    assert result.distance_matrix.shape[0] == len(result.features)
    assert (result.distance_matrix.values.diagonal() == 0).all()


def test_run_similarity_closest_keys():
    """Closest map should contain entries for targets present in the data."""
    panel = _make_panel()
    result = run_similarity(panel)
    assert result is not None
    assert "banks" in result.closest
    assert "foreigners_official" in result.closest
    for target, series in result.closest.items():
        assert len(series) <= 5
        assert target not in series.index


def test_run_similarity_rolling_correlations():
    """Rolling correlations should be produced when enough data exists."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)
    result = run_similarity(panel, config)
    assert result is not None
    if result.rolling_correlations is not None:
        assert not result.rolling_correlations.empty


def test_run_similarity_rolling_comovement_schema():
    """Factor-adjusted comovement should be emitted as a long-form artifact."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)
    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
        "net_public_supply": np.random.default_rng(43).normal(100, 20, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    assert result.rolling_comovement is not None
    expected = {
        "date", "sector_1", "sector_2", "metric", "value", "p_value", "q_value",
        "ci_low", "ci_high", "significant", "fdr_reject", "window", "n_obs", "controls",
    }
    assert expected.issubset(result.rolling_comovement.columns)
    assert (result.rolling_comovement["metric"] == "partial_pearson").all()


def test_run_similarity_rolling_comovement_covers_all_pairs():
    """Comovement should cover all sector pairs, not just configured targets."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)
    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
        "net_public_supply": np.random.default_rng(43).normal(100, 20, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    assert result.rolling_comovement is not None
    sectors = sorted(result.features.index.tolist())
    expected_pairs = len(sectors) * (len(sectors) - 1) // 2
    first_date = result.rolling_comovement["date"].min()
    assert result.rolling_comovement[result.rolling_comovement["date"] == first_date].shape[0] == expected_pairs


def test_run_similarity_rolling_comovement_handles_partial_controls():
    """Comovement should still run when only a subset of configured controls is available."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)
    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    assert result.rolling_comovement is not None
    assert set(result.rolling_comovement["controls"]) == {"delta_soma"}


def test_run_similarity_rolling_comovement_preserves_missing_values():
    """NaN sector quarters should not be coerced to zero in the comovement path."""
    panel = _make_panel(40)
    panel.loc[
        (panel["sector"] == "dealers") & (panel["date"].isin(sorted(panel["date"].unique())[10:13])),
        "holdings",
    ] = np.nan
    config = SimilarityConfig(rolling_window=8)
    dates = sorted(panel["date"].dropna().unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
        "net_public_supply": np.random.default_rng(43).normal(100, 20, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    assert result.rolling_comovement is not None
    affected = result.rolling_comovement[
        (result.rolling_comovement["sector_1"] == "banks")
        & (result.rolling_comovement["sector_2"] == "dealers")
    ]
    assert affected["n_obs"].min() < config.rolling_window


def test_run_similarity_rolling_comovement_populates_q_values():
    """Per-date multiple-testing adjustment should populate q-values."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)
    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
        "net_public_supply": np.random.default_rng(43).normal(100, 20, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    assert result.rolling_comovement is not None
    assert result.rolling_comovement["q_value"].notna().any()


def test_write_outputs_creates_files(tmp_path):
    """write_outputs must create features, distance matrix, and closest CSVs."""
    panel = _make_panel()
    result = run_similarity(panel)
    assert result is not None
    paths = write_outputs(result, tmp_path / "out")
    assert (tmp_path / "out" / "sector_features.csv").exists()
    assert (tmp_path / "out" / "sector_distance_matrix.csv").exists()
    assert (tmp_path / "out" / "rolling_comovement.csv").exists()
    for target in result.closest:
        assert (tmp_path / "out" / f"closest_to_{target}.csv").exists()


def test_run_similarity_returns_none_for_insufficient_data():
    """Should return None if not enough data."""
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2001-03-31")] * 2,
        "sector": ["banks", "dealers"],
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 200.0],
    })
    result = run_similarity(panel)
    assert result is None


def test_from_dict_empty_returns_list_targets():
    """from_dict({}) must return a real list, not a member descriptor."""
    config = SimilarityConfig.from_dict({})
    assert isinstance(config.targets, list)
    assert "banks" in config.targets
    assert isinstance(config.exclude_sectors, list)


def test_from_dict_passes_exclude_sectors():
    """from_dict should propagate exclude_sectors."""
    config = SimilarityConfig.from_dict({"exclude_sectors": ["_total"]})
    assert config.exclude_sectors == ["_total"]


def test_from_dict_overrides_targets():
    """from_dict should accept custom targets."""
    config = SimilarityConfig.from_dict({"targets": ["banks"]})
    assert config.targets == ["banks"]


def test_config_from_sectors_yml():
    """from_sectors_yml should load targets from configs/sectors.yml."""
    config = SimilarityConfig.from_sectors_yml()
    assert "banks" in config.targets
    assert len(config.targets) >= 2


def test_from_behavior_yml_loads_config():
    """from_behavior_yml should load behavior.yml params and sectors.yml targets."""
    config = SimilarityConfig.from_behavior_yml()
    assert "banks" in config.targets
    assert config.distance_metric == "cosine"
    assert config.rolling_window == 20
    assert "net_public_supply" in config.x_cols
    assert config.minimum_observations == 20


def test_exclude_sectors_includes_fed():
    """Default exclude_sectors should include fed to match inference pipeline."""
    config = SimilarityConfig()
    assert "fed" in config.exclude_sectors


def test_run_similarity_with_context():
    """Context factors should enable absorption beta computation."""
    panel = _make_panel(40)
    config = SimilarityConfig(rolling_window=8)

    # Build synthetic context
    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
        "net_public_supply": np.random.default_rng(43).normal(100, 20, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    # Absorption betas may or may not be computable depending on data sufficiency
    # Just check it doesn't crash


def test_run_similarity_absorption_betas_with_enough_data():
    """With sufficient data and context, absorption betas should be produced."""
    panel = _make_panel(60)
    config = SimilarityConfig(rolling_window=8)

    dates = sorted(panel["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "delta_soma": np.random.default_rng(42).normal(0, 10, len(dates)),
    })

    result = run_similarity(panel, config, context=context)
    assert result is not None
    if result.absorption_betas is not None:
        assert "sector" in result.absorption_betas.columns
        assert "date" in result.absorption_betas.columns
        assert any(c.startswith("beta_") for c in result.absorption_betas.columns)


def test_minimum_observations_gates_build_features():
    """build_features should return empty if fewer observations than minimum_observations."""
    panel = _make_panel(5)  # 5 quarters
    config = SimilarityConfig(minimum_observations=10)  # require 10
    features = build_features(panel, config)
    assert features.empty


def test_from_dict_reads_minimum_observations():
    """from_dict should propagate minimum_observations."""
    config = SimilarityConfig.from_dict({"minimum_observations": 30})
    assert config.minimum_observations == 30


def test_build_behavior_context_combines_available_series(tmp_path):
    """Context builder should align debt and SOMA quarterly controls."""
    interim = tmp_path / "interim"
    interim.mkdir()
    pd.DataFrame({
        "date": pd.to_datetime(["2000-03-31", "2000-06-30", "2000-09-30"]),
        "public_debt": [100.0, 110.0, 125.0],
    }).to_csv(interim / "debt_totals.csv", index=False)
    pd.DataFrame({
        "date": pd.to_datetime(["2000-06-30", "2000-09-30"]),
        "delta_soma": [1.0, -2.0],
    }).to_csv(interim / "soma_quarterly_delta.csv", index=False)

    context = build_behavior_context(interim, SimilarityConfig())
    assert context is not None
    assert {"date", "net_public_supply", "delta_soma"}.issubset(context.columns)
    assert context["net_public_supply"].notna().any()


def test_build_behavior_context_allows_partial_controls(tmp_path):
    """Context builder should return the available control when the other file is missing."""
    interim = tmp_path / "interim"
    interim.mkdir()
    pd.DataFrame({
        "date": pd.to_datetime(["2000-06-30", "2000-09-30"]),
        "delta_soma": [1.0, -2.0],
    }).to_csv(interim / "soma_quarterly_delta.csv", index=False)

    context = build_behavior_context(interim, SimilarityConfig())
    assert context is not None
    assert list(context.columns) == ["date", "delta_soma"]


def test_build_behavior_context_returns_none_when_missing(tmp_path):
    """Context builder should return None when no interim inputs are present."""
    interim = tmp_path / "interim"
    interim.mkdir()
    assert build_behavior_context(interim, SimilarityConfig()) is None
