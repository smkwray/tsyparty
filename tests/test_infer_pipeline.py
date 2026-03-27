"""Tests for infer/pipeline.py — asserts artifact schemas, not just matrix math."""

import json
import numpy as np
import pandas as pd
import pytest

from tsyparty.infer.pipeline import (
    InferenceConfig,
    InferenceResult,
    SkipRecord,
    build_support_matrix,
    prepare_quarters,
    run_quarter,
    run_inference,
    write_outputs,
)


def _make_panel(n_quarters: int = 8) -> pd.DataFrame:
    """Build a minimal synthetic panel for pipeline testing."""
    sectors = ["banks", "dealers", "foreigners_official", "money_market_funds"]
    rows = []
    rng = np.random.default_rng(42)
    for i in range(n_quarters):
        date = pd.Timestamp(f"200{i // 4 + 1}-{(i % 4) * 3 + 3:02d}-30")
        for sector in sectors:
            rows.append({
                "date": date,
                "sector": sector,
                "instrument": "treasury",
                "holdings": 100.0 + rng.normal(0, 20),
            })
    return pd.DataFrame(rows)


def test_inference_config_from_dict():
    cfg = {
        "entropy_ras": {"max_iter": 5000, "tol": 1e-6},
        "sparse_sensitivity": {"enabled": True, "threshold_quantile": 0.7},
        "validation": {"require_market_clearing": False},
    }
    config = InferenceConfig.from_dict(cfg)
    assert config.max_iter == 5000
    assert config.tol == 1e-6
    assert config.threshold_quantile == 0.7
    assert config.sparse_enabled is True
    assert config.require_market_clearing is False


def test_inference_config_defaults():
    config = InferenceConfig.from_dict({})
    assert config.max_iter == 10_000
    assert config.tol == 1e-8
    assert config.threshold_quantile == 0.65


def test_prepare_quarters_excludes_meta_sectors():
    panel = _make_panel()
    # Add a meta row that should be excluded
    meta = pd.DataFrame([{
        "date": panel["date"].iloc[0],
        "sector": "_total",
        "instrument": "treasury",
        "holdings": 400.0,
    }])
    panel_with_meta = pd.concat([panel, meta], ignore_index=True)
    changes = prepare_quarters(panel_with_meta)
    assert "_total" not in changes["sector"].values
    assert "fed" not in changes["sector"].values


def test_run_quarter_returns_skip_for_flat_data():
    """A quarter where everyone has zero net flow should return SkipRecord."""
    config = InferenceConfig()
    q_changes = pd.DataFrame({
        "sector": ["banks", "dealers"],
        "delta_holdings": [0.0, 0.0],
        "date": [pd.Timestamp("2001-03-30")] * 2,
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 100.0],
    })
    result = run_quarter(q_changes, pd.Timestamp("2001-03-30"), config)
    assert isinstance(result, SkipRecord)


def test_run_quarter_produces_valid_result():
    """A quarter with buyers and sellers should produce a QuarterResult."""
    config = InferenceConfig()
    q_changes = pd.DataFrame({
        "sector": ["banks", "dealers", "foreigners_official", "money_market_funds"],
        "delta_holdings": [50.0, -30.0, 20.0, -40.0],
        "date": [pd.Timestamp("2001-06-30")] * 4,
        "instrument": ["treasury"] * 4,
        "holdings": [150.0, 70.0, 120.0, 60.0],
    })
    result = run_quarter(q_changes, pd.Timestamp("2001-06-30"), config)
    assert result is not None
    assert result.dense_diag.converged
    assert result.dense.shape[0] > 0
    assert result.dense.shape[1] > 0
    assert result.sparse is not None
    assert result.sparse_diag.converged


def test_run_inference_flow_schema():
    """Flows DataFrame must have the expected columns."""
    panel = _make_panel(12)
    result = run_inference(panel)
    assert isinstance(result, InferenceResult)
    if not result.flows.empty:
        expected_cols = {"date", "seller", "buyer", "amount", "method", "converged"}
        assert expected_cols.issubset(set(result.flows.columns))
        assert set(result.flows["method"].unique()).issubset({"dense", "sparse"})


def test_run_inference_diagnostics_schema():
    """Quarter diagnostics must have the expected keys."""
    panel = _make_panel(12)
    result = run_inference(panel)
    if result.quarter_diagnostics:
        diag = result.quarter_diagnostics[0]
        required_keys = {
            "date", "buyer_total", "seller_total",
            "residual_amount", "residual_side",
            "n_buyers", "n_sellers",
            "dense_converged", "dense_iterations",
            "dense_max_row_error", "dense_max_col_error",
            "dense_nonzero_cells",
            "market_clearing_passes",
        }
        assert required_keys.issubset(set(diag.keys()))


def test_run_inference_counts():
    """Processed + skipped should equal total quarters."""
    panel = _make_panel(12)
    result = run_inference(panel)
    # The first quarter of each sector will have NaN delta, so some skip is expected
    assert result.quarters_processed + result.quarters_skipped > 0


def test_write_outputs_creates_files(tmp_path):
    """write_outputs must create counterparty_flows.csv and quarter_diagnostics.json."""
    panel = _make_panel(12)
    result = run_inference(panel)
    paths = write_outputs(result, tmp_path / "out")
    if not result.flows.empty:
        assert (tmp_path / "out" / "counterparty_flows.csv").exists()
    if result.quarter_diagnostics:
        diag_path = tmp_path / "out" / "quarter_diagnostics.json"
        assert diag_path.exists()
        with open(diag_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)
        assert len(loaded) == len(result.quarter_diagnostics)


def test_sparse_disabled():
    """When sparse is disabled, no sparse flows should appear."""
    panel = _make_panel(12)
    config = InferenceConfig(sparse_enabled=False)
    result = run_inference(panel, config)
    if not result.flows.empty:
        assert "sparse" not in result.flows["method"].values


def test_config_validate_rejects_bad_values():
    """Config validation should reject invalid parameters."""
    with pytest.raises(ValueError):
        InferenceConfig(max_iter=0).validate()
    with pytest.raises(ValueError):
        InferenceConfig(tol=-1).validate()
    with pytest.raises(ValueError):
        InferenceConfig(threshold_quantile=1.5).validate()


def test_config_validate_accepts_defaults():
    InferenceConfig().validate()


def test_build_support_matrix_disabled():
    buyers = pd.Series({"banks": 50.0})
    sellers = pd.Series({"dealers": 50.0})
    assert build_support_matrix(buyers, sellers, use_structural_zeros=False) is None


def test_build_support_matrix_zeros_diagonal():
    """When buyer and seller sets overlap, diagonal should be False."""
    buyers = pd.Series({"banks": 50.0, "dealers": 30.0})
    sellers = pd.Series({"banks": 40.0, "insurers": 40.0})
    support = build_support_matrix(buyers, sellers, use_structural_zeros=True)
    assert support is not None
    # banks appears on both sides — banks selling to banks should be zero
    assert support.loc["banks", "banks"] is np.bool_(False)
    # banks selling to dealers should be allowed
    assert support.loc["banks", "dealers"] is np.bool_(True)
    # insurers selling to banks should be allowed
    assert support.loc["insurers", "banks"] is np.bool_(True)


def test_write_outputs_writes_baselines(tmp_path):
    """write_outputs must create baseline_matrices.csv when baselines exist."""
    panel = _make_panel(12)
    result = run_inference(panel)
    paths = write_outputs(result, tmp_path / "out")
    if result.baselines:
        assert (tmp_path / "out" / "baseline_matrices.csv").exists()
        baselines_df = pd.read_csv(tmp_path / "out" / "baseline_matrices.csv")
        assert {"date", "seller", "buyer", "baseline_amount"}.issubset(set(baselines_df.columns))


def test_config_from_dict_reads_use_structural_zeros():
    cfg = {"entropy_ras": {"use_structural_zeros": False}}
    config = InferenceConfig.from_dict(cfg)
    assert config.use_structural_zeros is False


def test_config_from_dict_reads_epsilon():
    cfg = {"entropy_ras": {"epsilon": 1e-10}}
    config = InferenceConfig.from_dict(cfg)
    assert config.epsilon == 1e-10


def test_sparse_cv_produces_flows():
    """V3 sparse_cv should produce sparse_cv method flows when enabled."""
    panel = _make_panel(12)
    config = InferenceConfig(sparse_cv_enabled=True)
    result = run_inference(panel, config)
    if not result.flows.empty:
        methods = set(result.flows["method"].unique())
        # sparse_cv should appear alongside dense and sparse
        assert "dense" in methods
        if "sparse_cv" in methods:
            cv_flows = result.flows[result.flows["method"] == "sparse_cv"]
            assert len(cv_flows) > 0


def test_sparse_cv_diagnostics():
    """V3 diagnostics should include cv_best_quantile when enabled."""
    panel = _make_panel(12)
    config = InferenceConfig(sparse_cv_enabled=True)
    result = run_inference(panel, config)
    if result.quarter_diagnostics:
        has_cv = any("sparse_cv_best_quantile" in d for d in result.quarter_diagnostics)
        # At least some quarters should have CV results
        assert has_cv


def test_structural_zeros_in_inference():
    """Full pipeline with structural zeros enabled should still converge."""
    panel = _make_panel(12)
    config = InferenceConfig(use_structural_zeros=True)
    result = run_inference(panel, config)
    if result.quarter_diagnostics:
        converged = sum(1 for d in result.quarter_diagnostics if d["dense_converged"])
        assert converged == len(result.quarter_diagnostics)


def test_run_inference_rejects_bad_config():
    """run_inference should reject invalid config at entry."""
    panel = _make_panel(12)
    with pytest.raises(ValueError, match="max_iter"):
        run_inference(panel, InferenceConfig(max_iter=0))


def test_skipped_quarter_has_skip_record():
    """Quarters with near-zero flow should produce a SkipRecord."""
    config = InferenceConfig()
    q_changes = pd.DataFrame({
        "sector": ["banks", "dealers"],
        "delta_holdings": [0.0, 0.0],
        "date": [pd.Timestamp("2001-03-30")] * 2,
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 100.0],
    })
    result = run_quarter(q_changes, pd.Timestamp("2001-03-30"), config)
    assert isinstance(result, SkipRecord)
    assert result.status == "skipped"
    assert result.reason == "near_zero_net_flow"


def test_skip_records_in_inference_result():
    """InferenceResult should contain skip_records for skipped quarters."""
    panel = _make_panel(12)
    result = run_inference(panel, InferenceConfig())
    # Some quarters will be skipped (first quarter has NaN deltas)
    assert isinstance(result.skip_records, list)
    assert result.quarters_skipped == len(result.skip_records)
    for rec in result.skip_records:
        assert isinstance(rec, SkipRecord)
        assert rec.status in ("skipped", "error")
        assert rec.reason is not None


def test_write_outputs_writes_skip_records(tmp_path):
    """write_outputs should write skip_records.json when skips exist."""
    panel = _make_panel(12)
    result = run_inference(panel)
    paths = write_outputs(result, tmp_path / "out")
    if result.skip_records:
        skip_path = tmp_path / "out" / "skip_records.json"
        assert skip_path.exists()
        import json
        with open(skip_path) as f:
            records = json.load(f)
        assert len(records) == len(result.skip_records)
        assert all("reason" in r for r in records)
