"""Gold contract tests — assert manifest self-consistency and deterministic file sets.

Each pipeline gets a success-path and a no-data path test that asserts:
  - Expected files exist on disk
  - manifest.json lists itself in files_written
  - files_written matches actual files (no phantom entries, no missing entries)
  - Manifest fields have expected types and values
"""

import json

import numpy as np
import pandas as pd
import pytest

from tsyparty.infer.pipeline import InferenceConfig, run_inference, write_outputs as infer_write
from tsyparty.behavior.pipeline import (
    SimilarityConfig,
    run_similarity,
    write_outputs as sim_write,
    write_no_data_outputs as sim_write_no_data,
)


def _make_panel(n_quarters: int = 24, n_sectors: int = 5) -> pd.DataFrame:
    sectors = ["banks", "dealers", "foreigners_official", "money_market_funds", "insurers"][:n_sectors]
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


# ---------------------------------------------------------------------------
# Inference contracts
# ---------------------------------------------------------------------------

class TestInferenceContract:
    """Gold contract tests for the inference pipeline."""

    def test_success_path_files(self, tmp_path):
        """Success path must produce all expected artifact files."""
        panel = _make_panel(12, 4)
        result = run_inference(panel)
        config = InferenceConfig()
        out = tmp_path / "infer"
        paths = infer_write(result, out, config=config)

        expected_files = {
            "counterparty_flows.csv",
            "baseline_matrices.csv",
            "quarter_diagnostics.json",
            "skip_records.json",
            "manifest.json",
        }
        actual_files = {p.name for p in out.iterdir()}
        # Validation CSVs are conditional, so allow extras
        assert expected_files.issubset(actual_files)

    def test_success_path_manifest_self_consistency(self, tmp_path):
        """Manifest must list itself in files_written and match actual disk files."""
        panel = _make_panel(12, 4)
        result = run_inference(panel)
        config = InferenceConfig()
        out = tmp_path / "infer"
        infer_write(result, out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        # Manifest lists itself
        assert "manifest.json" in manifest["files_written"]

        # files_written matches what's on disk (excluding charts/PNGs)
        disk_files = sorted(p.name for p in out.iterdir() if p.suffix != ".png")
        assert manifest["files_written"] == disk_files

    def test_success_path_manifest_fields(self, tmp_path):
        """Manifest must contain expected top-level keys and config block."""
        panel = _make_panel(12, 4)
        result = run_inference(panel)
        config = InferenceConfig()
        out = tmp_path / "infer"
        infer_write(result, out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["schema_version"] == 1
        assert manifest["pipeline"] == "inference"
        assert "build_timestamp" in manifest
        assert isinstance(manifest["quarters_processed"], int)
        assert isinstance(manifest["quarters_skipped"], int)
        assert manifest["claims_label"] == "likely_net_counterparties"

        cfg = manifest["config"]
        assert "max_iter" in cfg
        assert "sparse_cv_quantiles" in cfg
        assert "require_market_clearing" in cfg
        assert "claims_label" in cfg
        assert "compare_to_fwtw_levels" in cfg

    def test_empty_flows_still_produces_all_files(self, tmp_path):
        """Even with zero processable quarters, all core files are written."""
        # Panel with one quarter — will produce NaN deltas → skip
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2001-03-31")] * 3,
            "sector": ["banks", "dealers", "insurers"],
            "instrument": ["treasury"] * 3,
            "holdings": [100.0, 200.0, 300.0],
        })
        result = run_inference(panel)
        out = tmp_path / "infer"
        infer_write(result, out)

        assert (out / "counterparty_flows.csv").exists()
        assert (out / "baseline_matrices.csv").exists()
        assert (out / "quarter_diagnostics.json").exists()
        assert (out / "skip_records.json").exists()
        assert (out / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Similarity contracts
# ---------------------------------------------------------------------------

class TestSimilarityContract:
    """Gold contract tests for the similarity pipeline."""

    def test_success_path_files(self, tmp_path):
        """Success path must produce all expected artifact files."""
        panel = _make_panel(24)
        config = SimilarityConfig()
        result = run_similarity(panel, config)
        assert result is not None
        out = tmp_path / "sim"
        sim_write(result, out, config=config)

        expected_files = {
            "sector_features.csv",
            "sector_distance_matrix.csv",
            "rolling_comovement.csv",
            "rolling_correlations.csv",
            "rolling_absorption_betas.csv",
            "manifest.json",
        }
        actual_files = {p.name for p in out.iterdir()}
        assert expected_files.issubset(actual_files)

    def test_success_path_manifest_self_consistency(self, tmp_path):
        """Manifest must list itself and match actual disk files."""
        panel = _make_panel(24)
        config = SimilarityConfig()
        result = run_similarity(panel, config)
        assert result is not None
        out = tmp_path / "sim"
        sim_write(result, out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        assert "manifest.json" in manifest["files_written"]

        disk_files = sorted(p.name for p in out.iterdir() if p.suffix != ".png")
        assert manifest["files_written"] == disk_files

    def test_success_path_manifest_fields(self, tmp_path):
        """Manifest must contain expected top-level keys and enriched config."""
        panel = _make_panel(24)
        config = SimilarityConfig()
        result = run_similarity(panel, config)
        assert result is not None
        out = tmp_path / "sim"
        sim_write(result, out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["schema_version"] == 1
        assert manifest["pipeline"] == "similarity"
        assert manifest["status"] == "ok"
        assert manifest["headline_behavior_metric"] == "partial_pearson"
        assert isinstance(manifest["n_sectors"], int)
        assert manifest["n_sectors"] > 0
        # date_range present and populated on success path
        assert isinstance(manifest["date_range"], dict)
        assert "min" in manifest["date_range"]
        assert "max" in manifest["date_range"]
        # Configured vs found targets
        assert isinstance(manifest["targets_configured"], list)
        assert isinstance(manifest["targets_found"], list)
        assert set(manifest["targets_found"]).issubset(set(manifest["targets_configured"]))

        cfg = manifest["config"]
        assert "targets" in cfg
        assert "top_n" in cfg
        assert "min_date" in cfg
        assert "minimum_observations" in cfg
        assert "distance_metric" in cfg

    def test_no_data_path_writes_manifest(self, tmp_path):
        """No-data path must produce manifest with status=no_data."""
        config = SimilarityConfig()
        out = tmp_path / "sim_empty"
        sim_write_no_data(out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["status"] == "no_data"
        assert manifest["headline_behavior_metric"] == "partial_pearson"
        assert manifest["date_range"] == {}
        assert manifest["n_sectors"] == 0
        assert manifest["targets_found"] == []
        assert manifest["targets_configured"] == config.targets
        assert "manifest.json" in manifest["files_written"]

    def test_no_data_path_files(self, tmp_path):
        """No-data path must still produce all structural files."""
        config = SimilarityConfig()
        out = tmp_path / "sim_empty"
        sim_write_no_data(out, config=config)

        expected = {
            "sector_features.csv",
            "sector_distance_matrix.csv",
            "rolling_comovement.csv",
            "rolling_correlations.csv",
            "rolling_absorption_betas.csv",
            "manifest.json",
        }
        actual = {p.name for p in out.iterdir()}
        assert expected == actual

    def test_no_data_manifest_matches_disk(self, tmp_path):
        """No-data manifest files_written should match actual files on disk."""
        config = SimilarityConfig()
        out = tmp_path / "sim_empty"
        sim_write_no_data(out, config=config)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        disk_files = sorted(p.name for p in out.iterdir())
        assert manifest["files_written"] == disk_files

    def test_no_data_csvs_have_schema_headers(self, tmp_path):
        """No-data CSVs should have column headers matching success-path schemas."""
        config = SimilarityConfig()
        out = tmp_path / "sim_empty"
        sim_write_no_data(out, config=config)

        features = pd.read_csv(out / "sector_features.csv", index_col=0)
        assert set(features.columns) == {"mean_delta", "volatility", "share_of_total_change", "share_of_total_holdings"}
        assert len(features) == 0

        betas = pd.read_csv(out / "rolling_absorption_betas.csv")
        assert "date" in betas.columns
        assert "sector" in betas.columns
        assert len(betas) == 0

        corr = pd.read_csv(out / "rolling_correlations.csv")
        assert "date" in corr.columns
        assert len(corr) == 0
