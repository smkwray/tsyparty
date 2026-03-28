"""Smoke tests for CLI commands — exercises the --panel-file code paths."""

import argparse

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_quarters: int = 12, tmp_path=None) -> pd.DataFrame:
    """Build a synthetic panel and optionally write to disk."""
    sectors = ["banks", "dealers", "foreigners_official", "money_market_funds"]
    rows = []
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2001-03-31")
    for i in range(n_quarters):
        date = base_date + pd.DateOffset(months=3 * i)
        for sector in sectors:
            rows.append({
                "date": date,
                "sector": sector,
                "instrument": "treasury",
                "holdings": 100.0 + rng.normal(0, 20) + i * 5,
            })
    panel = pd.DataFrame(rows)
    if tmp_path is not None:
        panel.to_csv(tmp_path / "panel.csv", index=False)
    return panel


def test_cmd_infer_smoke(tmp_path):
    """cmd_infer should run without error on a synthetic panel."""
    from tsyparty.cli import cmd_infer

    _make_panel(12, tmp_path)
    out = tmp_path / "inference_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        out=str(out),
    )
    cmd_infer(args)
    assert out.exists()
    assert (out / "counterparty_flows.csv").exists()
    assert (out / "quarter_diagnostics.json").exists()
    assert (out / "manifest.json").exists()
    assert (out / "skip_records.json").exists()
    assert (out / "baseline_matrices.csv").exists()


def test_cmd_similarity_smoke(tmp_path):
    """cmd_similarity should run without error on a synthetic panel."""
    from tsyparty.cli import cmd_similarity

    _make_panel(24, tmp_path)
    out = tmp_path / "similarity_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        out=str(out),
    )
    cmd_similarity(args)
    assert out.exists()
    assert (out / "sector_features.csv").exists()
    assert (out / "sector_distance_matrix.csv").exists()
    assert (out / "manifest.json").exists()
    assert (out / "rolling_comovement.csv").exists()
    assert (out / "rolling_correlations.csv").exists()
    assert (out / "rolling_absorption_betas.csv").exists()


def test_cmd_similarity_writes_comovement_when_context_available(tmp_path):
    """cmd_similarity should emit rolling_comovement.csv when interim context exists."""
    from tsyparty.cli import cmd_similarity

    _make_panel(24, tmp_path)
    interim = tmp_path.parent / "interim"
    interim.mkdir(exist_ok=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2001-06-30", "2001-09-30", "2001-12-31", "2002-03-31"]),
        "public_debt": [100.0, 105.0, 112.0, 120.0],
    }).to_csv(interim / "debt_totals.csv", index=False)
    pd.DataFrame({
        "date": pd.to_datetime(["2001-06-30", "2001-09-30", "2001-12-31", "2002-03-31"]),
        "delta_soma": [1.0, -2.0, 0.5, 3.0],
    }).to_csv(interim / "soma_quarterly_delta.csv", index=False)

    out = tmp_path / "similarity_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        out=str(out),
    )
    cmd_similarity(args)

    assert (out / "rolling_comovement.csv").exists()


def test_cmd_similarity_enriched_panel_writes_private_foreign_outputs(tmp_path):
    """Similarity should emit foreign-private artifacts when run on the enriched panel."""
    from tsyparty.cli import cmd_similarity

    panel = _make_panel(24, tmp_path)
    foreign_private = panel[panel["sector"] == "foreigners_official"].copy()
    foreign_private["sector"] = "foreigners_private"
    foreign_private["holdings"] = foreign_private["holdings"] * 0.4

    foreign_official = panel[panel["sector"] == "foreigners_official"].copy()
    foreign_official["holdings"] = foreign_official["holdings"] * 0.6

    enriched = pd.concat(
        [panel[panel["sector"] != "foreigners_official"], foreign_official, foreign_private],
        ignore_index=True,
    ).sort_values(["date", "sector"])
    enriched.to_csv(tmp_path / "panel_enriched.csv", index=False)

    out = tmp_path / "similarity_enriched_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel_enriched.csv"),
        derived=str(tmp_path),
        out=str(out),
    )
    cmd_similarity(args)

    assert (out / "closest_to_foreigners_private.csv").exists()


def test_cmd_similarity_no_data(tmp_path):
    """cmd_similarity with insufficient data should still write manifest + empty artifacts."""
    from tsyparty.cli import cmd_similarity
    import json

    # Only 2 sectors, 1 quarter — not enough for similarity
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2001-03-31")] * 2,
        "sector": ["banks", "dealers"],
        "instrument": ["treasury"] * 2,
        "holdings": [100.0, 200.0],
    })
    panel.to_csv(tmp_path / "panel.csv", index=False)
    out = tmp_path / "similarity_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        out=str(out),
    )
    cmd_similarity(args)

    # Must still produce output directory + manifest
    assert out.exists()
    manifest_path = out / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["status"] == "no_data"
    assert manifest["n_sectors"] == 0
    assert "manifest.json" in manifest["files_written"]
    assert (out / "sector_features.csv").exists()
    assert (out / "sector_distance_matrix.csv").exists()
    assert (out / "rolling_correlations.csv").exists()
    assert (out / "rolling_absorption_betas.csv").exists()


def test_cmd_validate_smoke(tmp_path):
    """cmd_validate should run without error on a synthetic panel (no external data)."""
    from tsyparty.cli import cmd_validate

    _make_panel(12, tmp_path)
    interim = tmp_path / "interim"
    interim.mkdir()
    out = tmp_path / "validation_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        interim=str(interim),
        out=str(out),
    )
    cmd_validate(args)
    # With no EFA/TIC data, validation prints a message but doesn't crash
    assert out.exists()


def test_cmd_enrich_foreign_smoke(tmp_path):
    """cmd_enrich_foreign should run without error on a synthetic panel."""
    from tsyparty.cli import cmd_enrich_foreign

    _make_panel(12, tmp_path)
    tic_dir = tmp_path / "tic"
    tic_dir.mkdir()
    out = tmp_path / "enrich_out"
    args = argparse.Namespace(
        panel_file=str(tmp_path / "panel.csv"),
        derived=str(tmp_path),
        tic_dir=str(tic_dir),
        out=str(out),
    )
    cmd_enrich_foreign(args)
    assert (out / "harmonized_panel_enriched.csv").exists()
    enriched = pd.read_csv(out / "harmonized_panel_enriched.csv")
    assert "foreigners_private" in enriched["sector"].values
    assert (out / "enrichment_metadata.json").exists()
