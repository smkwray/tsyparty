#!/usr/bin/env bash
set -euo pipefail

# Full reproducible pipeline for tsyparty
# Requires: virtualenv at ~/venvs/tsyparty with project installed

PYTHON="${PYTHON:-$HOME/venvs/tsyparty/bin/python}"
TSY="$PYTHON -B -m tsyparty"

echo "=== Step 1: Download public data ==="
$TSY download z1 fwtw investor_class debt_to_penny tic_slt efa soma

echo ""
echo "=== Step 2: Parse raw data ==="
$TSY parse-z1 data/raw_public/z1/z1_csv_files.zip --out data/interim
$TSY parse-fwtw data/raw_public/fwtw/fwtw_data.csv --out data/interim
$TSY parse-auction data/raw_public/auction/*.xls --out data/interim
$TSY parse-debt data/raw_public/fiscaldata/debt_to_penny_api.json --out data/interim
$TSY parse-tic data/raw_public/tic/slt1d_globl.csv --out data/interim
$TSY parse-efa --efa-dir data/raw_public/efa --out data/interim
$TSY parse-soma data/raw_public/soma/soma_holdings_page.json --out data/interim

echo ""
echo "=== Step 3: Harmonize and reconcile ==="
$TSY harmonize --interim data/interim --out data/derived

echo ""
echo "=== Step 4: Validate against EFA/TIC ==="
$TSY validate --derived data/derived --interim data/interim --out outputs/validation

echo ""
echo "=== Step 5: Descriptive baseline ==="
$TSY baseline --derived data/derived --out outputs/baseline

echo ""
echo "=== Step 6: Primary-market allocation ==="
$TSY primary-market --interim data/interim --out outputs/primary_market

echo ""
echo "=== Step 7: Counterparty inference ==="
$TSY infer --derived data/derived --out outputs/inference

echo ""
echo "=== Step 8: Enrich foreign (official/private split) ==="
$TSY enrich-foreign --derived data/derived --tic-dir data/raw_public/tic --out data/derived

echo ""
echo "=== Step 9: Behavior similarity ==="
$TSY similarity --derived data/derived --out outputs/similarity

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  data/derived/harmonized_panel.csv"
echo "  data/derived/harmonized_panel_enriched.csv"
echo "  data/derived/enrichment_metadata.json"
echo "  data/derived/reconciliation.csv"
echo "  data/interim/soma_weekly.csv"
echo "  data/interim/soma_quarterly_delta.csv"
echo "  outputs/validation/validation_summary.csv"
echo "  outputs/baseline/"
echo "  outputs/primary_market/"
echo "  outputs/inference/counterparty_flows.csv"
echo "  outputs/inference/quarter_diagnostics.json"
echo "  outputs/inference/baseline_matrices.csv"
echo "  outputs/similarity/sector_features.csv"
echo "  outputs/similarity/sector_distance_matrix.csv"
echo "  outputs/similarity/rolling_absorption_betas.csv"
