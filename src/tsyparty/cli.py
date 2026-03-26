from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsyparty.config import data_root
from tsyparty.registry import iter_sources
from tsyparty.infer.counterparty import ras_balance, sign_baseline_matrix, sparse_threshold_rebalance
from tsyparty.behavior.similarity import cosine_distance_matrix
from tsyparty.viz.charts import save_stacked_area


def cmd_show_plan(_: argparse.Namespace) -> None:
    print("tsyparty build order")
    print("1. sources -> 2. harmonize -> 3. reconcile -> 4. baseline -> 5. matrix -> 6. behavior -> 7. presentation")


def cmd_registry(args: argparse.Namespace) -> None:
    for spec in iter_sources(public_only=args.public_only):
        direct = spec.direct_url or spec.artifact_discovery or "-"
        print(f"{spec.key:28} | {spec.frequency:14} | {direct}")


def cmd_example(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    buyers = pd.Series({"banks": 50.0, "foreigners_private": 30.0, "money_market_funds": 20.0})
    sellers = pd.Series({"dealers": 40.0, "insurers": 25.0, "households_residual": 20.0, "mutual_funds_etfs": 15.0})

    prior = pd.DataFrame(
        1.0,
        index=sellers.index,
        columns=buyers.index,
    )
    support = pd.DataFrame(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        index=sellers.index,
        columns=buyers.index,
        dtype=bool,
    )

    baseline = sign_baseline_matrix(buyers, sellers, support=support)
    dense, dense_diag = ras_balance(prior, sellers, buyers, support=support)
    sparse, sparse_diag = sparse_threshold_rebalance(dense, sellers, buyers, support=support)

    baseline.to_csv(out_dir / "example_baseline_matrix.csv")
    dense.to_csv(out_dir / "example_dense_entropy_matrix.csv")
    sparse.to_csv(out_dir / "example_sparse_matrix.csv")

    stacked = dense.div(dense.sum(axis=0), axis=1).T
    save_stacked_area(stacked, "Example seller shares to buyers", out_dir / "example_seller_shares.png")

    features = pd.DataFrame(
        {
            "beta_net_public_supply": [0.9, 0.8, 0.3, -0.1],
            "beta_delta_soma": [-0.4, -0.2, 0.1, 0.2],
            "bill_share_preference": [0.2, 0.5, 0.8, 0.6],
        },
        index=["banks", "foreigners_private", "money_market_funds", "insurers"],
    )
    cosine_distance_matrix(features).to_csv(out_dir / "example_similarity_distances.csv")

    diagnostics = {
        "dense_converged": dense_diag.converged,
        "dense_iterations": dense_diag.iterations,
        "dense_row_error": dense_diag.max_abs_row_error,
        "dense_col_error": dense_diag.max_abs_col_error,
        "sparse_converged": sparse_diag.converged,
        "sparse_iterations": sparse_diag.iterations,
        "sparse_row_error": sparse_diag.max_abs_row_error,
        "sparse_col_error": sparse_diag.max_abs_col_error,
    }
    pd.Series(diagnostics).to_csv(out_dir / "example_diagnostics.csv", header=["value"])
    print(f"Wrote example outputs to {out_dir}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download public source data."""
    dest = Path(args.dest)
    sources_to_download = args.sources or ["z1", "fwtw", "investor_class"]

    for source in sources_to_download:
        if source == "z1":
            from tsyparty.ingest.fed import download_z1_current

            print(f"Downloading Z.1 CSV zip to {dest / 'z1'}")
            download_z1_current(dest / "z1")
        elif source == "fwtw":
            from tsyparty.ingest.fwtw import download_fwtw

            print(f"Downloading FWTW CSV to {dest / 'fwtw'}")
            download_fwtw(dest / "fwtw")
        elif source == "investor_class":
            from tsyparty.ingest.treasury import download_investor_class_recent

            print(f"Downloading investor-class XLS to {dest / 'auction'}")
            download_investor_class_recent(dest / "auction")
        else:
            print(f"Unknown source: {source}")


def cmd_parse_z1(args: argparse.Namespace) -> None:
    """Parse a downloaded Z.1 CSV zip."""
    from tsyparty.ingest.z1_parser import parse_z1_zip, z1_holdings_wide

    result = parse_z1_zip(args.zip_path)
    print(f"Parsed {result.source_file}: {len(result.holdings)} rows")
    if result.unmapped_series:
        print(f"  Unmapped series: {len(result.unmapped_series)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.holdings.to_csv(out / "z1_holdings_long.csv", index=False)
    wide = z1_holdings_wide(result)
    if not wide.empty:
        wide.to_csv(out / "z1_holdings_wide.csv")
    print(f"Wrote parsed Z.1 to {out}")


def cmd_parse_fwtw(args: argparse.Namespace) -> None:
    """Parse a downloaded FWTW CSV."""
    from tsyparty.ingest.fwtw import parse_fwtw_csv

    result = parse_fwtw_csv(args.csv_path)
    print(f"Parsed {result.raw_series_count} series: {len(result.holdings)} rows")
    if result.unmapped_series:
        print(f"  Unmapped series: {len(result.unmapped_series)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.holdings.to_csv(out / "fwtw_holdings.csv", index=False)
    print(f"Wrote parsed FWTW to {out}")


def cmd_parse_auction(args: argparse.Namespace) -> None:
    """Parse investor-class auction XLS files."""
    from tsyparty.ingest.auction_parser import parse_investor_class_xls

    for xls_path in args.xls_paths:
        result = parse_investor_class_xls(xls_path)
        print(f"Parsed {Path(xls_path).name}: {len(result.allotments)} auctions, instrument={result.instrument}")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.allotments.to_csv(out / f"{result.instrument}_allotments.csv", index=False)
        result.quarterly_composition.to_csv(out / f"{result.instrument}_quarterly_composition.csv", index=False)
    print(f"Wrote parsed auction data to {args.out}")


def cmd_harmonize(args: argparse.Namespace) -> None:
    """Build harmonized panel and run reconciliation."""
    from tsyparty.reconcile.harmonize import (
        build_harmonized_panel,
        reconcile_panel,
        save_panel_csv,
        save_reconciliation_csv,
    )

    interim = Path(args.interim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z1_path = interim / "z1_holdings_long.csv"
    fwtw_path = interim / "fwtw_holdings.csv"

    z1_df = pd.read_csv(z1_path, parse_dates=["date"]) if z1_path.exists() else None
    fwtw_df = pd.read_csv(fwtw_path, parse_dates=["date"]) if fwtw_path.exists() else None

    panel = build_harmonized_panel(z1_df, fwtw_df, priority=args.priority)
    save_panel_csv(panel, out / "harmonized_panel.csv")
    print(f"Harmonized panel: {len(panel.panel)} rows, sources={panel.sources_used}")

    if panel.date_range:
        print(f"  Date range: {panel.date_range[0].date()} to {panel.date_range[1].date()}")

    # Run reconciliation
    debt_path = interim / "debt_totals.csv"
    debt_df = pd.read_csv(debt_path, parse_dates=["date"]) if debt_path.exists() else None

    report = reconcile_panel(panel, debt_df)
    save_reconciliation_csv(report, out / "reconciliation.csv")
    print(f"Reconciliation: {report.summary}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsyparty")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_plan = subparsers.add_parser("show-plan", help="Print the build sequence")
    p_plan.set_defaults(func=cmd_show_plan)

    p_registry = subparsers.add_parser("registry", help="List configured sources")
    p_registry.add_argument("--public-only", action="store_true", help="List only public sources")
    p_registry.set_defaults(func=cmd_registry)

    p_example = subparsers.add_parser("example", help="Generate toy outputs from seed methods")
    p_example.add_argument("--out", default="outputs/sample", help="Output directory")
    p_example.set_defaults(func=cmd_example)

    # --- Ingest commands ---
    p_download = subparsers.add_parser("download", help="Download public source data")
    p_download.add_argument("--dest", default="data/raw_public", help="Destination directory")
    p_download.add_argument("sources", nargs="*", help="Sources to download (z1, fwtw, investor_class)")
    p_download.set_defaults(func=cmd_download)

    p_parse_z1 = subparsers.add_parser("parse-z1", help="Parse Z.1 CSV zip")
    p_parse_z1.add_argument("zip_path", help="Path to z1_csv_files.zip")
    p_parse_z1.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_z1.set_defaults(func=cmd_parse_z1)

    p_parse_fwtw = subparsers.add_parser("parse-fwtw", help="Parse FWTW CSV")
    p_parse_fwtw.add_argument("csv_path", help="Path to fwtw_data.csv")
    p_parse_fwtw.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_fwtw.set_defaults(func=cmd_parse_fwtw)

    p_parse_auction = subparsers.add_parser("parse-auction", help="Parse investor-class XLS files")
    p_parse_auction.add_argument("xls_paths", nargs="+", help="Path(s) to IC-*.xls files")
    p_parse_auction.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_auction.set_defaults(func=cmd_parse_auction)

    p_harmonize = subparsers.add_parser("harmonize", help="Build harmonized panel and reconcile")
    p_harmonize.add_argument("--interim", default="data/interim", help="Interim data directory")
    p_harmonize.add_argument("--out", default="data/derived", help="Output directory")
    p_harmonize.add_argument("--priority", default="z1", choices=["z1", "fwtw"], help="Source priority on overlap")
    p_harmonize.set_defaults(func=cmd_harmonize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
