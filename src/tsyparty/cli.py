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


def cmd_baseline(args: argparse.Namespace) -> None:
    """Compute descriptive baseline outputs from harmonized panel."""
    from tsyparty.baseline.flows import holdings_changes_from_levels, buyer_seller_margins
    from tsyparty.viz.charts import save_stacked_area, save_line_chart

    derived = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(derived / "harmonized_panel.csv", parse_dates=["date"])
    private = panel[~panel["sector"].isin(["_total", "_discrepancy"])].copy()

    # 1. Quarterly holding changes
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes.to_csv(out / "holding_changes.csv", index=False)
    recent = changes.dropna(subset=["delta_holdings"])
    print(f"Holding changes: {len(recent)} rows")

    # 2. Pivot to wide for charts
    wide_changes = recent.pivot_table(
        index="date", columns="sector", values="delta_holdings", aggfunc="sum"
    ).sort_index().fillna(0)
    # Keep only recent history for readability (post-2000)
    wide_recent = wide_changes.loc[wide_changes.index >= "2000-01-01"]
    if not wide_recent.empty:
        save_line_chart(wide_recent, "Quarterly Holding Changes by Sector", out / "holding_changes.png", ylabel="Millions USD")
        print(f"  Chart: {out / 'holding_changes.png'}")

    # 3. Latest-quarter buyer/seller margins
    latest_date = recent["date"].max()
    latest = recent[recent["date"] == latest_date].groupby("sector", as_index=False)["delta_holdings"].sum()
    latest = latest.rename(columns={"delta_holdings": "net_flow"})
    buyers, sellers = buyer_seller_margins(latest)
    margins = pd.DataFrame({
        "buyers": buyers.reindex(latest["sector"]).fillna(0),
        "sellers": sellers.reindex(latest["sector"]).fillna(0),
    })
    margins.to_csv(out / "latest_margins.csv")
    print(f"Latest quarter ({latest_date.date()}): {len(buyers)} buyers, {len(sellers)} sellers")

    # 4. Primary-market composition (if auction data exists)
    for instrument in ("bills", "nominal_coupons"):
        comp_path = derived.parent / "interim" / f"{instrument}_quarterly_composition.csv"
        if comp_path.exists():
            comp = pd.read_csv(comp_path, parse_dates=["date"])
            comp_wide = comp.pivot_table(index="date", columns="buyer_class", values="share", aggfunc="sum").fillna(0)
            comp_recent = comp_wide.loc[comp_wide.index >= "2010-01-01"] if len(comp_wide) > 40 else comp_wide
            if not comp_recent.empty:
                save_stacked_area(comp_recent, f"Primary Market Buyer Shares ({instrument})", out / f"primary_market_{instrument}.png")
                print(f"  Chart: {out / f'primary_market_{instrument}.png'}")

    # 5. Seller shares to banks and foreigners
    wide_levels = private.pivot_table(index="date", columns="sector", values="holdings", aggfunc="sum").sort_index()
    if "banks" in wide_levels.columns and "foreigners_official" in wide_levels.columns:
        shares = wide_levels.div(wide_levels.sum(axis=1), axis=0).fillna(0).clip(lower=0)
        shares_recent = shares.loc[shares.index >= "2000-01-01"]
        if not shares_recent.empty:
            save_stacked_area(shares_recent, "Sector Holding Shares", out / "sector_shares.png")
            print(f"  Chart: {out / 'sector_shares.png'}")

    print(f"Wrote baseline outputs to {out}")


def cmd_infer(args: argparse.Namespace) -> None:
    """Run counterparty inference from harmonized panel."""
    from tsyparty.baseline.flows import holdings_changes_from_levels, buyer_seller_margins
    from tsyparty.infer.counterparty import sign_baseline_matrix, ras_balance, sparse_threshold_rebalance, residual_bucket
    from tsyparty.config import load_yaml

    derived = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(derived / "harmonized_panel.csv", parse_dates=["date"])
    private = panel[~panel["sector"].isin(["_total", "_discrepancy", "fed"])].copy()

    inference_cfg = load_yaml("configs/inference.yml")

    # Compute quarterly changes
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes = changes.dropna(subset=["delta_holdings"])

    quarters = sorted(changes["date"].unique())
    results = []

    for q in quarters:
        q_data = changes[changes["date"] == q].groupby("sector", as_index=False)["delta_holdings"].sum()
        q_data = q_data.rename(columns={"delta_holdings": "net_flow"})

        # Skip quarters where everyone is flat
        if q_data["net_flow"].abs().sum() < 1.0:
            continue

        try:
            buyers, sellers = buyer_seller_margins(q_data)
        except Exception:
            continue

        if buyers.empty or sellers.empty:
            continue

        # Balance marginals: residual goes to explicit bucket
        buyer_total = float(buyers.sum())
        seller_total = float(sellers.sum())
        gap = buyer_total - seller_total

        if abs(gap) > 0.01:
            if gap > 0:
                sellers = pd.concat([sellers, pd.Series({"_residual": gap})])
            else:
                buyers = pd.concat([buyers, pd.Series({"_residual": -gap})])

        try:
            baseline = sign_baseline_matrix(buyers, sellers)
            prior = pd.DataFrame(1.0, index=sellers.index, columns=buyers.index)
            dense, dense_diag = ras_balance(prior, sellers, buyers)
            sparse, sparse_diag = sparse_threshold_rebalance(
                dense, sellers, buyers,
                threshold_quantile=inference_cfg.get("sparse_sensitivity", {}).get("threshold_quantile", 0.65),
            )
        except Exception as e:
            print(f"  {pd.Timestamp(q).date()}: skipped ({e})")
            continue

        # Save per-quarter result
        for label, matrix, diag in [("dense", dense, dense_diag), ("sparse", sparse, sparse_diag)]:
            for seller in matrix.index:
                for buyer in matrix.columns:
                    val = float(matrix.loc[seller, buyer])
                    if abs(val) < 0.01:
                        continue
                    results.append({
                        "date": q,
                        "seller": seller,
                        "buyer": buyer,
                        "amount": val,
                        "method": label,
                        "converged": diag.converged,
                    })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out / "counterparty_flows.csv", index=False)
        print(f"Counterparty inference: {len(df)} flow entries across {df['date'].nunique()} quarters")

        # Summary: who sells most to banks and foreigners?
        latest_q = df["date"].max()
        latest_sparse = df[(df["date"] == latest_q) & (df["method"] == "sparse")]
        if not latest_sparse.empty:
            print(f"\nLatest quarter ({pd.Timestamp(latest_q).date()}) — likely net sellers (sparse):")
            for buyer in ["banks", "foreigners_official"]:
                to_buyer = latest_sparse[latest_sparse["buyer"] == buyer].sort_values("amount", ascending=False)
                if not to_buyer.empty:
                    print(f"  To {buyer}:")
                    for _, row in to_buyer.head(5).iterrows():
                        print(f"    {row['seller']:30s} {row['amount']:>10,.0f}")
    else:
        print("No quarters produced valid inference results.")

    print(f"Wrote inference outputs to {out}")


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

    # --- Analysis commands ---
    p_baseline = subparsers.add_parser("baseline", help="Compute descriptive baseline outputs")
    p_baseline.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_baseline.add_argument("--out", default="outputs/baseline", help="Output directory")
    p_baseline.set_defaults(func=cmd_baseline)

    p_infer = subparsers.add_parser("infer", help="Run counterparty inference")
    p_infer.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_infer.add_argument("--out", default="outputs/inference", help="Output directory")
    p_infer.set_defaults(func=cmd_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
