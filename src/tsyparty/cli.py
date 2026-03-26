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
        elif source == "tic_slt":
            from tsyparty.ingest.treasury import download_direct_treasury_source

            print(f"Downloading TIC SLT to {dest / 'tic'}")
            download_direct_treasury_source("tic_slt_table1_txt", dest / "tic")
            download_direct_treasury_source("tic_slt_historical_global", dest / "tic")
        elif source == "debt_to_penny":
            from tsyparty.ingest.treasury import download_fiscaldata_api

            print(f"Downloading debt-to-penny to {dest / 'fiscaldata'}")
            download_fiscaldata_api(
                "debt_to_penny_api", dest / "fiscaldata",
                params={"fields": "record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt",
                        "sort": "-record_date", "page[size]": "10000"},
            )
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


def cmd_parse_debt(args: argparse.Namespace) -> None:
    """Parse debt-to-penny API JSON into quarterly debt totals."""
    import json

    json_path = Path(args.json_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    records = data.get("data", [])
    rows = []
    for rec in records:
        date_str = rec.get("record_date", "")
        held_public = rec.get("debt_held_public_amt", "null")
        if held_public == "null" or not held_public:
            continue
        try:
            ts = pd.Timestamp(date_str)
            val = float(held_public)
        except (ValueError, TypeError):
            continue
        rows.append({"date": ts, "public_debt": val / 1_000_000})  # Convert to millions

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid debt-to-penny records found.")
        return

    # Resample to quarter-end (take the last observation in each quarter)
    df = df.sort_values("date")
    df = df.set_index("date")
    quarterly = df.resample("QE").last().dropna().reset_index()
    quarterly.to_csv(out / "debt_totals.csv", index=False)
    print(f"Debt totals: {len(quarterly)} quarters, {quarterly['date'].min().date()} to {quarterly['date'].max().date()}")


def cmd_parse_tic(args: argparse.Namespace) -> None:
    """Parse TIC SLT historical global CSV into quarterly foreign Treasury holdings."""
    csv_path = Path(args.csv_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path, dtype=str, skiprows=14, header=None, on_bad_lines="skip", low_memory=False)
    # Columns: Country Name, Country Code, End of Month, Total Long-Term, Treasury, Agency, Corp Bonds, Corp Stocks
    if df_raw.shape[1] < 5:
        print("Unexpected TIC SLT format")
        return

    df_raw.columns = ["country", "country_code", "date", "total_longterm", "treasury", "agency", "corp_bonds", "corp_stocks"][:df_raw.shape[1]]

    # Parse treasury column
    df_raw["treasury_val"] = pd.to_numeric(
        df_raw["treasury"].str.replace(",", "").str.strip(), errors="coerce"
    )
    df_raw["date_ts"] = pd.to_datetime(df_raw["date"].str.strip(), format="%Y-%m", errors="coerce")
    df_raw = df_raw.dropna(subset=["date_ts", "treasury_val"])

    # Aggregate across all countries per month
    monthly = df_raw.groupby("date_ts", as_index=False)["treasury_val"].sum()

    # Resample to quarter-end
    monthly = monthly.sort_values("date_ts").set_index("date_ts")
    quarterly = monthly.resample("QE").last().dropna().reset_index()
    quarterly.columns = ["date", "tic_foreign_treasury"]

    quarterly.to_csv(out / "tic_foreign_holdings.csv", index=False)
    print(f"TIC foreign Treasury holdings: {len(quarterly)} quarters, {quarterly['date'].min().date()} to {quarterly['date'].max().date()}")


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


def cmd_similarity(args: argparse.Namespace) -> None:
    """Compute sector behavior similarity from harmonized panel."""
    import numpy as np
    from tsyparty.baseline.flows import holdings_changes_from_levels
    from tsyparty.behavior.similarity import cosine_distance_matrix, closest_sectors
    from tsyparty.viz.charts import save_line_chart

    derived = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(derived / "harmonized_panel.csv", parse_dates=["date"])
    private = panel[~panel["sector"].isin(["_total", "_discrepancy"])].copy()

    # Compute quarterly holding changes
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    changes = changes.dropna(subset=["delta_holdings"])

    # Pivot to wide: date x sector
    wide = changes.pivot_table(index="date", columns="sector", values="delta_holdings", aggfunc="sum").fillna(0)

    # Feature 1: Correlation of holding changes (post-2000 for stability)
    wide_recent = wide.loc[wide.index >= "2000-01-01"]
    if wide_recent.empty or wide_recent.shape[1] < 3:
        print("Not enough data for similarity analysis.")
        return

    # Feature construction: mean change, volatility, share of total, trend
    features = pd.DataFrame(index=wide_recent.columns)
    features["mean_delta"] = wide_recent.mean()
    features["volatility"] = wide_recent.std()
    features["share_of_total_change"] = wide_recent.abs().mean() / wide_recent.abs().mean().sum()

    # Bill-share approximation: ratio of level to total (from latest available)
    levels = private.pivot_table(index="date", columns="sector", values="holdings", aggfunc="sum")
    if not levels.empty:
        latest_levels = levels.iloc[-1]
        features["share_of_total_holdings"] = latest_levels / latest_levels.sum()

    features = features.dropna()
    if features.empty or len(features) < 3:
        print("Not enough features for similarity analysis.")
        return

    # Compute cosine distance matrix
    dist = cosine_distance_matrix(features)
    dist.to_csv(out / "sector_distance_matrix.csv")
    print(f"Distance matrix: {dist.shape[0]}x{dist.shape[1]} sectors")

    # Closest sectors for key targets
    for target in ["banks", "foreigners_official", "money_market_funds"]:
        if target in dist.index:
            closest = closest_sectors(dist, target, top_n=5)
            print(f"\n  Closest to {target}:")
            for sector, d in closest.items():
                print(f"    {sector:30s} {d:.4f}")

    # Heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(dist.columns)))
    ax.set_yticks(range(len(dist.index)))
    ax.set_xticklabels(dist.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(dist.index, fontsize=8)
    ax.set_title("Sector Behavior Distance (cosine)")
    fig.colorbar(im, ax=ax, label="Distance")
    plt.tight_layout()
    plt.savefig(out / "sector_distance_heatmap.png", dpi=200)
    plt.close()
    print(f"  Chart: {out / 'sector_distance_heatmap.png'}")

    # Rolling correlation of key pairs
    if "banks" in wide_recent.columns and "foreigners_official" in wide_recent.columns:
        rolling_corr = wide_recent["banks"].rolling(20).corr(wide_recent["foreigners_official"])
        corr_df = pd.DataFrame({"banks_vs_foreigners": rolling_corr}).dropna()
        if not corr_df.empty:
            save_line_chart(corr_df, "Rolling Correlation: Banks vs Foreigners (20Q)", out / "rolling_corr_banks_foreigners.png", ylabel="Correlation")
            print(f"  Chart: {out / 'rolling_corr_banks_foreigners.png'}")

    print(f"Wrote similarity outputs to {out}")


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

    p_parse_debt = subparsers.add_parser("parse-debt", help="Parse debt-to-penny API JSON")
    p_parse_debt.add_argument("json_path", help="Path to debt_to_penny_api.json")
    p_parse_debt.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_debt.set_defaults(func=cmd_parse_debt)

    p_parse_tic = subparsers.add_parser("parse-tic", help="Parse TIC SLT global CSV")
    p_parse_tic.add_argument("csv_path", help="Path to slt1d_globl.csv")
    p_parse_tic.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_tic.set_defaults(func=cmd_parse_tic)

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

    p_similarity = subparsers.add_parser("similarity", help="Compute sector behavior similarity")
    p_similarity.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_similarity.add_argument("--out", default="outputs/similarity", help="Output directory")
    p_similarity.set_defaults(func=cmd_similarity)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
