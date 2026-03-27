from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsyparty.registry import iter_sources
from tsyparty.infer.counterparty import ras_balance, sign_baseline_matrix, sparse_threshold_rebalance
from tsyparty.behavior.similarity import cosine_distance_matrix
from tsyparty.viz.charts import save_stacked_area

# NOTE: ras_balance, sign_baseline_matrix, sparse_threshold_rebalance, and
# cosine_distance_matrix are still imported for the `example` command.


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

    # Shorthand aliases map to specific download logic
    ALIASES = {
        "z1": "_z1",
        "fwtw": "_fwtw",
        "investor_class": "_investor_class",
        "tic_slt": "_tic_slt",
        "efa": "_efa",
        "debt_to_penny": "_debt_to_penny",
        "soma": "_soma",
        "h8": "_h8",
        "primary_dealer": "_primary_dealer",
    }

    for source in sources_to_download:
        if source in ALIASES:
            _download_aliased(ALIASES[source], dest)
        else:
            # Try the generic registry-driven dispatcher
            from tsyparty.ingest.download import download_source
            try:
                sub_dir = dest / source.replace("_", "/").split("/")[0]
                print(f"Downloading {source} to {sub_dir}")
                download_source(source, sub_dir)
            except (KeyError, ValueError) as e:
                print(f"Unknown source: {source} ({e})")


def _download_aliased(alias: str, dest: Path) -> None:
    """Handle legacy aliased download targets."""
    if alias == "_z1":
        from tsyparty.ingest.fed import download_z1_current
        print(f"Downloading Z.1 CSV zip to {dest / 'z1'}")
        download_z1_current(dest / "z1")
    elif alias == "_fwtw":
        from tsyparty.ingest.fwtw import download_fwtw
        print(f"Downloading FWTW CSV to {dest / 'fwtw'}")
        download_fwtw(dest / "fwtw")
    elif alias == "_investor_class":
        from tsyparty.ingest.treasury import download_investor_class_recent
        print(f"Downloading investor-class XLS to {dest / 'auction'}")
        download_investor_class_recent(dest / "auction")
    elif alias == "_tic_slt":
        from tsyparty.ingest.treasury import download_direct_treasury_source
        print(f"Downloading TIC SLT to {dest / 'tic'}")
        download_direct_treasury_source("tic_slt_table1_txt", dest / "tic")
        download_direct_treasury_source("tic_slt_historical_global", dest / "tic")
    elif alias == "_efa":
        from tsyparty.ingest.fed import download_direct_source
        for key in ("efa_mmf_holdings", "efa_banks", "efa_international_country"):
            print(f"Downloading {key} to {dest / 'efa'}")
            download_direct_source(key, dest / "efa")
    elif alias == "_debt_to_penny":
        from tsyparty.ingest.treasury import download_fiscaldata_api
        print(f"Downloading debt-to-penny to {dest / 'fiscaldata'}")
        download_fiscaldata_api(
            "debt_to_penny_api", dest / "fiscaldata",
            params={"fields": "record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt",
                    "sort": "-record_date", "page[size]": "10000"},
        )
    elif alias == "_soma":
        from tsyparty.ingest.download import download_source
        print(f"Downloading SOMA holdings to {dest / 'soma'}")
        download_source("soma_holdings_page", dest / "soma")
    elif alias == "_h8":
        from tsyparty.ingest.download import download_source
        print(f"Downloading H.8 to {dest / 'h8'}")
        download_source("h8_release_page", dest / "h8")
    elif alias == "_primary_dealer":
        from tsyparty.ingest.download import download_source
        print(f"Downloading primary dealer statistics to {dest / 'dealer'}")
        download_source("primary_dealer_statistics", dest / "dealer")


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
    from tsyparty.ingest.fiscaldata import parse_debt_to_penny

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quarterly = parse_debt_to_penny(args.json_path)
    if quarterly.empty:
        print("No valid debt-to-penny records found.")
        return

    quarterly.to_csv(out / "debt_totals.csv", index=False)
    print(f"Debt totals: {len(quarterly)} quarters, {quarterly['date'].min().date()} to {quarterly['date'].max().date()}")


def cmd_parse_tic(args: argparse.Namespace) -> None:
    """Parse TIC SLT historical global CSV into quarterly foreign Treasury holdings."""
    from tsyparty.ingest.tic import parse_slt_global

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quarterly = parse_slt_global(args.csv_path)
    quarterly.to_csv(out / "tic_foreign_holdings.csv", index=False)
    print(f"TIC foreign Treasury holdings: {len(quarterly)} quarters, {quarterly['date'].min().date()} to {quarterly['date'].max().date()}")


def cmd_parse_efa(args: argparse.Namespace) -> None:
    """Parse EFA data sources into validation targets."""
    from tsyparty.ingest.efa import parse_efa_mmf_treasury, parse_efa_bank_treasury, parse_efa_international

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    efa_dir = Path(args.efa_dir)

    mmf_path = efa_dir / "money-market-funds-investment-holdings-historical.csv"
    if mmf_path.exists():
        mmf_q = parse_efa_mmf_treasury(mmf_path)
        if not mmf_q.empty:
            mmf_q.to_csv(out / "efa_mmf_treasury.csv", index=False)
            print(f"EFA MMF Treasury: {len(mmf_q)} quarters, {mmf_q['date'].min().date()} to {mmf_q['date'].max().date()}")

    bank_path = efa_dir / "consolidated-bank-balance-sheet-data-historical.csv"
    if bank_path.exists():
        bank_df = parse_efa_bank_treasury(bank_path)
        if not bank_df.empty:
            bank_df.to_csv(out / "efa_bank_treasury.csv", index=False)
            print(f"EFA banks Treasury: {len(bank_df)} quarters, {bank_df['date'].min().date()} to {bank_df['date'].max().date()}")

    intl_path = efa_dir / "international-portfolio-investment-table1a-historical.csv"
    if intl_path.exists():
        intl_q = parse_efa_international(intl_path)
        if not intl_q.empty:
            intl_q.to_csv(out / "efa_foreign_total.csv", index=False)
            print(f"EFA foreign total LT: {len(intl_q)} quarters, {intl_q['date'].min().date()} to {intl_q['date'].max().date()}")


def cmd_parse_soma(args: argparse.Namespace) -> None:
    """Parse SOMA Treasury holdings JSON into weekly series and quarterly delta."""
    from tsyparty.context.soma import parse_soma_json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = parse_soma_json(args.json_path)
    if result.weekly.empty:
        print("No valid SOMA records found.")
        return

    result.weekly.to_csv(out / "soma_weekly.csv", index=False)
    result.quarterly_delta.to_csv(out / "soma_quarterly_delta.csv", index=False)
    print(f"SOMA holdings: {result.n_records} records → {len(result.weekly)} weekly obs, {len(result.quarterly_delta)} quarterly deltas")
    print(f"Wrote to {out}")


def cmd_parse_h8(args: argparse.Namespace) -> None:
    """Parse H.8 bank balance sheet CSV into weekly series."""
    from tsyparty.context.h8 import parse_h8_csv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = parse_h8_csv(args.csv_path)
    if result.weekly.empty:
        print("No valid H.8 records found.")
        return

    result.weekly.to_csv(out / "h8_weekly.csv", index=False)
    print(f"H.8: {result.n_series} series, {len(result.weekly)} weekly obs")
    if result.date_range:
        print(f"  Date range: {result.date_range[0].date()} to {result.date_range[1].date()}")
    print(f"Wrote to {out}")


def cmd_parse_dealer(args: argparse.Namespace) -> None:
    """Parse primary dealer statistics JSON into weekly series."""
    from tsyparty.context.dealer import parse_dealer_json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = parse_dealer_json(args.json_path)
    if result.weekly.empty:
        print("No valid dealer statistics found.")
        return

    result.weekly.to_csv(out / "dealer_weekly.csv", index=False)
    print(f"Dealer stats: {result.n_series} series, {len(result.weekly)} weekly obs")
    if result.date_range:
        print(f"  Date range: {result.date_range[0].date()} to {result.date_range[1].date()}")
    if result.source_periods:
        print(f"  Source periods: {result.source_periods}")
    print(f"Wrote to {out}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Cross-validate harmonized panel against EFA and TIC sources."""
    from tsyparty.validate.crosscheck import run_crosschecks

    derived = Path(args.derived)
    interim = Path(args.interim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel_file) if args.panel_file else derived / "harmonized_panel.csv"
    panel = pd.read_csv(panel_path, parse_dates=["date"])

    efa_bank = _read_optional_csv(interim / "efa_bank_treasury.csv")
    efa_mmf = _read_optional_csv(interim / "efa_mmf_treasury.csv")
    tic_foreign = _read_optional_csv(interim / "tic_foreign_holdings.csv")

    summary = run_crosschecks(panel, efa_bank=efa_bank, efa_mmf=efa_mmf, tic_foreign=tic_foreign)

    if not summary.empty:
        for _, row in summary.iterrows():
            print(f"{row['sector']} validation: {row['quarters']:.0f} quarters, mean diff {row['mean_diff_pct']:.1f}%")
        summary.to_csv(out / "validation_summary.csv", index=False)
        print(f"\nWrote validation summary to {out / 'validation_summary.csv'}")
    else:
        print("No validation sources found. Run parse-efa and parse-tic first.")


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


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
    from tsyparty.config import load_yaml
    from tsyparty.infer.pipeline import InferenceConfig, run_inference, validate_inference, write_outputs

    panel_path = Path(args.panel_file) if args.panel_file else Path(args.derived) / "harmonized_panel.csv"
    out = Path(args.out)
    interim = Path(args.derived).parent / "interim"

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    inference_cfg = load_yaml("configs/inference.yml")
    config = InferenceConfig.from_dict(inference_cfg)

    result = run_inference(panel, config)

    # Run validation checks against available data
    fwtw = _read_optional_csv(interim / "fwtw_holdings.csv")
    tic_foreign = _read_optional_csv(interim / "tic_foreign_holdings.csv")
    auction = _read_optional_csv(interim / "bills_allotments.csv")
    if auction is None:
        auction = _read_optional_csv(interim / "nominal_coupons_allotments.csv")
    result.validation_results = validate_inference(
        result, config, fwtw=fwtw, auction_allotments=auction, tic_foreign=tic_foreign,
    )

    paths = write_outputs(result, out)

    print(f"Counterparty inference: {len(result.flows)} flow entries across {result.quarters_processed} quarters ({result.quarters_skipped} skipped)")

    if not result.flows.empty:
        df = result.flows
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

    for check_name, check_df in result.validation_results.items():
        if not check_df.empty:
            print(f"  Validation {check_name}: {len(check_df)} comparisons")

    print(f"Wrote inference outputs to {out}")


def cmd_primary_market(args: argparse.Namespace) -> None:
    """Build primary-market allocation from auction data."""
    from tsyparty.baseline.primary_market import build_primary_allocation, primary_allocation_summary

    interim = Path(args.interim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bills_path = interim / "bills_allotments.csv"
    coupons_path = interim / "nominal_coupons_allotments.csv"

    bills = pd.read_csv(bills_path, parse_dates=["date"]) if bills_path.exists() else None
    coupons = pd.read_csv(coupons_path, parse_dates=["date"]) if coupons_path.exists() else None

    allocation = build_primary_allocation(bills, coupons)
    allocation.to_csv(out / "primary_allocation.csv", index=False)
    print(f"Primary allocation: {len(allocation)} rows, {allocation['date'].nunique()} quarters")

    summary = primary_allocation_summary(allocation)
    if not summary.empty:
        summary.to_csv(out / "primary_allocation_summary.csv", index=False)
        print(f"\nAverage primary-market shares:")
        for _, row in summary.iterrows():
            print(f"  {row['buyer_class']:25s} {row['mean_share']:6.1%}  ({row['mean_amount']:>10,.0f} avg)")

    print(f"Wrote to {out}")


def cmd_enrich_foreign(args: argparse.Namespace) -> None:
    """Enrich harmonized panel by splitting foreigners into official vs private."""
    from tsyparty.ingest.tic import parse_slt_table1_countries
    from tsyparty.reconcile.enrich import estimate_official_share, enrich_foreign_split, write_enrichment_metadata

    derived = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel_file) if args.panel_file else derived / "harmonized_panel.csv"
    panel = pd.read_csv(panel_path, parse_dates=["date"])

    tic_path = Path(args.tic_dir) / "slt_table1.txt"
    official_share = None
    if tic_path.exists():
        tic_countries = parse_slt_table1_countries(tic_path)
        if not tic_countries.empty:
            official_share = estimate_official_share(tic_countries)
    else:
        print(f"TIC SLT table1 not found at {tic_path}. Using default official share.")

    enriched_panel = enrich_foreign_split(panel, official_share)
    enriched_panel.to_csv(out / "harmonized_panel_enriched.csv", index=False)

    write_enrichment_metadata(out / "enrichment_metadata.json", official_share)

    mean_share = float(official_share.mean()) if official_share is not None and not official_share.empty else 0.65
    print(f"Foreign enrichment: mean official share = {mean_share:.1%}")
    print(f"Panel expanded from {len(panel)} to {len(enriched_panel)} rows (added foreigners_private)")
    print(f"Wrote to {out / 'harmonized_panel_enriched.csv'}")
    print(f"Wrote enrichment metadata to {out / 'enrichment_metadata.json'}")


def cmd_similarity(args: argparse.Namespace) -> None:
    """Compute sector behavior similarity from harmonized panel."""
    from tsyparty.behavior.pipeline import SimilarityConfig, run_similarity, write_outputs, write_charts

    panel_path = Path(args.panel_file) if args.panel_file else Path(args.derived) / "harmonized_panel.csv"
    out = Path(args.out)
    derived = Path(args.derived)

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    config = SimilarityConfig.from_sectors_yml()

    # Load context factors if available (SOMA quarterly delta, etc.)
    context = None
    soma_path = derived.parent / "interim" / "soma_quarterly_delta.csv"
    if soma_path.exists():
        context = pd.read_csv(soma_path, parse_dates=["date"])

    result = run_similarity(panel, config, context=context)

    if result is None:
        print("Not enough data for similarity analysis.")
        return

    paths = write_outputs(result, out)
    chart_paths = write_charts(result, out)

    print(f"Distance matrix: {result.distance_matrix.shape[0]}x{result.distance_matrix.shape[1]} sectors")
    for target, nearest in result.closest.items():
        print(f"\n  Closest to {target}:")
        for sector, d in nearest.items():
            print(f"    {sector:30s} {d:.4f}")

    if result.absorption_betas is not None:
        print(f"\n  Absorption betas: {len(result.absorption_betas)} obs across {result.absorption_betas['sector'].nunique()} sectors")

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

    p_parse_efa = subparsers.add_parser("parse-efa", help="Parse EFA data sources")
    p_parse_efa.add_argument("--efa-dir", default="data/raw_public/efa", help="EFA data directory")
    p_parse_efa.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_efa.set_defaults(func=cmd_parse_efa)

    p_parse_debt = subparsers.add_parser("parse-debt", help="Parse debt-to-penny API JSON")
    p_parse_debt.add_argument("json_path", help="Path to debt_to_penny_api.json")
    p_parse_debt.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_debt.set_defaults(func=cmd_parse_debt)

    p_parse_tic = subparsers.add_parser("parse-tic", help="Parse TIC SLT global CSV")
    p_parse_tic.add_argument("csv_path", help="Path to slt1d_globl.csv")
    p_parse_tic.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_tic.set_defaults(func=cmd_parse_tic)

    p_parse_soma = subparsers.add_parser("parse-soma", help="Parse SOMA Treasury holdings JSON")
    p_parse_soma.add_argument("json_path", help="Path to SOMA JSON file")
    p_parse_soma.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_soma.set_defaults(func=cmd_parse_soma)

    p_parse_h8 = subparsers.add_parser("parse-h8", help="Parse H.8 bank balance sheet CSV")
    p_parse_h8.add_argument("csv_path", help="Path to H.8 CSV file")
    p_parse_h8.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_h8.set_defaults(func=cmd_parse_h8)

    p_parse_dealer = subparsers.add_parser("parse-dealer", help="Parse primary dealer statistics JSON")
    p_parse_dealer.add_argument("json_path", help="Path to dealer JSON file")
    p_parse_dealer.add_argument("--out", default="data/interim", help="Output directory")
    p_parse_dealer.set_defaults(func=cmd_parse_dealer)

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
    p_infer.add_argument("--panel-file", default=None, help="Path to panel CSV (default: <derived>/harmonized_panel.csv)")
    p_infer.add_argument("--out", default="outputs/inference", help="Output directory")
    p_infer.set_defaults(func=cmd_infer)

    p_primary = subparsers.add_parser("primary-market", help="Build primary-market allocation")
    p_primary.add_argument("--interim", default="data/interim", help="Interim data directory")
    p_primary.add_argument("--out", default="outputs/primary_market", help="Output directory")
    p_primary.set_defaults(func=cmd_primary_market)

    p_enrich = subparsers.add_parser("enrich-foreign", help="Split foreign holdings into official/private")
    p_enrich.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_enrich.add_argument("--panel-file", default=None, help="Path to panel CSV (default: <derived>/harmonized_panel.csv)")
    p_enrich.add_argument("--tic-dir", default="data/raw_public/tic", help="TIC SLT data directory")
    p_enrich.add_argument("--out", default="data/derived", help="Output directory")
    p_enrich.set_defaults(func=cmd_enrich_foreign)

    p_validate = subparsers.add_parser("validate", help="Cross-validate panel against EFA/TIC")
    p_validate.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_validate.add_argument("--panel-file", default=None, help="Path to panel CSV (default: <derived>/harmonized_panel.csv)")
    p_validate.add_argument("--interim", default="data/interim", help="Interim data directory")
    p_validate.add_argument("--out", default="outputs/validation", help="Output directory")
    p_validate.set_defaults(func=cmd_validate)

    p_similarity = subparsers.add_parser("similarity", help="Compute sector behavior similarity")
    p_similarity.add_argument("--derived", default="data/derived", help="Derived data directory")
    p_similarity.add_argument("--panel-file", default=None, help="Path to panel CSV (default: <derived>/harmonized_panel.csv)")
    p_similarity.add_argument("--out", default="outputs/similarity", help="Output directory")
    p_similarity.set_defaults(func=cmd_similarity)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
