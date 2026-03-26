from __future__ import annotations

import argparse
from pathlib import Path

from tsyparty.ingest.fed import download_z1_current, download_direct_source, download_h41_pdf
from tsyparty.ingest.treasury import (
    download_direct_treasury_source,
    download_fiscaldata_api,
    download_investor_class_recent,
)
from tsyparty.ingest.sec import download_latest_sec_zip
from tsyparty.ingest.ffiec import ffiec_manual_instructions


def main() -> None:
    parser = argparse.ArgumentParser(description="Download seeded public-data sources for tsyparty")
    parser.add_argument("--dest", default="data/raw_public", help="Destination directory")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading Z.1 current CSV zip...")
    download_z1_current(dest / "z1")

    print("Downloading FWTW and EFA direct CSVs...")
    for key in [
        "fwtw_csv",
        "efa_mmf_holdings",
        "efa_hedge_funds",
        "efa_banks",
        "efa_international_country",
        "h41_current_pdf",
    ]:
        download_direct_source(key, dest / "fed")

    print("Downloading TIC / Treasury direct artifacts...")
    for key in [
        "tic_slt_table1_txt",
        "tic_slt_historical_global",
    ]:
        download_direct_treasury_source(key, dest / "treasury")

    print("Downloading recent investor-class allotment workbooks...")
    download_investor_class_recent(dest / "treasury" / "investor_class")

    print("Downloading FiscalData API snapshots...")
    download_fiscaldata_api("debt_to_penny_api", dest / "fiscaldata")
    download_fiscaldata_api("dts_public_debt_transactions", dest / "fiscaldata", params={"page[size]": 1000})
    download_fiscaldata_api("dts_operating_cash", dest / "fiscaldata", params={"page[size]": 1000})

    print("Downloading latest SEC datasets...")
    for key in ["sec_nmfp", "sec_nport"]:
        try:
            download_latest_sec_zip(key, dest / "sec")
        except Exception as exc:
            print(f"SEC downloader warning for {key}: {exc}")

    print("Writing FFIEC manual instructions placeholder...")
    ffiec_manual_instructions(dest / "ffiec")

    print("Done. Review manifests and source notes before parsing.")
    print("H.8, SOMA, and Primary Dealer Statistics still need source-specific parsers in the next build pass.")


if __name__ == "__main__":
    main()
