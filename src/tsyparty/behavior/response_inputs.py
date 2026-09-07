"""Reconstruct frozen quarterly response inputs from declared source cells."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tsyparty.behavior.issuance_maturity_response import (
    DEFAULT_SECTOR_GROUPS,
    build_sector_flows,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transaction_input(archive: Path, supplement: Path, contract: dict, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require complete source cells for every declared leaf and quarter."""
    included = contract["included_series"]
    expected_keys = {key for keys in DEFAULT_SECTOR_GROUPS.values() for key in keys}
    if set(included) != expected_keys or any(not codes for codes in included.values()):
        raise ValueError("Source contract must cover every configured included sector")
    codes = [code for values in included.values() for code in values]
    excluded = set(contract["excluded_table_series"])
    supplemental = contract["supplement_series"]
    if len(codes) != len(set(codes)) or set(codes) & excluded:
        raise ValueError("Overlapping source crosswalk")
    if any(not code.startswith("FU") or not code.endswith(".Q") for code in codes):
        raise ValueError("Only FU quarterly transaction sources are allowed")
    if set(supplemental) - set(codes) or contract["transaction_basis"] != "FU_quarterly_millions":
        raise ValueError("Invalid quarterly transaction source basis")
    with zipfile.ZipFile(archive) as source:
        table = pd.read_csv(io.BytesIO(source.read(contract["table_member"])), na_values=["ND"])
        dictionary = {parts[0]: parts for line in source.read(contract["dictionary_member"]).decode().splitlines()
                      if len(parts := line.split("\t")) >= 5}
    primary_codes = set(codes) - set(supplemental)
    if set(table.columns) - {"date"} != primary_codes | excluded:
        raise ValueError("Raw table crosswalk is not exhaustive")
    for code in primary_codes:
        if code not in dictionary or dictionary[code][-1] != contract["table_units"]:
            raise ValueError(f"Missing or incompatible source units: {code}")
    table["date"] = pd.PeriodIndex(table["date"].str.replace(":", ""), freq="Q").to_timestamp("Q")
    if table["date"].duplicated().any():
        raise ValueError("Duplicate primary source quarters")
    table = table.set_index("date")
    extra = pd.read_csv(supplement)
    extra["date"] = pd.to_datetime(extra["date"]).dt.to_period("Q").dt.to_timestamp("Q")
    selected_extra = extra[extra["series_code"].isin(supplemental)]
    if selected_extra.duplicated(["date", "series_code"]).any():
        raise ValueError("Duplicate supplementary source cells")
    quarters = pd.period_range(start, end, freq="Q").to_timestamp("Q")
    if not len(quarters):
        raise ValueError("Empty requested quarter window")
    cells, sectors = [], []
    for key, source_codes in included.items():
        components = []
        for code in source_codes:
            if code in supplemental:
                spec = supplemental[code]
                source_rows = selected_extra[selected_extra["series_code"] == code]
                if source_rows.empty or not source_rows["provider"].eq(spec["provider"]).all() or not source_rows["vintage"].eq(spec["vintage"]).all():
                    raise ValueError(f"Missing or conflicting supplementary provenance: {code}")
                values = source_rows.set_index("date")["value"].reindex(quarters)
                origin, units = supplement.name, spec["units"]
            else:
                values = table[code].reindex(quarters)
                origin, units = contract["table_member"], dictionary[code][-1]
            values = pd.to_numeric(values, errors="raise")
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete finite source coverage: {code}")
            components.append(values)
            cells.append(pd.DataFrame({"date": quarters, "sector_key": key, "series_code": code,
                                       "value": values.to_numpy(), "source_member": origin, "source_units": units}))
        total = pd.concat(components, axis=1).sum(axis=1, min_count=len(components))
        sectors.append(pd.DataFrame({"date": quarters, "sector_key": key, "transactions": total.to_numpy(),
                                     "transaction_basis": contract["transaction_basis"]}))
    panel = pd.concat(sectors, ignore_index=True)
    build_sector_flows(panel, transaction_basis=contract["transaction_basis"])
    return panel, pd.concat(cells, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--holder-panel", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output directory already exists; preserve earlier inputs")
    producer = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Commit the producing source before retaining inputs")
    contract = yaml.safe_load(args.contract.read_text())
    panel, cells = transaction_input(args.archive, args.supplement, contract, args.start, args.end)
    holder = pd.read_csv(args.holder_panel)
    if holder["source"].ne("z1").any():
        raise ValueError("Holder relabel requires the combined base Z.1 panel")
    if set(holder["sector"]) & set(contract["holder_sector_relabels"].values()):
        raise ValueError("Holder relabel would overlap an existing sector")
    holder["sector"] = holder["sector"].replace(contract["holder_sector_relabels"])
    args.out.mkdir(parents=True)
    panel.to_csv(args.out / "maturity_transactions.csv", index=False)
    cells.to_csv(args.out / "maturity_source_cells.csv", index=False)
    holder.to_csv(args.out / "holder_panel.csv", index=False)
    inputs = {name: {"path": str(path), "sha256": sha256(path)} for name, path in
              [("archive", args.archive), ("supplement", args.supplement), ("holder_panel", args.holder_panel), ("contract", args.contract)]}
    receipt = {"producer_commit": producer, "inputs": inputs, "effective_contract": contract,
               "start": args.start, "end": args.end, "sector_rows": len(panel), "source_cells": len(cells),
               "outputs": {p.name: sha256(p) for p in sorted(args.out.glob("*.csv"))},
               "claim_boundary": "Frozen-input reproduction; no causal or thesis empirical admission"}
    (args.out / "input_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"producer_commit": producer, "sector_rows": len(panel), "source_cells": len(cells)}))


if __name__ == "__main__":
    main()
