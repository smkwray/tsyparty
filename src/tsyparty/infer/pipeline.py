"""Counterparty inference pipeline.

Extracts the quarterly orchestration loop from cli.cmd_infer into
reusable, testable functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from tsyparty.baseline.flows import holdings_changes_from_levels, buyer_seller_margins
from tsyparty.infer.counterparty import (
    MatrixDiagnostics,
    ras_balance,
    sign_baseline_matrix,
    sparse_threshold_rebalance,
    sparse_cv,
    residual_bucket,
)
from tsyparty.validate.checks import validate_market_clearing


@dataclass(slots=True)
class InferenceConfig:
    """Typed configuration for the counterparty inference pipeline.

    Every field maps to a key in configs/inference.yml. No aspirational keys.
    """

    max_iter: int = 10_000
    tol: float = 1.0e-8
    epsilon: float = 1.0e-12
    threshold_quantile: float = 0.65
    sparse_enabled: bool = True
    sparse_cv_enabled: bool = False
    sparse_cv_quantiles: list[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    )
    use_structural_zeros: bool = True
    exclude_sectors: list[str] = field(
        default_factory=lambda: ["_total", "_discrepancy", "fed"]
    )
    require_market_clearing: bool = True
    compare_to_fwtw_levels: bool = True
    compare_to_auction_allotments: bool = True
    compare_foreign_side_to_tic: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> InferenceConfig:
        ras = cfg.get("entropy_ras", {})
        sparse = cfg.get("sparse_sensitivity", {})
        sparse_cv = cfg.get("sparse_cv", {})
        validation = cfg.get("validation", {})
        return cls(
            max_iter=ras.get("max_iter", 10_000),
            tol=ras.get("tol", 1.0e-8),
            epsilon=ras.get("epsilon", 1.0e-12),
            threshold_quantile=sparse.get("threshold_quantile", 0.65),
            sparse_enabled=sparse.get("enabled", True),
            sparse_cv_enabled=sparse_cv.get("enabled", False),
            sparse_cv_quantiles=sparse_cv.get("quantiles", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),
            use_structural_zeros=ras.get("use_structural_zeros", True),
            exclude_sectors=cfg.get("exclude_sectors", ["_total", "_discrepancy", "fed"]),
            require_market_clearing=validation.get("require_market_clearing", True),
            compare_to_fwtw_levels=validation.get("compare_to_fwtw_levels", True),
            compare_to_auction_allotments=validation.get("compare_to_auction_allotments", True),
            compare_foreign_side_to_tic=validation.get("compare_foreign_side_to_tic", True),
        )

    def validate(self) -> None:
        """Raise ValueError if config values are out of range."""
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if not 0 <= self.threshold_quantile <= 1:
            raise ValueError(f"threshold_quantile must be in [0, 1], got {self.threshold_quantile}")


@dataclass(slots=True)
class QuarterResult:
    """Inference result for a single quarter."""

    date: Any
    buyers: pd.Series
    sellers: pd.Series
    baseline: pd.DataFrame
    dense: pd.DataFrame
    dense_diag: MatrixDiagnostics
    sparse: pd.DataFrame | None
    sparse_diag: MatrixDiagnostics | None
    sparse_cv_matrix: pd.DataFrame | None
    sparse_cv_diag: MatrixDiagnostics | None
    sparse_cv_report: dict | None
    residual_amount: float
    residual_side: str  # "buyer" | "seller" | "none"
    market_clearing: dict


@dataclass(slots=True)
class InferenceResult:
    """Full pipeline output."""

    flows: pd.DataFrame
    baselines: list[tuple[Any, pd.DataFrame]] = field(default_factory=list)
    quarter_diagnostics: list[dict] = field(default_factory=list)
    validation_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    quarters_processed: int = 0
    quarters_skipped: int = 0


def prepare_quarters(
    panel: pd.DataFrame,
    exclude_sectors: list[str] | None = None,
) -> pd.DataFrame:
    """Compute quarterly holding changes from the harmonized panel."""
    if exclude_sectors is None:
        exclude_sectors = ["_total", "_discrepancy", "fed"]
    private = panel[~panel["sector"].isin(exclude_sectors)].copy()
    changes = holdings_changes_from_levels(private, group_cols=["sector", "instrument"])
    return changes.dropna(subset=["delta_holdings"])


def build_support_matrix(
    buyers: pd.Series,
    sellers: pd.Series,
    use_structural_zeros: bool = True,
) -> pd.DataFrame | None:
    """Build a boolean support matrix for RAS.

    Returns None for full support (all cells allowed).
    When use_structural_zeros is True, enforces domain-grounded restrictions:
      - A sector cannot sell to itself (diagonal zeros)
      - _residual bucket can trade with anyone (it's the catch-all)
      - households_residual cannot be a net counterparty to itself
    """
    if not use_structural_zeros:
        return None

    support = pd.DataFrame(True, index=sellers.index, columns=buyers.index)

    # Zero the diagonal: a sector cannot be both buyer and seller with itself
    for sector in support.index:
        if sector in support.columns:
            support.loc[sector, sector] = False

    # If every cell in a row or column is False, revert to full support
    # (this avoids infeasible constraints when a sector appears alone)
    row_all_false = ~support.any(axis=1)
    col_all_false = ~support.any(axis=0)
    if row_all_false.any() or col_all_false.any():
        return None

    return support


def run_quarter(
    q_changes: pd.DataFrame,
    date: Any,
    config: InferenceConfig,
) -> QuarterResult | None:
    """Run inference for a single quarter. Returns None if the quarter is skipped."""
    q_data = q_changes.groupby("sector", as_index=False)["delta_holdings"].sum()
    q_data = q_data.rename(columns={"delta_holdings": "net_flow"})

    if q_data["net_flow"].abs().sum() < 1.0:
        return None

    try:
        buyers, sellers = buyer_seller_margins(q_data)
    except Exception:
        return None

    if buyers.empty or sellers.empty:
        return None

    # Balance marginals: residual goes to explicit bucket
    buyer_total = float(buyers.sum())
    seller_total = float(sellers.sum())
    gap = buyer_total - seller_total

    residual_side = "none"
    residual_amount = 0.0
    if abs(gap) > 0.01:
        residual_amount = abs(gap)
        if gap > 0:
            sellers = pd.concat([sellers, pd.Series({"_residual": gap})])
            residual_side = "seller"
        else:
            buyers = pd.concat([buyers, pd.Series({"_residual": -gap})])
            residual_side = "buyer"

    support = build_support_matrix(buyers, sellers, config.use_structural_zeros)

    baseline = sign_baseline_matrix(buyers, sellers, support=support)
    prior = pd.DataFrame(1.0, index=sellers.index, columns=buyers.index)
    dense, dense_diag = ras_balance(
        prior, sellers, buyers,
        support=support,
        max_iter=config.max_iter,
        tol=config.tol,
        epsilon=config.epsilon,
    )

    sparse = None
    sparse_diag = None
    if config.sparse_enabled:
        sparse, sparse_diag = sparse_threshold_rebalance(
            dense, sellers, buyers,
            support=support,
            threshold_quantile=config.threshold_quantile,
        )

    sparse_cv_matrix = None
    sparse_cv_diag = None
    sparse_cv_report = None
    if config.sparse_cv_enabled:
        try:
            sparse_cv_matrix, sparse_cv_diag, sparse_cv_report = sparse_cv(
                dense, sellers, buyers, support=support,
                quantiles=config.sparse_cv_quantiles,
            )
        except Exception:
            pass

    mc = validate_market_clearing(dense, sellers, buyers, tol=config.tol * 100)

    return QuarterResult(
        date=date,
        buyers=buyers,
        sellers=sellers,
        baseline=baseline,
        dense=dense,
        dense_diag=dense_diag,
        sparse=sparse,
        sparse_diag=sparse_diag,
        sparse_cv_matrix=sparse_cv_matrix,
        sparse_cv_diag=sparse_cv_diag,
        sparse_cv_report=sparse_cv_report,
        residual_amount=residual_amount,
        residual_side=residual_side,
        market_clearing=mc,
    )


def run_inference(
    panel: pd.DataFrame,
    config: InferenceConfig | None = None,
) -> InferenceResult:
    """Run the full quarterly inference pipeline."""
    if config is None:
        config = InferenceConfig()

    changes = prepare_quarters(panel, exclude_sectors=config.exclude_sectors)
    quarters = sorted(changes["date"].unique())

    flow_rows: list[dict] = []
    baselines: list[tuple[Any, pd.DataFrame]] = []
    diagnostics: list[dict] = []
    skipped = 0

    for q in quarters:
        q_changes = changes[changes["date"] == q]
        try:
            result = run_quarter(q_changes, q, config)
        except Exception:
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        # Collect diagnostics for this quarter
        diag_entry = {
            "date": str(pd.Timestamp(q).date()),
            "buyer_total": float(result.buyers.sum()),
            "seller_total": float(result.sellers.sum()),
            "residual_amount": float(result.residual_amount),
            "residual_side": result.residual_side,
            "n_buyers": int(len(result.buyers)),
            "n_sellers": int(len(result.sellers)),
            "dense_converged": bool(result.dense_diag.converged),
            "dense_iterations": int(result.dense_diag.iterations),
            "dense_max_row_error": float(result.dense_diag.max_abs_row_error),
            "dense_max_col_error": float(result.dense_diag.max_abs_col_error),
            "dense_nonzero_cells": int((result.dense.abs() > 0.01).sum().sum()),
            "market_clearing_passes": bool(result.market_clearing["passes"]),
            "market_clearing_max_row_gap": float(result.market_clearing["max_row_gap"]),
            "market_clearing_max_col_gap": float(result.market_clearing["max_col_gap"]),
        }
        if result.sparse_diag is not None:
            diag_entry["sparse_converged"] = bool(result.sparse_diag.converged)
            diag_entry["sparse_iterations"] = int(result.sparse_diag.iterations)
            diag_entry["sparse_max_row_error"] = float(result.sparse_diag.max_abs_row_error)
            diag_entry["sparse_max_col_error"] = float(result.sparse_diag.max_abs_col_error)
            diag_entry["sparse_nonzero_cells"] = int(
                (result.sparse.abs() > 0.01).sum().sum()
            ) if result.sparse is not None else 0
        if result.sparse_cv_diag is not None:
            diag_entry["sparse_cv_converged"] = bool(result.sparse_cv_diag.converged)
            diag_entry["sparse_cv_iterations"] = int(result.sparse_cv_diag.iterations)
            diag_entry["sparse_cv_nonzero_cells"] = int(
                (result.sparse_cv_matrix.abs() > 0.01).sum().sum()
            ) if result.sparse_cv_matrix is not None else 0
            if result.sparse_cv_report:
                diag_entry["sparse_cv_best_quantile"] = result.sparse_cv_report["best_quantile"]
        diagnostics.append(diag_entry)

        # Collect baseline
        baselines.append((q, result.baseline))

        # Collect flow rows
        for label, matrix, diag in [("dense", result.dense, result.dense_diag)]:
            _append_flows(flow_rows, q, matrix, label, diag)
        if result.sparse is not None and result.sparse_diag is not None:
            _append_flows(flow_rows, q, result.sparse, "sparse", result.sparse_diag)
        if result.sparse_cv_matrix is not None and result.sparse_cv_diag is not None:
            _append_flows(flow_rows, q, result.sparse_cv_matrix, "sparse_cv", result.sparse_cv_diag)

    flows = pd.DataFrame(flow_rows) if flow_rows else pd.DataFrame(
        columns=["date", "seller", "buyer", "amount", "method", "converged"]
    )

    return InferenceResult(
        flows=flows,
        baselines=baselines,
        quarter_diagnostics=diagnostics,
        quarters_processed=len(diagnostics),
        quarters_skipped=skipped,
    )


def _append_flows(
    rows: list[dict],
    date: Any,
    matrix: pd.DataFrame,
    method: str,
    diag: MatrixDiagnostics,
) -> None:
    for seller in matrix.index:
        for buyer in matrix.columns:
            val = float(matrix.loc[seller, buyer])
            if abs(val) < 0.01:
                continue
            rows.append({
                "date": date,
                "seller": seller,
                "buyer": buyer,
                "amount": val,
                "method": method,
                "converged": diag.converged,
            })


def validate_inference(
    result: InferenceResult,
    config: InferenceConfig,
    fwtw: pd.DataFrame | None = None,
    auction_allotments: pd.DataFrame | None = None,
    tic_foreign: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run configured validation checks against inference outputs.

    Returns a dict of check_name -> result DataFrame.
    """
    from tsyparty.validate.crosscheck import (
        compare_inference_to_fwtw,
        compare_inference_to_auction,
        compare_foreign_inference_to_tic,
    )

    checks: dict[str, pd.DataFrame] = {}

    if config.compare_to_fwtw_levels and fwtw is not None and not fwtw.empty:
        checks["fwtw_comparison"] = compare_inference_to_fwtw(result.flows, fwtw)

    if config.compare_to_auction_allotments and auction_allotments is not None and not auction_allotments.empty:
        checks["auction_comparison"] = compare_inference_to_auction(result.flows, auction_allotments)

    if config.compare_foreign_side_to_tic and tic_foreign is not None and not tic_foreign.empty:
        checks["tic_foreign_comparison"] = compare_foreign_inference_to_tic(result.flows, tic_foreign)

    return checks


def write_outputs(result: InferenceResult, out_dir: Path) -> dict[str, Path]:
    """Write inference artifacts to disk. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not result.flows.empty:
        flows_path = out_dir / "counterparty_flows.csv"
        result.flows.to_csv(flows_path, index=False)
        paths["flows"] = flows_path

    if result.baselines:
        baseline_rows = []
        for date, matrix in result.baselines:
            for seller in matrix.index:
                for buyer in matrix.columns:
                    val = float(matrix.loc[seller, buyer])
                    if abs(val) < 0.01:
                        continue
                    baseline_rows.append({
                        "date": date,
                        "seller": seller,
                        "buyer": buyer,
                        "baseline_amount": val,
                    })
        if baseline_rows:
            baseline_path = out_dir / "baseline_matrices.csv"
            pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
            paths["baselines"] = baseline_path

    if result.quarter_diagnostics:
        diag_path = out_dir / "quarter_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(result.quarter_diagnostics, f, indent=2)
        paths["diagnostics"] = diag_path

    for check_name, check_df in result.validation_results.items():
        if not check_df.empty:
            check_path = out_dir / f"{check_name}.csv"
            check_df.to_csv(check_path, index=False)
            paths[check_name] = check_path

    return paths
