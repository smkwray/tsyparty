from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1.0e-12


@dataclass(slots=True)
class MatrixDiagnostics:
    converged: bool
    iterations: int
    max_abs_row_error: float
    max_abs_col_error: float


def _validate_nonnegative(series: pd.Series, name: str) -> None:
    if (series < -EPS).any():
        raise ValueError(f"{name} contains negative values")


def sign_baseline_matrix(
    buyers: pd.Series,
    sellers: pd.Series,
    support: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Allocate buyer inflows proportionally to seller outflows.

    buyers: positive target inflows indexed by buyer sector.
    sellers: positive target outflows indexed by seller sector.
    support: optional boolean frame indexed by sellers x buyers.

    Returns a seller x buyer matrix.
    """
    _validate_nonnegative(buyers, "buyers")
    _validate_nonnegative(sellers, "sellers")

    if not np.isclose(float(buyers.sum()), float(sellers.sum()), atol=1e-8):
        raise ValueError("Buyer and seller margins must sum to the same total")

    weights = sellers / sellers.sum()
    matrix = pd.DataFrame(index=sellers.index, columns=buyers.index, dtype=float)
    for buyer, inflow in buyers.items():
        matrix[buyer] = inflow * weights

    if support is not None:
        aligned = support.reindex(index=matrix.index, columns=matrix.columns).fillna(False)
        masked = matrix.where(aligned, 0.0)
        return ras_balance(masked + aligned.astype(float) * EPS, sellers, buyers, aligned)[0]
    return matrix


def ras_balance(
    prior: pd.DataFrame,
    row_targets: pd.Series,
    col_targets: pd.Series,
    support: pd.DataFrame | None = None,
    max_iter: int = 10_000,
    tol: float = 1.0e-8,
) -> tuple[pd.DataFrame, MatrixDiagnostics]:
    """
    Balance a nonnegative prior matrix to match row and column targets.

    Rows correspond to sellers, columns correspond to buyers.
    """
    _validate_nonnegative(row_targets, "row_targets")
    _validate_nonnegative(col_targets, "col_targets")

    if not np.isclose(float(row_targets.sum()), float(col_targets.sum()), atol=1e-8):
        raise ValueError("Row and column targets must sum to the same total")

    matrix = prior.reindex(index=row_targets.index, columns=col_targets.index).fillna(0.0).astype(float)
    if support is None:
        support = matrix > 0
        if not support.to_numpy().any():
            support.loc[:, :] = True
    else:
        support = support.reindex(index=row_targets.index, columns=col_targets.index).fillna(False)

    arr = matrix.to_numpy(dtype=float)
    mask = support.to_numpy(dtype=bool)
    arr = np.where(mask, np.maximum(arr, EPS), 0.0)

    row_target_arr = row_targets.to_numpy(dtype=float)
    col_target_arr = col_targets.to_numpy(dtype=float)

    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration

        row_sums = arr.sum(axis=1)
        for i, target in enumerate(row_target_arr):
            if target <= EPS:
                arr[i, :] = 0.0
                continue
            if row_sums[i] <= EPS:
                allowed = mask[i, :]
                if not allowed.any():
                    raise ValueError(f"Row {row_targets.index[i]!r} has positive target but no allowed support")
                arr[i, allowed] = target / allowed.sum()
            else:
                arr[i, :] *= target / row_sums[i]

        col_sums = arr.sum(axis=0)
        for j, target in enumerate(col_target_arr):
            if target <= EPS:
                arr[:, j] = 0.0
                continue
            if col_sums[j] <= EPS:
                allowed = mask[:, j]
                if not allowed.any():
                    raise ValueError(f"Column {col_targets.index[j]!r} has positive target but no allowed support")
                arr[allowed, j] = target / allowed.sum()
            else:
                arr[:, j] *= target / col_sums[j]

        arr = np.where(mask, arr, 0.0)

        row_error = np.max(np.abs(arr.sum(axis=1) - row_target_arr))
        col_error = np.max(np.abs(arr.sum(axis=0) - col_target_arr))
        if max(row_error, col_error) <= tol:
            converged = True
            break

    result = pd.DataFrame(arr, index=row_targets.index, columns=col_targets.index)
    diagnostics = MatrixDiagnostics(
        converged=converged,
        iterations=iterations,
        max_abs_row_error=float(np.max(np.abs(result.sum(axis=1).to_numpy() - row_target_arr))),
        max_abs_col_error=float(np.max(np.abs(result.sum(axis=0).to_numpy() - col_target_arr))),
    )
    return result, diagnostics


def sparse_threshold_rebalance(
    dense_matrix: pd.DataFrame,
    row_targets: pd.Series,
    col_targets: pd.Series,
    support: pd.DataFrame | None = None,
    threshold_quantile: float = 0.65,
) -> tuple[pd.DataFrame, MatrixDiagnostics]:
    values = dense_matrix.to_numpy()
    positives = values[values > 0]
    if positives.size == 0:
        raise ValueError("dense_matrix has no positive cells")
    threshold = float(np.quantile(positives, threshold_quantile))

    if support is None:
        support = dense_matrix > 0
    support = support.reindex(index=row_targets.index, columns=col_targets.index).fillna(False)

    sparse_prior = dense_matrix.where(dense_matrix >= threshold, EPS)
    sparse_prior = sparse_prior.where(support, 0.0)
    return ras_balance(sparse_prior, row_targets, col_targets, support=support)


def residual_bucket(buyers: pd.Series, sellers: pd.Series) -> float:
    return float(buyers.sum() - sellers.sum())
