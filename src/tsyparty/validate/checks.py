from __future__ import annotations

import pandas as pd


def assert_close_series(left: pd.Series, right: pd.Series, tol: float = 1.0e-8) -> None:
    joined = pd.concat({"left": left, "right": right}, axis=1).fillna(0.0)
    diff = (joined["left"] - joined["right"]).abs()
    if (diff > tol).any():
        raise AssertionError(f"Series differ by more than {tol}: {diff[diff > tol].to_dict()}")


def validate_market_clearing(matrix: pd.DataFrame, row_targets: pd.Series, col_targets: pd.Series, tol: float = 1.0e-8) -> dict:
    row_gap = (matrix.sum(axis=1) - row_targets).abs().max()
    col_gap = (matrix.sum(axis=0) - col_targets).abs().max()
    return {
        "max_row_gap": float(row_gap),
        "max_col_gap": float(col_gap),
        "passes": row_gap <= tol and col_gap <= tol,
    }
