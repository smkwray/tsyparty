import numpy as np
import pandas as pd

from tsyparty.infer.counterparty import ras_balance, sign_baseline_matrix, sparse_threshold_rebalance, sparse_cv


def test_sign_baseline_balances():
    buyers = pd.Series({"banks": 60.0, "foreigners_private": 40.0})
    sellers = pd.Series({"dealers": 50.0, "insurers": 25.0, "households_residual": 25.0})
    matrix = sign_baseline_matrix(buyers, sellers)
    assert np.isclose(matrix.sum().sum(), 100.0)
    assert np.allclose(matrix.sum(axis=0).to_numpy(), buyers.to_numpy())
    assert np.allclose(matrix.sum(axis=1).to_numpy(), sellers.to_numpy())


def test_ras_balance_hits_targets():
    row_targets = pd.Series({"a": 30.0, "b": 70.0})
    col_targets = pd.Series({"x": 20.0, "y": 80.0})
    prior = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=row_targets.index, columns=col_targets.index)

    matrix, diagnostics = ras_balance(prior, row_targets, col_targets)
    assert diagnostics.converged
    assert np.allclose(matrix.sum(axis=1).to_numpy(), row_targets.to_numpy(), atol=1e-7)
    assert np.allclose(matrix.sum(axis=0).to_numpy(), col_targets.to_numpy(), atol=1e-7)


def test_sparse_rebalance_preserves_targets():
    row_targets = pd.Series({"a": 30.0, "b": 70.0})
    col_targets = pd.Series({"x": 20.0, "y": 80.0})
    prior = pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=row_targets.index, columns=col_targets.index)

    dense, _ = ras_balance(prior, row_targets, col_targets)
    sparse, diag = sparse_threshold_rebalance(dense, row_targets, col_targets)
    assert diag.converged
    assert np.allclose(sparse.sum(axis=1).to_numpy(), row_targets.to_numpy(), atol=1e-7)
    assert np.allclose(sparse.sum(axis=0).to_numpy(), col_targets.to_numpy(), atol=1e-7)


def test_sparse_cv_selects_best_quantile():
    row_targets = pd.Series({"a": 30.0, "b": 70.0, "c": 50.0})
    col_targets = pd.Series({"x": 40.0, "y": 60.0, "z": 50.0})
    prior = pd.DataFrame(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        index=row_targets.index, columns=col_targets.index,
    )
    dense, _ = ras_balance(prior, row_targets, col_targets)
    best, diag, report = sparse_cv(dense, row_targets, col_targets)

    assert diag.converged
    assert np.allclose(best.sum(axis=1).to_numpy(), row_targets.to_numpy(), atol=1e-6)
    assert np.allclose(best.sum(axis=0).to_numpy(), col_targets.to_numpy(), atol=1e-6)
    assert "best_quantile" in report
    assert len(report["results"]) > 1
