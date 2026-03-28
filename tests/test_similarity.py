import pandas as pd

from tsyparty.behavior.similarity import (
    closest_sectors,
    cosine_distance_matrix,
    partial_corr_row,
    rolling_partial_correlations,
)


def test_cosine_distance_matrix_has_zero_diagonal():
    frame = pd.DataFrame(
        {
            "beta_net_public_supply": [1.0, 0.9, -0.3],
            "beta_delta_soma": [-0.2, -0.1, 0.4],
        },
        index=["banks", "foreigners_private", "insurers"],
    )
    dist = cosine_distance_matrix(frame)
    assert (dist.values.diagonal() == 0).all()


def test_closest_sectors_orders_smallest_distance():
    frame = pd.DataFrame(
        {
            "beta_net_public_supply": [1.0, 0.95, -0.4],
            "beta_delta_soma": [-0.2, -0.21, 0.5],
        },
        index=["banks", "foreigners_private", "insurers"],
    )
    dist = cosine_distance_matrix(frame)
    closest = closest_sectors(dist, "banks", top_n=1)
    assert closest.index[0] == "foreigners_private"


def test_partial_corr_row_uses_available_controls():
    dates = pd.date_range("2000-03-31", periods=8, freq="QE")
    y1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=dates)
    y2 = pd.Series([2, 4, 5, 8, 11, 12, 15, 17], index=dates)
    controls = pd.DataFrame({"delta_soma": [0, 1, 0, 1, 0, 1, 0, 1]}, index=dates)

    row = partial_corr_row(y1, y2, controls)
    assert row is not None
    assert "value" in row
    assert row["n_obs"] == 8


def test_rolling_partial_correlations_returns_long_form():
    dates = pd.date_range("2000-03-31", periods=10, freq="QE")
    wide = pd.DataFrame(
        {
            "banks": range(10),
            "dealers": [v * 2 for v in range(10)],
            "insurers": [(-1) ** v * v for v in range(10)],
        },
        index=dates,
    )
    context = pd.DataFrame(
        {
            "date": dates,
            "delta_soma": [0, 1] * 5,
            "net_public_supply": range(10, 20),
        }
    )

    result = rolling_partial_correlations(wide, context=context, x_cols=["net_public_supply", "delta_soma"], window=6)
    assert not result.empty
    assert {"sector_1", "sector_2", "q_value", "controls"}.issubset(result.columns)
    assert set(result["controls"]) == {"net_public_supply,delta_soma"}
