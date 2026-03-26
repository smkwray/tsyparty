import pandas as pd

from tsyparty.behavior.similarity import cosine_distance_matrix, closest_sectors


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
