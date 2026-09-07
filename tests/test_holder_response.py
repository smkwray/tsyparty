import json

import pandas as pd

from tsyparty.behavior.holder_response import (
    HolderResponseConfig,
    build_response_panel,
    estimate_holder_response,
    load_shock_artifact,
    run_holder_response,
    write_outputs,
)


def _panel(n_quarters: int = 16) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")
    for i, date in enumerate(dates):
        shock = float(i % 4)
        values = {
            "banks": 1000 + i * 20 + shock * 30,
            "foreigners_official": 2000 + i * 10 - shock * 15,
            "money_market_funds": 500 + i * 5 + shock * 10,
        }
        for sector, holdings in values.items():
            rows.append({"date": date, "sector": sector, "instrument": "treasury", "holdings": holdings})
        rows.append({"date": date, "sector": "_total", "instrument": "treasury", "holdings": sum(values.values()) + 25.0})
    return pd.DataFrame(rows)


def _shock(n_quarters: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")
    return pd.DataFrame({"date": dates, "ati_baseline_bn": [float(i % 4) for i in range(n_quarters)]})


def test_load_shock_artifact_accepts_quarter(tmp_path):
    path = tmp_path / "shock.csv"
    pd.DataFrame({"quarter": ["2020Q1", "2020Q2"], "ati_baseline_bn": [10.0, -5.0]}).to_csv(path, index=False)
    shock = load_shock_artifact(path, "ati_baseline_bn")
    assert list(shock.columns) == ["date", "ati_baseline_bn"]
    assert shock["date"].iloc[0] == pd.Timestamp("2020-03-31")


def test_build_response_panel_adds_residual_from_total():
    config = HolderResponseConfig(min_observations=4)
    response, outcome_source, residual_method = build_response_panel(_panel(), _shock(), config)
    assert outcome_source == "holdings_change_proxy"
    assert residual_method == "source_total"
    assert "_residual" in set(response["sector"])


def test_estimate_holder_response_schema():
    config = HolderResponseConfig(min_observations=4, max_horizon_quarters=2)
    response, _, _ = build_response_panel(_panel(), _shock(), config)
    coefficients, cumulative = estimate_holder_response(response, config)
    assert {"sector", "horizon", "response", "ci_low", "ci_high", "claim_label"}.issubset(coefficients.columns)
    assert set(cumulative["horizon"]) == {2}


def test_run_and_write_outputs(tmp_path):
    config = HolderResponseConfig(min_observations=4, max_horizon_quarters=1)
    result = run_holder_response(_panel(), _shock(), config)
    paths = write_outputs(result, tmp_path, config=config, shock_path="shock.csv")
    assert (tmp_path / "holder_response_bundle.json").exists()
    assert (tmp_path / "sector_response_coefficients.csv").exists()
    bundle = json.loads((tmp_path / "holder_response_bundle.json").read_text())
    assert bundle["pipeline"] == "holder_response"
    assert bundle["residual_method"] == "source_total"
    assert "bundle" in paths


def test_missing_requested_control_is_not_silently_removed():
    import pytest

    with pytest.raises(ValueError, match="Requested controls unavailable"):
        run_holder_response(_panel(), _shock(), controls=["missing"])


def test_unmeasured_residual_stays_missing():
    panel = _panel().query("sector != '_total'")
    response, _, method = build_response_panel(panel, _shock())
    assert method == "unavailable_no_total"
    assert response.loc[response["sector"] == "_residual", "outcome"].isna().all()


def test_cumulative_horizon_does_not_bridge_deleted_quarter():
    from tsyparty.behavior.holder_response import _horizon_frame

    base = pd.DataFrame({"date": pd.to_datetime(["2020-03-31", "2020-09-30", "2020-12-31"]),
                         "outcome": [1., 3., 4.], "ati_baseline_bn": [1., 2., 3.]})
    frame = _horizon_frame(base, 1, HolderResponseConfig(), [])
    assert frame["date"].tolist() == [pd.Timestamp("2020-09-30")]
    assert frame["cumulative_outcome"].tolist() == [7.]


def test_duplicate_shock_quarters_are_rejected(tmp_path):
    import pytest

    path = tmp_path / "shock.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-03-31"], "ati_baseline_bn": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row per quarter"):
        load_shock_artifact(path, "ati_baseline_bn")
