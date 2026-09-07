import json

import numpy as np
import pandas as pd
import pytest

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
    config = HolderResponseConfig(min_observations=4, total_matches_holder_perimeter=True)
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
    config = HolderResponseConfig(min_observations=4, max_horizon_quarters=1, total_matches_holder_perimeter=True)
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


def test_overlapping_horizons_use_hac(monkeypatch, tmp_path):
    from statsmodels.regression.linear_model import OLS

    calls = []
    original = OLS.fit

    def capture(self, *args, **kwargs):
        calls.append((kwargs["cov_type"], kwargs["cov_kwds"]["maxlags"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OLS, "fit", capture)
    config = HolderResponseConfig(min_observations=4, max_horizon_quarters=3)
    result = run_holder_response(_panel(), _shock(), config)
    assert calls and set(calls) == {("HAC", 1), ("HAC", 2), ("HAC", 3)}
    assert result.coefficients["covariance"].eq("HAC").all()
    assert result.coefficients["maxlags"].eq(result.coefficients["horizon"].clip(lower=1)).all()
    write_outputs(result, tmp_path, config)
    bundle = json.loads((tmp_path / "holder_response_bundle.json").read_text())
    assert bundle["covariance"] == "HAC"
    assert bundle["maxlags_by_horizon"] == {"0": 1, "1": 1, "2": 2, "3": 3}


def test_holdings_proxy_is_not_labeled_transaction_flow():
    result = run_holder_response(_panel(), _shock(), HolderResponseConfig(min_observations=4))
    assert result.coefficients["outcome"].str.endswith("treasury_holdings_change_proxy").all()
    assert result.coefficients["transaction_basis"].eq("not_applicable_holdings_change").all()
    assert result.response_panel["outcome_units"].eq("quarterly_billions").all()


def test_broad_debt_change_is_not_holder_residual():
    context = _shock().rename(columns={"ati_baseline_bn": "net_public_supply"})
    response, _, method = build_response_panel(_panel(), _shock(), context=context)
    assert method == "unavailable_no_total"
    assert response.loc[response["sector"] == "_residual", "outcome"].isna().all()


def test_same_perimeter_total_can_define_residual():
    panel = _panel().rename(columns={"holdings": "transactions"})
    config = HolderResponseConfig(transaction_basis="FU_quarterly_millions", total_matches_holder_perimeter=True)
    response, _, method = build_response_panel(panel, _shock(), config)
    assert method == "source_total"
    np.testing.assert_allclose(response.loc[response["sector"] == "_residual", "outcome"], .025)
    missing = panel.drop(panel.query("sector == 'banks'").index[0])
    incomplete, _, _ = build_response_panel(missing, _shock(), config)
    assert pd.isna(incomplete.query("sector == '_residual'")["outcome"].iloc[0])


@pytest.mark.parametrize("basis,divisor", [("FA_SAAR_millions", 4000), ("FU_quarterly_millions", 1000), ("prequarterized_billions", 1)])
def test_holder_transaction_basis_conversion(basis, divisor):
    panel = _panel().rename(columns={"holdings": "transactions"})
    config = HolderResponseConfig(transaction_basis=basis, min_observations=4)
    result = run_holder_response(panel, _shock(), config)
    expected = panel.query("sector == 'banks'")["transactions"].to_numpy() / divisor
    np.testing.assert_allclose(result.response_panel.query("sector == 'banks'")["outcome"], expected)
    assert result.coefficients["transaction_basis"].eq(basis).all()
    assert result.coefficients["outcome"].str.endswith("treasury_transaction_flow").all()


def test_holder_unknown_and_mixed_transaction_basis_rejected():
    panel = _panel().rename(columns={"holdings": "transactions"})
    for basis in [None, "unknown"]:
        with pytest.raises(ValueError, match="transaction_basis"):
            build_response_panel(panel, _shock(), HolderResponseConfig(transaction_basis=basis))
    panel["transaction_basis"] = "FA_SAAR_millions"
    with pytest.raises(ValueError, match="conflicting transaction_basis"):
        build_response_panel(panel, _shock(), HolderResponseConfig(transaction_basis="FU_quarterly_millions"))


@pytest.mark.parametrize("attestation", ["false", "true", 1, None])
def test_same_perimeter_total_requires_literal_boolean(attestation):
    config = HolderResponseConfig(total_matches_holder_perimeter=attestation)
    with pytest.raises(ValueError, match="literal boolean"):
        build_response_panel(_panel(), _shock(), config)
