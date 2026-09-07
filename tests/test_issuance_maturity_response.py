import json

import pandas as pd

from tsyparty.behavior.issuance_maturity_response import (
    IssuanceMaturityResponseConfig,
    build_issuance_maturity_treatment,
    build_outcome_panel,
    build_sector_flows,
    run_issuance_maturity_response,
    write_outputs,
)


def _auctions(n_quarters: int = 24) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2018-03-31", periods=n_quarters, freq="QE")
    for i, q_end in enumerate(dates):
        q_start = q_end - pd.offsets.QuarterBegin(startingMonth=q_end.month)
        short_amt = 90_000_000_000 + i * 1_000_000_000
        long_amt = 40_000_000_000 + (i % 5) * 12_000_000_000
        rows.extend(
            [
                {
                    "issue_date": q_start + pd.Timedelta(days=10),
                    "maturity_date": q_start + pd.Timedelta(days=192),
                    "total_accepted": short_amt,
                },
                {
                    "issue_date": q_start + pd.Timedelta(days=40),
                    "maturity_date": q_start + pd.Timedelta(days=3652),
                    "total_accepted": long_amt,
                },
            ]
        )
    return pd.DataFrame(rows)


def _sector_panel(n_quarters: int = 24) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2018-03-31", periods=n_quarters, freq="QE")
    keys = {
        "bank_us_chartered": 20.0,
        "bank_foreign_banking_offices_us": 6.0,
        "bank_us_affiliated_areas": 2.0,
        "foreigners_total": 30.0,
        "money_market_funds": 18.0,
        "mutual_funds": 12.0,
        "exchange_traded_funds": 4.0,
        "security_brokers_and_dealers": 8.0,
        "life_insurers": 5.0,
        "households_nonprofits": 14.0,
    }
    for i, date in enumerate(dates):
        cycle = float((i % 5) - 2)
        for key, base in keys.items():
            bank_like = key.startswith("bank_") or key == "foreigners_total"
            transactions = (base + cycle * (3.0 if bank_like else -1.5)) * 1000.0
            rows.append({"date": date, "sector_key": key, "transactions": transactions})
    return pd.DataFrame(rows)


def test_treatment_uses_whole_curve_wam_gap():
    config = IssuanceMaturityResponseConfig(min_observations=8)
    treatment = build_issuance_maturity_treatment(_auctions(), config)
    assert {"issuance_wam_years", "issuance_wam_gap_years", "bill_share", "long_7y_plus_share"}.issubset(treatment.columns)
    assert treatment["issuance_wam_years"].notna().all()
    assert treatment["issuance_wam_gap_years"].notna().sum() > 10


def test_sector_flow_groups_include_bank_components():
    flows = build_sector_flows(_sector_panel())
    latest = flows[flows["date"] == flows["date"].min()].set_index("sector")
    banks = latest.loc["Banks", "transactions_bn"]
    assert banks == 10.0
    assert "Foreign holders" in set(flows["sector"])


def test_outcome_panel_reports_absorption_shares_with_denominator():
    config = IssuanceMaturityResponseConfig(min_observations=8)
    treatment = build_issuance_maturity_treatment(_auctions(), config)
    flows = build_sector_flows(_sector_panel())
    outcomes = build_outcome_panel(flows, treatment, config)
    assert {"value", "flow_bn", "positive_absorption_bn", "issuance_wam_gap_years"}.issubset(outcomes.columns)
    assert "Banks + foreign holders absorption share" in set(outcomes["outcome"])
    assert outcomes["positive_absorption_bn"].gt(0).all()


def test_run_and_write_outputs(tmp_path):
    config = IssuanceMaturityResponseConfig(min_observations=8, factor_controls_enabled=False)
    result = run_issuance_maturity_response(_auctions(), _sector_panel(), config)
    assert not result.estimates.empty
    assert "dependent_variable" in result.estimates.columns
    paths = write_outputs(result, tmp_path)
    assert (tmp_path / "maturity_response_bundle.json").exists()
    assert (tmp_path / "maturity_response_estimates.csv").exists()
    bundle = json.loads((tmp_path / "maturity_response_bundle.json").read_text())
    assert bundle["pipeline"] == "issuance_maturity_response"
    assert "maturity_response_bundle.json" in paths


def test_offering_amount_fallback_without_total_accepted():
    auctions = _auctions().rename(columns={"total_accepted": "offering_amt"})
    result = build_issuance_maturity_treatment(auctions, IssuanceMaturityResponseConfig())
    assert result["issuance_volume_bn"].gt(0).all()


def test_missing_core_controls_do_not_get_full_model_label():
    config = IssuanceMaturityResponseConfig(min_observations=8, factor_controls_enabled=False)
    result = run_issuance_maturity_response(_auctions(), _sector_panel(), config)
    assert set(result.estimates["model"]) == {"no_controls"}
    assert "dgs10_l1" in result.design_summary["missing_core_controls"]
    assert result.placebo.empty


def test_missing_factor_input_fails_when_enabled(tmp_path):
    import pytest
    from tsyparty.behavior.issuance_maturity_response import build_factor_controls

    with pytest.raises(ValueError, match="factor-control universe is unavailable"):
        build_factor_controls(pd.DataFrame(), tmp_path / "absent.csv", IssuanceMaturityResponseConfig())


def test_calendar_gap_is_rejected_before_hac_or_horizon():
    import pytest
    from tsyparty.behavior.issuance_maturity_response import _build_cumulative_absorption_share

    sub = pd.DataFrame({"date": pd.to_datetime(["2020-03-31", "2020-09-30"]),
                        "flow_bn": [1., 3.], "positive_absorption_bn": [10., 10.]})
    with pytest.raises(ValueError, match="contiguous"):
        _build_cumulative_absorption_share(sub, 1, 100.)
