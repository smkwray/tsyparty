import json

import numpy as np
import pandas as pd
import pytest

from tsyparty.behavior.issuance_maturity_response import (
    DEFAULT_SECTOR_GROUPS,
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
    for date in dates:
        for key in {key for keys in DEFAULT_SECTOR_GROUPS.values() for key in keys}.difference(keys):
            rows.append({"date": date, "sector_key": key, "transactions": 0.0})
    return pd.DataFrame(rows)


def test_treatment_uses_whole_curve_wam_gap():
    config = IssuanceMaturityResponseConfig(transaction_basis="FU_quarterly_millions", min_observations=8)
    treatment = build_issuance_maturity_treatment(_auctions(), config)
    assert {"issuance_wam_years", "issuance_wam_gap_years", "bill_share", "long_7y_plus_share"}.issubset(treatment.columns)
    assert treatment["issuance_wam_years"].notna().all()
    assert treatment["issuance_wam_gap_years"].notna().sum() > 10


def test_sector_flow_groups_include_bank_components():
    flows = build_sector_flows(_sector_panel(), transaction_basis="FU_quarterly_millions")
    latest = flows[flows["date"] == flows["date"].min()].set_index("sector")
    banks = latest.loc["Banks", "transactions_bn"]
    assert banks == 10.0
    assert "Foreign holders" in set(flows["sector"])


def test_outcome_panel_reports_absorption_shares_with_denominator():
    config = IssuanceMaturityResponseConfig(transaction_basis="FU_quarterly_millions", min_observations=8)
    treatment = build_issuance_maturity_treatment(_auctions(), config)
    flows = build_sector_flows(_sector_panel(), transaction_basis="FU_quarterly_millions")
    outcomes = build_outcome_panel(flows, treatment, config)
    assert {"value", "flow_bn", "positive_absorption_bn", "issuance_wam_gap_years"}.issubset(outcomes.columns)
    assert "Banks + foreign holders share of non-Fed positive net acquisition" in set(outcomes["outcome"])
    assert outcomes["positive_absorption_bn"].gt(0).all()


def test_run_and_write_outputs(tmp_path):
    config = IssuanceMaturityResponseConfig(transaction_basis="FU_quarterly_millions", min_observations=8, factor_controls_enabled=False)
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
    config = IssuanceMaturityResponseConfig(transaction_basis="FU_quarterly_millions", min_observations=8, factor_controls_enabled=False)
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


def test_unmapped_sector_key_rejected():
    panel = _sector_panel()
    extra = panel.iloc[[0]].assign(sector_key="unknown_holder")
    with pytest.raises(ValueError, match="Unmapped sector keys"):
        build_sector_flows(pd.concat([panel, extra]), transaction_basis="FU_quarterly_millions")


def test_missing_denominator_group_blocks_share():
    panel = _sector_panel()
    missing = panel.drop(panel.query("sector_key == 'money_market_funds'").index[0])
    with pytest.raises(ValueError, match="Incomplete denominator coverage"):
        build_sector_flows(missing, transaction_basis="FU_quarterly_millions")
    flows = build_sector_flows(panel, transaction_basis="FU_quarterly_millions")
    flows = flows.drop(flows.query("sector == 'Money funds'").index[0])
    with pytest.raises(ValueError, match="Incomplete denominator group coverage"):
        build_outcome_panel(flows, pd.DataFrame(), IssuanceMaturityResponseConfig())


def test_sector_crosswalk_rejects_overlap():
    groups = {**DEFAULT_SECTOR_GROUPS, "Duplicate banks": ["bank_us_chartered"]}
    with pytest.raises(ValueError, match="crosswalk overlap"):
        build_sector_flows(_sector_panel(), groups, transaction_basis="FU_quarterly_millions")
    with pytest.raises(ValueError, match="crosswalk overlap"):
        build_sector_flows(_sector_panel(), transaction_basis="FU_quarterly_millions", excluded_sector_keys=["fed", "bank_us_chartered"])


def test_z1_fa_saar_is_quarterized():
    panel = _sector_panel()
    fa = build_sector_flows(panel, transaction_basis="FA_SAAR_millions")
    fu = build_sector_flows(panel, transaction_basis="FU_quarterly_millions")
    bn = build_sector_flows(panel.assign(transactions=panel["transactions"] / 1000), transaction_basis="prequarterized_billions")
    np.testing.assert_allclose(fa["transactions_bn"] * 4, fu["transactions_bn"])
    np.testing.assert_allclose(bn["transactions_bn"], fu["transactions_bn"])
    assert fu.attrs["coverage"]["unmapped_row_count"] == 0
    assert len(fu.attrs["coverage"]["quarter_coverage"]) == 24
    assert set(fu.attrs["coverage"]["included_groups"]) == set(DEFAULT_SECTOR_GROUPS)
    fed = panel.iloc[[0]].assign(sector_key="fed", transactions=1e10)
    excluded = build_sector_flows(pd.concat([panel, fed]), transaction_basis="FU_quarterly_millions")
    np.testing.assert_allclose(excluded["transactions_bn"], fu["transactions_bn"])


@pytest.mark.parametrize("basis", [None, "unknown", "millions"])
def test_unknown_transaction_basis_rejected(basis):
    with pytest.raises(ValueError, match="transaction_basis"):
        build_sector_flows(_sector_panel(), transaction_basis=basis)


def test_mixed_transaction_basis_rejected():
    panel = _sector_panel().assign(transaction_basis="FA_SAAR_millions")
    panel.loc[0, "transaction_basis"] = "FU_quarterly_millions"
    with pytest.raises(ValueError, match="conflicting transaction_basis"):
        build_sector_flows(panel, transaction_basis="FA_SAAR_millions")


def test_factor_models_do_not_emit_naive_inference(tmp_path):
    from tsyparty.behavior.issuance_maturity_response import estimate_local_projections

    config = IssuanceMaturityResponseConfig(transaction_basis="FU_quarterly_millions", min_observations=8,
                                            core_controls=[], horizons=[0, 2], k_grid=[1], factor_controls_enabled=False)
    result = run_issuance_maturity_response(_auctions(), _sector_panel(), config)
    factors = result.treatment_panel[["date"]].copy()
    factors["factor_k1_1"] = np.sin(np.arange(len(factors)))
    estimates = estimate_local_projections(result.outcome_panel, config, factor_controls=factors)
    selected = estimates.query("model == 'factor_controls_k1'")
    fixed = estimates.query("model == 'no_controls'")
    assert not selected.empty and selected["beta_pp_per_1y"].notna().all()
    assert selected[["std_error", "ci_low", "ci_high", "p_value"]].isna().all().all()
    assert selected["inference_status"].eq("exploratory_post_selection").all()
    assert fixed["std_error"].notna().all()
    assert estimates["dependent_variable"].str.contains("non-Fed positive net acquisition").all()
    assert estimates["outcome"].str.contains("non-Fed positive net acquisition").all()
    assert estimates["transaction_basis"].eq("FU_quarterly_millions").all()
    assert not result.placebo.empty
    assert result.placebo["transaction_basis"].eq("FU_quarterly_millions").all()
    assert result.placebo["maxlags"].eq(result.placebo["placebo_lead_quarters"].clip(lower=1)).all()
    result.estimates = estimates
    write_outputs(result, tmp_path)
    saved = pd.read_csv(tmp_path / "maturity_response_estimates.csv")
    assert saved.query("model == 'factor_controls_k1'")[["std_error", "ci_low", "ci_high", "p_value"]].isna().all().all()
    bundle = json.loads((tmp_path / "maturity_response_bundle.json").read_text())
    assert "point estimates only" in bundle["selected_factor_inference"]
    assert bundle["transaction_basis"] == "FU_quarterly_millions"
    assert bundle["denominator_perimeter"] == "non_fed_positive_net_acquisition"
