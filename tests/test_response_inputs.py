import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tsyparty.behavior.response_inputs import transaction_input


def _inputs(tmp_path):
    contract = yaml.safe_load(Path('configs/response_input_sources.yml').read_text())
    codes = {code for sources in contract['included_series'].values() for code in sources}
    primary = sorted(codes - set(contract['supplement_series']))
    columns = primary + sorted(contract['excluded_table_series'])
    table = pd.DataFrame([['2025:Q3', *([1] * len(columns))], ['2025:Q4', *([2] * len(columns))]], columns=['date', *columns])
    dictionary = '\n'.join(f'{code}\tdescription\tline\ttable\t{contract["table_units"]}' for code in primary)
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(contract['table_member'], table.to_csv(index=False))
        z.writestr(contract['dictionary_member'], dictionary)
    extra = pd.DataFrame([{'date': date, 'series_code': code, 'value': 3, 'provider': 'fred', 'vintage': '2026-03-23'}
                          for code in contract['supplement_series'] for date in ['2025-09-30', '2025-12-31']])
    supplement = tmp_path / 'supplement.csv'
    extra.to_csv(supplement, index=False)
    return archive, supplement, contract


def test_response_input_preserves_source_cells_and_composite(tmp_path):
    archive, supplement, contract = _inputs(tmp_path)
    panel, cells = transaction_input(archive, supplement, contract, '2025Q3', '2025Q4')
    assert len(panel) == 25 * 2 and len(cells) == 26 * 2
    assert panel.query("sector_key == 'credit_unions_marketable_proxy'").transactions.eq(6).all()
    assert cells.series_code.str.startswith('FU').all()


def test_response_input_missing_component_blocks_build(tmp_path):
    archive, supplement, contract = _inputs(tmp_path)
    extra = pd.read_csv(supplement).iloc[1:]
    extra.to_csv(supplement, index=False)
    with pytest.raises(ValueError, match='Incomplete finite source coverage'):
        transaction_input(archive, supplement, contract, '2025Q3', '2025Q4')


def test_response_input_rejects_wrong_units_or_unmapped_series(tmp_path):
    archive, supplement, contract = _inputs(tmp_path)
    contract['table_units'] = 'annual rates'
    with pytest.raises(ValueError, match='incompatible source units'):
        transaction_input(archive, supplement, contract, '2025Q3', '2025Q4')
    contract['table_units'] = 'Millions of dollars; transactions, not seasonally adjusted'
    contract['excluded_table_series'].pop(next(iter(contract['excluded_table_series'])))
    with pytest.raises(ValueError, match='crosswalk is not exhaustive'):
        transaction_input(archive, supplement, contract, '2025Q3', '2025Q4')
