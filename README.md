# tsyparty

`tsyparty` maps sector-level US Treasury holdings, quarterly holding changes, likely net counterparty flows, and sector behavior similarity from public Federal Reserve and US Treasury data. It also runs quarterly reduced-form regressions relating sector Treasury transactions to issuance maturity shifts.

## Data

`tsyparty download` fetches public files directly with no API keys required:

- Financial Accounts of the United States (Z.1) CSV archive (Table L.210 sector Treasury holdings) and From-Whom-to-Whom (FWTW) issuer-holder CSV (Federal Reserve).
- Enhanced Financial Accounts (EFA) historical CSVs for money market funds, commercial banks, and international portfolio holdings (Federal Reserve).
- Treasury International Capital (TIC) SLT foreign holdings tables and Investor Class Auction Allotments XLS files (US Treasury).
- Debt to the Penny JSON via the Fiscal Data API (US Treasury).
- System Open Market Account (SOMA) holdings JSON and Primary Dealer Statistics JSON (Federal Reserve Bank of New York).
- H.8 commercial bank balance-sheet CSV (Federal Reserve).

## Install

Requires Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

Run an offline toy example (writes matrices, charts, and diagnostics to `outputs/sample/`):

```bash
tsyparty example --out outputs/sample
```

Run the full public-data pipeline (download, parse, harmonize, validate, quarterly holding changes, primary-market shares, RAS counterparty inference, official/private foreign split, and sector similarity):

```bash
./reproduce.sh
```

Or run individual stages directly via the `tsyparty` CLI:

```bash
tsyparty registry --public-only
tsyparty download z1 fwtw investor_class debt_to_penny tic_slt efa soma
tsyparty parse-z1 data/raw_public/z1/z1_csv_files.zip --out data/interim
tsyparty parse-fwtw data/raw_public/fwtw/fwtw_data.csv --out data/interim
tsyparty parse-auction data/raw_public/auction/*.xls --out data/interim
tsyparty parse-debt data/raw_public/fiscaldata/debt_to_penny_api.json --out data/interim
tsyparty parse-tic data/raw_public/tic/slt1d_globl.csv --out data/interim
tsyparty parse-efa --efa-dir data/raw_public/efa --out data/interim
tsyparty parse-soma data/raw_public/soma/soma_holdings_page.json --out data/interim
tsyparty harmonize --interim data/interim --out data/derived
tsyparty validate --derived data/derived --interim data/interim --out outputs/validation
tsyparty baseline --derived data/derived --out outputs/baseline
tsyparty primary-market --interim data/interim --out outputs/primary_market
tsyparty infer --derived data/derived --out outputs/inference
tsyparty enrich-foreign --derived data/derived --tic-dir data/raw_public/tic --out data/derived
tsyparty similarity --derived data/derived --out outputs/similarity
```

Additional parsers exist for weekly H.8 (`parse-h8`) and primary dealer statistics (`parse-dealer`).

To serve the static dashboard under `site/` locally:

```bash
python3 -m http.server 8899
```

Then open `http://localhost:8899/site/index.html`.

### Holder-response and issuance-maturity regressions

Both quarterly regression commands require `--transaction-basis` when given transaction columns:

| `--transaction-basis` value | Meaning |
| --- | --- |
| `FA_SAAR_millions` | Seasonally adjusted annual rate in USD millions (divided by 4,000 to quarterly USD billions) |
| `FU_quarterly_millions` | Unadjusted quarterly flow in USD millions (divided by 1,000 to quarterly USD billions) |
| `prequarterized_billions` | Quarterly flow already in USD billions (unchanged) |

```bash
tsyparty holder-response \
  --panel-file path/to/sector_transactions.csv \
  --transaction-basis FU_quarterly_millions \
  --shock path/to/maturity_tilt_shock.csv \
  --out outputs/holder_response

tsyparty issuance-maturity-response \
  --auction-file path/to/auctions.csv \
  --sector-panel path/to/sector_transactions.csv \
  --transaction-basis FU_quarterly_millions \
  --no-factor-controls \
  --out outputs/holder_response/issuance_maturity_response
```

Key options:

- `holder-response`: `--controls` requires an explicit `--context-file` CSV; `--same-perimeter-total` computes the residual sector when the panel total matches the sector set, timing, and units of the individual sectors. Cumulative horizons use Newey-West HAC standard errors with `maxlags = max(horizon, 1)`.
- `issuance-maturity-response`: pass `--control-universe` to enable screened factor controls or `--no-factor-controls` to skip them, and optional `--sample-start-quarter YYYYQn` to set the estimation start while keeping earlier quarters for trailing expectations and lagged controls.
- `python -m tsyparty.behavior.response_inputs --help` builds quarterly unadjusted transaction inputs from a Z.1 zip archive and supplementary CSV using `configs/response_input_sources.yml`.

## Output

Raw downloads (`data/raw_public/`), parsed interim tables (`data/interim/`), derived panels (`data/derived/`), and analysis outputs (`outputs/`) are gitignored.

| Path | Description |
| --- | --- |
| `data/derived/harmonized_panel.csv` | Quarterly sector Treasury holdings panel (plus enriched foreign split and reconciliation CSVs) |
| `outputs/validation/` | EFA and TIC cross-check summary tables |
| `outputs/baseline/` | Quarterly holding changes, net buyer/seller margins, and PNG charts |
| `outputs/primary_market/` | Primary-market auction allocation shares by investor class |
| `outputs/inference/` | RAS dense and sparse net counterparty flow matrices (with residual bucket) |
| `outputs/similarity/` | Sector feature tables, distance matrix, rolling betas, and charts |
| `outputs/holder_response/` | Holder-response and issuance-maturity regression tables, JSON bundles, and charts |

## Limits

- Counterparty matrices from `infer` are iterative proportional fitting (RAS) estimates of likely net seller-to-buyer flows from quarterly holding changes; public data do not observe exact gross or bilateral trades, and an unexplained residual bucket is always retained.
- `holder-response` and `issuance-maturity-response` produce reduced-form descriptive associations rather than structural causal elasticities.
- In `issuance-maturity-response`, outcomes are shares of positive net non-Fed Treasury acquisition rather than shares of gross Treasury issuance. Missing quarters, missing required sector keys, or unmapped keys raise errors rather than being filled with zero. Screened factor-control specifications are exploratory and omit standard errors, confidence intervals, and p-values.

## Tests

```bash
python -B -m pytest -q
ruff check src tests scripts
```

## License

MIT. See [LICENSE](LICENSE).
