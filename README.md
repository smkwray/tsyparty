# tsyparty

`tsyparty` maps quarterly US Treasury holdings by sector from public Federal Reserve and US Treasury data. It computes holding changes, estimates likely net flows from sellers to buyers by iterative proportional fitting (RAS), and measures how alike sectors behave. It also runs quarterly reduced-form regressions of sector Treasury transactions on shifts in issuance maturity. A dashboard of the results is published at [smkwray.github.io/tsyparty](https://smkwray.github.io/tsyparty/).

## Data

`tsyparty download` fetches public files directly, with no API keys:

- Financial Accounts of the United States (Z.1) CSV archive (Table L.210 sector Treasury holdings) and the From-Whom-to-Whom (FWTW) issuer-holder CSV, from the Federal Reserve.
- Enhanced Financial Accounts (EFA) historical CSVs for money market funds, commercial banks and international portfolio holdings, from the Federal Reserve.
- Treasury International Capital (TIC) SLT tables of foreign holdings of long-term US securities, and investor-class auction allotment XLS files, from the US Treasury.
- Debt to the Penny JSON from the Treasury Fiscal Data API.
- System Open Market Account (SOMA) holdings and Primary Dealer Statistics JSON, from the Federal Reserve Bank of New York.
- The Federal Reserve's weekly H.8 commercial bank balance-sheet CSV.

## Install

Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

Run an offline toy example, which writes matrices, charts and diagnostics to `outputs/sample/`:

```bash
tsyparty example --out outputs/sample
```

Run the full public-data pipeline: download, parse, harmonize, check against EFA and TIC, compute holding changes and primary-market shares, infer counterparty flows, split foreign holdings into official and private, and measure sector similarity:

```bash
./reproduce.sh
```

`reproduce.sh` lists the command for each stage, and `tsyparty --help` lists every command, including parsers for weekly H.8 (`parse-h8`) and primary dealer statistics (`parse-dealer`). The dashboard in `site/` reads a committed copy of the outputs. To serve it locally, run `python3 -m http.server 8899` and open `http://localhost:8899/site/index.html`.

### Holder-response and issuance-maturity regressions

Both regressions require `--transaction-basis`, which states the units, when the input has reported transaction columns:

| `--transaction-basis` | Meaning |
| --- | --- |
| `FA_SAAR_millions` | Seasonally adjusted annual rate in USD millions (divided by 4,000 to give quarterly USD billions) |
| `FU_quarterly_millions` | Unadjusted quarterly flow in USD millions (divided by 1,000) |
| `prequarterized_billions` | Quarterly flow already in USD billions |

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

In `holder-response`, `--controls` requires a `--context-file` CSV, and `--same-perimeter-total` computes a residual sector when the panel's total covers the same sectors, timing and units as the listed sectors. Cumulative horizons use Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) standard errors with `maxlags = max(horizon, 1)`. In `issuance-maturity-response`, pass `--control-universe` for screened factor controls or `--no-factor-controls` to skip them, and `--sample-start-quarter YYYYQn` to start estimation later while keeping earlier quarters for lags. `python -m tsyparty.behavior.response_inputs --help` builds quarterly unadjusted transaction inputs from a Z.1 archive using `configs/response_input_sources.yml`.

## Output

Raw downloads (`data/raw_public/`), parsed tables (`data/interim/`), derived panels (`data/derived/`) and analysis outputs (`outputs/`) are ignored by Git.

| Path | Contents |
| --- | --- |
| `data/derived/harmonized_panel.csv` | Quarterly sector Treasury holdings, with the enriched foreign split and reconciliation CSVs beside it |
| `outputs/validation/` | Cross-checks against EFA and TIC |
| `outputs/baseline/` | Quarterly holding changes, net buyer and seller margins, and PNG charts |
| `outputs/primary_market/` | Auction allotment shares by investor class |
| `outputs/inference/` | RAS net counterparty flow matrices, dense and sparse, with a residual bucket |
| `outputs/similarity/` | Sector feature tables, distance matrix, rolling betas and charts |
| `outputs/holder_response/` | Regression tables, JSON bundles and charts |

## Limits

- Public data record holdings by sector, not trades. The RAS matrices are estimates of likely net seller-to-buyer flows from quarterly holding changes, not observed gross or bilateral trades, and they always keep an unexplained residual bucket.
- `holder-response` and `issuance-maturity-response` estimate reduced-form associations, not structural causal elasticities.
- In `issuance-maturity-response`, outcomes are shares of positive net non-Fed Treasury acquisition, not shares of gross issuance. Missing quarters or unmapped sector keys raise errors instead of being filled with zero. Specifications with screened factor controls are exploratory and report no standard errors, confidence intervals or p-values.

## Tests

```bash
python -B -m pytest -q
ruff check src tests scripts
```

## License

MIT. See [LICENSE](LICENSE).
