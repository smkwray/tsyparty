# tsyparty

Public-data Treasury counterparty inference and sector-behavior mapping.

**[Live Dashboard](https://smkwray.github.io/tsyparty/)** — interactive site with holdings charts, sector behavior heatmaps, factor-adjusted comovement, and pipeline status.

To serve locally with your own pipeline data:

```bash
python3 -m http.server 8899   # then open http://localhost:8899/site/index.html
```

`tsyparty` builds a reproducible, public-data-only map of who holds U.S. Treasury securities, how holdings flow between sectors, and which sectors behave most similarly. It answers four questions:

1. When banks add Treasury holdings, who are the most likely **net sellers**?
2. When foreigners add Treasury holdings, who are the most likely **net sellers**?
3. Who buys new Treasury supply in the **primary market**?
4. Which sectors behave most like banks or foreigners?

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick start

Run the full pipeline:

```bash
./reproduce.sh
```

Or step by step:

```bash
# Download public data
tsyparty download z1 fwtw investor_class debt_to_penny tic_slt efa

# Parse raw data
tsyparty parse-z1 data/raw_public/z1/z1_csv_files.zip --out data/interim
tsyparty parse-fwtw data/raw_public/fwtw/fwtw_data.csv --out data/interim
tsyparty parse-auction data/raw_public/auction/*.xls --out data/interim
tsyparty parse-debt data/raw_public/fiscaldata/debt_to_penny_api.json --out data/interim
tsyparty parse-tic data/raw_public/tic/slt1d_globl.csv --out data/interim
tsyparty parse-efa --efa-dir data/raw_public/efa --out data/interim

# Harmonize and reconcile
tsyparty harmonize --interim data/interim --out data/derived

# Validate against independent sources
tsyparty validate --derived data/derived --interim data/interim --out outputs/validation

# Descriptive analysis
tsyparty baseline --derived data/derived --out outputs/baseline
tsyparty primary-market --interim data/interim --out outputs/primary_market

# Counterparty inference
tsyparty infer --derived data/derived --out outputs/inference

# Behavior similarity
tsyparty similarity --derived data/derived --out outputs/similarity

# Holder-response mechanism artifacts
tsyparty holder-response \
  --derived data/derived \
  --shock path/to/qrawatch_maturity_tilt.csv \
  --out outputs/holder_response

# Issuance maturity response for descriptive mechanism analysis
tsyparty issuance-maturity-response \
  --auction-file path/to/auctions.csv \
  --sector-panel path/to/sector_panel.csv \
  --no-factor-controls \
  --out outputs/holder_response/issuance_maturity_response
```

## Available commands

| Command | Description |
|---------|-------------|
| `download` | Download public source data (Z.1, FWTW, TIC, EFA, Fiscal Data, SOMA, H.8, dealer) |
| `parse-z1` | Parse Z.1 Financial Accounts CSV zip (table L.210) |
| `parse-fwtw` | Parse FWTW issuer-holder CSV |
| `parse-auction` | Parse Treasury investor-class auction XLS files |
| `parse-debt` | Parse Fiscal Data API debt-to-penny JSON |
| `parse-tic` | Parse TIC SLT historical global CSV |
| `parse-efa` | Parse EFA data (MMF, banks, international) |
| `parse-soma` | Parse SOMA Treasury holdings JSON into weekly series and quarterly deltas |
| `parse-h8` | Parse H.8 bank balance sheet data into weekly series |
| `parse-dealer` | Parse primary dealer statistics JSON into weekly series |
| `harmonize` | Build harmonized quarterly panel and run reconciliation |
| `validate` | Cross-validate panel against EFA/TIC sources |
| `enrich-foreign` | Split foreign holdings into official vs private |
| `baseline` | Compute holding changes, margins, and composition charts |
| `primary-market` | Build primary-market allocation from auction data |
| `infer` | Run RAS/sparse counterparty inference with validation checks |
| `similarity` | Compute sector behavior distance, factor-adjusted comovement, absorption betas, and charts |
| `holder-response` | Estimate quarterly sector Treasury transaction responses to imported bill-heavy issuance / maturity-tilt shocks |
| `issuance-maturity-response` | Estimate sector absorption-share responses to whole-curve issuance weighted-average maturity changes |
| `show-plan` | Print the build sequence |
| `registry` | List configured public data sources |
| `example` | Generate example outputs from toy data |

## Public data sources

- **Z.1 Financial Accounts** — quarterly sector Treasury holdings (Fed)
- **FWTW** — from-whom-to-whom issuer-holder data (Fed)
- **Enhanced Financial Accounts** — MMF holdings, bank balance sheets, international holdings (Fed)
- **TIC / SLT** — foreign Treasury holdings by country (Treasury)
- **Investor Class Auction Allotments** — primary-market buyer composition (Treasury)
- **Debt to the Penny** — total debt outstanding (Fiscal Data API)
- **SOMA Holdings** — weekly Fed Treasury holdings by security type (NY Fed)
- **H.8** — weekly bank Treasury/agency securities (Fed via FRED)
- **Primary Dealer Statistics** — weekly dealer positions, financing, transactions (NY Fed)

## Method

1. **Harmonized panel**: Merge Z.1 and FWTW into a quarterly sector-level holdings panel with source-priority overlap resolution. Cross-validate against EFA and TIC.

2. **Reconciliation**: Compare sector totals against debt held by the public (Fiscal Data API), separating SOMA from private-sector holdings.

3. **Baseline flows**: Compute quarterly holding changes and buyer/seller margins per sector.

4. **Counterparty inference**: Use RAS (iterative proportional fitting) to estimate likely net seller-to-buyer flows, with a sparse sensitivity variant. The residual/unexplained bucket is always preserved.

5. **Behavior similarity**: Cosine distance over sector features (mean delta, volatility, holding shares) plus rolling factor-adjusted comovement from quarterly holding changes.

6. **Holder-response sidecar**: Imported QRA or maturity-tilt shock artifacts can be aligned to the quarterly sector panel to estimate reduced-form sector transaction responses. This is descriptive mechanism evidence for the buyer-mix channel, not a counterparty matrix and not a settled causal policy elasticity.

## Claims discipline

This tool labels outputs as **likely net counterparties**, not exact bilateral trades. It:

- separates primary-market allocation from secondary-market reallocation,
- preserves and displays a residual bucket in all inference outputs,
- does not claim exact gross trades or exact immediate counterparties.

## Repository layout

```text
src/tsyparty/
  ingest/         # Downloaders and parsers for each data source
  reconcile/      # Harmonization, accounting, enrichment
  baseline/       # Holding changes, buyer/seller flows, primary market
  infer/          # RAS counterparty matrix estimation
  behavior/       # Sector similarity, holder-response, and issuance-maturity response sidecars
  validate/       # Cross-validation against independent sources
  viz/            # Chart generation
  cli.py          # CLI entry points
configs/          # Sector, source, inference, and instrument configs
tests/            # Unit, contract, and CLI checks
data/             # Raw, interim, and derived data (gitignored)
outputs/          # Charts, CSVs, and inference results (gitignored)
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT

Holder response requires an explicit `--shock` file and uses `--panel-file` when
provided, otherwise `harmonized_panel.csv` under `--derived`. Requested `--controls`
require an explicit `--context-file` CSV with a date column. No debt context is
discovered automatically. A residual is available only with `--same-perimeter-total`,
which attests that the panel's total has the same holder perimeter, clock, and
transaction basis. Broad public debt changes cannot define that residual.

Both response commands require an explicit `--transaction-basis` for transaction
inputs: `FA_SAAR_millions` divides annualized millions by 4000,
`FU_quarterly_millions` divides quarterly millions by 1000, and
`prequarterized_billions` uses the supplied quarterly billions unchanged. Shipped
configuration leaves the basis unset; it does not certify historical inputs.
Holdings-change proxies use separate holdings units and are labeled as proxies in
coefficient rows. Holder cumulative horizons use HAC with `maxlags=max(horizon,1)`.

Maturity outcomes measure shares of **non-Fed positive net acquisition**, not shares
of Treasury issuance. The configured included groups and excluded keys exhaust the
input crosswalk. Unknown keys, overlapping mappings, and missing included keys in
any quarter fail before shares are built; an absent row is not an observed zero.
The bundle records included groups, exclusions, and quarter coverage.

For factor robustness, supply `--control-universe`; `--no-factor-controls`
explicitly omits that tier. Same-sample screening and factor extraction make the
factor tier exploratory: it reports point estimates and omits standard errors,
intervals, and p-values. Missing core controls are reported, and a partial set is
not labeled as the full model. Quarterly gaps are rejected by the maturity-response
estimator. These diagnostics require a separate evidence review before use as
empirical findings.


Frozen response inputs can be reconstructed with
`python -m tsyparty.behavior.response_inputs --help` using
`configs/response_input_sources.yml`. The builder preserves each FU source cell,
requires complete requested quarters, checks primary table units and the exhaustive
crosswalk, and records input hashes and the committed producer. Supplementary
credit-union cells retain their own source and retrieval date. It never refreshes
data or replaces missing observations with zero.

The issuance-maturity command accepts `--sample-start-quarter YYYYQn` for an explicit
estimation and factor-screening sample. Treatment expectations and lagged controls
retain earlier input history. This option does not fill missing control quarters;
calendar-contiguity checks still apply to every fitted sample.
