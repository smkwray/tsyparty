# tsyparty

Public-data Treasury counterparty inference and sector-behavior mapping.

`tsyparty` is a seeded research repo for building a defensible, presentation-ready project that asks:

- when banks add Treasury holdings, who are the most likely **ultimate net sellers**,
- when foreigners add Treasury holdings, who are the most likely **ultimate net sellers**,
- who buys new Treasury supply in the **primary market**,
- which sectors behave most like banks or foreigners across issuance and balance-sheet regimes.

The repo is designed to stay **public-data-first**, **quarterly-canonical**, and **institutionally transparent**. It treats exact gross bilateral trades and immediate dealer counterparties as outside the claim set of the public backbone, while still producing useful estimates of likely net counterparties and marginal absorbers.

## Why this repo exists

The broader thesis draft already contains a Treasury-deposit-component accounting framework, a stock-flow-consistent simulation, and a large mixed-frequency econometrics program. This repo isolates the Treasury-holder / counterparty module so it can be built, defended, and presented as a standalone research product.

## Frozen estimands

1. **Ultimate net seller to banks** by quarter and instrument bucket.
2. **Ultimate net seller to foreigners** by quarter and instrument bucket.
3. **Primary-market buyer composition** by investor class and instrument bucket.
4. **Behavioral proximity**: which sectors absorb supply like banks or foreigners.

## Core method stack

1. **Accounting and reconciliation layer**
   - debt held by the public
   - SOMA/Fed holdings
   - sector Treasury holdings
   - issuance, maturities, and runoff

2. **Transparent baseline**
   - sector holding changes
   - issuance mix
   - auction allotments
   - dealer bridge context
   - foreign official/private changes

3. **Two-layer counterparty estimator**
   - primary-market matrix: Treasury -> sectors
   - secondary-market matrix: sector -> sector

4. **Main inference approach**
   - sign-based baseline attribution
   - constrained entropy / RAS / IPFP reconstruction
   - sparse sensitivity variant
   - explicit residual bucket

5. **Behavior module**
   - supply-absorption betas
   - rolling similarity scores
   - bank-closeness and foreign-closeness rankings
   - clustering / distance views

## Public-data backbone

- Federal Reserve **Z.1 Financial Accounts**
- Federal Reserve **FWTW** issuer-holder data
- Federal Reserve **Enhanced Financial Accounts (EFA)**
- Treasury **TIC / expanded SLT**
- Treasury **Investor Class Auction Allotments**
- FiscalData **Debt to the Penny** and DTS endpoints
- SOMA holdings history
- H.4.1, H.8, Primary Dealer Statistics
- FFIEC Call Reports
- SEC N-MFP and N-PORT

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

tsyparty show-plan
tsyparty example --out outputs/sample
tsyparty registry --public-only
```

To start ingestion work:

```bash
python scripts/download_all_public.py --dest data/raw_public
```

## Repository layout

```text
tsyparty/
  AGENTS.md
  README.md
  pyproject.toml
  Makefile
  reproduce.sh
  configs/
  data/
  docs/
  notebooks/
  outputs/
  paper/
  scripts/
  slides/
  src/tsyparty/
  tests/
```

## What this repo will and will not claim

Allowed:

- likely ultimate **net** sellers to banks or foreigners,
- primary-market buyer mix,
- regime-dependent marginal absorbers,
- similarity rankings and clusters.

Not allowed:

- exact gross bilateral trades,
- exact immediate dealer counterparties,
- exact beneficial foreign owners,
- literal interpretation of every household residual move.

## Build sequence

1. Freeze crosswalks and estimands.
2. Land downloaders and source manifests.
3. Build reconciliation checks.
4. Publish descriptive baseline.
5. Implement counterparty matrix V1/V2/V3.
6. Add behavior module.
7. Add event and presentation layers.
8. Write paper and slides.

## Included seed content

- build plan
- methods memo
- source registry
- sector and instrument config files
- downloader stubs for all core public sources
- counterparty matrix code
- similarity code
- reconciliation helpers
- toy sample data and tests
- paper and slides outlines
- coding-agent instructions in `AGENTS.md`

## Recommended first milestone

Produce three charts from public data only:

1. quarterly Treasury holding changes by sector,
2. seller-share estimate to banks,
3. seller-share estimate to foreigners.
