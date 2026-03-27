# AGENTS.md

This file is for coding agents and future contributors.

## Mission

Build `tsyparty` into a reproducible, public-data Treasury counterparty map with strong institutional grounding, clear claim discipline, and thesis-quality presentation.

## Non-negotiables

1. Keep the **public-data core runnable** without TRACE or other licensed inputs.
2. Keep the **canonical estimation layer quarterly** unless a module is explicitly event-frequency.
3. Separate **primary-market allocation** from **secondary-market reallocation**.
4. Never overclaim exact gross trades or exact immediate counterparties.
5. Preserve and display a **residual / unexplained bucket**.
6. Keep all transformations auditable through configs, manifests, and source notes.

## Frozen estimands

- Ultimate net seller to banks
- Ultimate net seller to foreigners
- Primary-market buyer composition
- Behavioral proximity to banks / foreigners

## Build order

### Phase 0
Freeze sector and instrument crosswalks in `configs/`.

### Phase 1
Complete public downloaders and raw snapshot manifests.

### Phase 2
Build stock and flow reconciliation tables.

### Phase 3
Ship descriptive baseline charts and tables.

### Phase 4
Implement three counterparty estimators:
- V1 sign baseline
- V2 entropy / RAS
- V3 sparse sensitivity

### Phase 5
Implement behavior module.

### Phase 6
Add event/regime overlays and presentation layer.

## Data discipline

- Every source gets an entry in `configs/sources.yml` and `docs/source_audit.md`.
- Every derived series gets a row in `docs/series_registry.csv`.
- Every caveat gets surfaced in `docs/claims_and_limits.md`.
- Optional / licensed inputs belong under `data/licensed_optional/` and must never be required by the core pipeline.

## Coding conventions

- Python 3.11+
- Type hints on public functions
- Prefer pure functions over side effects
- Use small, testable modules
- Keep notebooks lightweight; logic belongs in `src/`
- If a source is not stable, scrape only the landing page and record the discovered artifact URL

## Presentation conventions

- Every output chart should have a subtitle that states whether it is holdings, transactions, inferred seller shares, or behavior similarity.
- Counterparty charts must say **likely net counterparties**.
- Validation panels should always show source alignment and residuals.

## Definition of done for the first publishable version

1. Public-data download path works.
2. Reconciliation passes for at least one complete quarterly sample.
3. Seller-share estimates to banks and foreigners render from code.
4. Behavior similarity table renders.
5. README, methods note, and limitations note are consistent.
6. Paper and slide outlines reflect actual outputs.
