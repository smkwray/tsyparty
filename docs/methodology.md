# Methodology

## What is being estimated

This repo targets **ultimate net counterparties**, not exact gross bilateral trades.

That means the main estimand is:

- if a target sector increases Treasury holdings over a quarter,
- and another set of sectors reduces holdings over that quarter,
- which sectors are the most likely net sources of that increase,
- after accounting for primary issuance, maturities, and central-bank balance-sheet effects.

## Why not use a pure reduced-form screen as the main method

Reduced-form rolling VAR / VECM / mixed-frequency screeners are useful for identifying co-movers and regime shifts, but they do not impose the accounting structure that a counterparty question needs.

The core object here is a matrix with row and column constraints. That argues for an accounting-first estimator.

## Canonical accounting identity

For sector `i`, quarter `t`, instrument bucket `k`:

\[
\Delta H_{i,t,k}
=
A_{i,t,k}
+
\sum_j X_{j \to i,t,k}
-
\sum_j X_{i \to j,t,k}
-
M_{i,t,k}
\]

Where:

- `ΔH` = observed holdings change,
- `A` = primary-market allocation from the Treasury,
- `X` = secondary-market sector-to-sector reallocation,
- `M` = maturities, redemptions, runoff, and other exits from the holder bucket.

This immediately implies a **two-layer model**:

1. Treasury -> sectors (primary market)
2. sectors -> sectors (secondary reallocation)

## Estimation sequence

### Step 1 — baseline descriptive accounting

Before matrix inference, compute:

- holdings changes by sector,
- issuance totals by bucket,
- SOMA changes,
- foreign official/private changes,
- auction buyer shares,
- dealer inventory context.

This is the first truth layer.

### Step 2 — sign baseline

Construct buyer and seller margins from net changes:

- buyers are sectors with positive net additions after primary and maturity adjustments,
- sellers are sectors with negative net changes after the same adjustments.

Allocate buyer inflows across seller outflows proportionally to available sales.

This is not the final estimator, but it creates a transparent baseline.

### Step 3 — constrained entropy / RAS

Given:

- row sums = seller outflows,
- column sums = buyer inflows,
- a support mask for allowed links,
- an optional prior matrix,

estimate `X` by minimizing divergence from the prior while satisfying market-clearing constraints.

The default objective is relative entropy:

\[
\min_{X \ge 0} \sum_{i,j} x_{ij}\log\left(\frac{x_{ij}}{\pi_{ij}}\right)
\]

subject to row and column sums.

Operationally, this is implemented with IPFP / RAS.

### Step 4 — sparse sensitivity

Maximum-entropy methods can generate unrealistically dense networks. To guard against that, threshold small cells and rebalance with the same margins.

Report:

- dense entropy estimate,
- sparse sensitivity estimate,
- residual and constraint diagnostics.

## Priors

Acceptable priors include:

- lagged inferred matrices,
- FWTW levels,
- auction investor-class shares,
- structural zeros from institutional logic.

Avoid using a prior that simply hard-codes the desired result.

## Behavioral similarity module

Counterparty inference and behavioral similarity are separate modules.

For behavior, estimate sector response vectors to supply and balance-sheet conditions, for example:

\[
\Delta H_{i,t,k}
=
\alpha_i + \beta_i \Delta \text{NetSupply}_{t,k}
+ \gamma_i \Delta \text{SOMA}_{t,k}
+ \delta_i \text{Regime}_{t}
+ \varepsilon_{i,t,k}
\]

Then compare sectors using:

- standardized beta vectors,
- rolling factor-adjusted partial correlations,
- cosine distance,
- clustering.

Outputs:

- bank-closeness score,
- foreign-closeness score,
- most similar / most dissimilar sectors.

## Validation

Validation must be explicit.

1. Compare levels to FWTW.
2. Compare primary allocations to investor-class allotments.
3. Compare foreign results to TIC.
4. Compare bank-side holdings changes to Call Reports / EFA.
5. Compare dealer bridge episodes to Primary Dealer Statistics.
6. Always report residuals and row/column constraint errors.

## What not to claim

Do not claim:

- exact gross trades,
- exact immediate counterparties,
- exact beneficial foreign ownership,
- household residuals as literal observed sector behavior.

## Preferred presentation language

Use:

- likely net seller
- likely net counterparty
- marginal absorber
- public-data inferred matrix

Avoid:

- sold directly to
- exact trade path
- observed bilateral transaction
