# Source audit

This file summarizes the intended role of each source and the main caveat to remember during implementation.

## Z.1 Financial Accounts
Role: quarterly sector holdings / transactions backbone.  
Main caveat: sector granularity is still coarse for some use cases.

## FWTW
Role: issuer-holder level prior and validation target.  
Main caveat: currently levels-oriented.

## Enhanced Financial Accounts
Role: adds institution detail, including MMFs, hedge funds, bank balance-sheet detail, and long-term international holdings by country.  
Main caveat: project-specific tables have different structures and update cadences.

## TIC / expanded SLT
Role: foreign holdings, transactions, official/private splits, and country detail.  
Main caveat: residence / custodial bias and post-2023 series design changes.

## Investor Class Auction Allotments
Role: primary-market buyer composition.  
Main caveat: investor classes are coarse and links can rotate over time.

## Debt to the Penny / DTS
Role: held-by-public totals, debt operations, and cash/TGA context.  
Main caveat: definitions differ from some other Treasury publications.

## SOMA / H.4.1
Role: Fed holdings and balance-sheet context.  
Main caveat: not a full holder map.

## H.8 / Call Reports / EFA bank data
Role: banking system Treasury context and validation.  
Main caveat: some weekly aggregates mix Treasuries with agency debt.

## Primary Dealer Statistics
Role: dealer bridge and market-plumbing context.  
Main caveat: aggregate only.

## SEC N-MFP / N-PORT
Role: fund-side holdings detail.  
Main caveat: public release cadence and large file size.
