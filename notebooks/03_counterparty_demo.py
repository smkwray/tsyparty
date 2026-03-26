# %%
import pandas as pd
from tsyparty.infer.counterparty import ras_balance

buyers = pd.Series({"banks": 50.0, "foreigners_private": 30.0, "money_market_funds": 20.0})
sellers = pd.Series({"dealers": 40.0, "insurers": 25.0, "households_residual": 20.0, "mutual_funds_etfs": 15.0})
prior = pd.DataFrame(1.0, index=sellers.index, columns=buyers.index)
matrix, diagnostics = ras_balance(prior, sellers, buyers)
matrix, diagnostics
