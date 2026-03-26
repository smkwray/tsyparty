# %%
import pandas as pd
from tsyparty.reconcile.accounting import summarize_gap_frame

toy = pd.DataFrame(
    {
        "date": ["2024Q4", "2025Q1"],
        "public_debt": [27000.0, 27200.0],
        "soma_holdings": [4500.0, 4450.0],
        "sector_total": [22400.0, 22630.0],
    }
)
summarize_gap_frame(toy)
