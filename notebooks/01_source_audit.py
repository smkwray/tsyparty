# %%
from tsyparty.registry import load_sources

sources = load_sources()
[(key, spec.frequency, spec.landing_url) for key, spec in sources.items()]
