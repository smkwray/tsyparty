/* Consistent sector color palette across all charts */
const SECTOR_COLORS = {
  banks:                   "#58a6ff",
  dealers:                 "#f97583",
  foreigners_official:     "#d29922",
  foreigners_private:      "#e3b341",
  money_market_funds:      "#3fb950",
  mutual_funds_etfs:       "#bc8cff",
  insurers:                "#79c0ff",
  pensions:                "#56d364",
  households_residual:     "#8b949e",
  fed:                     "#f85149",
  state_local_governments: "#db61a2",
  other_financial:         "#7ee787",
  nonfinancial_corporates: "#e3b341",
  hedge_funds:             "#b392f0",
  _residual:               "#6e7681",
  _total:                  "#c9d1d9",
};

/* Human-readable labels */
const SECTOR_LABELS = {
  banks:                   "Banks",
  dealers:                 "Dealers",
  foreigners_official:     "Foreign Official",
  foreigners_private:      "Foreign Private",
  money_market_funds:      "Money Market Funds",
  mutual_funds_etfs:       "Mutual Funds & ETFs",
  insurers:                "Insurers",
  pensions:                "Pensions",
  households_residual:     "Households (Residual)",
  fed:                     "Federal Reserve",
  state_local_governments: "State & Local Gov't",
  other_financial:         "Other Financial",
  nonfinancial_corporates: "Nonfinancial Corps",
  hedge_funds:             "Hedge Funds",
  _residual:               "Residual",
  _total:                  "Total",
};

function sectorColor(sector) {
  return SECTOR_COLORS[sector] || "#6e7681";
}

function sectorLabel(sector) {
  return SECTOR_LABELS[sector] || sector.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
