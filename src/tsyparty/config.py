from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(relative_path: str | Path) -> dict[str, Any]:
    path = repo_root() / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def data_root() -> Path:
    return repo_root() / "data"


TRANSACTION_BASIS_SCALES = {
    "FA_SAAR_millions": 1.0 / 4000.0,
    "FU_quarterly_millions": 1.0 / 1000.0,
    "prequarterized_billions": 1.0,
}


def transaction_scale_to_quarterly_billions(basis: str | None) -> float:
    """Require a declared source rate basis rather than infer one from values."""
    if basis not in TRANSACTION_BASIS_SCALES:
        raise ValueError(f"Unknown or unspecified transaction_basis: {basis!r}")
    return TRANSACTION_BASIS_SCALES[basis]
