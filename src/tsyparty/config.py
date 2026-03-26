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
