from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tsyparty.config import load_yaml


@dataclass(slots=True)
class SourceSpec:
    key: str
    category: str
    frequency: str
    landing_url: str
    purpose: str | None = None
    direct_url: str | None = None
    artifact_discovery: str | None = None
    download_strategy: str | None = None
    api_url: str | None = None
    raw: dict[str, Any] | None = None


def load_sources() -> dict[str, SourceSpec]:
    payload = load_yaml("configs/sources.yml")
    out: dict[str, SourceSpec] = {}
    for key, value in payload["sources"].items():
        out[key] = SourceSpec(
            key=key,
            category=value["category"],
            frequency=value["frequency"],
            landing_url=value["landing_url"],
            purpose=value.get("purpose"),
            direct_url=value.get("direct_url"),
            artifact_discovery=value.get("artifact_discovery"),
            download_strategy=value.get("download_strategy"),
            api_url=value.get("api_url"),
            raw=value,
        )
    return out


def iter_sources(public_only: bool = False) -> Iterable[SourceSpec]:
    sources = load_sources()
    for spec in sources.values():
        if public_only:
            yield spec
        else:
            yield spec
