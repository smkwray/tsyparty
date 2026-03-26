from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import requests

from tsyparty.ingest.base import discover_links, download_with_manifest
from tsyparty.registry import load_sources


def download_investor_class_recent(dest_dir: str | Path) -> list[Path]:
    sources = load_sources()
    spec = sources["investor_class_page"]
    links: list[str] = []
    try:
        links = discover_links(
            spec.landing_url,
            href_pattern=r"IC-(Coupons|Bills)\.xls",
        )
    except requests.HTTPError:
        pass
    if not links:
        fallback = [v for k, v in spec.raw.items() if k.startswith("sample_") and str(v).endswith(".xls")]
        links = fallback
    out: list[Path] = []
    for url in links:
        dest = Path(dest_dir) / Path(url).name
        out.append(download_with_manifest(url, dest, {"source": spec.key, "landing_url": spec.landing_url}))
    return out


def download_direct_treasury_source(source_key: str, dest_dir: str | Path) -> Path:
    sources = load_sources()
    spec = sources[source_key]
    if not spec.direct_url:
        raise ValueError(f"Source {source_key} has no direct_url")
    dest = Path(dest_dir) / Path(spec.direct_url).name
    return download_with_manifest(spec.direct_url, dest, {"source": spec.key, "landing_url": spec.landing_url})


def download_fiscaldata_api(source_key: str, dest_dir: str | Path, params: dict | None = None) -> Path:
    sources = load_sources()
    spec = sources[source_key]
    api_url = spec.raw["api_url"]
    dest = Path(dest_dir) / f"{source_key}.json"
    query = f"{api_url}?{urlencode(params or {})}" if params else api_url
    response = requests.get(query, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    manifest = dest.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"source": source_key, "query_url": query, "landing_url": spec.landing_url}, indent=2), encoding="utf-8")
    return dest
