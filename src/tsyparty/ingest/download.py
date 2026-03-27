"""Registry-driven download dispatcher.

Routes download requests through the appropriate strategy based on
the `download_strategy` field in sources.yml.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import requests

from tsyparty.ingest.base import download_with_manifest
from tsyparty.registry import load_sources, SourceSpec


def download_source(source_key: str, dest_dir: str | Path, **kwargs) -> Path:
    """Download a source using its configured strategy.

    Parameters
    ----------
    source_key : key in sources.yml
    dest_dir : destination directory
    **kwargs : extra parameters (e.g. params for API sources)

    Returns the path to the downloaded file.
    """
    sources = load_sources()
    if source_key not in sources:
        raise KeyError(f"Unknown source: {source_key}. Available: {sorted(sources.keys())}")

    spec = sources[source_key]
    strategy = spec.download_strategy

    if strategy == "direct_url":
        return _download_direct(spec, dest_dir)
    elif strategy == "json_api":
        return _download_json_api(spec, dest_dir, params=kwargs.get("params"))
    elif strategy == "fiscaldata_api":
        return _download_fiscaldata_api(spec, dest_dir, params=kwargs.get("params"))
    elif strategy == "discover_artifact":
        return _download_discover(spec, dest_dir)
    elif strategy is None:
        # Fallback: try direct_url, then api_url
        if spec.direct_url:
            return _download_direct(spec, dest_dir)
        elif spec.api_url:
            return _download_json_api(spec, dest_dir, params=kwargs.get("params"))
        else:
            raise ValueError(
                f"Source {source_key} has no download_strategy and no direct_url or api_url"
            )
    else:
        raise ValueError(f"Unknown download_strategy '{strategy}' for source {source_key}")


def _download_direct(spec: SourceSpec, dest_dir: str | Path) -> Path:
    if not spec.direct_url:
        raise ValueError(f"Source {spec.key} has no direct_url")
    # Use source key as filename if the URL has query params that mangle Path.name
    url_path = spec.direct_url.split("?")[0]
    filename = Path(url_path).name
    if not filename or "/" in filename or len(filename) > 100:
        ext = ".csv" if "csv" in spec.direct_url.lower() else ".dat"
        filename = f"{spec.key}{ext}"
    dest = Path(dest_dir) / filename
    return download_with_manifest(
        spec.direct_url, dest,
        {"source": spec.key, "landing_url": spec.landing_url},
    )


def _download_json_api(
    spec: SourceSpec, dest_dir: str | Path, params: dict | None = None,
) -> Path:
    url = spec.api_url
    if not url:
        raise ValueError(f"Source {spec.key} has no api_url")
    query = f"{url}?{urlencode(params)}" if params else url
    response = requests.get(query, timeout=120)
    response.raise_for_status()
    dest = Path(dest_dir) / f"{spec.key}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    manifest = dest.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps({"source": spec.key, "query_url": query, "landing_url": spec.landing_url}, indent=2),
        encoding="utf-8",
    )
    return dest


def _download_fiscaldata_api(
    spec: SourceSpec, dest_dir: str | Path, params: dict | None = None,
) -> Path:
    url = spec.api_url or spec.raw.get("api_url", "")
    if not url:
        raise ValueError(f"Source {spec.key} has no api_url")
    return _download_json_api(
        SourceSpec(
            key=spec.key,
            category=spec.category,
            frequency=spec.frequency,
            landing_url=spec.landing_url,
            api_url=url,
            raw=spec.raw,
        ),
        dest_dir, params,
    )


def _download_discover(spec: SourceSpec, dest_dir: str | Path) -> Path:
    """Discover artifact URL from landing page, then download."""
    from tsyparty.ingest.base import discover_links

    # Use artifact_discovery hints from the spec
    href_pattern = None
    if spec.raw and "artifact_discovery" in spec.raw:
        # Try to derive a pattern; fall back to any .zip or .csv
        href_pattern = r"\.(zip|csv|xls|xlsx|json)$"

    links = discover_links(spec.landing_url, href_pattern=href_pattern)
    if not links:
        # Fall back to sample URLs in raw config
        fallback_urls = [
            v for k, v in (spec.raw or {}).items()
            if k.startswith("sample_") and isinstance(v, str)
        ]
        if not fallback_urls:
            raise RuntimeError(f"No artifacts discovered for {spec.key} at {spec.landing_url}")
        links = fallback_urls

    url = links[0]
    dest = Path(dest_dir) / Path(url).name
    return download_with_manifest(url, dest, {"source": spec.key, "landing_url": spec.landing_url})
