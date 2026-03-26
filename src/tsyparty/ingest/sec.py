from __future__ import annotations

from pathlib import Path

from tsyparty.ingest.base import discover_links, download_with_manifest
from tsyparty.registry import load_sources


def discover_sec_zip_links(source_key: str) -> list[str]:
    sources = load_sources()
    spec = sources[source_key]
    return discover_links(spec.landing_url, href_pattern=r"\.zip$")


def download_latest_sec_zip(source_key: str, dest_dir: str | Path) -> Path:
    links = discover_sec_zip_links(source_key)
    if not links:
        raise RuntimeError(f"No ZIP links discovered for {source_key}")
    url = links[0]
    dest = Path(dest_dir) / Path(url).name
    return download_with_manifest(url, dest, {"source": source_key})
