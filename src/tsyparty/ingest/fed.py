from __future__ import annotations

from pathlib import Path

from tsyparty.ingest.base import discover_links, download_with_manifest
from tsyparty.registry import load_sources


def download_z1_current(dest_dir: str | Path) -> Path:
    sources = load_sources()
    spec = sources["z1_release_page"]
    links = discover_links(
        spec.landing_url,
        href_pattern=r"releases/z1/\d{8}/z1_csv_files\.zip",
    )
    if not links:
        sample = spec.raw.get("sample_artifact_url")
        if not sample:
            raise RuntimeError("Could not discover current Z.1 CSV zip")
        links = [sample]
    url = links[0]
    dest = Path(dest_dir) / Path(url).name
    return download_with_manifest(url, dest, {"source": spec.key, "landing_url": spec.landing_url})


def download_direct_source(source_key: str, dest_dir: str | Path) -> Path:
    sources = load_sources()
    spec = sources[source_key]
    if not spec.direct_url:
        raise ValueError(f"Source {source_key} has no direct_url")
    dest = Path(dest_dir) / Path(spec.direct_url).name
    return download_with_manifest(spec.direct_url, dest, {"source": spec.key, "landing_url": spec.landing_url})


def download_h41_pdf(dest_dir: str | Path) -> Path:
    return download_direct_source("h41_current_pdf", dest_dir)
