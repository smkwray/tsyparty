from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from tsyparty.utils_http import fetch_binary, soup


def discover_links(
    page_url: str,
    href_pattern: str | None = None,
    text_pattern: str | None = None,
) -> list[str]:
    page = soup(page_url)
    out: list[str] = []
    href_re = re.compile(href_pattern) if href_pattern else None
    text_re = re.compile(text_pattern, flags=re.IGNORECASE) if text_pattern else None

    for anchor in page.find_all("a"):
        href = anchor.get("href")
        text = anchor.get_text(" ", strip=True)
        if not href:
            continue
        href_ok = href_re.search(href) if href_re else True
        text_ok = text_re.search(text) if text_re else True
        if href_ok and text_ok:
            out.append(urljoin(page_url, href))
    return list(dict.fromkeys(out))


def download_with_manifest(url: str, dest: Path, meta: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = fetch_binary(url, dest)
    manifest = dest.with_suffix(dest.suffix + ".manifest.json")
    manifest.write_text(json.dumps(meta | {"download_url": url}, indent=2), encoding="utf-8")
    return result
