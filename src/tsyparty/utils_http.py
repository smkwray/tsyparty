from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "tsyparty/0.1 (+research use)"
}


def fetch_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_binary(url: str, dest: str | Path, timeout: int = 60) -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)
    return dest_path


def soup(url: str, timeout: int = 60) -> BeautifulSoup:
    return BeautifulSoup(fetch_text(url, timeout=timeout), "lxml")


def absolute_links(page_url: str, hrefs: Iterable[str]) -> list[str]:
    return [urljoin(page_url, href) for href in hrefs]
