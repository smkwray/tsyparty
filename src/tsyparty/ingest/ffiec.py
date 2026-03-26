from __future__ import annotations

from pathlib import Path

from tsyparty.registry import load_sources


def ffiec_manual_instructions(dest_dir: str | Path) -> Path:
    """
    Seed placeholder for FFIEC bulk downloads.

    The FFIEC bulk page is a form-driven endpoint. Build a proper downloader after
    inspecting the available call-date options and form payload on the landing page.

    This function writes a note into the destination directory so the manual path is
    visible in the repo before full automation lands.
    """
    spec = load_sources()["ffiec_bulk_data"]
    dest = Path(dest_dir) / "FFIEC_BULK_DOWNLOAD_INSTRUCTIONS.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "\n".join(
            [
                "FFIEC bulk data downloader not yet automated.",
                f"Landing page: {spec.landing_url}",
                "Next codex task:",
                "1. Inspect the page form fields.",
                "2. Submit a valid call date selection.",
                "3. Save the returned ZIP or CSV artifact.",
                "4. Record the artifact URL and form payload in a manifest.",
            ]
        ),
        encoding="utf-8",
    )
    return dest
