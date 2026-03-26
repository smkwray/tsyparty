from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_stacked_area(frame: pd.DataFrame, title: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ax = frame.plot.area(figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel("Share / amount")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(dest, dpi=200)
    plt.close()
    return dest
