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


def save_heatmap(frame: pd.DataFrame, title: str, dest: str | Path, label: str = "Value") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(frame.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(frame.columns)))
    ax.set_yticks(range(len(frame.index)))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(frame.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.savefig(dest, dpi=200)
    plt.close()
    return dest


def save_line_chart(frame: pd.DataFrame, title: str, dest: str | Path, ylabel: str = "") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ax = frame.plot(figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(dest, dpi=200)
    plt.close()
    return dest
