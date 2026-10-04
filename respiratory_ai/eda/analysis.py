from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(labels: Sequence[str], output_path: str | Path):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_summary(waveforms: Sequence[np.ndarray], output_path: str | Path):
    fig, axes = plt.subplots(1, min(len(waveforms), 3), figsize=(15, 4), squeeze=False)
    for idx, waveform in enumerate(waveforms[:3]):
        ax = axes[0, idx]
        ax.plot(np.asarray(waveform), linewidth=1)
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
