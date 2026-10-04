from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def list_audio_files(data_dir: str | Path, extensions: Iterable[str] = (".wav", ".flac", ".mp3", ".ogg")) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    ext_set = {str(ext).lower() for ext in extensions}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    return sorted(files)


def summarize_audio_inventory(data_dir: str | Path) -> pd.DataFrame:
    rows = []
    for file_path in list_audio_files(data_dir):
        rows.append({
            "file_name": file_path.name,
            "relative_path": str(file_path.relative_to(data_dir)),
            "suffix": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
        })
    return pd.DataFrame(rows)


def clean_audio_manifest(data_dir: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    df = summarize_audio_inventory(data_dir)
    if df.empty:
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
        return df

    df = df.drop_duplicates(subset=["relative_path"]).copy()
    df["is_valid"] = True
    df["record_count"] = 1

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def export_label_mapping(label_map: Dict[str, int], out_path: str | Path) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2, sort_keys=True)
