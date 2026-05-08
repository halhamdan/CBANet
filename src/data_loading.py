from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

from src.config import LABEL_MAP


def _extract_driver_id(stem: str) -> str:
    """
    Extract a driver identifier from a session filename stem.

    Convention assumed: the part before the first underscore is the driver ID.
    Examples:
        "Driver1_Session2"  -> "Driver1"
        "D01_Trip3"         -> "D01"
        "participant_01_r2" -> "participant"
        "single_file"       -> "single"
        "noUnderscore"      -> "noUnderscore"
    """
    parts = stem.split('_')
    return parts[0] if len(parts) > 1 else stem


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and concatenate all labeled CSV files matching the given glob pattern.

    Adds two metadata columns to the returned DataFrame:
      - session_id : filename stem (e.g. "Driver1_Session2")
      - driver_id  : driver portion extracted from the filename stem
    """
    files = glob.glob(data_path)
    if not files:
        raise ValueError(f"No files found at {data_path}. Check your path!")

    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(f, low_memory=False)
            stem = Path(f).stem
            df_temp['session_id'] = stem
            df_temp['driver_id']  = _extract_driver_id(stem)
            dfs.append(df_temp)
        except Exception as e:
            print(f"Warning: Skipped {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df['Label'] = df['Label'].astype(str).str.strip()
    df = df[df['Label'].isin(LABEL_MAP.keys())].copy()
    return df
