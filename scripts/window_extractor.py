from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, butter, filtfilt, detrend

SAMPLE_RATE_HZ = 100.0 
LOWPASS_CUTOFF_HZ = 5.0

def lowpass_and_detrend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a 5 Hz low-pass filter and remove drift (linear detrend) to all numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df

    nyq = 0.5 * SAMPLE_RATE_HZ
    cutoff_norm = LOWPASS_CUTOFF_HZ / nyq

    b, a = butter(4, cutoff_norm, btype="low")

    df_filtered = df.copy()
    for col in numeric_cols:
        values = df_filtered[col].to_numpy(dtype=float)
        if len(values) < 3:
            continue
        filtered = filtfilt(b, a, values)
        filtered_detrended = detrend(filtered, type="linear")
        df_filtered[col] = filtered_detrended

    return df_filtered

def find_minima_indices(df: pd.DataFrame) -> List[int]:
    """
    Find local minima indices from the first numeric column using scipy.signal.argrelextrema.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return []

    values = df[numeric_cols[1]].to_numpy()
    if len(values) < 3:
        return []

    idx = argrelextrema(values, np.less, order=50)[0]
    return idx.tolist()

def split_into_windows(df: pd.DataFrame, minima_indices: List[int]) -> List[pd.DataFrame]:
    """
    Split the dataframe into windows using minima indices as internal boundaries.

    Windows are defined between:
        [0, min1), [min1, min2), ..., [minN, len)

    The first and last windows are discarded.
    """
    if not minima_indices:
        return []

    n_rows = len(df)
    boundaries = [0] + minima_indices + [n_rows]

    windows: List[pd.DataFrame] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            windows.append(df.iloc[start:end].copy())

    # Discard the first and last windows
    if len(windows) <= 2:
        return []

    return windows[1:-1]

def process_file(input_path: Path, save_root_dir: Path | None = None, plot: bool = False) -> None:
    """
    Load the CSV, preprocess (low-pass + detrend), find minima, split into windows,
    and optionally save each window as a separate CSV or just plot.
    """
    df = pd.read_csv(input_path)

    df_preprocessed = lowpass_and_detrend(df)

    minima_indices = find_minima_indices(df_preprocessed)
    windows = split_into_windows(df_preprocessed, minima_indices)

    # If no save directory provided, always plot
    should_plot = plot or (save_root_dir is None)

    if should_plot:
        # Plot numeric columns with vertical lines at window boundaries (minima)
        numeric_cols = df_preprocessed.select_dtypes(include=[np.number]).columns
        x = np.arange(len(df_preprocessed))

        plt.figure(figsize=(12, 6))
        for col in numeric_cols:
            plt.plot(x, df_preprocessed[col], label=str(col))

        for idx in minima_indices:
            plt.axvline(idx, color="red", linestyle="--", alpha=0.5)

        plt.xlabel("Index")
        plt.ylabel("Amplitude (mV)")
        if len(numeric_cols) > 0:
            plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    if save_root_dir is None:
        if not windows:
            print("No valid windows found (not enough minima or data).")
        else:
            print(f"Found {len(windows)} windows (plotting only, no save directory provided).")
        return

    if not windows:
        print("No valid windows to save (not enough minima or data).")
        return

    output_dir = save_root_dir / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, window_df in enumerate(windows, start=1):
        out_path = output_dir / f"{i}.csv"
        window_df.to_csv(out_path, index=False)

    print(f"Saved {len(windows)} windows to '{output_dir}'")

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python window_extractor.py <input_csv_path> [save_root_dir] [--plot]"
        )

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Check if save_root_dir is provided (not --plot flag)
    save_root_dir = None
    plot = False
    
    for arg in sys.argv[2:]:
        if arg == "--plot":
            plot = True
        elif save_root_dir is None:
            save_root_dir = Path(arg)

    process_file(input_path, save_root_dir, plot=plot)

if __name__ == "__main__":
    main()

