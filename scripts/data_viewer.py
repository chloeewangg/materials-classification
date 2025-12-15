"""
This script is used to plot the data from a single CSV file.
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from scipy.signal import butter, filtfilt, detrend

def plot_file(filename, low_pass=True, detrend_data=True):
    """
    Load a CSV file, apply a low-pass filter, and plot the three sensor columns.

    Args:
        filename: Path to the CSV file.
        cutoff_hz: Low-pass cutoff frequency in Hz.
        detrend_data: If True, remove linear trend after filtering.

    Returns:
        (fig, ax): Matplotlib figure and axes for further customization.
    """
    df = pd.read_csv(filename, header=None, skiprows=1)

    time = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d %H:%M:%S.%f')
    time = (time - time.iloc[0]).dt.total_seconds()
    sensor1 = df.iloc[:, 1].to_numpy()  # ai28
    sensor2 = df.iloc[:, 2].to_numpy()  # ai30
    sensor3 = df.iloc[:, 3].to_numpy()  # ai31

    if low_pass:
        cutoff_hz = 5.0
        dt = time.diff().median()
        fs = 1.0 / dt
        nyq = 0.5 * fs
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        sensor1 = filtfilt(b, a, sensor1)
        sensor2 = filtfilt(b, a, sensor2)
        sensor3 = filtfilt(b, a, sensor3)

    if detrend_data:
        sensor1 = detrend(sensor1, type="linear")
        sensor2 = detrend(sensor2, type="linear")
        sensor3 = detrend(sensor3, type="linear")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, sensor1, label="ai28")
    ax.plot(time, sensor2, label="ai30")
    ax.plot(time, sensor3, label="ai31")

    plot_title = os.path.basename(filename)
    ax.set_title(plot_title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    plot_file(sys.argv[1])
    plt.show()