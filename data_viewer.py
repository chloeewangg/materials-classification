import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from scipy.signal import butter, filtfilt

filename = sys.argv[1]

df = pd.read_csv(filename, header=None, skiprows=1)

time = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d %H:%M:%S.%f')
sensor1 = df.iloc[:, 1].to_numpy()  # ai28
sensor2 = df.iloc[:, 2].to_numpy()  # ai30
sensor3 = df.iloc[:, 3].to_numpy()  # ai31

dt = time.diff().dt.total_seconds().median()
fs = 1.0 / dt
cutoff = 5.0  # Hz
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(4, normal_cutoff, btype="low", analog=False)
sensor1 = filtfilt(b, a, sensor1)
sensor2 = filtfilt(b, a, sensor2)
sensor3 = filtfilt(b, a, sensor3)

plt.figure(figsize=(10, 6))
plt.plot(time, sensor1, label="ai28")
plt.plot(time, sensor2, label="ai30")
plt.plot(time, sensor3, label="ai31")

plot_title = os.path.basename(filename)
plt.title(plot_title)

plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()