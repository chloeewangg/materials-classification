import numpy as np
from scipy.fft import fft, fftfreq

def mean(data):
    return np.mean(data)

def mean_absolute_value(data):
    return np.mean(np.abs(data))

def std(data):
    return np.std(data)

def variance(data):
    return np.var(data)

def min_value(data):
    return np.min(data)

def max_value(data):
    return np.max(data)

def zero_crossing_rate(data):
    if len(data) < 2:
        return 0.0
    signs = np.sign(data)
    # Treat zeros as previous sign to avoid spurious crossings
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.sum(np.diff(signs) != 0)
    return crossings / (len(data) - 1)

def slope_sign_changes(data):
    if len(data) < 3:
        return 0.0
    diff = np.diff(data)
    diff_signs = np.sign(diff)
    for i in range(1, len(diff_signs)):
        if diff_signs[i] == 0:
            diff_signs[i] = diff_signs[i - 1]
    changes = np.sum(np.diff(diff_signs) != 0)
    return changes / (len(diff))

def peak_frequency(data, sampling_rate=100.0):
    n = len(data)
    fft_values = fft(data)
    fft_magnitude = np.abs(fft_values)
    frequencies = fftfreq(n, 1/sampling_rate)
    
    # Only consider positive frequencies
    positive_freq_idx = frequencies > 0
    positive_frequencies = frequencies[positive_freq_idx]
    positive_magnitude = fft_magnitude[positive_freq_idx]
    
    if len(positive_magnitude) > 0:
        max_mag_idx = np.argmax(positive_magnitude)
        return positive_frequencies[max_mag_idx]
    else:
        return 0.0

def spectral_energy(data, sampling_rate=100.0):
    n = len(data)
    fft_values = fft(data)
    fft_magnitude = np.abs(fft_values)
    frequencies = fftfreq(n, 1/sampling_rate)
    
    # Only consider positive frequencies
    positive_freq_idx = frequencies > 0
    positive_magnitude = fft_magnitude[positive_freq_idx]
    
    if len(positive_magnitude) > 0:
        return np.sum(positive_magnitude ** 2)
    else:
        return 0.0

