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

# Cross-channel feature functions
def frequency_ratio(freq1, freq2, eps=1e-10):
    """
    Compute ratio between two frequencies.
    Args:
        freq1: First frequency value
        freq2: Second frequency value
        eps: Small value to avoid division by zero
    Returns:
        Ratio freq1/freq2, or 0 if both are near zero
    """
    if abs(freq2) < eps:
        return 0.0 if abs(freq1) < eps else np.sign(freq1) * 1e10
    return freq1 / (freq2 + eps)

def energy_ratio(energy1, energy2, eps=1e-10):
    """
    Compute ratio between two energy values.
    Args:
        energy1: First energy value
        energy2: Second energy value
        eps: Small value to avoid division by zero
    Returns:
        Ratio energy1/energy2, or 0 if both are near zero
    """
    if abs(energy2) < eps:
        return 0.0 if abs(energy1) < eps else 1e10
    return energy1 / (energy2 + eps)

def feature_ratio(val1, val2, eps=1e-10):
    """
    Compute ratio between two feature values.
    Args:
        val1: First value
        val2: Second value
        eps: Small value to avoid division by zero
    Returns:
        Ratio val1/val2, or 0 if both are near zero
    """
    if abs(val2) < eps:
        return 0.0 if abs(val1) < eps else np.sign(val1) * 1e10
    return val1 / (val2 + eps)

def feature_difference(val1, val2):
    """
    Compute difference between two feature values.
    Args:
        val1: First value
        val2: Second value
    Returns:
        Difference val1 - val2
    """
    return val1 - val2

def cross_channel_correlation(data1, data2):
    """
    Compute Pearson correlation coefficient between two channels.
    Args:
        data1: First channel data array
        data2: Second channel data array
    Returns:
        Correlation coefficient, or 0 if insufficient data
    """
    if len(data1) != len(data2) or len(data1) < 2:
        return 0.0
    
    # Handle constant arrays
    if np.std(data1) < 1e-10 or np.std(data2) < 1e-10:
        return 0.0
    
    correlation = np.corrcoef(data1, data2)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def cross_channel_mean(values):
    """
    Compute mean across multiple channel feature values.
    Args:
        values: List or array of feature values from different channels
    Returns:
        Mean value
    """
    return np.mean(values)

def cross_channel_std(values):
    """
    Compute standard deviation across multiple channel feature values.
    Args:
        values: List or array of feature values from different channels
    Returns:
        Standard deviation
    """
    return np.std(values)

def cross_channel_max(values):
    """
    Compute maximum across multiple channel feature values.
    Args:
        values: List or array of feature values from different channels
    Returns:
        Maximum value
    """
    return np.max(values)

def cross_channel_min(values):
    """
    Compute minimum across multiple channel feature values.
    Args:
        values: List or array of feature values from different channels
    Returns:
        Minimum value
    """
    return np.min(values)

def cross_channel_range(values):
    """
    Compute range (max - min) across multiple channel feature values.
    Args:
        values: List or array of feature values from different channels
    Returns:
        Range value
    """
    return np.max(values) - np.min(values)

