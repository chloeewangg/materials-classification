"""
This script is used to extract features from the data windows located in the data_windows directory.
The features are saved to a new file called features_dataset.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import signal
from features import *

def get_features(data, time):
    """
    Extract features from a single channel.
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = mean(data)
    features['mav'] = mean_absolute_value(data)
    features['std'] = std(data)
    features['min'] = min_value(data)
    features['max'] = max_value(data)
    features['zero_crossing_rate'] = zero_crossing_rate(data)
    features['slope_sign_changes'] = slope_sign_changes(data)
    
    # Frequency domain features
    sampling_rate = 100.0
    features['peak_frequency'] = peak_frequency(data, sampling_rate)
    features['spectral_energy'] = spectral_energy(data, sampling_rate)
    
    return features

def get_folder_name(folder_name):
    """
    Parse folder name like 'FEP_bottom_10mm' or 'Al_center_50mm'
    Returns: (material, location, distance)
    """
    folder_name = Path(folder_name).stem if '.' in folder_name else folder_name
    
    parts = folder_name.split('_')
    
    if len(parts) >= 3:
        material = parts[0]
        location = parts[1]
        distance_str = parts[2]
        
        distance_match = re.search(r'(\d+)', distance_str)
        if distance_match:
            distance = int(distance_match.group(1))
        else:
            distance = None
        
        return material, location, distance
    else:
        return parts[0], None, None

def process_csv(csv_path, folder_name):
    """
    Process a single CSV file and extract features.
    Returns a dictionary with all features and labels.
    """
    material, location, distance = get_folder_name(folder_name)
    
    df = pd.read_csv(csv_path)
    df = df.iloc[1:].reset_index(drop=True)  # Drop first data row
    
    columns = df.columns.tolist()
    
    time_col = columns[0]
    data_cols = columns[1:4]  
    
    time = pd.to_datetime(df[time_col]).astype(np.int64) / 1e9  # Convert to seconds
    time = time.values
    
    # Initialize feature dictionary with labels
    features = {
        'material': material,
        'location': location,
        # 'distance': distance
    }
    
    # Store channel data and features for cross-channel computations
    channel_data_list = []
    channel_features_list = []
    
    # Extract features from each channel
    for i, col in enumerate(data_cols):
        channel_data = df[col].values
        channel_data_list.append(channel_data)
        
        # Extract features for this channel
        channel_features = get_features(channel_data, time)
        channel_features_list.append(channel_features)
        
        # Add features with format: feature_name_channel_index
        for feat_name, feat_value in channel_features.items():
            features[f'{feat_name}_{i}'] = feat_value
    
    # Compute cross-channel features
    num_channels = len(channel_features_list)
    
    if num_channels >= 2:
        # Cross-channel correlations
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                corr = cross_channel_correlation(channel_data_list[i], channel_data_list[j])
                features[f'correlation_{i}_{j}'] = corr
        
        # Frequency ratios
        freq_key = 'peak_frequency'
        if all(freq_key in cf for cf in channel_features_list):
            freqs = [cf[freq_key] for cf in channel_features_list]
            # Ratios between all pairs
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    ratio = frequency_ratio(freqs[i], freqs[j])
                    features[f'freq_ratio_{i}_{j}'] = ratio
                    # Also add inverse ratio
                    features[f'freq_ratio_{j}_{i}'] = frequency_ratio(freqs[j], freqs[i])
        
        # Energy ratios
        energy_key = 'spectral_energy'
        if all(energy_key in cf for cf in channel_features_list):
            energies = [cf[energy_key] for cf in channel_features_list]
            # Ratios between all pairs
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    ratio = energy_ratio(energies[i], energies[j])
                    features[f'energy_ratio_{i}_{j}'] = ratio
                    # Also add inverse ratio
                    features[f'energy_ratio_{j}_{i}'] = energy_ratio(energies[j], energies[i])
        
        # Mean ratios and differences
        mean_key = 'mean'
        if all(mean_key in cf for cf in channel_features_list):
            means = [cf[mean_key] for cf in channel_features_list]
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    features[f'mean_ratio_{i}_{j}'] = feature_ratio(means[i], means[j])
                    features[f'mean_diff_{i}_{j}'] = feature_difference(means[i], means[j])
        
        # MAV (Mean Absolute Value) ratios and differences
        mav_key = 'mav'
        if all(mav_key in cf for cf in channel_features_list):
            mavs = [cf[mav_key] for cf in channel_features_list]
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    features[f'mav_ratio_{i}_{j}'] = feature_ratio(mavs[i], mavs[j])
                    features[f'mav_diff_{i}_{j}'] = feature_difference(mavs[i], mavs[j])
        
        # Std ratios and differences
        std_key = 'std'
        if all(std_key in cf for cf in channel_features_list):
            stds = [cf[std_key] for cf in channel_features_list]
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    features[f'std_ratio_{i}_{j}'] = feature_ratio(stds[i], stds[j])
                    features[f'std_diff_{i}_{j}'] = feature_difference(stds[i], stds[j])
        
        # Cross-channel statistics (aggregate across channels for each feature type)
        feature_types = ['mean', 'mav', 'std', 'peak_frequency', 'spectral_energy']
        for feat_type in feature_types:
            if all(feat_type in cf for cf in channel_features_list):
                values = [cf[feat_type] for cf in channel_features_list]
                features[f'{feat_type}_cross_mean'] = cross_channel_mean(values)
                features[f'{feat_type}_cross_std'] = cross_channel_std(values)
                features[f'{feat_type}_cross_max'] = cross_channel_max(values)
                features[f'{feat_type}_cross_min'] = cross_channel_min(values)
                features[f'{feat_type}_cross_range'] = cross_channel_range(values)
    
    return features

def make_feature_df(directory_path):
    """
    Extract features from all CSV files in a directory and subdirectories.
    Returns a pandas DataFrame.
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Find all CSV files
    csv_files = list(directory.rglob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_features = []
    for csv_file in csv_files:
        # Get the parent folder name (the one containing the CSV)
        folder_name = csv_file.parent.name
        
        try:
            features = process_csv(csv_file, folder_name)
            all_features.append(features)
            print(f"Processed: {csv_file.relative_to(directory)}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    return df

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    print(f"Extracting features from: {directory_path}")
    
    df = make_feature_df(directory_path)
    
    output_file = Path("features_dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Saved to: {output_file}")
    
    print(df.head())

if __name__ == "__main__":
    main()

