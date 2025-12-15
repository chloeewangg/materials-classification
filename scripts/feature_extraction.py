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
    features['variance'] = variance(data)
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
        # 'location': location,
        # 'distance': distance
    }
    
    # Extract features from each channel
    for i, col in enumerate(data_cols):
        channel_data = df[col].values
        
        # Extract features for this channel
        channel_features = get_features(channel_data, time)
        
        # Add features with format: feature_name_channel_index
        for feat_name, feat_value in channel_features.items():
            features[f'{feat_name}_{i}'] = feat_value
    
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

