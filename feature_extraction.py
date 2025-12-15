import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import signal
from scipy.fft import fft, fftfreq

def extract_features_from_channel(data, time):
    """
    Extract features from a single channel.
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    
    # Frequency domain features
    sampling_rate = 100.0
    
    # Perform FFT
    n = len(data)
    fft_values = fft(data)
    fft_magnitude = np.abs(fft_values)
    frequencies = fftfreq(n, 1/sampling_rate)
    
    # Only consider positive frequencies
    positive_freq_idx = frequencies > 0
    positive_frequencies = frequencies[positive_freq_idx]
    positive_magnitude = fft_magnitude[positive_freq_idx]
    
    if len(positive_magnitude) > 0:
        # Frequency with max magnitude
        max_mag_idx = np.argmax(positive_magnitude)
        features['freq_max_magnitude'] = positive_frequencies[max_mag_idx]
        
        # Total spectral energy (sum of squared magnitudes)
        features['spectral_energy'] = np.sum(positive_magnitude ** 2)
    else:
        features['freq_max_magnitude'] = 0.0
        features['spectral_energy'] = 0.0
    
    return features

def parse_folder_name(folder_name):
    """
    Parse folder name like 'FEP_bottom_10mm' or 'Al_center_50mm'
    Returns: (material, location, distance)
    """
    # Remove file extension if present
    folder_name = Path(folder_name).stem if '.' in folder_name else folder_name
    
    # Split by underscore
    parts = folder_name.split('_')
    
    if len(parts) >= 3:
        # Material is the first part
        material = parts[0]
        location = parts[1]
        distance_str = parts[2]
        
        # Extract numeric distance (remove 'mm' and any other non-numeric characters)
        distance_match = re.search(r'(\d+)', distance_str)
        if distance_match:
            distance = int(distance_match.group(1))
        else:
            distance = None
        
        return material, location, distance
    else:
        # Fallback if format doesn't match
        return None, None, None

def process_csv_file(csv_path, folder_name):
    """
    Process a single CSV file and extract features.
    Returns a dictionary with all features and labels.
    """
    # Parse labels from folder name
    material, location, distance = parse_folder_name(folder_name)
    
    # Read CSV file (with header), then ignore first data row
    df = pd.read_csv(csv_path)
    df = df.iloc[1:].reset_index(drop=True)  # Drop first data row
    
    # Get column names
    columns = df.columns.tolist()
    
    # First column should be time, rest are data channels
    time_col = columns[0]
    data_cols = columns[1:4]  # Get first 3 data columns
    
    # Extract time and data
    time = pd.to_datetime(df[time_col]).astype(np.int64) / 1e9  # Convert to seconds
    time = time.values
    
    # Initialize feature dictionary with labels
    features = {
        'material': material,
        'location': location,
        'distance': distance
    }
    
    # Extract features from each channel
    for i, col in enumerate(data_cols):
        channel_data = df[col].values
        
        # Extract features for this channel
        channel_features = extract_features_from_channel(channel_data, time)
        
        # Add features with format: feature_name_channel_index
        for feat_name, feat_value in channel_features.items():
            features[f'{feat_name}_{i}'] = feat_value
    
    return features

def extract_features_from_directory(directory_path):
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
            features = process_csv_file(csv_file, folder_name)
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
    
    # Extract features
    df = extract_features_from_directory(directory_path)
    
    # Save to CSV
    output_file = Path("features_dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\nExtraction complete!")
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Saved to: {output_file}")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    main()

