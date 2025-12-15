import subprocess
import sys
from pathlib import Path

SAVE_ROOT_DIR = Path(r"C:\Users\chloe\OneDrive\Desktop\materials classification\data_windows")

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_generator.py <root_directory>")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory")
        sys.exit(1)
    
    # Find all CSV files recursively
    csv_files = list(root_dir.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{root_dir}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        # Get relative path from root directory
        rel_path = csv_file.relative_to(root_dir)
        
        # Get parent directory structure (e.g., Positive/Al/ or Negative/FEP/)
        parent_dir = rel_path.parent
        
        # Construct save directory: data_windows/Positive/Al/
        save_dir = SAVE_ROOT_DIR / parent_dir
        
        # Run window_extractor.py
        print(f"Processing: {rel_path}")
        result = subprocess.run(
            [
                sys.executable,
                "window_extractor.py",
                str(csv_file),
                str(save_dir)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  Error processing {rel_path}: {result.stderr}")
        else:
            print(f"  Success: {result.stdout.strip()}")
    
    print(f"\nCompleted processing {len(csv_files)} files")

if __name__ == "__main__":
    main()

