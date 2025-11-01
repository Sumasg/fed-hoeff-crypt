"""
Combine CICIDS2017 CSV files into a single dataset
"""
import pandas as pd
import os
from pathlib import Path

def combine_cicids2017_files():
    """Combine all CICIDS2017 CSV files into one"""
    
    data_dir = Path("../data/CICIDS2017")
    output_file = Path("../data/CICIDS2017.csv")
    
    print("Combining CICIDS2017 CSV files...")
    
    # List all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    
    combined_data = []
    total_rows = 0
    
    for csv_file in csv_files:
        print(f"  Processing: {csv_file.name}")
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Add to combined data
            combined_data.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"    Error reading {csv_file.name}: {e}")
    
    if combined_data:
        # Combine all dataframes
        print(f"\nCombining {len(combined_data)} files...")
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        print(f"Combined dataset:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total columns: {len(combined_df.columns)}")
        
        # Check label distribution
        if 'Label' in combined_df.columns:
            print(f"\nLabel distribution:")
            print(combined_df['Label'].value_counts())
        elif ' Label' in combined_df.columns:
            print(f"\nLabel distribution:")
            print(combined_df[' Label'].value_counts())
        
        # Save combined dataset
        print(f"\nSaving to: {output_file}")
        combined_df.to_csv(output_file, index=False)
        print("CICIDS2017.csv created successfully!")
        
        return str(output_file)
    else:
        print(" No data to combine!")
        return None

if __name__ == "__main__":
    combine_cicids2017_files()