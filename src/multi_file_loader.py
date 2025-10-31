"""
Multi-File Dataset Loader
Handles loading multiple CSV files from CICIDS2017/2018 datasets

Use this when your dataset is split across multiple CSV files
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from data_preprocessing import IDSDataPreprocessor, FederatedDataDistributor


class MultiFileDatasetLoader:
    """
    Load and combine multiple CSV files from IDS datasets
    """
    
    def __init__(self, dataset_name='CICIDS2017'):
        """
        Initialize the loader
        
        Args:
            dataset_name: Name of dataset ('CICIDS2017', 'CICIDS2018', 'EdgeIIoT')
        """
        self.dataset_name = dataset_name
        self.preprocessor = IDSDataPreprocessor(dataset_name)
        
    def load_multiple_files(self, folder_path, file_pattern='*.csv', 
                           sample_size=None, verbose=True):
        """
        Load multiple CSV files from a folder
        
        Args:
            folder_path: Path to folder containing CSV files
            file_pattern: Pattern to match files (e.g., '*.csv', 'Friday*.csv')
            sample_size: Optional - limit samples per file (for memory management)
            verbose: Print loading progress
            
        Returns:
            Combined DataFrame
        """
        # Get all CSV files matching pattern
        file_paths = glob(os.path.join(folder_path, file_pattern))
        
        if not file_paths:
            raise FileNotFoundError(
                f"No files found in {folder_path} matching pattern {file_pattern}"
            )
        
        file_paths = sorted(file_paths)
        
        if verbose:
            print(f"Found {len(file_paths)} CSV files:")
            for fp in file_paths:
                print(f"  - {os.path.basename(fp)}")
        
        # Load and combine all files
        dfs = []
        total_rows = 0
        
        for i, file_path in enumerate(file_paths):
            if verbose:
                print(f"\nLoading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                # Try different encodings
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                except:
                    df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
                
                # Clean column names (remove leading/trailing spaces)
                df.columns = df.columns.str.strip()
                
                if verbose:
                    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Sample if requested (for memory management)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    if verbose:
                        print(f"  Sampled to: {len(df)} rows")
                
                dfs.append(df)
                total_rows += len(df)
                
            except Exception as e:
                print(f"  ⚠️ Error loading {file_path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No files were successfully loaded!")
        
        # Combine all dataframes
        if verbose:
            print(f"\nCombining {len(dfs)} dataframes...")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if verbose:
            print(f"✓ Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            print(f"  Total samples: {total_rows}")
        
        return combined_df
    
    def load_specific_files(self, file_paths, sample_size=None, verbose=True):
        """
        Load specific CSV files by providing list of paths
        
        Args:
            file_paths: List of file paths to load
            sample_size: Optional - limit samples per file
            verbose: Print loading progress
            
        Returns:
            Combined DataFrame
        """
        if verbose:
            print(f"Loading {len(file_paths)} specific files:")
            for fp in file_paths:
                print(f"  - {os.path.basename(fp)}")
        
        dfs = []
        
        for i, file_path in enumerate(file_paths):
            if verbose:
                print(f"\nLoading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                except:
                    df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
                
                # Clean column names (remove leading/trailing spaces)
                df.columns = df.columns.str.strip()
                
                if verbose:
                    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    if verbose:
                        print(f"  Sampled to: {len(df)} rows")
                
                dfs.append(df)
                
            except Exception as e:
                print(f"  ⚠️ Error loading {file_path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No files were successfully loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if verbose:
            print(f"\n✓ Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        return combined_df


def prepare_federated_dataset_multifile(
    folder_path=None,
    file_paths=None,
    file_pattern='*.csv',
    dataset_name='CICIDS2017',
    n_clients=10,
    distribution='iid',
    test_size=0.2,
    sample_size_per_file=None,
    random_state=42,
    verbose=True
):
    """
    Complete pipeline to prepare federated dataset from multiple files
    
    Args:
        folder_path: Path to folder containing CSV files (use this OR file_paths)
        file_paths: List of specific file paths (use this OR folder_path)
        file_pattern: Pattern to match files (only if using folder_path)
        dataset_name: Name of dataset
        n_clients: Number of federated clients
        distribution: 'iid' or 'non-iid'
        test_size: Test set proportion
        sample_size_per_file: Limit samples per file (for memory management)
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        client_data: List of client datasets
        X_test: Global test set features
        y_test: Global test set labels
        preprocessor: Fitted preprocessor object
    """
    print("="*70)
    print(f"LOADING MULTIPLE FILES - {dataset_name}")
    print("="*70)
    
    # Initialize loader
    loader = MultiFileDatasetLoader(dataset_name)
    
    # Load files
    if folder_path:
        # Load all files matching pattern from folder
        df = loader.load_multiple_files(
            folder_path, 
            file_pattern, 
            sample_size_per_file, 
            verbose
        )
    elif file_paths:
        # Load specific files
        df = loader.load_specific_files(
            file_paths, 
            sample_size_per_file, 
            verbose
        )
    else:
        raise ValueError("Must provide either folder_path or file_paths")
    
    # Clean data
    if verbose:
        print("\n" + "="*70)
        print("PREPROCESSING DATA")
        print("="*70)
    
    df = loader.preprocessor.clean_data(df)
    
    # Preprocess features
    X, y = loader.preprocessor.preprocess_features(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize
    X_train, X_test = loader.preprocessor.normalize_features(X_train, X_test)
    
    # Distribute among clients
    if verbose:
        print("\n" + "="*70)
        print("DISTRIBUTING TO FEDERATED CLIENTS")
        print("="*70)
    
    distributor = FederatedDataDistributor(n_clients, random_state)
    
    if distribution == 'iid':
        client_data = distributor.distribute_iid(X_train, y_train)
    else:
        client_data = distributor.distribute_non_iid(X_train, y_train)
    
    if verbose:
        print("\n" + "="*70)
        print("DATASET PREPARATION COMPLETE!")
        print("="*70)
        print(f"Total training samples: {len(X_train)}")
        print(f"Total testing samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Number of clients: {n_clients}")
        print("="*70 + "\n")
    
    return client_data, X_test, y_test, loader.preprocessor


# =============================================================================
# EXAMPLE USAGE FOR YOUR CICIDS2017 FILES
# =============================================================================

if __name__ == "__main__":
    print("Multi-File Dataset Loader Examples\n")
    
    # =========================================================================
    # OPTION 1: Load ALL CSV files from a folder
    # =========================================================================
    print("="*70)
    print("OPTION 1: Load ALL CSV files from folder")
    print("="*70)
    print("""
    # Replace 'path/to/your/folder' with your actual folder path
    folder_path = 'C:/Users/YourName/Downloads/CICIDS2017'
    
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        folder_path=folder_path,
        file_pattern='*.csv',  # Load all CSV files
        dataset_name='CICIDS2017',
        n_clients=10,
        distribution='iid',
        test_size=0.2,
        sample_size_per_file=50000,  # Limit to 50k rows per file for memory
        verbose=True
    )
    """)
    
    # =========================================================================
    # OPTION 2: Load specific files by name pattern
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 2: Load files matching specific pattern")
    print("="*70)
    print("""
    # Load only Friday files
    folder_path = 'C:/Users/YourName/Downloads/CICIDS2017'
    
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        folder_path=folder_path,
        file_pattern='Friday*.csv',  # Only Friday files
        dataset_name='CICIDS2017',
        n_clients=10,
        distribution='iid'
    )
    """)
    
    # =========================================================================
    # OPTION 3: Load specific files by providing exact paths
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 3: Load specific files with exact paths")
    print("="*70)
    print("""
    # Specify exactly which files to load
    file_paths = [
        'C:/path/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'C:/path/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'C:/path/Friday-WorkingHours-Morning.pcap_ISCX.csv'
    ]
    
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        file_paths=file_paths,
        dataset_name='CICIDS2017',
        n_clients=10,
        distribution='iid'
    )
    """)
    
    # =========================================================================
    # OPTION 4: Load files by day (Monday, Tuesday, etc.)
    # =========================================================================
    print("\n" + "="*70)
    print("OPTION 4: Load files for specific days")
    print("="*70)
    print("""
    # Load all Monday files
    folder_path = 'C:/Users/YourName/Downloads/CICIDS2017'
    
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        folder_path=folder_path,
        file_pattern='Monday*.csv',
        dataset_name='CICIDS2017',
        n_clients=10
    )
    
    # Or load Tuesday + Wednesday + Thursday
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        folder_path=folder_path,
        file_pattern='T*.csv',  # Starts with T
        dataset_name='CICIDS2017',
        n_clients=10
    )
    """)
    
    print("\n" + "="*70)
    print("Ready to use! Choose the option that fits your needs.")
    print("="*70)