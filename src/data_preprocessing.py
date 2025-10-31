"""
Functionality 1: Data Loading and Preprocessing Module
For Secure FL-based IDS with Homomorphic Encryption and Watermarking

This module handles:
- Loading CICIDS2017, CICIDS2018, and EdgeIIoT datasets
- Data preprocessing and feature engineering
- Data distribution among federated learning clients
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

class IDSDataPreprocessor:
    """
    Preprocessor for IDS datasets (CICIDS2017, CICIDS2018, EdgeIIoT)
    """
    
    def __init__(self, dataset_name='CICIDS2017'):
        """
        Initialize the preprocessor
        
        Args:
            dataset_name: Name of dataset ('CICIDS2017', 'CICIDS2018', 'EdgeIIoT')
        """
        self.dataset_name = dataset_name
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.label_names = None
        
    def load_dataset(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading {self.dataset_name} dataset from {file_path}...")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except:
            df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and infinite values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"Data cleaned: {df.shape[0]} rows remaining")
        return df
    
    def preprocess_features(self, df, label_column='Label'):
        """
        Preprocess features and labels
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            
        Returns:
            X (features), y (labels)
        """
        print("Preprocessing features...")
        
        # Clean column names first (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Separate features and labels
        if label_column not in df.columns:
            # Try common label column names
            possible_labels = ['Label', 'label', 'Attack_type', 'Attack_label', 'class']
            for label in possible_labels:
                if label in df.columns:
                    label_column = label
                    break
        
        if label_column not in df.columns:
            raise ValueError(f"Label column not found. Available columns: {df.columns.tolist()}")
        
        # Extract labels
        y = df[label_column].copy()
        X = df.drop(columns=[label_column])
        
        # Remove non-numeric columns that can't be used for ML
        # Keep only timestamp if needed, otherwise remove it
        cols_to_drop = []
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    X[col] = pd.to_numeric(X[col])
                except:
                    cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"Dropping non-numeric columns: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels (convert attack names to numeric)
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_names = self.label_encoder.classes_
        
        print(f"Features shape: {X.shape}")
        print(f"Number of classes: {len(self.label_names)}")
        print(f"Class distribution:\n{pd.Series(y).value_counts()}")
        
        return X, y_encoded
    
    def normalize_features(self, X_train, X_test):
        """
        Normalize features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Normalized X_train, X_test
        """
        print("Normalizing features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test


class FederatedDataDistributor:
    """
    Distribute data among federated learning clients
    """
    
    def __init__(self, n_clients=10, random_state=42):
        """
        Initialize the distributor
        
        Args:
            n_clients: Number of federated learning clients
            random_state: Random seed
        """
        self.n_clients = n_clients
        self.random_state = random_state
        np.random.seed(random_state)
    
    def distribute_iid(self, X_train, y_train):
        """
        Distribute data randomly (IID) among clients
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            List of (X_client, y_client) tuples for each client
        """
        print(f"Distributing data IID among {self.n_clients} clients...")
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Split into chunks
        X_chunks = np.array_split(X_shuffled, self.n_clients)
        y_chunks = np.array_split(y_shuffled, self.n_clients)
        
        client_data = []
        for i in range(self.n_clients):
            client_data.append((X_chunks[i], y_chunks[i]))
            print(f"Client {i}: {len(X_chunks[i])} samples")
        
        return client_data
    
    def distribute_non_iid(self, X_train, y_train, n_classes_per_client=2):
        """
        Distribute data non-IID among clients (each client gets subset of classes)
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_classes_per_client: Number of classes per client
            
        Returns:
            List of (X_client, y_client) tuples for each client
        """
        print(f"Distributing data Non-IID among {self.n_clients} clients...")
        print(f"Each client gets {n_classes_per_client} classes")
        
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        # Assign classes to clients
        client_classes = []
        for i in range(self.n_clients):
            np.random.seed(self.random_state + i)
            selected_classes = np.random.choice(
                unique_classes, 
                size=min(n_classes_per_client, n_classes), 
                replace=False
            )
            client_classes.append(selected_classes)
        
        # Distribute data
        client_data = []
        for i in range(self.n_clients):
            # Get indices for this client's classes
            mask = np.isin(y_train, client_classes[i])
            X_client = X_train[mask]
            y_client = y_train[mask]
            
            client_data.append((X_client, y_client))
            print(f"Client {i}: {len(X_client)} samples, classes {client_classes[i]}")
        
        return client_data


def prepare_federated_dataset(file_path, dataset_name='CICIDS2017', 
                             n_clients=10, distribution='iid', 
                             test_size=0.2, random_state=42):
    """
    Complete pipeline to prepare federated dataset
    
    Args:
        file_path: Path to dataset CSV file
        dataset_name: Name of dataset
        n_clients: Number of federated clients
        distribution: 'iid' or 'non-iid'
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        client_data: List of client datasets
        X_test: Global test set features
        y_test: Global test set labels
        preprocessor: Fitted preprocessor object
    """
    # Initialize preprocessor
    preprocessor = IDSDataPreprocessor(dataset_name)
    
    # Load and clean data
    df = preprocessor.load_dataset(file_path)
    df = preprocessor.clean_data(df)
    
    # Preprocess features
    X, y = preprocessor.preprocess_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X.values, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize
    X_train, X_test = preprocessor.normalize_features(X_train, X_test)
    
    # Distribute among clients
    distributor = FederatedDataDistributor(n_clients, random_state)
    
    if distribution == 'iid':
        client_data = distributor.distribute_iid(X_train, y_train)
    else:
        client_data = distributor.distribute_non_iid(X_train, y_train)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print("="*60 + "\n")
    
    return client_data, X_test, y_test, preprocessor


# Example usage
if __name__ == "__main__":
    # Example: Prepare CICIDS2017 dataset for federated learning
    
    # Uncomment and modify the path to your dataset
    # file_path = 'path/to/CICIDS2017.csv'
    
    # client_data, X_test, y_test, preprocessor = prepare_federated_dataset(
    #     file_path=file_path,
    #     dataset_name='CICIDS2017',
    #     n_clients=10,
    #     distribution='iid',
    #     test_size=0.2,
    #     random_state=42
    # )
    
    # # Access client data
    # for i, (X_client, y_client) in enumerate(client_data):
    #     print(f"Client {i}: X shape = {X_client.shape}, y shape = {y_client.shape}")
    
    print("Data preprocessing module ready!")
    print("\nTo use this module:")
    print("1. Load your dataset CSV file")
    print("2. Call prepare_federated_dataset() with your file path")
    print("3. Get distributed client data ready for federated learning")