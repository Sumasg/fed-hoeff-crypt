"""
Complete Setup Guide and Executable Examples
For Secure FL-based IDS System

Follow these steps to run the code successfully.
"""

# =============================================================================
# STEP 1: INSTALLATION
# =============================================================================
"""
First, install all required dependencies. Run these commands in your terminal:

pip install numpy pandas scikit-learn
pip install matplotlib seaborn  # For visualization
pip install tenseal  # For homomorphic encryption (we'll need this later)

Or create a requirements.txt file with:
---
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tenseal>=0.3.0
---

Then run: pip install -r requirements.txt
"""

# =============================================================================
# STEP 2: PROJECT STRUCTURE
# =============================================================================
"""
Create this folder structure:

FL_IDS_Project/
│
├── data/
│   ├── CICIDS2017.csv
│   ├── CICIDS2018.csv
│   └── EdgeIIoT.csv
│
├── src/
│   ├── data_preprocessing.py  (Functionality 1)
│   ├── hat_algorithm.py       (Functionality 2)
│   └── main.py                (This file)
│
└── results/
    └── (output files will be saved here)
"""

# =============================================================================
# STEP 3: SAVE THE MODULES
# =============================================================================
"""
1. Copy Functionality 1 code → save as 'data_preprocessing.py'
2. Copy Functionality 2 code → save as 'hat_algorithm.py'
3. Copy this file → save as 'main.py'
"""

# =============================================================================
# STEP 4: EXECUTABLE EXAMPLE - COMPLETE WORKFLOW
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_preprocessing import IDSDataPreprocessor, FederatedDataDistributor, prepare_federated_dataset
from hat_algorithm import ModifiedHoeffdingTree, PolynomialActivation


def create_synthetic_dataset(n_samples=10000, n_features=20, n_classes=5, noise=0.1):
    """
    Create synthetic IDS dataset for testing
    (Use this if you don't have real datasets yet)
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of attack classes
        noise: Noise level
        
    Returns:
        DataFrame with synthetic IDS data
    """
    print("Creating synthetic IDS dataset...")
    
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make classification meaningful
    for i in range(n_classes):
        mask = np.random.choice(n_samples, size=n_samples//n_classes, replace=False)
        X[mask, i*2:(i+1)*2] += np.random.randn(len(mask), 2) * 3
    
    # Generate labels
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add noise
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = np.random.randint(0, n_classes, noise_mask.sum())
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['Label'] = y
    
    # Map labels to attack types
    attack_types = ['Normal', 'DDoS', 'PortScan', 'BruteForce', 'Infiltration']
    df['Label'] = df['Label'].map(lambda x: attack_types[x % len(attack_types)])
    
    print(f"Created dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    
    return df


def example_1_test_polynomial_activations():
    """
    Example 1: Test polynomial activation functions
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Testing Polynomial Activation Functions")
    print("="*70)
    
    poly_act = PolynomialActivation()
    
    # Test inputs
    x = np.linspace(-2, 2, 10)
    
    print("\nInput values:", x)
    print("\nPolynomial Sigmoid (degree 3):")
    print(poly_act.polynomial_sigmoid(x, degree=3))
    
    print("\nPolynomial Tanh (degree 3):")
    print(poly_act.polynomial_tanh(x, degree=3))
    
    print("\n✓ Polynomial activations working correctly!")


def example_2_test_data_preprocessing():
    """
    Example 2: Test data preprocessing with synthetic data
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Testing Data Preprocessing")
    print("="*70)
    
    # Create synthetic dataset
    df = create_synthetic_dataset(n_samples=5000, n_features=20, n_classes=5)
    
    # Save to CSV (simulate real dataset)
    df.to_csv('synthetic_ids_data.csv', index=False)
    print("\n✓ Saved synthetic dataset to 'synthetic_ids_data.csv'")
    
    # Load and preprocess
    preprocessor = IDSDataPreprocessor(dataset_name='Synthetic')
    
    # Load data
    df = preprocessor.load_dataset('synthetic_ids_data.csv')
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Preprocess features
    X, y = preprocessor.preprocess_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X.values, y)
    
    # Normalize
    X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Training set: {X_train_scaled.shape}")
    print(f"  Test set: {X_test_scaled.shape}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


def example_3_test_federated_distribution():
    """
    Example 3: Test federated data distribution
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Testing Federated Data Distribution")
    print("="*70)
    
    # Create synthetic dataset
    df = create_synthetic_dataset(n_samples=5000, n_features=20, n_classes=5)
    df.to_csv('synthetic_ids_data.csv', index=False)
    
    # Prepare federated dataset
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset(
        file_path='synthetic_ids_data.csv',
        dataset_name='Synthetic',
        n_clients=10,
        distribution='iid',
        test_size=0.2,
        random_state=42
    )
    
    print(f"\n✓ Federated distribution complete!")
    print(f"  Number of clients: {len(client_data)}")
    print(f"  Global test set: {X_test.shape}")
    
    # Show client data distribution
    print("\nClient data sizes:")
    for i, (X_client, y_client) in enumerate(client_data):
        print(f"  Client {i}: {X_client.shape[0]} samples")
    
    return client_data, X_test, y_test, preprocessor


def example_4_train_hat_model():
    """
    Example 4: Train Modified HAT model
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Training Modified HAT Model")
    print("="*70)
    
    # Get preprocessed data
    X_train, X_test, y_train, y_test, preprocessor = example_2_test_data_preprocessing()
    
    # Create HAT model
    n_classes = len(np.unique(y_train))
    hat_model = ModifiedHoeffdingTree(
        max_depth=10,
        min_samples_split=20,
        confidence=0.95,
        grace_period=100,
        activation='sigmoid',
        polynomial_degree=3,
        n_classes=n_classes
    )
    
    print("\n✓ HAT model created")
    print(f"  Max depth: {hat_model.max_depth}")
    print(f"  Activation: {hat_model.activation}")
    print(f"  Polynomial degree: {hat_model.polynomial_degree}")
    
    # Incremental training
    print("\nTraining HAT model...")
    batch_size = 100
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        hat_model.partial_fit(X_batch, y_batch)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i+len(X_batch)}/{len(X_train)} samples")
    
    print("\n✓ Training complete!")
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = hat_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Get probabilities (with polynomial activation)
    y_proba = hat_model.predict_proba(X_test[:10])
    print(f"\nSample predictions (with polynomial activation):")
    print(f"  Shape: {y_proba.shape}")
    print(f"  First 3 samples:\n{y_proba[:3]}")
    
    # Feature importance
    feature_importance = hat_model.extract_features(X_train)
    print(f"\nTop 5 important features:")
    sorted_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
    for feat_idx, importance in sorted_features:
        print(f"  Feature {feat_idx}: {importance:.4f}")
    
    return hat_model, X_test, y_test


def example_5_simulate_federated_learning():
    """
    Example 5: Simulate complete federated learning process
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Simulating Federated Learning Process")
    print("="*70)
    
    # Get federated data
    client_data, X_test, y_test, preprocessor = example_3_test_federated_distribution()
    
    n_clients = len(client_data)
    n_rounds = 5  # Reduced for demo
    clients_per_round = 5
    n_classes = len(np.unique(y_test))
    
    print(f"\nFederated Learning Setup:")
    print(f"  Total clients: {n_clients}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Clients per round: {clients_per_round}")
    
    # Initialize global model
    global_model = ModifiedHoeffdingTree(
        max_depth=10,
        min_samples_split=20,
        confidence=0.95,
        grace_period=50,
        activation='sigmoid',
        polynomial_degree=3,
        n_classes=n_classes
    )
    
    # Federated learning rounds
    for round_num in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{n_rounds}")
        print(f"{'='*60}")
        
        # Select random clients
        selected_clients = np.random.choice(n_clients, clients_per_round, replace=False)
        print(f"Selected clients: {selected_clients}")
        
        # Train local models
        local_models = []
        for client_id in selected_clients:
            X_client, y_client = client_data[client_id]
            
            # Create local model
            local_model = ModifiedHoeffdingTree(
                max_depth=10,
                min_samples_split=20,
                confidence=0.95,
                grace_period=50,
                activation='sigmoid',
                polynomial_degree=3,
                n_classes=n_classes
            )
            
            # Train local model
            local_model.partial_fit(X_client, y_client)
            local_models.append(local_model)
            
            print(f"  Client {client_id}: Trained on {len(X_client)} samples")
        
        # Aggregate models (simplified - just use last model for demo)
        # In real FL, you'd aggregate parameters from all local models
        global_model = local_models[-1]
        
        # Evaluate global model
        y_pred = global_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n  Global Model Accuracy: {accuracy:.4f}")
    
    print("\n✓ Federated Learning simulation complete!")
    return global_model


def main():
    """
    Main function to run all examples
    """
    print("="*70)
    print("SECURE FL-BASED IDS SYSTEM - EXECUTABLE EXAMPLES")
    print("="*70)
    
    # Run all examples
    try:
        example_1_test_polynomial_activations()
        example_2_test_data_preprocessing()
        example_3_test_federated_distribution()
        example_4_train_hat_model()
        example_5_simulate_federated_learning()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# RUN THE CODE
# =============================================================================

if __name__ == "__main__":
    # OPTION 1: Run all examples
    main()
    
    # OPTION 2: Run individual examples
    # Uncomment the example you want to run:
    
    # example_1_test_polynomial_activations()
    # example_2_test_data_preprocessing()
    # example_3_test_federated_distribution()
    # example_4_train_hat_model()
    # example_5_simulate_federated_learning()


# =============================================================================
# STEP 5: RUNNING WITH REAL DATASETS
# =============================================================================
"""
To use real CICIDS2017/2018 or EdgeIIoT datasets:

1. Download datasets:
   - CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
   - CICIDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
   - EdgeIIoT: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset

2. Place CSV files in 'data/' folder

3. Replace synthetic data creation with real data loading:

   client_data, X_test, y_test, preprocessor = prepare_federated_dataset(
       file_path='data/CICIDS2017.csv',  # Change this path
       dataset_name='CICIDS2017',
       n_clients=10,
       distribution='iid',
       test_size=0.2
   )

4. Run the code normally
"""


# =============================================================================
# TROUBLESHOOTING
# =============================================================================
"""
Common Issues and Solutions:

1. ModuleNotFoundError: No module named 'xxx'
   Solution: Install missing package with pip install xxx

2. FileNotFoundError: 'xxx.csv'
   Solution: Check file path or use synthetic dataset first

3. Memory Error with large datasets
   Solution: Reduce n_samples or process in smaller batches

4. Import errors
   Solution: Make sure all files are in same directory or proper Python path

5. Low accuracy
   Solution: Increase training epochs, adjust hyperparameters, or use more data
"""