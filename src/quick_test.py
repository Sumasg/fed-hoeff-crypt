"""
QUICK START SCRIPT
Run this script first to test if everything works!

This script will:
1. Test all dependencies
2. Create synthetic data
3. Run a simple federated learning demo
4. Show you it's working!
"""

print("Starting Quick Test...")
print("="*70)

# ============================================================================
# STEP 1: Check Dependencies
# ============================================================================
print("\n Checking dependencies...")

required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas', 
    'sklearn': 'scikit-learn'
}

missing_packages = []
for package, install_name in required_packages.items():
    try:
        __import__(package)
        print(f"  {package}")
    except ImportError:
        print(f" {package} - MISSING")
        missing_packages.append(install_name)

if missing_packages:
    print(f"\n Missing packages: {', '.join(missing_packages)}")
    print(f"\nInstall them with:")
    print(f"pip install {' '.join(missing_packages)}")
    exit()

print("\n All dependencies installed!")

# ============================================================================
# STEP 2: Import Required Modules
# ============================================================================
print("\n Importing modules...")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Set random seed
np.random.seed(42)

print(" Modules imported successfully!")

# ============================================================================
# STEP 3: Create Simple Test Data
# ============================================================================
print("\n Creating test dataset...")

# Create simple 2-class dataset
n_samples = 1000
n_features = 10

# Generate features
X = np.random.randn(n_samples, n_features)
# Create simple pattern: if feature_0 > 0, class 1, else class 0
y = (X[:, 0] > 0).astype(int)

print(f" Created dataset: {n_samples} samples, {n_features} features")
print(f"  Class 0: {np.sum(y==0)} samples")
print(f"  Class 1: {np.sum(y==1)} samples")

# ============================================================================
# STEP 4: Split Data for Federated Learning
# ============================================================================
print("\n Distributing data to 10 clients...")

# Split into train/test
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Distribute to 10 clients
n_clients = 10
client_data = []

indices = np.random.permutation(len(X_train))
X_shuffled = X_train[indices]
y_shuffled = y_train[indices]

X_chunks = np.array_split(X_shuffled, n_clients)
y_chunks = np.array_split(y_shuffled, n_clients)

for i in range(n_clients):
    client_data.append((X_chunks[i], y_chunks[i]))
    print(f"  Client {i}: {len(X_chunks[i])} samples")

print(f"\n Data distributed!")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ============================================================================
# STEP 5: Simple Federated Learning Simulation
# ============================================================================
print("\n Running simple FL simulation...")
print("  (Using basic decision tree for demo)")

from sklearn.tree import DecisionTreeClassifier

# Global model
global_predictions = None

# 3 rounds of federated learning
n_rounds = 3
clients_per_round = 5

for round_num in range(n_rounds):
    print(f"\n  Round {round_num + 1}/{n_rounds}")
    
    # Select random clients
    selected_clients = np.random.choice(n_clients, clients_per_round, replace=False)
    print(f"    Selected clients: {list(selected_clients)}")
    
    # Train local models
    local_predictions = []
    for client_id in selected_clients:
        X_client, y_client = client_data[client_id]
        
        # Train local model
        local_model = DecisionTreeClassifier(max_depth=5, random_state=client_id)
        local_model.fit(X_client, y_client)
        
        # Get predictions
        pred = local_model.predict(X_test)
        local_predictions.append(pred)
    
    # Aggregate predictions (majority voting)
    local_predictions = np.array(local_predictions)
    global_predictions = np.round(np.mean(local_predictions, axis=0)).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test, global_predictions)
    print(f"    Global accuracy: {accuracy:.4f}")

print("\n FL simulation complete!")

# ============================================================================
# STEP 6: Final Results
# ============================================================================
print("\n" + "="*70)
print(" FINAL RESULTS")
print("="*70)

final_accuracy = accuracy_score(y_test, global_predictions)
print(f"\n Final Global Model Accuracy: {final_accuracy:.4f}")

# Show confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, global_predictions)
print(f"\nConfusion Matrix:")
print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

print("\n" + "="*70)
print(" SUCCESS! Everything is working!")
print("="*70)

print("\n Next Steps:")
print("  1. Dependencies are installed")
print("  2. Basic FL is working")
print("  3. Now use the full modules (data_preprocessing.py & hat_algorithm.py)")
print("  4. Run main.py for complete examples")
print("  5. Add your real datasets (CICIDS2017/2018/EdgeIIoT)")
print("\n  To test the full modules:")
print("  1. Save Functionality 1 as 'data_preprocessing.py'")
print("  2. Save Functionality 2 as 'hat_algorithm.py'")
print("  3. Save the setup guide as 'main.py'")
print("  4. Run: python main.py")

print("\n You're all set to proceed!")