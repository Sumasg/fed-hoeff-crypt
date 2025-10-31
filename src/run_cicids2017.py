"""
COMPLETE EXAMPLE - CICIDS2017 with Your Downloaded Files

This script loads all your CICIDS2017 CSV files and trains the FL-IDS system
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our modules
from data_preprocessing import IDSDataPreprocessor, FederatedDataDistributor
from hat_algorithm import ModifiedHoeffdingTree
from multi_file_loader import prepare_federated_dataset_multifile


def main():
    """
    Main function to run complete FL-IDS with your CICIDS2017 files
    """
    
    print("="*70)
    print("SECURE FL-BASED IDS WITH CICIDS2017")
    print("="*70)
    
    # =========================================================================
    # STEP 1: SET YOUR FOLDER PATH
    # =========================================================================
    # CHANGE THIS to match your actual folder location
    folder_path = r'D:\phdcode\data\CICIDS2017'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        print("\n📝 Please update the folder_path variable to match your location")
        print("   Example: folder_path = r'C:\\Users\\YourName\\Downloads\\CICIDS2017'")
        return
    
    print(f"✓ Found folder: {folder_path}\n")
    
    # =========================================================================
    # STEP 2: LOAD AND PREPARE DATA
    # =========================================================================
    print("Loading your CICIDS2017 CSV files...\n")
    
    try:
        # Option A: Load ALL CSV files from folder
        client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
            folder_path=folder_path,
            file_pattern='*.csv',           # Load all CSV files
            dataset_name='CICIDS2017',
            n_clients=10,                   # 10 FL clients as per your requirements
            distribution='iid',             # Random distribution
            test_size=0.2,                  # 80/20 split
            sample_size_per_file=100000,    # Limit to 100k rows per file for memory
            random_state=42,
            verbose=True
        )
        
        print("\n✓ Data loading and preparation complete!")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the folder path is correct")
        print("2. Check that CSV files are in the folder")
        print("3. Try reducing sample_size_per_file if memory error")
        return
    
    # =========================================================================
    # STEP 3: SETUP FEDERATED LEARNING
    # =========================================================================
    print("\n" + "="*70)
    print("FEDERATED LEARNING SETUP")
    print("="*70)
    
    n_clients = len(client_data)
    n_rounds = 60                    # As per your requirements
    clients_per_round = 10            # As per your requirements
    n_classes = len(np.unique(y_test))
    
    print(f"Total clients: {n_clients}")
    print(f"Training rounds: {n_rounds}")
    print(f"Clients per round: {clients_per_round}")
    print(f"Number of classes: {n_classes}")
    print(f"Test set size: {len(y_test)}")
    
    # =========================================================================
    # STEP 4: FEDERATED LEARNING WITH HAT
    # =========================================================================
    print("\n" + "="*70)
    print("STARTING FEDERATED LEARNING")
    print("="*70)
    
    # Initialize global model
    global_model = ModifiedHoeffdingTree(
        max_depth=10,
        min_samples_split=20,
        confidence=0.95,
        grace_period=200,            # As per your requirements
        activation='sigmoid',
        polynomial_degree=3,         # For HE compatibility
        n_classes=n_classes
    )
    
    # Store metrics for each round
    round_accuracies = []
    
    # Federated learning rounds
    for round_num in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{n_rounds}")
        print(f"{'='*60}")
        
        # Select random clients for this round
        selected_clients = np.random.choice(
            n_clients, 
            clients_per_round, 
            replace=False
        )
        print(f"Selected clients: {selected_clients.tolist()}")
        
        # Train local models
        local_models = []
        for client_id in selected_clients:
            X_client, y_client = client_data[client_id]
            
            # Create local model
            local_model = ModifiedHoeffdingTree(
                max_depth=10,
                min_samples_split=20,
                confidence=0.95,
                grace_period=200,
                activation='sigmoid',
                polynomial_degree=3,
                n_classes=n_classes
            )
            
            # Train with 10 epochs as per requirements
            print(f"  Training Client {client_id}...")
            for epoch in range(10):  # 10 epochs per round
                # Shuffle data for each epoch
                indices = np.random.permutation(len(X_client))
                X_shuffled = X_client[indices]
                y_shuffled = y_client[indices]
                
                # Train incrementally
                batch_size = 100
                for i in range(0, len(X_shuffled), batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    local_model.partial_fit(X_batch, y_batch)
            
            local_models.append(local_model)
            print(f"    ✓ Client {client_id} training complete")
        
        # Aggregate models (simplified - use last trained model)
        # In production, implement proper federated averaging
        global_model = local_models[-1]
        
        # Evaluate global model
        y_pred = global_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        round_accuracies.append(accuracy)
        
        print(f"\n  📊 Round {round_num + 1} Results:")
        print(f"     Global Model Accuracy: {accuracy:.4f}")
        
        # Show progress every 10 rounds
        if (round_num + 1) % 10 == 0:
            print(f"\n  ✓ Completed {round_num + 1}/{n_rounds} rounds")
            print(f"     Average accuracy (last 10 rounds): {np.mean(round_accuracies[-10:]):.4f}")
    
    # =========================================================================
    # STEP 5: FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Final predictions
    y_pred = global_model.predict(X_test)
    y_proba = global_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Final Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(cm)
    
    # Calculate TP, TN, FP, FN (for binary classification)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\n   True Positives (TP): {tp}")
        print(f"   True Negatives (TN): {tn}")
        print(f"   False Positives (FP): {fp}")
        print(f"   False Negatives (FN): {fn}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    target_names = [preprocessor.label_encoder.classes_[i] 
                   for i in range(min(n_classes, len(preprocessor.label_encoder.classes_)))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance
    print(f"\n🔍 Top 10 Most Important Features:")
    feature_importance = global_model.extract_features(X_test)
    sorted_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for feat_idx, importance in sorted_features:
        feat_name = preprocessor.feature_names[feat_idx] if feat_idx < len(preprocessor.feature_names) else f"Feature_{feat_idx}"
        print(f"   {feat_name}: {importance:.4f}")
    
    # Training curve
    print(f"\n📉 Accuracy over Rounds:")
    for i in range(0, len(round_accuracies), 10):
        print(f"   Rounds {i+1}-{min(i+10, len(round_accuracies))}: {np.mean(round_accuracies[i:i+10]):.4f}")
    
    print("\n" + "="*70)
    print("✅ FEDERATED LEARNING COMPLETE!")
    print("="*70)
    
    # =========================================================================
    # STEP 6: SAVE RESULTS (OPTIONAL)
    # =========================================================================
    print("\n💾 Saving results...")
    
    # Save predictions to a dedicated results folder
    import pickle
    import datetime as dt
    import pandas as pd

    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"fl_ids_results_{timestamp}.csv")
    pkl_path = os.path.join(results_dir, f"fl_ids_model_params_{timestamp}.pkl")

    # Handle probability shape robustly
    proba = y_proba
    if proba.ndim == 1:
        proba0 = proba
        proba1 = 1 - proba
    else:
        proba0 = proba[:, 0]
        proba1 = proba[:, 1] if proba.shape[1] > 1 else 1 - proba0

    results_df = pd.DataFrame({
        "True_Label": y_test,
        "Predicted_Label": y_pred,
        "Probability_Class_0": proba0,
        "Probability_Class_1": proba1
    })
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to '{csv_path}'")

    # Save model parameters (for later use)
    model_params = global_model.get_params()
    with open(pkl_path, "wb") as f:
        pickle.dump(model_params, f)
    print(f"✓ Model parameters saved to '{pkl_path}'")

    print("\n🎉 All done! Your FL-IDS system is trained and evaluated.")


# =============================================================================
# ALTERNATIVE: LOAD ONLY SPECIFIC FILES
# =============================================================================

def load_specific_files_example():
    """
    Example: Load only specific files instead of all files
    """
    print("Loading specific CICIDS2017 files...\n")
    
    # Option 1: Specify exact file paths
    file_paths = [
        r'C:\path\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        r'C:\path\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        r'C:\path\Friday-WorkingHours-Morning.pcap_ISCX.csv',
        r'C:\path\Monday-WorkingHours.pcap_ISCX.csv',
        r'C:\path\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        r'C:\path\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        r'C:\path\Tuesday-WorkingHours.pcap_ISCX.csv',
        r'C:\path\Wednesday-workingHours.pcap_ISCX.csv'
    ]
    
    client_data, X_test, y_test, preprocessor = prepare_federated_dataset_multifile(
        file_paths=file_paths,
        dataset_name='CICIDS2017',
        n_clients=10,
        distribution='iid',
        sample_size_per_file=50000  # Adjust based on your RAM
    )
    
    # Continue with training...


# =============================================================================
# RUN THE CODE
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    # IMPORTANT: Update this path to your actual folder location!
    # Example Windows path: r'C:\Users\YourName\Downloads\CICIDS2017'
    # Example Linux path: '/home/yourname/Downloads/CICIDS2017'
    
    # Run the main function
    main()
    
    # Or run with specific files
    # load_specific_files_example()