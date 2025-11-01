"""
FILE: fedhoeffcrypt_integration.py
Functionality 4: Complete FedHoeffCrypt Integration & Training Pipeline
Save this as: fedhoeffcrypt_integration.py

This integrates all functionalities:
- Data preprocessing (Functionality 1)
- HAT algorithm (Functionality 2)
- CKKS encryption + Watermarking (Functionality 3)

Implements complete federated learning workflow:
- 60 rounds of training
- 10 clients with 5 selected per round
- 10 training epochs per round
- Watermark embedding
- Performance evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_preprocessing import prepare_federated_dataset
from hat_algorithm import ModifiedHoeffdingTree
from ckks_watermarking import (
    CKKSEncryption, TriggerSetConstructor,
    FeatureProjector, DynamicWatermarking,
    FedHoeffCryptClient, FedHoeffCryptServer
)


class FedHoeffCryptExperiment:
    """
    Complete FedHoeffCrypt experiment manager
    Implements Algorithms 1 & 2 with full integration
    """
    
    def __init__(self,
                 dataset_name='CICIDS2017',
                 n_clients=10,
                 clients_per_round=5,
                 n_rounds=60,
                 n_epochs=10,
                 watermark_enabled=True,
                 trigger_size=100,
                 random_state=42):
        """
        Initialize FedHoeffCrypt experiment
        
        Args:
            dataset_name: Name of dataset
            n_clients: Total number of clients
            clients_per_round: Clients selected per round
            n_rounds: Number of FL rounds
            n_epochs: Training epochs per round
            watermark_enabled: Enable watermarking
            trigger_size: Size of trigger set
            random_state: Random seed
        """
        self.dataset_name = dataset_name
        self.n_clients = n_clients
        self.clients_per_round = clients_per_round
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.watermark_enabled = watermark_enabled
        self.trigger_size = trigger_size
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Components
        self.ckks = None
        self.server = None
        self.clients = []
        self.watermarking = None
        
        # Data
        self.client_data = None
        self.X_test = None
        self.y_test = None
        self.preprocessor = None
        
        # Results
        self.results = {
            'round_accuracies': [],
            'round_precisions': [],
            'round_recalls': [],
            'round_f1_scores': [],
            'round_times': [],
            'watermark_accuracies': [],
            'confusion_matrices': []
        }
        
        print("="*70)
        print("FedHoeffCrypt Experiment Initialized")
        print("="*70)
        print(f"Dataset: {dataset_name}")
        print(f"Clients: {n_clients} (selecting {clients_per_round} per round)")
        print(f"Rounds: {n_rounds}")
        print(f"Epochs per round: {n_epochs}")
        print(f"Watermarking: {'Enabled' if watermark_enabled else 'Disabled'}")
        print("="*70)
    
    def load_and_prepare_data(self, file_path):
        """
        Load and prepare dataset for federated learning
        
        Args:
            file_path: Path to dataset CSV file
        """
        print("\n" + "="*70)
        print("STEP 1: Loading and Preparing Data")
        print("="*70)
        
        # Prepare federated dataset
        self.client_data, self.X_test, self.y_test, self.preprocessor = \
            prepare_federated_dataset(
                file_path=file_path,
                dataset_name=self.dataset_name,
                n_clients=self.n_clients,
                distribution='iid',
                test_size=0.2,
                random_state=self.random_state
            )
        
        # Get number of classes
        self.n_classes = len(np.unique(self.y_test))
        self.n_features = self.X_test.shape[1]
        
        print(f"\n Data prepared:")
        print(f"  Features: {self.n_features}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Client datasets: {len(self.client_data)}")
    
    def initialize_encryption_and_watermarking(self):
        """
        Initialize CKKS encryption and watermarking components
        """
        print("\n" + "="*70)
        print("STEP 2: Initializing Encryption & Watermarking")
        print("="*70)
        
        # Initialize CKKS encryption
        print("\nInitializing CKKS encryption...")
        self.ckks = CKKSEncryption(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale=2**40
        )
        
        if self.watermark_enabled:
            # Initialize trigger set constructor
            print("\nInitializing trigger set constructor...")
            trigger_constructor = TriggerSetConstructor(
                trigger_size=self.trigger_size,
                random_state=self.random_state
            )
            
            # Initialize feature projector
            print("\nInitializing feature projector...")
            projector = FeatureProjector(
                projection_dim=min(64, self.n_features),
                random_state=self.random_state
            )
            
            # Initialize watermarking
            print("\nInitializing watermarking system...")
            self.watermarking = DynamicWatermarking(
                ckks_encryption=self.ckks,
                trigger_constructor=trigger_constructor,
                feature_projector=projector,
                watermark_weight=0.1
            )
        
        print("\n Encryption and watermarking initialized")
    
    def initialize_server_and_clients(self):
        """
        Initialize FL server and clients
        """
        print("\n" + "="*70)
        print("STEP 3: Initializing Server and Clients")
        print("="*70)
        
        # Initialize server
        print("\nInitializing server...")
        self.server = FedHoeffCryptServer(
            n_clients=self.n_clients,
            ckks_encryption=self.ckks,
            watermarking=self.watermarking if self.watermark_enabled else None
        )
        
        # Initialize global model
        print("\nInitializing global model...")
        self.server.initialize_global_model({
            'max_depth': 10,
            'min_samples_split': 20,
            'confidence': 0.95,
            'grace_period': 100,
            'activation': 'sigmoid',
            'polynomial_degree': 3,
            'n_classes': self.n_classes
        })
        
        # Initialize clients
        print(f"\nInitializing {self.n_clients} clients...")
        self.clients = []
        for i in range(self.n_clients):
            if self.watermark_enabled:
                client = FedHoeffCryptClient(
                    client_id=i,
                    ckks_encryption=self.ckks,
                    key_extract=self.watermarking.trigger_constructor.key_extract,
                    v_proj=self.watermarking.projector
                )
            else:
                # Create simplified client without watermarking
                client = FedHoeffCryptClient(
                    client_id=i,
                    ckks_encryption=self.ckks,
                    key_extract=None,
                    v_proj=None
                )
            self.clients.append(client)
        
        print(f" Server and {self.n_clients} clients initialized")
    
    def train_federated_learning(self):
        """
        Execute complete federated learning training
        Implements Algorithm 1 & 2
        """
        print("\n" + "="*70)
        print("STEP 4: Federated Learning Training")
        print("="*70)
        print(f"Starting {self.n_rounds} rounds of federated learning...")
        
        start_time = time.time()
        
        for round_num in range(self.n_rounds):
            round_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"Round {round_num + 1}/{self.n_rounds}")
            print(f"{'='*70}")
            
            # Select random clients for this round
            selected_clients_ids = np.random.choice(
                self.n_clients,
                self.clients_per_round,
                replace=False
            )
            print(f"Selected clients: {selected_clients_ids}")
            
            # Distribute encrypted global model to selected clients
            print("\nDistributing encrypted global model...")
            encrypted_model = self.server.distribute_encrypted_model()
            
            # Client local training
            print("\nLocal training on selected clients...")
            client_updates = []
            
            for client_id in selected_clients_ids:
                client = self.clients[client_id]
                X_client, y_client = self.client_data[client_id]
                
                # Receive encrypted model
                client.receive_encrypted_model(encrypted_model)
                
                # Local training with multiple epochs
                print(f"\n  Client {client_id}:")
                for epoch in range(self.n_epochs):
                    # Shuffle data for each epoch
                    indices = np.random.permutation(len(X_client))
                    X_shuffled = X_client[indices]
                    y_shuffled = y_client[indices]
                    
                    # Train in batches
                    batch_size = 50
                    for i in range(0, len(X_shuffled), batch_size):
                        X_batch = X_shuffled[i:i+batch_size]
                        y_batch = y_shuffled[i:i+batch_size]
                        client.local_model.partial_fit(X_batch, y_batch)
                    
                    if (epoch + 1) % 5 == 0:
                        print(f"    Epoch {epoch + 1}/{self.n_epochs} completed")
                
                # Send encrypted update
                update = client.send_encrypted_update()
                client_updates.append(update)
            
            # Server aggregates models
            print("\nAggregating client models...")
            global_model = self.server.aggregate_models(client_updates)
            
            # Embed watermark if enabled
            if self.watermark_enabled and round_num % 10 == 0:
                print("\nEmbedding watermark in global model...")
                # Use combined data from all clients for trigger set
                X_combined = np.vstack([self.client_data[i][0] for i in range(min(3, self.n_clients))])
                y_combined = np.hstack([self.client_data[i][1] for i in range(min(3, self.n_clients))])
                
                global_model = self.watermarking.embed_watermark(
                    global_model, X_combined[:500], y_combined[:500]
                )
            
            # Evaluate global model
            print("\nEvaluating global model...")
            y_pred = global_model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            round_time = time.time() - round_start
            
            # Store results
            self.results['round_accuracies'].append(accuracy)
            self.results['round_precisions'].append(precision)
            self.results['round_recalls'].append(recall)
            self.results['round_f1_scores'].append(f1)
            self.results['round_times'].append(round_time)
            
            # Verify watermark if enabled
            if self.watermark_enabled and round_num % 10 == 0:
                is_watermarked, watermark_acc = self.watermarking.verify_watermark(
                    global_model, threshold=0.7
                )
                self.results['watermark_accuracies'].append({
                    'round': round_num + 1,
                    'is_watermarked': bool(is_watermarked),
                    'accuracy': float(watermark_acc)
                })
            
            # Print round summary
            print(f"\n{'='*70}")
            print(f"Round {round_num + 1} Summary:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Time:      {round_time:.2f}s")
            print(f"{'='*70}")
            
            # Save confusion matrix every 10 rounds
            if (round_num + 1) % 10 == 0:
                cm = confusion_matrix(self.y_test, y_pred)
                self.results['confusion_matrices'].append({
                    'round': round_num + 1,
                    'confusion_matrix': cm.tolist()
                })
        
        total_time = time.time() - start_time
        print(f"\n Training completed in {total_time:.2f}s")
        print(f"  Average time per round: {total_time/self.n_rounds:.2f}s")
    
    def evaluate_final_model(self):
        """
        Comprehensive evaluation of final model
        """
        print("\n" + "="*70)
        print("STEP 5: Final Model Evaluation")
        print("="*70)
        
        # Get final predictions
        final_model = self.server.global_model
        y_pred = final_model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        print("\nFinal Model Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate per-class metrics
        print("\nPer-Class Metrics:")
        class_report = classification_report(
            self.y_test, y_pred,
            target_names=[str(i) for i in range(self.n_classes)],
            zero_division=0
        )
        print(class_report)
        
        # Extract TP, TN, FP, FN for binary classification
        if self.n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            print("\nBinary Classification Metrics:")
            print(f"  True Positives (TP):  {tp}")
            print(f"  True Negatives (TN):  {tn}")
            print(f"  False Positives (FP): {fp}")
            print(f"  False Negatives (FN): {fn}")
        
        # Watermark verification if enabled
        if self.watermark_enabled:
            print("\n" + "="*70)
            print("Watermark Verification")
            print("="*70)
            is_watermarked, watermark_acc = self.watermarking.verify_watermark(
                final_model, threshold=0.7
            )
            
            print(f"\nWatermark Status: {' VERIFIED' if is_watermarked else ' NOT VERIFIED'}")
            print(f"Trigger Accuracy: {watermark_acc:.4f}")
        
        # Store final results
        self.results['final_metrics'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        return accuracy, precision, recall, f1, cm
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """
        Plot and save training curves
        """
        try:
            import matplotlib.pyplot as plt
            
            print("\n" + "="*70)
            print("STEP 6: Plotting Training Curves")
            print("="*70)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            rounds = list(range(1, self.n_rounds + 1))
            
            # Accuracy
            axes[0, 0].plot(rounds, self.results['round_accuracies'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy over Rounds')
            axes[0, 0].grid(True)
            
            # Precision
            axes[0, 1].plot(rounds, self.results['round_precisions'], 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision over Rounds')
            axes[0, 1].grid(True)
            
            # Recall
            axes[1, 0].plot(rounds, self.results['round_recalls'], 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].set_title('Recall over Rounds')
            axes[1, 0].grid(True)
            
            # F1-Score
            axes[1, 1].plot(rounds, self.results['round_f1_scores'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_title('F1-Score over Rounds')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training curves saved to {save_path}")
            plt.close()
            
        except ImportError:
            print(" Matplotlib not installed. Skipping plots.")
    
    def save_results(self, save_path='fedhoeffcrypt_results.json'):
        """
        Save experiment results to JSON file
        """
        print("\n" + "="*70)
        print("STEP 7: Saving Results")
        print("="*70)
        
        results_summary = {
            'experiment_info': {
                'dataset': self.dataset_name,
                'n_clients': self.n_clients,
                'clients_per_round': self.clients_per_round,
                'n_rounds': self.n_rounds,
                'n_epochs': self.n_epochs,
                'watermark_enabled': self.watermark_enabled,
                'timestamp': datetime.now().isoformat()
            },
            'training_results': {
                'round_accuracies': self.results['round_accuracies'],
                'round_precisions': self.results['round_precisions'],
                'round_recalls': self.results['round_recalls'],
                'round_f1_scores': self.results['round_f1_scores'],
                'round_times': self.results['round_times'],
                'watermark_accuracies': self.results['watermark_accuracies'],
                'confusion_matrices': self.results['confusion_matrices']
            },
            'final_metrics': self.results.get('final_metrics', {})
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f" Results saved to {save_path}")
    
    def run_complete_experiment(self, file_path):
        """
        Run complete FedHoeffCrypt experiment
        
        Args:
            file_path: Path to dataset CSV file
        """
        print("\n" + "="*70)
        print("STARTING FEDHOEFFCRYPT EXPERIMENT")
        print("="*70)
        
        # Step 1: Load data
        self.load_and_prepare_data(file_path)
        
        # Step 2: Initialize encryption and watermarking
        self.initialize_encryption_and_watermarking()
        
        # Step 3: Initialize server and clients
        self.initialize_server_and_clients()
        
        # Step 4: Train federated learning
        self.train_federated_learning()
        
        # Step 5: Evaluate final model
        self.evaluate_final_model()
        
        # Step 6: Plot results
        self.plot_training_curves()
        
        # Step 7: Save results
        self.save_results()
        
        print("\n" + "="*70)
        print(" EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)


# Example usage
if __name__ == "__main__":
    print("FedHoeffCrypt Integration Module")
    print("✓ Ready for complete federated learning experiments")
    print("\nUsage:")
    print("  from fedhoeffcrypt_integration import FedHoeffCryptExperiment")
    print("  experiment = FedHoeffCryptExperiment(dataset_name='CICIDS2017')")
    print("  experiment.run_complete_experiment('path/to/dataset.csv')")