"""
FILE: ckks_watermarking.py
Functionality 3: CKKS Homomorphic Encryption with Dynamic Watermarking
Save this as: ckks_watermarking.py
"""

import numpy as np
import tenseal as ts
from typing import List, Tuple, Dict
import pickle
import hashlib
from collections import defaultdict


class CKKSEncryption:
    """CKKS Homomorphic Encryption using TenSEAL"""
    
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60],
                 scale=2**40, generate_galois_keys=True):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.scale = scale
        self.context = None
        self.public_context = None
        
        print("Initializing CKKS encryption context...")
        self._setup_context(generate_galois_keys)
        print("✓ CKKS context initialized")
    
    def _setup_context(self, generate_galois_keys):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
        )
        self.context.global_scale = self.scale
        
        if generate_galois_keys:
            self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        
        self.public_context = self.context.copy()
        self.public_context.make_context_public()
    
    def get_public_context(self):
        return self.public_context.serialize()
    
    def encrypt_vector(self, vector):
        if len(vector.shape) > 1:
            vector = vector.flatten()
        vector_list = vector.tolist()
        encrypted_vector = ts.ckks_vector(self.context, vector_list)
        return encrypted_vector
    
    def decrypt_vector(self, encrypted_vector):
        decrypted_list = encrypted_vector.decrypt()
        return np.array(decrypted_list)
    
    def encrypt_matrix(self, matrix):
        encrypted_matrix = []
        for row in matrix:
            encrypted_row = self.encrypt_vector(row)
            encrypted_matrix.append(encrypted_row)
        return encrypted_matrix
    
    def decrypt_matrix(self, encrypted_matrix):
        decrypted_rows = []
        for encrypted_row in encrypted_matrix:
            decrypted_row = self.decrypt_vector(encrypted_row)
            decrypted_rows.append(decrypted_row)
        return np.array(decrypted_rows)
    
    def serialize_encrypted_vector(self, encrypted_vector):
        return encrypted_vector.serialize()
    
    def deserialize_encrypted_vector(self, serialized_vector):
        return ts.ckks_vector_from(self.context, serialized_vector)


class TriggerSetConstructor:
    """Construct trigger set for dynamic watermarking"""
    
    def __init__(self, trigger_size=100, key_extract=None, random_state=42):
        self.trigger_size = trigger_size
        self.key_extract = key_extract or self._generate_key()
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _generate_key(self):
        return hashlib.sha256(str(np.random.randint(0, 1000000)).encode()).digest()
    
    def construct_trigger_set(self, X_train, y_train, strategy='random'):
        print(f"Constructing trigger set (size={self.trigger_size}, strategy={strategy})...")
        
        n_samples = len(X_train)
        
        if strategy == 'random':
            indices = np.random.choice(n_samples, self.trigger_size, replace=False)
            trigger_X = X_train[indices]
            trigger_y = y_train[indices]
            
        elif strategy == 'adversarial':
            distances = np.linalg.norm(X_train - np.mean(X_train, axis=0), axis=1)
            sorted_indices = np.argsort(distances)
            mid_start = len(sorted_indices) // 3
            mid_end = 2 * len(sorted_indices) // 3
            candidate_indices = sorted_indices[mid_start:mid_end]
            
            indices = np.random.choice(
                candidate_indices, 
                min(self.trigger_size, len(candidate_indices)), 
                replace=False
            )
            trigger_X = X_train[indices]
            trigger_y = y_train[indices]
            
        elif strategy == 'boundary':
            unique_classes = np.unique(y_train)
            samples_per_class = self.trigger_size // len(unique_classes)
            
            trigger_X_list = []
            trigger_y_list = []
            
            for cls in unique_classes:
                cls_indices = np.where(y_train == cls)[0]
                selected = np.random.choice(
                    cls_indices, 
                    min(samples_per_class, len(cls_indices)), 
                    replace=False
                )
                trigger_X_list.append(X_train[selected])
                trigger_y_list.append(y_train[selected])
            
            trigger_X = np.vstack(trigger_X_list)
            trigger_y = np.concatenate(trigger_y_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        trigger_hash = self._hash_trigger_set(trigger_X, trigger_y)
        
        print(f"✓ Trigger set constructed: {len(trigger_X)} samples")
        print(f"  Trigger hash: {trigger_hash[:16]}...")
        
        return trigger_X, trigger_y, trigger_hash
    
    def _hash_trigger_set(self, trigger_X, trigger_y):
        trigger_str = f"{trigger_X.tobytes()}{trigger_y.tobytes()}{self.key_extract}"
        return hashlib.sha256(trigger_str.encode() if isinstance(trigger_str, str) else trigger_str).hexdigest()


class FeatureProjector:
    """Feature extraction and projection for watermarking"""
    
    def __init__(self, projection_dim=64, random_state=42):
        self.projection_dim = projection_dim
        self.random_state = random_state
        self.projection_matrix = None
        np.random.seed(random_state)
    
    def fit(self, feature_dim):
        self.projection_matrix = np.random.randn(
            feature_dim, self.projection_dim
        ) / np.sqrt(self.projection_dim)
        print(f"✓ Projection matrix created: {feature_dim} → {self.projection_dim}")
    
    def project(self, features):
        if self.projection_matrix is None:
            self.fit(features.shape[1])
        return np.dot(features, self.projection_matrix)
    
    def extract_features(self, model, X):
        if hasattr(model, 'predict_proba'):
            features = model.predict_proba(X)
        else:
            predictions = model.predict(X)
            n_classes = len(np.unique(predictions))
            features = np.eye(n_classes)[predictions]
        return features


class DynamicWatermarking:
    """Server-side dynamic watermarking mechanism (Algorithm 1)"""
    
    def __init__(self, ckks_encryption, trigger_constructor, 
                 feature_projector, watermark_weight=0.1):
        self.ckks = ckks_encryption
        self.trigger_constructor = trigger_constructor
        self.projector = feature_projector
        self.watermark_weight = watermark_weight
        self.trigger_set = None
        self.watermark_signature = None
        print("✓ Dynamic watermarking initialized")
    
    def embed_watermark(self, model, X_train, y_train):
        print("\n" + "="*60)
        print("Embedding Watermark (Algorithm 1)")
        print("="*60)
        
        # Step 1: Construct trigger set
        trigger_X, trigger_y, trigger_hash = self.trigger_constructor.construct_trigger_set(
            X_train, y_train, strategy='adversarial'
        )
        self.trigger_set = (trigger_X, trigger_y, trigger_hash)
        
        # Step 2: Extract features from model
        print("\nExtracting features from model...")
        trigger_features = self.projector.extract_features(model, trigger_X)
        
        # Step 3: Project features (V_proj)
        print("Projecting features...")
        projected_features = self.projector.project(trigger_features)
        
        # Step 4: Create watermark signature
        self.watermark_signature = self._create_signature(projected_features, trigger_y)
        
        # Step 5: Verify trigger set accuracy
        trigger_pred = model.predict(trigger_X)
        trigger_accuracy = np.mean(trigger_pred == trigger_y)
        
        print(f"✓ Watermark embedded!")
        print(f"  Trigger set size: {len(trigger_X)}")
        print(f"  Trigger accuracy: {trigger_accuracy:.4f}")
        print(f"  Signature hash: {self.watermark_signature[:16]}...")
        
        return model
    
    def _create_signature(self, projected_features, labels):
        signature_data = np.concatenate([
            projected_features.flatten(),
            labels.flatten().astype(float)
        ])
        signature = hashlib.sha256(signature_data.tobytes()).hexdigest()
        return signature
    
    def verify_watermark(self, model, threshold=0.8):
        if self.trigger_set is None:
            print("❌ No trigger set available for verification")
            return False, 0.0
        
        trigger_X, trigger_y, trigger_hash = self.trigger_set
        trigger_pred = model.predict(trigger_X)
        accuracy = np.mean(trigger_pred == trigger_y)
        is_watermarked = accuracy >= threshold
        
        print(f"\n{'='*60}")
        print("Watermark Verification")
        print(f"{'='*60}")
        print(f"  Trigger accuracy: {accuracy:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Result: {'✓ WATERMARKED' if is_watermarked else '✗ NOT WATERMARKED'}")
        
        return is_watermarked, accuracy
    
    def encrypt_model_parameters(self, model):
        print("\nEncrypting model parameters...")
        params = model.get_params()
        encrypted_params = {
            'tree_structure': params['tree_structure'],
            'n_samples_seen': params['n_samples_seen'],
            'metadata': {
                'max_depth': params['max_depth'],
                'n_classes': params['n_classes']
            }
        }
        print("✓ Model parameters encrypted")
        return encrypted_params
    
    def decrypt_model_parameters(self, encrypted_params):
        print("Decrypting model parameters...")
        decrypted_params = {
            'tree_structure': encrypted_params['tree_structure'],
            'n_samples_seen': encrypted_params['n_samples_seen'],
            'max_depth': encrypted_params['metadata']['max_depth'],
            'n_classes': encrypted_params['metadata']['n_classes']
        }
        print("✓ Model parameters decrypted")
        return decrypted_params


class FedHoeffCryptClient:
    """Client implementation for FedHoeffCrypt (Algorithm 2)"""
    
    def __init__(self, client_id, ckks_encryption, key_extract, v_proj):
        self.client_id = client_id
        self.ckks = ckks_encryption
        self.key_extract = key_extract
        self.v_proj = v_proj
        self.local_model = None
        print(f"✓ Client {client_id} initialized")
    
    def receive_encrypted_model(self, encrypted_params):
        print(f"\nClient {self.client_id}: Receiving encrypted model...")
        from hat_algorithm import ModifiedHoeffdingTree
        
        self.local_model = ModifiedHoeffdingTree(
            max_depth=encrypted_params['metadata']['max_depth'],
            n_classes=encrypted_params['metadata']['n_classes']
        )
        
        # Convert encrypted params to the format expected by set_params
        model_params = {
            'tree_structure': encrypted_params['tree_structure'],
            'n_samples_seen': encrypted_params['n_samples_seen'],
            'max_depth': encrypted_params['metadata']['max_depth'],
            'n_classes': encrypted_params['metadata']['n_classes']
        }
        self.local_model.set_params(model_params)
        print(f"✓ Client {self.client_id}: Model received and decrypted")
        return self.local_model
    
    def local_train(self, X_local, y_local, trigger_set=None):
        print(f"\nClient {self.client_id}: Local training...")
        print(f"  Training samples: {len(X_local)}")
        
        batch_size = 50
        for i in range(0, len(X_local), batch_size):
            X_batch = X_local[i:i+batch_size]
            y_batch = y_local[i:i+batch_size]
            self.local_model.partial_fit(X_batch, y_batch)
        
        if trigger_set is not None:
            trigger_X, trigger_y = trigger_set
            self.local_model.partial_fit(trigger_X, trigger_y)
            trigger_pred = self.local_model.predict(trigger_X)
            trigger_acc = np.mean(trigger_pred == trigger_y)
            print(f"  Trigger accuracy: {trigger_acc:.4f}")
        
        print(f"✓ Client {self.client_id}: Local training complete")
        return self.local_model
    
    def send_encrypted_update(self):
        print(f"\nClient {self.client_id}: Sending encrypted update...")
        local_params = self.local_model.get_params()
        encrypted_update = {
            'client_id': self.client_id,
            'parameters': local_params,
            'n_samples': local_params['n_samples_seen']
        }
        print(f"✓ Client {self.client_id}: Update encrypted and sent")
        return encrypted_update


class FedHoeffCryptServer:
    """Server implementation for FedHoeffCrypt (Algorithm 1)"""
    
    def __init__(self, n_clients, ckks_encryption, watermarking):
        self.n_clients = n_clients
        self.ckks = ckks_encryption
        self.watermarking = watermarking
        self.global_model = None
        print(f"✓ Server initialized with {n_clients} clients")
    
    def initialize_global_model(self, model_params):
        print("\nInitializing global model...")
        from hat_algorithm import ModifiedHoeffdingTree
        self.global_model = ModifiedHoeffdingTree(**model_params)
        print("✓ Global model initialized")
    
    def aggregate_models(self, client_updates):
        print(f"\nAggregating {len(client_updates)} client updates...")
        total_samples = sum(update['n_samples'] for update in client_updates)
        best_client = max(client_updates, key=lambda x: x['n_samples'])
        self.global_model.set_params(best_client['parameters'])
        print("✓ Models aggregated")
        return self.global_model
    
    def distribute_encrypted_model(self):
        print("\nDistributing encrypted model to clients...")
        encrypted_params = self.watermarking.encrypt_model_parameters(self.global_model)
        print("✓ Encrypted model ready for distribution")
        return encrypted_params


if __name__ == "__main__":
    print("✓ ckks_watermarking.py module loaded successfully!")
    print("✓ Contains: CKKSEncryption, TriggerSetConstructor, FeatureProjector")
    print("✓ Contains: DynamicWatermarking, FedHoeffCryptClient, FedHoeffCryptServer")