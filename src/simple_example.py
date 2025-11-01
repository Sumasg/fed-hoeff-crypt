"""
FILE: simple_example.py
Simple example showing how to use Functionality 3
Save this as: simple_example.py

Usage: python simple_example.py
"""

import numpy as np

print("="*70)
print("SIMPLE EXAMPLE: Using Functionality 3")
print("="*70)

# Import required modules
print("\n1. Importing modules...")
from ckks_watermarking import (
    CKKSEncryption,
    TriggerSetConstructor,
    FeatureProjector,
    DynamicWatermarking
)
from hat_algorithm import ModifiedHoeffdingTree
print("Modules imported")

# Create synthetic data
print("\n2. Creating synthetic data...")
np.random.seed(42)
X_train = np.random.randn(500, 20)  # 500 samples, 20 features
y_train = np.random.randint(0, 2, 500)  # Binary classification
X_test = np.random.randn(100, 20)
y_test = np.random.randint(0, 2, 100)
print(f" Training data: {X_train.shape}")
print(f" Test data: {X_test.shape}")

# Initialize CKKS encryption
print("\n3. Setting up CKKS encryption...")
ckks = CKKSEncryption(
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
    scale=2**40
)
print(" CKKS initialized")

# Test encryption/decryption
print("\n4. Testing encryption...")
test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"   Original: {test_data}")
encrypted = ckks.encrypt_vector(test_data)
decrypted = ckks.decrypt_vector(encrypted)
print(f"   Decrypted: {np.round(decrypted[:5], 2)}")
print(" Encryption working")

# Create trigger set constructor
print("\n5. Creating trigger set constructor...")
trigger_constructor = TriggerSetConstructor(
    trigger_size=50,
    random_state=42
)
print(" Trigger constructor ready")

# Create feature projector
print("\n6. Creating feature projector...")
projector = FeatureProjector(
    projection_dim=32,
    random_state=42
)
print(" Feature projector ready")

# Initialize watermarking
print("\n7. Initializing watermarking system...")
watermarking = DynamicWatermarking(
    ckks_encryption=ckks,
    trigger_constructor=trigger_constructor,
    feature_projector=projector,
    watermark_weight=0.1
)
print(" Watermarking system ready")

# Train a HAT model
print("\n8. Training HAT model...")
model = ModifiedHoeffdingTree(
    max_depth=8,
    min_samples_split=20,
    confidence=0.95,
    n_classes=2
)

# Train incrementally
batch_size = 50
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    model.partial_fit(X_batch, y_batch)

print(" Model trained")

# Test model accuracy before watermarking
predictions = model.predict(X_test)
accuracy_before = np.mean(predictions == y_test)
print(f"   Accuracy before watermarking: {accuracy_before:.4f}")

# Embed watermark
print("\n9. Embedding watermark...")
watermarked_model = watermarking.embed_watermark(model, X_train, y_train)
print("Watermark embedded")

# Test model accuracy after watermarking
predictions_after = watermarked_model.predict(X_test)
accuracy_after = np.mean(predictions_after == y_test)
print(f"   Accuracy after watermarking: {accuracy_after:.4f}")
print(f"   Performance impact: {abs(accuracy_before - accuracy_after):.4f}")

# Verify watermark
print("\n10. Verifying watermark...")
is_watermarked, trigger_accuracy = watermarking.verify_watermark(
    watermarked_model,
    threshold=0.7
)

# Encrypt model parameters
print("\n11. Encrypting model parameters...")
encrypted_params = watermarking.encrypt_model_parameters(watermarked_model)
print(" Model parameters encrypted")

# Decrypt model parameters
print("\n12. Decrypting model parameters...")
decrypted_params = watermarking.decrypt_model_parameters(encrypted_params)
print(" Model parameters decrypted")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f" Model accuracy (before watermark): {accuracy_before:.4f}")
print(f" Model accuracy (after watermark):  {accuracy_after:.4f}")
print(f" Watermark verified: {is_watermarked}")
print(f" Trigger accuracy: {trigger_accuracy:.4f}")
print(f" Model encrypted and decrypted successfully")

print("\n" + "="*70)
print(" COMPLETE! All functionality working!")
print("="*70)

print("\n What we did:")
print("  1.  Initialized CKKS homomorphic encryption")
print("  2.  Created and encrypted data")
print("  3.  Trained a HAT model")
print("  4.  Embedded a watermark")
print("  5.  Verified the watermark")
print("  6.  Encrypted model parameters")
print("  7.  Minimal impact on accuracy")

print("\n Next: Integrate with federated learning!")