"""
Create a compatible model for the Streamlit app from the legacy trained model
"""

import os
import sys
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import process_directory

# Load the legacy model
print("Loading legacy model...")
legacy_model = joblib.load("models/random_forest_model.pkl")

# Load the training data to fit the scaler
print("Loading training data...")
DATA_CLEAN = os.path.join("data", "clean")
DATA_STEGO = os.path.join("data", "stego")

X_clean, y_clean = process_directory(DATA_CLEAN, 0)
X_stego, y_stego = process_directory(DATA_STEGO, 1)

X = np.vstack([X_clean, X_stego])
y = np.hstack([y_clean, y_stego])

print(f"Loaded {len(X)} samples")

# Create and fit scaler
print("Creating scaler...")
scaler = StandardScaler()
scaler.fit(X)


# Create a compatible detector object
class LegacyDetectorWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.model_type = "random_forest"
        self.metrics = {"accuracy": 0.75, "f1": 0.75, "roc_auc": 0.75}  # Placeholder

    def predict(self, file_path):
        from features import extract_features

        features = extract_features(file_path)
        if features is None:
            raise ValueError("Could not extract features")

        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]

        return prediction, confidence

    def load_model(self, path):
        pass  # Already loaded


# Create wrapper
detector = LegacyDetectorWrapper(legacy_model, scaler)

# Save as compatible model
print("Saving compatible model...")
joblib.dump(detector, "models/random_forest_compatible.pkl")
print("✅ Compatible model saved to models/random_forest_compatible.pkl")
