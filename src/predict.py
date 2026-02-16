import argparse
import os
import sys
import joblib

# Fix for relative imports when running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

MODEL_PATH = "model.pkl"


def predict(file_path: str) -> None:
    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: Model not found at {MODEL_PATH}. First train the model using src/train.py"
        )
        return

    # Load Model
    model = joblib.load(MODEL_PATH)

    # Extract Features
    features = extract_features(file_path)
    if features is None:
        print("Error extracting features.")
        return

    # Predict
    # Reshape for single sample
    features_reshaped = features.reshape(1, -1)
    prediction = model.predict(features_reshaped)[0]
    probability = model.predict_proba(features_reshaped)[0]

    print(f"\nPrediction for {file_path}:")
    if prediction == 1:
        print(
            f"Result: STEGANOGRAPHY DETECTED! (Confidence: {probability[1]*100:.2f}%)"
        )
    else:
        print(f"Result: CLEAN AUDIO. (Confidence: {probability[0]*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect steganography in an audio file."
    )
    parser.add_argument("file", help="Path to the audio file (.wav)")
    args = parser.parse_args()

    predict(args.file)
