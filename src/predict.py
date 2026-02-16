"""
Command-line prediction script
Uses the new enhanced model system
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model_train import SteganographyDetector
except ImportError:
    # Fallback to old system
    import joblib
    from features import extract_features

    MODEL_PATH = "model.pkl"

    def predict_legacy(file_path: str) -> None:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            print("Please train a model first using: python src/model_train.py")
            return

        model = joblib.load(MODEL_PATH)
        features = extract_features(file_path)

        if features is None:
            print("Error extracting features.")
            return

        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        probability = model.predict_proba(features_reshaped)[0]

        print(f"\n{'='*60}")
        print(f"Prediction for: {file_path}")
        print(f"{'='*60}")

        if prediction == 1:
            print(f"Result: ⚠️  STEGANOGRAPHY DETECTED")
            print(f"Confidence: {probability[1]*100:.2f}%")
        else:
            print(f"Result: ✅ CLEAN AUDIO")
            print(f"Confidence: {probability[0]*100:.2f}%")

        print(f"{'='*60}\n")


def predict_enhanced(file_path: str, model_path: str = None) -> None:
    """Predict using enhanced model system"""

    # Find available models
    if model_path is None:
        model_dir = "models"
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
            if models:
                model_path = os.path.join(model_dir, models[0])
                print(f"Using model: {models[0]}")

    if model_path is None or not os.path.exists(model_path):
        print("Error: No trained model found.")
        print("Please train a model first using: python src/model_train.py")
        return

    # Load model
    detector = SteganographyDetector()
    detector.load_model(model_path)

    # Predict
    try:
        prediction, confidence = detector.predict(file_path)

        print(f"\n{'='*60}")
        print(f"Prediction for: {file_path}")
        print(f"Model: {detector.model_type}")
        print(f"{'='*60}")

        if prediction == 1:
            print(f"Result: ⚠️  STEGANOGRAPHY DETECTED")
            print(f"Confidence: {confidence*100:.2f}%")
        else:
            print(f"Result: ✅ CLEAN AUDIO")
            print(f"Confidence: {confidence*100:.2f}%")

        # Show model metrics if available
        if detector.metrics:
            print(f"\nModel Performance Metrics:")
            print(f"  Accuracy:  {detector.metrics.get('accuracy', 0)*100:.2f}%")
            print(f"  F1 Score:  {detector.metrics.get('f1', 0)*100:.2f}%")
            print(f"  ROC AUC:   {detector.metrics.get('roc_auc', 0)*100:.2f}%")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect steganography in audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py audio.wav
  python src/predict.py audio.wav --model models/random_forest_model.pkl
        """,
    )

    parser.add_argument("file", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model", "-m", help="Path to specific model file", default=None
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Try enhanced prediction first
    try:
        predict_enhanced(args.file, args.model)
    except NameError:
        # Fall back to legacy
        predict_legacy(args.file)
