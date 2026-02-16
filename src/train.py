import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix for relative imports when running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import process_directory

MODEL_PATH = "model.pkl"
DATA_CLEAN = os.path.join("data", "clean")
DATA_STEGO = os.path.join("data", "stego")


def train_model() -> None:
    print("Loading datasets...")

    if not os.path.exists(DATA_CLEAN) or not os.path.exists(DATA_STEGO):
        print("Please generate the dataset first using src/generate_dataset.py")
        return

    # Load clean data (Label 0)
    X_clean, y_clean = process_directory(DATA_CLEAN, 0)

    # Load stego data (Label 1)
    X_stego, y_stego = process_directory(DATA_STEGO, 1)

    if len(X_clean) == 0 or len(X_stego) == 0:
        print("Dataset is empty. Run src/generate_dataset.py first.")
        return

    print(
        f"Loaded {len(X_clean)} clean samples and {len(X_stego)} steganography samples."
    )

    # Combine data
    X = np.vstack([X_clean, X_stego])
    y = np.hstack([y_clean, y_stego])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
