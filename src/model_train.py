"""
Model Training Module with Multiple Algorithms
Supports Random Forest, SVM, Gradient Boosting, and Neural Networks
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from typing import Tuple, Dict, Any
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import AudioFeatureExtractor, process_directory


class SteganographyDetector:
    """Train and evaluate steganography detection models"""

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize detector with specified model type.

        Args:
            model_type: One of 'random_forest', 'svm', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.extractor = AudioFeatureExtractor()
        self.metrics = {}

    def get_model(self, hyperparams: Dict[str, Any] = None) -> Any:
        """Get model instance based on type"""
        if hyperparams is None:
            hyperparams = {}

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=hyperparams.get("n_estimators", 200),
                max_depth=hyperparams.get("max_depth", 20),
                min_samples_split=hyperparams.get("min_samples_split", 5),
                min_samples_leaf=hyperparams.get("min_samples_leaf", 2),
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "svm":
            return SVC(
                C=hyperparams.get("C", 1.0),
                kernel=hyperparams.get("kernel", "rbf"),
                gamma=hyperparams.get("gamma", "scale"),
                probability=True,
                random_state=42,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=hyperparams.get("n_estimators", 100),
                learning_rate=hyperparams.get("learning_rate", 0.1),
                max_depth=hyperparams.get("max_depth", 5),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparams: Dict[str, Any] = None,
        tune_hyperparams: bool = False,
    ) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            hyperparams: Model hyperparameters
            tune_hyperparams: Whether to perform hyperparameter tuning
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            self.model = self.get_model(hyperparams)
            print(f"Training {self.model_type} model...")
            self.model.fit(X_train_scaled, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Perform grid search for hyperparameter tuning"""
        if self.model_type == "random_forest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif self.model_type == "svm":
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            }
        elif self.model_type == "gradient_boosting":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            }
        else:
            return self.get_model()

        grid_search = GridSearchCV(
            self.get_model(), param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model and return metrics.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        print("\n" + "=" * 50)
        print(f"Model Evaluation - {self.model_type.upper()}")
        print("=" * 50)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1']:.4f}")
        print(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Stego"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self.metrics

    def plot_confusion_matrix(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None
    ) -> None:
        """Plot confusion matrix"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Clean", "Stego"],
            yticklabels=["Clean", "Stego"],
        )
        plt.title(f"Confusion Matrix - {self.model_type}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None
    ) -> None:
        """Plot ROC curve"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.model_type}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance (for tree-based models)"""
        if not hasattr(self.model, "feature_importances_"):
            print("Model does not have feature_importances_ attribute")
            return

        feature_names = self.extractor.get_feature_names()
        importances = self.model.feature_importances_

        # Get top N features
        indices = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances - {self.model_type}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_model(self, filepath: str, include_metadata: bool = True) -> None:
        """Save model, scaler, and metadata"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "extractor_params": {
                "n_mfcc": self.extractor.n_mfcc,
                "n_fft": self.extractor.n_fft,
                "hop_length": self.extractor.hop_length,
            },
        }

        if include_metadata:
            model_data["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics,
                "feature_names": self.extractor.get_feature_names(),
            }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

        # Save metadata as JSON
        if include_metadata:
            metadata_path = filepath.replace(".pkl", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(model_data["metadata"], f, indent=2)

    def load_model(self, filepath: str) -> None:
        """Load model, scaler, and metadata"""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]

        # Restore extractor params
        params = model_data["extractor_params"]
        self.extractor = AudioFeatureExtractor(
            n_mfcc=params["n_mfcc"],
            n_fft=params["n_fft"],
            hop_length=params["hop_length"],
        )

        if "metadata" in model_data:
            self.metrics = model_data["metadata"].get("metrics", {})

        print(f"Model loaded from {filepath}")

    def predict(self, file_path: str) -> Tuple[int, float]:
        """
        Predict if an audio file contains steganography.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (prediction, confidence)
        """
        features = self.extractor.extract_features(file_path)
        if features is None:
            raise ValueError("Could not extract features from file")

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0][prediction]

        return prediction, confidence


def train_model(
    data_clean_dir: str,
    data_stego_dir: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    tune_hyperparams: bool = False,
    save_path: str = "models/model.pkl",
) -> SteganographyDetector:
    """
    Train a steganography detection model.

    Args:
        data_clean_dir: Directory containing clean audio files
        data_stego_dir: Directory containing stego audio files
        model_type: Type of model to train
        test_size: Proportion of data for testing
        tune_hyperparams: Whether to tune hyperparameters
        save_path: Path to save the trained model

    Returns:
        Trained SteganographyDetector instance
    """
    print("Loading datasets...")

    if not os.path.exists(data_clean_dir) or not os.path.exists(data_stego_dir):
        raise ValueError("Data directories not found. Please generate dataset first.")

    # Initialize extractor
    extractor = AudioFeatureExtractor()

    # Load data
    X_clean, y_clean = process_directory(data_clean_dir, 0, extractor)
    X_stego, y_stego = process_directory(data_stego_dir, 1, extractor)

    if len(X_clean) == 0 or len(X_stego) == 0:
        raise ValueError("Dataset is empty. Please generate dataset first.")

    print(f"Loaded {len(X_clean)} clean samples and {len(X_stego)} stego samples.")
    print(f"Feature vector size: {X_clean.shape[1]}")

    # Combine data
    X = np.vstack([X_clean, X_stego])
    y = np.hstack([y_clean, y_stego])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    detector = SteganographyDetector(model_type=model_type)
    detector.train(X_train, y_train, tune_hyperparams=tune_hyperparams)

    # Evaluate
    detector.evaluate(X_test, y_test)

    # Create visualizations
    os.makedirs("visualizations", exist_ok=True)
    detector.plot_confusion_matrix(
        X_test, y_test, f"visualizations/confusion_matrix_{model_type}.png"
    )
    detector.plot_roc_curve(
        X_test, y_test, f"visualizations/roc_curve_{model_type}.png"
    )
    if model_type in ["random_forest", "gradient_boosting"]:
        detector.plot_feature_importance(
            save_path=f"visualizations/feature_importance_{model_type}.png"
        )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    detector.save_model(save_path)

    return detector


if __name__ == "__main__":
    # Train all models
    DATA_CLEAN = os.path.join("data", "clean")
    DATA_STEGO = os.path.join("data", "stego")

    for model_type in ["random_forest", "gradient_boosting", "svm"]:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}\n")

        try:
            detector = train_model(
                DATA_CLEAN,
                DATA_STEGO,
                model_type=model_type,
                save_path=f"models/{model_type}_model.pkl",
            )
        except Exception as e:
            print(f"Error training {model_type}: {e}")
