"""
CNN Model for Spectrogram-based Steganography Detection
Uses deep learning on mel-spectrogram images
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


class SpectrogramCNN:
    """CNN model for steganography detection using spectrograms"""

    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1)):
        """
        Initialize CNN model.

        Args:
            input_shape: Shape of input spectrograms (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.metrics = {}

    def build_model(self, architecture: str = "standard") -> None:
        """
        Build CNN architecture.

        Args:
            architecture: Model architecture ('standard', 'deep', 'resnet')
        """
        if architecture == "standard":
            self.model = self._build_standard_cnn()
        elif architecture == "deep":
            self.model = self._build_deep_cnn()
        elif architecture == "resnet":
            self.model = self._build_resnet()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall", "AUC"],
        )

    def _build_standard_cnn(self) -> models.Model:
        """Build standard CNN architecture"""
        model = models.Sequential(
            [
                # First convolutional block
                layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Dense layers
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        return model

    def _build_deep_cnn(self) -> models.Model:
        """Build deeper CNN architecture"""
        model = models.Sequential(
            [
                layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape
                ),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        return model

    def _build_resnet(self) -> models.Model:
        """Build ResNet-inspired architecture"""
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Residual blocks
        for filters in [32, 64, 128]:
            # Residual connection
            shortcut = x

            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(filters, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)

            # Match dimensions for residual connection
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)

            x = layers.Add()([x, shortcut])
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        return models.Model(inputs=inputs, outputs=outputs)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """
        Train the CNN model.

        Args:
            X_train: Training spectrograms
            y_train: Training labels
            X_val: Validation spectrograms
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        """
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                "models/cnn_best.h5", monitor="val_accuracy", save_best_only=True
            ),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model and return metrics"""
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        loss, accuracy, precision, recall, auc = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        self.metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
            "f1": float(2 * (precision * recall) / (precision + recall + 1e-7)),
        }

        print("\n" + "=" * 60)
        print("CNN Model Evaluation")
        print("=" * 60)
        print(f"Loss:      {self.metrics['loss']:.4f}")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1']:.4f}")
        print(f"AUC:       {self.metrics['auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Stego"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self.metrics

    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history["accuracy"], label="Train")
        axes[0, 0].plot(self.history.history["val_accuracy"], label="Validation")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history["loss"], label="Train")
        axes[0, 1].plot(self.history.history["val_loss"], label="Validation")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history["precision"], label="Train")
        axes[1, 0].plot(self.history.history["val_precision"], label="Validation")
        axes[1, 0].set_title("Model Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history["recall"], label="Train")
        axes[1, 1].plot(self.history.history["val_recall"], label="Validation")
        axes[1, 1].set_title("Model Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_model(self, filepath: str) -> None:
        """Save model and metadata"""
        self.model.save(filepath)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_shape": self.input_shape,
            "metrics": self.metrics,
        }

        metadata_path = filepath.replace(".h5", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load saved model"""
        self.model = keras.models.load_model(filepath)

        # Load metadata if available
        metadata_path = filepath.replace(".h5", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.metrics = metadata.get("metrics", {})

        print(f"Model loaded from {filepath}")


def audio_to_spectrogram(
    file_path: str, target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Convert audio file to mel-spectrogram image.

    Args:
        file_path: Path to audio file
        target_size: Target size for spectrogram

    Returns:
        Spectrogram as numpy array
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[0])
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )

    # Resize if needed
    if mel_spec_norm.shape[1] != target_size[1]:
        from scipy.ndimage import zoom

        zoom_factor = target_size[1] / mel_spec_norm.shape[1]
        mel_spec_norm = zoom(mel_spec_norm, (1, zoom_factor))

    # Take first target_size[1] frames
    mel_spec_norm = mel_spec_norm[:, : target_size[1]]

    # Add channel dimension
    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)

    return mel_spec_norm


def prepare_cnn_dataset(
    clean_dir: str, stego_dir: str, target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset of spectrograms for CNN training.

    Args:
        clean_dir: Directory with clean audio files
        stego_dir: Directory with stego audio files
        target_size: Target size for spectrograms

    Returns:
        Tuple of (spectrograms, labels)
    """
    spectrograms = []
    labels = []

    # Process clean files
    print("Processing clean files...")
    for filename in os.listdir(clean_dir):
        if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
            filepath = os.path.join(clean_dir, filename)
            try:
                spec = audio_to_spectrogram(filepath, target_size)
                spectrograms.append(spec)
                labels.append(0)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Process stego files
    print("Processing stego files...")
    for filename in os.listdir(stego_dir):
        if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
            filepath = os.path.join(stego_dir, filename)
            try:
                spec = audio_to_spectrogram(filepath, target_size)
                spectrograms.append(spec)
                labels.append(1)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return np.array(spectrograms), np.array(labels)


def train_cnn_model(
    clean_dir: str,
    stego_dir: str,
    architecture: str = "standard",
    epochs: int = 50,
    batch_size: int = 32,
) -> SpectrogramCNN:
    """
    Train CNN model on spectrograms.

    Args:
        clean_dir: Directory with clean audio
        stego_dir: Directory with stego audio
        architecture: CNN architecture to use
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Trained SpectrogramCNN instance
    """
    print("Preparing dataset...")
    X, y = prepare_cnn_dataset(clean_dir, stego_dir)

    print(f"Dataset size: {len(X)} samples")
    print(f"Spectrogram shape: {X.shape[1:]}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Build and train model
    cnn = SpectrogramCNN(input_shape=X.shape[1:])
    cnn.build_model(architecture=architecture)

    print(f"\nModel architecture: {architecture}")
    cnn.model.summary()

    print("\nTraining model...")
    cnn.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    # Evaluate
    print("\nEvaluating model...")
    cnn.evaluate(X_test, y_test)

    # Save visualizations
    os.makedirs("visualizations", exist_ok=True)
    cnn.plot_training_history("visualizations/cnn_training_history.png")

    # Save model
    os.makedirs("models", exist_ok=True)
    cnn.save_model(f"models/cnn_{architecture}_model.h5")

    return cnn


if __name__ == "__main__":
    # Train CNN models
    DATA_CLEAN = os.path.join("data", "clean")
    DATA_STEGO = os.path.join("data", "stego")

    for arch in ["standard", "deep"]:
        print(f"\n{'='*70}")
        print(f"Training {arch.upper()} CNN Model")
        print(f"{'='*70}\n")

        try:
            cnn = train_cnn_model(
                DATA_CLEAN, DATA_STEGO, architecture=arch, epochs=30, batch_size=16
            )
        except Exception as e:
            print(f"Error training {arch} CNN: {e}")
