"""
Feature Extraction Module for Audio Steganography Detection
Extracts comprehensive features from audio files for ML model training
"""

import os
import numpy as np
import librosa
from scipy.stats import entropy, skew, kurtosis
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class AudioFeatureExtractor:
    """Extract features from audio files for steganography detection"""

    def __init__(self, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, file_path: str) -> Optional[np.ndarray]:
        """
        Extract comprehensive features from an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Feature vector as numpy array, or None if extraction fails
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=None)

            # 1. Statistical Features
            stat_features = self._extract_statistical_features(y)

            # 2. Spectral Features
            spectral_features = self._extract_spectral_features(y, sr)

            # 3. MFCC Features
            mfcc_features = self._extract_mfcc_features(y, sr)

            # 4. LSB-specific Features
            lsb_features = self._extract_lsb_features(y)

            # 5. Temporal Features
            temporal_features = self._extract_temporal_features(y, sr)

            # Combine all features
            features = np.hstack(
                [
                    stat_features,
                    spectral_features,
                    mfcc_features,
                    lsb_features,
                    temporal_features,
                ]
            )

            return features

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def _extract_statistical_features(self, y: np.ndarray) -> np.ndarray:
        """Extract statistical features from audio signal"""
        features = []

        # Basic statistics
        features.append(np.mean(np.abs(y)))
        features.append(np.std(y))
        features.append(np.max(np.abs(y)))
        features.append(np.min(y))
        features.append(skew(y))
        features.append(kurtosis(y))

        # Energy
        features.append(np.sum(y**2))

        return np.array(features)

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral features"""
        features = []

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend(
            [
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.max(spectral_centroid),
                np.min(spectral_centroid),
            ]
        )

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop_length
        )
        features.extend([np.mean(zcr), np.std(zcr)])

        return np.array(features)

    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Statistics of each MFCC coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        return np.hstack([mfcc_mean, mfcc_std])

    def _extract_lsb_features(self, y: np.ndarray) -> np.ndarray:
        """Extract LSB-specific features for steganography detection"""
        features = []

        # Convert to 16-bit integer
        if y.dtype != np.int16:
            y_int = (y * 32767).astype(np.int16)
        else:
            y_int = y

        # LSB extraction
        lsb = y_int & 1

        # LSB statistics
        features.append(np.mean(lsb))
        features.append(np.std(lsb))

        # LSB entropy
        lsb_counts = np.bincount(lsb.flatten())
        if len(lsb_counts) > 0:
            features.append(entropy(lsb_counts))
        else:
            features.append(0.0)

        # Chi-square test statistic for LSB randomness
        expected = len(lsb) / 2
        if expected > 0:
            chi_square = np.sum((lsb_counts - expected) ** 2 / expected)
            features.append(chi_square)
        else:
            features.append(0.0)

        # Consecutive LSB patterns
        lsb_diff = np.diff(lsb)
        features.append(np.mean(np.abs(lsb_diff)))
        features.append(np.std(lsb_diff))

        return np.array(features)

    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract temporal features"""
        features = []

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features.extend([np.mean(onset_env), np.std(onset_env)])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)

        return np.array(features)

    def get_feature_names(self) -> list:
        """Return names of all extracted features"""
        names = []

        # Statistical
        names.extend(
            [
                "mean_amplitude",
                "std_amplitude",
                "max_amplitude",
                "min_amplitude",
                "skewness",
                "kurtosis",
                "energy",
            ]
        )

        # Spectral
        names.extend(
            [
                "spectral_centroid_mean",
                "spectral_centroid_std",
                "spectral_centroid_max",
                "spectral_centroid_min",
                "spectral_bandwidth_mean",
                "spectral_bandwidth_std",
                "spectral_rolloff_mean",
                "spectral_rolloff_std",
                "spectral_contrast_mean",
                "spectral_contrast_std",
                "zcr_mean",
                "zcr_std",
            ]
        )

        # MFCC
        for i in range(self.n_mfcc):
            names.append(f"mfcc_{i}_mean")
        for i in range(self.n_mfcc):
            names.append(f"mfcc_{i}_std")

        # LSB
        names.extend(
            [
                "lsb_mean",
                "lsb_std",
                "lsb_entropy",
                "lsb_chi_square",
                "lsb_diff_mean",
                "lsb_diff_std",
            ]
        )

        # Temporal
        names.extend(["onset_strength_mean", "onset_strength_std", "tempo"])

        return names


def process_directory(
    directory: str, label: int, extractor: Optional[AudioFeatureExtractor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all audio files in a directory and extract features.

    Args:
        directory: Path to directory containing audio files
        label: Label for the files (0 for clean, 1 for stego)
        extractor: AudioFeatureExtractor instance (creates new if None)

    Returns:
        Tuple of (features array, labels array)
    """
    if extractor is None:
        extractor = AudioFeatureExtractor()

    features_list = []
    labels_list = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return np.array([]), np.array([])

    for filename in os.listdir(directory):
        if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
            filepath = os.path.join(directory, filename)
            features = extractor.extract_features(filepath)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

    return np.array(features_list), np.array(labels_list)


def get_feature_importance_dict(
    model, extractor: AudioFeatureExtractor
) -> Dict[str, float]:
    """
    Get feature importance as a dictionary.

    Args:
        model: Trained model with feature_importances_ attribute
        extractor: AudioFeatureExtractor instance

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not hasattr(model, "feature_importances_"):
        return {}

    feature_names = extractor.get_feature_names()
    importances = model.feature_importances_

    return dict(zip(feature_names, importances))
