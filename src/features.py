import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
from scipy.stats import entropy


def extract_features(file_path):
    """
    Extracts features from an audio file suitable for steganography detection.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)

        # 1. Statistical Features
        mean_amp = np.mean(np.abs(y))
        std_amp = np.std(y)
        zero_crossings = np.sum(librosa.zero_crossings(y))

        # 2. Spectral Features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfccs = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1
        )  # 13 features

        # 3. LSB specific features (simplistic)
        # Convert to 16-bit integer if not already
        if y.dtype != np.int16:
            y_int = (y * 32767).astype(np.int16)
        else:
            y_int = y

        lsb = y_int & 1
        lsb_mean = np.mean(lsb)
        lsb_std = np.std(lsb)
        # Calculate entropy of LSB distribution (0 vs 1)
        lsb_counts = np.bincount(lsb.flatten())
        lsb_entropy = entropy(lsb_counts) if len(lsb_counts) > 0 else 0

        features = np.hstack(
            [
                mean_amp,
                std_amp,
                zero_crossings,
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff,
                mfccs,
                lsb_mean,
                lsb_std,
                lsb_entropy,
            ]
        )

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_directory(directory, label):
    """
    Iterates through a directory, extracts features, and returns X (features) and y (labels).
    """
    features_list = []
    labels_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            features = extract_features(filepath)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

    return np.array(features_list), np.array(labels_list)
