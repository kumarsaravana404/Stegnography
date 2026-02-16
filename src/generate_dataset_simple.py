"""
Simple Dataset Generation - Non-Interactive
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import random

# Constants
SAMPLE_RATE = 44100
DURATION = 3  # seconds
DATA_DIR = "data"
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
STEGO_DIR = os.path.join(DATA_DIR, "stego")


def generate_sine_wave(filename: str, frequency: int) -> None:
    """Generates a simple sine wave and saves it as a WAV file."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise

    # Normalize to 16-bit integer range
    audio = np.int16(audio * 32767)

    wav.write(filename, SAMPLE_RATE, audio)
    print(f"Generated: {filename}")


def embed_lsb(input_file: str, output_file: str) -> None:
    """Embeds random data using LSB modification."""
    rate, data = wav.read(input_file)

    # Ensure data is 1D
    if len(data.shape) > 1:
        data = data[:, 0]  # Take first channel if stereo

    num_samples = len(data)
    secret_bits = np.random.randint(0, 2, num_samples, dtype=np.int16)

    # Modify LSBs
    data = (data & ~1) | secret_bits

    wav.write(output_file, rate, data)
    print(f"Generated: {output_file}")


def create_dataset(num_samples: int = 20) -> None:
    """Creates a dataset of clean and stego audio files."""
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(STEGO_DIR, exist_ok=True)

    print(f"Generating {num_samples} audio file pairs...")

    for i in range(num_samples):
        clean_filename = os.path.join(CLEAN_DIR, f"sample_{i}.wav")
        stego_filename = os.path.join(STEGO_DIR, f"sample_{i}.wav")

        # Generate clean audio with random frequency
        freq = random.randint(300, 1500)
        generate_sine_wave(clean_filename, freq)

        # Generate stego version
        embed_lsb(clean_filename, stego_filename)

    print(f"\nDataset generation complete!")
    print(f"Clean samples: {CLEAN_DIR}")
    print(f"Stego samples: {STEGO_DIR}")


if __name__ == "__main__":
    create_dataset(num_samples=20)
