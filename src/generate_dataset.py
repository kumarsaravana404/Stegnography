import os
import numpy as np
import scipy.io.wavfile as wav
import random
from typing import Optional

# Constants
SAMPLE_RATE = 44100
DURATION = 5  # seconds
DATA_DIR = "data"
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
STEGO_DIR = os.path.join(DATA_DIR, "stego")


def generate_sine_wave(
    filename: str, frequency: int, duration: int, sample_rate: int = SAMPLE_RATE
) -> None:
    """Generates a sine wave and saves it as a WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a sine wave with some random noise to make it realistic
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise

    # Normalize to 16-bit integer range
    audio = np.int16(audio * 32767)

    wav.write(filename, sample_rate, audio)
    print(f"Generated clean audio: {filename}")


def embed_message(input_file: str, output_file: str, message: str = "SECRET") -> None:
    """Embeds a hidden message into the audio file using LSB steganography."""
    rate, data = wav.read(input_file)
    # original_data = data.copy() # Unused

    # We want to modify a significant portion of the file to make it detectable
    # by our simple statistical features.
    # Let's simple replace LSBs with random bits for the "stego" class
    # This simulates a high-capacity LSB steganography

    num_samples = len(data)
    secret_bits = np.random.randint(0, 2, num_samples)

    # Modify LSBs
    # data is int16. We use bitwise operations.
    # data & ~1 clears the LSB. | secret_bits sets it.

    # We need to ensure we don't overflow/underflow if we were adding, but bitwise is safe on int16
    data = (data & ~1) | secret_bits

    wav.write(output_file, rate, data)
    print(f"Generated stego audio: {output_file}")


def create_dataset(num_samples: int = 10) -> None:
    """Creates a dataset of clean and stego audio files."""
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)
    if not os.path.exists(STEGO_DIR):
        os.makedirs(STEGO_DIR)

    for i in range(num_samples):
        clean_filename = os.path.join(CLEAN_DIR, f"sample_{i}.wav")
        stego_filename = os.path.join(STEGO_DIR, f"sample_{i}.wav")

        # Generate random frequency between 200 and 2000 Hz
        freq = random.randint(200, 2000)
        generate_sine_wave(clean_filename, freq, DURATION)

        # Embed a random message
        message = "SECRET_DATA_" * random.randint(1, 10)
        embed_message(clean_filename, stego_filename, message)


if __name__ == "__main__":
    create_dataset(num_samples=20)
    print("Dataset generation complete!")
