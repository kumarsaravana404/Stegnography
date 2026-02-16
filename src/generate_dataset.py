"""
Enhanced Dataset Generation with Multiple Steganography Techniques
"""

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


def generate_complex_audio(
    filename: str, duration: int, sample_rate: int = SAMPLE_RATE
) -> None:
    """Generate more complex audio with multiple frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Mix multiple frequencies
    frequencies = [random.randint(200, 2000) for _ in range(3)]
    audio = np.zeros_like(t)

    for freq in frequencies:
        amplitude = random.uniform(0.1, 0.3)
        audio += amplitude * np.sin(2 * np.pi * freq * t)

    # Add noise
    noise = np.random.normal(0, 0.02, audio.shape)
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = np.int16(audio * 32767)

    wav.write(filename, sample_rate, audio)
    print(f"Generated complex audio: {filename}")


def embed_lsb_random(
    input_file: str, output_file: str, message: str = "SECRET"
) -> None:
    """Embeds hidden data using random LSB modification."""
    rate, data = wav.read(input_file)

    num_samples = len(data)
    secret_bits = np.random.randint(0, 2, num_samples)

    # Ensure types match for bitwise operations
    secret_bits = secret_bits.astype(data.dtype)

    # Modify LSBs
    data = (data & ~1) | secret_bits

    wav.write(output_file, rate, data)
    print(f"Generated stego audio (LSB random): {output_file}")


def embed_lsb_message(input_file: str, output_file: str, message: str) -> None:
    """Embed actual message in LSB."""
    rate, data = wav.read(input_file)

    # Convert message to binary
    message_with_end = message + "###END###"
    binary_message = "".join(format(ord(char), "08b") for char in message_with_end)

    if len(binary_message) > len(data):
        print(f"Warning: Message too long, truncating...")
        binary_message = binary_message[: len(data)]

    # Embed message
    for i, bit in enumerate(binary_message):
        data[i] = (data[i] & ~1) | int(bit)

    wav.write(output_file, rate, data)
    print(f"Generated stego audio (LSB message): {output_file}")


def embed_echo_hiding(input_file: str, output_file: str) -> None:
    """Simple echo hiding technique."""
    rate, data = wav.read(input_file)

    # Add subtle echo
    delay_samples = int(0.05 * rate)  # 50ms delay
    echo_amplitude = 0.3

    # Create echo
    echo = np.zeros_like(data)
    echo[delay_samples:] = data[:-delay_samples] * echo_amplitude

    # Mix original with echo
    stego_data = data + echo.astype(data.dtype)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(stego_data))
    if max_val > 32767:
        stego_data = (stego_data / max_val * 32767).astype(np.int16)

    wav.write(output_file, rate, stego_data)
    print(f"Generated stego audio (echo hiding): {output_file}")


def create_dataset(num_samples: int = 20, use_complex_audio: bool = True) -> None:
    """Creates a dataset of clean and stego audio files."""
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)
    if not os.path.exists(STEGO_DIR):
        os.makedirs(STEGO_DIR)

    for i in range(num_samples):
        clean_filename = os.path.join(CLEAN_DIR, f"sample_{i}.wav")
        stego_filename = os.path.join(STEGO_DIR, f"sample_{i}.wav")

        # Generate clean audio
        if use_complex_audio and i % 2 == 0:
            generate_complex_audio(clean_filename, DURATION)
        else:
            freq = random.randint(200, 2000)
            generate_sine_wave(clean_filename, freq, DURATION)

        # Generate stego version with different techniques
        technique = i % 3
        if technique == 0:
            # Random LSB
            embed_lsb_random(clean_filename, stego_filename)
        elif technique == 1:
            # Message LSB
            message = f"Secret message {i}: " + "X" * random.randint(50, 200)
            embed_lsb_message(clean_filename, stego_filename, message)
        else:
            # Echo hiding
            embed_echo_hiding(clean_filename, stego_filename)


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Steganography Dataset Generation")
    print("=" * 60)

    num_samples = int(
        input("Enter number of samples to generate (default 20): ") or "20"
    )
    use_complex = input("Use complex audio? (y/n, default y): ").lower() != "n"

    create_dataset(num_samples=num_samples, use_complex_audio=use_complex)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Clean samples: {CLEAN_DIR}")
    print(f"Stego samples: {STEGO_DIR}")
    print("=" * 60)
