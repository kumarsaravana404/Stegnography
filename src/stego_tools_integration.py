"""
Image Steganography Tools
LSB-based steganography for images
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os


class ImageSteganography:
    """LSB-based image steganography"""

    @staticmethod
    def encode_text(image_path: str, text: str, output_path: str) -> bool:
        """
        Hide text in an image using LSB steganography.

        Args:
            image_path: Path to cover image
            text: Text to hide
            output_path: Path to save stego image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Add delimiter to text
            text_with_delimiter = text + "###END###"

            # Convert text to binary
            binary_text = "".join(
                format(ord(char), "08b") for char in text_with_delimiter
            )

            # Check if image can hold the text
            max_bytes = img_array.size
            if len(binary_text) > max_bytes:
                print("Error: Text too large for image")
                return False

            # Flatten image array
            flat_img = img_array.flatten()

            # Embed text in LSB
            for i, bit in enumerate(binary_text):
                flat_img[i] = (flat_img[i] & ~1) | int(bit)

            # Reshape and save
            stego_img = flat_img.reshape(img_array.shape)
            Image.fromarray(stego_img.astype(np.uint8)).save(output_path)

            print(f"Text successfully hidden in {output_path}")
            return True

        except Exception as e:
            print(f"Error encoding text: {e}")
            return False

    @staticmethod
    def decode_text(image_path: str) -> Optional[str]:
        """
        Extract hidden text from an image.

        Args:
            image_path: Path to stego image

        Returns:
            Extracted text or None if failed
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)

            # Flatten image
            flat_img = img_array.flatten()

            # Extract LSBs
            binary_text = "".join(str(pixel & 1) for pixel in flat_img)

            # Convert binary to text
            text = ""
            for i in range(0, len(binary_text), 8):
                byte = binary_text[i : i + 8]
                if len(byte) == 8:
                    char = chr(int(byte, 2))
                    text += char

                    # Check for delimiter
                    if text.endswith("###END###"):
                        return text[:-9]  # Remove delimiter

            return text

        except Exception as e:
            print(f"Error decoding text: {e}")
            return None

    @staticmethod
    def detect_steganography(image_path: str) -> Tuple[bool, float]:
        """
        Simple detection based on LSB entropy.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (is_stego, confidence)
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)

            # Extract LSBs
            lsb = img_array.flatten() & 1

            # Calculate entropy
            unique, counts = np.unique(lsb, return_counts=True)
            probabilities = counts / len(lsb)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            # High entropy suggests steganography
            # Perfect randomness would have entropy close to 1
            threshold = 0.95
            is_stego = entropy > threshold
            confidence = min(entropy, 1.0)

            return is_stego, confidence

        except Exception as e:
            print(f"Error detecting steganography: {e}")
            return False, 0.0


def create_stego_image_dataset(clean_dir: str, stego_dir: str, num_samples: int = 10):
    """
    Create a dataset of clean and stego images.

    Args:
        clean_dir: Directory to save clean images
        stego_dir: Directory to save stego images
        num_samples: Number of image pairs to create
    """
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    for i in range(num_samples):
        # Create random image
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        clean_path = os.path.join(clean_dir, f"image_{i}.png")
        Image.fromarray(img_array).save(clean_path)

        # Create stego version
        secret_text = f"Secret message {i}: " + "X" * np.random.randint(50, 200)
        stego_path = os.path.join(stego_dir, f"image_{i}.png")
        ImageSteganography.encode_text(clean_path, secret_text, stego_path)

    print(f"Created {num_samples} image pairs")


if __name__ == "__main__":
    # Example usage
    img_stego = ImageSteganography()

    # Create test image
    test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(test_img).save("test_image.png")

    # Encode
    secret = "This is a secret message hidden in the image!"
    img_stego.encode_text("test_image.png", secret, "stego_image.png")

    # Decode
    extracted = img_stego.decode_text("stego_image.png")
    print(f"Extracted text: {extracted}")

    # Detect
    is_stego, conf = img_stego.detect_steganography("stego_image.png")
    print(f"Steganography detected: {is_stego} (confidence: {conf:.2f})")
